"""CLI tool to run and evaluate analyser agent experiments with an LLM judge.

Every entry in eval_data/analyser/prompt.yaml is one experiment: it names the
data file to analyse (CSV/Excel in eval_data/analyser/), the user prompt to try
and optionally a ground truth reference from the shared ground_truth/ directory -
see src/analyser/experiment_spec.py for the format.

For each experiment the data file is uploaded to the backend and the analyser is
run on the prompt. The run is then scored:
- execution_success: 1 if the backend reported no error, else 0 (the generated
  code is not executed again),
- code_simplicity, code_correctness: judged from the generated code, the
  execution result and a profile of the input data, whenever there is code,
- data_correctness, task_relevance, chart_suitability, completeness,
  readability: judged from the plot, the request, the input data and the
  reference, only if execution succeeded and a plot exists (NaN otherwise).
The judge model, prompts and 1-5 rubrics live in src/analyser/judge_config.yaml;
the judge reads OPENAI_API_KEY from the environment.

Metrics, params, judge explanations and outputs (plot, code, ...) are logged to
MLflow (sqlite:///mlflow.db) under an MLflow experiment named
analyser/{experiment_name} and a run named run_name - failed runs included.

If --experiment_name is omitted, every experiment is run in turn; one missing its
data file or named ground truth is logged as skipped (NaN metrics) instead of
aborting the batch.

Usage:
    python evaluate_analyser.py --experiment_name kontoauszug --run_name gemma4_v1
    python evaluate_analyser.py --run_name gemma4_v1
"""

import argparse
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import yaml
from loguru import logger
from src.analyser import (
    JUDGE_CONFIG_PATH,
    MLFLOW_PREFIX,
)
from src.analyser.evaluate import AnalyserEvaluation, AnalyserEvaluator
from src.analyser.experiment_spec import (
    AnalyserExperimentSpec,
    load_experiment_entries,
    load_experiment_spec,
)
from src.analyser.judge import LLMJudge
from src.analyser.judge_config import (
    ALL_METRIC_NAMES,
    judge_config_sha,
    load_judge_config,
)
from src.analyser.run_analyser import AnalyserRunner, AnalyserRunOutcome
from src.experiments import (
    configure_mlflow,
    log_run_to_mlflow,
    mlflow_experiment_name,
    run_all_experiments,
)

# MLflow rejects tag values above 5000 characters.
MAX_TAG_LENGTH = 5000


@dataclass(frozen=True)
class EvaluationContext:
    """Everything shared by all experiments of one CLI run."""

    runner: AnalyserRunner
    evaluator: AnalyserEvaluator
    analyser_config: dict[str, Any]
    judge_model: str
    judge_config_sha: str


def write_artifacts(
    artifact_dir: Path,
    outcome: AnalyserRunOutcome,
    evaluation: AnalyserEvaluation,
    analyser_config: dict[str, Any],
) -> list[Path]:
    """Writes this run's outputs to a temp directory for MLflow to upload.

    Args:
        artifact_dir (Path): Temporary directory to write into.
        outcome (AnalyserRunOutcome): What the analyser returned.
        evaluation (AnalyserEvaluation): The judge results.
        analyser_config (dict[str, Any]): The analyser agent config used.
    Returns:
        list[Path]: The written files.
    """
    paths: list[Path] = []

    def write(name: str, content: str | bytes) -> None:
        path = artifact_dir / name
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content)
        paths.append(path)

    if outcome.plot_png is not None:
        write("plot.png", outcome.plot_png)
    if outcome.output_csv is not None:
        write("output_data.csv", outcome.output_csv)
    if outcome.python_code:
        write("analyser_code.py", outcome.python_code)
    if outcome.code_summary:
        write("code_summary.txt", outcome.code_summary)
    errors = [
        f"{label}:\n{message}"
        for label, message in (
            ("Analyser error", outcome.error_message),
            ("Download error", outcome.download_error),
            *(("Judge error", message) for message in evaluation.judge_errors),
        )
        if message
    ]
    if errors:
        write("error.txt", "\n\n".join(errors))
    write("judge_results.json", json.dumps(evaluation.judgements, indent=2))
    write("analyser_config.yaml", yaml.safe_dump(analyser_config, sort_keys=False))

    judge_config_copy = artifact_dir / JUDGE_CONFIG_PATH.name
    shutil.copyfile(JUDGE_CONFIG_PATH, judge_config_copy)
    paths.append(judge_config_copy)
    return paths


def log_to_mlflow(
    spec: AnalyserExperimentSpec,
    run_name: str,
    outcome: AnalyserRunOutcome,
    evaluation: AnalyserEvaluation,
    context: EvaluationContext,
) -> None:
    """Logs one analyser experiment run's params, metrics, tags and outputs to
    MLflow.

    Args:
        spec (AnalyserExperimentSpec): The experiment that was run.
        run_name (str): Name of this run.
        outcome (AnalyserRunOutcome): What the analyser returned.
        evaluation (AnalyserEvaluation): The judge results.
        context (EvaluationContext): Shared run settings (models, config hash).
    """
    tags = {"agent": "analyser"}
    if outcome.error_message:
        tags["error_message"] = outcome.error_message[:MAX_TAG_LENGTH]
    if outcome.download_error:
        tags["download_error"] = outcome.download_error[:MAX_TAG_LENGTH]
    if evaluation.judge_errors:
        tags["judge_error"] = "; ".join(evaluation.judge_errors)[:MAX_TAG_LENGTH]

    # The artifacts only need to exist until MLflow has copied them into its
    # own artifact store.
    with tempfile.TemporaryDirectory() as tmp_dir:
        artifact_paths = write_artifacts(
            Path(tmp_dir), outcome, evaluation, context.analyser_config
        )
        log_run_to_mlflow(
            mlflow_experiment_name(spec.name, MLFLOW_PREFIX),
            run_name,
            params={
                "user_prompt": spec.user_prompt,
                "input_file": spec.input_file.name,
                "ground_truth": (
                    spec.ground_truth_path.name if spec.ground_truth_path else "none"
                ),
                "analyser_model": context.analyser_config.get("model"),
                "analyser_model_provider": context.analyser_config.get(
                    "model_provider"
                ),
                "judge_model": context.judge_model,
                "judge_config_sha": context.judge_config_sha,
            },
            metrics=evaluation.metrics,
            artifact_paths=artifact_paths,
            tags=tags,
        )


def run_experiment(
    experiment_name: str, run_name: str, context: EvaluationContext
) -> None:
    """Runs the analyser on one experiment, judges the result and logs it.

    Args:
        experiment_name (str): Name of the experiment (key in
            eval_data/analyser/prompt.yaml).
        run_name (str): Name of this run.
        context (EvaluationContext): Shared runner, evaluator and settings.
    """
    # Fail fast (before any backend call) if the data file, prompt or a named
    # ground truth is missing.
    spec = load_experiment_spec(experiment_name)

    logger.info(f"Running experiment '{experiment_name}' (run '{run_name}')...")
    outcome = context.runner.run(spec.input_file, spec.user_prompt)
    logger.info(
        f"Analyser finished: execution_success={outcome.execution_success}, "
        f"plot={'yes' if outcome.plot_png is not None else 'no'}."
    )

    logger.info("Judging...")
    evaluation = context.evaluator.evaluate(spec, outcome)
    log_to_mlflow(spec, run_name, outcome, evaluation, context)

    scores = ", ".join(
        f"{name}={value if value is not None else '-'}"
        for name, value in evaluation.metrics.items()
    )
    print(
        f"\nExperiment '{experiment_name}', run '{run_name}' complete: {scores}. "
        f"Logged to MLflow."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run and evaluate analyser agent experiment(s) with an LLM judge."
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help=(
            "Name of the experiment (key in eval_data/analyser/prompt.yaml). "
            "If omitted, every experiment in that file is run."
        ),
    )
    parser.add_argument(
        "--run_name",
        type=str,
        required=True,
        help="Name of this run, used as the MLflow run name.",
    )
    args = parser.parse_args()

    # Everything that can fail without touching the analyser is checked first:
    # the judge config, the API key and the judge model itself.
    judge_config = load_judge_config()
    # Covers both "openai:" and "openai-responses:" models.
    if judge_config.model.startswith("openai") and not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY for the judge model first.")
    judge = LLMJudge(judge_config)
    logger.info(f"Checking judge model {judge_config.model}...")
    judge.check_model()

    configure_mlflow()
    runner = AnalyserRunner()
    context = EvaluationContext(
        runner=runner,
        evaluator=AnalyserEvaluator(judge),
        analyser_config=runner.prepare(),
        judge_model=judge_config.model,
        judge_config_sha=judge_config_sha(),
    )

    if args.experiment_name is None:
        run_all_experiments(
            list(load_experiment_entries()),
            args.run_name,
            partial(run_experiment, context=context),
            ALL_METRIC_NAMES,
            mlflow_prefix=MLFLOW_PREFIX,
        )
    else:
        run_experiment(args.experiment_name, args.run_name, context)


if __name__ == "__main__":
    main()
