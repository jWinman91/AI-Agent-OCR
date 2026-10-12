"""CLI tool to run and evaluate data_extractor agent experiments.

Runs the data extractor against the test files in test_data/{experiment_name}/,
using the prompt for that experiment from test_data/prompt.yaml, then evaluates
the result against ground_truth/{experiment_name}.csv and reports metrics.
Metrics, params, and output files are logged to MLflow (sqlite:///mlflow.db) under
an MLflow experiment named experiment_name and a run named run_name, so past runs
can be browsed and compared via `mlflow ui --backend-store-uri sqlite:///mlflow.db`.
Only the extracted data CSV is kept permanently on disk (extracted_data/); the
per-run config yaml and false positive/negative CSVs are written to the system
temp directory just long enough to be logged as MLflow artifacts, then removed.

If --experiment_name is omitted, every subdirectory of test_data/ is run in turn.
In that mode, an experiment missing its prompt, ground truth, or test files is
logged as an error and skipped (with NaN metrics logged to MLflow) rather than
aborting the whole batch.

Usage:
    python evaluate_extractor.py --experiment_name body_weight --run_name gemma4_v1
    python evaluate_extractor.py --run_name gemma4_v1
"""

import argparse
import math
from pathlib import Path
from typing import Any

import mlflow
import yaml
from loguru import logger
from src.evaluate import ALL_OVERALL_METRIC_NAMES, DataExtractorEvaluator
from src.extract_data import DataExtractor

TEST_DATA_DIR = Path("test_data")
PROMPT_PATH = TEST_DATA_DIR / "prompt.yaml"
GROUND_TRUTH_DIR = Path("ground_truth")
EXTRACTED_DATA_DIR = Path("extracted_data")
MLFLOW_DB_PATH = Path("mlflow.db")

mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
logger.info(
    f"MLflow tracking URI set to {mlflow.get_tracking_uri()} "
    f"(resolved from cwd: {Path.cwd()})"
)


class ExperimentDataMissing(Exception):
    """Raised when an experiment is missing its prompt, ground truth, or test
    data files - i.e. data the user needs to add, not a run/backend failure."""


def discover_experiment_names() -> list[str]:
    """Lists every experiment subdirectory under test_data/.

    Returns:
        list[str]: Sorted experiment names (test_data/ subdirectory names).
    """
    return sorted(path.name for path in TEST_DATA_DIR.iterdir() if path.is_dir())


def check_experiment_ready(experiment_name: str) -> None:
    """Validates that an experiment has everything needed to run and evaluate.

    Checked upfront (before any backend call) so a missing prompt, ground
    truth, or test file is reported immediately instead of wasting a real
    extraction call on an experiment that can't be evaluated anyway.

    Args:
        experiment_name (str): Name of the experiment to validate.
    Raises:
        ExperimentDataMissing: If the prompt, ground truth, or test data files
            for this experiment are missing.
    """
    try:
        load_user_prompt(experiment_name)
    except (FileNotFoundError, ValueError) as e:
        raise ExperimentDataMissing(str(e)) from e

    ground_truth_path = GROUND_TRUTH_DIR / f"{experiment_name}.csv"
    if not ground_truth_path.exists():
        raise ExperimentDataMissing(
            f"Ground truth file not found at {ground_truth_path}."
        )

    experiment_dir = TEST_DATA_DIR / experiment_name
    if not experiment_dir.is_dir() or not any(
        path.is_file() for path in experiment_dir.iterdir()
    ):
        raise ExperimentDataMissing(
            f"No test data files found for experiment '{experiment_name}' "
            f"in {experiment_dir}."
        )


def load_user_prompt(experiment_name: str) -> str:
    """Loads the user prompt for an experiment from the shared prompt.yaml file.

    Args:
        experiment_name (str): Name of the experiment to load the prompt for.
    Returns:
        str: The user prompt string.
    """
    # All experiment prompts live in one file, keyed by experiment_name, rather
    # than one prompt.yaml per experiment folder - easier to see/edit them all
    # at a glance and keep in sync with the test_data/ subfolders.
    if not PROMPT_PATH.exists():
        raise FileNotFoundError(
            f"Prompt file not found at {PROMPT_PATH}. "
            f"Create it with a '{experiment_name}' key."
        )

    with PROMPT_PATH.open("r") as f:
        prompts = yaml.safe_load(f)

    if not prompts or experiment_name not in prompts:
        raise ValueError(
            f"{PROMPT_PATH} must define a '{experiment_name}' key "
            f"with the user prompt for that experiment."
        )

    return prompts[experiment_name]


def flatten_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    """Flattens the per-field + overall metrics dict into MLflow's flat
    {name: float} metric namespace, suffixing each metric name with the field
    it belongs to (or "overall" for the macro-averaged summary).

    Each field only has the metric keys its strategy produced (e.g. mae for a
    single-value field, accuracy/precision/recall/f1_score for a multi-value
    one), so this iterates whatever keys are present rather than assuming a
    fixed set. E.g. {"fields": {"Saldo": {"accuracy": 0.9, ...}, "body_weight":
    {"mae": 1.2}}, "overall": {...}} becomes {"accuracy_Saldo": 0.9, ...,
    "mae_body_weight": 1.2, "accuracy_overall": ..., "mae_overall": ...}.

    A metric value of None (the field was never extracted, or no field
    contributed to that overall metric) is logged as NaN instead, since MLflow
    rejects None for metric values but accepts NaN as its native "missing
    value" representation (renders as blank in the UI).

    Args:
        metrics (dict[str, Any]): The dict returned by
            DataExtractorEvaluator.evaluate_extracted_data.
    Returns:
        dict[str, float]: Flat metric name -> value dict ready for
        mlflow.log_metrics.
    """
    flat: dict[str, float] = {}
    for field, field_metrics in metrics["fields"].items():
        for metric_name, value in field_metrics.items():
            flat[f"{metric_name}_{field}"] = math.nan if value is None else value
    for metric_name, value in metrics["overall"].items():
        flat[f"{metric_name}_overall"] = math.nan if value is None else value
    return flat


def log_to_mlflow(
    experiment_name: str,
    run_name: str,
    user_prompt: str,
    metrics: dict[str, Any],
    config_path: Path,
    mismatch_paths: list[Path],
) -> None:
    """Logs one experiment run's params, metrics, and output files to MLflow.

    Args:
        experiment_name (str): Name of the experiment; mapped 1:1 to an MLflow
            experiment of the same name.
        run_name (str): Name of this run; mapped 1:1 to an MLflow run of the
            same name, nested under the MLflow experiment above.
        user_prompt (str): The prompt used for this run, logged as a param.
        metrics (dict[str, Any]): The metrics dict returned by
            DataExtractorEvaluator.evaluate_extracted_data (per-field metrics
            plus a macro-averaged "overall" set).
        config_path (Path): Path to the temporary config yaml file to log as an
            artifact.
        mismatch_paths (list[Path]): Paths to the temporary mismatch csv files
            (false positives/negatives and/or single-value mismatches,
            whichever the evaluator produced) to log as artifacts.
    """
    with config_path.open("r") as f:
        config = yaml.safe_load(f)

    # The extracted data CSV is the only output kept permanently on disk
    # (alongside ground_truth/); everything else is logged from /tmp and
    # cleaned up by the caller once this upload succeeds.
    artifact_paths = [
        EXTRACTED_DATA_DIR / f"{experiment_name}_{run_name}.csv",
        config_path,
        *mismatch_paths,
    ]

    # set_experiment creates the MLflow experiment on first use and reuses it on
    # every later call with the same name, so repeated runs of the same
    # experiment_name accumulate as multiple runs under one MLflow experiment.
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        logger.info(
            f"Logging to MLflow: experiment '{experiment_name}', run '{run_name}' "
            f"(run_id={run.info.run_id})..."
        )
        mlflow.log_param("user_prompt", user_prompt)
        mlflow.log_params(
            {
                "model": config.get("model"),
                "model_provider": config.get("model_provider"),
            }
        )
        mlflow.log_metrics(flatten_metrics(metrics))
        logger.info("Logged params and metrics to MLflow.")
        for artifact_path in artifact_paths:
            mlflow.log_artifact(str(artifact_path))
        logger.info(f"Logged {len(artifact_paths)} artifact(s) to MLflow.")


def log_skipped_to_mlflow(experiment_name: str, run_name: str, reason: str) -> None:
    """Logs a skipped experiment to MLflow with NaN metrics in place of real ones.

    Used in "run all experiments" mode when an experiment is missing required
    data and never actually ran, so there are no real metrics/params/artifacts
    to log - only a record that this run was attempted and why it was skipped.

    Args:
        experiment_name (str): Name of the experiment; mapped 1:1 to an MLflow
            experiment of the same name.
        run_name (str): Name of this run; mapped 1:1 to an MLflow run of the
            same name, nested under the MLflow experiment above.
        reason (str): Human-readable reason the experiment was skipped, logged
            as a tag so it's visible in the MLflow UI.
    """
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        logger.info(
            f"Logging skipped experiment to MLflow: experiment '{experiment_name}', "
            f"run '{run_name}' (run_id={run.info.run_id})..."
        )
        mlflow.set_tag("skipped_reason", reason)
        # No fields were ever extracted, so there's nothing to report per
        # field - log the full superset of possible "overall" metrics, all NaN
        # (not None, which MLflow rejects; NaN is its native "missing value"
        # representation and shows as blank in the UI), so skipped and real
        # runs keep consistent overall columns.
        mlflow.log_metrics(
            {f"{name}_overall": math.nan for name in ALL_OVERALL_METRIC_NAMES}
        )
        logger.info("Logged NaN metrics and skipped_reason tag to MLflow.")


def run_experiment(experiment_name: str, run_name: str) -> None:
    """Runs a full extract-and-evaluate experiment for the data_extractor agent.

    Args:
        experiment_name (str): Name of the experiment (subfolder of test_data/).
        run_name (str): Name of this run, used to tag the saved outputs.
    """
    # Fail fast (before any backend call) if the prompt, ground truth, or test
    # files for this experiment are missing.
    check_experiment_ready(experiment_name)
    user_prompt = load_user_prompt(experiment_name)

    logger.info(f"Running experiment '{experiment_name}' (run '{run_name}')...")
    # DataExtractor.__init__ triggers /update_agents, so the backend picks up
    # whatever config change the user made before starting this run.
    extractor = DataExtractor()

    files = extractor.load_files(experiment_name)
    logger.info(f"Uploading {len(files)} file(s)...")
    extractor.upload_files(files)

    logger.info("Extracting data...")
    data_file_path = extractor.extract_data(user_prompt)

    # The extracted data CSV is the one output kept permanently on disk; the
    # config yaml is only needed transiently to attach as an MLflow artifact.
    extractor.save_extracted_data(data_file_path, experiment_name, run_name)
    config_path = extractor.save_config(experiment_name, run_name)
    logger.info("Extraction complete, extracted data and config saved.")

    # Evaluation reads back the extracted data CSV just written above (by
    # experiment_name + run_name), so it always scores exactly this run's
    # extraction output.
    evaluator = DataExtractorEvaluator()
    metrics = evaluator.evaluate_extracted_data(experiment_name, run_name)
    mismatch_paths = evaluator.save_mismatches(experiment_name, run_name)

    # Log to MLflow last, once every file it attaches as an artifact already
    # exists on disk (as temp files for the config/mismatch CSVs).
    log_to_mlflow(
        experiment_name,
        run_name,
        user_prompt,
        metrics,
        config_path,
        mismatch_paths,
    )

    # The temp files have been uploaded to MLflow's own artifact store now, so
    # they don't need to stick around locally.
    for tmp_path in (config_path, *mismatch_paths):
        tmp_path.unlink(missing_ok=True)

    print(
        f"\nExperiment '{experiment_name}', run '{run_name}' complete. "
        f"Metrics logged to MLflow."
    )


def run_all_experiments(run_name: str) -> None:
    """Runs every experiment found under test_data/, skipping any that are
    missing required data instead of aborting the whole batch.

    Args:
        run_name (str): Name of this run, used to tag the saved outputs and
            the MLflow run for every experiment.
    """
    experiment_names = discover_experiment_names()
    if not experiment_names:
        raise ValueError(f"No experiment subdirectories found in {TEST_DATA_DIR}.")

    logger.info(f"Running {len(experiment_names)} experiment(s): {experiment_names}")
    for experiment_name in experiment_names:
        try:
            run_experiment(experiment_name, run_name)
        except ExperimentDataMissing as e:
            # Only data the user needs to add (missing prompt/ground
            # truth/test files) is skipped - anything else (backend errors,
            # extraction failures, ...) still propagates and stops the batch.
            logger.error(f"Skipping experiment '{experiment_name}': {e}")
            log_skipped_to_mlflow(experiment_name, run_name, reason=str(e))


def main() -> None:
    # experiment_name and run_name are the only CLI args - everything else
    # (prompt, test data, ground truth) is looked up from disk by convention.
    parser = argparse.ArgumentParser(
        description="Run and evaluate data_extractor agent experiment(s)."
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help=(
            "Name of the experiment (subfolder of test_data/). "
            "If omitted, every experiment under test_data/ is run."
        ),
    )
    parser.add_argument(
        "--run_name",
        type=str,
        required=True,
        help="Name of this run, used to tag the saved outputs.",
    )
    args = parser.parse_args()

    if args.experiment_name is None:
        run_all_experiments(args.run_name)
    else:
        run_experiment(args.experiment_name, args.run_name)


if __name__ == "__main__":
    main()
