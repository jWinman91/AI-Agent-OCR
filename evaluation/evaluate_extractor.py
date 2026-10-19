"""CLI tool to run and evaluate data_extractor agent experiments.

Runs the data extractor against the test files in
eval_data/extractor/{experiment_name}/, using the prompt for that experiment from
eval_data/extractor/prompt.yaml, then evaluates the result against the shared
ground_truth/{experiment_name}.csv and reports metrics.
Metrics, params, and output files are logged to MLflow (sqlite:///mlflow.db) under
an MLflow experiment named experiment_name and a run named run_name, so past runs
can be browsed and compared via `mlflow ui --backend-store-uri sqlite:///mlflow.db`.
Only the extracted data CSV is kept permanently on disk (extracted_data/); the
per-run config yaml and false positive/negative CSVs are written to the system
temp directory just long enough to be logged as MLflow artifacts, then removed.

If --experiment_name is omitted, every subdirectory of eval_data/extractor/ is run
in turn. In that mode, an experiment missing its prompt, ground truth, or test files is
logged as an error and skipped (with NaN metrics logged to MLflow) rather than
aborting the whole batch.

Usage:
    python evaluate_extractor.py --experiment_name body_weight --run_name gemma4_v1
    python evaluate_extractor.py --run_name gemma4_v1
"""

import argparse
from pathlib import Path
from typing import Any

import yaml
from loguru import logger
from src.experiments import (
    GROUND_TRUTH_DIR,
    check_experiment_ready,
    configure_mlflow,
    discover_experiment_names,
    flatten_metrics,
    load_user_prompt,
    log_run_to_mlflow,
    run_all_experiments,
)
from src.extractor import EXTRACTED_DATA_DIR, EXTRACTOR_TEST_DATA_DIR
from src.extractor.evaluate import ALL_OVERALL_METRIC_NAMES, DataExtractorEvaluator
from src.extractor.extract_data import DataExtractor


def log_to_mlflow(
    experiment_name: str,
    run_name: str,
    user_prompt: str,
    metrics: dict[str, Any],
    config_path: Path,
    mismatch_paths: list[Path],
) -> None:
    """Logs one extractor experiment run's params, metrics, and output files to
    MLflow.

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
    log_run_to_mlflow(
        experiment_name,
        run_name,
        params={
            "user_prompt": user_prompt,
            "model": config.get("model"),
            "model_provider": config.get("model_provider"),
        },
        metrics=flatten_metrics(metrics),
        artifact_paths=[
            EXTRACTED_DATA_DIR / f"{experiment_name}_{run_name}.csv",
            config_path,
            *mismatch_paths,
        ],
    )


def run_experiment(experiment_name: str, run_name: str) -> None:
    """Runs a full extract-and-evaluate experiment for the data_extractor agent.

    Args:
        experiment_name (str): Name of the experiment (subfolder of
            eval_data/extractor/).
        run_name (str): Name of this run, used to tag the saved outputs.
    """
    # Fail fast (before any backend call) if the prompt, ground truth, or test
    # files for this experiment are missing.
    check_experiment_ready(
        EXTRACTOR_TEST_DATA_DIR,
        experiment_name,
        ground_truth_path=GROUND_TRUTH_DIR / f"{experiment_name}.csv",
    )
    user_prompt = load_user_prompt(EXTRACTOR_TEST_DATA_DIR, experiment_name)

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
            "Name of the experiment (subfolder of eval_data/extractor/). "
            "If omitted, every experiment under eval_data/extractor/ is run."
        ),
    )
    parser.add_argument(
        "--run_name",
        type=str,
        required=True,
        help="Name of this run, used to tag the saved outputs.",
    )
    args = parser.parse_args()

    configure_mlflow()
    if args.experiment_name is None:
        run_all_experiments(
            discover_experiment_names(EXTRACTOR_TEST_DATA_DIR),
            args.run_name,
            run_experiment,
            [f"{name}_overall" for name in ALL_OVERALL_METRIC_NAMES],
        )
    else:
        run_experiment(args.experiment_name, args.run_name)


if __name__ == "__main__":
    main()
