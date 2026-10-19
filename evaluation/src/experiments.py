"""Shared experiment helpers for the extractor and analyser evaluators.

An experiment is a subdirectory of an agent's test data directory (e.g.
test_data/extractor/body_weight/) holding the test files, with its user prompt
stored under the experiment's key in that directory's shared prompt.yaml. Each
experiment run is logged to MLflow with a 1:1 mapping: MLflow experiment =
experiment_name, MLflow run = run_name.

All paths are resolved relative to the evaluation/ directory, so the CLIs work
regardless of the directory they are started from.
"""

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import mlflow
import yaml
from loguru import logger

EVALUATION_DIR = Path(__file__).resolve().parent.parent
TEST_DATA_DIR = EVALUATION_DIR / "eval_data"
# Shared by both evaluators: ground_truth/{experiment_name}.* holds whatever
# reference an experiment is scored against.
GROUND_TRUTH_DIR = EVALUATION_DIR / "ground_truth"
MLFLOW_DB_PATH = EVALUATION_DIR / "mlflow.db"
PROMPT_FILE_NAME = "prompt.yaml"


class ExperimentDataMissing(Exception):
    """Raised when an experiment is missing its prompt, ground truth, or test
    data files - i.e. data the user needs to add, not a run/backend failure."""


def configure_mlflow() -> None:
    """Points MLflow at the shared evaluation/mlflow.db tracking database."""
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
    logger.info(f"MLflow tracking URI set to {mlflow.get_tracking_uri()}")


def discover_experiment_names(test_data_dir: Path) -> list[str]:
    """Lists every experiment subdirectory under an agent's test data directory.

    Args:
        test_data_dir (Path): The agent's test data directory, e.g.
            test_data/extractor/.
    Returns:
        list[str]: Sorted experiment names (subdirectory names).
    """
    return sorted(path.name for path in test_data_dir.iterdir() if path.is_dir())


def load_user_prompt(test_data_dir: Path, experiment_name: str) -> str:
    """Loads the user prompt for an experiment from the agent's shared
    prompt.yaml file.

    Args:
        test_data_dir (Path): The agent's test data directory containing
            prompt.yaml.
        experiment_name (str): Name of the experiment to load the prompt for.
    Returns:
        str: The user prompt string.
    """
    # All experiment prompts of one agent live in one file, keyed by
    # experiment_name, rather than one prompt.yaml per experiment folder -
    # easier to see/edit them all at a glance and keep in sync with the
    # experiment subfolders.
    prompt_path = test_data_dir / PROMPT_FILE_NAME
    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Prompt file not found at {prompt_path}. "
            f"Create it with a '{experiment_name}' key."
        )

    with prompt_path.open("r") as f:
        prompts = yaml.safe_load(f)

    if not prompts or experiment_name not in prompts:
        raise ValueError(
            f"{prompt_path} must define a '{experiment_name}' key "
            f"with the user prompt for that experiment."
        )

    return str(prompts[experiment_name])


def check_experiment_ready(
    test_data_dir: Path, experiment_name: str, ground_truth_path: Path
) -> None:
    """Validates that an experiment has everything needed to run and evaluate.

    Checked upfront (before any backend call) so a missing prompt, ground
    truth, or test file is reported immediately instead of wasting a real
    agent call on an experiment that can't be evaluated anyway.

    Args:
        test_data_dir (Path): The agent's test data directory.
        experiment_name (str): Name of the experiment to validate.
        ground_truth_path (Path): The reference file this experiment is scored
            against.
    Raises:
        ExperimentDataMissing: If the prompt, ground truth, or test data files
            for this experiment are missing.
    """
    try:
        load_user_prompt(test_data_dir, experiment_name)
    except (FileNotFoundError, ValueError) as e:
        raise ExperimentDataMissing(str(e)) from e

    if not ground_truth_path.exists():
        raise ExperimentDataMissing(
            f"Ground truth file not found at {ground_truth_path}."
        )

    experiment_dir = test_data_dir / experiment_name
    if not experiment_dir.is_dir() or not any(
        path.is_file() for path in experiment_dir.iterdir()
    ):
        raise ExperimentDataMissing(
            f"No test data files found for experiment '{experiment_name}' "
            f"in {experiment_dir}."
        )


def flatten_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    """Flattens a per-item + overall metrics dict into MLflow's flat
    {name: float} metric namespace, suffixing each metric name with the item
    it belongs to (or "overall" for the summary).

    Every top-level key other than "overall" is treated as a group of
    {item: {metric: value}} (e.g. the extractor's "fields"), and each item only
    contributes the metric keys it actually has. E.g. {"fields": {"Saldo":
    {"accuracy": 0.9, ...}, "body_weight": {"mae": 1.2}}, "overall": {...}}
    becomes {"accuracy_Saldo": 0.9, ..., "mae_body_weight": 1.2,
    "accuracy_overall": ..., "mae_overall": ...}.

    A metric value of None is logged as NaN instead, since MLflow rejects None
    for metric values but accepts NaN as its native "missing value"
    representation (renders as blank in the UI).

    Args:
        metrics (dict[str, Any]): Metrics dict with an "overall" summary and one
            or more per-item groups.
    Returns:
        dict[str, float]: Flat metric name -> value dict ready for
        mlflow.log_metrics.
    """
    flat: dict[str, float] = {}
    for group, items in metrics.items():
        if group == "overall":
            continue
        for item, item_metrics in items.items():
            for metric_name, value in item_metrics.items():
                flat[f"{metric_name}_{item}"] = math.nan if value is None else value
    for metric_name, value in metrics["overall"].items():
        flat[f"{metric_name}_overall"] = math.nan if value is None else value
    return flat


def log_run_to_mlflow(
    experiment_name: str,
    run_name: str,
    params: dict[str, Any],
    metrics: dict[str, Any],
    artifact_paths: list[Path],
) -> None:
    """Logs one experiment run's params, metrics, and output files to MLflow.

    Args:
        experiment_name (str): Name of the experiment; mapped 1:1 to an MLflow
            experiment of the same name.
        run_name (str): Name of this run; mapped 1:1 to an MLflow run of the
            same name, nested under the MLflow experiment above.
        params (dict[str, Any]): Run params (prompt, model, ...) to log.
        metrics (dict[str, Any]): Nested metrics dict, flattened with
            flatten_metrics before logging.
        artifact_paths (list[Path]): Output files to attach as artifacts.
    """
    # set_experiment creates the MLflow experiment on first use and reuses it on
    # every later call with the same name, so repeated runs of the same
    # experiment_name accumulate as multiple runs under one MLflow experiment.
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        logger.info(
            f"Logging to MLflow: experiment '{experiment_name}', run '{run_name}' "
            f"(run_id={run.info.run_id})..."
        )
        mlflow.log_params(params)
        mlflow.log_metrics(flatten_metrics(metrics))
        logger.info("Logged params and metrics to MLflow.")
        for artifact_path in artifact_paths:
            mlflow.log_artifact(str(artifact_path))
        logger.info(f"Logged {len(artifact_paths)} artifact(s) to MLflow.")


def log_skipped_to_mlflow(
    experiment_name: str,
    run_name: str,
    reason: str,
    overall_metric_names: list[str],
) -> None:
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
        overall_metric_names (list[str]): Every metric name the evaluator can
            report in its "overall" summary.
    """
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        logger.info(
            f"Logging skipped experiment to MLflow: experiment '{experiment_name}', "
            f"run '{run_name}' (run_id={run.info.run_id})..."
        )
        mlflow.set_tag("skipped_reason", reason)
        # Nothing was evaluated, so there's nothing to report per item - log the
        # full superset of possible "overall" metrics, all NaN (not None, which
        # MLflow rejects; NaN is its native "missing value" representation and
        # shows as blank in the UI), so skipped and real runs keep consistent
        # overall columns.
        mlflow.log_metrics(
            {f"{name}_overall": math.nan for name in overall_metric_names}
        )
        logger.info("Logged NaN metrics and skipped_reason tag to MLflow.")


def run_all_experiments(
    test_data_dir: Path,
    run_name: str,
    run_experiment: Callable[[str, str], None],
    overall_metric_names: list[str],
) -> None:
    """Runs every experiment found under an agent's test data directory,
    skipping any that are missing required data instead of aborting the whole
    batch.

    Args:
        test_data_dir (Path): The agent's test data directory.
        run_name (str): Name of this run, used to tag the saved outputs and
            the MLflow run for every experiment.
        run_experiment (Callable[[str, str], None]): The agent's single
            experiment runner, called as run_experiment(experiment_name,
            run_name).
        overall_metric_names (list[str]): Every metric name the evaluator can
            report in its "overall" summary, logged as NaN for skipped
            experiments.
    """
    experiment_names = discover_experiment_names(test_data_dir)
    if not experiment_names:
        raise ValueError(f"No experiment subdirectories found in {test_data_dir}.")

    logger.info(f"Running {len(experiment_names)} experiment(s): {experiment_names}")
    for experiment_name in experiment_names:
        try:
            run_experiment(experiment_name, run_name)
        except ExperimentDataMissing as e:
            # Only data the user needs to add (missing prompt/ground
            # truth/test files) is skipped - anything else (backend errors,
            # agent failures, ...) still propagates and stops the batch.
            logger.error(f"Skipping experiment '{experiment_name}': {e}")
            log_skipped_to_mlflow(
                experiment_name,
                run_name,
                reason=str(e),
                overall_metric_names=overall_metric_names,
            )
