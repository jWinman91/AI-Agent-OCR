"""Shared experiment helpers for the extractor and analyser evaluators.

An experiment is either a subdirectory of an agent's test data directory
holding the test files (extractor, e.g. eval_data/extractor/body_weight/) or a
single input file in it (analyser, e.g. eval_data/analyser/body_weight.csv), with
its user prompt stored under the experiment's key in that directory's shared
prompt.yaml. Each experiment run is logged to MLflow with a 1:1 mapping: MLflow
experiment = experiment_name (prefixed with "analyser/" for the analyser), MLflow
run = run_name.

All paths are resolved relative to the evaluation/ directory, so the CLIs work
regardless of the directory they are started from.
"""

import math
from collections.abc import Callable, Mapping
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
# With a sqlite tracking store, MLflow would otherwise put new experiments'
# artifacts into ./mlruns of whatever directory the CLI was started from.
MLFLOW_ARTIFACT_DIR = EVALUATION_DIR / "mlruns"
PROMPT_FILE_NAME = "prompt.yaml"


class ExperimentDataMissing(Exception):
    """Raised when an experiment is missing its prompt, ground truth, or test
    data files - i.e. data the user needs to add, not a run/backend failure."""


def configure_mlflow() -> None:
    """Points MLflow at the shared evaluation/mlflow.db tracking database."""
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
    logger.info(f"MLflow tracking URI set to {mlflow.get_tracking_uri()}")


def _set_mlflow_experiment(mlflow_experiment_name: str) -> None:
    """Activates an MLflow experiment, creating it with its artifacts stored
    under evaluation/mlruns/ on first use, so runs started from any directory
    keep their artifacts in one place. Existing experiments keep the artifact
    location they were created with.
    """
    if mlflow.get_experiment_by_name(mlflow_experiment_name) is None:
        mlflow.create_experiment(
            mlflow_experiment_name,
            artifact_location=(MLFLOW_ARTIFACT_DIR / mlflow_experiment_name).as_uri(),
        )
    mlflow.set_experiment(mlflow_experiment_name)


def discover_experiment_names(test_data_dir: Path) -> list[str]:
    """Lists every experiment subdirectory under an agent's test data directory.

    Args:
        test_data_dir (Path): The agent's test data directory, e.g.
            eval_data/extractor/.
    Returns:
        list[str]: Sorted experiment names (subdirectory names).
    """
    return sorted(path.name for path in test_data_dir.iterdir() if path.is_dir())


def discover_experiment_files(
    test_data_dir: Path, suffixes: tuple[str, ...]
) -> dict[str, Path]:
    """Lists every experiment input file directly under an agent's test data
    directory, for agents whose experiments are single files rather than
    subdirectories (e.g. one data file per analyser experiment).

    Args:
        test_data_dir (Path): The agent's test data directory, e.g.
            eval_data/analyser/.
        suffixes (tuple[str, ...]): Lower-case file suffixes that count as
            experiment inputs (e.g. (".csv", ".xlsx")); anything else, such as
            prompt.yaml, is ignored.
    Returns:
        dict[str, Path]: Experiment name (file stem) -> input file, sorted by
        name.
    Raises:
        ValueError: If two input files share a stem (e.g. a.csv and a.xlsx),
            since the experiment name would be ambiguous.
    """
    files: dict[str, Path] = {}
    for path in sorted(test_data_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        if path.stem in files:
            raise ValueError(
                f"Experiment name '{path.stem}' is ambiguous: both "
                f"{files[path.stem].name} and {path.name} exist in {test_data_dir}."
            )
        files[path.stem] = path
    return files


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
    mlflow_experiment_name: str,
    run_name: str,
    params: dict[str, Any],
    metrics: Mapping[str, float | None],
    artifact_paths: list[Path],
    tags: dict[str, str] | None = None,
) -> None:
    """Logs one experiment run's params, metrics, tags, and output files to
    MLflow.

    Args:
        mlflow_experiment_name (str): MLflow experiment to log to - the
            experiment name itself for the extractor, prefixed for the analyser
            (see mlflow_experiment_name).
        run_name (str): Name of this run; mapped 1:1 to an MLflow run of the
            same name, nested under the MLflow experiment above.
        params (dict[str, Any]): Run params (prompt, model, ...) to log.
        metrics (Mapping[str, float | None]): Flat metric name -> value dict (see
            flatten_metrics for nested metrics); None is logged as NaN.
        artifact_paths (list[Path]): Output files to attach as artifacts.
        tags (dict[str, str] | None): Optional run tags, e.g. error messages.
    """
    # The MLflow experiment is created on first use and reused on every later
    # call with the same name, so repeated runs of the same experiment
    # accumulate as multiple runs under one MLflow experiment.
    _set_mlflow_experiment(mlflow_experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        logger.info(
            f"Logging to MLflow: experiment '{mlflow_experiment_name}', "
            f"run '{run_name}' (run_id={run.info.run_id})..."
        )
        mlflow.log_params(params)
        mlflow.log_metrics(
            {
                name: math.nan if value is None else value
                for name, value in metrics.items()
            }
        )
        if tags:
            mlflow.set_tags(tags)
        logger.info("Logged params, metrics and tags to MLflow.")
        for artifact_path in artifact_paths:
            mlflow.log_artifact(str(artifact_path))
        logger.info(f"Logged {len(artifact_paths)} artifact(s) to MLflow.")


def log_skipped_to_mlflow(
    mlflow_experiment_name: str,
    run_name: str,
    reason: str,
    metric_names: list[str],
) -> None:
    """Logs a skipped experiment to MLflow with NaN metrics in place of real ones.

    Used in "run all experiments" mode when an experiment is missing required
    data and never actually ran, so there are no real metrics/params/artifacts
    to log - only a record that this run was attempted and why it was skipped.

    Args:
        mlflow_experiment_name (str): MLflow experiment to log to.
        run_name (str): Name of this run; mapped 1:1 to an MLflow run of the
            same name, nested under the MLflow experiment above.
        reason (str): Human-readable reason the experiment was skipped, logged
            as a tag so it's visible in the MLflow UI.
        metric_names (list[str]): Every summary metric name the evaluator can
            report, logged as NaN.
    """
    _set_mlflow_experiment(mlflow_experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        logger.info(
            "Logging skipped experiment to MLflow: experiment "
            f"'{mlflow_experiment_name}', run '{run_name}' "
            f"(run_id={run.info.run_id})..."
        )
        mlflow.set_tag("skipped_reason", reason)
        # Nothing was evaluated - log the full set of summary metrics, all NaN
        # (not None, which MLflow rejects; NaN is its native "missing value"
        # representation and shows as blank in the UI), so skipped and real
        # runs keep consistent columns.
        mlflow.log_metrics({name: math.nan for name in metric_names})
        logger.info("Logged NaN metrics and skipped_reason tag to MLflow.")


def mlflow_experiment_name(experiment_name: str, prefix: str = "") -> str:
    """Maps an experiment to its MLflow experiment name.

    The extractor uses no prefix (keeping its existing MLflow history), the
    analyser uses "analyser/" so experiments sharing a name with an extractor
    experiment (e.g. body_weight) land in a separate MLflow experiment.
    """
    return f"{prefix}{experiment_name}"


def run_all_experiments(
    experiment_names: list[str],
    run_name: str,
    run_experiment: Callable[[str, str], None],
    metric_names: list[str],
    mlflow_prefix: str = "",
) -> None:
    """Runs every given experiment, skipping any that are missing required
    data instead of aborting the whole batch.

    Args:
        experiment_names (list[str]): The agent's experiments, e.g. from
            discover_experiment_names or discover_experiment_files.
        run_name (str): Name of this run, used to tag the saved outputs and
            the MLflow run for every experiment.
        run_experiment (Callable[[str, str], None]): The agent's single
            experiment runner, called as run_experiment(experiment_name,
            run_name).
        metric_names (list[str]): Every summary metric name the evaluator can
            report, logged as NaN for skipped experiments.
        mlflow_prefix (str): Prefix for the MLflow experiment names (see
            mlflow_experiment_name).
    """
    if not experiment_names:
        raise ValueError("No experiments found.")

    logger.info(f"Running {len(experiment_names)} experiment(s): {experiment_names}")
    for experiment_name in experiment_names:
        try:
            run_experiment(experiment_name, run_name)
        except ExperimentDataMissing as e:
            # Only data the user needs to add (missing prompt/ground
            # truth/test files) is skipped - anything else (backend errors,
            # ...) still propagates and stops the batch.
            logger.error(f"Skipping experiment '{experiment_name}': {e}")
            log_skipped_to_mlflow(
                mlflow_experiment_name(experiment_name, mlflow_prefix),
                run_name,
                reason=str(e),
                metric_names=metric_names,
            )
