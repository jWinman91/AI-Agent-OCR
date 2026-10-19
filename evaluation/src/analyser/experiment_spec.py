"""Resolves analyser experiments from eval_data/analyser/prompt.yaml.

Every top-level key of prompt.yaml is one experiment. It maps to the data file to
analyse (in eval_data/analyser/), the user prompt to try, and optionally a ground
truth file in the shared ground_truth/ directory:

    kontoauszug:
      data_file: kontoauszug.csv
      ground_truth: monthly_transaction_sum.png
      prompt: |
        Plot the monthly sum of incoming and outgoing transactions.

Several experiments can use the same data file with different prompts. A named
ground truth is required to exist; it is shown to the judge as a reference for a
good result - a .png reference plot, or a .csv with the expected analysis output.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from src.analyser import ANALYSER_TEST_DATA_DIR, DATA_FILE_SUFFIXES
from src.experiments import GROUND_TRUTH_DIR, PROMPT_FILE_NAME, ExperimentDataMissing

GROUND_TRUTH_SUFFIXES = (".png", ".csv")


@dataclass(frozen=True)
class AnalyserExperimentSpec:
    name: str
    input_file: Path
    user_prompt: str
    ground_truth_path: Path | None


def load_experiment_entries(
    test_data_dir: Path = ANALYSER_TEST_DATA_DIR,
) -> dict[str, Any]:
    """Loads all experiment entries from the analyser's prompt.yaml.

    Args:
        test_data_dir (Path): The analyser's test data directory.
    Returns:
        dict[str, Any]: Experiment name -> raw entry, in file order; empty if
        prompt.yaml is missing or empty.
    """
    prompt_path = test_data_dir / PROMPT_FILE_NAME
    if not prompt_path.exists():
        return {}
    with prompt_path.open("r") as f:
        # An empty prompt.yaml loads as None.
        entries = yaml.safe_load(f) or {}
    if not isinstance(entries, dict):
        raise ValueError(f"{prompt_path} must map experiment names to entries.")
    return entries


def load_experiment_spec(
    experiment_name: str,
    test_data_dir: Path = ANALYSER_TEST_DATA_DIR,
    ground_truth_dir: Path = GROUND_TRUTH_DIR,
) -> AnalyserExperimentSpec:
    """Loads and validates everything an analyser experiment needs.

    Called before any backend or judge call, so missing data is reported
    without wasting a real agent run.

    Args:
        experiment_name (str): Name of the experiment (key in prompt.yaml).
        test_data_dir (Path): The analyser's test data directory.
        ground_truth_dir (Path): The shared ground truth directory.
    Returns:
        AnalyserExperimentSpec: The experiment's data file, prompt and
        optional ground truth.
    Raises:
        ExperimentDataMissing: If the prompt.yaml entry is missing or invalid,
            or its data file or named ground truth file doesn't exist.
    """
    prompt_path = test_data_dir / PROMPT_FILE_NAME
    entry = load_experiment_entries(test_data_dir).get(experiment_name)
    if entry is None:
        raise ExperimentDataMissing(
            f"{prompt_path} must define a '{experiment_name}' entry with "
            f"'data_file' and 'prompt'."
        )
    if not (
        isinstance(entry, dict)
        and isinstance(entry.get("data_file"), str)
        and isinstance(entry.get("prompt"), str)
    ):
        raise ExperimentDataMissing(
            f"'{experiment_name}' in {prompt_path} must be a mapping with "
            f"'data_file' and 'prompt' keys (and an optional 'ground_truth')."
        )

    input_file = test_data_dir / entry["data_file"]
    if input_file.suffix.lower() not in DATA_FILE_SUFFIXES:
        raise ExperimentDataMissing(
            f"Data file {input_file.name} for '{experiment_name}' must be one of "
            f"{', '.join(DATA_FILE_SUFFIXES)}."
        )
    if not input_file.is_file():
        raise ExperimentDataMissing(f"Data file not found at {input_file}.")

    ground_truth_path = None
    if (ground_truth_name := entry.get("ground_truth")) is not None:
        ground_truth_path = ground_truth_dir / str(ground_truth_name)
        if ground_truth_path.suffix.lower() not in GROUND_TRUTH_SUFFIXES:
            raise ExperimentDataMissing(
                f"Ground truth {ground_truth_path.name} for '{experiment_name}' "
                f"must be one of {', '.join(GROUND_TRUTH_SUFFIXES)}."
            )
        if not ground_truth_path.is_file():
            raise ExperimentDataMissing(
                f"Ground truth file not found at {ground_truth_path}."
            )

    return AnalyserExperimentSpec(
        name=experiment_name,
        input_file=input_file,
        user_prompt=entry["prompt"],
        ground_truth_path=ground_truth_path,
    )
