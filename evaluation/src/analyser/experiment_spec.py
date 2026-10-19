"""Resolves an analyser experiment's input file, prompt and optional ground truth.

An analyser experiment is one data file in eval_data/analyser/ (experiment name =
file stem). Its entry in eval_data/analyser/prompt.yaml is either the prompt
itself or a mapping that also names a ground truth file in ground_truth/:

    kontoauszug:
      prompt: |
        Plot the monthly sum of incoming and outgoing transactions.
      ground_truth: monthly_transaction_sum.png
    body_weight: |
      Plot my body weight over time.

A named ground truth is required to exist; it is shown to the judge as a
reference for a good result - a .png reference plot, or a .csv with the expected
analysis output.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml
from src.analyser import ANALYSER_TEST_DATA_DIR, DATA_FILE_SUFFIXES
from src.experiments import (
    GROUND_TRUTH_DIR,
    PROMPT_FILE_NAME,
    ExperimentDataMissing,
    discover_experiment_files,
)

GROUND_TRUTH_SUFFIXES = (".png", ".csv")


@dataclass(frozen=True)
class AnalyserExperimentSpec:
    name: str
    input_file: Path
    user_prompt: str
    ground_truth_path: Path | None


def load_experiment_spec(
    experiment_name: str,
    test_data_dir: Path = ANALYSER_TEST_DATA_DIR,
    ground_truth_dir: Path = GROUND_TRUTH_DIR,
) -> AnalyserExperimentSpec:
    """Loads and validates everything an analyser experiment needs.

    Called before any backend or judge call, so missing data is reported
    without wasting a real agent run.

    Args:
        experiment_name (str): Name of the experiment (stem of its data file).
        test_data_dir (Path): The analyser's test data directory.
        ground_truth_dir (Path): The shared ground truth directory.
    Returns:
        AnalyserExperimentSpec: The experiment's input file, prompt and
        optional ground truth.
    Raises:
        ExperimentDataMissing: If the data file, the prompt entry or a named
            ground truth file is missing or invalid.
    """
    input_files = discover_experiment_files(test_data_dir, DATA_FILE_SUFFIXES)
    if experiment_name not in input_files:
        raise ExperimentDataMissing(
            f"No data file ({', '.join(DATA_FILE_SUFFIXES)}) named "
            f"'{experiment_name}' found in {test_data_dir}."
        )

    prompt_path = test_data_dir / PROMPT_FILE_NAME
    if not prompt_path.exists():
        raise ExperimentDataMissing(
            f"Prompt file not found at {prompt_path}. "
            f"Create it with a '{experiment_name}' key."
        )
    with prompt_path.open("r") as f:
        # An empty prompt.yaml loads as None.
        entries = yaml.safe_load(f) or {}
    if experiment_name not in entries:
        raise ExperimentDataMissing(
            f"{prompt_path} must define a '{experiment_name}' key "
            f"with the user prompt for that experiment."
        )

    entry = entries[experiment_name]
    ground_truth_name: str | None = None
    if isinstance(entry, str):
        user_prompt = entry
    elif isinstance(entry, dict) and isinstance(entry.get("prompt"), str):
        user_prompt = entry["prompt"]
        ground_truth_name = entry.get("ground_truth")
    else:
        raise ExperimentDataMissing(
            f"'{experiment_name}' in {prompt_path} must be a prompt string or a "
            f"mapping with a 'prompt' key (and an optional 'ground_truth')."
        )

    ground_truth_path = None
    if ground_truth_name is not None:
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
        input_file=input_files[experiment_name],
        user_prompt=user_prompt,
        ground_truth_path=ground_truth_path,
    )
