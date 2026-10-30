"""Resolves extractor experiments from eval_data/extractor/.

An experiment is a subdirectory of test files (images or one PDF), scored against
ground_truth/{experiment_name}.csv. Its entry in eval_data/extractor/prompt.yaml
holds the user prompt to try and optionally the data extractor's
domain_knowledge_prompt (the domain knowledge about the use case the user would
enter in the UI):

    body_weight:
      prompt: |
        Extract the body weight value given in kg from the image.
      domain_knowledge_prompt: |
        The values are always of the format XX.X kg.

Without domain_knowledge_prompt the data extractor runs with an empty one.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from src.experiments import GROUND_TRUTH_DIR, PROMPT_FILE_NAME, ExperimentDataMissing
from src.extractor import EXTRACTOR_TEST_DATA_DIR


@dataclass(frozen=True)
class ExtractorExperimentSpec:
    name: str
    user_prompt: str
    domain_knowledge_prompt: str = ""


def load_experiment_spec(
    experiment_name: str,
    test_data_dir: Path = EXTRACTOR_TEST_DATA_DIR,
    ground_truth_dir: Path = GROUND_TRUTH_DIR,
) -> ExtractorExperimentSpec:
    """Loads and validates everything an extractor experiment needs.

    Called before any backend call, so a missing prompt, ground truth, or test
    file is reported immediately instead of wasting a real agent call on an
    experiment that can't be evaluated anyway.

    Args:
        experiment_name (str): Name of the experiment (subdirectory of
            test_data_dir and key in its prompt.yaml).
        test_data_dir (Path): The extractor's test data directory.
        ground_truth_dir (Path): The shared ground truth directory.
    Returns:
        ExtractorExperimentSpec: The experiment's prompt and
        domain_knowledge_prompt.
    Raises:
        ExperimentDataMissing: If the prompt.yaml entry is missing or invalid,
            or the ground truth or test data files don't exist.
    """
    # All experiment prompts live in one file, keyed by experiment_name, rather
    # than one prompt.yaml per experiment folder - easier to see/edit them all
    # at a glance and keep in sync with the experiment subfolders.
    prompt_path = test_data_dir / PROMPT_FILE_NAME
    entries: Any = {}
    if prompt_path.exists():
        with prompt_path.open("r") as f:
            # An empty prompt.yaml loads as None.
            entries = yaml.safe_load(f) or {}
    entry = entries.get(experiment_name) if isinstance(entries, dict) else None
    if entry is None:
        raise ExperimentDataMissing(
            f"{prompt_path} must define a '{experiment_name}' entry with 'prompt'."
        )
    if not (
        isinstance(entry, dict)
        and isinstance(entry.get("prompt"), str)
        # A key without a value loads as None.
        and isinstance(entry.get("domain_knowledge_prompt") or "", str)
    ):
        raise ExperimentDataMissing(
            f"'{experiment_name}' in {prompt_path} must be a mapping with a "
            f"'prompt' key (and an optional 'domain_knowledge_prompt' key)."
        )

    ground_truth_path = ground_truth_dir / f"{experiment_name}.csv"
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

    return ExtractorExperimentSpec(
        name=experiment_name,
        user_prompt=entry["prompt"],
        domain_knowledge_prompt=entry.get("domain_knowledge_prompt") or "",
    )
