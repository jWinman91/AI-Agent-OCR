"""Analyser evaluation implementation."""

from pathlib import Path

from src.experiments import TEST_DATA_DIR

# One test data directory per agent, so running all experiments of one agent
# never picks up the other agent's experiments.
ANALYSER_TEST_DATA_DIR = TEST_DATA_DIR / "analyser"
JUDGE_CONFIG_PATH = Path(__file__).resolve().parent / "judge_config.yaml"

# An analyser experiment's data file must have one of the suffixes the backend's
# /uploaddatafile accepts.
DATA_FILE_SUFFIXES = (".csv", ".xlsx", ".xls")

# Analyser experiments can share a name with an extractor experiment (e.g.
# body_weight), so they are logged to their own MLflow experiments.
MLFLOW_PREFIX = "analyser/"
