"""Analyser evaluation implementation."""

from src.experiments import TEST_DATA_DIR

# One test data directory per agent, so running all experiments of one agent
# never picks up the other agent's experiments.
ANALYSER_TEST_DATA_DIR = TEST_DATA_DIR / "analyser"
