"""Data extractor evaluation implementation."""

from src.experiments import EVALUATION_DIR, TEST_DATA_DIR

# One test data directory per agent, so running all experiments of one agent
# never picks up the other agent's experiments.
EXTRACTOR_TEST_DATA_DIR = TEST_DATA_DIR / "extractor"
EXTRACTED_DATA_DIR = EVALUATION_DIR / "extracted_data"
