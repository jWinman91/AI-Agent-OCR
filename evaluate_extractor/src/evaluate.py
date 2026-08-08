"""A module for evaluating the data extractor agent by comparing extracted data with
ground truth data."""

from pathlib import Path


class DataExtractorEvaluator:
    def __init__(
        self,
        extracted_data_dir: Path = Path("extracted_data"),
        ground_truth_data_dir: Path = Path("ground_truth_data"),
        false_values_dir: Path = Path("false_values"),
    ) -> None:
        self._EXTRACTED_DATA_DIR = extracted_data_dir
        self._GROUND_TRUTH_DATA_DIR = ground_truth_data_dir
        self._FALSE_VALUES_DIR = false_values_dir

    def evaluate_extracted_data(
        self,
        experiment_name: str,
    ) -> dict[str, float]:
        """Evaluates the extracted data for a given experiment against the ground truth
        data.

        For a given experiment, it compares the extracted data with the ground truth
        data and then returns a dictionary containing evaluation metrics such as
        accuracy, precision, recall, and F1 score.

        Args:
            experiment_name (str): Name of the experiment to evaluate.
        Returns:
            dict[str, float]: Dictionary containing evaluation metrics.
        """
        pass

    def save_false_values(
        self,
        experiment_name: str,
        run_name: str,
    ) -> None:
        """Saves the false values from the evaluation of the extracted data for a given
        experiment.

        For a given experiment, it compares the extracted data against the ground truth
        data and saves the false positives and false negatives into separate csv files.
        In the csv files, the false positives and false negatives are saved with their
        corresponding ground truth values.
        The false positives are saved in a file named
        {experiment_name}_{run_name}_false_positives.csv and the false negatives are
        saved in a file named {experiment_name}_{run_name}_false_negatives.csv inside
        the FALSE_VALUES_DIR directory.

        Args:
            experiment_name (str): Name of the experiment to save false values for.
            run_name (str): Name of the run to save false values for.
        """
        pass
