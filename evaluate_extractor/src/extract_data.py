"""A module for testing the data extractor agent by uploading test data and extracting
data based on user prompts."""

from pathlib import Path

from fastapi import UploadFile
from src.backend_client.request_be import BeRequest
from src.backend_client.response_be import BeResponse


class DataExtractor:
    def __init__(
        self,
        ip: str = "127.0.0.1",
        port: int = 8000,
        protocol: str = "http",
        extracted_data_dir: Path = Path("extracted_data"),
        test_data_dir: Path = Path("test_data"),
    ) -> None:
        self._be_request = BeRequest(ip=ip, port=port, protocol=protocol)

        # update agents
        self._be_request.post(path="update_agents")

        # some directories as constants
        self._EXTRACTED_DATA_DIR = extracted_data_dir
        self._TEST_DATA_DIR = test_data_dir

    def load_files(self, experiment_name: Path) -> list[UploadFile]:
        """Loads all files from the directory inside TEST_DATA_DIR and returns a list of
          UploadFile objects.

        Uses glob to find all files in the directory and creates an UploadFile object
        for each file. The files can then be sent to the backend for processing.

        Args:
            experiment_name (Path): Name of the experiment containing the files to load.
        Returns:
            list[UploadFile]: List of UploadFile objects containing the loaded data.

        """
        files = list((self._TEST_DATA_DIR / experiment_name).glob("*"))
        upload_files = []
        for file in files:
            if file.is_file():
                upload_file = UploadFile(filename=file.name, file=open(file, "rb"))
                upload_files.append(upload_file)
        return upload_files

    def upload_files(self, files: list[UploadFile]) -> BeResponse:
        """Uploads the given list of UploadFile objects to the backend.

        It uses the BeRequest class to send a POST request to the backend with the
        files. The files are either send to /uploadimages or /uploadpdf depending on the
        file type. The backend will then process the files and return a response.

        Args:
            files (list[UploadFile]): List of UploadFile objects to upload.
        Returns:
            BeResponse: Response from the backend after uploading the files.
        """
        pass

    def extract_data(self, user_prompt: str) -> Path:
        """Extracts data from the uploaded files based on the user prompt.

        First the user prompt is sent to the /run_single_agent endpoint of the backend
        with `agent_name` set to `data_extractor`.
        The backend will then process the uploaded files and extract the data based on
        the user prompt. This route returns a JSON response with a file path to the
        data.
        This function then returns the file path to the extracted data from this JSON.

        Args:
            user_prompt (str): User prompt to guide the data extraction.
        Returns:
            Path: Path to the extracted data file.
        """
        pass

    def save_extracted_data(
        self,
        data_file_path: Path,
        experiment_name: str,
        run_name: str,
    ) -> None:
        """Saves the extracted data to the specified path inside the EXTRACTED_DATA_DIR
        directory.

        It uses the /df_path/{df_path:path}/download_data route of the backend to
        download the extracted data and saves it into the EXTRACTED_DATA_DIR directory.
        The experiment_name parameter and the run_name parameter are used to create a
        unique name for the saved file. The file is saved in CSV format.

        E.g., if experiment_name is "body_weight" and run_name is "1",
        the file will be saved as EXTRACTED_DATA_DIR/body_weight_1.csv

        Args:
            data_file_path (Path): Path to the extracted data file from the backend.
            experiment_name (str): Name of the experiment from where the test data was
            loaded. Used to create a unique name for the saved file.
            run_name (str): Name of the run. Used to create a unique name for the saved
            file.
        """
        pass
