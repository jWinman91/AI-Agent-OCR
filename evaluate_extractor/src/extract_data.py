"""A module for testing the data extractor agent by uploading test data and extracting
data based on user prompts."""

import tempfile
from pathlib import Path

import yaml
from fastapi import UploadFile
from src.backend_client.request_be import BeRequest
from src.backend_client.response_be import BeResponse

UPLOADABLE_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tiff"}


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
        self._be_request.post("update_agents")

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
        # load_files() globs the whole experiment folder, which can also contain
        # OS junk (.DS_Store) - only forward real data files.
        uploadable_files = [
            file
            for file in files
            if Path(file.filename).suffix.lower() in UPLOADABLE_SUFFIXES
        ]
        skipped_files = [file for file in files if file not in uploadable_files]
        for file in skipped_files:
            file.file.close()

        if not uploadable_files:
            raise ValueError("No uploadable files found.")

        try:
            # PDF-first routing: a single PDF goes to /uploadpdf (the backend
            # converts it into per-page images itself); anything else is a set
            # of already-rendered images that goes straight to /uploadimages.
            is_pdf = [
                file.filename.lower().endswith(".pdf") for file in uploadable_files
            ]
            if any(is_pdf):
                if len(uploadable_files) > 1:
                    raise ValueError(
                        "Only one PDF file can be uploaded at a time, "
                        f"found {len(uploadable_files)} files."
                    )
                return self._be_request.post(
                    "uploadpdf",
                    {"resize": "medium"},
                    uploadable_files[0],
                )

            return self._be_request.post(
                "uploadimages",
                {"resize": "medium"},
                uploadable_files,
            )
        finally:
            # Always release the open file handles from load_files(), whether
            # the upload succeeded or raised.
            for file in uploadable_files:
                file.file.close()

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
        # Same route/payload shape as the Streamlit "Run Data Extractor Agent"
        # button (streamlit-app/src/run_single_agent.py).
        response_json = self._be_request.post(
            "run_single_agent",
            {
                "user_prompt": user_prompt,
                "agent_name": "data_extractor",
                "previous_error": None,
            },
            payload_type="data",
        )

        if response_json.get("error_message"):
            raise RuntimeError(response_json["error_message"])

        return Path(response_json["data_file_path"])

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

        E.g., if
            experiment_name is "body_weight",
            run_name is "1",
        then
            the file will be saved as:
              EXTRACTED_DATA_DIR/body_weight_1.csv

        Args:
            data_file_path (Path): Path to the extracted data file from the backend.
            experiment_name (str): Name of the experiment from where the test data was
            loaded. Used to create a unique name for the saved file.
            run_name (str): Name of the run. Used to create a unique name for the saved
            file.
        """
        # Same download route the Streamlit app uses (parent_run.py's _get_csv),
        # but written straight to disk instead of parsed into a DataFrame.
        csv_bytes = self._be_request.get(
            path=f"df_path/{data_file_path}/download_data",
            params=None,
            data_type="data",
        )

        self._EXTRACTED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        output_path = self._EXTRACTED_DATA_DIR / f"{experiment_name}_{run_name}.csv"
        output_path.write_bytes(csv_bytes)

    def save_config(
        self,
        experiment_name: str,
        run_name: str,
    ) -> Path:
        """Gets the configuration for the data_extractor agents and saves it as a
        temporary yaml file.

        The configuration is retrieved from the backend using the /get_config route
        with the `config_name` set to `data_extractor`. The configuration is written
        to a uniquely named file under the system temp directory rather than a
        permanent local directory - the caller is expected to upload it as an
        MLflow artifact and then remove it.

        Args:
            experiment_name (str): Name of the experiment from where the test data
                was loaded. Used to create a unique name for the saved file.
            run_name (str): Name of the run. Used to create a unique name for the
                saved file.
        Returns:
            Path: Path to the saved config yaml file under the system temp directory.
        """
        config = self._be_request.get(
            path="get_config",
            params={
                "config_name": "data_extractor",
                "full_prompt": "true",
            },
            data_type=None,
        )

        output_path = (
            Path(tempfile.gettempdir()) / f"{experiment_name}_{run_name}_config.yaml"
        )
        with output_path.open("w") as f:
            yaml.safe_dump(config, f, sort_keys=False)
        return output_path
