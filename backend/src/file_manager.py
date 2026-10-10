from pathlib import Path

from fastapi import HTTPException, UploadFile
from loguru import logger


class FileManager:
    """Class to manage file operations."""

    def __init__(
        self,
        data_dir: Path,
        plot_dir: Path,
        image_dir: Path,
        text_dir: Path,
    ) -> None:
        """
        Initializes the FileManager class.

        :param data_dir: Directory to store data files.
        :param plot_dir: Directory to store plot files.
        :param image_dir: Directory to store image files.
        """
        self.DATA_DIR = data_dir
        self.PLOT_DIR = plot_dir
        self.IMAGE_DIR = image_dir
        self.TEXT_DIR = text_dir

        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.PLOT_DIR.mkdir(parents=True, exist_ok=True)
        self.IMAGE_DIR.mkdir(parents=True, exist_ok=True)
        self.TEXT_DIR.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def validate_filename(filename: str | None) -> str:
        if (
            not filename
            or filename in {".", ".."}
            or "/" in filename
            or "\\" in filename
            or "\x00" in filename
            or Path(filename).is_absolute()
        ):
            raise HTTPException(status_code=422, detail="Invalid upload filename.")
        return filename

    @staticmethod
    def upload_path(directory: Path, filename: str | None) -> Path:
        path = directory / FileManager.validate_filename(filename)
        if not path.resolve().is_relative_to(directory.resolve()):
            raise HTTPException(
                status_code=422, detail="Upload path escapes its directory."
            )
        return path

    @staticmethod
    def download_path(filename: str, directories: list[Path]) -> Path:
        path = Path(filename).resolve()
        if not any(
            path.is_relative_to(directory.resolve()) for directory in directories
        ):
            raise HTTPException(
                status_code=403, detail="File is outside allowed directories."
            )
        if not path.is_file():
            raise HTTPException(status_code=404, detail="File not found.")
        return path

    @staticmethod
    def _clear_directory(directory: Path) -> None:
        for file_path in directory.iterdir():
            if file_path.is_file():
                file_path.unlink()

    def get_data_dir(self) -> Path:
        """
        Returns the data directory path.

        :return: Data directory path.
        """
        return self.DATA_DIR

    def get_plot_dir(self) -> Path:
        """
        Returns the plot directory path.

        :return: Plot directory path.
        """
        return self.PLOT_DIR

    def upload_images(self, files: list[UploadFile]) -> list[Path]:
        """
        Simulates uploading images and returns their URLs.

        :param files: List of UploadFile objects representing the images.
        :return: List of file paths for the uploaded images.
        """

        destinations = [
            self.upload_path(self.IMAGE_DIR, file.filename) for file in files
        ]
        # delete * in image dir
        self._clear_directory(self.IMAGE_DIR)

        # save images to image dir
        uploaded_urls = []
        for file, file_location in zip(files, destinations):
            with file_location.open("wb+") as file_object:
                file_object.write(file.file.read())
            logger.info(f"Image saved at {file_location}")
            uploaded_urls.append(file_location)

        return uploaded_urls

    def upload_txt_files(self, files: dict[str, str]) -> list[Path]:
        """
        Simulates uploading text files and returns their URLs.

        :param files: Dictionary where keys are filenames and values are file contents.
        :return: List of file paths for the uploaded text files.
        """

        destinations = {
            filename: self.upload_path(self.TEXT_DIR, filename) for filename in files
        }
        # delete * in text dir
        self._clear_directory(self.TEXT_DIR)

        # save text files to text dir
        uploaded_urls = []
        for filename, content in files.items():
            file_location = destinations[filename]
            with file_location.open("w") as file_object:
                file_object.write(content)
            logger.info(f"Text file saved at {file_location}")
            uploaded_urls.append(file_location)

        return uploaded_urls

    def upload_data_file(self, file: UploadFile) -> Path:
        """
        Uploads a data file to the data directory.

        :param file: The data file to upload.
        :return: The path where the data file is saved.
        """
        file_location = self.upload_path(self.DATA_DIR, file.filename)
        # delete * in data dir
        self._clear_directory(self.DATA_DIR)

        # save data file to data dir
        with file_location.open("wb+") as file_object:
            file_object.write(file.file.read())
        logger.info(f"Data file saved at {file_location}")
        return file_location

    def clean_up(self) -> None:
        """
        Cleans up the data, plot, and image directories by deleting all files.
        """
        self._clear_directory(self.DATA_DIR)
        self._clear_directory(self.PLOT_DIR)
        self._clear_directory(self.IMAGE_DIR)
        self._clear_directory(self.TEXT_DIR)
        logger.info("Cleaned up data, plot, image, and text directories.")
