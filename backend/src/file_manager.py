from pathlib import Path

from fastapi import UploadFile
from loguru import logger


class FileManager:
    """Class to manage file operations."""

    def __init__(
        self,
        data_dir: Path,
        plot_dir: Path,
        image_dir: Path,
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

        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.PLOT_DIR.mkdir(parents=True, exist_ok=True)
        self.IMAGE_DIR.mkdir(parents=True, exist_ok=True)

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

        :param image_paths: List of image file paths to upload.
        :return: List of URLs for the uploaded images.
        """

        # delete * in image dir
        self._clear_directory(self.IMAGE_DIR)

        # save images to image dir
        uploaded_urls = []
        for file in files:
            file_location = self.IMAGE_DIR / file.filename
            with file_location.open("wb+") as file_object:
                file_object.write(file.file.read())
            logger.info(f"Image saved at {file_location}")
            uploaded_urls.append(file_location)

        return uploaded_urls

    def upload_data_file(self, file: UploadFile) -> Path:
        """
        Uploads a data file to the data directory.

        :param file: The data file to upload.
        :return: The path where the data file is saved.
        """
        # delete * in data dir
        self._clear_directory(self.DATA_DIR)

        # save data file to data dir
        file_location = self.DATA_DIR / file.filename
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
        logger.info("Cleaned up data, plot, and image directories.")
