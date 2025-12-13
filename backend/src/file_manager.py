import os

from fastapi import UploadFile
from loguru import logger


class FileManager:
    """Class to manage file operations."""

    def __init__(self, data_dir: str, plot_dir: str, image_dir: str) -> None:
        """
        Initializes the FileManager class.

        :param data_dir: Directory to store data files.
        :param plot_dir: Directory to store plot files.
        """
        self.DATA_DIR = data_dir
        self.PLOT_DIR = plot_dir
        self.IMAGE_DIR = image_dir

        os.makedirs(self.DATA_DIR, exist_ok=True)
        os.makedirs(self.PLOT_DIR, exist_ok=True)
        os.makedirs(image_dir, exist_ok=True)

    def get_data_dir(self) -> str:
        """
        Returns the data directory path.

        :return: Data directory path.
        """
        return self.DATA_DIR

    def get_plot_dir(self) -> str:
        """
        Returns the plot directory path.

        :return: Plot directory path.
        """
        return self.PLOT_DIR

    def upload_images(self, files: list[UploadFile]) -> list[str]:
        """
        Simulates uploading images and returns their URLs.

        :param image_paths: List of image file paths to upload.
        :return: List of URLs for the uploaded images.
        """

        # delete * in image dir
        for file_name in os.listdir(self.IMAGE_DIR):
            os.remove(os.path.join(self.IMAGE_DIR, file_name))

        # save images to image dir
        uploaded_urls = []
        for file in files:
            file_location = os.path.join(self.IMAGE_DIR, str(file.filename))
            with open(file_location, "wb+") as file_object:
                file_object.write(file.file.read())
            logger.info(f"Image saved at {file_location}")
            uploaded_urls.append(file_location)

        return uploaded_urls

    def upload_data_file(self, file: UploadFile) -> str:
        """
        Uploads a data file to the data directory.

        :param file: The data file to upload.
        :return: The path where the data file is saved.
        """
        # delete * in data dir
        for existing_file in os.listdir(self.DATA_DIR):
            os.remove(os.path.join(self.DATA_DIR, existing_file))

        # save data file to data dir
        file_location = os.path.join(self.DATA_DIR, str(file.filename))
        with open(file_location, "wb+") as file_object:
            file_object.write(file.file.read())
        logger.info(f"Data file saved at {file_location}")
        return file_location
