import requests
from src.plot_agent_client.response_be import BeResponse
from streamlit.runtime.uploaded_file_manager import UploadedFile

from typing import List


class BeRequest:
    def __init__(self, ip: str = "127.0.0.1", port: int = 8000, protocol: str = "http"):
        """
        Initializes the BeRequest with the backend server's IP, port, and protocol.

        :param ip: IP address of the backend server.
        :param port: Port number of the backend server.
        :param protocol: Protocol to use for the request (default is "http").
        """
        self._url = f"{protocol}://{ip}:{port}"

    def get(self, path: str, image_file: bool = True) -> dict:
        """
        Sends a GET request to the backend server.

        :param path: Path to the endpoint on the backend server.
        :param image_file: Whether the response is expected to be an image file (default is True).
        :return: Dictionary containing the response from the backend.
        """
        response = requests.get(f"{self._url}/{path}")
        if response.status_code != 200:
            raise RuntimeError(f"GET request failed with status code {response.status_code}: {response.reason}")
        if not image_file:
            return BeResponse(response=response).data_bytes()
        else:
            return BeResponse(response=response).image_bytes()

    def post(self, path: str, user_prompt: str | None = None, data_files: List[UploadedFile] | None = None) -> dict:
        """
        Sends a POST request to the backend with a user prompt and a data file.

        :param path: Path to the endpoint on the backend server.
        :param user_prompt: User's prompt to be sent with the request.
        :param data_files: Data files to be uploaded with the request.
        :return: Dictionary containing the response from the backend.
        """
        if user_prompt is not None:
            print(f"Sending POST request to {self._url}/{path} with user prompt '{user_prompt}'")
            response = BeResponse(response=requests.post(f"{self._url}/{path}", data={"user_prompt": user_prompt}))
        elif data_files is not None and len(data_files) > 0:
            file_names = ", ". join([data_file.name for data_file in data_files])
            files = [("files", data_file) for data_file in data_files]
            print(f"Sending POST request to {self._url}/{path} with data file {file_names}.")
            response = BeResponse(response=requests.post(f"{self._url}/{path}", files=files))
        else:
            raise ValueError("Either data_file or user_prompt must be provided.")

        if response.is_error():
            raise RuntimeError("Something went wrong with the post request...")
        else:
            return response.json()
