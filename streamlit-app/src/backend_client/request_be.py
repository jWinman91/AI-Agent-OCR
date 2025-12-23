from typing import Any, List, Literal

import requests
from loguru import logger
from multipledispatch import dispatch
from src.backend_client.response_be import BeResponse
from streamlit.runtime.uploaded_file_manager import UploadedFile


class BeRequest:
    def __init__(
        self,
        ip: str = "127.0.0.1",
        port: int = 8000,
        protocol: str = "http",
    ) -> None:
        """
        Initializes the BeRequest with the backend server's IP, port, and protocol.

        :param ip: IP address of the backend server.
        :param port: Port number of the backend server.
        :param protocol: Protocol to use for the request (default is "http").
        """
        self._url = f"{protocol}://{ip}:{port}"

    def raise_error(
        self,
        request_type: Literal["post", "get"],
        response: BeResponse,
    ) -> BeResponse:
        """
        Checks if the response indicates an error and raises an exception if so.

        :param response: BeResponse object containing the response from the backend.
        :return: BeResponse object if no error is found.
        """
        if response.is_error():
            raise RuntimeError(
                f"{request_type} failed with reason: {response.reason()}"
            )
        else:
            return response

    def get(
        self,
        path: str,
        params: dict[str, str] | None = None,
        data_type: Literal["data", "image"] | None = None,
    ) -> bytes | dict[str, Any] | bool:
        """
        Sends a GET request to the backend server.

        :param path: Path to the endpoint on the backend server.
        :param params: Parameters to be sent with the request.
        :param data_type: Type of data expected in the response ("data", "image").
        :return: Dictionary containing the response from the backend.
        """
        logger.info(f"Sending GET request to {self._url}/{path}")
        url = f"{self._url}/{path}"
        if params:
            response = requests.get(url, params=params)
        else:
            response = requests.get(url)

        _ = self.raise_error("GET", BeResponse(response=response))

        if data_type == "data":
            return BeResponse(response=response).data_bytes()
        elif data_type == "image":
            return BeResponse(response=response).image_bytes()
        else:
            return BeResponse(response=response).json()

    @dispatch(str)
    def post(
        self,
        path: str,
    ) -> dict[str, Any] | bool:
        """
        Sends a POST request to the backend server without any params.

        :param path: Path to the endpoint on the backend server.
        :return: Dictionary containing the response from the backend.
        """
        logger.info(f"Sending POST request to {self._url}/{path} without params.")

        response = requests.post(f"{self._url}/{path}")
        response = self.raise_error("POST", BeResponse(response=response))

        return response.json()

    @dispatch(str, dict)
    def post(
        self,
        path: str,
        data: dict[str, str],
        payload_type: Literal["json", "params", "data"],
    ) -> dict[str, Any] | bool:
        """
        Sends a POST request to the backend server.

        :param path: Path to the endpoint on the backend server.
        :param params: Parameters to be sent with the request.
        :return: Dictionary containing the response from the backend.
        """
        logger.info(f"Sending POST request to {self._url}/{path} with data {data}")

        if payload_type == "json":
            response = requests.post(f"{self._url}/{path}", json=data)
        elif payload_type == "params":
            response = requests.post(f"{self._url}/{path}", params=data)
        elif payload_type == "data":
            response = requests.post(f"{self._url}/{path}", data=data)
        else:
            raise ValueError(f"Invalid payload_type: {payload_type}")

        response = self.raise_error("POST", BeResponse(response=response))

        return response.json()

    @dispatch(str, str, dict)
    def post(
        self,
        path: str,
        config_name: Literal["extractor", "plotter"],
        config: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Sends a POST request to the backend with a configuration dictionary.

        :param path: Path to the endpoint on the backend server.
        :param config_name: Name of the configuration to be sent with the request.
        :param config: Configuration dictionary to be sent with the request.
        :return: Dictionary containing the response from the backend.
        """
        logger.info(f"Sending POST request to {self._url}/{path} with config {config}")

        response = requests.post(
            f"{self._url}/{path}", params={"config_name": config_name}, json=config
        )

        return self.raise_error("POST", BeResponse(response))

    @dispatch(str, dict, list)
    def post(
        self,
        path: str,
        params: dict[str, str],
        data_files: List[UploadedFile],
    ) -> dict:
        """
        Sends a POST request to the backend with a user prompt and a data file.

        :param path: Path to the endpoint on the backend server.
        :param user_prompt: User's prompt to be sent with the request.
        :param data_files: Data files to be uploaded with the request.
        :return: Dictionary containing the response from the backend.
        """
        print(data_files[0])
        logger.info(
            f"Sending POST request to {self._url}/{path} "
            f"with params {params} and "
            f"data files {''.join(file.name for file in data_files)}."
        )
        if not data_files or len(data_files) < 2:
            raise ValueError("No data files provided.")

        files = [("files", data_file) for data_file in data_files]

        if params:
            response = requests.post(f"{self._url}/{path}", params=params, files=files)
        else:
            response = requests.post(f"{self._url}/{path}", files=files)

        return self.raise_error("POST", BeResponse(response)).json()

    @dispatch(str, dict, UploadedFile)
    def post(
        self,
        path: str,
        params: dict[str, str],
        data_file: UploadedFile,
    ) -> dict:
        """
        Sends a POST request to the backend with a user prompt and a data file.

        :param path: Path to the endpoint on the backend server.
        :param user_prompt: User's prompt to be sent with the request.
        :param data_file: Data file to be uploaded with the request.
        :return: Dictionary containing the response from the backend.
        """
        logger.info(
            f"Sending POST request to {self._url}/{path} "
            f"with params {params} and "
            f"data file {data_file.name}."
        )

        files = [("file", data_file)]
        if params:
            response = requests.post(f"{self._url}/{path}", params=params, files=files)
        else:
            response = requests.post(f"{self._url}/{path}", files=files)

        return self.raise_error("POST", BeResponse(response)).json()
