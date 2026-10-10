import os
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

    def _request(self, method: str, path: str, **kwargs: object) -> requests.Response:
        read_timeout = 300 if "files" in kwargs or "/download_" in path else 60
        if path.strip("/") in {"run_agents", "run_single_agent"}:
            try:
                read_timeout = float(
                    os.environ.get("STREAMLIT_AGENT_READ_TIMEOUT_SECONDS", "1800")
                )
                if not 0 < read_timeout < float("inf"):
                    raise ValueError
            except ValueError as error:
                raise RuntimeError(
                    "Agent read timeout must be a positive number."
                ) from error
        try:
            return requests.request(
                method, f"{self._url}/{path}", timeout=(5, read_timeout), **kwargs
            )
        except requests.Timeout as error:
            raise RuntimeError(
                "Backend request timed out. Agent work may still be running; "
                "do not retry immediately."
            ) from error
        except requests.RequestException as error:
            raise RuntimeError(
                "Cannot reach the backend. Check that it is running."
            ) from error

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
        if params:
            response = self._request("GET", path, params=params)
        else:
            response = self._request("GET", path)

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

        response = self._request("POST", path)
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
            response = self._request("POST", path, json=data)
        elif payload_type == "params":
            response = self._request("POST", path, params=data)
        elif payload_type == "data":
            response = self._request("POST", path, data=data)
        else:
            raise ValueError(f"Invalid payload_type: {payload_type}")

        response = self.raise_error("POST", BeResponse(response=response))

        return response.json()

    @dispatch(str, str, dict)
    def post(
        self,
        path: str,
        config_name: Literal["data_extractor", "analyser"],
        config: dict[str, Any],
    ) -> bool | None:
        """
        Sends a POST request to the backend with a configuration dictionary.

        :param path: Path to the endpoint on the backend server.
        :param config_name: Name of the configuration to be sent with the request.
        :param config: Configuration dictionary to be sent with the request.
        :return: Dictionary containing the response from the backend.
        """
        logger.info(f"Sending POST request to {self._url}/{path} with config {config}")

        response = self._request(
            "POST", path, params={"config_name": config_name}, json=config
        )

        result = self.raise_error("POST", BeResponse(response)).json()
        if result is not None and type(result) is not bool:
            raise RuntimeError(
                "Backend returned an invalid configuration-save response."
            )
        return result

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
        logger.info(
            f"Sending POST request to {self._url}/{path} "
            f"with params {params} and "
            f"data files {''.join(file.name for file in data_files)}."
        )
        if not data_files:
            raise ValueError("No data files provided.")

        files = [("files", data_file) for data_file in data_files]

        if params:
            response = self._request("POST", path, params=params, files=files)
        else:
            response = self._request("POST", path, files=files)

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
            response = self._request("POST", path, params=params, files=files)
        else:
            response = self._request("POST", path, files=files)

        return self.raise_error("POST", BeResponse(response)).json()
