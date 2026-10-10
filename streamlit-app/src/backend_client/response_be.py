from io import BytesIO
from typing import Any

from PIL import Image
from requests import Response


class BeResponse:
    response: Response
    response_json: dict

    def __init__(self, response: Response) -> None:
        self.response = response

    def is_error(self) -> bool:
        return not self.response.status_code == 200

    def reason(self) -> str:
        return self.response.reason

    def json(self) -> dict[str, Any] | list[Any] | str | int | float | bool | None:
        try:
            return self.response.json()
        except ValueError as error:
            raise RuntimeError("Backend returned invalid JSON.") from error

    def image_bytes(self) -> Image:
        """
        Returns the content of the response as Image
        """
        return Image.open(BytesIO(self.response.content))

    def data_bytes(self) -> bytes:
        """
        Returns the content of the response as bytes
        """
        return self.response.content
