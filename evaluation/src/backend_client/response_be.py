from io import BytesIO
from typing import Any

from PIL import Image
from requests import Response


class BeResponse:
    response: Response
    response_json: dict[str, Any]

    def __init__(self, response: Response) -> None:
        self.response = response

    def is_error(self) -> bool:
        return not self.response.status_code == 200

    def reason(self) -> str:
        return self.response.reason

    def json(self) -> Any:  # noqa: ANN401 - shape depends on the endpoint
        return self.response.json()

    def image_bytes(self) -> Image.Image:
        """
        Returns the content of the response as Image
        """
        return Image.open(BytesIO(self.response.content))

    def data_bytes(self) -> bytes:
        """
        Returns the content of the response as bytes
        """
        return self.response.content
