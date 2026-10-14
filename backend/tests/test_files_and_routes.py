import os
import sys
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import AiAgentOcrApp  # noqa: E402
from src.file_manager import FileManager  # noqa: E402
from src.model_retriever import ModelRetriever  # noqa: E402
from src.preprocessor import Preprocessor  # noqa: E402


class FileRoutes(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.manager = FileManager(
            self.root / "data",
            self.root / "plots",
            self.root / "images",
            self.root / "texts",
        )
        self.server = AiAgentOcrApp.__new__(AiAgentOcrApp)
        self.server.FILE_MANAGER = self.manager
        self.server.MODEL_RETRIEVER = ModelRetriever()
        self.server.PREPROCESSOR = Preprocessor()
        self.server._image_urls = []
        self.server._text_file_urls = []
        self.server._data_file_path = None
        self.server.app = FastAPI()
        self.server._register_routes()
        self.client = TestClient(self.server.app)

    def test_startup_sets_ollama_url_before_agents_and_preserves_override(self) -> None:
        for environment, expected_url in (
            ({}, "http://localhost:11434/v1"),
            ({"OLLAMA_BASE_URL": "http://ollama:11434/v1"}, "http://ollama:11434/v1"),
        ):
            with self.subTest(environment=environment):

                def create_runner(**kwargs: object) -> MagicMock:
                    self.assertEqual(os.environ["OLLAMA_BASE_URL"], expected_url)
                    return MagicMock()

                with (
                    patch.dict(os.environ, environment, clear=True),
                    patch("app.ConfigBuilder"),
                    patch("app.ModelRetriever"),
                    patch("app.FileManager"),
                    patch("app.Preprocessor"),
                    patch("app.AgentRunner", side_effect=create_runner) as runner,
                ):
                    AiAgentOcrApp("127.0.0.1", 8000)
                    self.assertEqual(os.environ["OLLAMA_BASE_URL"], expected_url)
                    runner.assert_called_once()

    def test_model_catalog_routes(self) -> None:
        self.assertEqual(
            self.client.get("/get_model_providers/").json(), ["openai", "ollama"]
        )
        response = self.client.get("/get_models/", params={"model_provider": "openai"})
        self.assertEqual(response.status_code, 200)
        self.assertIn("gpt-5", response.json())
        self.assertEqual(
            self.client.get(
                "/get_models/", params={"model_provider": "missing"}
            ).status_code,
            404,
        )

    def test_model_retriever_takes_precedence_and_reports_failure(self) -> None:
        provider = self.server.MODEL_RETRIEVER.providers["ollama"]
        provider.available_models = ["static-model"]
        retriever = MagicMock(return_value=["local-model"])
        with patch.object(self.server.MODEL_RETRIEVER, "get_ollama_models", retriever):
            response = self.client.get(
                "/get_models/", params={"model_provider": "ollama"}
            )
            self.assertEqual(response.json(), ["local-model"])
            retriever.assert_called_once_with()
            retriever.side_effect = RuntimeError("Ollama is unavailable")
            response = self.client.get(
                "/get_models/", params={"model_provider": "ollama"}
            )
            self.assertEqual(response.status_code, 502)
            self.assertEqual(response.json()["detail"], "Ollama is unavailable")
        provider.retriever_function = "missing_retriever"
        self.assertEqual(
            self.client.get(
                "/get_models/", params={"model_provider": "ollama"}
            ).status_code,
            502,
        )

    def test_model_catalog_supports_new_static_provider(self) -> None:
        config = self.root / "models.yaml"
        config.write_text(
            "model_provider:\n  custom:\n    available_models: [custom-model]\n"
        )
        self.server.MODEL_RETRIEVER = ModelRetriever(config)
        self.assertEqual(self.client.get("/get_model_providers/").json(), ["custom"])
        self.assertEqual(
            self.client.get("/get_models/", params={"model_provider": "custom"}).json(),
            ["custom-model"],
        )

    def test_ollama_retriever_uses_backend_url_and_validates_response(self) -> None:
        response = MagicMock()
        for base_url in (
            "http://ollama:11434/",
            "http://ollama:11434/v1",
            "http://ollama:11434/v1/",
        ):
            with self.subTest(base_url=base_url):
                response.json.return_value = {"models": [{"name": "local-model"}]}
                with (
                    patch.dict("os.environ", {"OLLAMA_BASE_URL": base_url}),
                    patch(
                        "src.model_retriever.requests.get", return_value=response
                    ) as get,
                ):
                    self.assertEqual(
                        ModelRetriever.get_ollama_models(), ["local-model"]
                    )
                    get.assert_called_once_with(
                        "http://ollama:11434/api/tags", timeout=(3, 10)
                    )
                    response.json.return_value = {"models": [{"name": ""}]}
                    with self.assertRaisesRegex(RuntimeError, "invalid model list"):
                        ModelRetriever.get_ollama_models()

    def test_upload_paths_rejected_before_cleanup(self) -> None:
        previous = self.manager.IMAGE_DIR / "previous.png"
        previous.write_bytes(b"previous")
        for filename in (
            None,
            "",
            ".",
            "..",
            "../outside.png",
            "a/b.png",
            "a\\b.png",
            "/tmp/outside.png",
        ):
            with self.subTest(filename=filename), self.assertRaises(HTTPException):
                self.manager.upload_images(
                    [
                        UploadFile(filename="valid.png", file=BytesIO(b"valid")),
                        UploadFile(filename=filename, file=BytesIO(b"invalid")),
                    ]
                )
            self.assertEqual(previous.read_bytes(), b"previous")
        outside = self.root / "outside.png"
        outside.write_bytes(b"outside")
        link = self.manager.IMAGE_DIR / "link.png"
        link.symlink_to(outside)
        with self.assertRaises(HTTPException):
            self.manager.upload_images(
                [UploadFile(filename="link.png", file=BytesIO())]
            )
        self.assertEqual(outside.read_bytes(), b"outside")

    def test_download_roots_and_errors(self) -> None:
        for directory in (self.manager.DATA_DIR, self.manager.PLOT_DIR):
            path = directory / "result.csv"
            path.write_text("value\n1\n")
            response = self.client.get(f"/df_path/{path}/download_data")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.text, "value\n1\n")
        plot = self.manager.PLOT_DIR / "plot.png"
        plot.write_bytes(b"plot")
        self.assertEqual(
            self.client.get(f"/plot_path/{plot}/download_plot").status_code, 200
        )
        outside_dir = self.root / "plots-other"
        outside_dir.mkdir()
        outside = outside_dir / "outside.csv"
        outside.write_text("outside")
        self.assertEqual(
            self.client.get(f"/df_path/{outside}/download_data").status_code, 403
        )
        link = self.manager.DATA_DIR / "link.csv"
        link.symlink_to(outside)
        self.assertEqual(
            self.client.get(f"/df_path/{link}/download_data").status_code, 403
        )
        missing = self.manager.DATA_DIR / "missing.csv"
        self.assertEqual(
            self.client.get(f"/df_path/{missing}/download_data").status_code, 404
        )
        relative = Path("backend/tests/test_files_and_routes.py")
        self.assertEqual(
            self.client.get(f"/df_path/{relative}/download_data").status_code, 403
        )

    def test_photo_upload_clears_previous_pdf_text(self) -> None:
        self.server._text_file_urls = [self.manager.TEXT_DIR / "previous.txt"]
        image = BytesIO()
        Image.new("RGB", (4, 4)).save(image, format="PNG")
        response = self.client.post(
            "/uploadimages/",
            files=[("files", ("photo.PNG", image.getvalue(), "image/png"))],
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.server._text_file_urls, [])
        self.assertEqual(len(self.server._image_urls), 1)

    def test_excel_first_sheet_and_csv_unchanged(self) -> None:
        workbook = BytesIO()
        with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
            pd.DataFrame({"amount": [1, 2]}).to_excel(
                writer, sheet_name="First", index=False
            )
            pd.DataFrame({"other": [99]}).to_excel(
                writer, sheet_name="Second", index=False
            )
        response = self.client.post(
            "/uploaddatafile",
            files={"file": ("book.XLSX", workbook.getvalue())},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.server._data_file_path.name, "book.csv")
        self.assertEqual(
            pd.read_csv(self.server._data_file_path).to_dict("list"), {"amount": [1, 2]}
        )
        csv = b"amount\n3\n"
        self.client.post("/uploaddatafile", files={"file": ("input.csv", csv)})
        self.assertEqual(self.server._data_file_path.read_bytes(), csv)
        invalid = self.client.post(
            "/uploaddatafile", files={"file": ("bad.xlsx", b"bad")}
        )
        self.assertEqual(invalid.status_code, 422)
        self.assertEqual(self.server._data_file_path.read_bytes(), csv)

    def test_xls_uses_first_sheet(self) -> None:
        with patch(
            "src.preprocessor.pd.read_excel", return_value=pd.DataFrame({"value": [1]})
        ) as read:
            result = Preprocessor().normalize_data_file(
                UploadFile(filename="input.xls", file=BytesIO(b"workbook"))
            )
        self.assertEqual(result.filename, "input.csv")
        self.assertEqual(read.call_args.kwargs["sheet_name"], 0)

    def test_blank_pdf_pages_preserve_alignment(self) -> None:
        document = MagicMock()
        document.__enter__.return_value = document
        document.page_count = 3
        with (
            patch("src.preprocessor.fitz.open", return_value=document),
            patch(
                "src.preprocessor.pymupdf4llm.to_text", side_effect=["", "middle", ""]
            ),
        ):
            texts = Preprocessor().convert_pdf_to_text(
                UploadFile(filename="input.pdf", file=BytesIO(b"pdf"))
            )
        self.assertEqual(
            list(texts), [f"input.pdf_page_{page}.txt" for page in (1, 2, 3)]
        )
        self.assertEqual(list(texts.values()), ["", "middle", ""])
        document.__exit__.assert_called_once()

    def test_all_blank_pdf_pages_are_retained(self) -> None:
        document = MagicMock()
        document.__enter__.return_value = document
        document.page_count = 2
        with (
            patch("src.preprocessor.fitz.open", return_value=document),
            patch("src.preprocessor.pymupdf4llm.to_text", return_value=""),
        ):
            texts = Preprocessor().convert_pdf_to_text(
                UploadFile(filename="blank.pdf", file=BytesIO(b"pdf"))
            )
        self.assertEqual(list(texts.values()), ["", ""])
        self.assertEqual(len(texts), 2)


if __name__ == "__main__":
    unittest.main()
