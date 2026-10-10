import ast
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import requests
from streamlit.testing.v1 import AppTest

FRONTEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FRONTEND))

from src.backend_client.model_retriever import (  # noqa: E402
    OPENAI_MODELS,
    get_ollama_models,
    get_openai_models,
)
from src.backend_client.request_be import BeRequest  # noqa: E402
from src.parent_run import ParentRun  # noqa: E402


def class_source(filename: str) -> str:
    tree = ast.parse((FRONTEND / "src" / filename).read_text())
    return ast.unparse(
        next(node for node in tree.body if isinstance(node, ast.ClassDef))
    )


class HttpTests(unittest.TestCase):
    @patch("requests.request")
    def test_save_contract(self, request: Mock) -> None:
        client = BeRequest()
        response = Mock(status_code=200)
        request.return_value = response
        for result in (True, False, None):
            response.json.return_value = result
            self.assertIs(client.post("upload_config", "analyser", {}), result)
        for result in ({"success": True}, "true", 1):
            response.json.return_value = result
            with self.assertRaises(RuntimeError):
                client.post("upload_config", "analyser", {})
        response.json.side_effect = ValueError("invalid JSON")
        with self.assertRaises(RuntimeError):
            client.post("upload_config", "analyser", {})

    @patch("requests.request")
    def test_timeouts_and_errors(self, request: Mock) -> None:
        client = BeRequest()
        request.return_value = Mock(status_code=200)
        client.get("get_config")
        self.assertEqual(request.call_args.kwargs["timeout"], (5, 60))
        client.get("df_path/data.csv/download_data", data_type="data")
        self.assertEqual(request.call_args.kwargs["timeout"], (5, 300))
        client.post("uploadimages", {}, [SimpleNamespace(name="image.png")])
        self.assertEqual(request.call_args.kwargs["timeout"], (5, 300))
        with patch.dict(os.environ, {"STREAMLIT_AGENT_READ_TIMEOUT_SECONDS": "2400"}):
            client.post("run_agents", {}, payload_type="data")
            self.assertEqual(request.call_args.kwargs["timeout"], (5, 2400))
        for exception in (requests.Timeout(), requests.ConnectionError()):
            request.side_effect = exception
            with self.assertRaises(RuntimeError):
                client.get("get_config")

    @patch("requests.get")
    def test_discovery(self, request: Mock) -> None:
        request.return_value = Mock()
        request.return_value.json.return_value = {"models": [{"name": "gemma4"}]}
        self.assertEqual(get_ollama_models(), ["gemma4"])
        self.assertEqual(request.call_args.kwargs["timeout"], (3, 10))
        for body in ({}, {"models": None}, {"models": [{}]}):
            request.return_value.json.return_value = body
            with self.assertRaises(RuntimeError):
                get_ollama_models()
        request.side_effect = requests.ConnectionError()
        with self.assertRaises(RuntimeError):
            get_ollama_models()

    @patch("requests.get")
    def test_openai_discovery(self, request: Mock) -> None:
        models = get_openai_models()
        self.assertEqual(models, list(OPENAI_MODELS))
        self.assertTrue(models)
        self.assertEqual(len(models), len(set(models)))
        models.clear()
        self.assertEqual(get_openai_models(), list(OPENAI_MODELS))
        request.assert_not_called()


class ResultTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ui = MagicMock()
        self.ui.session_state = {"df": "old data", "image": "old plot"}
        self.instance = ParentRun.__new__(ParentRun)
        self.instance._request_be = Mock()
        self.frame = pd.DataFrame({"total": [3]})
        self.instance._get_csv = Mock(return_value=self.frame)
        self.ui_patch = patch("src.parent_run.st", self.ui)
        self.ui_patch.start()
        self.addCleanup(self.ui_patch.stop)

    def test_generated_csv_and_absent_plots(self) -> None:
        for empty in (None, "", "."):
            self.instance._load_result(
                {
                    "df_file_path": "analysis.csv",
                    "data_file_path": "input.csv",
                    "plot_path": empty,
                }
            )
            self.instance._get_csv.assert_called_with("analysis.csv")
            self.instance._request_be.get.assert_not_called()
            self.assertIsNone(self.ui.session_state["image"])

    def test_plot_only_result_does_not_display_input_csv(self) -> None:
        self.instance._request_be.get.return_value = "generated plot"
        for empty in (None, "", "."):
            with self.subTest(df_file_path=empty):
                self.ui.reset_mock()
                self.instance._request_be.get.reset_mock()
                self.instance._load_result(
                    {
                        "df_file_path": empty,
                        "data_file_path": "input.csv",
                        "plot_path": "plot.png",
                    }
                )
                self.instance._get_csv.assert_not_called()
                self.assertIsNone(self.ui.session_state["df"])
                self.instance._request_be.get.assert_called_once_with(
                    path="plot_path/plot.png/download_plot", data_type="image"
                )
                self.assertEqual(self.ui.session_state["image"], "generated plot")
                self.ui.error.assert_not_called()
                self.instance._render_result()
                self.ui.image.assert_called_once()
                self.ui.dataframe.assert_not_called()
                self.ui.download_button.assert_not_called()

    def test_partial_downloads_do_not_mix_generations(self) -> None:
        self.instance._request_be.get.side_effect = RuntimeError("missing plot")
        self.instance._load_result(
            {"df_file_path": "analysis.csv", "plot_path": "plot.png"}
        )
        self.assertIs(self.ui.session_state["df"], self.frame)
        self.assertIsNone(self.ui.session_state["image"])
        self.instance._get_csv.side_effect = RuntimeError("missing CSV")
        self.instance._request_be.get.side_effect = None
        self.instance._request_be.get.return_value = "new plot"
        self.instance._load_result(
            {
                "df_file_path": "missing.csv",
                "data_file_path": "input.csv",
                "plot_path": "plot.png",
            }
        )
        self.assertIsNone(self.ui.session_state["df"])
        self.assertEqual(self.ui.session_state["image"], "new plot")
        self.instance._get_csv.assert_called_with("missing.csv")

    def test_render_both_and_no_output(self) -> None:
        self.ui.session_state.update(df=self.frame, image="plot")
        for _ in range(2):
            self.instance._render_result()
        self.assertEqual(self.ui.dataframe.call_count, 2)
        self.assertEqual(self.ui.image.call_count, 2)
        self.instance._load_result({"data_file_path": "input.csv"})
        self.ui.error.assert_called_with("No plot or analysis data generated.")
        self.instance._get_csv.assert_not_called()
        self.assertIsNone(self.ui.session_state["df"])
        self.ui.error.reset_mock()
        self.instance._load_result({"data_file_path": "extracted.csv"}, extractor=True)
        self.ui.error.assert_not_called()
        self.instance._get_csv.assert_called_once_with("extracted.csv")
        self.assertIs(self.ui.session_state["df"], self.frame)

    def test_upload_failure(self) -> None:
        self.instance._request_be.post.side_effect = RuntimeError("offline")
        self.assertFalse(self.instance._upload_files("uploadimages", {}, []))


class StreamlitTests(unittest.TestCase):
    def configuration_app(
        self,
        discovery: str = "return ['gemma4']",
        save: str = "True",
        initial_model: str | None = None,
    ) -> AppTest:
        script = """import streamlit as st
from typing import Literal
class BeRequest:
    def __init__(self, *args): pass
    def get(self, **kwargs):
        return dict(model_provider='openai', model='gpt-5', system_prompt_user='same')
    def post(self, *args, **kwargs):
        return SAVE_RESULT
def get_ollama_models():
    DISCOVERY
def get_openai_models():
    return ['gpt-5', 'gpt-4.1']
""".replace("SAVE_RESULT", save).replace("DISCOVERY", discovery)
        script += class_source("agent_configurations.py")
        script += "\nAgentConfigurations().build_page()"
        app = AppTest.from_string(script)
        if initial_model is not None:
            app.session_state["openai_model_selectbox_analyser"] = initial_model
        return app.run()

    def test_widgets_replacement_and_reset(self) -> None:
        app = self.configuration_app()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.text_input), 2)
        app.selectbox(key="model_provider_selectbox_analyser").select("ollama").run()
        self.assertFalse(app.exception)
        self.assertEqual(
            app.selectbox(key="ollama_model_selectbox_analyser").value, "gemma4"
        )
        app.button(key="reset_button_analyser").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(
            app.selectbox(key="model_provider_selectbox_analyser").value, "openai"
        )

    def test_openai_catalog_only(self) -> None:
        app = self.configuration_app()
        self.assertFalse(app.exception)
        self.assertEqual(app.text_input(key="api_key_analyser").value, "")
        selector = app.selectbox(key="openai_model_selectbox_analyser")
        self.assertEqual(selector.options, ["gpt-5", "gpt-4.1"])
        self.assertEqual(selector.value, "gpt-5")
        selector.select("gpt-4.1").run()
        self.assertFalse(app.exception)
        self.assertEqual(
            app.selectbox(key="openai_model_selectbox_analyser").value, "gpt-4.1"
        )
        app.text_input(key="api_key_analyser").input("different-test-key").run()
        self.assertFalse(app.exception)
        self.assertEqual(
            app.selectbox(key="openai_model_selectbox_analyser").value,
            "gpt-4.1",
        )
        self.assertEqual(
            [widget.label for widget in app.text_input], ["API Key", "API Key"]
        )
        app = self.configuration_app(initial_model="Enter model manually")
        self.assertFalse(app.exception)
        self.assertEqual(
            app.selectbox(key="openai_model_selectbox_analyser").value, "gpt-5"
        )

    def test_discovery_outage_and_save_failure(self) -> None:
        for discovery in ("return []", "raise RuntimeError('Ollama offline')"):
            app = self.configuration_app(discovery)
            app.selectbox(key="model_provider_selectbox_analyser").select(
                "ollama"
            ).run()
            self.assertFalse(app.exception)
            self.assertTrue(
                app.button(key="save_configuration_button_analyser").disabled
            )
            app.selectbox(key="model_provider_selectbox_analyser").select(
                "openai"
            ).run()
            self.assertFalse(app.exception)
            self.assertFalse(
                app.button(key="save_configuration_button_analyser").disabled
            )
        for result in ("False", "None"):
            app = self.configuration_app(save=result)
            app.button(key="save_configuration_button_analyser").click().run()
            self.assertFalse(app.exception)
            self.assertTrue(any("Failed to save" in error.value for error in app.error))

    def test_startup_failure(self) -> None:
        script = f"""import sys
sys.path.insert(0, {str(FRONTEND)!r})
from unittest.mock import patch
from src.parent_run import ParentRun
with patch('src.parent_run.BeRequest') as client:
    client.return_value.post.side_effect = RuntimeError('backend offline')
    ParentRun()
"""
        app = AppTest.from_string(script).run()
        self.assertFalse(app.exception)
        self.assertTrue(any("Cannot initialize" in error.value for error in app.error))


class WorkflowTests(unittest.TestCase):
    def test_retry_errors_and_cleanup(self) -> None:
        for filename, name, method, workflow, upload_key in (
            (
                "run_single_agent.py",
                "SingleAgent",
                "build_extractor_widget",
                "extractor",
                "uploaded_images",
            ),
            (
                "run_single_agent.py",
                "SingleAgent",
                "build_analyser_widget",
                "analyser",
                "uploaded_data_file",
            ),
            (
                "run_ai_agent_ocr.py",
                "RunAiAgentOcr",
                "build_user_prompt",
                "pipeline",
                "uploaded_images",
            ),
        ):
            definition = ast.parse(class_source(filename)).body[0]
            definition.bases = [ast.Name(id="object", ctx=ast.Load())]
            ui = MagicMock()
            ui.text_area.return_value = "Analyse data"
            ui.button.return_value = True
            ui.session_state = {
                f"previous_error_{workflow}": "original traceback",
                "previous_error_unrelated": "unrelated traceback",
                upload_key: "uploaded",
            }
            scope = {"st": ui, "logger": Mock()}
            module = ast.fix_missing_locations(
                ast.Module(body=[definition], type_ignores=[])
            )
            exec(compile(module, filename, "exec"), scope)
            instance = scope[name]()
            instance.build_upload_widget = Mock(return_value=True)
            instance.build_upload_data_widget = Mock(return_value=True)
            instance._request_be = Mock()
            instance._load_result = Mock()
            instance._render_result = Mock()
            instance._request_be.post.return_value = {"error_message": "new traceback"}
            getattr(instance, method)()
            self.assertEqual(
                ui.session_state[f"previous_error_{workflow}"], "new traceback"
            )
            instance._load_result.assert_not_called()
            instance._request_be.post.return_value = {"df_file_path": "analysis.csv"}
            instance._load_result.side_effect = RuntimeError("download error")
            getattr(instance, method)()
            self.assertIsNone(ui.session_state[f"previous_error_{workflow}"])
            self.assertEqual(
                ui.session_state["previous_error_unrelated"], "unrelated traceback"
            )
            self.assertEqual(
                ui.session_state[upload_key],
                None if upload_key == "uploaded_data_file" else [],
            )
            self.assertEqual(
                instance._request_be.post.call_args.args[1]["previous_error"],
                "new traceback",
            )
            ui.button.return_value = False
            getattr(instance, method)()
            instance._render_result.assert_called()


if __name__ == "__main__":
    unittest.main()
