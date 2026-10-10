import sys
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image  # noqa: E402
from pydantic_ai import BinaryContent  # noqa: E402
from pydantic_ai.exceptions import ModelAPIError  # noqa: E402
from src.agents.agents_factory import AgentsFactory  # noqa: E402
from src.agents.agents_runner import AgentRunner  # noqa: E402
from src.agents.output_validator import OutputValidator  # noqa: E402
from src.server.plot_server import execute_python_code  # noqa: E402
from src.utils.data_models import (  # noqa: E402
    AnalyserResponse,
    AnalyserResult,
    ConfigModelBuild,
    DataDownloadResponse,
    DataDownloadResult,
    DataExtractorPrompt,
    DataExtractorResponse,
    FixJsonResponse,
    OrchestratorResponse,
)
from src.utils.sqlite_db_handler import SqliteDBHandler  # noqa: E402


class AgentContracts(unittest.IsolatedAsyncioTestCase):
    async def test_malformed_downloader_does_not_execute(self) -> None:
        factory = AgentsFactory.__new__(AgentsFactory)
        repair = AsyncMock(return_value=FixJsonResponse(fixed_json='{"data": []}'))
        factory.OUTPUT_VALIDATOR = OutputValidator(
            create_agent_func=Mock(),
            run_agent_func=repair,
            config=ConfigModelBuild(model="test", system_prompt="test"),
        )
        agent = Mock()
        agent.name = "data_downloader"
        agent.run = AsyncMock(return_value=SimpleNamespace(output='{"data": []}'))
        callback = Mock()
        result = await factory.run_agent(
            agent,
            DataDownloadResponse(download_request=[], file_name=Path("data.csv")),
            "request",
            tool_execution_func=callback,
            error_result_factory=lambda message: DataDownloadResult(
                data_file_path=None, error_message=message
            ),
        )
        self.assertTrue(result.error_message)
        callback.assert_not_called()

    async def test_validation_and_provider_failures_skip_execution(self) -> None:
        factory = AgentsFactory.__new__(AgentsFactory)
        factory.OUTPUT_VALIDATOR = OutputValidator(
            create_agent_func=Mock(),
            run_agent_func=AsyncMock(),
            config=ConfigModelBuild(model="test", system_prompt="test"),
        )
        example = AnalyserResponse(
            python_code="placeholder", code_summary="placeholder"
        )
        agent = Mock(name="agent")
        agent.name = "analyser"
        agent.run = AsyncMock(return_value=SimpleNamespace(output="missing tags"))
        callback = Mock()

        def failed(message: str) -> AnalyserResult:
            return AnalyserResult(
                df_file_path=None,
                data_file_path=None,
                plot_path=None,
                python_code="",
                code_summary="",
                error_message=message,
            )

        result = await factory.run_agent(
            agent,
            example,
            "request",
            tool_execution_func=callback,
            error_result_factory=failed,
        )
        self.assertIn("CODE and SUMMARY", result.error_message)
        callback.assert_not_called()
        plain = await factory.run_agent(agent, example, "request")
        self.assertTrue(plain.error_message)
        agent.run.side_effect = ModelAPIError(
            model_name="test", message="Connection error"
        )
        result = await factory.run_agent(
            agent,
            example,
            "request",
            tool_execution_func=callback,
            error_result_factory=failed,
        )
        self.assertIn("Connection error", result.error_message)
        callback.assert_not_called()

    async def test_expected_schema_is_enforced_after_repair(self) -> None:
        expected = OrchestratorResponse(
            analyser_prompt="analyse",
            data_download_prompt="",
            data_extractor_prompt=DataExtractorPrompt(
                user_prompt="extract", json_details={}
            ),
        )
        repair = AsyncMock(
            return_value=FixJsonResponse(fixed_json=expected.model_dump_json())
        )
        validator = OutputValidator(
            create_agent_func=Mock(),
            run_agent_func=repair,
            config=ConfigModelBuild(model="test", system_prompt="test"),
        )
        response = await validator.validate_output(
            '{"data": []}', "orchestrator", expected
        )
        self.assertIsInstance(response, OrchestratorResponse)
        repair.assert_awaited_once()
        repair.return_value = FixJsonResponse(fixed_json='{"data": []}')
        with self.assertRaises(ValueError):
            await validator.validate_output('{"data": []}', "orchestrator", expected)
        with self.assertRaises(ValueError):
            await validator.parse_json("[]", expected)

    async def test_no_code_failure_does_not_run_repair(self) -> None:
        runner = AgentRunner.__new__(AgentRunner)
        runner.CONFIG = {"analyser": Mock()}
        runner.agents = Mock()
        runner.agents.run_agent = AsyncMock(
            return_value=AnalyserResult(
                df_file_path=None,
                data_file_path=None,
                plot_path=None,
                python_code="",
                code_summary="",
                error_message="Provider unavailable",
            )
        )
        with patch("src.agents.agents_runner.get_dataset_metadata", return_value=""):
            result = await runner.run_analyser(
                "request", {"data_file_path": None, "downloaded_data_file_path": None}
            )
        self.assertIn("Provider unavailable", result.error_message)
        self.assertEqual(runner.agents.create_agent.call_count, 1)

    async def test_parsed_response_and_execution_callback(self) -> None:
        factory = AgentsFactory.__new__(AgentsFactory)
        response = AnalyserResponse(python_code="pass", code_summary="summary")
        factory.OUTPUT_VALIDATOR = Mock()
        factory.OUTPUT_VALIDATOR.validate_output = AsyncMock(return_value=response)
        agent = Mock()
        agent.name = "analyser"
        agent.run = AsyncMock(return_value=SimpleNamespace(output="<CODE>pass</CODE>"))
        parsed = await factory.run_agent(agent, response, "request")
        self.assertIs(parsed, response)
        callback = Mock(return_value=execute_python_code(response, None, None))
        result = await factory.run_agent(
            agent, response, "request", tool_execution_func=callback
        )
        callback.assert_called_once_with(response)
        self.assertEqual(result.python_code, "pass")
        self.assertIsNone(result.error_message)

    async def test_pdf_text_and_image_bytes(self) -> None:
        factory = AgentsFactory.__new__(AgentsFactory)
        response = DataExtractorResponse(data=[{"weight": 70.5}])
        agent = Mock()
        agent.name = "data_extractor"
        agent.run = AsyncMock(return_value=SimpleNamespace(output=response))
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "page.png"
            text = Path(directory) / "page.txt"
            content = BytesIO()
            Image.new("RGB", (2, 2)).save(content, format="PNG")
            image.write_bytes(content.getvalue())
            text.write_text("PDF text")
            result = await factory.run_agent(
                agent, response, "extract", image_url=image, text_file_url=text
            )
        self.assertIs(result, response)
        message = agent.run.call_args.args[0]
        self.assertIn("<EXTRACTED PDF TEXT>:\nPDF text", message[0])
        self.assertIsInstance(message[1], BinaryContent)
        self.assertEqual(message[1].data, content.getvalue())
        self.assertEqual(message[1].media_type, "image/png")

    async def test_jpeg_bytes_are_labelled_as_jpeg(self) -> None:
        factory = AgentsFactory.__new__(AgentsFactory)
        expected = DataExtractorResponse(data=[])
        agent = Mock()
        agent.name = "extractor"
        agent.run = AsyncMock(return_value=SimpleNamespace(output=expected))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "misleading.png"
            content = BytesIO()
            Image.new("RGB", (2, 2)).save(content, format="JPEG")
            path.write_bytes(content.getvalue())
            result = await factory.run_agent(agent, expected, "request", image_url=path)
        self.assertIsNone(result.error_message)
        self.assertEqual(agent.run.call_args.args[0][1].media_type, "image/jpeg")

    async def test_validator_json_tags_and_fix_agent(self) -> None:
        validator = OutputValidator(
            create_agent_func=Mock(),
            run_agent_func=AsyncMock(
                return_value=FixJsonResponse(fixed_json='{"data": []}')
            ),
            config=ConfigModelBuild(model="test", system_prompt="test"),
        )
        expected = DataExtractorResponse(data=[])
        self.assertEqual(
            await validator.parse_json('{"data": []}', expected), {"data": []}
        )
        self.assertEqual(await validator.parse_json("broken", expected), {"data": []})
        response = await validator.validate_output(
            "<CODE>pass</CODE><SUMMARY>summary</SUMMARY>",
            "analyser",
            AnalyserResponse(python_code="", code_summary=""),
        )
        self.assertIsInstance(response, AnalyserResponse)
        self.assertEqual(response.python_code, "pass")
        self.assertEqual(await validator.validate_output("plain", "test", ""), "plain")

    async def test_analyser_repair_order(self) -> None:
        runner = AgentRunner.__new__(AgentRunner)
        runner.CONFIG = {}
        factory = Mock()
        factory.create_agent.side_effect = lambda **kwargs: kwargs["agent_name"]
        codes = iter(["raise ValueError('original')", "pass"])

        async def run_agent(**kwargs: object) -> object:
            response = AnalyserResponse(python_code=next(codes), code_summary="test")
            return kwargs["tool_execution_func"](response)

        factory.run_agent = AsyncMock(side_effect=run_agent)
        runner.agents = factory
        runner.CONFIG = {"analyser": Mock(), "code_repair": Mock()}
        with patch("src.agents.agents_runner.get_dataset_metadata", return_value=""):
            result = await runner.run_analyser(
                "request", {"data_file_path": None, "downloaded_data_file_path": None}
            )
        self.assertIsNone(result.error_message)
        self.assertEqual(factory.run_agent.await_count, 2)
        repair = factory.run_agent.await_args_list[1].kwargs
        self.assertEqual(repair["agent"], "code_repair")
        self.assertIn("raise ValueError('original')", repair["user_prompt"])
        self.assertIn("ValueError: original", repair["user_prompt"])


class SqliteContracts(unittest.TestCase):
    def test_config_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            handler = SqliteDBHandler.__new__(SqliteDBHandler)
            handler._table_name = "configs"
            handler._db_path = str(Path(directory) / "config.db")
            handler._db_schema = {
                "config_name": "TEXT NOT NULL UNIQUE",
                "config_dict": "TEXT NOT NULL",
            }
            self.assertTrue(handler.create_table())
            self.assertTrue(handler.add_config({"model": "test"}, "analyser"))
            self.assertEqual(handler.get_config("analyser"), {"model": "test"})


if __name__ == "__main__":
    unittest.main()
