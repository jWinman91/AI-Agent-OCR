import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic_ai import BinaryContent  # noqa: E402
from src.agents.agents_factory import AgentsFactory  # noqa: E402
from src.agents.agents_runner import AgentRunner  # noqa: E402
from src.agents.output_validator import OutputValidator  # noqa: E402
from src.server.plot_server import execute_python_code  # noqa: E402
from src.utils.data_models import (  # noqa: E402
    AnalyserResponse,
    ConfigModelBuild,
    DataExtractorResponse,
    FixJsonResponse,
)
from src.utils.sqlite_db_handler import SqliteDBHandler  # noqa: E402


class AgentContracts(unittest.IsolatedAsyncioTestCase):
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
            image.write_bytes(b"image bytes")
            text.write_text("PDF text")
            result = await factory.run_agent(
                agent, response, "extract", image_url=image, text_file_url=text
            )
        self.assertIs(result, response)
        message = agent.run.call_args.args[0]
        self.assertIn("<EXTRACTED PDF TEXT>:\nPDF text", message[0])
        self.assertIsInstance(message[1], BinaryContent)
        self.assertEqual(message[1].data, b"image bytes")

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
