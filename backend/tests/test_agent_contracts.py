import sys
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image  # noqa: E402
from pydantic_ai import BinaryContent  # noqa: E402
from pydantic_ai.exceptions import ModelAPIError  # noqa: E402
from src.agents.agents_factory import AgentsFactory  # noqa: E402
from src.agents.agents_runner import AgentRunner  # noqa: E402
from src.agents.output_validator import OutputValidator  # noqa: E402
from src.server.analysis_policy import compile_analysis  # noqa: E402
from src.server.plot_server import execute_python_code  # noqa: E402
from src.utils.data_models import (  # noqa: E402
    AnalyserResponse,
    AnalyserResult,
    ConfigModelBuild,
    ConfigModelDB,
    ConfigModelUser,
    DataDownloadResponse,
    DataDownloadResult,
    DataExtractorPrompt,
    DataExtractorResponse,
    FixJsonResponse,
    OrchestratorResponse,
)
from src.utils.sqlite_db_handler import SqliteDBHandler  # noqa: E402


class AgentContracts(unittest.IsolatedAsyncioTestCase):
    def test_create_agent_passes_ollama_config_and_structured_output(self) -> None:
        factory = AgentsFactory.__new__(AgentsFactory)
        factory.AGENT_NAMES = ["analyser"]
        config = ConfigModelBuild(
            model="gemma4:latest",
            model_provider="ollama",
            output_type=True,
            model_settings={"temperature": 0.2, "top_p": 0.9},
            system_prompt="Analyse {dataset}",
        )
        model = Mock()
        agent = Mock()

        with (
            patch("src.agents.agents_factory.infer_model", return_value=model) as infer,
            patch("src.agents.agents_factory.Agent", return_value=agent) as agent_ctor,
        ):
            result = factory.create_agent(
                "analyser",
                config,
                AnalyserResponse,
                system_prompt_args={"dataset": "transactions"},
            )
            infer.assert_called_once_with("ollama:gemma4:latest")

        self.assertIs(result, agent)
        self.assertEqual(agent.name, "analyser")
        agent_ctor.assert_called_once_with(
            model=model,
            system_prompt="Analyse transactions",
            model_settings={"temperature": 0.2, "top_p": 0.9},
            output_type=AnalyserResponse,
        )

    def test_create_agent_uses_named_provider_without_optional_output_type(
        self,
    ) -> None:
        factory = AgentsFactory.__new__(AgentsFactory)
        factory.AGENT_NAMES = ["analyser"]
        for provider_name in ("openai", "anthropic", "ollama"):
            with self.subTest(provider=provider_name):
                config = ConfigModelBuild(
                    model="model-name",
                    model_provider=provider_name,
                    output_type=False,
                    system_prompt="Analyse the data",
                )
                agent = Mock()
                model = Mock()
                with (
                    patch(
                        "src.agents.agents_factory.infer_model", return_value=model
                    ) as infer,
                    patch(
                        "src.agents.agents_factory.Agent", return_value=agent
                    ) as agent_ctor,
                ):
                    result = factory.create_agent("analyser", config, AnalyserResponse)
                    infer.assert_called_once_with(f"{provider_name}:model-name")
                self.assertIs(result, agent)
                agent_ctor.assert_called_once_with(
                    model=model,
                    system_prompt="Analyse the data",
                )

    def test_config_models_ignore_legacy_base_url(self) -> None:
        legacy_config = {
            "model": "gemma4:latest",
            "model_provider": "ollama",
            "base_url": "http://legacy:11434/v1",
            "system_prompt": "Analyse data",
            "system_prompt_user": "Analyse data",
            "system_prompt_template": "$user_part",
        }
        for config_type in (ConfigModelBuild, ConfigModelDB, ConfigModelUser):
            with self.subTest(config_type=config_type.__name__):
                config = config_type.model_validate(legacy_config)
                self.assertNotIn("base_url", config.model_dump())
                self.assertNotIn("base_url", config_type.model_fields)
                self.assertNotIn(
                    "base_url", config_type.model_json_schema()["properties"]
                )

    def test_ollama_uses_environment_url_not_legacy_config(self) -> None:
        factory = AgentsFactory.__new__(AgentsFactory)
        factory.AGENT_NAMES = ["analyser"]
        config = ConfigModelBuild.model_validate(
            {
                "model": "gemma4:latest",
                "model_provider": "ollama",
                "base_url": "http://legacy:11434/v1",
                "system_prompt": "Analyse data",
            }
        )
        with patch.dict("os.environ", {"OLLAMA_BASE_URL": "http://ollama:11434/v1"}):
            agent = factory.create_agent("analyser", config, AnalyserResponse)
        self.assertEqual(agent.model.system, "ollama")
        self.assertEqual(
            str(agent.model.base_url).rstrip("/"), "http://ollama:11434/v1"
        )

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

    async def test_unsafe_analysis_failure_does_not_run_repair(self) -> None:
        for error_message in (
            "ValueError: Unsafe analysis code: import 'os' is not allowed",
            "Traceback (most recent call last):\n"
            "ValueError: unsafe analysis code: "
            "output must stay in the output directory",
            "UNSAFE ANALYSIS CODE: attribute '__class__' is not allowed",
        ):
            with self.subTest(error_message=error_message):
                failure = AnalyserResult(
                    df_file_path=None,
                    data_file_path=None,
                    plot_path=None,
                    python_code="import os",
                    code_summary="summary",
                    error_message=error_message,
                )
                runner = AgentRunner.__new__(AgentRunner)
                runner.CONFIG = {"analyser": Mock()}
                runner.agents = Mock()
                runner.agents.run_agent = AsyncMock(return_value=failure)
                with patch(
                    "src.agents.agents_runner.get_dataset_metadata", return_value=""
                ):
                    result = await runner.run_analyser(
                        "request",
                        {"data_file_path": None, "downloaded_data_file_path": None},
                    )
                self.assertIs(result, failure)
                self.assertEqual(result.error_message, error_message)
                self.assertEqual(runner.agents.create_agent.call_count, 1)
                runner.agents.run_agent.assert_awaited_once()
                self.assertEqual(
                    runner.agents.create_agent.call_args.kwargs["agent_name"],
                    "analyser",
                )

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
        with patch(
            "src.server.plot_server.pd.read_csv",
            return_value=pd.DataFrame({"value": [1]}),
        ):
            executed = execute_python_code(response, Path("data.csv"), None)
        callback = Mock(return_value=executed)
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
        with (
            patch("src.agents.agents_runner.get_dataset_metadata", return_value=""),
            patch(
                "src.server.plot_server.pd.read_csv",
                return_value=pd.DataFrame({"value": [1]}),
            ),
        ):
            result = await runner.run_analyser(
                "request",
                {"data_file_path": Path("data.csv"), "downloaded_data_file_path": None},
            )
        self.assertIsNone(result.error_message)
        self.assertEqual(factory.run_agent.await_count, 2)
        repair = factory.run_agent.await_args_list[1].kwargs
        self.assertEqual(repair["agent"], "code_repair")
        self.assertIn("raise ValueError('original')", repair["user_prompt"])
        self.assertIn("ValueError: original", repair["user_prompt"])


class ExecutionDataContracts(unittest.TestCase):
    def test_unsafe_analysis_code_is_rejected_before_execution(self) -> None:
        programs = (
            "import os\nos.system('echo unsafe')",
            "from subprocess import run\nrun(['echo', 'unsafe'])",
            "open('secret.txt').read()",
            "eval('1 + 1')",
            "getattr(df, 'to_pickle')('unsafe.pkl')",
            "df.__class__.__mro__",
            "import pandas as pd\npd.read_csv('https://example.com/data.csv')",
            "from pandas import read_pickle as load\nload('unsafe.pkl')",
            "import numpy as np\nnp.load('unsafe.npy', allow_pickle=True)",
            "import numpy.ctypeslib",
            "from matplotlib import *",
            "import pandas as pd\npd.eval('1 + 1')",
            "df.query('value > 0')",
            "reader = df.to_pickle",
            "import pandas as pd\npd.io.common.os.system('echo unsafe')",
            "'{0.__class__}'.format(df)",
            "from . import pandas",
            "from numpy import mean as eval",
            "import numpy as np\nnp.__dict__['load']('unsafe.npy')",
            "writer = df.to_csv",
            "import matplotlib.pyplot as plt\nwriter = plt.savefig",
            "while True:\n    pass",
            "class Unsafe:\n    pass",
            "df.to_csv = lambda path: None",
        )
        for program in programs:
            with self.subTest(program=program):
                with (
                    patch(
                        "src.server.plot_server.pd.read_csv",
                        return_value=pd.DataFrame({"value": [1]}),
                    ),
                    patch("builtins.exec") as execute,
                ):
                    result = execute_python_code(
                        AnalyserResponse(python_code=program, code_summary="summary"),
                        Path("data.csv"),
                        None,
                    )
                execute.assert_not_called()
                self.assertIn("Unsafe analysis code", result.error_message)
                self.assertIsNone(result.plot_path)
                self.assertIsNone(result.df_file_path)

    def test_representative_imports_from_all_allowed_libraries(self) -> None:
        programs = (
            "import pandas as pd\nresult = df.groupby('value').mean()",
            "import polars as pl\nresult = pl.from_pandas(df).select(pl.col('value'))",
            "import matplotlib.pyplot as plt\nplt.plot(df['value'])",
            "import seaborn as sns\nsns.histplot(data=df, x='value')",
            "import numpy as np\nresult = np.mean(df['value'])",
            "from sklearn.preprocessing import StandardScaler\n"
            "model = StandardScaler()",
            "import statsmodels.api as sm\nmodel = sm.OLS(df['value'], df[['value']])",
            "from sktime.forecasting.naive import NaiveForecaster\n"
            "model = NaiveForecaster()",
        )
        for program in programs:
            with self.subTest(program=program):
                self.assertIsNotNone(compile_analysis(program))

    def test_output_escape_and_invalid_extensions_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_dir = root / "outputs"
            output_dir.mkdir()
            (output_dir / "link").symlink_to(root, target_is_directory=True)
            targets = (
                root / "escaped.csv",
                output_dir / ".." / "traversal.csv",
                output_dir / "link" / "symlink.csv",
                output_dir / "not_csv.py",
            )
            for target in targets:
                with self.subTest(target=target):
                    with patch(
                        "src.server.plot_server.pd.read_csv",
                        return_value=pd.DataFrame({"value": [1]}),
                    ):
                        result = execute_python_code(
                            AnalyserResponse(
                                python_code=f"df.to_csv({str(target)!r})",
                                code_summary="summary",
                            ),
                            Path("data.csv"),
                            None,
                            output_dir=output_dir,
                        )
                    self.assertIn("Unsafe analysis code", result.error_message)
                    self.assertFalse(target.exists())

    def test_polars_analysis_and_keyword_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "polars.csv"
            program = (
                "import polars as pl\n"
                "result = pl.from_pandas(df).select(pl.col('value').mean())\n"
                f"df_file_path = {str(csv_path)!r}\n"
                "result.write_csv(file=df_file_path)\n"
            )
            with patch(
                "src.server.plot_server.pd.read_csv",
                return_value=pd.DataFrame({"value": [1, 2, 3]}),
            ):
                result = execute_python_code(
                    AnalyserResponse(python_code=program, code_summary="summary"),
                    Path("data.csv"),
                    None,
                    output_dir=Path(directory),
                )
            self.assertIsNone(result.error_message)
            self.assertEqual(result.df_file_path, csv_path)
            self.assertIn("2.0", csv_path.read_text())

    def test_invalid_syntax_returns_an_error_without_execution(self) -> None:
        with (
            patch(
                "src.server.plot_server.pd.read_csv",
                return_value=pd.DataFrame({"value": [1]}),
            ),
            patch("builtins.exec") as execute,
        ):
            result = execute_python_code(
                AnalyserResponse(python_code="if:", code_summary="summary"),
                Path("data.csv"),
                None,
            )
        execute.assert_not_called()
        self.assertIn("SyntaxError", result.error_message)

    def test_allowed_analysis_and_plotting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            plot_path = Path(directory) / "plot.png"
            csv_path = Path(directory) / "analysis.csv"
            program = (
                "import pandas as pd\n"
                "import numpy as np\n"
                "import matplotlib.pyplot as plt\n"
                "from sklearn.linear_model import LinearRegression\n"
                "data = df.copy() if df is not None else df_download.copy()\n"
                "model = LinearRegression().fit(data[['value']], data['value'])\n"
                "data['prediction'] = model.predict(data[['value']])\n"
                "data['mean'] = np.mean(data['value'])\n"
                "fig, ax = plt.subplots()\n"
                "ax.plot(data['value'], data['prediction'])\n"
                f"plot_path = {str(plot_path)!r}\n"
                "fig.savefig(plot_path)\n"
                "plt.close(fig)\n"
                f"df_file_path = {str(csv_path)!r}\n"
                "data.to_csv(df_file_path, index=False)\n"
            )
            with patch(
                "src.server.plot_server.pd.read_csv",
                return_value=pd.DataFrame({"value": [1, 2, 3]}),
            ):
                result = execute_python_code(
                    AnalyserResponse(python_code=program, code_summary="summary"),
                    Path("data.csv"),
                    None,
                    output_dir=Path(directory),
                )
            self.assertIsNone(result.error_message)
            self.assertTrue(plot_path.is_file())
            self.assertTrue(csv_path.is_file())
            self.assertEqual(result.plot_path, plot_path)
            self.assertEqual(result.df_file_path, csv_path)

    def test_missing_or_empty_datasets_skip_execution(self) -> None:
        empty = pd.DataFrame(columns=["value"])
        for extracted, downloaded in (
            (None, None),
            (empty, None),
            (None, empty),
            (empty, empty),
        ):
            with self.subTest(
                extracted=extracted is not None, downloaded=downloaded is not None
            ):
                values = [
                    frame for frame in (extracted, downloaded) if frame is not None
                ]
                with (
                    patch("src.server.plot_server.pd.read_csv", side_effect=values),
                    patch("builtins.exec") as execute,
                ):
                    result = execute_python_code(
                        AnalyserResponse(python_code="pass", code_summary="summary"),
                        Path("extracted.csv") if extracted is not None else None,
                        Path("downloaded.csv") if downloaded is not None else None,
                    )
                execute.assert_not_called()
                self.assertIn("missing or empty", result.error_message)
                self.assertIsNone(result.plot_path)
                self.assertIsNone(result.df_file_path)

    def test_either_nonempty_dataset_allows_execution(self) -> None:
        usable = pd.DataFrame({"value": [1]})
        empty = pd.DataFrame(columns=["value"])
        for extracted, downloaded in (
            (usable, None),
            (None, usable),
            (usable, empty),
            (empty, usable),
        ):
            values = [frame for frame in (extracted, downloaded) if frame is not None]
            with patch("src.server.plot_server.pd.read_csv", side_effect=values):
                result = execute_python_code(
                    AnalyserResponse(python_code="pass", code_summary="summary"),
                    Path("extracted.csv") if extracted is not None else None,
                    Path("downloaded.csv") if downloaded is not None else None,
                )
            self.assertIsNone(result.error_message)


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
