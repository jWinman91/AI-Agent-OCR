"""Tests for the analyser evaluation: experiment spec, judge config, backend
run, judge, evaluator and the end-to-end CLI run, without backend or API key."""

import io
import json
import tempfile
import unittest
from functools import partial
from pathlib import Path
from typing import Any
from unittest import mock

import mlflow
import pandas as pd
import yaml
from evaluate_analyser import EvaluationContext, run_experiment
from pydantic import ValidationError
from pydantic_ai import BinaryContent
from pydantic_ai.messages import ModelRequest, UserPromptPart
from src.analyser import JUDGE_CONFIG_PATH
from src.analyser.evaluate import AnalyserEvaluation, AnalyserEvaluator
from src.analyser.experiment_spec import (
    AnalyserExperimentSpec,
    load_experiment_entries,
    load_experiment_spec,
)
from src.analyser.judge import LLMJudge, describe_table, profile_table
from src.analyser.judge_config import (
    ALL_METRIC_NAMES,
    CODE_CRITERIA,
    VISUALISATION_CRITERIA,
    JudgeConfig,
    load_judge_config,
)
from src.analyser.run_analyser import AnalyserRunner, AnalyserRunOutcome
from src.experiments import ExperimentDataMissing

from tests.helpers import FakeAgentClient, ScriptedJudgeModel
from tests.test_experiments import MlflowTestCase

SUCCESS_RESULT = {
    "df_file_path": "plots/out.csv",
    "data_file_path": "data/in.csv",
    "plot_path": "plots/plot.png",
    "code_summary": "Plots weight.",
    "python_code": "print('plot')",
    "error_message": None,
}


def make_experiment_dir(root: Path, prompts: object) -> tuple[Path, Path]:
    """Creates eval_data/analyser-like and ground_truth-like directories."""
    test_data_dir = root / "analyser"
    ground_truth_dir = root / "ground_truth"
    test_data_dir.mkdir()
    ground_truth_dir.mkdir()
    (test_data_dir / "weight.csv").write_text("day,kg\n1,80.0\n2,81.5\n")
    (ground_truth_dir / "ref.png").write_bytes(b"REFPNG")
    (ground_truth_dir / "ref.csv").write_text("month,sum\n1,10\n")
    (test_data_dir / "prompt.yaml").write_text(yaml.safe_dump(prompts))
    return test_data_dir, ground_truth_dir


def entry(
    prompt: str = "Plot my weight.",
    data_file: str = "weight.csv",
    ground_truth: str | None = None,
    system_prompt_user: str | None = None,
) -> dict[str, str]:
    """A prompt.yaml entry for the weight.csv created by make_experiment_dir."""
    values = {"data_file": data_file, "prompt": prompt}
    if ground_truth is not None:
        values["ground_truth"] = ground_truth
    if system_prompt_user is not None:
        values["system_prompt_user"] = system_prompt_user
    return values


def outcome(**overrides: object) -> AnalyserRunOutcome:
    values: dict[str, object] = {
        "execution_success": 1,
        "error_message": None,
        "python_code": "print('plot')",
        "code_summary": "Plots weight.",
        "plot_png": b"PNG",
        "output_csv": b"a,b\n1,2\n",
    }
    values.update(overrides)
    return AnalyserRunOutcome(**values)  # type: ignore[arg-type]


def user_parts(messages: list[Any]) -> list[Any]:
    request = messages[-1]
    assert isinstance(request, ModelRequest)
    parts = [p for p in request.parts if isinstance(p, UserPromptPart)]
    content = parts[0].content
    return list(content) if isinstance(content, list) else [content]


class ExperimentSpecTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def load(self, prompts: object) -> AnalyserExperimentSpec:
        test_data_dir, ground_truth_dir = make_experiment_dir(self.root, prompts)
        return load_experiment_spec("weight", test_data_dir, ground_truth_dir)

    def test_entry_without_ground_truth(self) -> None:
        spec = self.load({"weight": entry()})
        self.assertEqual(spec.name, "weight")
        self.assertEqual(spec.user_prompt, "Plot my weight.")
        self.assertEqual(spec.input_file.name, "weight.csv")
        self.assertIsNone(spec.ground_truth_path)
        self.assertEqual(spec.system_prompt_user, "")

    def test_entry_with_system_prompt_user(self) -> None:
        spec = self.load({"weight": entry(system_prompt_user="Weight is in kg.")})
        self.assertEqual(spec.system_prompt_user, "Weight is in kg.")

    def test_empty_system_prompt_user(self) -> None:
        spec = self.load({"weight": {**entry(), "system_prompt_user": None}})
        self.assertEqual(spec.system_prompt_user, "")

    def test_entry_with_ground_truth(self) -> None:
        spec = self.load({"weight": entry(ground_truth="ref.png")})
        assert spec.ground_truth_path is not None
        self.assertEqual(spec.ground_truth_path.name, "ref.png")

    def test_experiments_can_share_a_data_file(self) -> None:
        test_data_dir, ground_truth_dir = make_experiment_dir(
            self.root,
            {"weight": entry(), "weight_trend": entry(prompt="Plot the trend.")},
        )
        self.assertEqual(
            list(load_experiment_entries(test_data_dir)), ["weight", "weight_trend"]
        )
        spec = load_experiment_spec("weight_trend", test_data_dir, ground_truth_dir)
        self.assertEqual(spec.input_file.name, "weight.csv")
        self.assertEqual(spec.user_prompt, "Plot the trend.")

    def test_missing_prompt_file_means_no_experiments(self) -> None:
        self.assertEqual(load_experiment_entries(self.root), {})

    def test_missing_or_invalid_data_is_reported(self) -> None:
        cases = [
            None,  # empty prompt.yaml
            {"other": entry()},
            {"weight": "Plot my weight."},  # no data_file
            {"weight": {"data_file": "weight.csv"}},  # no prompt
            {"weight": entry(data_file="missing.csv")},
            {"weight": entry(data_file="notes.txt")},
            {"weight": entry(ground_truth="missing.png")},
            {"weight": entry(ground_truth="ref.txt")},
        ]
        for prompts in cases:
            with self.subTest(prompts=prompts), tempfile.TemporaryDirectory() as tmp:
                test_data_dir, ground_truth_dir = make_experiment_dir(
                    Path(tmp), prompts
                )
                (test_data_dir / "notes.txt").write_text("x")
                (ground_truth_dir / "ref.txt").write_text("x")
                with self.assertRaises(ExperimentDataMissing):
                    load_experiment_spec("weight", test_data_dir, ground_truth_dir)


class JudgeConfigTest(unittest.TestCase):
    def test_shipped_config_is_valid(self) -> None:
        config = load_judge_config()
        self.assertEqual(config.model, "openai-responses:gpt-6.1-sol")
        rubrics = config.visualisation_judge.render_rubrics()
        for name in VISUALISATION_CRITERIA:
            self.assertIn(f"{name}:", rubrics)

    def test_missing_criterion_or_level_fails(self) -> None:
        without_criterion = yaml.safe_load(JUDGE_CONFIG_PATH.read_text())
        del without_criterion["visualisation_judge"]["criteria"]["readability"]
        with self.assertRaises(ValidationError):
            JudgeConfig.model_validate(without_criterion)

        without_level = yaml.safe_load(JUDGE_CONFIG_PATH.read_text())
        del without_level["visualisation_judge"]["criteria"]["readability"]["rubric"][3]
        with self.assertRaises(ValidationError):
            JudgeConfig.model_validate(without_level)


class AnalyserRunnerTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.input_file = Path(self._tmp.name) / "weight.csv"
        self.input_file.write_text("day,kg\n1,80\n")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def run_with(self, client: FakeAgentClient) -> AnalyserRunOutcome:
        return AnalyserRunner(client).run(self.input_file, "Plot.")  # type: ignore[arg-type]

    def test_success_with_plot(self) -> None:
        client = FakeAgentClient(SUCCESS_RESULT)
        result = self.run_with(client)
        self.assertEqual(client.uploaded, ["weight.csv"])
        self.assertEqual(result.execution_success, 1)
        self.assertEqual(result.plot_png, b"PNG")
        self.assertEqual(result.output_csv, b"a,b\n1,2\n")
        self.assertIsNone(result.download_error)

    def test_success_without_plot(self) -> None:
        result = self.run_with(
            FakeAgentClient({**SUCCESS_RESULT, "plot_path": None, "df_file_path": None})
        )
        self.assertEqual(result.execution_success, 1)
        self.assertIsNone(result.plot_png)
        self.assertIsNone(result.output_csv)

    def test_agent_failure_keeps_code(self) -> None:
        result = self.run_with(
            FakeAgentClient(
                {**SUCCESS_RESULT, "plot_path": None, "error_message": "Traceback"}
            )
        )
        self.assertEqual(result.execution_success, 0)
        self.assertEqual(result.error_message, "Traceback")
        self.assertEqual(result.python_code, "print('plot')")

    def test_orchestrator_shaped_failure(self) -> None:
        result = self.run_with(
            FakeAgentClient({"analyser_prompt": "", "error_message": "bad json"})
        )
        self.assertEqual(result.execution_success, 0)
        self.assertEqual(result.python_code, "")
        self.assertIsNone(result.plot_png)

    def test_plot_download_failure(self) -> None:
        result = self.run_with(FakeAgentClient(SUCCESS_RESULT, plot=None))
        self.assertEqual(result.execution_success, 1)
        self.assertIsNone(result.plot_png)
        assert result.download_error is not None
        self.assertIn("plots/plot.png", result.download_error)

    def test_configure_uploads_system_prompt_user(self) -> None:
        client = FakeAgentClient(SUCCESS_RESULT)
        config = AnalyserRunner(client).configure("Weight is in kg.")  # type: ignore[arg-type]
        self.assertEqual(
            client.calls,
            [
                "get_config:analyser:False",
                "upload_config:analyser",
                "update_agents",
                "get_config:analyser:True",
            ],
        )
        self.assertEqual(client.config["system_prompt_user"], "Weight is in kg.")
        self.assertEqual(client.config["model"], "gemma4:latest")
        self.assertEqual(config["system_prompt_user"], "FULL[Weight is in kg.]")

    def test_current_system_prompt_user(self) -> None:
        client = FakeAgentClient(SUCCESS_RESULT)
        client.config["system_prompt_user"] = "from the UI"
        runner = AnalyserRunner(client)  # type: ignore[arg-type]
        self.assertEqual(runner.current_system_prompt_user(), "from the UI")


class JudgeAndEvaluatorTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.config = load_judge_config()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def spec(self, ground_truth: str | None = None) -> AnalyserExperimentSpec:
        root = self.root / (ground_truth or "none")
        root.mkdir()
        test_data_dir, ground_truth_dir = make_experiment_dir(
            root, {"weight": entry(ground_truth=ground_truth)}
        )
        return load_experiment_spec("weight", test_data_dir, ground_truth_dir)

    def evaluate(
        self,
        judge_model: ScriptedJudgeModel,
        spec: AnalyserExperimentSpec,
        run: AnalyserRunOutcome,
    ) -> AnalyserEvaluation:
        judge = LLMJudge(self.config, model=judge_model.model)
        return AnalyserEvaluator(judge).evaluate(spec, run)

    def test_full_run_is_scored(self) -> None:
        judge_model = ScriptedJudgeModel(code_score=4, visualisation_score=3)

        evaluation = self.evaluate(judge_model, self.spec(), outcome())

        self.assertEqual(list(evaluation.metrics), ALL_METRIC_NAMES)
        self.assertEqual(evaluation.metrics["execution_success"], 1.0)
        for name in CODE_CRITERIA:
            self.assertEqual(evaluation.metrics[name], 4.0)
        for name in VISUALISATION_CRITERIA:
            self.assertEqual(evaluation.metrics[name], 3.0)
        self.assertEqual(
            evaluation.judgements["readability"]["explanation"], "scored 3"
        )
        self.assertEqual(evaluation.judge_errors, [])

        # The code judge sees the code, the execution result and the input
        # data profile; the visualisation judge sees the plot and the input
        # data, but never the code.
        code_text = user_parts(judge_model.requests[0])[0]
        self.assertIn("print('plot')", code_text)
        self.assertIn("<EXECUTION_RESULT>\nsucceeded\n</EXECUTION_RESULT>", code_text)
        self.assertIn("2 data rows, 2 columns.", code_text)
        self.assertIn("- kg: dtype float64", code_text)
        visualisation_parts = user_parts(judge_model.requests[1])
        self.assertIn("80.0", visualisation_parts[0])
        self.assertNotIn("print('plot')", visualisation_parts[0])
        images = [p for p in visualisation_parts if isinstance(p, BinaryContent)]
        self.assertEqual([i.data for i in images], [b"PNG"])

    def test_png_reference_is_attached_as_second_image(self) -> None:
        judge_model = ScriptedJudgeModel()
        self.evaluate(judge_model, self.spec("ref.png"), outcome())

        parts = user_parts(judge_model.requests[1])
        images = [p.data for p in parts if isinstance(p, BinaryContent)]
        self.assertEqual(images, [b"PNG", b"REFPNG"])
        self.assertIn("reference plot", parts[0])

    def test_csv_reference_is_shown_with_output_data(self) -> None:
        judge_model = ScriptedJudgeModel()
        self.evaluate(judge_model, self.spec("ref.csv"), outcome())

        text = user_parts(judge_model.requests[1])[0]
        self.assertIn("<REFERENCE_DATA>", text)
        self.assertIn("month,sum", text)
        self.assertIn("<OUTPUT_DATA>", text)

    def test_failed_execution_skips_visualisation_but_judges_code(self) -> None:
        judge_model = ScriptedJudgeModel()
        evaluation = self.evaluate(
            judge_model,
            self.spec(),
            outcome(
                execution_success=0,
                error_message="Traceback: KeyError 'weight'",
                plot_png=None,
            ),
        )
        self.assertEqual(evaluation.metrics["execution_success"], 0.0)
        code_text = user_parts(judge_model.requests[0])[0]
        self.assertIn("failed:\nTraceback: KeyError 'weight'", code_text)
        self.assertIn("day,kg", code_text)
        for name in CODE_CRITERIA:
            self.assertEqual(evaluation.metrics[name], 4.0)
        for name in VISUALISATION_CRITERIA:
            self.assertIsNone(evaluation.metrics[name])
        self.assertEqual(len(judge_model.requests), 1)

    def test_no_plot_and_no_code_skip_both_judges(self) -> None:
        judge_model = ScriptedJudgeModel()
        evaluation = self.evaluate(
            judge_model, self.spec(), outcome(python_code="", plot_png=None)
        )
        self.assertEqual(evaluation.metrics["execution_success"], 1.0)
        for name in CODE_CRITERIA:
            self.assertIsNone(evaluation.metrics[name])
        self.assertEqual(
            evaluation.judgements["readability"], {"skipped": "no plot was produced"}
        )
        self.assertEqual(judge_model.requests, [])

    def test_out_of_range_score_becomes_judge_error(self) -> None:
        judge_model = ScriptedJudgeModel(code_score=7)
        evaluation = self.evaluate(judge_model, self.spec(), outcome())
        for name in CODE_CRITERIA:
            self.assertIsNone(evaluation.metrics[name])
        self.assertEqual(evaluation.metrics["readability"], 3.0)
        self.assertEqual(len(evaluation.judge_errors), 1)
        self.assertIn("code judge failed", evaluation.judge_errors[0])

    def test_judge_exception_becomes_judge_error(self) -> None:
        evaluation = self.evaluate(
            ScriptedJudgeModel(fail=True), self.spec(), outcome()
        )
        self.assertTrue(
            all(evaluation.metrics[n] is None for n in ALL_METRIC_NAMES[1:])
        )
        self.assertEqual(len(evaluation.judge_errors), 2)


class CheckModelTest(unittest.TestCase):
    def test_preflight_uses_structured_output(self) -> None:
        judge_model = ScriptedJudgeModel()
        LLMJudge(load_judge_config(), model=judge_model.model).check_model()
        self.assertEqual(len(judge_model.requests), 1)

    def test_preflight_failure_propagates(self) -> None:
        judge = LLMJudge(load_judge_config(), model=ScriptedJudgeModel(fail=True).model)
        with self.assertRaises(RuntimeError):
            judge.check_model()


class DescribeTableTest(unittest.TestCase):
    def test_truncates_long_tables(self) -> None:
        content = ("a,b\n" + "1,2\n" * 100).encode()
        text = describe_table(content, "x.csv", max_chars=20)
        self.assertIn("100 data rows. Header: a,b", text)
        self.assertIn("truncated: showing the first 20 of 404 characters", text)

    def test_keeps_short_tables_whole(self) -> None:
        text = describe_table(b"a,b\n1,2\n", "x.csv", max_chars=100)
        self.assertNotIn("truncated", text)
        self.assertTrue(text.endswith("a,b\n1,2\n"))


class ProfileTableTest(unittest.TestCase):
    def test_small_table_is_shown_whole_with_raw_values(self) -> None:
        content = 'Datum,Typ,Saldo\n01. Jan 26,Zinsen,"60.394,59 €"\n'.encode()
        text = profile_table(content, "x.csv", sample_rows=100, max_chars=10000)
        self.assertIn("1 data rows, 3 columns.", text)
        self.assertIn("- Saldo: dtype object, 0 missing, 1 distinct", text)
        self.assertIn("All rows", text)
        self.assertIn('0,01. Jan 26,Zinsen,"60.394,59 €"', text)

    def test_low_cardinality_columns_list_their_values(self) -> None:
        content = b"Typ,x\nZinsen,1\nErtrag,2\nZinsen,\n"
        text = profile_table(content, "x.csv", sample_rows=100, max_chars=10000)
        self.assertIn("- Typ: dtype object, 0 missing, 2 distinct; values: ", text)
        self.assertIn("'Zinsen' (2), 'Ertrag' (1)", text)
        self.assertIn("- x: dtype float64, 1 missing", text)

    def test_large_table_shows_edges_and_a_fixed_random_sample(self) -> None:
        content = ("value\n" + "".join(f"v{i}\n" for i in range(1000))).encode()
        text = profile_table(content, "x.csv", sample_rows=100, max_chars=100000)
        self.assertEqual(
            text, profile_table(content, "x.csv", sample_rows=100, max_chars=100000)
        )
        self.assertNotIn("values:", text)  # 1000 distinct values are not listed
        rows = text.split("row,value\n")[1].strip().splitlines()
        positions = [int(row.split(",")[0]) for row in rows]
        self.assertEqual(len(positions), 110)
        self.assertEqual(positions, sorted(positions))
        self.assertEqual(positions[:5], [0, 1, 2, 3, 4])
        self.assertEqual(positions[-5:], [995, 996, 997, 998, 999])
        self.assertIn(f"{positions[50]},v{positions[50]}", rows)

    def test_excel_input(self) -> None:
        buffer = io.BytesIO()
        pd.DataFrame({"kg": [80.0, 81.5]}).to_excel(buffer, index=False)
        text = profile_table(buffer.getvalue(), "x.xlsx", 100, max_chars=10000)
        self.assertIn("2 data rows, 1 columns.", text)
        self.assertIn("1,81.5", text)

    def test_truncates_at_max_chars(self) -> None:
        content = ("value\n" + "".join(f"v{i}\n" for i in range(1000))).encode()
        text = profile_table(content, "x.csv", sample_rows=100, max_chars=50)
        self.assertIn("[... truncated: showing the first 50 of", text)


class RunExperimentTest(MlflowTestCase):
    def test_logs_scored_run_with_artifacts(self) -> None:
        test_data_dir, ground_truth_dir = make_experiment_dir(
            self.tmp, {"weight": entry(system_prompt_user="Weight is in kg.")}
        )
        judge_model = ScriptedJudgeModel()
        client = FakeAgentClient(SUCCESS_RESULT)
        context = EvaluationContext(
            runner=AnalyserRunner(client),  # type: ignore[arg-type]
            evaluator=AnalyserEvaluator(
                LLMJudge(load_judge_config(), model=judge_model.model)
            ),
            judge_model="openai-responses:gpt-6.1-sol",
            judge_config_sha="abc",
        )
        with mock.patch(
            "evaluate_analyser.load_experiment_spec",
            partial(
                load_experiment_spec,
                test_data_dir=test_data_dir,
                ground_truth_dir=ground_truth_dir,
            ),
        ):
            run_experiment("weight", "run1", context)

        self.assertEqual(client.config["system_prompt_user"], "Weight is in kg.")
        self.assertLess(
            client.calls.index("upload_config:analyser"),
            client.calls.index("run:analyser"),
        )
        run = self.only_run("analyser/weight")
        self.assertEqual(set(run.data.metrics), set(ALL_METRIC_NAMES))
        self.assertEqual(run.data.metrics["code_simplicity"], 4.0)
        self.assertEqual(run.data.metrics["code_correctness"], 4.0)
        self.assertEqual(run.data.params["analyser_model"], "gemma4:latest")
        self.assertEqual(run.data.params["ground_truth"], "none")
        self.assertEqual(run.data.params["judge_config_sha"], "abc")
        self.assertEqual(run.data.tags["agent"], "analyser")
        artifacts = {
            a.path for a in mlflow.artifacts.list_artifacts(run_id=run.info.run_id)
        }
        self.assertEqual(
            artifacts,
            {
                "plot.png",
                "output_data.csv",
                "analyser_code.py",
                "code_summary.txt",
                "judge_results.json",
                "analyser_config.yaml",
                "judge_config.yaml",
            },
        )
        logged_config = yaml.safe_load(
            mlflow.artifacts.load_text(f"runs:/{run.info.run_id}/analyser_config.yaml")
        )
        self.assertEqual(logged_config["system_prompt_user"], "FULL[Weight is in kg.]")
        judgements = json.loads(
            mlflow.artifacts.load_text(f"runs:/{run.info.run_id}/judge_results.json")
        )
        self.assertEqual(judgements["readability"]["score"], 3)


if __name__ == "__main__":
    unittest.main()
