"""Tests for the extractor evaluation's experiment spec and domain knowledge
configuration, without backend."""

import tempfile
import unittest
from pathlib import Path

from src.experiments import ExperimentDataMissing
from src.extractor.experiment_spec import (
    ExtractorExperimentSpec,
    load_experiment_spec,
)
from src.extractor.extract_data import DataExtractor

from tests.helpers import FakeAgentClient


class ExperimentSpecTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.test_data_dir = root / "extractor"
        self.ground_truth_dir = root / "ground_truth"
        (self.test_data_dir / "body_weight").mkdir(parents=True)
        (self.test_data_dir / "body_weight" / "scale.jpg").write_bytes(b"JPG")
        self.ground_truth_dir.mkdir()
        (self.ground_truth_dir / "body_weight.csv").write_text("body_weight\n80.1\n")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def load(
        self, prompt_yaml: str, experiment_name: str = "body_weight"
    ) -> ExtractorExperimentSpec:
        (self.test_data_dir / "prompt.yaml").write_text(prompt_yaml)
        return load_experiment_spec(
            experiment_name, self.test_data_dir, self.ground_truth_dir
        )

    def test_entry_with_domain_knowledge_prompt(self) -> None:
        spec = self.load(
            "body_weight:\n"
            "  prompt: Extract the weight.\n"
            "  domain_knowledge_prompt: Always XX.X kg.\n"
        )
        self.assertEqual(spec.user_prompt, "Extract the weight.")
        self.assertEqual(spec.domain_knowledge_prompt, "Always XX.X kg.")

    def test_missing_or_empty_domain_knowledge_prompt(self) -> None:
        for prompt_yaml in (
            "body_weight:\n  prompt: Extract.\n",
            "body_weight:\n  prompt: Extract.\n  domain_knowledge_prompt:\n",
        ):
            with self.subTest(prompt_yaml=prompt_yaml):
                spec = self.load(prompt_yaml)
                self.assertEqual(spec.domain_knowledge_prompt, "")

    def test_missing_or_invalid_data_is_reported(self) -> None:
        cases = {
            "missing entry": ("other:\n  prompt: Extract.\n", "body_weight"),
            "plain string entry": ("body_weight: Extract.\n", "body_weight"),
            "no prompt": (
                "body_weight:\n  domain_knowledge_prompt: kg\n",
                "body_weight",
            ),
            "no ground truth": ("missing:\n  prompt: Extract.\n", "missing"),
        }
        for case, (prompt_yaml, experiment_name) in cases.items():
            with self.subTest(case=case), self.assertRaises(ExperimentDataMissing):
                self.load(prompt_yaml, experiment_name)

    def test_missing_test_files_are_reported(self) -> None:
        (self.test_data_dir / "body_weight" / "scale.jpg").unlink()
        with self.assertRaises(ExperimentDataMissing):
            self.load("body_weight:\n  prompt: Extract.\n")


class DataExtractorConfigureTest(unittest.TestCase):
    def test_configure_uploads_domain_knowledge_prompt(self) -> None:
        client = FakeAgentClient({})
        config = DataExtractor(client=client).configure("Always XX.X kg.")  # type: ignore[arg-type]
        self.assertEqual(
            client.calls,
            [
                "get_config:data_extractor:False",
                "upload_config:data_extractor",
                "update_agents",
                "get_config:data_extractor:True",
            ],
        )
        self.assertEqual(client.config["domain_knowledge_prompt"], "Always XX.X kg.")
        self.assertEqual(client.config["model"], "gemma4:latest")
        self.assertEqual(config["domain_knowledge_prompt"], "FULL[Always XX.X kg.]")

    def test_current_domain_knowledge_prompt(self) -> None:
        client = FakeAgentClient({})
        client.config["domain_knowledge_prompt"] = "from the UI"
        extractor = DataExtractor(client=client)  # type: ignore[arg-type]
        self.assertEqual(extractor.current_domain_knowledge_prompt(), "from the UI")
        self.assertEqual(client.calls, ["get_config:data_extractor:False"])
