"""Tests for the shared experiment helpers in src/experiments.py."""

import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import mlflow
from src import experiments
from src.experiments import (
    ExperimentDataMissing,
    flatten_metrics,
    log_run_to_mlflow,
    log_skipped_to_mlflow,
    run_all_experiments,
)


class MlflowTestCase(unittest.TestCase):
    """Points MLflow at a throwaway sqlite store and artifact directory."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self._previous_uri = mlflow.get_tracking_uri()
        mlflow.set_tracking_uri(f"sqlite:///{self.tmp / 'mlflow.db'}")
        patcher = mock.patch.object(
            experiments, "MLFLOW_ARTIFACT_DIR", self.tmp / "mlruns"
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self) -> None:
        mlflow.set_tracking_uri(self._previous_uri)
        self._tmp.cleanup()

    def only_run(self, mlflow_experiment_name: str) -> mlflow.entities.Run:
        experiment = mlflow.get_experiment_by_name(mlflow_experiment_name)
        assert experiment is not None
        runs = mlflow.search_runs([experiment.experiment_id], output_format="list")
        self.assertEqual(len(runs), 1)
        return runs[0]


class FlattenMetricsTest(unittest.TestCase):
    def test_extractor_metric_names_unchanged(self) -> None:
        metrics = {
            "fields": {"Saldo": {"accuracy": 0.9}, "body_weight": {"mae": None}},
            "overall": {"accuracy": 0.9, "mae": 1.5},
        }
        flat = flatten_metrics(metrics)
        self.assertEqual(
            set(flat),
            {"accuracy_Saldo", "mae_body_weight", "accuracy_overall", "mae_overall"},
        )
        self.assertTrue(math.isnan(flat["mae_body_weight"]))


class LogToMlflowTest(MlflowTestCase):
    def test_logs_metrics_params_tags_and_artifacts(self) -> None:
        artifact = self.tmp / "plot.png"
        artifact.write_bytes(b"PNG")

        log_run_to_mlflow(
            "analyser/body_weight",
            "run1",
            params={"user_prompt": "plot it"},
            metrics={"execution_success": 1.0, "readability": None},
            artifact_paths=[artifact],
            tags={"agent": "analyser"},
        )

        run = self.only_run("analyser/body_weight")
        self.assertEqual(run.info.run_name, "run1")
        self.assertEqual(run.data.metrics["execution_success"], 1.0)
        self.assertTrue(math.isnan(run.data.metrics["readability"]))
        self.assertEqual(run.data.params["user_prompt"], "plot it")
        self.assertEqual(run.data.tags["agent"], "analyser")
        # Artifacts land under the configured directory, not the cwd.
        self.assertTrue(
            run.info.artifact_uri.startswith((self.tmp / "mlruns").as_uri())
        )
        artifacts = mlflow.artifacts.list_artifacts(run_id=run.info.run_id)
        self.assertEqual([a.path for a in artifacts], ["plot.png"])

    def test_skipped_run_logs_nan_metrics_and_reason(self) -> None:
        log_skipped_to_mlflow("x", "run1", "no prompt", ["a", "b"])

        run = self.only_run("x")
        self.assertEqual(run.data.tags["skipped_reason"], "no prompt")
        self.assertEqual(set(run.data.metrics), {"a", "b"})
        self.assertTrue(all(math.isnan(v) for v in run.data.metrics.values()))


class RunAllExperimentsTest(MlflowTestCase):
    def test_skips_missing_data_with_prefixed_experiment(self) -> None:
        ran: list[str] = []

        def run_experiment(experiment_name: str, run_name: str) -> None:
            if experiment_name == "missing":
                raise ExperimentDataMissing("no prompt")
            ran.append(experiment_name)

        run_all_experiments(
            ["missing", "ok"], "run1", run_experiment, ["a"], mlflow_prefix="p/"
        )

        self.assertEqual(ran, ["ok"])
        run = self.only_run("p/missing")
        self.assertEqual(run.data.tags["skipped_reason"], "no prompt")

    def test_other_errors_stop_the_batch(self) -> None:
        def run_experiment(experiment_name: str, run_name: str) -> None:
            raise RuntimeError("backend down")

        with self.assertRaises(RuntimeError):
            run_all_experiments(["a", "b"], "run1", run_experiment, ["a"])


if __name__ == "__main__":
    unittest.main()
