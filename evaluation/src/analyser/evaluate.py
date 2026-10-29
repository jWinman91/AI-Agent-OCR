"""Turns one analyser run into metrics: execution success from the backend plus
LLM judge scores for the code and, if a plot exists, the visualisation, and a
line-by-line bug report of the code."""

from dataclasses import dataclass, field
from typing import Any

from loguru import logger
from src.analyser.experiment_spec import AnalyserExperimentSpec
from src.analyser.judge import BugReport, LLMJudge
from src.analyser.judge_config import BUG_COUNT, CODE_CRITERIA, VISUALISATION_CRITERIA
from src.analyser.run_analyser import AnalyserRunOutcome


@dataclass
class AnalyserEvaluation:
    # Metric name -> value; None for criteria that were not judged (logged as
    # NaN), so every run has the same metric columns.
    metrics: dict[str, float | None]
    # Criterion -> {explanation, evidence, score}, or the reason it was skipped.
    judgements: dict[str, Any] = field(default_factory=dict)
    # One message per judge call that failed (API error, invalid output, ...).
    judge_errors: list[str] = field(default_factory=list)
    # The code's bugs by line; None if there was no code or the bug judge failed.
    bug_report: BugReport | None = None


class AnalyserEvaluator:
    def __init__(self, judge: LLMJudge) -> None:
        self._judge = judge

    def evaluate(
        self, spec: AnalyserExperimentSpec, outcome: AnalyserRunOutcome
    ) -> AnalyserEvaluation:
        """Scores one analyser run. Each judge is only called when there is
        something to judge, and a failing judge leaves its scores None instead
        of aborting the run, so failed runs are still logged.

        Args:
            spec (AnalyserExperimentSpec): The experiment that was run.
            outcome (AnalyserRunOutcome): What the analyser returned.
        Returns:
            AnalyserEvaluation: Metrics, judge explanations and judge errors.
        """
        evaluation = AnalyserEvaluation(
            metrics={"execution_success": float(outcome.execution_success)}
        )
        for name in CODE_CRITERIA + VISUALISATION_CRITERIA + [BUG_COUNT]:
            evaluation.metrics[name] = None

        # Code is judged whenever there is any, including the failed code of a
        # run whose execution (and code repair) failed.
        if not outcome.python_code:
            self._skip(evaluation, CODE_CRITERIA, "no code was generated")
        else:
            try:
                code_judgement = self._judge.judge_code(
                    spec.user_prompt,
                    outcome.code_summary,
                    outcome.python_code,
                    spec.input_file,
                    outcome.error_message,
                )
                self._record(evaluation, code_judgement.model_dump())
            except Exception as e:
                self._fail(evaluation, CODE_CRITERIA, "code judge", e)

        if not outcome.execution_success:
            self._skip(evaluation, VISUALISATION_CRITERIA, "execution failed")
        elif outcome.plot_png is None:
            self._skip(evaluation, VISUALISATION_CRITERIA, "no plot was produced")
        else:
            try:
                visualisation_judgement = self._judge.judge_visualisation(
                    spec.user_prompt,
                    spec.input_file,
                    outcome.plot_png,
                    spec.ground_truth_path,
                    outcome.output_csv,
                )
                self._record(evaluation, visualisation_judgement.model_dump())
            except Exception as e:
                self._fail(evaluation, VISUALISATION_CRITERIA, "visualisation judge", e)

        # Like the code judge, the bug judge also runs on failed code, where
        # finding the root cause matters most.
        if not outcome.python_code:
            self._skip(evaluation, ["bug_report"], "no code was generated")
        else:
            try:
                evaluation.bug_report = self._judge.judge_bugs(
                    spec.user_prompt,
                    outcome.code_summary,
                    outcome.python_code,
                    spec.input_file,
                    outcome.error_message,
                    outcome.output_csv,
                )
                evaluation.metrics[BUG_COUNT] = float(len(evaluation.bug_report.issues))
                evaluation.judgements["bug_report"] = evaluation.bug_report.model_dump()
            except Exception as e:
                self._fail(evaluation, ["bug_report"], "bug judge", e)

        return evaluation

    @staticmethod
    def _record(evaluation: AnalyserEvaluation, judgement: dict[str, Any]) -> None:
        for name, criterion in judgement.items():
            evaluation.metrics[name] = float(criterion["score"])
            evaluation.judgements[name] = criterion

    @staticmethod
    def _skip(evaluation: AnalyserEvaluation, criteria: list[str], reason: str) -> None:
        for name in criteria:
            evaluation.judgements[name] = {"skipped": reason}

    @staticmethod
    def _fail(
        evaluation: AnalyserEvaluation,
        criteria: list[str],
        judge_name: str,
        error: Exception,
    ) -> None:
        # Any judge failure (API error, refusal, output still invalid after
        # retries) only costs this judge's scores, not the whole run.
        message = f"{judge_name} failed: {type(error).__name__}: {error}"
        logger.error(message)
        evaluation.judge_errors.append(message)
        for name in criteria:
            evaluation.judgements[name] = {"error": message}
