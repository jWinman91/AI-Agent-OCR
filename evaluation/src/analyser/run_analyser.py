"""Runs the analyser agent on one input file via the backend and collects its
outputs (code, plot, output data) for judging."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import UploadFile
from loguru import logger
from src.backend_client.agent_client import AgentClient


@dataclass
class AnalyserRunOutcome:
    # 1 if the backend reported no error_message, else 0. Taken from the
    # response only - the generated code is never executed a second time.
    execution_success: int
    error_message: str | None
    python_code: str
    code_summary: str
    plot_png: bytes | None
    output_csv: bytes | None
    # Set when the agent reported a plot/data file that could not be
    # downloaded; the run still counts as executed.
    download_error: str | None = None


class AnalyserRunner:
    def __init__(self, client: AgentClient | None = None) -> None:
        self._client = client or AgentClient()

    def current_system_prompt_user(self) -> str:
        """Returns the analyser's stored system_prompt_user, so it can be restored
        after the experiments.

        Returns:
            str: The user-editable part of the analyser's system prompt.
        """
        return str(
            self._client.get_config("analyser", full_prompt=False)["system_prompt_user"]
        )

    def configure(self, system_prompt_user: str) -> dict[str, Any]:
        """Sets the analyser's system_prompt_user via /upload_config, keeping the
        rest of its stored config (model, provider, settings), and rebuilds the
        backend agents so the next run uses it.

        Args:
            system_prompt_user (str): The experiment's domain knowledge about the
                use case.
        Returns:
            dict[str, Any]: The resulting analyser config with the full system
            prompt, so the run can log which model and prompt it used.
        """
        config = self._client.get_config("analyser", full_prompt=False)
        config["system_prompt_user"] = system_prompt_user
        self._client.upload_config("analyser", config)
        self._client.update_agents()
        return self._client.get_config("analyser")

    def run(self, input_file: Path, user_prompt: str) -> AnalyserRunOutcome:
        """Uploads the input file, runs the analyser on it and downloads the
        resulting plot and output data.

        An agent failure is returned as an outcome with execution_success=0, not
        raised; transport errors (e.g. backend not running) propagate.

        Args:
            input_file (Path): The experiment's CSV/Excel data file.
            user_prompt (str): The experiment's analysis request.
        Returns:
            AnalyserRunOutcome: Execution result plus whatever outputs exist.
        """
        with input_file.open("rb") as f:
            self._client.upload_data_file(UploadFile(filename=input_file.name, file=f))

        result = self._client.run_single_agent("analyser", user_prompt)

        # If the orchestrator fails first, the backend returns its response
        # shape instead of an AnalyserResult, so every field may be missing.
        error_message = result.get("error_message") or None
        outcome = AnalyserRunOutcome(
            execution_success=0 if error_message else 1,
            error_message=error_message,
            python_code=result.get("python_code") or "",
            code_summary=result.get("code_summary") or "",
            plot_png=None,
            output_csv=None,
        )

        download_errors = []
        if plot_path := result.get("plot_path"):
            try:
                outcome.plot_png = self._client.download_plot(Path(plot_path))
            except RuntimeError as e:
                download_errors.append(f"plot {plot_path}: {e}")
        if df_file_path := result.get("df_file_path"):
            try:
                outcome.output_csv = self._client.download_data(Path(df_file_path))
            except RuntimeError as e:
                download_errors.append(f"data {df_file_path}: {e}")
        if download_errors:
            outcome.download_error = "; ".join(download_errors)
            logger.error(f"Download failed: {outcome.download_error}")

        return outcome
