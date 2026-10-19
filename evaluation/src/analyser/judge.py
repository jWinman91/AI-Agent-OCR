"""LLM-as-a-judge for the analyser agent: one call scores the generated code, one
(vision) call scores the plot."""

import io
from pathlib import Path
from string import Template
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field
from pydantic_ai import Agent, BinaryContent
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings
from src.analyser.judge_config import (
    JudgeConfig,
)


class CriterionJudgement(BaseModel):
    # Field order is the generation order: the judge writes its reasoning and
    # evidence before committing to a score.
    explanation: str = Field(description="Reasoning for the score.")
    evidence: list[str] = Field(
        description="Concrete observations from the input supporting the score."
    )
    score: int = Field(ge=1, le=5, description="Score from 1 (worst) to 5 (best).")


class CodeJudgement(BaseModel):
    code_simplicity: CriterionJudgement


class VisualisationJudgement(BaseModel):
    data_correctness: CriterionJudgement
    task_relevance: CriterionJudgement
    chart_suitability: CriterionJudgement
    completeness: CriterionJudgement
    readability: CriterionJudgement


def describe_table(content: bytes, file_name: str, max_chars: int) -> str:
    """Renders a CSV/Excel table as text for the judge prompt: row count,
    header, then the CSV text cut at max_chars.

    CSV text is passed on as-is (keeping e.g. "60.394,59 €" formatting the plot
    has to interpret); Excel files are converted to CSV first.

    Args:
        content (bytes): The raw file content.
        file_name (str): File name, used to detect Excel files.
        max_chars (int): Maximum number of characters of table text.
    Returns:
        str: The table description.
    """
    if Path(file_name).suffix.lower() in (".xlsx", ".xls"):
        text = pd.read_excel(io.BytesIO(content)).to_csv(index=False)
    else:
        text = content.decode("utf-8", errors="replace")

    lines = text.strip().splitlines()
    header = lines[0] if lines else ""
    description = f"{max(len(lines) - 1, 0)} data rows. Header: {header}\n\n"
    if len(text) <= max_chars:
        return description + text
    return (
        description
        + text[:max_chars]
        + f"\n[... truncated: showing the first {max_chars} of {len(text)} "
        f"characters]"
    )


class LLMJudge:
    def __init__(self, config: JudgeConfig, model: Model | str | None = None) -> None:
        """Creates the code and visualisation judge agents.

        Args:
            config (JudgeConfig): The validated judge configuration.
            model (Model | str | None): Model override (tests pass a
                pydantic_ai test model); defaults to config.model.
        """
        self._config = config
        model = model if model is not None else config.model
        model_settings = ModelSettings(**config.model_settings)  # type: ignore[typeddict-item]
        self._code_agent = Agent(
            model,
            output_type=CodeJudgement,
            system_prompt=config.code_judge.system_prompt,
            model_settings=model_settings,
            output_retries=config.retries,
        )
        self._visualisation_agent = Agent(
            model,
            output_type=VisualisationJudgement,
            system_prompt=config.visualisation_judge.system_prompt,
            model_settings=model_settings,
            output_retries=config.retries,
        )
        self._preflight_agent = Agent(model, model_settings=model_settings)

    def check_model(self) -> None:
        """Sends one tiny request, so a wrong model name, API key or unsupported
        model setting fails before any analyser run instead of after it."""
        self._preflight_agent.run_sync("Reply with OK.")

    def judge_code(
        self, user_request: str, code_summary: str, python_code: str
    ) -> CodeJudgement:
        """Scores the generated code.

        Args:
            user_request (str): The user's analysis request.
            code_summary (str): The agent's summary of its code.
            python_code (str): The generated Python code.
        Returns:
            CodeJudgement: Explanation, evidence and score per code criterion.
        """
        prompt = Template(self._config.code_judge.user_prompt_template).substitute(
            user_request=user_request,
            code_summary=code_summary or "(none)",
            python_code=python_code,
            rubrics=self._config.code_judge.render_rubrics(),
        )
        return self._code_agent.run_sync(prompt).output

    def judge_visualisation(
        self,
        user_request: str,
        input_file: Path,
        plot_png: bytes,
        ground_truth_path: Path | None,
        output_csv: bytes | None,
    ) -> VisualisationJudgement:
        """Scores the plot against the request, the input data and the optional
        reference. The judge deliberately never sees the generated code, so it
        scores what is drawn rather than what the code claims to draw.

        Args:
            user_request (str): The user's analysis request.
            input_file (Path): The data file the analyser was given.
            plot_png (bytes): The plot under evaluation.
            ground_truth_path (Path | None): Reference .png plot or .csv
                expected output, if the experiment names one.
            output_csv (bytes | None): The analyser's output data, shown next to
                a .csv reference.
        Returns:
            VisualisationJudgement: Explanation, evidence and score per
            visualisation criterion.
        """
        max_chars = self._config.max_data_chars
        images = [BinaryContent(data=plot_png, media_type="image/png")]

        if ground_truth_path is None:
            reference_section = (
                "No reference is available; judge from the request and the "
                "input data only."
            )
        elif ground_truth_path.suffix.lower() == ".png":
            images.append(
                BinaryContent(
                    data=ground_truth_path.read_bytes(), media_type="image/png"
                )
            )
            reference_section = (
                "The second image attached is a reference plot showing what a good "
                "result for this request looks like."
            )
        else:
            reference = describe_table(
                ground_truth_path.read_bytes(), ground_truth_path.name, max_chars
            )
            output = (
                describe_table(output_csv, "output.csv", max_chars)
                if output_csv is not None
                else "(the agent produced no output data file)"
            )
            reference_section = (
                "The expected analysis result for this request (reference):\n"
                f"<REFERENCE_DATA>\n{reference}\n</REFERENCE_DATA>\n\n"
                "The data the agent's analysis produced:\n"
                f"<OUTPUT_DATA>\n{output}\n</OUTPUT_DATA>"
            )

        prompt = Template(
            self._config.visualisation_judge.user_prompt_template
        ).substitute(
            user_request=user_request,
            input_data=describe_table(
                input_file.read_bytes(), input_file.name, max_chars
            ),
            reference_section=reference_section,
            rubrics=self._config.visualisation_judge.render_rubrics(),
        )
        message: list[Any] = [prompt, *images]
        return self._visualisation_agent.run_sync(message).output
