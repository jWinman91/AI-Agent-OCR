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


class PreflightCheck(BaseModel):
    ok: bool


class CodeJudgement(BaseModel):
    code_simplicity: CriterionJudgement
    code_correctness: CriterionJudgement


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
    return description + _truncate(text, max_chars)


# Rows always shown at the start and end of the code judge's sample, besides
# the random ones, so it sees the row order and the edges of the file.
EDGE_ROWS = 5
# Columns with at most this many distinct values get all of them listed.
MAX_LISTED_VALUES = 20


def profile_table(
    content: bytes, file_name: str, sample_rows: int, max_chars: int
) -> str:
    """Renders a profile of a CSV/Excel table for the code judge, so it can spot
    code that doesn't fit the actual data (unexpected values, rows, formats or
    row order): shape, per-column dtype as the backend's pandas loads it,
    missing and distinct counts, all values of low-cardinality columns, and
    sample rows (first and last EDGE_ROWS plus sample_rows random ones, with a
    fixed seed so every run shows the same rows). Sample values are raw text,
    keeping formats like "60.394,59 €" the code has to parse.

    Args:
        content (bytes): The raw file content.
        file_name (str): File name, used to detect Excel files.
        sample_rows (int): Number of random rows between the first and last ones.
        max_chars (int): Maximum number of characters of the profile.
    Returns:
        str: The table profile.
    """
    inferred = _read_table(content, file_name)
    raw = _read_table(content, file_name, raw=True)

    lines = [f"{len(raw)} data rows, {len(raw.columns)} columns.", "", "Columns:"]
    for column in raw.columns:
        values = raw[column]
        line = (
            f"- {column}: dtype {inferred[column].dtype}, "
            f"{int(inferred[column].isna().sum())} missing, "
            f"{values.nunique()} distinct"
        )
        if values.nunique() <= MAX_LISTED_VALUES:
            counts = values.value_counts()
            line += "; values: " + ", ".join(
                f"{value!r} ({count})" for value, count in counts.items()
            )
        lines.append(line)

    if len(raw) <= sample_rows + 2 * EDGE_ROWS:
        sample = raw
        lines += ["", "All rows (row = 0-based position in the file):"]
    else:
        middle = raw.iloc[EDGE_ROWS:-EDGE_ROWS].sample(n=sample_rows, random_state=0)
        positions = [
            *range(EDGE_ROWS),
            *middle.index,
            *range(len(raw) - EDGE_ROWS, len(raw)),
        ]
        sample = raw.loc[sorted(positions)]
        lines += [
            "",
            f"Sample rows: the first {EDGE_ROWS}, {sample_rows} random ones and "
            f"the last {EDGE_ROWS}, in file order (row = 0-based position in "
            "the file):",
        ]
    lines.append(sample.to_csv(index_label="row"))
    return _truncate("\n".join(lines), max_chars)


def _read_table(content: bytes, file_name: str, raw: bool = False) -> pd.DataFrame:
    # raw=True keeps every value as its original text (no type inference, no
    # NaN conversion); otherwise pandas infers dtypes like the backend does.
    dtype = str if raw else None
    if Path(file_name).suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(io.BytesIO(content), dtype=dtype, keep_default_na=not raw)
    return pd.read_csv(io.BytesIO(content), dtype=dtype, keep_default_na=not raw)


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return (
        text[:max_chars]
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
        # Uses structured output like the real judges, so a model/API that
        # can't return it fails here too, not just a wrong name or key.
        self._preflight_agent = Agent(
            model, output_type=PreflightCheck, model_settings=model_settings
        )

    def check_model(self) -> None:
        """Sends one tiny structured-output request, so a wrong model name, API
        key, unsupported model setting or structured-output problem fails
        before any analyser run instead of after it."""
        self._preflight_agent.run_sync("Set ok to true.")

    def judge_code(
        self,
        user_request: str,
        code_summary: str,
        python_code: str,
        input_file: Path,
        error_message: str | None,
    ) -> CodeJudgement:
        """Scores the generated code. Besides the code it sees the execution
        result and a profile of the input data, so it can find bugs such as a
        crash's root cause or code that doesn't fit the data.

        Args:
            user_request (str): The user's analysis request.
            code_summary (str): The agent's summary of its code.
            python_code (str): The generated Python code.
            input_file (Path): The data file the analyser was given.
            error_message (str | None): The backend's error (traceback), None if
                execution succeeded.
        Returns:
            CodeJudgement: Explanation, evidence and score per code criterion.
        """
        max_chars = self._config.max_data_chars
        prompt = Template(self._config.code_judge.user_prompt_template).substitute(
            user_request=user_request,
            code_summary=code_summary or "(none)",
            python_code=python_code,
            execution_result=(
                "succeeded"
                if error_message is None
                else "failed:\n" + _truncate(error_message, max_chars)
            ),
            input_data=profile_table(
                input_file.read_bytes(),
                input_file.name,
                self._config.code_judge_sample_rows,
                max_chars,
            ),
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
