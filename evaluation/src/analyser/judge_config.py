"""Loads and validates the LLM judge configuration (judge_config.yaml)."""

import hashlib
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from src.analyser import JUDGE_CONFIG_PATH

CODE_CRITERIA = ["code_simplicity"]
VISUALISATION_CRITERIA = [
    "data_correctness",
    "task_relevance",
    "chart_suitability",
    "completeness",
    "readability",
]
# Every metric the analyser evaluation logs, in a fixed order: execution
# success from the backend plus one 1-5 score per judged criterion.
ALL_METRIC_NAMES = ["execution_success", *CODE_CRITERIA, *VISUALISATION_CRITERIA]

SCORE_LEVELS = {1, 2, 3, 4, 5}


class CriterionConfig(BaseModel):
    description: str
    rubric: dict[int, str]

    @field_validator("rubric")
    @classmethod
    def _check_levels(cls, rubric: dict[int, str]) -> dict[int, str]:
        if set(rubric) != SCORE_LEVELS:
            raise ValueError(
                f"rubric must define exactly the levels 1-5, got {sorted(rubric)}"
            )
        return rubric


class JudgeSectionConfig(BaseModel):
    """One judge: its prompts and the criteria it scores."""

    system_prompt: str
    user_prompt_template: str
    criteria: dict[str, CriterionConfig]

    def render_rubrics(self) -> str:
        """Renders the description and 1-5 rubric of each of this judge's
        criteria as plain text for the judge prompt.

        Returns:
            str: One block per criterion, in config order.
        """
        blocks = []
        for name, criterion in self.criteria.items():
            levels = "\n".join(
                f"  {level}: {criterion.rubric[level]}"
                for level in sorted(criterion.rubric)
            )
            blocks.append(f"{name}: {criterion.description}\n{levels}")
        return "\n\n".join(blocks)


class JudgeConfig(BaseModel):
    model: str
    model_settings: dict[str, Any] = Field(default_factory=dict)
    max_data_chars: int = Field(gt=0)
    retries: int = Field(ge=0)
    code_judge: JudgeSectionConfig
    visualisation_judge: JudgeSectionConfig

    @model_validator(mode="after")
    def _check_criteria(self) -> "JudgeConfig":
        # Each judge's structured output (CodeJudgement/VisualisationJudgement)
        # has a fixed set of fields, so its config must define exactly those.
        for judge_name, judge, expected in (
            ("code_judge", self.code_judge, CODE_CRITERIA),
            ("visualisation_judge", self.visualisation_judge, VISUALISATION_CRITERIA),
        ):
            if set(judge.criteria) != set(expected):
                raise ValueError(
                    f"{judge_name}.criteria must be exactly {sorted(expected)}, "
                    f"got {sorted(judge.criteria)}"
                )
        return self


def load_judge_config(path: Path = JUDGE_CONFIG_PATH) -> JudgeConfig:
    """Loads and validates the judge configuration, so a broken config fails
    before any backend or judge call.

    Args:
        path (Path): Path to the judge config yaml.
    Returns:
        JudgeConfig: The validated configuration.
    """
    with path.open("r") as f:
        return JudgeConfig.model_validate(yaml.safe_load(f))


def judge_config_sha(path: Path = JUDGE_CONFIG_PATH) -> str:
    """Short content hash of the judge config, logged with every run so scores
    are only compared between runs judged with the same prompts and rubrics.

    Args:
        path (Path): Path to the judge config yaml.
    Returns:
        str: The first 12 hex characters of the file's SHA-256.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]
