from pydantic import BaseModel, Field
from typing import Any, Dict, Union


class ExtractorPrompt(BaseModel):
    user_prompt: str
    json_details: Dict[str, Any]


class OrchestratorResult(BaseModel):
    extractor_prompt: ExtractorPrompt
    plot_prompt: str
    error: str | None


class PlotterResultOutput(BaseModel):
    plot_path: str
    tool_used: str
    code_summary: str


class PlotterResult(BaseModel):
    body: PlotterResultOutput
    error: str | None = None


class ExtractorResult(BaseModel):
    data: Dict[str, Any]
    error: str | None | bool = None


class OrchestratorOutput(PlotterResultOutput):
    df_path: str


class FixJsonResult(BaseModel):
    text: str
    error: str | None = None


AgentResult = Union[OrchestratorResult, PlotterResult, ExtractorResult]


class ConfigModel(BaseModel):
    name: str
    config: Dict[str, Any] = Field(default_factory=dict)