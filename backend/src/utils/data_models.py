from typing import Any, Optional, Union

from pydantic import BaseModel


class ConfigModelBase(BaseModel):
    model: str
    model_provider: Optional[str] = None
    mcp_server: Optional[str] = None
    output_type: Optional[bool] = False
    base_url: Optional[str] = "http://localhost:11434/v1"


class ConfigModelBuild(ConfigModelBase):
    system_prompt: str


class ConfigModelUser(ConfigModelBase):
    system_prompt_user: str


class ConfigModelDB(ConfigModelUser):
    system_prompt_template: str


class ExtractorResponse(BaseModel):
    data: dict[str, Any]
    error_message: str | None = None


class ExtractorPrompt(BaseModel):
    user_prompt: str
    json_details: dict[str, Any]


class OrchestratorResponse(BaseModel):
    plot_prompt: str
    extractor_prompt: ExtractorPrompt
    error_message: str | None = None


class PlotterResponse(BaseModel):
    df_file_path: str
    plot_path: str
    tool_used: str
    code_summary: str
    error_message: str | None = None


class FixJsonResult(BaseModel):
    fixed_json: str
    error_message: str | None = None


AgentResponse = Union[
    ExtractorResponse, OrchestratorResponse, PlotterResponse, FixJsonResult
]


class AgentResult(BaseModel):
    plot_path: str = ""
    tool_used: str = ""
    code_summary: str = ""
    data_file_path: str = ""
