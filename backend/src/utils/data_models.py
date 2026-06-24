from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field


class YFinanceRequest(BaseModel):
    share_name: str = Field(
        ..., description="Name of the share to download data for, e.g. 'E.ON'"
    )
    period: Optional[str] = Field(
        "1mo", description="Data period, e.g. 1d, 5d, 1mo, 1y"
    )
    interval: Optional[str] = Field("1d", description="Data interval, e.g. 1m, 1h, 1d")
    start: Optional[str] = Field(None, description="Start date YYYY-MM-DD")
    end: Optional[str] = Field(None, description="End date YYYY-MM-DD")


# Config models
class ConfigModelBase(BaseModel):
    """
    Base configuration model including only base infos.
    """

    model: str
    model_provider: Optional[str] = "openai"
    mcp_servers: list[dict[str, Any]] | None = None
    output_type: Optional[bool] = False
    base_url: Optional[str] = "http://localhost:11434/v1"


class ConfigModelBuild(ConfigModelBase):
    """
    Configuration model for the config after building.
    """

    system_prompt: str


class ConfigModelUser(ConfigModelBase):
    """
    Configuration model for fields set by the user.
    """

    system_prompt_user: str


class ConfigModelDB(ConfigModelUser):
    """
    Configuration model for fields stored in the database.
    """

    system_prompt_template: str


# Prompt model for the data extractor agent
class DataExtractorPrompt(BaseModel):
    """
    Prompt model for the data extractor agent.
    """

    user_prompt: str
    json_details: dict[str, Any]


# Response models for the LLM agents
class DataExtractorResponse(BaseModel):
    """
    Response model for the data extractor agent.
    """

    data: dict[str, Any]
    error_message: str | None = None


class OrchestratorResponse(BaseModel):
    """
    Response model for the orchestrator agent.
    """

    analyser_prompt: str
    data_download_prompt: str
    data_extractor_prompt: DataExtractorPrompt
    error_message: str | None = None


class AnalyserResponse(BaseModel):
    """
    Response model for the analyser agent.
    """

    python_code: str
    code_summary: str


class DataDownloadResponse(BaseModel):
    """
    Response model for the data download agent.
    """

    download_request: list[YFinanceRequest]
    file_name: Path


class FixJsonResponse(BaseModel):
    """
    Response model for the fix/repair JSON agent.
    """

    fixed_json: str
    error_message: str | None = None


AgentResponse = (
    DataExtractorResponse
    | OrchestratorResponse
    | AnalyserResponse
    | FixJsonResponse
    | DataDownloadResponse
)


# Result models after execution of the agents
class AnalyserResult(BaseModel):
    """
    Response model for the analyser agent.
    """

    df_file_path: Path | None
    plot_path: Path | None
    code_summary: str
    error_message: str | None = None


class DataDownloadResult(BaseModel):
    """
    Result model for the data download agent.
    """

    data_file_path: Path | None
    error_message: str | None = None


class DataExtractorResult(BaseModel):
    """
    Result model for the data extractor agent.
    """

    data_file_path: Path | None
    error_message: str | None = None


AgentResult = (
    DataExtractorResponse | OrchestratorResponse | AnalyserResult | DataDownloadResult
)
