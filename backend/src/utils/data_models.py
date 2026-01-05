from typing import Any, Optional

from pydantic import BaseModel


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


class DataExtractorResponse(BaseModel):
    """
    Response model for the data extractor agent.
    """

    data: dict[str, Any]
    error_message: str | None = None


class DataExtractorPrompt(BaseModel):
    """
    Prompt model for the data extractor agent.
    """

    user_prompt: str
    json_details: dict[str, Any]


class OrchestratorResponse(BaseModel):
    """
    Response model for the orchestrator agent.
    """

    analyser_prompt: str
    data_extractor_prompt: DataExtractorPrompt
    error_message: str | None = None


class AnalyserResponse(BaseModel):
    """
    Response model for the analyser agent.
    """

    df_file_path: str | None
    plot_path: str | None
    code_summary: str
    error_message: str | None = None


class FixJsonResult(BaseModel):
    """
    Response model for the fix/repair JSON agent.
    """

    fixed_json: str
    error_message: str | None = None


AgentResponse = (
    DataExtractorResponse | OrchestratorResponse | AnalyserResponse | FixJsonResult
)


class AgentResult(BaseModel):
    """
    Result model for the agent.
    """

    plot_path: str | None = None
    code_summary: str | None = None
    data_file_path: str | None = None
