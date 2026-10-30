"""Backend operations shared by the extractor and analyser evaluators: refreshing
agents, uploading inputs, running a single agent, and downloading its outputs."""

from pathlib import Path
from typing import Any, Literal

from fastapi import UploadFile
from src.backend_client.request_be import BeRequest

AgentName = Literal["data_extractor", "analyser"]


class AgentClient:
    def __init__(
        self,
        ip: str = "127.0.0.1",
        port: int = 8000,
        protocol: str = "http",
    ) -> None:
        self._be_request = BeRequest(ip=ip, port=port, protocol=protocol)

    def update_agents(self) -> None:
        """Rebuilds the backend agents from the current config, so the backend picks
        up whatever config change the user made before starting a run."""
        self._be_request.post("update_agents")

    def upload_images(self, files: list[UploadFile]) -> None:
        """Uploads images for the data extractor via /uploadimages.

        Args:
            files (list[UploadFile]): Image files to upload.
        """
        self._be_request.post("uploadimages", {"resize": "medium"}, files)

    def upload_pdf(self, file: UploadFile) -> None:
        """Uploads a single PDF for the data extractor via /uploadpdf; the backend
        converts it into per-page images and texts itself.

        Args:
            file (UploadFile): PDF file to upload.
        """
        self._be_request.post("uploadpdf", {"resize": "medium"}, file)

    def upload_data_file(self, file: UploadFile) -> None:
        """Uploads a CSV/Excel dataset for the analyser via /uploaddatafile.

        Args:
            file (UploadFile): Data file to upload.
        """
        self._be_request.post("uploaddatafile", {}, file)

    def run_single_agent(
        self, agent_name: AgentName, user_prompt: str
    ) -> dict[str, Any]:
        """Runs one agent on the previously uploaded inputs via /run_single_agent.

        The backend reports agent failures as a 200 response with error_message
        set, so the returned result is not checked here - the caller decides
        whether a failure aborts the run or is scored as a result.

        Args:
            agent_name (AgentName): The agent to run.
            user_prompt (str): User prompt to guide the agent.
        Returns:
            dict[str, Any]: The agent result, e.g. data_file_path for the data
            extractor or plot_path/df_file_path/python_code for the analyser,
            plus error_message.
        """
        # Same route/payload shape as the frontend's single agent workflow.
        result = self._be_request.post(
            "run_single_agent",
            {
                "user_prompt": user_prompt,
                "agent_name": agent_name,
                "previous_error": None,
            },
            payload_type="data",
        )
        assert isinstance(result, dict)
        return result

    def get_config(
        self, agent_name: AgentName, full_prompt: bool = True
    ) -> dict[str, Any]:
        """Gets an agent's configuration.

        Args:
            agent_name (AgentName): The agent to get the configuration for.
            full_prompt (bool): If True, domain_knowledge_prompt holds the full built
                system prompt; otherwise it holds only the user-editable part, as
                stored and as upload_config expects it.
        Returns:
            dict[str, Any]: The agent configuration.
        """
        config = self._be_request.get(
            path="get_config",
            params={
                "config_name": agent_name,
                "full_prompt": "true" if full_prompt else "false",
            },
            data_type=None,
        )
        assert isinstance(config, dict)
        return config

    def upload_config(self, agent_name: AgentName, config: dict[str, Any]) -> None:
        """Stores an agent's configuration via /upload_config, like saving it in
        the UI. Takes effect for runs after the next update_agents.

        Args:
            agent_name (AgentName): The agent to configure.
            config (dict[str, Any]): The configuration, in the shape get_config
                returns with full_prompt=False.
        """
        self._be_request.post("upload_config/", agent_name, config)

    def download_data(self, df_path: Path) -> bytes:
        """Downloads a CSV produced by an agent via /df_path/.../download_data.

        Args:
            df_path (Path): Backend path of the data file.
        Returns:
            bytes: The raw CSV content.
        """
        content = self._be_request.get(
            path=f"df_path/{df_path}/download_data", params=None, data_type="data"
        )
        assert isinstance(content, bytes)
        return content

    def download_plot(self, plot_path: Path) -> bytes:
        """Downloads a plot produced by the analyser via
        /plot_path/.../download_plot.

        Args:
            plot_path (Path): Backend path of the plot.
        Returns:
            bytes: The raw PNG content.
        """
        # data_type="data" returns the raw bytes, so the PNG can be written to
        # disk or passed to a judge model without re-encoding.
        content = self._be_request.get(
            path=f"plot_path/{plot_path}/download_plot", params=None, data_type="data"
        )
        assert isinstance(content, bytes)
        return content


def current_domain_knowledge_prompt(client: AgentClient, agent_name: AgentName) -> str:
    """Returns an agent's stored domain_knowledge_prompt, so it can be restored
    after the experiments.

    Args:
        client (AgentClient): The backend client.
        agent_name (AgentName): The agent to read the prompt of.
    Returns:
        str: The user-editable part of the agent's system prompt.
    """
    return str(
        client.get_config(agent_name, full_prompt=False)["domain_knowledge_prompt"]
    )


def configure_domain_knowledge_prompt(
    client: AgentClient, agent_name: AgentName, domain_knowledge_prompt: str
) -> dict[str, Any]:
    """Sets an agent's domain_knowledge_prompt via /upload_config, keeping the rest
    of its stored config (model, provider, settings), and rebuilds the backend
    agents so the next run uses it.

    Args:
        client (AgentClient): The backend client.
        agent_name (AgentName): The agent to configure.
        domain_knowledge_prompt (str): The experiment's domain knowledge about the
            use case.
    Returns:
        dict[str, Any]: The resulting agent config with the full system prompt,
        so the run can log which model and prompt it used.
    """
    config = client.get_config(agent_name, full_prompt=False)
    config["domain_knowledge_prompt"] = domain_knowledge_prompt
    client.upload_config(agent_name, config)
    client.update_agents()
    return client.get_config(agent_name)
