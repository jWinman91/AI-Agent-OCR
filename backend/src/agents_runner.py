import datetime
import logging
import time
from typing import Any, Dict, List, Literal, Tuple

import pandas
import pandas as pd
from loguru import logger
from PIL import Image
from pydantic_ai import Agent, BinaryContent
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
from src.utils.data_models import (
    AgentResponse,
    AgentResult,
    AnalyserResponse,
    ConfigModelBuild,
    DataExtractorPrompt,
    DataExtractorResponse,
    OrchestratorResponse,
)
from src.utils.output_validator import OutputValidator

logging.basicConfig(level=logging.INFO)


class AgentRunner:
    def __init__(
        self,
        config: Dict[str, ConfigModelBuild],
        data_dir: str = "data",
        plots_dir: str = "plots",
    ) -> None:
        """
        Initialize the AgentRunner with configuration and directories.

        param config: Configuration dictionary for agents
        param data_dir: Directory to store data files
        param plots_dir: Directory to store plot files
        """
        # Create directories for data and plots if they don't exist
        self.DATA_DIR = data_dir
        self.PLOTS_DIR = plots_dir

        # Save config
        self.CONFIG = config
        self.AGENT_NAMES = list(config.keys())

        # Initialize output validator
        self.OUTPUT_VALIDATOR = OutputValidator(
            create_agent_func=self.create_agent,
            run_agent_func=self.run_agent,
            config=self.CONFIG["fix_json"],
        )

    @staticmethod
    def _fill_system_prompt(
        system_prompt: str, args: Dict[str, Any] | None = None
    ) -> str:
        """
        Fill in the system prompt with provided arguments.

        param system_prompt: The system prompt template
        param args: Arguments to format the system prompt
        return: Formatted system prompt
        """
        if args is None:
            return system_prompt
        return system_prompt.format(**args)

    @staticmethod
    def _error_handling_prompt(
        agent_name: Literal["data_extractor", "orchestrator", "analyser"],
        error_message: str,
        user_prompt: str,
        previous_errors: Dict[str, str] | None = None,
    ) -> Dict[str, str]:
        """
        Build or update the error handling prompt for the agent.

        param agent_name: Name of the agent
        param error_message: The error message to include
        param user_prompt: The original user prompt
        param previous_errors: Previous errors to include
        return: Updated dictionary of previous errors
        """
        if previous_errors is None:
            previous_errors = {user_prompt: error_message}
        else:
            previous_errors[user_prompt] += f"{agent_name}: {error_message} \n"

        if len(previous_errors) > 3:
            raise ValueError(
                f"Too many errors in {agent_name}: {previous_errors}."
                f"Stopping execution."
            )
        return previous_errors

    @staticmethod
    def _get_image_timestamp(image_url: str) -> datetime.datetime:
        """
        Load in an image and get the timestamp from the image metadata.

        :param image_url: Path to the image file
        :return: Timestamp extracted from the image metadata or current time
                 if not available
        """
        if not image_url.endswith(".pdf"):
            exif = Image.open(image_url)._getexif()
            if exif is not None and 36867 in list(exif.keys()):
                return datetime.datetime.strptime(exif[36867], "%Y:%m:%d %H:%M:%S")

        return datetime.datetime.strptime(
            datetime.datetime.now().strftime("%Y:%m:%d %H:%M:%S"), "%Y:%m:%d %H:%M:%S"
        )

    @staticmethod
    def _build_error_messages(previous_errors: Dict[str, str] | None) -> str:
        """
        Build a formatted string of previous error messages.

        param previous_errors: Dictionary of previous errors
        return: Formatted string of errors
        """
        if not previous_errors:
            return ""
        return "\n".join(
            [f"- User Prompt: {k}\nError: {v}" for k, v in previous_errors.items()]
        )

    def create_agent(
        self,
        agent_name: str,
        config: ConfigModelBuild,
        output_type: type[AgentResponse],
        system_prompt_args: Dict[str, Any] | None = None,
    ) -> Agent:
        """
        Create and configure an agent based on the provided configuration.

        param agent_name: Name of the agent to create
        param config: Configuration model for the agent
        param output_type: Expected output type of the agent
        param system_prompt_args: Arguments to format the system prompt
        return: Configured Agent instance
        """
        if agent_name not in self.AGENT_NAMES:
            raise ValueError(
                f"Invalid agent {agent_name}. Agent must be one of {self.AGENT_NAMES}"
            )

        cfg = config.model_dump()
        cfg["system_prompt"] = self._fill_system_prompt(
            config.system_prompt, system_prompt_args
        )

        # set base url if given in config and delete it from cfg
        base_url = cfg.pop("base_url", "http://localhost:11434/v1")

        # set output type if given in config
        if cfg.pop("output_type", False):
            cfg["output_type"] = output_type

        # initiate model with model provider if ollama should be used
        model_provider = cfg.pop("model_provider", "openai")
        if model_provider == "ollama":
            cfg["model"] = OpenAIChatModel(
                model_name=cfg["model"],
                provider=OllamaProvider(base_url=base_url),
            )
        else:
            cfg["model"] = f"{model_provider}:{cfg['model']}"

        # set MCP server if necessary
        mcp_servers_cfg = cfg.pop("mcp_servers", None)
        if mcp_servers_cfg is not None:
            mcp_servers = []
            for mcp_server_cfg in mcp_servers_cfg:
                mcp_server = MCPServerStdio(**mcp_server_cfg)
                mcp_servers.append(mcp_server)
            cfg["toolsets"] = mcp_servers
            logger.info(
                f"Configured MCP servers for agent {agent_name} and model"
                f" {cfg['model']}: {cfg['toolsets']}"
            )
        agent = Agent(**cfg)
        agent.name = agent_name
        return agent

    async def run_agent(
        self,
        agent: Agent,
        output_type_example: AgentResponse,
        user_prompt: str,
        image_url: str | None = None,
    ) -> AgentResponse:
        """
        Runs an agent with the given user prompt and optional image.

        :param agent: The agent to run.
        :param output_type_example: Example of the expected output type for validation.
        :param user_prompt: The user prompt to provide to the agent.
        :param image_url: Optional path to an image file to provide to the agent.
        :return: The agent's response, validated explicitly if it's a string output.
        """
        t0 = time.time()
        if image_url is not None:
            with open(image_url, "rb") as f:
                file_content = f.read()
            binary_content = BinaryContent(
                data=file_content,
                media_type="image/png",
            )
            response = await agent.run(
                [
                    user_prompt,
                    binary_content,
                ]
            )
        else:
            response = await agent.run([user_prompt])

        image_log = f" with image {image_url}" if image_url is not None else ""

        logger.info(
            f"{agent.name} response in {time.time() - t0:.2f} sec"
            f"{image_log}: {response}"
        )

        if type(response.output) is str:
            return await self.OUTPUT_VALIDATOR.validate_output(
                response.output, agent.name if agent.name else "", output_type_example
            )
        else:
            return response.output

    async def run_orchestrator(
        self, user_prompt: str, system_prompt_args: dict[str, Any] | None = None
    ) -> OrchestratorResponse:
        """
        Runs the orchestrator agent to determine extraction and plotting instructions.

        :param user_prompt: The user prompt to provide to the orchestrator.
        :param system_prompt_args: Optional arguments to format the system prompt.
        :return: The orchestrator's response containing extraction and plotting
        instructions.
        """
        # Orchestrator decides how to extract and plot
        # based on user prompt and previous error
        orchestrator = self.create_agent(
            agent_name="orchestrator",
            config=self.CONFIG["orchestrator"],
            system_prompt_args=system_prompt_args,
            output_type=OrchestratorResponse,
        )
        orchestrator_res = await self.run_agent(
            agent=orchestrator,
            user_prompt=user_prompt,
            output_type_example=OrchestratorResponse(
                plot_prompt="str",
                data_extractor_prompt=DataExtractorPrompt(
                    user_prompt="str",
                    json_details={"field_name": "str", "data_type": "str"},
                ),
                error_message="null or error message",
            ),
        )

        if orchestrator_res.error_message and orchestrator_res.error_message != "":
            logger.error(f"Error in orchestrator: {orchestrator_res.error_message}")
            raise ValueError(f"Error in orchestrator: {orchestrator_res.error_message}")

        return orchestrator_res

    async def run_data_extractor(
        self,
        user_prompt: str,
        image_urls: List[str],
        previous_errors_data_extractor: dict[str, str] | None = None,
        previous_errors_analyser: dict[str, str] | None = None,
    ) -> Tuple[pandas.DataFrame, str]:
        """
        Runs the data extractor agent on a list of image URLs.

        :param user_prompt: The user prompt to provide to the data extractor.
        :param image_urls: List of image URLs to process.
        :param previous_errors_data_extractor: Previous errors from the data extractor
                                               agent.
        :param previous_errors_analyser: Previous errors from the analyser agent.
        :return: A tuple containing the extracted data as a DataFrame and the plot
                 prompt.
        """
        system_prompt_args_orchestrator = {
            "previous_errors_data_extractor": self._build_error_messages(
                previous_errors_data_extractor
            ),
            "previous_errors_analyser": self._build_error_messages(
                previous_errors_analyser
            ),
        }

        # Run the orchestrator to get extraction instructions
        orchestrator_res = await self.run_orchestrator(
            user_prompt, system_prompt_args_orchestrator
        )

        # Create data_extractor agent based on orchestrator's instructions
        data_extractor = self.create_agent(
            "data_extractor",
            self.CONFIG["data_extractor"],
            DataExtractorResponse,
            {"json_details": orchestrator_res.data_extractor_prompt.json_details},
        )

        # Run data_extractor for each image
        user_prompt_data_extractor = orchestrator_res.data_extractor_prompt.user_prompt
        df = pd.DataFrame()
        timestamps = []
        for image_url in image_urls:
            data_extractor_res = await self.run_agent(
                agent=data_extractor,
                user_prompt=user_prompt_data_extractor,
                image_url=image_url,
                output_type_example=DataExtractorResponse(
                    data={"field_name": "value"},
                    error_message="null or error message",
                ),
            )

            # Retry if error in data extractor
            if (
                data_extractor_res.error_message
                and data_extractor_res.error_message != ""
            ):
                logger.error(
                    f"Error in data extractor for image {image_url}:"
                    f"{data_extractor_res.error_message}"
                )
                previous_errors_data_extractor = self._error_handling_prompt(
                    agent_name="data_extractor",
                    error_message=data_extractor_res.error_message,
                    user_prompt=user_prompt,
                    previous_errors=previous_errors_data_extractor,
                )
                return await self.run_data_extractor(
                    user_prompt=user_prompt,
                    image_urls=image_urls,
                    previous_errors_data_extractor=previous_errors_data_extractor,
                    previous_errors_analyser=previous_errors_analyser,
                )

            timestamps.append(self._get_image_timestamp(image_url))

            df = pd.concat(
                [df, pd.DataFrame([data_extractor_res.data])], ignore_index=True
            )
        df["timestamp"] = timestamps
        df["image_url"] = image_urls

        # Explode dataframe if any column contains a list or tuple
        cols_to_explode = [
            col
            for col in df.columns
            if df[col].apply(lambda x: isinstance(x, (list, tuple))).any()
        ]

        if cols_to_explode:
            df = df.explode(cols_to_explode, ignore_index=True)

        logger.info(f"Extracted DataFrame:\n{df}")

        return df, orchestrator_res.analyser_prompt

    async def run_analyser(
        self,
        user_prompt: str,
        system_prompt_args: dict,
    ) -> AnalyserResponse:
        """
        Runs the analyser agent to generate plots based on extracted data.

        :param user_prompt: The user prompt to provide to the analyser.
        :param previous_errors_analyser: Previous errors from the analyser agent.
        :param system_prompt_args: Additional arguments for the system prompt.
        :return: The analyser's response containing plot details.
        """
        # Create analyser agent based on orchestrator's instructions
        analyser = self.create_agent(
            agent_name="analyser",
            config=self.CONFIG["analyser"],
            output_type=AnalyserResponse,
            system_prompt_args=system_prompt_args,
        )
        return await self.run_agent(
            agent=analyser,
            user_prompt=user_prompt,
            output_type_example=AnalyserResponse(
                df_file_path="str",
                plot_path="str",
                tool_used="str",
                code_summary="str",
                error_message="null or error message",
            ),
        )

    async def run_ai_ocr_agents(
        self,
        user_prompt: str,
        image_urls: List[str],
        previous_errors_data_extractor: Dict[str, Any] | None = None,
        previous_errors_analyser: Dict[str, Any] | None = None,
    ) -> AgentResult:
        """
        Runs the full AI OCR pipeline: data extractor and analyser agents.

        :param user_prompt: The user prompt to provide to the agents.
        :param image_urls: List of image URLs to process.
        :param previous_errors_data_extractor: Previous errors from the data extractor
        agent.
        :param previous_errors_analyser: Previous errors from the analyser agent.
        :return: The final result containing plot and data details.
        """
        df, plot_prompt = await self.run_data_extractor(
            user_prompt=user_prompt,
            image_urls=image_urls,
            previous_errors_data_extractor=previous_errors_data_extractor,
            previous_errors_analyser=previous_errors_analyser,
        )
        df_path = f"{self.DATA_DIR}/extracted_data_from_ai_ocr_agents.csv"
        df.to_csv(df_path, index=False)

        # Analyser generates plot based on orchestrator's instructions and extracted data
        system_prompt_args_analyser = {
            "data_file_path": df_path,
            "output_dir": self.PLOTS_DIR,
        }
        plot_res = await self.run_analyser(
            plot_prompt,
            system_prompt_args_analyser,
        )

        if plot_res.error_message and plot_res.error_message != "":
            logger.error(f"Error in analyser: {plot_res.error_message}")
            previous_errors_analyser = self._error_handling_prompt(
                agent_name="analyser",
                error_message=plot_res.error_message,
                user_prompt=user_prompt,
                previous_errors=previous_errors_analyser,
            )
            return await self.run_ai_ocr_agents(
                user_prompt=user_prompt,
                image_urls=image_urls,
                previous_errors_data_extractor=previous_errors_data_extractor,
                previous_errors_analyser=previous_errors_analyser,
            )

        return AgentResult(
            plot_path=plot_res.plot_path,
            code_summary=plot_res.code_summary,
            data_file_path=df_path,
        )
