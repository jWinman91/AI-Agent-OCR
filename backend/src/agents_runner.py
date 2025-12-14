import datetime
import logging
from pathlib import Path
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
    ConfigModelBuild,
    ExtractorPrompt,
    ExtractorResponse,
    OrchestratorResponse,
    PlotterResponse,
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
        )

    @staticmethod
    def _fill_system_prompt(
        system_prompt: str, args: Dict[str, Any] | None = None
    ) -> str:
        if args is None:
            return system_prompt
        return system_prompt.format(**args)

    @staticmethod
    def _error_handling_prompt(
        agent_name: Literal["extractor", "orchestrator", "plotter"],
        error_message: str,
        user_prompt: str,
        previous_errors: Dict[str, str] | None = None,
    ) -> Dict[str, str]:
        if previous_errors is None:
            previous_errors = {user_prompt: error_message}
        else:
            previous_errors[user_prompt] += error_message

        if len(previous_errors) > 3:
            raise ValueError(
                f"Too many errors in {agent_name}: {previous_errors}."
                f"Stopping execution."
            )
        return previous_errors

    @staticmethod
    def _get_image_timestamp(image_url: str) -> datetime.datetime:
        """
        Load in data and get the timestamp from the image metadata.
        :param image_url:
        :return:
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
        if cfg.pop("model_provider", None) == "ollama":
            cfg["model"] = OpenAIChatModel(
                model_name=cfg["model"],
                provider=OllamaProvider(base_url=base_url),
            )

        # set MCP server if necessary
        if cfg.pop("mcp_server", None):
            mcp_server = MCPServerStdio(**cfg["mcp_server"])
            cfg["toolsets"] = [mcp_server]

        return Agent(**cfg)

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
        :param user_prompt: The user prompt to provide to the agent.
        :param image_url: Optional path to an image file to provide to the agent.
        :return: The agent's response, validated explicitly if it's a string output.
        """
        if image_url is not None:
            with open(image_url, "rb") as f:
                file_content = f.read()
            binary_content = BinaryContent(
                filename=Path(image_url).name,
                content=file_content,
                mime_type="image/png",
            )
            response = await agent.run(user_prompt=user_prompt, file=binary_content)
        else:
            response = await agent.run(user_prompt=user_prompt)

        if type(response.output) is str:
            return await self.OUTPUT_VALIDATOR.validate_output(
                response.output, agent.name if agent.name else "", output_type_example
            )
        else:
            return response

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
                extractor_prompt=ExtractorPrompt(
                    user_prompt="str",
                    json_details={"field_name": "str", "data_type": "str"},
                ),
                error_message="null or error message",
            ),
        )

        if orchestrator_res.error_message and orchestrator_res.error_message != "":
            logger.error(f"Error in orchestrator: {orchestrator_res.error_message}")
            raise ValueError(f"Error in orchestrator: {orchestrator_res.error_message}")

        logger.info(f"orchestrator: {orchestrator_res}")

        return orchestrator_res

    async def run_extractor(
        self,
        user_prompt: str,
        image_urls: List[str],
        previous_errors_extractor: dict[str, str] | None = None,
        previous_errors_plotter: dict[str, str] | None = None,
    ) -> Tuple[pandas.DataFrame, str]:
        """
        Runs the extractor agent on a list of image URLs.

        :param user_prompt: The user prompt to provide to the extractor.
        :param image_urls: List of image URLs to process.
        :param previous_errors_extractor: Previous errors from the extractor agent.
        :param previous_errors_plotter: Previous errors from the plotter agent.
        :return: A tuple containing the extracted data as a DataFrame and the plot prompt.
        """
        system_prompt_args_orchestrator = {
            "previous_errors_extractor": self._build_error_messages(
                previous_errors_extractor
            ),
            "previous_errors_plotter": self._build_error_messages(
                previous_errors_plotter
            ),
        }

        # Run the orchestrator to get extraction instructions
        orchestrator_res = await self.run_orchestrator(
            user_prompt, system_prompt_args_orchestrator
        )

        # Create extractor agent based on orchestrator's instructions
        extractor = self.create_agent(
            "extractor",
            self.CONFIG["extractor"],
            ExtractorResponse,
            {"json_details": orchestrator_res.extractor_prompt.json_details},
        )

        # Run extractor for each image
        user_prompt_extractor = orchestrator_res.extractor_prompt.user_prompt
        df = pd.DataFrame()
        timestamps = []
        for image_url in image_urls:
            extractor_res = await self.run_agent(
                agent=extractor,
                user_prompt=user_prompt_extractor,
                image_url=image_url,
                output_type_example=ExtractorResponse(
                    data={"field_name": "value"},
                    error_message="null or error message",
                ),
            )

            logger.info(f"extractor for image {image_url}: {extractor_res}")

            # Retry if error in extractor
            if extractor_res.error_message and extractor_res.error_message != "":
                logger.error(
                    f"Error in extractor for image {image_url}:"
                    f"{extractor_res.error_message}"
                )
                previous_errors_extractor = self._error_handling_prompt(
                    agent_name="extractor",
                    error_message=extractor_res.error_message,
                    user_prompt=user_prompt_extractor,
                    previous_errors=previous_errors_extractor,
                )
                return await self.run_extractor(
                    user_prompt=user_prompt,
                    image_urls=image_urls,
                    previous_errors_extractor=previous_errors_extractor,
                    previous_errors_plotter=previous_errors_plotter,
                )

            timestamps.append(self._get_image_timestamp(image_url))

            df = pd.concat([df, pd.DataFrame([extractor_res.data])], ignore_index=True)
        df["timestamp"] = timestamps

        return df, orchestrator_res.plot_prompt

    async def run_plotter(
        self,
        user_prompt: str,
        system_prompt_args: dict,
    ) -> PlotterResponse:
        """
        Runs the plotter agent to generate plots based on extracted data.

        :param user_prompt: The user prompt to provide to the plotter.
        :param previous_errors_plotter: Previous errors from the plotter agent.
        :param system_prompt_args: Additional arguments for the system prompt.
        :return: The plotter's response containing plot details.
        """
        # Create plotter agent based on orchestrator's instructions
        plotter = self.create_agent(
            agent_name="plotter",
            config=self.CONFIG["plotter"],
            output_type=PlotterResponse,
            system_prompt_args=system_prompt_args,
        )
        return await self.run_agent(
            agent=plotter,
            user_prompt=user_prompt,
            output_type_example=PlotterResponse(
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
        previous_errors_extractor: Dict[str, Any] | None = None,
        previous_errors_plotter: Dict[str, Any] | None = None,
    ) -> AgentResult:
        """
        Runs the full AI OCR pipeline: extractor and plotter agents.

        :param user_prompt: The user prompt to provide to the agents.
        :param image_urls: List of image URLs to process.
        :param previous_errors_extractor: Previous errors from the extractor agent.
        :param previous_errors_plotter: Previous errors from the plotter agent.
        :return: The final result containing plot and data details.
        """
        df, plot_prompt = self.run_extractor(
            user_prompt=user_prompt,
            image_urls=image_urls,
            previous_errors_extractor=previous_errors_extractor,
            previous_errors_plotter=previous_errors_plotter,
        )
        df_path = f"{self.DATA_DIR}/extracted_data_from_ai_ocr_agents.csv"
        df.to_csv(df_path, index=False)

        # Plotter generates plot based on orchestrator's instructions and extracted data
        system_prompt_args_plotter = {
            "data_path": df_path,
            "output_dir": self.PLOTS_DIR,
        }
        plot_res = await self.run_plotter(
            plot_prompt,
            system_prompt_args_plotter,
        )

        if plot_res.error_message and plot_res.error_message != "":
            logger.error(f"Error in plotter: {plot_res.error_message}")
            previous_errors_plotter = self._error_handling_prompt(
                agent_name="plotter",
                error_message=plot_res.error_message,
                user_prompt=plot_prompt,
                previous_errors=previous_errors_plotter,
            )
            return await self.run_ai_ocr_agents(
                user_prompt=user_prompt,
                image_urls=image_urls,
                previous_errors_extractor=previous_errors_extractor,
                previous_errors_plotter=previous_errors_plotter,
            )

        return AgentResult(
            plot_path=plot_res.plot_path,
            tool_used=plot_res.tool_used,
            code_summary=plot_res.code_summary,
            data_file_path=df_path,
        )
