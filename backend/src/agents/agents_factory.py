import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict

from loguru import logger
from pydantic_ai import Agent, BinaryContent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
from src.agents.output_validator import OutputValidator
from src.utils.data_models import AgentResponse, AgentResult, ConfigModelBuild

logging.basicConfig(level=logging.INFO)


class AgentsFactory:
    def __init__(
        self,
        agent_names: list[str],
        output_validator_config: ConfigModelBuild,
    ) -> None:
        """Initializes the AgentsFactory class.

        :param output_validator: An instance of the OutputValidator class to validate
        agent outputs.
        """
        # Initialize output validator
        self.AGENT_NAMES = agent_names
        self.OUTPUT_VALIDATOR = OutputValidator(
            create_agent_func=self.create_agent,
            run_agent_func=self.run_agent,
            config=output_validator_config,
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

        agent = Agent(**cfg)
        agent.name = agent_name
        return agent

    async def run_agent(
        self,
        agent: Agent,
        output_type_example: AgentResponse,
        user_prompt: str,
        image_url: str | Path | None = None,
        tool_execution_func: Callable[[AgentResponse], Any] | None = None,
    ) -> AgentResult:
        """
        Runs an agent with the given user prompt and optional image.

        :param agent: The agent to run.
        :param output_type_example: Example of the expected output type for validation.
        :param user_prompt: The user prompt to provide to the agent.
        :param image_url: Optional path to an image file to provide to the agent.
        param tool_execution_func: Optional function to execute tools
        :return: The agent's response, validated explicitly if it's a string output.
        """
        t0 = time.time()
        if image_url is not None:
            image_path = Path(image_url)
            with image_path.open("rb") as f:
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
            structured_output = await self.OUTPUT_VALIDATOR.validate_output(
                response.output, agent.name if agent.name else "", output_type_example
            )
        else:
            structured_output = response.output

        # execute tools if tool execution function is given
        # NOTE: the tool execution function will always return a result,
        # error handling should be done inside the tool execution function
        if tool_execution_func is not None:
            return tool_execution_func(structured_output)
        else:
            return structured_output
