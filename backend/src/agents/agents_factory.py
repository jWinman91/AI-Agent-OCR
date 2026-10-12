import copy
import logging
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Dict, TypeVar, cast, overload

from loguru import logger
from PIL import Image
from pydantic_ai import Agent, BinaryContent
from pydantic_ai.exceptions import (
    ModelAPIError,
    UnexpectedModelBehavior,
    UsageLimitExceeded,
)
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
from src.agents.output_validator import OutputValidator
from src.utils.data_models import AgentResponse, ConfigModelBuild

ResponseT = TypeVar("ResponseT", bound=AgentResponse)
ResultT = TypeVar("ResultT")

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

        agent_config = config.model_dump(
            exclude={
                "model",
                "model_provider",
                "base_url",
                "output_type",
                "model_settings",
            }
        )
        agent_config["system_prompt"] = self._fill_system_prompt(
            config.system_prompt, system_prompt_args
        )

        if config.model_settings:
            agent_config["model_settings"] = config.model_settings

        if config.output_type:
            agent_config["output_type"] = output_type

        if config.model_provider == "ollama":
            agent_config["model"] = OpenAIChatModel(
                model_name=config.model,
                provider=OllamaProvider(base_url=config.base_url),
            )
        else:
            agent_config["model"] = f"{config.model_provider}:{config.model}"

        agent = Agent(**agent_config)
        agent.name = agent_name
        return agent

    @overload
    async def run_agent(
        self,
        agent: Agent,
        output_type_example: ResponseT,
        user_prompt: str,
        image_url: str | Path | None = None,
        text_file_url: str | Path | None = None,
        tool_execution_func: None = None,
    ) -> ResponseT: ...

    @overload
    async def run_agent(
        self,
        agent: Agent,
        output_type_example: ResponseT,
        user_prompt: str,
        image_url: str | Path | None = None,
        text_file_url: str | Path | None = None,
        *,
        tool_execution_func: Callable[[ResponseT], ResultT],
        error_result_factory: Callable[[str], ResultT] | None = None,
    ) -> ResultT: ...

    async def run_agent(
        self,
        agent: Agent,
        output_type_example: ResponseT,
        user_prompt: str,
        image_url: str | Path | None = None,
        text_file_url: str | Path | None = None,
        tool_execution_func: Callable[[ResponseT], ResultT] | None = None,
        error_result_factory: Callable[[str], ResultT] | None = None,
    ) -> ResponseT | ResultT:
        """
        Runs an agent with the given user prompt and optional image.

        :param agent: The agent to run.
        :param output_type_example: Example of the expected output type for validation.
        :param user_prompt: The user prompt to provide to the agent.
        :param image_url: Optional path to an image file to provide to the agent.
        :param text_file_url: Optional path to a text file to provide to the agent.
        :param tool_execution_func: Optional function to execute tools
        :return: The agent's response, validated explicitly if it's a string output.
        """

        def failure(message: str) -> ResponseT | ResultT:
            logger.error("Agent {} failed: {}", agent.name, message)
            if tool_execution_func is not None:
                if error_result_factory is None:
                    raise RuntimeError(message)
                return error_result_factory(message)
            failed_response = copy.deepcopy(output_type_example)
            failed_response.error_message = message
            return failed_response

        t0 = time.time()
        # check if image_url and text_file_url are not empty
        if text_file_url:
            with Path(text_file_url).open("r") as f:
                file_content = f.read()

            user_prompt = (
                f"{user_prompt}\n\n"
                f"<EXTRACTED PDF TEXT>:\n"
                f"{file_content}\n"
                f"</EXTRACTED PDF TEXT>"
            )

        messages: list[str | BinaryContent]
        if image_url:
            image_path = Path(image_url)
            try:
                image_content = image_path.read_bytes()
                with Image.open(BytesIO(image_content)) as image:
                    media_type = {"PNG": "image/png", "JPEG": "image/jpeg"}.get(
                        image.format or ""
                    )
                    image.verify()
                if media_type is None:
                    raise ValueError("Only PNG and JPEG images are supported.")
            except (OSError, ValueError) as error:
                return failure(f"Cannot read image input: {error}")
            binary_content = BinaryContent(
                data=image_content,
                media_type=media_type,
            )
            messages = [user_prompt, binary_content]
        else:
            messages = [user_prompt]

        try:
            response = await agent.run(messages)
        except (ModelAPIError, UnexpectedModelBehavior, UsageLimitExceeded) as error:
            logger.exception("Model request failed for {}", agent.name)
            return failure(f"Model request failed: {error}")

        image_log = f" with image {image_url}" if image_url is not None else ""

        logger.info(
            f"{agent.name} response in {time.time() - t0:.2f} sec"
            f"{image_log}: {response}"
        )

        try:
            if type(response.output) is str:
                structured_output = cast(
                    ResponseT,
                    await self.OUTPUT_VALIDATOR.validate_output(
                        response.output,
                        agent.name if agent.name else "",
                        output_type_example,
                    ),
                )
            else:
                structured_output = cast(
                    ResponseT, type(output_type_example).model_validate(response.output)
                )
        except (ValueError, TypeError) as error:
            return failure(f"Error validating output: {error}")

        if structured_output.error_message:
            return failure(structured_output.error_message)

        # execute tools if tool execution function is given
        # NOTE: the tool execution function will always return a result,
        # error handling should be done inside the tool execution function
        if tool_execution_func is not None:
            return tool_execution_func(structured_output)
        else:
            return structured_output
