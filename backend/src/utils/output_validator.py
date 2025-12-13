import json
import re
from collections.abc import Callable
from typing import Any, Awaitable

from loguru import logger
from pydantic import ValidationError
from pydantic_ai import Agent
from src.utils.data_models import AgentResponse


class OutputValidator:
    PROMPT = (
        "Expected json format: {expected_json}."
        "Corrupted JSON that you need to fix: {text}"
    )
    AGENT_NAME = "fix_json"

    def __init__(
        self,
        create_agent_func: Callable[..., Agent],
        run_agent_func: Callable[..., Awaitable[AgentResponse]],
    ) -> None:
        """
        Initializes the OutputValidator class.

        :param create_agent_func: Function to create an agent.
        :param run_agent_func: Function to run an agent asynchronously.
        """

        self._create_agent_func = create_agent_func
        self._run_agent_func = run_agent_func

    async def parse_json(
        self,
        text: str,
        expected_json: AgentResponse,
    ) -> dict[str, Any]:
        """
        Parses a JSON string, attempting to fix it if it's malformed.

        :param text: The JSON string to parse.
        :param expected_json: The expected JSON structure as a Pydantic model.
        :return: A dictionary representing the parsed JSON.
        """
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.error(
                f"Initial JSON parsing failed: {e},"
                f"attempting to extract JSON code block."
            )
            json_match = re.search(r"```json\s*[\s\S]*?```", text, re.DOTALL)

            if json_match or "None" not in text:
                json_str = (
                    json_match.group(0)
                    .replace("```json", "")
                    .replace("```", "")
                    .strip()
                )
                return json.loads(json_str)

            user_prompt = self.PROMPT.format(
                expected_json=expected_json.model_dump(),
                text=text,
            )
            fixed_json = await self._run_agent_func(
                agent=self.AGENT_NAME,
                user_prompt=user_prompt,
                output_type_example=AgentResponse(
                    fixed_json="str",
                    error_message="null or error message",
                ),
            )
            if not fixed_json.error_message:
                return json.loads(fixed_json.fixed_json)
            else:
                logger.error(
                    f"Could not parse JSON."
                    f"fix_json failed with error: {fixed_json.error_message}"
                )
                raise (
                    ValueError(
                        f"Could not parse JSON."
                        f"fix_json failed with error: {fixed_json.error_message}"
                    )
                )

    async def validate_output(
        self, agent_response: str, agent_name: str, expected_json: AgentResponse
    ) -> AgentResponse:
        """
        Validates the output of an agent against the expected Pydantic model.

        :param agent_response: The output string from the agent.
        :param agent_name: The name of the agent.
        :param expected_json: The expected JSON structure as a Pydantic model.
        :return: The validated agent response as a Pydantic model.
        """
        parsed_output = await self.parse_json(
            text=agent_response, expected_json=expected_json
        )
        try:
            return AgentResponse(**parsed_output)
        except ValidationError as e:
            logger.error(
                f"Validation error for {agent_name}: {e},"
                f"attempting to fix output: {parsed_output}"
            )
            user_prompt = self.PROMPT.format(
                expected_json=expected_json.model_dump(),
                text=agent_response,
            )
            fixed_json = await self._run_agent_func(
                agent=self.AGENT_NAME,
                user_prompt=user_prompt,
                output_type_example=AgentResponse(
                    fixed_json="str",
                    error_message="null or error message",
                ),
            )
            if not fixed_json.error_message:
                return AgentResponse(**json.loads(fixed_json.fixed_json))
            else:
                raise (
                    ValueError(
                        f"Could not parse output."
                        f"fix_json failed with error: {fixed_json.error_message}"
                    )
                )
