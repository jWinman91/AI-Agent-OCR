import json
import re
from collections.abc import Callable
from typing import Any, Awaitable, cast

from loguru import logger
from pydantic import ValidationError
from pydantic_ai import Agent
from src.utils.data_models import (
    AgentResponse,
    AnalyserResponse,
    ConfigModelBuild,
    DataDownloadResponse,
    DataExtractorResponse,
    FixJsonResponse,
    OrchestratorResponse,
)


class OutputValidator:
    PROMPT = (
        "Expected json format: {expected_json}."
        "Corrupted JSON that you need to fix: {text}"
        "Error message from previous parsing attempt: {error_message}"
    )
    AGENT: object | None = None

    def __init__(
        self,
        create_agent_func: Callable[..., Agent],
        run_agent_func: Callable[..., Awaitable[AgentResponse | str]],
        config: ConfigModelBuild,
    ) -> None:
        """
        Initializes the OutputValidator class.

        :param create_agent_func: Function to create an agent.
        :param run_agent_func: Function to run an agent asynchronously.
        """

        self.AGENT = create_agent_func(
            agent_name="fix_json",
            config=config,
            output_type=FixJsonResponse,
        )
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
            return self._json_object(text)
        except json.JSONDecodeError as e:
            logger.error(
                f"Initial JSON parsing failed: {e},"
                f"attempting to extract JSON code block."
            )
            # Fallback for models that prepend/append explanations around JSON.
            # Try to decode the first valid inline JSON object in the text.
            inline_json = self.extract_first_json_object(text)
            if inline_json is not None:
                return inline_json
            else:
                logger.error(
                    "Could not extract valid JSON, "
                    "attempting to fix output with FixJson agent now."
                )

            user_prompt = self.PROMPT.format(
                expected_json=expected_json.model_dump(),
                text=text,
                error_message=str(e),
            )
            fixed_json = await self._run_agent_func(
                agent=self.AGENT,
                user_prompt=user_prompt,
                output_type_example=self.try_output_models(
                    {
                        "fixed_json": "str",
                        "error_message": "null or error message",
                    }
                ),
            )
            if type(fixed_json) is not FixJsonResponse and type(fixed_json) is str:
                fixed_json = FixJsonResponse(
                    **await self.parse_json(
                        fixed_json,
                        FixJsonResponse(
                            fixed_json="text",
                            error_message=None,
                        ),
                    )
                )
            fixed_json = cast(FixJsonResponse, fixed_json)
            if not fixed_json.error_message:
                return self._json_object(fixed_json.fixed_json)
            else:
                logger.error(
                    f"Error in parsing JSON. "
                    f"FixJson agent failed with error: {fixed_json.error_message}",
                )
                raise (
                    ValueError(
                        f"Error in parsing JSON. "
                        f"FixJson agent failed with error: {fixed_json.error_message}"
                    )
                )

    @staticmethod
    def _json_object(text: str) -> dict[str, Any]:
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError("Agent output must be a JSON object.")
        return parsed

    @staticmethod
    def extract_first_json_object(text: str) -> dict[str, Any] | None:
        """
        Extract the first valid JSON object from free-form text.

        This handles outputs like: "Explanation... { ...valid json... }".
        """
        decoder = json.JSONDecoder()

        # Find the first JSON object start and treat it as authoritative.
        # If this object is broken, return None so the fixer can repair it
        # instead of accidentally decoding nested objects.
        start_index = text.find("{")
        if start_index == -1:
            return None

        try:
            parsed, end_index = decoder.raw_decode(text[start_index:])
        except json.JSONDecodeError:
            return None

        return parsed if isinstance(parsed, dict) else None

    async def validate_output(
        self,
        agent_response: str,
        agent_name: str,
        expected_json: AgentResponse | str,
    ) -> AgentResponse | str:
        """
        Validates the output of an agent against the expected Pydantic model.

        :param agent_response: The output string from the agent.
        :param agent_name: The name of the agent.
        :param expected_json: The expected JSON structure as a Pydantic model.
        :return: The validated agent response as a Pydantic model.
        """
        # Special handling for AnalyserResponse which contains code
        if type(expected_json) is AnalyserResponse:
            code = self.extract_tag("CODE", agent_response)
            summary = self.extract_tag("SUMMARY", agent_response)
            if not code or summary is None:
                raise ValueError("Analyser output requires CODE and SUMMARY sections.")
            return AnalyserResponse(
                python_code=code,
                code_summary=summary,
            )
        if type(expected_json) is str:
            return agent_response

        expected_json = cast(AgentResponse, expected_json)
        parsed_output = await self.parse_json(
            text=agent_response,
            expected_json=expected_json,
        )
        try:
            return type(expected_json).model_validate(parsed_output)
        except ValidationError as e:
            logger.error(
                f"Validation error for {agent_name}: {e},"
                f"attempting to fix output: {parsed_output}"
            )
            user_prompt = self.PROMPT.format(
                expected_json=expected_json.model_dump(),
                text=agent_response,
                error_message=str(e),
            )
            fixed_json = await self._run_agent_func(
                agent=self.AGENT,
                user_prompt=user_prompt,
                output_type_example=self.try_output_models(
                    {
                        "fixed_json": "str",
                        "error_message": "null or error message",
                    }
                ),
            )
            fixed_json = cast(FixJsonResponse, fixed_json)
            if not fixed_json.error_message:
                return type(expected_json).model_validate(
                    self._json_object(fixed_json.fixed_json)
                )
            else:
                raise (
                    ValueError(
                        f"Error in validating model."
                        f"fix_json failed with error: {fixed_json.error_message}"
                    )
                )

    @staticmethod
    def try_output_models(parsed_output: dict[str, Any]) -> AgentResponse:
        """
        Attempts to parse the output using different models in order of preference.
        """
        models: list[type[AgentResponse]] = [
            DataExtractorResponse,
            OrchestratorResponse,
            FixJsonResponse,
            DataDownloadResponse,
        ]

        for model in models:
            try:
                return model(**parsed_output)
            except ValidationError:
                continue

        raise ValueError(
            f"Could not parse output: no valid model matched from models: {models}."
        )

    @staticmethod
    def extract_tag(tag: str, text: str) -> str | None:
        """
        Extracts content from a specific tag in the text.

        :param tag: The tag to extract content from.
        :param text: The text to search within.
        :return: The content within the specified tag, or None if not found.
        """
        pattern = rf"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else None
