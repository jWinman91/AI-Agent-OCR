import json, re
from collections.abc import Callable

from loguru import logger
from pydantic import ValidationError
from typing import List, Any, Dict, Awaitable

from src.utils.data_models import AgentResult

class OutputValidator:
    def __init__(self, agent_names: List[str], agent_output_examples: Dict[str, Any], agent_outputs: dict[str, Any]):
        self._agent_names = agent_names
        self._agent_output_examples = agent_output_examples
        self._agent_outputs = agent_outputs

    async def parse_json(self, text: str, expected_json: AgentResult,
                         run_agent_func: Callable[..., Awaitable[AgentResult]]) -> dict:
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.error(f"Initial JSON parsing failed: {e}, attempting to extract JSON code block.")
            json_match = re.search(r"```json\s*[\s\S]*?```", text, re.DOTALL)
            if not json_match or "None" in text:
                user_prompt = f"""Expected json format: {expected_json.model_dump()}.
                                  Corrupted JSON that you need to fix: {text}"""
                fixed_json = await run_agent_func("fix_json", user_prompt=user_prompt)
                if not fixed_json.error:
                    return json.loads(fixed_json.text)
                else:
                    logger.error(f"Could not parse JSON and fix_json failed with error: {fixed_json.error}")
                    raise(ValueError(f"Could not parse JSON and fix_json failed with error: {fixed_json.error}"))

            json_str = json_match.group(0).replace("```json", "").replace("```", "").strip()
            return json.loads(json_str)

    async def validate_output(self, agent_response: str, agent_name: str,
                              run_agent_func: Callable[..., Awaitable[AgentResult]]) -> AgentResult:
        if agent_name not in self._agent_names:
            raise ValueError(f"Invalid agent {agent_name}. Agent must be one of {self._agent_names}")

        parsed_output = await self.parse_json(agent_response, self._agent_output_examples[agent_name])
        try:
            return self._agent_outputs[agent_name](**parsed_output)
        except ValidationError as e:
            logger.error(f"Validation error for {agent_name}: {e}, attempting to fix output: {parsed_output}")
            user_prompt = f"""Expected output format: {self._agent_output_examples[agent_name].model_dump()}.
                              Corrupted output that you need to fix: {agent_response}"""
            fixed_output = await run_agent_func("fix_json", user_prompt=user_prompt)
            if not fixed_output.error:
                return self._agent_outputs[agent_name](**json.loads(fixed_output.text))
            else:
                raise(ValueError(f"Could not parse output and fix_json failed with error: {fixed_output.error}"))


