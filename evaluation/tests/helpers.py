"""Shared fakes for the evaluation tests: a scripted judge model and a fake
backend client, so no backend or API key is needed."""

import json
from pathlib import Path
from typing import Any

from fastapi import UploadFile
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel


def criterion(score: int) -> dict[str, Any]:
    return {"explanation": f"scored {score}", "evidence": ["seen"], "score": score}


class ScriptedJudgeModel:
    """FunctionModel wrapper answering each judge call with fixed scores and
    recording every request it received."""

    def __init__(
        self,
        code_score: int = 4,
        visualisation_score: int = 3,
        fail: bool = False,
        issues: list[dict[str, Any]] | None = None,
    ) -> None:
        self.code_score = code_score
        self.visualisation_score = visualisation_score
        self.fail = fail
        self.issues = (
            issues
            if issues is not None
            else [{"line": 1, "kind": "wrong_result", "reason": "Plots nothing."}]
        )
        self.requests: list[list[ModelMessage]] = []
        self.model = FunctionModel(self._respond)

    def _respond(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        self.requests.append(messages)
        if self.fail:
            raise RuntimeError("judge unavailable")
        if not info.output_tools:
            return ModelResponse(parts=[TextPart("OK")])
        tool = info.output_tools[0]
        properties = tool.parameters_json_schema["properties"]
        if "ok" in properties:  # LLMJudge.check_model
            args: dict[str, Any] = {"ok": True}
        elif "issues" in properties:  # LLMJudge.judge_bugs
            args = {"issues": self.issues}
        elif "code_simplicity" in properties:
            args = {name: criterion(self.code_score) for name in properties}
        else:
            args = {name: criterion(self.visualisation_score) for name in properties}
        return ModelResponse(
            parts=[ToolCallPart(tool_name=tool.name, args=json.dumps(args))]
        )


class FakeAgentClient:
    """Stands in for AgentClient, returning a canned /run_single_agent result."""

    def __init__(
        self,
        result: dict[str, Any],
        plot: bytes | None = b"PNG",
        data: bytes | None = b"a,b\n1,2\n",
    ) -> None:
        self.result = result
        self.plot = plot
        self.data = data
        self.calls: list[str] = []
        self.uploaded: list[str] = []
        self.config: dict[str, Any] = {
            "model": "gemma4:latest",
            "model_provider": "ollama",
            "domain_knowledge_prompt": "",
        }

    def update_agents(self) -> None:
        self.calls.append("update_agents")

    def get_config(self, agent_name: str, full_prompt: bool = True) -> dict[str, Any]:
        self.calls.append(f"get_config:{agent_name}:{full_prompt}")
        config = dict(self.config)
        if full_prompt:
            config["domain_knowledge_prompt"] = (
                f"FULL[{config['domain_knowledge_prompt']}]"
            )
        return config

    def upload_config(self, agent_name: str, config: dict[str, Any]) -> None:
        self.calls.append(f"upload_config:{agent_name}")
        self.config = dict(config)

    def upload_data_file(self, file: UploadFile) -> None:
        self.uploaded.append(file.filename or "")
        file.file.read()

    def run_single_agent(self, agent_name: str, user_prompt: str) -> dict[str, Any]:
        self.calls.append(f"run:{agent_name}")
        return self.result

    def download_plot(self, plot_path: Path) -> bytes:
        if self.plot is None:
            raise RuntimeError("GET failed with reason: Not Found")
        return self.plot

    def download_data(self, df_path: Path) -> bytes:
        if self.data is None:
            raise RuntimeError("GET failed with reason: Not Found")
        return self.data
