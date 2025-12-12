import os, datetime, time, importlib

import pandas
import pandas as pd
from pathlib import Path

from loguru import logger
from pydantic_ai import Agent, BinaryContent
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
from PIL import Image
from typing import List, Dict, Any, Literal

from src.utils.data_models import (OrchestratorOutput,
                                   PlotterResult,
                                   OrchestratorResult,
                                   ExtractorResult,
                                   AgentResult,
                                   ExtractorPrompt,
                                   PlotterResultOutput)
from src.output_validator import OutputValidator

import logging
logging.basicConfig(level=logging.INFO)


class AgentCreator:
    def __init__(self,
                 config: Dict[str, Dict[str, Any]],
                 data_dir: str = "data",
                 output_dir: str = "plots") -> None:
        self.config = config

        self.agent_outputs = {}
        self._agent_names = []
        for agent_name, agent_config in config:
            self._agent_names.append(agent_name)
            exec(f"self.agent_outputs['{agent_name}'] = {"".join(map(
                lambda x: x.capitalize(), agent_name.split("_")))}Result")
            exec(f"self.{agent_name} = self._create_agent(name)")

        extractor_example = ExtractorResult(
            data={"": "corrected dictionary of extracted data"},
            error="null or error message"
        )
        extractor_prompt_example = ExtractorPrompt(
            user_prompt="string",
            json_details={"column1": "string", "column2": "integer"}
        )
        orchestrator_example = OrchestratorResult(
            extractor_prompt=extractor_prompt_example,
            plot_prompt="str",
            error="null or error message"
        )
        plotter_output_example = PlotterResultOutput(
            plot_path="string",
            tool_used="string",
            code_summary="string"
        )
        plotter_example = PlotterResult(
            body=plotter_output_example,
            error="null or error message"
        )

        self._agent_output_examples = {
            "extractor": extractor_example,
            "orchestrator": orchestrator_example,
            "plotter": plotter_example
        }

        self._output_validator = OutputValidator(
            self._agent_names,
            self._agent_output_examples,
            self.agent_outputs
        )

        self._data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self._output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)


    @staticmethod
    def fill_system_prompt(cfg: Dict[str, Any],args: Dict[str, Any] | None = None) -> Dict[str, Any]:
        cfg = cfg.copy()
        if args:
            cfg["system_prompt"] = cfg["system_prompt"].format(**args)
            return cfg
        return cfg

    @staticmethod
    def error_handling_prompt(agent_name: Literal["extractor", "orchestrator", "plotter"],
                              error_message: str,
                              user_prompt: str,
                              previous_errors: Dict[str, Any] | None = None) -> Dict[str, Any]:
        if previous_errors is None:
            previous_errors = {user_prompt: error_message}
        else:
            previous_errors[user_prompt] += error_message

        if len(previous_errors) > 3:
            raise ValueError(f"Too many errors in {agent_name}: {previous_errors}. Stopping execution.")
        return previous_errors

    @staticmethod
    def get_image_timestamp(image_url: str) -> datetime.datetime:
        """
        Load in data and get the timestamp from the image metadata.
        :param image_url:
        :return:
        """
        if not image_url.endswith(".pdf"):
            exif = Image.open(image_url)._getexif()
            if exif is not None and 36867 in list(exif.keys()):
                return datetime.datetime.strptime(exif[36867], "%Y:%m:%d %H:%M:%S")

        return datetime.datetime.strptime(datetime.datetime.now().strftime("%Y:%m:%d %H:%M:%S"),
                                          "%Y:%m:%d %H:%M:%S")

    @staticmethod
    def build_error_messages(previous_errors: Dict[str, Any] | None) -> str:
        if not previous_errors:
            return ""
        return "\n".join([f"- User Prompt: {k}\nError: {v}" for k, v in previous_errors.items()])

    def _create_agent(self, agent_name: str, system_prompt_args: Dict[str, Any] | None = None) -> Agent:
        if agent_name not in self._agent_names:
            raise ValueError(f"Invalid agent {agent_name}. Agent must be one of {self._agent_names}")

        # set agent_cls by default to Agent
        # (we might overwrite it later with a custom agent)
        agent_cls = Agent

        # process configuration
        cfg = self.fill_system_prompt(self.config.get(agent_name, {}), system_prompt_args)
        # set output type if given in config
        if cfg.pop("output_type", None):
            cfg["output_type"] = self.agent_outputs[agent_name]
        # initiate model with model provider if ollama should be used
        if cfg.pop("model_provider", None) == "ollama":
            cfg["model"] = OpenAIChatModel(
                model_name=cfg.pop("model", "gpt-oss:20b"),
                provider=OllamaProvider(base_url=cfg.pop("base_url", "http://localhost:11434/v1"))
            )
        # set MCP server if necessary
        if "mcp_server" in cfg:
            mcp_server = MCPServerStdio(**cfg["mcp_server"])
            cfg["toolsets"] = [mcp_server]
            cfg.pop("mcp_server")
        # set custom model
        if custom := cfg.get("custom", {}):
            module = importlib.import_module(custom.get("module_name"))
            agent_cls = getattr(module, custom.get("class_name"))

        return agent_cls(**cfg)

    async def run_agent(self, agent_name: str, **kwargs) -> AgentResult:
        agent = getattr(self, agent_name)
        if agent_name == "extractor":
            agent_input = [
                kwargs["user_prompt"],
                BinaryContent(data=Path(kwargs["image_url"]).read_bytes(), media_type="image/png")
            ]
        else:
            agent_input = kwargs["user_prompt"]

        t0 = time.time()
        agent_response = await agent.run(agent_input)
        logger.info(f"Time taken for {agent_name}: {time.time() - t0:.2f} seconds")
        print(agent_response.output)
        if type(agent_response.output) != str:
            return agent_response.output
        return await self._output_validator.validate_output(agent_response.output, agent_name, self.run_agent)

    async def run_orchestrator(self, user_prompt: str, system_prompt_args: dict | None = None) -> OrchestratorResult:
        # Orchestrator decides how to extract and plot based on user prompt and previous errors
        self.orchestrator = self._create_agent("orchestrator", system_prompt_args)
        orchestrator_res = await self.run_agent("orchestrator", user_prompt=user_prompt)
        if orchestrator_res.error and orchestrator_res.error != "":
            logger.error(f"Error in orchestrator: {orchestrator_res.error}")
            raise ValueError(f"Error in orchestrator: {orchestrator_res.error}")

        logger.info(f"orchestrator: {orchestrator_res}")

        return orchestrator_res

    async def run_extractor(self, image_urls: List[str], previous_errors_extractor: dict,
                            orchestrator_res: OrchestratorResult) -> pandas.DataFrame | dict:
        # Extractor extracts data from each image based on orchestrator's instructions
        system_prompt_args_extractor = {"json_details": orchestrator_res.extractor_prompt.json_details}
        user_prompt_extractor = orchestrator_res.extractor_prompt.user_prompt
        self.extractor = self._create_agent("extractor", system_prompt_args_extractor)

        df = pd.DataFrame()
        timestamps = []
        for image_url in image_urls:
            extractor_res = await self.run_agent("extractor",
                                                 user_prompt=user_prompt_extractor,
                                                 image_url=image_url)

            logger.info(f"extractor for image {image_url}: " f"{extractor_res}")
            if extractor_res.error:
                logger.error(f"Error in extractor for image {image_url}: {extractor_res.error}")
                previous_errors_extractor = self.error_handling_prompt(
                    "extractor", extractor_res.error, user_prompt_extractor, previous_errors_extractor
                )
                return previous_errors_extractor
            timestamps.append(self.get_image_timestamp(image_url))

            df = pd.concat([df, pd.DataFrame([extractor_res.data])], ignore_index=True)

        df["timestamp"] = timestamps

    async def run_plotter(self, user_prompt: str, previous_errors_plotter: dict, system_prompt_args: dict) \
            -> PlotterResult | dict:
        self.plotter = self._create_agent("plotter", system_prompt_args)
        plot_res = await self.run_agent("plotter", user_prompt=user_prompt)

        if plot_res.error is not None:
            logger.error(f"Error in plotter: {plot_res.error}")
            return self.error_handling_prompt(
                "plotter", plot_res.error, user_prompt, previous_errors_plotter
            )
        return plot_res

    async def run_logic(self, user_prompt: str, image_urls: List[str],
                        previous_errors_extractor: Dict[str, Any] | None = None,
                        previous_errors_plotter: Dict[str, Any] | None = None) -> OrchestratorOutput:
        # Build error messages if any of previous errors exist
        system_prompt_args_orchestrator = {
            "previous_errors_extractor": self.build_error_messages(previous_errors_extractor),
            "previous_errors_plotter": self.build_error_messages(previous_errors_plotter)
        }
        orchestrator_res = await self.run_orchestrator(user_prompt, system_prompt_args_orchestrator)
        extractor_res = await self.run_extractor(image_urls, previous_errors_extractor, orchestrator_res)

        if type(extractor_res) is dict:
            return await self.run_logic(user_prompt, image_urls, extractor_res, previous_errors_plotter)
        else:
            df = extractor_res
            df_path = f"{self._data_dir}/extracted_data.csv"
            df.to_csv(df_path, index=False)

        # Plotter generates plot based on orchestrator's instructions and extracted data
        system_prompt_args_plotter = {"data_path": df_path, "output_dir": self._output_dir}
        plot_res = await self.run_plotter(orchestrator_res.plot_prompt, previous_errors_plotter,
                                          system_prompt_args_plotter)
        if type(plot_res) is not ExtractorResult:
            return await self.run_logic(user_prompt, image_urls, previous_errors_extractor, plot_res)

        return OrchestratorOutput(
            plot_path=plot_res.body.plot_path,
            tool_used=plot_res.body.tool_used,
            code_summary=plot_res.body.code_summary,
            df_path=df_path
        )
