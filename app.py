import uvicorn, os, yaml, subprocess
import pandas as pd

from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from typing import List, Literal
from loguru import logger

from src.agents_creator import AgentCreator
from src.config_builder import ConfigBuilder
from src.utils.data_models import OrchestratorOutput, ConfigModel, AgentResult


class AiAgentOcrApp:
    def __init__(self, ip: str = "127.0.0.1", port: int = 8000,
                 data_dir: str = "data", output_dir: str = "plots", image_dir: str = "images",
                 agent_config_path: str = "configs/agents.yaml"):
        """
        Initializes the PlotAgentApp with the given parameters.

        :param ip:
        :param port:
        :param output_dir:
        :param agent_config_path:
        """
        self._ip = ip
        self._port = port
        self._data_dir = data_dir
        self._output_dir = output_dir
        self._data_file_path = ""

        self._config_builder = ConfigBuilder(agent_config_path)
        self._ai_ocr_agents = AgentCreator(self._config_builder.build_config(),
                                           data_dir=data_dir, output_dir=output_dir)

        os.makedirs(image_dir, exist_ok=True)
        self._image_urls = []

        self.app = FastAPI()
        self._register_routes()

    @staticmethod
    def load_yaml(config_path: str) -> dict:
        """
        Loads in a system prompt (and additional parameters) for the LLM and returns them as a tuple.
        :param config_path: path to yaml file
        :return: Tuple containing the system prompt and additional parameters
        """
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def _register_routes(self):
        @self.app.post("/upload_config/")
        async def upload_config(config: ConfigModel) -> bool | None:
            return self._config_builder.write_config_to_db(config.name, config.config)

        @self.app.get("/get_config/")
        async def get_config(name: Literal["extractor", "plotter"]) -> ConfigModel:
            return ConfigModel(name=name, config=self._config_builder.load_config_from_db(name))

        @self.app.get("/get_all_configs")
        async def get_all_configs() -> List[ConfigModel]:
            configs = self._config_builder.load_all_configs_from_db()
            return [ConfigModel(name=name, config=config) for name, config in configs.items()]

        @self.app.get("/get_custom_all_agents")
        async def get_all_custom_agents() -> List[str]:
            return []

        @self.app.get("/update_agents/")
        async def update_agents() -> bool:
            """
            Recreates the agents with the current configuration.
            :return: True if successful
            """
            self._ai_ocr_agents = AgentCreator(self._config_builder.build_config())
            return True

        @self.app.post("/uploadimages/")
        async def upload_images(files: List[UploadFile]) -> bool:
            subprocess.call(f"rm {self._output_dir}/*", shell=True)
            subprocess.call(f"rm {self._data_dir}/*", shell=True)
            self._image_urls = []
            for file in files:
                file_path = os.path.join("images", file.filename)
                with open(file_path, "wb") as f:
                    content = file.file.read()
                    f.write(content)
                logger.info(f"File saved at: {file_path}")
                self._image_urls.append(file_path)
            return True

        @self.app.post("/uploaddatafile")
        async def upload_datafile(file: UploadFile) -> bool:
            if not file.filename.endswith((".csv", ".xlsx", ".xls")):
                raise HTTPException(status_code=400, detail="Only CSV or Excel files are allowed")

            subprocess.call(f"rm {self._data_dir}/*", shell=True)

            self._data_file_path = os.path.join(self._data_dir, file.filename)
            with open(self._data_file_path, "wb") as f:
                content = file.file.read()
                f.write(content)
            return True

        @self.app.post("/run_single_agent")
        async def run_single_agent(user_prompt: str = Form(...), agent_name: str = Form(...)) -> OrchestratorOutput:
            if agent_name == "extractor":
                if len(self._image_urls) == 0:
                    raise ValueError(f"No images to analyse!")

                previous_errors_extractor = {}
                system_prompt_args_orchestrator = {"previous_errors_extractor": ""}
                orchestrator_res = await self._ai_ocr_agents.run_orchestrator(user_prompt, system_prompt_args_orchestrator)
                extractor_res = await self._ai_ocr_agents.run_extractor(self._image_urls, previous_errors_extractor,
                                                                        orchestrator_res)

                if type(extractor_res) is dict:
                    logger.error(f"Error in extractor agent: {extractor_res}.")
                    raise HTTPException(status_code=500, detail=f"Extractor failed to process image: {extractor_res}.")
                else:
                    df = extractor_res
                    df_path = f"{self._data_dir}/extracted_data.csv"
                    df.to_csv(df_path, index=False)
                    return OrchestratorOutput(df_path=df_path, plot_path="", code_summary="", tool_used="")

            elif agent_name == "plotter":
                if self._data_file_path.endswith(".csv"):
                    df = pd.read_csv(self._data_file_path)
                else:
                    df = pd.read_excel(self._data_file_path)
                df_path = f"{self._output_dir}/extracted_data.csv"
                df.to_csv(df_path, index=False)

                system_prompt_args_plotter = {"data_path": df_path, "output_dir": self._output_dir}
                previous_errors_plotter = {}
                plot_res = await self._ai_ocr_agents.run_plotter(user_prompt, previous_errors_plotter,
                                                                 system_prompt_args_plotter)
                if type(plot_res) is not AgentResult:
                    logger.error(f"Error in plotter agent: {plot_res}.")
                    raise HTTPException(status_code=500, detail=f"Plotter failed to plot data: {plot_res}.")
                else:
                    return OrchestratorOutput(
                        plot_path=plot_res.body.plot_path,
                        tool_used=plot_res.body.tool_used,
                        code_summary=plot_res.body.code_summary,
                        df_path=df_path
                    )
            else:
                raise ValueError(f"Invalid agent {agent_name}. Agent must be one of {['extractor', 'plotter']}")

        @self.app.post("/run_agents")
        async def run_agents(user_prompt: str = Form(...)) -> OrchestratorOutput:
            """
            Endpoint to run the agents.
            :param user_prompt: User prompt to process
            :return: Dictionary with the plot path
            """
            try:
                return await self._ai_ocr_agents.run_logic(user_prompt, image_urls=self._image_urls)
            except ValueError as e:
                logger.error(f"Error in run_agents: {e}")
                raise HTTPException(status_code=422, detail=f"Your input could not be processed by the agents: {e}")

        @self.app.get("/plot_path/{plot_path:path}/download_plot")
        async def download_plot(plot_path: str) -> FileResponse:
            """
            Endpoint to download a file.
            :param plot_path: Name of the file to download
            :return: FileResponse with the requested file
            """
            if os.path.exists(plot_path):
                return FileResponse(plot_path, media_type='image/png', filename=os.path.basename(plot_path))
            else:
                raise FileNotFoundError(f"File {plot_path} not found")

        @self.app.get("/df_path/{df_path:path}/download_data")
        async def download_data(df_path: str) -> FileResponse:
            """
            Endpoint to download a file.
            :param df_path: Name of the file to download
            :return: FileResponse with the requested file
            """
            if os.path.exists(df_path):
                return FileResponse(df_path, media_type='csv', filename=os.path.basename(df_path))
            else:
                raise FileNotFoundError(f"File {df_path} not found")

    def run(self) -> None:
        """
        Run the api
        :return: None
        """
        uvicorn.run(self.app, host=self._ip, port=self._port)


if __name__ == "__main__":
    app = AiAgentOcrApp()
    app.run()