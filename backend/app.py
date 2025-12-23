import os
from typing import List, Literal

import uvicorn
from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from loguru import logger
from src.agents_runner import AgentRunner
from src.config_builder import ConfigBuilder
from src.file_manager import FileManager
from src.preprocessor import Preprocessor
from src.utils.data_models import (
    AgentResult,
    ConfigModelUser,
)


class AiAgentOcrApp:
    def __init__(
        self,
        ip: str = "127.0.0.1",
        port: int = 8000,
        data_dir: str = "data",
        output_dir: str = "plots",
        image_dir: str = "images",
        agent_config_path: str = "configs/agents.yaml",
    ) -> None:
        """
        Initializes the PlotAgentApp with the given parameters.

        :param ip: IP address to bind the server
        :param port: Port to bind the server
        :param data_dir: Directory to store data files
        :param output_dir: Directory to store output plots
        :param agent_config_path: Path to the agent configuration file
        """
        self.IP = ip
        self.PORT = port

        # MUTUABLE STATES
        self._image_urls = []
        self._data_file_path = ""

        self.CONFIG_BUILDER = ConfigBuilder(agent_config_path)
        self.FILE_MANAGER = FileManager(
            data_dir=data_dir, plot_dir=output_dir, image_dir=image_dir
        )
        self.FILE_MANAGER.clean_up()
        self.PREPROCESSOR = Preprocessor()
        self.ai_agent_runner = AgentRunner(
            config=self.CONFIG_BUILDER.build_config(),
            data_dir=data_dir,
            plots_dir=output_dir,
        )

        self.app = FastAPI()
        self._register_routes()

    def _register_routes(self) -> None:
        @self.app.post("/upload_config/")
        async def upload_config(
            config_name: Literal["extractor", "plotter"], config: ConfigModelUser
        ) -> bool | None:
            """
            Uploads a new configuration to the database.

            :param config: Configuration model to upload
            :return: True if successful
            """
            return self.CONFIG_BUILDER.update_config_in_db(config_name, config)

        @self.app.get("/get_config/")
        async def get_config(
            config_name: Literal["extractor", "plotter"],
        ) -> ConfigModelUser:
            """
            Retrieves a configuration from the database.

            :param name: Name of the configuration to retrieve
            :return: Configuration model
            """
            config_model_db = self.CONFIG_BUILDER.load_config_from_db(
                config_name=config_name
            )
            return ConfigModelUser(**config_model_db.model_dump())

        @self.app.get("/get_all_configs")
        async def get_all_configs() -> dict[str, ConfigModelUser]:
            """
            Retrieves all configurations from the database.

            :return: List of configuration models
            """
            configs = self.CONFIG_BUILDER.load_all_configs_from_db()
            return {
                name: ConfigModelUser(**config.model_dump())
                for name, config in configs.items()
                if name in ["extractor", "plotter"]
            }

        @self.app.post("/reset_config/")
        async def reset_config(
            config_name: Literal["extractor", "plotter"],
        ) -> bool | None:
            """
            Resets a configuration in the database to its default state.

            :param config_name: Name of the configuration to reset
            :return: True if successful
            """
            return self.CONFIG_BUILDER.reset_config_db(config_name)

        @self.app.get("/get_mcp_servers/")
        async def get_mcp_servers() -> dict[str, List[str]]:
            """
            Retrieves the list of MCP servers in the src/server directory.

            :return: Dictionary with the list of MCP servers
            """
            try:
                server_files = os.listdir("src/server")
                mcp_servers = [
                    f.split("/")[-1] for f in server_files if f.endswith(".py")
                ]
                return {"mcp_servers": mcp_servers}
            except Exception as e:
                logger.error(f"Error retrieving MCP servers: {e}")
                raise HTTPException(
                    status_code=500, detail="Could not retrieve MCP servers"
                )

        @self.app.post("/set_openai_api_key/")
        async def set_openai_api_key(api_key: str) -> bool:
            """
            Sets the OpenAI API key in the configuration.

            :param api_key: The OpenAI API key
            :return: True if successful
            """
            # set api key in environment variable
            os.environ["OPENAI_API_KEY"] = api_key
            return True

        @self.app.post("/update_agents/")
        async def update_agents() -> bool:
            """
            Recreates the agents with the current configuration.
            :return: True if successful
            """
            self._ai_agent_runner = AgentRunner(
                config=self.CONFIG_BUILDER.build_config(),
                data_dir=self.FILE_MANAGER.get_data_dir(),
                plots_dir=self.FILE_MANAGER.get_plot_dir(),
            )
            return True

        @self.app.post("/run_single_agent")
        async def run_single_agent(
            user_prompt: str = Form(...),
            agent_name: Literal["extractor", "plotter"] = Form(...),
        ) -> AgentResult:
            if agent_name == "extractor":
                # Get image URLs from the file manager
                data_dir = self.FILE_MANAGER.get_data_dir()
                if len(self._image_urls) == 0:
                    raise ValueError("Extractor agent requires images to process!")

                # Run the extractor agent and catch ValueErrors from the LLMs
                try:
                    df, _ = self.ai_agent_runner.run_extractor(
                        user_prompt=user_prompt,
                        image_urls=self._image_urls,
                        previous_errors_extractor={},
                        previous_errors_plotter={},
                    )
                except ValueError as e:
                    logger.error(f"Error in extractor: {e}")
                    raise HTTPException(
                        status_code=422,
                        detail=f"Extractor agent failed to process your input: {e}",
                    )

                df_path = f"{data_dir}/extracted_data.csv"
                df.to_csv(df_path, index=False)
                return AgentResult(
                    data_file_path=df_path,
                    plot_path="",
                    code_summary="",
                    tool_used="",
                )
            elif agent_name == "plotter":
                # Get data file path from the file manager
                output_dir = self.FILE_MANAGER.get_plot_dir()

                # Run the plotter agent and catch ValueErrors from the LLMs
                try:
                    plot_res = await self.ai_agent_runner.run_plotter(
                        user_prompt=user_prompt,
                        system_prompt_args={
                            "data_file_path": self._data_file_path,
                            "output_dir": output_dir,
                        },
                    )
                    if plot_res.error_message and plot_res.error_message != "":
                        logger.error(f"Error in plotter: {plot_res.error_message}")
                        raise ValueError(plot_res.error_message)

                except ValueError as e:
                    logger.error(f"Error in plotter: {e}")
                    raise HTTPException(
                        status_code=422,
                        detail=f"Plotter agent failed to process your input: {e}",
                    )

                # Reset data file path after use
                self._data_file_path = ""

                return AgentResult(
                    data_file_path=plot_res.df_file_path,
                    plot_path=plot_res.plot_path,
                    code_summary=plot_res.code_summary,
                    tool_used=plot_res.tool_used,
                )

            else:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Invalid agent {agent_name}."
                        f"Agent must be eihter 'extractor' or 'plotter'",
                    ),
                )

        @self.app.post("/run_agents")
        async def run_ai_ocr_agents(user_prompt: str = Form(...)) -> AgentResult:
            """
            Endpoint to run the agents.
            :param user_prompt: User prompt to process
            :return: Dictionary with the plot path
            """
            try:
                agent_result = await self.ai_agent_runner.run_ai_ocr_agents(
                    user_prompt=user_prompt, image_urls=self._image_urls
                )

                # Reset image URLs after use
                self._image_urls = []
                return agent_result

            except ValueError as e:
                logger.error(f"Error in run_agents: {e}")
                raise HTTPException(
                    status_code=422,
                    detail=f"Your input could not be processed by the agents: {e}",
                )

        @self.app.post("/uploadimages/")
        async def upload_images(
            files: List[UploadFile],
            resize: Literal["original", "medium", "small"] = Form(default="medium"),
        ) -> bool:
            """
            Endpoint to upload images.
            :param files: List of image files to upload
            :return: True if successful
            """
            if not all(
                file.filename.endswith((".png", ".jpg", ".jpeg")) for file in files
            ):
                raise HTTPException(
                    status_code=400,
                    detail="Only image files with extensions png, jpg, jpeg are allowed",
                )

            if resize == "medium":
                resize_files = []
                for file in files:
                    file = self.PREPROCESSOR.resize_image(
                        file,
                        max_size=800,
                    )
                    resize_files.append(file)
            elif resize == "small":
                resize_files = []
                for file in files:
                    file = self.PREPROCESSOR.resize_image(
                        file,
                        max_size=400,
                    )
                    resize_files.append(file)
            else:
                resize_files = files
            self._image_urls = self.FILE_MANAGER.upload_images(resize_files)
            logger.info(f"Uploaded {len(self._image_urls)} images.")
            return True

        @self.app.post("/uploaddatafile")
        async def upload_datafile(file: UploadFile = File(...)) -> bool:
            if not file.filename.endswith((".csv", ".xlsx", ".xls")):
                raise HTTPException(
                    status_code=422, detail="Only CSV or Excel files are allowed"
                )

            self._data_file_path = self.FILE_MANAGER.upload_data_file(file)
            logger.info(f"Uploaded data file: {self._data_file_path}")
            return True

        @self.app.get("/plot_path/{plot_path:path}/download_plot")
        async def download_plot(plot_path: str) -> FileResponse:
            """
            Endpoint to download a file.
            :param plot_path: Name of the file to download
            :return: FileResponse with the requested file
            """
            if os.path.exists(plot_path):
                return FileResponse(
                    plot_path,
                    media_type="image/png",
                    filename=os.path.basename(plot_path),
                )
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
                return FileResponse(
                    df_path,
                    media_type="csv",
                    filename=os.path.basename(df_path),
                )
            else:
                raise FileNotFoundError(f"File {df_path} not found")

    def run(self) -> None:
        """
        Run the api
        :return: None
        """
        uvicorn.run(self.app, host=self.IP, port=self.PORT)


if __name__ == "__main__":
    app = AiAgentOcrApp()
    app.run()
