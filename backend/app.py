import argparse
import os
from pathlib import Path
from typing import List, Literal

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from loguru import logger
from src.agents.agents_runner import AgentRunner
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
        ip: str,
        port: int,
        data_dir: Path = Path("data"),
        output_dir: Path = Path("plots"),
        image_dir: Path = Path("images"),
        agent_config_path: Path = Path("configs/agents.yaml"),
    ) -> None:
        """
        Initializes the AiAgentOcrApp with the given parameters.

        :param ip: IP address to bind the server
        :param port: Port to bind the server
        :param data_dir: Directory to store data files
        :param output_dir: Directory to store output plots
        :param agent_config_path: Path to the agent configuration file
        """
        self.IP = ip
        self.PORT = port

        # MUTUABLE STATES
        self._image_urls: list[Path] = []
        self._data_file_path: Path | None = None

        self.CONFIG_BUILDER = ConfigBuilder(agent_config_path)
        self.FILE_MANAGER = FileManager(
            data_dir=data_dir,
            plot_dir=output_dir,
            image_dir=image_dir,
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
            config_name: Literal["data_extractor", "analyser"], config: ConfigModelUser
        ) -> bool | None:
            """
            Uploads a new configuration to the database.

            :param config: Configuration model to upload
            :return: True if successful
            """
            return self.CONFIG_BUILDER.update_config_in_db(config_name, config)

        @self.app.get("/get_config/")
        async def get_config(
            config_name: Literal["data_extractor", "analyser"],
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
                if name in ["data_extractor", "analyser"]
            }

        @self.app.post("/reset_config/")
        async def reset_config(
            config_name: Literal["data_extractor", "analyser"],
        ) -> bool | None:
            """
            Resets a configuration in the database to its default state.

            :param config_name: Name of the configuration to reset
            :return: True if successful
            """
            return self.CONFIG_BUILDER.reset_config_db(config_name)

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
            agent_name: Literal["data_extractor", "analyser"] = Form(...),
            previous_error: str | None = Form(default=None),
        ) -> AgentResult:
            """
            Endpoint to run a single agent.

            :param user_prompt: User prompt to process
            :param agent_name: Name of the agent to run
            :return: Dictionary with the plot path
            """
            if agent_name == "data_extractor":
                # Get image URLs from the file manager
                if len(self._image_urls) == 0:
                    raise ValueError("Data extractor agent requires images to process!")
                # Run the data extractor agent
                res = await self.ai_agent_runner.run_ai_ocr_agents(
                    user_prompt=user_prompt,
                    image_urls=self._image_urls,
                    data_file_path=None,
                    list_of_agents_to_run=["orchestrator", "data_extractor"],
                    previous_error=previous_error,
                )
                # reset image URLs after use
                self._image_urls = []
                return res

            elif agent_name == "analyser":
                # Run the analyser agent
                res = await self.ai_agent_runner.run_ai_ocr_agents(
                    user_prompt=user_prompt,
                    image_urls=[],
                    data_file_path=self._data_file_path,
                    list_of_agents_to_run=["orchestrator", "analyser"],
                    previous_error=previous_error,
                )

                # reset data file path after use
                self._data_file_path = None
                return res

            else:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"Invalid agent {agent_name}."
                        f"Agent must be either 'data_extractor' or 'analyser'",
                    ),
                )

        @self.app.post("/run_agents")
        async def run_ai_ocr_agents(
            user_prompt: str = Form(...),
            previous_error: str | None = Form(default=None),
        ) -> AgentResult:
            """
            Endpoint to run the agents (both Data Extractor and Analyser) in sequence.

            :param user_prompt: User prompt to process
            :return: Dictionary with the plot path
            """
            agent_result = await self.ai_agent_runner.run_ai_ocr_agents(
                user_prompt=user_prompt,
                image_urls=self._image_urls,
                data_file_path=None,
                list_of_agents_to_run=[
                    "orchestrator",
                    "data_extractor",
                    "analyser",
                ],
                previous_error=previous_error,
            )

            # Reset image URLs after use
            self._image_urls = []
            return agent_result

        @self.app.post("/uploadimages/")
        async def upload_images(
            files: List[UploadFile],
            resize: Literal["original", "medium", "small"] = Form(default="medium"),
        ) -> bool:
            """
            Endpoint to upload images.

            :param files: List of image files to upload
            :param resize: Resize option for images
            :return: True if successful
            """
            if not all(
                file.filename.endswith((".png", ".jpg", ".jpeg")) for file in files
            ):
                raise HTTPException(
                    status_code=422,
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
            """
            Endpoint to upload a data file.

            :param file: Data file to upload
            :return: True if successful
            """
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
            file_path = Path(plot_path)
            if file_path.exists() and file_path.is_file():
                return FileResponse(
                    str(file_path),
                    media_type="image/png",
                    filename=file_path.name,
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
            file_path = Path(df_path)
            if file_path.exists() and file_path.is_file():
                return FileResponse(
                    file_path,
                    media_type="text/csv",
                    filename=file_path.name,
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
    arguments = [
        {
            "name_or_flags": "--ip",
            "type": str,
            "default": "127.0.0.1",
            "help": "IP address to run the app on",
        },
        {
            "name_or_flags": "--port",
            "type": int,
            "default": 8000,
            "help": "Port to run the app on",
        },
        {
            "name_or_flags": "--data_dir",
            "type": str,
            "default": "data",
            "help": "Directory to store data files",
        },
        {
            "name_or_flags": "--output_dir",
            "type": str,
            "default": "plots",
            "help": "Directory to store output plots",
        },
        {
            "name_or_flags": "--image_dir",
            "type": str,
            "default": "images",
            "help": "Directory to store image files",
        },
        {
            "name_or_flags": "--agent_config_path",
            "type": str,
            "default": "configs/agents.yaml",
            "help": "Path to the agent configuration file",
        },
    ]

    parser = argparse.ArgumentParser(description="Run AI Agent OCR Backend App")
    for arg in arguments:
        name_or_flags = arg["name_or_flags"]
        options = {k: v for k, v in arg.items() if k != "name_or_flags"}
        parser.add_argument(name_or_flags, **options)

    args = parser.parse_args()

    app = AiAgentOcrApp(
        ip=args.ip,
        port=args.port,
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        image_dir=Path(args.image_dir),
        agent_config_path=Path(args.agent_config_path),
    )
    app.run()
