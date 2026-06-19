import datetime
import logging
from typing import Any, Dict, List, Literal, Tuple

import pandas
import pandas as pd
from loguru import logger
from PIL import Image
from src.agents.agents_factory import AgentsFactory
from src.server.download_server import (
    YFinanceRequest,
    handle_data_download,
)
from src.server.plot_server import execute_python_code, get_dataset_metadata
from src.utils.data_models import (
    AgentResult,
    AnalyserResponse,
    AnalyserResult,
    ConfigModelBuild,
    DataDownloadResponse,
    DataDownloadResult,
    DataExtractorPrompt,
    DataExtractorResponse,
    OrchestratorResponse,
)

logging.basicConfig(level=logging.INFO)


class AgentRunner:
    def __init__(
        self,
        config: Dict[str, ConfigModelBuild],
        data_dir: str = "data",
        plots_dir: str = "plots",
    ) -> None:
        """
        Initialize the AgentRunner with configuration and directories.

        param config: Configuration dictionary for agents
        param data_dir: Directory to store data files
        param plots_dir: Directory to store plot files
        """
        # Create directories for data and plots if they don't exist
        self.DATA_DIR = data_dir
        self.PLOTS_DIR = plots_dir

        # Save config
        self.CONFIG = config

        # Initialize agents factory with agent names and output validator config
        self.agents = AgentsFactory(
            agent_names=list(config.keys()),
            output_validator_config=config["fix_json"],
        )

    @staticmethod
    def _error_handling_prompt(
        agent_name: Literal["data_extractor", "orchestrator", "analyser"],
        error_message: str,
        user_prompt: str,
        previous_errors: Dict[str, str] | None = None,
    ) -> Dict[str, str]:
        """
        Build or update the error handling prompt for the agent.

        param agent_name: Name of the agent
        param error_message: The error message to include
        param user_prompt: The original user prompt
        param previous_errors: Previous errors to include
        return: Updated dictionary of previous errors
        """
        if previous_errors is None:
            previous_errors = {user_prompt: error_message}
        else:
            previous_errors[user_prompt] += f"{agent_name}: {error_message} \n"

        if len(previous_errors) > 3:
            raise ValueError(
                f"Too many errors in {agent_name}: {previous_errors}."
                f"Stopping execution."
            )
        return previous_errors

    @staticmethod
    def _get_image_timestamp(image_url: str) -> datetime.datetime:
        """
        Load in an image and get the timestamp from the image metadata.

        :param image_url: Path to the image file
        :return: Timestamp extracted from the image metadata or current time
                 if not available
        """
        if not image_url.endswith(".pdf"):
            exif = Image.open(image_url)._getexif()
            if exif is not None and 36867 in list(exif.keys()):
                return datetime.datetime.strptime(exif[36867], "%Y:%m:%d %H:%M:%S")

        return datetime.datetime.strptime(
            datetime.datetime.now().strftime("%Y:%m:%d %H:%M:%S"), "%Y:%m:%d %H:%M:%S"
        )

    @staticmethod
    def _build_error_messages(previous_errors: Dict[str, str] | None) -> str:
        """
        Build a formatted string of previous error messages.

        param previous_errors: Dictionary of previous errors
        return: Formatted string of errors
        """
        if not previous_errors:
            return ""
        return "\n".join(
            [f"- User Prompt: {k}\nError: {v}" for k, v in previous_errors.items()]
        )

    async def run_orchestrator(
        self,
        user_prompt: str,
        system_prompt_args: dict[str, Any] | None = None,
    ) -> OrchestratorResponse:
        """
        Runs the orchestrator agent to determine extraction and plotting instructions.

        :param user_prompt: The user prompt to provide to the orchestrator.
        :param system_prompt_args: Optional arguments to format the system prompt.
        :return: The orchestrator's response containing extraction and plotting
        instructions.
        """
        # Orchestrator decides how to extract and plot
        # based on user prompt and previous error
        orchestrator = self.agents.create_agent(
            agent_name="orchestrator",
            config=self.CONFIG["orchestrator"],
            system_prompt_args=system_prompt_args,
            output_type=OrchestratorResponse,
        )
        orchestrator_res = await self.agents.run_agent(
            agent=orchestrator,
            user_prompt=user_prompt,
            output_type_example=OrchestratorResponse(
                analyser_prompt="str",
                data_download_prompt="str",
                data_extractor_prompt=DataExtractorPrompt(
                    user_prompt="str",
                    json_details={"field_name": "str", "data_type": "str"},
                ),
                error_message="null or error message",
            ),
        )

        if orchestrator_res.error_message and orchestrator_res.error_message != "":
            logger.error(f"Error in orchestrator: {orchestrator_res.error_message}")
            raise ValueError(f"Error in orchestrator: {orchestrator_res.error_message}")

        return orchestrator_res

    async def run_data_extractor(
        self,
        user_prompt: str,
        image_urls: List[str],
        previous_errors_data_extractor: dict[str, str] | None = None,
        previous_errors_analyser: dict[str, str] | None = None,
    ) -> Tuple[pandas.DataFrame, str, str]:
        """
        Runs the data extractor agent on a list of image URLs.

        :param user_prompt: The user prompt to provide to the data extractor.
        :param image_urls: List of image URLs to process.
        :param previous_errors_data_extractor: Previous errors from the data extractor
                                               agent.
        :param previous_errors_analyser: Previous errors from the analyser agent.
        :return: A tuple containing the extracted data as a DataFrame and the plot
                 prompt.
        """
        system_prompt_args_orchestrator = {
            "previous_errors_data_extractor": self._build_error_messages(
                previous_errors_data_extractor
            ),
            "previous_errors_analyser": self._build_error_messages(
                previous_errors_analyser
            ),
        }

        # Run the orchestrator to get extraction instructions
        orchestrator_res = await self.run_orchestrator(
            user_prompt, system_prompt_args_orchestrator
        )

        # Create data_extractor agent based on orchestrator's instructions
        data_extractor = self.agents.create_agent(
            "data_extractor",
            self.CONFIG["data_extractor"],
            DataExtractorResponse,
            {"json_details": orchestrator_res.data_extractor_prompt.json_details},
        )

        # Run data_extractor for each image
        user_prompt_data_extractor = orchestrator_res.data_extractor_prompt.user_prompt
        df = pd.DataFrame()
        timestamps = []
        for image_url in image_urls:
            data_extractor_res = await self.agents.run_agent(
                agent=data_extractor,
                user_prompt=user_prompt_data_extractor,
                image_url=image_url,
                output_type_example=DataExtractorResponse(
                    data={"field_name": "value"},
                    error_message="null or error message",
                ),
            )

            # Retry if error in data extractor
            if (
                data_extractor_res.error_message
                and data_extractor_res.error_message != ""
            ):
                logger.error(
                    f"Error in data extractor for image {image_url}:"
                    f"{data_extractor_res.error_message}"
                )
                previous_errors_data_extractor = self._error_handling_prompt(
                    agent_name="data_extractor",
                    error_message=data_extractor_res.error_message,
                    user_prompt=user_prompt,
                    previous_errors=previous_errors_data_extractor,
                )
                return await self.run_data_extractor(
                    user_prompt=user_prompt,
                    image_urls=image_urls,
                    previous_errors_data_extractor=previous_errors_data_extractor,
                    previous_errors_analyser=previous_errors_analyser,
                )

            timestamps.append(self._get_image_timestamp(image_url))

            df = pd.concat(
                [df, pd.DataFrame([data_extractor_res.data])], ignore_index=True
            )
        df["timestamp"] = timestamps
        df["image_url"] = image_urls

        # Explode dataframe if any column contains a list or tuple
        cols_to_explode = [
            col
            for col in df.columns
            if df[col].apply(lambda x: isinstance(x, (list, tuple))).any()
        ]

        if cols_to_explode:
            df = df.explode(cols_to_explode, ignore_index=True)

        logger.info(f"Extracted DataFrame:\n{df}")

        return (
            df,
            orchestrator_res.analyser_prompt,
            orchestrator_res.data_download_prompt,
        )

    async def run_downloader(
        self,
        data_download_prompt: str,
        system_prompt_args: dict[str, str],
    ) -> DataDownloadResponse:
        """
        Runs the data downloader agent based on the provided prompt.

        :param data_download_prompt: The prompt to provide to the data downloader.
        :return: The response from the data downloader agent.
        """
        # Create data_downloader agent based on orchestrator's instructions
        data_downloader = self.agents.create_agent(
            agent_name="data_downloader",
            config=self.CONFIG["data_downloader"],
            output_type=DataDownloadResponse,
            system_prompt_args=system_prompt_args,
        )
        data_downloader_res = await self.agents.run_agent(
            agent=data_downloader,
            user_prompt=data_download_prompt,
            output_type_example=DataDownloadResponse(
                file_name="str",
                download_request=[
                    YFinanceRequest(
                        share_name="E.ON",
                        period="1mo",
                        interval="1d",
                        start=None,
                        end=None,
                    )
                ],
            ),
        )

        try:
            file_path = handle_data_download(
                reqs=data_downloader_res.download_request,
                file_name=data_downloader_res.file_name,
            )
            error_message = None
            return DataDownloadResult(
                data_file_path=file_path,
                error_message=error_message,
            )
        except ValueError as e:
            logger.error(f"Error in data downloader: {str(e)}")
            return DataDownloadResult(
                data_file_path=None,
                error_message=str(e),
            )

    async def run_analyser(
        self,
        user_prompt: str,
        data_download_prompt: str | None,
        system_prompt_args: dict[str, str],
    ) -> AnalyserResult:
        """
        Runs the analyser agent to generate plots based on extracted data.

        :param user_prompt: The user prompt to provide to the analyser.
        :param data_download_prompt: Optional prompt to download data for analysis.
        :param previous_errors_analyser: Previous errors from the analyser agent.
        :param system_prompt_args: Additional arguments for the system prompt.
        :return: The analyser's response containing plot details.
        """
        # If orchestrator or user did not provide a specific data download prompt,
        # run it to get one
        if data_download_prompt is None:
            orchestrator_res = await self.run_orchestrator(user_prompt=user_prompt)
            user_prompt = orchestrator_res.analyser_prompt
            data_download_prompt = orchestrator_res.data_download_prompt

        # If data download prompt provided, run downloader agent to get data
        # else set downloaded_data_file_path to None in system prompt args for analyser
        if data_download_prompt != "":
            data_download_res = await self.run_downloader(
                data_download_prompt=data_download_prompt,
                system_prompt_args=system_prompt_args,
            )
            system_prompt_args["downloaded_data_file_path"] = (
                data_download_res.data_file_path
            )

            # If error in data downloader, return error message in analyser result
            if data_download_res.error_message:
                return AnalyserResult(
                    df_file_path=None,
                    plot_path=None,
                    code_summary="",
                    error_message=data_download_res.error_message,
                )
        else:
            system_prompt_args["downloaded_data_file_path"] = None

        # Add dataset metadata to system prompt args for analyser
        system_prompt_args["dataset_metadata"] = get_dataset_metadata(
            system_prompt_args["data_file_path"]
        )
        system_prompt_args["downloaded_dataset_metadata"] = get_dataset_metadata(
            system_prompt_args["downloaded_data_file_path"]
        )

        # Create analyser agent based on orchestrator's instructions
        analyser = self.agents.create_agent(
            agent_name="analyser",
            config=self.CONFIG["analyser"],
            output_type=AnalyserResponse,
            system_prompt_args=system_prompt_args,
        )

        analyser_res = await self.agents.run_agent(
            agent=analyser,
            user_prompt=user_prompt,
            output_type_example=AnalyserResponse(
                python_code="str",
                code_summary="str",
            ),
        )

        code_res = execute_python_code(
            code=analyser_res.python_code,
            data_file_path=system_prompt_args["data_file_path"],
            downloaded_data_file_path=system_prompt_args["downloaded_data_file_path"],
        )

        return AnalyserResult(
            df_file_path=code_res["df_file_path"],
            plot_path=code_res["plot_path"],
            code_summary=analyser_res.code_summary,
            error_message=code_res["error_message"],
        )

    async def run_ai_ocr_agents(
        self,
        user_prompt: str,
        image_urls: List[str],
        previous_errors_data_extractor: Dict[str, Any] | None = None,
        previous_errors_analyser: Dict[str, Any] | None = None,
    ) -> AgentResult:
        """
        Runs the full AI OCR pipeline: data extractor and analyser agents.

        :param user_prompt: The user prompt to provide to the agents.
        :param image_urls: List of image URLs to process.
        :param previous_errors_data_extractor: Previous errors from the data extractor
        agent.
        :param previous_errors_analyser: Previous errors from the analyser agent.
        :return: The final result containing plot and data details.
        """
        df, plot_prompt, data_download_prompt = await self.run_data_extractor(
            user_prompt=user_prompt,
            image_urls=image_urls,
            previous_errors_data_extractor=previous_errors_data_extractor,
            previous_errors_analyser=previous_errors_analyser,
        )
        df_path = f"{self.DATA_DIR}/extracted_data_from_ai_ocr_agents.csv"
        df.to_csv(df_path, index=False)

        # Analyser generates plot based on orchestrator's instructions and extracted data
        system_prompt_args_analyser = {
            "data_file_path": df_path,
            "output_dir": self.PLOTS_DIR,
        }
        plot_res = await self.run_analyser(
            plot_prompt=plot_prompt,
            data_download_prompt=data_download_prompt,
            system_prompt_args=system_prompt_args_analyser,
        )

        if plot_res.error_message and plot_res.error_message != "":
            logger.error(f"Error in analyser: {plot_res.error_message}")
            previous_errors_analyser = self._error_handling_prompt(
                agent_name="analyser",
                error_message=plot_res.error_message,
                user_prompt=user_prompt,
                previous_errors=previous_errors_analyser,
            )
            return await self.run_ai_ocr_agents(
                user_prompt=user_prompt,
                image_urls=image_urls,
                previous_errors_data_extractor=previous_errors_data_extractor,
                previous_errors_analyser=previous_errors_analyser,
            )

        return AgentResult(
            plot_path=plot_res.plot_path,
            code_summary=plot_res.code_summary,
            data_file_path=df_path,
        )
