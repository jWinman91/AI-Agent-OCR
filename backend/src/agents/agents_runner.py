import datetime
import logging
from pathlib import Path
from typing import Any, Dict, List, Literal

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
    DataExtractorPrompt,
    DataExtractorResponse,
    DataExtractorResult,
    OrchestratorResponse,
)

logging.basicConfig(level=logging.INFO)


class AgentRunner:
    def __init__(
        self,
        config: Dict[str, ConfigModelBuild],
        data_dir: Path = Path("data"),
        plots_dir: Path = Path("plots"),
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

        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.PLOTS_DIR.mkdir(parents=True, exist_ok=True)

        # Save config
        self.CONFIG = config

        # Initialize agents factory with agent names and output validator config
        self.agents = AgentsFactory(
            agent_names=list(config.keys()),
            output_validator_config=config["fix_json"],
        )

    @staticmethod
    def _error_handling(
        agent_name: Literal["data_extractor", "orchestrator", "analyser"],
        agent_result: AgentResult,
    ) -> AgentResult:
        """
        Adds the agent name to the error message in the agent result if an error occurred.

        param agent_name: Name of the agent
        param agent_result: The result object of the agent containing error information
        return: Updated agent result with error message
        """
        if agent_result.error_message:
            agent_result.error_message = (
                f"Previous {agent_name} agent error:\n{agent_result.error_message}"
            )
        return agent_result

    @staticmethod
    def _get_image_timestamp(image_url: Path) -> datetime.datetime:
        """
        Load in an image and get the timestamp from the image metadata.

        :param image_url: Path to the image file
        :return: Timestamp extracted from the image metadata or current time
                 if not available
        """
        if image_url.suffix.lower() != ".pdf":
            exif = Image.open(image_url)._getexif()
            if exif is not None and 36867 in list(exif.keys()):
                return datetime.datetime.strptime(exif[36867], "%Y:%m:%d %H:%M:%S")

        return datetime.datetime.strptime(
            datetime.datetime.now().strftime("%Y:%m:%d %H:%M:%S"), "%Y:%m:%d %H:%M:%S"
        )

    @staticmethod
    def _post_process_and_save_df(
        df: pd.DataFrame,
        file_path: Path,
        timestamps: List[datetime.datetime],
        image_urls: List[Path],
    ) -> Path:
        """
        Post-process the extracted DataFrame by exploding columns with list or tuple
        values.

        :param df: The DataFrame to post-process
        :param file_path: The path to save the post-processed DataFrame
        :param timestamps: List of timestamps corresponding to each row in the DataFrame
        :param image_urls: List of image URLs corresponding to each row in the DataFrame
        :return: The path to the saved DataFrame
        """
        df["timestamp"] = timestamps
        df["image_url"] = image_urls

        cols_to_explode = [
            col
            for col in df.columns
            if df[col].apply(lambda x: isinstance(x, (list, tuple))).any()
        ]

        if cols_to_explode:
            df = df.explode(cols_to_explode, ignore_index=True)

        df.to_csv(file_path, index=False)
        logger.info(f"Extracted DataFrame:\n{df} \n saved to {file_path}.")
        return file_path

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

        return self._error_handling(orchestrator_res)

    async def run_data_extractor(
        self,
        extractor_prompt: DataExtractorPrompt,
        image_urls: List[Path],
    ) -> DataExtractorResult:
        """
        Runs the data extractor agent on a list of image URLs.

        :param extractor_prompt: The prompt to provide to the data extractor.
        :param image_urls: List of image URLs to process.
        :param previous_errors_data_extractor: Previous errors from the data extractor
                                               agent.
        :return: A tuple containing the extracted data as a DataFrame and the plot
                 prompt.
        """
        # Create data_extractor agent based on orchestrator's instructions
        data_extractor = self.agents.create_agent(
            "data_extractor",
            self.CONFIG["data_extractor"],
            DataExtractorResponse,
            {"json_details": extractor_prompt.json_details},
        )

        # Run data_extractor for each image
        df = pd.DataFrame()
        timestamps = []
        for image_url in image_urls:
            data_extractor_res = await self.agents.run_agent(
                agent=data_extractor,
                user_prompt=extractor_prompt.user_prompt,
                image_url=image_url,
                output_type_example=DataExtractorResponse(
                    data={"field_name": "value"},
                    error_message="null or error message",
                ),
            )
            timestamps.append(self._get_image_timestamp(image_url))

            # return error if data extractor fails, no point in continuing
            if (
                data_extractor_res.error_message
                and data_extractor_res.error_message != ""
            ):
                logger.error(
                    f"Error in data extractor for image {image_url}:"
                    f"{data_extractor_res.error_message}"
                )
                return DataExtractorResult(
                    data_file_path=None,
                    error_message=(
                        f"Previous error in data extractor for image {image_url}:"
                        f"{data_extractor_res.error_message}"
                    ),
                )

            df = pd.concat(
                [df, pd.DataFrame([data_extractor_res.data])],
                ignore_index=True,
            )

        # Post-process and save extracted DataFrame
        df_path = self._post_process_and_save_df(
            df=df,
            file_path=self.DATA_DIR / "extracted_data_from_ai_ocr_agents.csv",
            timestamps=timestamps,
            image_urls=image_urls,
        )

        return DataExtractorResult(
            data_file_path=df_path,
            error_message=None,
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
            tool_execution_func=handle_data_download,
        )

        return self._error_handling(data_downloader_res)

    async def run_analyser(
        self,
        analyser_prompt: str,
        system_prompt_args: dict[str, str],
    ) -> AnalyserResult:
        """
        Runs the analyser agent to generate plots based on extracted data.

        :param analyser_prompt: The prompt to provide to the analyser.
        :param system_prompt_args: Additional arguments for the system prompt.
        :return: The analyser's response containing plot details.
        """
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
            user_prompt=analyser_prompt,
            output_type_example=AnalyserResponse(
                python_code="str",
                code_summary="str",
            ),
            tool_execution_func=lambda analyser_response: execute_python_code(
                analyser_response,
                system_prompt_args.get("data_file_path"),
                system_prompt_args.get("downloaded_data_file_path"),
            ),
        )

        return self._error_handling(analyser_res)

    async def run_ai_ocr_agents(
        self,
        user_prompt: str,
        image_urls: List[Path],
        data_file_path: Path | None,
        list_of_agents_to_run: List[
            Literal[
                "orchestrator",
                "data_extractor",
                "analyser",
            ]
        ],
        previous_error: str | None = None,
    ) -> AgentResult:
        """
        Runs the full AI OCR pipeline: data extractor and analyser agents.

        :param user_prompt: The user prompt to provide to the agents.
        :param image_urls: List of image URLs to process.
        :param list_of_agents_to_run: List of agent names to run in the pipeline.
        :param previous_error: Previous error message, if any.
        :return: The final result containing plot and data details.
        """
        # Initialize agent results and functions and parameters for each agent
        agent_res = {
            "orchestrator": None,
            "data_extractor": None,
            "data_downloader": None,
            "analyser": None,
        }
        agent_funcs = {
            "orchestrator": self.run_orchestrator,
            "data_extractor": self.run_data_extractor,
            "data_downloader": self.run_downloader,
            "analyser": self.run_analyser,
        }
        agent_params = {
            "orchestrator": {
                "user_prompt": user_prompt,
                "system_prompt_args": {"previous_error": previous_error},
            },
            "data_extractor": {
                "extractor_prompt": agent_res[
                    "orchestrator"
                ].data_extractor_prompt.user_prompt
                if agent_res["orchestrator"]
                else user_prompt,
                "image_urls": image_urls,
            },
            "data_downloader": {
                "data_download_prompt": agent_res["orchestrator"].data_download_prompt
                if agent_res["orchestrator"]
                and agent_res["orchestrator"].data_download_prompt
                else None,
                "system_prompt_args": {},
            },
            "analyser": {
                "analyser_prompt": agent_res["orchestrator"].analyser_prompt
                if agent_res["orchestrator"]
                else user_prompt,
                "system_prompt_args": {
                    "data_file_path": agent_res["data_extractor"].data_file_path
                    if agent_res["data_extractor"]
                    else data_file_path,
                    "downloaded_data_file_path": agent_res["data_downloader"].file_name
                    if agent_res["data_downloader"]
                    else None,
                },
            },
        }

        # check that each agent type occurs maximum once in the list of agents to run
        if len(list_of_agents_to_run) != len(set(list_of_agents_to_run)):
            logger.error(
                "Duplicate agent types found in list_of_agents_to_run."
                "Each agent type should occur at most once."
            )
            raise ValueError(
                "Duplicate agent types found in list_of_agents_to_run."
                "Each agent type should occur at most once."
            )

        # Run each agent in the list of agents to run
        for agent_name in list_of_agents_to_run:
            if agent_name not in agent_funcs:
                logger.error(f"Unknown agent name: {agent_name}")
                raise ValueError(f"Unknown agent name: {agent_name}")

            logger.info(f"Running {agent_name} agent...")
            agent_res[agent_name] = await agent_funcs[agent_name](
                **agent_params[agent_name]
            )

            if agent_res[agent_name].error_message:
                logger.error(agent_res[agent_name].error_message)
                return agent_res[agent_name]

            if (
                agent_name == "orchestrator"
                and agent_res["orchestrator"]
                and agent_res["orchestrator"].data_download_prompt
            ):
                # Run the data downloader agent
                # if orchestrator provides a data download prompt
                logger.info("Running data_downloader agent...")
                agent_res["data_downloader"] = await self.run_downloader(
                    data_download_prompt=agent_res["orchestrator"].data_download_prompt,
                    system_prompt_args={},
                )

                if agent_res["data_downloader"].error_message:
                    logger.error(agent_res["data_downloader"].error_message)
                    return agent_res["data_downloader"]

        # return the result of the last agent in the list of agents to run
        return agent_res[list_of_agents_to_run[-1]]
