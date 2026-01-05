import streamlit as st
from loguru import logger
from src.parent_run import ParentRun


class SingleAgent(ParentRun):
    """
    Class to run a single AI agent based on user input and uploaded data files.
    """

    def build_extractor_widget(self) -> None:
        """
        Builds the Streamlit widget to run the Data Extractor Agent in single mode.
        """
        files_uploaded = self.build_upload_widget("data_extractor_agent")
        user_prompt = st.text_area(
            "Enter your prompt for the Data Extractor Agent...",
            height=300,
        )

        submit_button = st.button("Run Data Extractor Agent")
        if submit_button and user_prompt and files_uploaded:
            logger.info("Running Data Extractor Agent...")
            with st.spinner("Running Data Extractor Agent..."):
                try:
                    response_json = self._request_be.post(
                        "run_single_agent",
                        {
                            "user_prompt": user_prompt,
                            "agent_name": "extractor",
                        },
                        payload_type="data",
                    )

                    # Fetch generated data and plot from backend
                    st.session_state["df"] = self._get_csv(
                        response_json["data_file_path"]
                    )

                    with st.container(border=True):
                        # display head of dataframe
                        st.dataframe(st.session_state["df"].head())
                        st.download_button(
                            label="Download data as CSV",
                            data=self._convert_df(st.session_state["df"]),
                            file_name="df.csv",
                            mime="text/csv",
                        )
                except Exception as e:
                    st.error(f"Error running Data Extractor Agent: {e}")
                    return
        elif st.session_state.get("df") is not None:
            st.dataframe(st.session_state["df"].head())
            st.download_button(
                label="Download data as CSV",
                data=self._convert_df(st.session_state["df"]),
                file_name="df.csv",
                mime="text/csv",
            )

            logger.info("Data Extractor Agent run completed.")

    def build_upload_data_widget(self) -> None:
        """
        Builds the Streamlit widget to run the Analyser Agent in single mode.
        """
        with st.form("Upload_data", clear_on_submit=True):
            uploaded_file = st.file_uploader(
                "Upload your data files here (CSV, Excel, ...)",
                type=["csv", "xlsx"],
                accept_multiple_files=False,
            )
            submit_button = st.form_submit_button("Upload Data File")

            if submit_button and uploaded_file is not None:
                success = self._request_be.post(
                    "uploaddatafile",
                    {},
                    uploaded_file,
                )
                st.session_state["uploaded_data_file"] = uploaded_file.name
                logger.info(f"Upload successful: {success}.")

            # Display success message
            if st.session_state.get("uploaded_data_file") is not None:
                st.success(
                    f"Successfully uploaded the file: "
                    f"{st.session_state['uploaded_data_file']}."
                )
                return True
            return False

    def build_analyser_widget(self) -> None:
        """
        Builds the Streamlit widget to run the Analyser Agent in single mode.
        """
        data_uploaded = self.build_upload_data_widget()
        user_prompt = st.text_area(
            "Enter your plotting requirements here...",
            height=300,
        )
        submit_button = st.button("Generate Plot")

        # Get the config file for the analyser agent and check for available tools
        analyser_config = self._request_be.get(
            "get_config",
            {"config_name": "analyser"},
        )
        mcp_servers = analyser_config.get("mcp_servers", [])
        download_tool_available = any(
            map(lambda x: "download" in str(x["args"]), mcp_servers)
        )

        if user_prompt and submit_button and (data_uploaded or download_tool_available):
            with st.spinner("Generating plot..."):
                try:
                    # Send request to backend to generate plot
                    response_json = self._request_be.post(
                        "run_single_agent",
                        {
                            "user_prompt": user_prompt,
                            "agent_name": "analyser",
                        },
                        payload_type="data",
                    )
                    st.session_state["code_summary"] = response_json["code_summary"]
                    logger.info(f"Plot generation response: {response_json}.")

                    image_path = response_json.get("plot_path", None)
                    data_file_path = response_json.get("data_file_path", None)

                    if data_file_path:
                        st.session_state["df"] = self._get_csv(data_file_path)
                        with st.container(border=True):
                            st.dataframe(st.session_state["df"])

                    if image_path:
                        st.session_state["image"] = self._request_be.get(
                            path=f"plot_path/{response_json.get('plot_path')}/download_plot",
                            params=None,
                            data_type="image",
                        )

                        with st.container(border=True):
                            st.image(
                                st.session_state["image"],
                                caption=st.session_state["code_summary"],
                            )

                    if not data_file_path and not image_path:
                        st.error("No plot or data generated.")

                except Exception as e:
                    logger.error(f"Error generating plot: {e}")
                    st.error(f"Error generating plot: {e}")
        elif st.session_state.get("df") is not None:
            with st.container(border=True):
                st.dataframe(st.session_state["df"])
        elif st.session_state.get("image") is not None:
            with st.container(border=True):
                st.image(
                    st.session_state["image"], caption=st.session_state["code_summary"]
                )

    def build_page(self) -> None:
        """
        Builds the Streamlit page for the single AI agent.
        """
        select_agent = st.selectbox(
            "Select AI Agent",
            options=["Data Extractor", "Analyser"],
            index=0,
        )

        if select_agent == "Data Extractor":
            self.build_extractor_widget()
        elif select_agent == "Analyser":
            self.build_analyser_widget()
        else:
            st.error("Invalid agent selected.")


single_agent = SingleAgent()
single_agent.build_page()
