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
                            "agent_name": "data_extractor",
                            "previous_error": st.session_state.get(
                                "previous_error_extractor", None
                            ),
                        },
                        payload_type="data",
                    )
                    if response_json.get("error_message"):
                        st.error(response_json["error_message"])
                        st.session_state["previous_error_extractor"] = response_json[
                            "error_message"
                        ]

                        return

                    st.session_state["previous_error_extractor"] = None
                    self._load_result(response_json, extractor=True)
                except Exception as e:
                    st.error(f"Error running Data Extractor Agent: {e}")
                    return
                finally:
                    st.session_state["uploaded_images"] = []
        self._render_result(preview=True)

    def build_upload_data_widget(self) -> bool:
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
                success = self._upload_files(
                    "uploaddatafile",
                    {},
                    uploaded_file,
                )
                if not success:
                    st.session_state["uploaded_data_file"] = None
                    return False
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

        if user_prompt and submit_button:
            if not data_uploaded:
                st.warning(
                    "No data file was uploaded."
                    "Download agent will try to download data based on prompt "
                    "requirements, but this may lead to errors if not done correctly."
                )
            with st.spinner("Generating plot..."):
                try:
                    # Send request to backend to generate plot
                    response_json = self._request_be.post(
                        "run_single_agent",
                        {
                            "user_prompt": user_prompt,
                            "agent_name": "analyser",
                            "previous_error": st.session_state.get(
                                "previous_error_analyser", None
                            ),
                        },
                        payload_type="data",
                    )
                    if response_json.get("error_message"):
                        st.error(response_json["error_message"])
                        st.session_state["previous_error_analyser"] = response_json[
                            "error_message"
                        ]
                        return
                    st.session_state["previous_error_analyser"] = None
                    logger.info(f"Plot generation response: {response_json}.")
                    self._load_result(response_json)

                except Exception as e:
                    logger.error(f"Error generating plot: {e}")
                    st.error(f"Error generating plot: {e}")
                finally:
                    st.session_state["uploaded_data_file"] = None
        self._render_result()

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
