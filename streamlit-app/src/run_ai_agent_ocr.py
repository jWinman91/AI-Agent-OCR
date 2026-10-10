import streamlit as st
from loguru import logger
from src.parent_run import ParentRun


class RunAiAgentOcr(ParentRun):
    """
    Class to generate plots based on user input and uploaded data files.
    """

    def build_user_prompt(self) -> None:
        """
        Builds the user prompt input widget for the Streamlit app.
        Allows users to specify their plotting requirements.
        """
        files_uploaded = self.build_upload_widget("ai_agent_ocr")
        user_prompt = st.text_area(
            "Enter your plotting requirements here...",
            height=300,
        )
        submit_button = st.button("Generate Plot")

        if user_prompt and submit_button and files_uploaded:
            with st.spinner("Generating plot..."):
                try:
                    # Send request to backend to generate plot
                    response_json = self._request_be.post(
                        "run_agents",
                        {
                            "user_prompt": user_prompt,
                            "previous_error": st.session_state.get(
                                "previous_error_pipeline", None
                            ),
                        },
                        payload_type="data",
                    )
                    if response_json.get("error_message"):
                        st.error(response_json["error_message"])
                        st.session_state["previous_error_pipeline"] = response_json[
                            "error_message"
                        ]
                        return
                    st.session_state["previous_error_pipeline"] = None
                    logger.info(f"Plot generation response: {response_json}.")
                    self._load_result(response_json)

                except Exception as e:
                    st.error(f"Error generating plot: {e}")
                    return
                finally:
                    st.session_state["uploaded_images"] = []
        self._render_result()

    def build_page(self) -> None:
        """
        Builds the main page of the Streamlit app.
        Combines the upload widget and user prompt for generating plots.
        """
        self.build_user_prompt()


run_ai_agent_ocr = RunAiAgentOcr()
run_ai_agent_ocr.build_page()
