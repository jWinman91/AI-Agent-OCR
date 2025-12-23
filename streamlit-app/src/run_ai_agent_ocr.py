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
                        {"user_prompt": user_prompt},
                        payload_type="data",
                    )
                    st.session_state["tool_used"] = response_json["tool_used"]
                    st.session_state["code_summary"] = response_json["code_summary"]
                    logger.info(f"Plot generation response: {response_json}.")

                    # Fetch generated data and plot from backend
                    st.session_state["df"] = self._get_csv(
                        response_json["data_file_path"]
                    )
                    st.session_state["image"] = self._request_be.get(
                        path=f"plot_path/{response_json.get('plot_path')}/download_plot",
                        params=None,
                        data_type="image",
                    )

                    with st.container(border=True):
                        st.write(f"**Tool used:** {response_json['tool_used']}")
                        st.download_button(
                            label="Download data as CSV",
                            data=self._convert_df(st.session_state["df"]),
                            file_name="df.csv",
                            mime="text/csv",
                        )
                        st.image(
                            st.session_state["image"],
                            caption=response_json["code_summary"],
                        )
                        logger.info(f"Plot generation successful: {response_json}.")

                except RuntimeError as e:
                    st.error(f"Error generating plot: {e}")
                    return
        elif st.session_state.get("image") is not None:
            with st.container(border=True):
                st.write(f"**Tool used:** {st.session_state['tool_used']}")
                st.download_button(
                    label="Download data as CSV",
                    data=self._convert_df(st.session_state["df"]),
                    file_name="df.csv",
                    mime="text/csv",
                )
                st.image(
                    st.session_state["image"], caption=st.session_state["code_summary"]
                )

    def build_page(self) -> None:
        """
        Builds the main page of the Streamlit app.
        Combines the upload widget and user prompt for generating plots.
        """
        self.build_user_prompt()


run_ai_agent_ocr = RunAiAgentOcr()
run_ai_agent_ocr.build_page()
