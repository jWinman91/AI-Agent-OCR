import streamlit as st
import pandas as pd

from loguru import logger
from io import BytesIO
from src.plot_agent_client.request_be import BeRequest


class GeneratePlot:
    """
    Class to generate plots based on user input and uploaded data files.
    """
    def __init__(self, ip: str = "127.0.0.1", port: int = 8000, protocol: str = "http") -> None:
        self._request_be = BeRequest(ip, port, protocol)

        st.session_state["uploaded_images"] = st.session_state.get("uploaded_images", [])
        st.session_state["df"] = st.session_state.get("df", None)
        st.session_state["image"] = st.session_state.get("image", None)
        st.session_state["tool_used"] = st.session_state.get("tool_used", None)
        st.session_state["code_summary"] = st.session_state.get("code_summary", None)

    def get_csv(self, file_name: str) -> pd.DataFrame:
        """
        Fetches a CSV file from the backend and returns it as a pandas DataFrame.
        """
        csv_data = self._request_be.get(f"df_path/{file_name}/download_data", image_file=False)
        df = pd.read_csv(BytesIO(csv_data))
        return df

    @staticmethod
    @st.cache_data
    def convert_df(df: pd.DataFrame):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode("utf-8")

    def build_upload_widget(self) -> bool:
        """
        Builds the upload widget for the Streamlit app.
        Allows users to upload data files for plotting.
        """
        with st.form("Upload_widget", clear_on_submit=True):
            uploaded_images = st.file_uploader("Upload your images here...", accept_multiple_files=True)
            upload_button = st.form_submit_button("Upload images")

            if len(uploaded_images) > 0 and upload_button:
                with st.spinner("Uploading images..."):
                    success = self._request_be.post("uploadimages", data_files=uploaded_images)
                    logger.info(f"Upload successful: {success}.")
                    st.session_state["uploaded_images"] = [image.name for image in uploaded_images]

            if len(st.session_state["uploaded_images"]) > 0:
                uploaded_images_str = "\n".join([f"- {image}" for image in st.session_state["uploaded_images"]])
                st.success(f"Successfully uploaded the following files:\n{uploaded_images_str}.")
                return True
            return False

    def build_user_prompt(self) -> None:
        """
        Builds the user prompt input widget for the Streamlit app.
        Allows users to specify their plotting requirements.
        """
        uploaded_file = self.build_upload_widget()
        user_prompt = st.text_area("Enter your plotting requirements here...", height=300)
        submit_button = st.button("Generate Plot")

        if user_prompt and submit_button and uploaded_file:
            with st.spinner("Generating plot..."):
                print(f"User prompt: {user_prompt}")
                try:
                    response_json = self._request_be.post("run_agents", user_prompt)
                    st.session_state["df"] = self.get_csv(response_json["df_path"])
                    st.session_state["image"] = self._request_be.get(f"plot_path/{response_json.get('plot_path')}/download_plot")
                    st.session_state["tool_used"] = response_json["tool_used"]
                    st.session_state["code_summary"] = response_json["code_summary"]
                    with st.container(border=True):
                        st.write(f"**Tool used:** {response_json['tool_used']}")
                        st.download_button(
                            label="Download data as CSV",
                            data=self.convert_df(st.session_state["df"]),
                            file_name="df.csv",
                            mime="text/csv"
                        )
                        st.image(st.session_state["image"], caption=response_json["code_summary"])
                        logger.info(f"Plot generation successful: {response_json}.")

                except RuntimeError as e:
                    st.error(f"Error generating plot: {e}")
                    return
        elif st.session_state.get("image") is not None:
            with st.container(border=True):
                st.write(f"**Tool used:** {st.session_state['tool_used']}")
                st.download_button(
                    label="Download data as CSV",
                    data=self.convert_df(st.session_state["df"]),
                    file_name="df.csv",
                    mime="text/csv"
                )
                st.image(st.session_state["image"], caption=st.session_state["code_summary"])

    def build_page(self) -> None:
        """
        Builds the main page of the Streamlit app.
        Combines the upload widget and user prompt for generating plots.
        """
        self.build_user_prompt()


generate_plot = GeneratePlot()
generate_plot.build_page()
