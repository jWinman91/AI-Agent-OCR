from io import BytesIO

import pandas as pd
import streamlit as st
from loguru import logger
from src.backend_client.request_be import BeRequest


class ParentRun:
    """
    Class to run different AI agents in different modes based on user input and uploaded
    data files:
    - Single Agent Mode: Run a single AI agent (Data Extractor or Analyser).
    - AI Agent OCR Mode: Generate plots by running both Data Extractor and Analyser
    agents in sequence.
    """

    def __init__(
        self,
        ip: str = "127.0.0.1",
        port: int = 8000,
        protocol: str = "http",
    ) -> None:
        """
        Initializes the ParentRun with backend connection details.

        :param ip: IP address of the backend server.
        :param port: Port number of the backend server.
        :param protocol: Protocol to use for the request (default is "http").
        """
        self._request_be = BeRequest(ip, port, protocol)
        st.session_state["update_agents"] = self._request_be.post("update_agents")

        st.session_state["uploaded_data_file"] = st.session_state.get(
            "uploaded_data_file", None
        )

        st.session_state["uploaded_images"] = st.session_state.get(
            "uploaded_images", []
        )
        st.session_state["df"] = st.session_state.get("df", None)
        st.session_state["image"] = st.session_state.get("image", None)
        st.session_state["code_summary"] = st.session_state.get("code_summary", None)

    def _get_csv(self, file_name: str) -> pd.DataFrame:
        """
        Fetches a CSV file from the backend and returns it as a pandas DataFrame.

        :param file_name: Name of the CSV file to fetch.
        :return: pandas DataFrame containing the CSV data.
        """
        route = f"df_path/{file_name}/download_data"
        csv_data = self._request_be.get(
            path=route,
            params=None,
            data_type="data",
        )
        df = pd.read_csv(BytesIO(csv_data))
        return df

    @staticmethod
    @st.cache_data
    def _convert_df(df: pd.DataFrame) -> bytes:
        """
        Converts a pandas DataFrame to a CSV byte-encoded format for download.
        :param df: pandas DataFrame to convert.
        :return: Byte-encoded CSV data.
        """
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode("utf-8")

    def build_upload_widget(self, page_name: str) -> bool:
        """
        Builds the upload widget for the Streamlit app.
        Allows users to upload data files for plotting.

        :param page_name: Name of the current page for unique widget identification.
        :return: True if files were uploaded successfully, False otherwise.
        """
        with st.form(f"Upload_widget_{page_name}", clear_on_submit=True):
            uploaded_images = st.file_uploader(
                "Upload your images here (PNG, JPG, JPEG, TIFF, BMP)...",
                type=["png", "jpg", "jpeg", "tiff", "bmp"],
                accept_multiple_files=True,
            )
            resize = st.selectbox(
                "Resize images before uploading?",
                options=["Original", "Medium", "Small"],
                index=1,
            )
            upload_button = st.form_submit_button("Upload images")

            if len(uploaded_images) > 0 and upload_button:
                with st.spinner("Uploading images..."):
                    # Upload images to the backend
                    success = self._request_be.post(
                        "uploadimages",
                        {"resize": resize.lower()},
                        uploaded_images,
                    )
                    logger.info(f"Upload successful: {success}.")
                    st.session_state["uploaded_images"] = [
                        image.name for image in uploaded_images
                    ]

            # Display success message
            if len(st.session_state["uploaded_images"]) > 0:
                uploaded_images_str = "\n".join(
                    [f"- {image}" for image in st.session_state["uploaded_images"]]
                )
                st.success(
                    f"Successfully uploaded the following files:\n{uploaded_images_str}."
                )
                return True
            return False
