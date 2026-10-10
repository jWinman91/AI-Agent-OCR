from io import BytesIO
from typing import Any

import pandas as pd
import streamlit as st
from loguru import logger
from src.backend_client.request_be import BeRequest
from streamlit.runtime.uploaded_file_manager import UploadedFile


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
        try:
            if self._request_be.post("update_agents") is not True:
                raise RuntimeError("Backend could not initialize agents.")
        except Exception as error:
            st.error(f"Cannot initialize agents: {error}")
            st.stop()

        st.session_state["uploaded_data_file"] = st.session_state.get(
            "uploaded_data_file", None
        )

        st.session_state["uploaded_images"] = st.session_state.get(
            "uploaded_images", []
        )
        st.session_state["df"] = st.session_state.get("df", None)
        st.session_state["image"] = st.session_state.get("image", None)
        st.session_state["code_summary"] = st.session_state.get("code_summary", None)
        for workflow in ("extractor", "analyser", "pipeline"):
            st.session_state.setdefault(f"previous_error_{workflow}", None)

    @staticmethod
    def _output_path(value: object) -> str | None:
        return str(value) if value is not None and str(value) not in {"", "."} else None

    def _load_result(self, response: dict[str, Any], extractor: bool = False) -> None:
        generated_csv = self._output_path(response.get("df_file_path"))
        input_csv = self._output_path(response.get("data_file_path"))
        plot = self._output_path(response.get("plot_path"))
        st.session_state["df"] = None
        st.session_state["image"] = None
        st.session_state["code_summary"] = response.get("code_summary", "")
        if not (generated_csv or plot or (extractor and input_csv)):
            st.error("No plot or analysis data generated.")
        csv_path = generated_csv or (input_csv if extractor else None)
        if csv_path:
            try:
                st.session_state["df"] = self._get_csv(csv_path)
            except Exception as error:
                st.error(f"Cannot download data: {error}")
        if plot:
            try:
                st.session_state["image"] = self._request_be.get(
                    path=f"plot_path/{plot}/download_plot", data_type="image"
                )
            except Exception as error:
                st.error(f"Cannot download plot: {error}")

    def _render_result(self, preview: bool = False) -> None:
        frame = st.session_state.get("df")
        if frame is not None:
            with st.container(border=True):
                st.dataframe(frame.head() if preview else frame)
                st.download_button(
                    "Download data as CSV",
                    self._convert_df(frame),
                    file_name="df.csv",
                    mime="text/csv",
                )
        image = st.session_state.get("image")
        if image is not None:
            with st.container(border=True):
                st.image(image, caption=st.session_state.get("code_summary"))

    def _upload_files(
        self,
        path: str,
        params: dict[str, str],
        files: UploadedFile | list[UploadedFile],
    ) -> bool:
        try:
            if self._request_be.post(path, params, files) is not True:
                raise RuntimeError("Backend did not confirm the upload.")
            return True
        except Exception as error:
            st.error(f"Upload failed: {error}")
            return False

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
                "Upload your images here (PNG, JPG, JPEG, TIFF, BMP, PDF)...",
                type=["png", "jpg", "jpeg", "tiff", "bmp", "pdf"],
                accept_multiple_files=True,
            )
            resize = st.selectbox(
                "Resize images before uploading?",
                options=["Original", "Medium", "Small"],
                index=1,
            )
            upload_button = st.form_submit_button("Upload images")

            if len(uploaded_images) > 0 and upload_button:
                # check if the file is a pdf and send it to the uploadpdf endpoint
                if any(file.name.lower().endswith(".pdf") for file in uploaded_images):
                    # not more than 1 pdf file can be uploaded though
                    if len(uploaded_images) > 1:
                        st.error("Please upload only one PDF file at a time.")
                        return False

                    pdf_file = uploaded_images[0]
                    with st.spinner("Uploading PDF..."):
                        success = self._upload_files(
                            "uploadpdf",
                            {"resize": resize.lower()},
                            pdf_file,
                        )
                        if not success:
                            st.session_state["uploaded_images"] = []
                            return False
                        st.session_state["uploaded_images"] = [pdf_file.name]

                else:
                    with st.spinner("Uploading images..."):
                        # Upload images to the backend
                        success = self._upload_files(
                            "uploadimages",
                            {"resize": resize.lower()},
                            uploaded_images,
                        )
                        if not success:
                            st.session_state["uploaded_images"] = []
                            return False
                        st.session_state["uploaded_images"] = [
                            image.name for image in uploaded_images
                        ]
                logger.info(f"Upload successful: {success}.")

                # Display success message
                if len(st.session_state["uploaded_images"]) > 0:
                    uploaded_images_str = "\n".join(
                        [f"- {image}" for image in st.session_state["uploaded_images"]]
                    )
                    st.success(
                        f"Successfully uploaded the following files:\n"
                        f"{uploaded_images_str}."
                    )
                    return True
                return False
            return bool(st.session_state.get("uploaded_images"))
