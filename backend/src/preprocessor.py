import io

import cv2
import fitz  # PyMuPDF
import numpy as np
import pymupdf4llm
from fastapi import UploadFile
from pdf2image import convert_from_bytes
from PIL import Image


class Preprocessor:
    def _get_pdf_bytes(self, pdf_file: UploadFile) -> bytes:
        """Read PDF bytes safely from an UploadFile by rewinding the stream first."""
        pdf_file.file.seek(0)
        pdf_bytes = pdf_file.file.read()
        if not pdf_bytes:
            raise ValueError("Uploaded PDF is empty.")
        return pdf_bytes

    def resize_image(self, image: UploadFile, max_size: int) -> UploadFile:
        """
        Resize the image to fit within max_size while maintaining aspect ratio,
        using Lanczos algorithm.

        :param image: UploadFile object representing the image
        :param max_size: Maximum width and height for the resized image
        :return: resized image as UploadFile
        """
        pil_image = Image.open(image.file)
        exif = pil_image.getexif()

        pil_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        img_byte_arr = io.BytesIO()
        pil_image.save(
            img_byte_arr,
            format=pil_image.format,
            exif=exif,
        )
        img_byte_arr.seek(0)

        resized_image = UploadFile(
            filename=image.filename,
            file=img_byte_arr,
        )
        return resized_image

    def apply_clahe(self, image: UploadFile) -> UploadFile:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        to enhance the image.

        :param image: UploadFile object representing the image
        :return: enhanced image as UploadFile
        """
        pil_image = Image.open(image.file)
        exif = pil_image.getexif()

        # Convert to grayscale for CLAHE
        gray_image = pil_image.convert("L")

        # Apply CLAHE
        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8),
        )
        np_image = np.array(gray_image)
        clahe_image = clahe.apply(np_image)

        # Convert back to PIL Image
        enhanced_image = Image.fromarray(clahe_image)

        img_byte_arr = io.BytesIO()
        enhanced_image.save(
            img_byte_arr,
            format=pil_image.format,
            exif=exif,
        )
        img_byte_arr.seek(0)

        enhanced_upload_file = UploadFile(
            filename=image.filename,
            file=img_byte_arr,
        )
        return enhanced_upload_file

    def convert_pdf_to_images(
        self,
        pdf_file: UploadFile,
        dpi: int = 300,
    ) -> list[UploadFile]:
        """
        Convert each page of the PDF to an image.

        :param pdf_file: UploadFile object representing the PDF
        :param dpi: Resolution for the converted images
        :return: List of images as UploadFile objects
        """
        pdf_bytes = self._get_pdf_bytes(pdf_file)
        pil_images = convert_from_bytes(pdf_bytes, dpi=dpi)

        image_files = []
        for i, pil_image in enumerate(pil_images):
            img_byte_arr = io.BytesIO()
            pil_image.save(img_byte_arr, format="PNG")
            img_byte_arr.seek(0)

            image_file = UploadFile(
                filename=f"{pdf_file.filename}_page_{i + 1}.png",
                file=img_byte_arr,
            )
            image_files.append(image_file)

        return image_files

    def convert_pdf_to_text(
        self,
        pdf_file: UploadFile,
    ) -> dict[str, str]:
        """
        Convert each page of the PDF to text (markdown).

        Uses pymupdf4llm to extract text from the PDF and returns a dictionary
        where keys are the page numbers and values are the extracted text.

        :param pdf_file: UploadFile object representing the PDF
        :return: Dictionary mapping text filenames to extracted page text
        """

        pdf_bytes = self._get_pdf_bytes(pdf_file)

        doc = fitz.open(
            stream=pdf_bytes,
            filetype="pdf",
        )
        num_pages = doc.page_count

        text_files = {
            f"{pdf_file.filename}_page_{i + 1}.txt": pymupdf4llm.to_text(
                doc,
                pages=[i],
                force_ocr=True,
                ocr_language="eng+deu",
            )
            for i in range(num_pages)
        }

        # if any page text is empty, return empty dictionary
        if any(not text.strip() for text in text_files.values()):
            return {}

        return text_files
