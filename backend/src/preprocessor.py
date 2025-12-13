import io
from typing import Tuple

from fastapi import UploadFile
from PIL import Image


class Preprocessor:
    def resize_image(self, image: UploadFile, max_size: Tuple[int, int]) -> UploadFile:
        """
        Resize the image to fit within max_size while maintaining aspect ratio,
        using Lanczos algorithm.

        :param image: UploadFile object representing the image
        :param max_size: Maximum width and height
        :return: resized image as UploadFile
        """
        pil_image = Image.open(image.file)
        pil_image.thumbnail(max_size, Image.Resampling.LANCZOS)

        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format=pil_image.format)
        img_byte_arr.seek(0)

        resized_image = UploadFile(
            filename=image.filename,
            file=img_byte_arr,
        )
        return resized_image

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
        from pdf2image import convert_from_bytes

        pdf_bytes = pdf_file.file.read()
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
