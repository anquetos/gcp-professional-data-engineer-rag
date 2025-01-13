from pathlib import Path
from typing import Dict

import pdfplumber
from pdfplumber.page import Page


class PDFExtractor:
    def __init__(self, path: Path):
        self.path = path

    def open_document(self) -> bool:
        if not self.path:
            return False

        with pdfplumber.open(self.path) as pdf:
            self.pdf = pdf
            return True

    def close_document(self) -> bool:
        if self.pdf is not None:
            self.pdf.close()
            self.pdf = None
            return True
        print("PDF is already closed.")
        return False

    def get_pages(
        self,
        first_page_offset: int = 0,
        start_page_number: int = 1,
        end_page_number: int = None,
    ) -> Dict[int, Page]:
        if not self.pdf:
            raise ValueError("PDF document is not opened.")

        total_pages = len(self.pdf.pages)
        if not end_page_number:
            end_page_number = total_pages

        page_range = range(start_page_number, end_page_number + 1)

        pdf_pages = []
        for page_idx, page in enumerate(self.pdf.pages):
            page_number = page_idx - first_page_offset + 1
            if page_number in page_range:
                pdf_pages.append({page_number: page})

        self.pdf_pages = pdf_pages

        return self.pdf_pages


    # TODO : Add method to filter the extracted lines on their font for each page.
    # TODO : Add method for removing hyphens.
    # TODO : Add method to format text.
    # TODO : Add method to create a list of dictionnaries with the relevant pages.
