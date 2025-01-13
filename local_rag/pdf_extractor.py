from pathlib import Path

import pdfplumber
import pdfplumber.page


class PDFExtractor:
    def __init__(self, path: Path):
        self.pdf_path = path
        self.pdf_pages = None

    def extract_and_filter_pages(
        self,
        page_offset: int = 0,
        first_page_number: int = 1,
        last_page_number: int = None,
    ) -> list[dict[int: pdfplumber.page.Page]]:
        """
        Extracts and filters pages from a PDF file.

        Args:
            page_offset (int, optional): The offset to apply to the page numbers. Defaults to 0.
            first_page_number (int, optional): The first page number to include. Defaults to 1.
            last_page_number (int, optional): The last page number to include. Defaults to None.

        Raises:
            TypeError: If the provided 'pdf_path' is not a Pathlib object.
            FileNotFoundError: If the file at 'pdf_path' does not exist.

        Returns:
            list[dict[int, pdfplumber.page.Page]]: A list of dictionaries where the key is the page number and the value
            is the pdfplumber Page object.
        """

        if not isinstance(self.pdf_path, Path):
            raise TypeError("The provided 'pdf_path' is not a Pathlib object")

        if not self.pdf_path.exists():
            raise FileNotFoundError(f"The file at '{self.pdf_path}' does not exist.")

        with pdfplumber.open(self.pdf_path) as pdf:
            self.pdf_pages = pdf.pages

            if not last_page_number:
                last_page_number = len(self.pdf_pages)
            page_range = range(first_page_number, last_page_number + 1)

            assert last_page_number > first_page_number, (
                "The number of the last page cannot be less than or equal to the number of the first page."
            )

            pdf_pages = []
            for page_idx, page in enumerate(self.pdf_pages):
                page_number = page_idx - page_offset + 1
                if page_number in page_range:
                    pdf_pages.append({page_number: page})
            self.pdf_pages = pdf_pages

        return self.pdf_pages
