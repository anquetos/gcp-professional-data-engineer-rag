"""PDFExtractor Module.

This module provides a PDFExtractor class for extracting and processing text from PDF files.

Classes:
    PDFExtractor: A class to load, extract pages, filter, and process text from a PDF file.
    The text can be further processed to remove hyphens, reorder pages, and extract text by specific fonts.
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

import pdfplumber

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PDFExtractor:
    def __init__(self, path: Path):
        """Initializes the PDFExtractor with a given PDF file path.

        Args:
            path (Path): The path to the PDF file.
        """
        self.path = path
        self.pdf = None
        self.pdf_pages: List[Dict[str, Optional[int]]] = []
        self.pages_text: List[Dict[str, str]] = []
        self._load_file()
        self._extract_pages()

    def _load_file(self) -> None:
        """Loads the PDF file specified by the path and creates a `pdfplumber.PDF` instance.

        Raises:
            FileNotFoundError: If the PDF file does not exist.
        """
        if not self.path.exists():
            logger.error(f"The file '{self.path}' does not exist.")
            raise FileNotFoundError(f"The file '{self.path}' does not exist.")
        self.pdf = pdfplumber.open(self.path)

    def _extract_pages(self) -> None:
        """Extracts all pages (one `pdfplumber.Page` instance per page) from the loaded PDF file.

        Raises:
            ValueError: If no PDF file is loaded.
        """
        if not self.pdf:
            logger.error("No PDF file loaded.")
            raise ValueError("No PDF file loaded.")
        for page_number, page in enumerate(self.pdf.pages):
            self.pdf_pages.append({"page_number": page_number + 1, "page_object": page})
        logger.info(f"Extracted {len(self.pdf_pages)} pages from the PDF.")

    def unload_file(self) -> None:
        """Closes the loaded PDF file."""
        if self.pdf:
            self.pdf.close()
            self.pdf = None
            logger.info("PDF file closed successfully.")

    def reorder_pages(self, page_offset: int = 0) -> None:
        """Reorders the pages by a given offset.

        Args:
            page_offset (int, optional): The offset to apply to page numbers. Defaults to 0.
        """
        if page_offset == 0:
            return
        for page in self.pdf_pages:
            page["page_number"] += page_offset
        logger.info(f"Reordered pages with an offset of {page_offset}.")

    def filter_pages_range(
        self, first_page_number: int = 1, last_page_number: Optional[int] = None
    ) -> None:
        """Filters the pages to include only those within a specified range.

        Args:
            first_page_number (int, optional): The first page number to include. Defaults to 1.
            last_page_number (Optional[int], optional): The last page number to include. Defaults to None.
        """
        if not last_page_number:
            last_page_number = len(self.pdf_pages)
        page_range = range(first_page_number, last_page_number + 1)
        self.pdf_pages = [
            page for page in self.pdf_pages if page["page_number"] in page_range
        ]
        logger.info(
            f"Filtered pages from range {first_page_number} to {last_page_number}."
        )

    def extract_pages_text_by_font(self, fontnames: list[str]) -> List[Dict[str, str]]:
        """Extracts text from pages using specified font names.

        Args:
            fontnames (list[str]): A list of font names to filter text by.

        Raises:
            ValueError: A list of dictionaries containing page numbers and their extracted text.

        Returns:
            List[Dict[str, str]]: If no pages are available to extract text from.
        """
        if not self.pdf_pages:
            logger.error("No pages to extract text from.")
            raise ValueError("No pages to extract text from.")
        filtered_lines = []
        for page in self.pdf_pages:
            lines = page["page_object"].extract_text_lines(
                return_chars=True, keep_blank_chars=True
            )
            for line in lines:
                line_char = [
                    char["text"]
                    for char in line["chars"]
                    if char.get("fontname") in fontnames
                ]
                line_text = "".join(line_char)
                filtered_lines.append(line_text)
            text = "\n".join(filtered_lines).rstrip()
            text = self._remove_hyphens(text)
            text = self._basic_text_formatter(text)
            if text:
                self.pages_text.append(
                    {"page_number": page["page_number"], "page_text": text}
                )
        logger.info("Extracted text from pages by font.")
        return self.pages_text

    def _dehyphenate(self, lines: list[str], line_no: int) -> list[str]:
        """Helper function to remove hyphens from a specific line and join with the next line.

        Args:
            lines (list[str]): The list of text lines.
            line_no (int): The line number to process.

        Returns:
            list[str]: The updated list of text lines.
        """
        next_line = lines[line_no + 1]
        word_suffix = next_line.split(" ")[0]

        lines[line_no] = lines[line_no][:-1] + word_suffix
        lines[line_no + 1] = lines[line_no + 1][len(word_suffix) :]
        return lines

    def _remove_hyphens(self, text: str) -> str:
        """Removes hyphens from the text and rejoins split words.

        Args:
            text (str): The text to process.

        Returns:
            str: The processed text without hyphens.
        """
        lines = [line.rstrip() for line in text.split("\n")]

        for i in range(len(lines) - 1):
            if lines[i].endswith("-"):
                lines = self._dehyphenate(lines, i)

        return " ".join(lines)

    def _basic_text_formatter(self, text: str) -> str:
        """converts to lowercase, removes newlines and extra spaces, and fixes common ligature issues.

        Args:
            text (str): The text to format.

        Returns:
            str: The formatted text.
        """
        formatted_text = re.sub(r"\s+", " ", text.casefold())
        formatted_text = (
            formatted_text.replace("\n", " ")
            .replace("fifi", "fi")
            .replace("flfl", "fl")
            .split()
        )

        return " ".join(formatted_text)
