from pathlib import Path

from pdf_extractor import PDFExtractor

# Set the path of the document
pdf_filepath = Path(__file__).resolve().parent.parent / "pdf/source.pdf"    

if __name__ == "__main__":
    pdf = PDFExtractor(pdf_filepath)
    pages = pdf.extract_and_filter_pages(page_offset=40, last_page_number=305)
