from pathlib import Path

from pdf_extractor import PDFExtractor

# Set the path of the document
pdf_filepath = Path(__file__).resolve().parent.parent / "pdf/source.pdf"

if __name__ == "__main__":
    pdf = PDFExtractor(pdf_filepath)
    if pdf.open_document():
        pages = pdf.get_pages(first_page_offset=40)
        print(pages)
        pdf.close_document()
