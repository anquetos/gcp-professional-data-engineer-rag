from pathlib import Path


from pdf_extractor import PDFExtractor
from text_embedder import TextEmbedder

pdf_filepath = Path(__file__).resolve().parent.parent / "pdf/source.pdf"
embeddings_filepath = Path(__file__).resolve().parent.parent / "datasets/embeddings.json"

fontnames = [
    "GHSRZR+SabonLTStd-Roman",
    "GHSRZR+SourceCodePro-Regular",
    "GHSRZR+SabonLTStd-Bold",
    "GHSRZR+SabonLTStd-Italic",
    "URTXBU+SourceCodePro-Bold",
]

extractor = PDFExtractor(pdf_filepath)
extractor.reorder_pages(page_offset=-40)
extractor.filter_pages_range(2, 3)
pages_and_text = extractor.extract_pages_text_by_font(fontnames)
extractor.unload_file()

embedder = TextEmbedder(pages_and_text)
embedder.chunk_text()
embedder.calculate_embeddings()
embedder.export_embeddings_to_json(embeddings_filepath)
