import json
from pathlib import Path

from content_generation.text_retriever import TextRetriever

# Set path
pages_text_filepath = (
    Path(__file__).resolve().parent.parent / "datasets/pages-text.json"
)
embeddings_filepath = (
    Path(__file__).resolve().parent.parent / "datasets/embeddings-overlap-50.json"
)

# Load pages text from the JSON file
with open(pages_text_filepath, "r") as f:
    json_string = f.read()
    pages_text_data = json.loads(json_string)

# Load embeddings from the JSON file
with open(embeddings_filepath, "r") as f:
    json_string = f.read()
    embeddings_data = json.loads(json_string)


retriever = TextRetriever(embeddings=embeddings_data, source_document=pages_text_data)
print(retriever.retrieve_relevant_sources(query="stream"))
