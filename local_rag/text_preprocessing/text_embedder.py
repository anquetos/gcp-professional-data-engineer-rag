"""TextEmbedder Module.

This module provides a TextEmbedder class for embedding and processing text.

Classes:
    TextEmbedder: A class to load text data, split it into chunks, calculate embeddings, and export them to a JSON file.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import torch
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TextEmbedder:
    def __init__(
        self, text: List[Dict[str, Any]], model_name: str = "all-mpnet-base-v2"
    ):
        """Initializes the TextEmbedder class.

        Args:
            text (List[Dict[str, Any]]): List of dictionaries containing text data.
            model_name (str, optional): The name of the pre-trained model to use. Defaults to "all-mpnet-base-v2".
        """
        self.text = text
        self.model_name = model_name
        self._set_device()
        self.embeddings: List[Dict[str, Any]] = []

    def _set_device(self) -> str:
        """Sets the device (CPU or GPU) for embedding.

        Returns:
            str: Device type ("cpu" or "cuda").
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    def chunk_text(self, chunk_overlap: int = 50) -> List[Dict[str, Any]]:
        """Splits the text into chunks.

        Args:
            chunk_overlap (int, optional): Number of overlapping tokens. Defaults to 50.

        Raises:
            ValueError: If text data is not available for chunking.

        Returns:
            List[Dict[str, Any]]: Text split into chunks with metadata.
        """
        logger.info("Starting text chunking...")
        if not self.text:
            logger.error("Text data is required for chunking.")
            raise ValueError("Text data is required for chunking.")
        text_splitter = SentenceTransformersTokenTextSplitter(
            chunk_overlap=chunk_overlap,
            model_name=self.model_name,
        )
        with tqdm(total=len(self.text), desc="Chunking text") as pbar:
            for page in self.text:
                page["page_chunks"] = text_splitter.split_text(text=page["page_text"])
                page["page_chunks_max_tokens_count"] = max(
                    text_splitter.count_tokens(text=chunk)
                    for chunk in page["page_chunks"]
                )
                page["page_chunks_count"] = len(page["page_chunks"])
                pbar.update(1)

        logger.info("Text chunking completed successfully.")
        return self.text

    def calculate_embeddings(self) -> List[Dict[str, Any]]:
        """Calculates embeddings for the text chunks.

        Raises:
            ValueError: If text is not chunked before calculating embeddings.

        Returns:
            List[Dict[str, Any]]: List of embeddings for each text chunk with metadata.
        """
        logger.info("Starting embedding calculation...")
        if not hasattr(self, "text") or not any(
            "page_chunks" in page for page in self.text
        ):
            logger.error("Text must be chunked before calculating embeddings.")
            raise ValueError("Text must be chunked before calculating embeddings.")

        embedding_model = SentenceTransformer(model_name_or_path=self.model_name)

        self.embeddings = []
        for page in self.text:
            for chunk in page["page_chunks"]:
                embedding = embedding_model.encode(
                    sentences=chunk,
                    batch_size=32,
                    device=self.device,
                    normalize_embeddings=True,
                )
                self.embeddings.append(
                    {
                        "id": page["page_number"],
                        "text": chunk,
                        "embedding": embedding.tolist(),  # JSON can not serialize numpy arrays
                    }
                )

        logger.info("Embedding calculation completed successfully.")
        return self.embeddings

    def export_embeddings_to_json(
        self, filepath: Path, force_overwrite: bool = False
    ) -> None:
        """Exports the embeddings to a JSON file.

        Args:
            filepath (Path): The file path to save embeddings.
            force_overwrite (bool, optional): Whether to overwrite existing file. Defaults to False.

        Raises:
            ValueError: If there are no embeddings to export.
        """
        logger.info("Starting export of embeddings to JSON...")
        if not self.embeddings:
            logger.error("No embeddings to export.")
            raise ValueError("No embeddings to export.")
        if not filepath.is_file() or force_overwrite:
            with open(filepath, "w") as f:
                json.dump(self.embeddings, f, indent=2)
            logger.info(f"Embeddings exported to {filepath}.")
        else:
            logger.info(
                "Embeddings already exist. If you want to overwrite it, turn 'force_overwrite' to 'True'."
            )
