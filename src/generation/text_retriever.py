"""TextRetriever Module.

This module provides a TextRetriever class for retrieving relevant text sources based on similarity embeddings.

Classes:
    TextRetriever: A class to load text data, calculate embeddings, and retrieve relevant sources based on similarity scores.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Union

import torch
from sentence_transformers import SentenceTransformer, util

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TextRetriever:
    def __init__(
        self,
        pages_text_filepath: Union[str, Path],
        embeddings_filepath: Union[str, Path],
        model_name: str = "all-mpnet-base-v2",
    ):
        self.pages_text_filepath = Path(pages_text_filepath)
        self.embeddings_filepath = Path(embeddings_filepath)
        self._load_pages_text_data()
        self._load_embeddings_data()
        self._initialize_tensor()
        self._set_device()
        self.model_name = model_name
        self.embedding_model = SentenceTransformer(model_name_or_path=self.model_name)

    def _load_pages_text_data(self) -> None:
        """Loads pages text data (extrated from the source document) from a JSON file.

        Raises:
            ValueError: If the loaded pages text is empty or not a list.
        """
        with open(self.pages_text_filepath, "r") as f:
            self.pages_text = json.load(f)

        if not self.pages_text or not isinstance(self.pages_text, list):
            raise ValueError("Pages text must be a non-empty list.")

        logger.info("Pages text data loaded successfully.")

    def _load_embeddings_data(self) -> None:
        """Loads embeddings data from a JSON file.

        Raises:
            ValueError: If the loaded embeddings are empty or not a list.
        """
        with open(self.embeddings_filepath, "r") as f:
            self.embeddings = json.load(f)

        # Input validation
        if not self.embeddings or not isinstance(self.embeddings, list):
            raise ValueError("Embeddings must be a non-empty list.")

        logger.info("Embeddings data loaded successfully.")

    def _initialize_tensor(self) -> None:
        """Initializes the tensor from embeddings.

        Raises:
            ValueError: If embeddings list is empty.
        """
        if not self.embeddings:
            logger.error("Embeddings list is empty.")
            raise ValueError("Embeddings list is empty.")

        self.vectors_tensor = torch.tensor(
            [embedding["embedding"] for embedding in self.embeddings],
            dtype=torch.float32,
        )
        logger.info(
            f"Initialized tensor with shape '{self.vectors_tensor.shape}' and type '{self.vectors_tensor.dtype}'."
        )

    def _set_device(self) -> str:
        """Sets the device (CPU or GPU) for embedding.

        Returns:
            str: Device type ("cpu" or "cuda").
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Device set to '{self.device}'.")

    def search_top_k_vectors(self, query: str, k: int = 5) -> torch.topk:
        """Searches and retrieves the top k vectors most similar to the query.

        Args:
            query (str): The query text.
            k (int, optional): The number of top results to retrieve. Defaults to 5.

        Raises:
            ValueError: If query is empty or k is not a positive integer.

        Returns:
            torch.topk: The top k most similar vectors and their indices.
        """
        if not query:
            logger.error("Query is empty.")
            raise ValueError("Query is empty.")
        if k <= 0:
            logger.error("k must be a positive integer.")
            raise ValueError("k must be a positive integer.")

        query_embedding = self.embedding_model.encode(
            sentences=query,
            batch_size=32,
            device=self.device,
            normalize_embeddings=False,
        )

        cos_sim_scores = util.cos_sim(a=query_embedding, b=self.vectors_tensor)
        self.top_k_vectors = torch.topk(cos_sim_scores[0], k=k)

        return self.top_k_vectors

    def retrieve_relevant_sources(
        self,
        query: str,
        k: int = 5,
    ) -> Dict[str, Union[str, List[Dict[str, Union[int, float, str]]]]]:
        """Retrieves relevant source documents based on the query.

        Args:
            query (str): he query text.
            k (int, optional): The number of top results to retrieve. Defaults to 5.

        Returns:
            Dict[str, Union[str, List[Dict[str, Union[int, float, str]]]]]:
            A dictionary containing the query and the retrieved source documents.
        """

        top_k = self.search_top_k_vectors(query=query, k=k)

        self.retrieved_sources = {"query": query, "outputs": []}

        # Loop through the top k results and append the output to the dictionary
        for score, idx in zip(top_k[0], top_k[1]):
            source_id = int(self.embeddings[idx.item()]["id"])
            source_text = next(
                page["page_text"]
                for page in self.pages_text
                if page["page_number"] == source_id
            )
            self.retrieved_sources["outputs"].append(
                {
                    "id": source_id,
                    "score": score.item(),
                    "text": source_text,
                }
            )

        logger.info("Sources successfully retrieved.")
        return self.retrieved_sources
