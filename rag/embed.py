"""Embedding utilities for RAG using Azure OpenAI."""
from openai import AzureOpenAI
import numpy as np
from typing import List, Union
import logging

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Wrapper for Azure OpenAI embedding model."""

    def __init__(
        self,
        azure_endpoint: str,
        azure_key: str,
        azure_deployment: str = "text-embedding-ada-002",
        api_version: str = "2024-12-01-preview"
    ):
        """
        Initialize Azure OpenAI embedding model.
        
        Args:
            azure_endpoint: Azure OpenAI endpoint URL
            azure_key: Azure OpenAI API key
            azure_deployment: Deployment name for embeddings
            api_version: API version
        """
        self.azure_endpoint = azure_endpoint
        self.azure_deployment = azure_deployment
        logger.info(f"Initializing Azure OpenAI embedding model: {azure_deployment}")
        
        self.client = AzureOpenAI(
            api_key=azure_key,
            api_version=api_version,
            azure_endpoint=azure_endpoint
        )

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for text(s).
        
        Args:
            texts: Single text string or list of texts
            
        Returns:
            Numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            response = self.client.embeddings.create(
                model=self.azure_deployment,
                input=texts
            )
            
            # Extract embeddings from response
            embeddings = [item.embedding for item in response.data]
            return np.array(embeddings)
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a single query.
        
        Args:
            query: Query text
            
        Returns:
            Numpy array embedding
        """
        return self.embed(query)[0]

    def embed_documents(self, documents: List[str], batch_size: int = 100) -> np.ndarray:
        """
        Generate embeddings for multiple documents, in batches to stay under
        the API's per-request input limits.

        Args:
            documents: List of document texts
            batch_size: Texts per API request

        Returns:
            Numpy array of embeddings
        """
        if not documents:
            return np.empty((0, self.dimension))
        batches = [
            self.embed(documents[i:i + batch_size])
            for i in range(0, len(documents), batch_size)
        ]
        return np.vstack(batches)

    @property
    def dimension(self) -> int:
        """Get embedding dimension (1536 for text-embedding-ada-002)."""
        return 1536
