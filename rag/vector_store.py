"""Vector store using FAISS."""
import faiss
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """FAISS-based vector store for document embeddings."""

    def __init__(self, dimension: int, index_path: Optional[Path] = None):
        """
        Initialize FAISS vector store.
        
        Args:
            dimension: Embedding dimension
            index_path: Path to save/load index
        """
        self.dimension = dimension
        self.index_path = index_path
        self.index = None
        self.documents = []  # Store document metadata
        self.initialize_index()

    def initialize_index(self):
        """Initialize or load FAISS index."""
        if self.index_path and self.index_path.exists():
            self.load()
        else:
            # Use IndexFlatIP for inner product (cosine similarity after normalization)
            self.index = faiss.IndexFlatIP(self.dimension)
            logger.info(f"Initialized new FAISS index with dimension {self.dimension}")

    def add_documents(
        self, 
        embeddings: np.ndarray, 
        documents: List[Dict[str, Any]]
    ):
        """
        Add documents and their embeddings to the index.
        
        Args:
            embeddings: Numpy array of embeddings (N x D)
            documents: List of document metadata dicts
        """
        if len(embeddings) != len(documents):
            raise ValueError("Number of embeddings must match number of documents")
        
        # Ensure embeddings are float32 numpy array
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)
        
        # Convert to float32 and ensure contiguous
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to index
        self.index.add(embeddings)
        self.documents.extend(documents)
        
        logger.info(f"Added {len(embeddings)} documents to index. Total: {self.index.ntotal}")

    def search(
        self, 
        query_embedding: np.ndarray, 
        k: int = 10
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of (document, score) tuples
        """
        return [(self.documents[idx], score) for idx, score in self.search_ids(query_embedding, k)]

    def search_ids(
        self,
        query_embedding: np.ndarray,
        k: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Search for similar documents, returning corpus indices.

        Args:
            query_embedding: Query embedding vector
            k: Number of results to return

        Returns:
            List of (document index, score) tuples, best first
        """
        if self.index.ntotal == 0:
            logger.warning("Index is empty")
            return []

        # Normalize query
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_embedding)

        # Search
        k = min(k, self.index.ntotal)
        distances, indices = self.index.search(query_embedding, k)

        # FAISS pads with -1 when fewer than k results exist
        return [
            (int(idx), float(score))
            for idx, score in zip(indices[0], distances[0])
            if 0 <= idx < len(self.documents)
        ]

    def save(self, path: Optional[Path] = None):
        """
        Save index and documents to disk.
        
        Args:
            path: Path to save (uses self.index_path if not provided)
        """
        save_path = path or self.index_path
        if not save_path:
            raise ValueError("No save path specified")
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_file = str(save_path.with_suffix('.faiss'))
        faiss.write_index(self.index, index_file)
        
        # Save documents metadata
        docs_file = str(save_path.with_suffix('.pkl'))
        with open(docs_file, 'wb') as f:
            pickle.dump(self.documents, f)
        
        logger.info(f"Saved index to {index_file} and metadata to {docs_file}")

    def load(self, path: Optional[Path] = None):
        """
        Load index and documents from disk.
        
        Args:
            path: Path to load (uses self.index_path if not provided)
        """
        load_path = path or self.index_path
        if not load_path:
            raise ValueError("No load path specified")
        
        load_path = Path(load_path)
        
        # Load FAISS index
        index_file = str(load_path.with_suffix('.faiss'))
        if not Path(index_file).exists():
            logger.warning(f"Index file not found: {index_file}")
            return False
        
        self.index = faiss.read_index(index_file)
        
        # Load documents metadata
        docs_file = str(load_path.with_suffix('.pkl'))
        if Path(docs_file).exists():
            with open(docs_file, 'rb') as f:
                self.documents = pickle.load(f)
        
        logger.info(f"Loaded index with {self.index.ntotal} vectors from {index_file}")
        return True

    def clear(self):
        """Clear the index."""
        self.index.reset()
        self.documents = []
        logger.info("Cleared index")

    @property
    def size(self) -> int:
        """Get number of documents in index."""
        return self.index.ntotal if self.index else 0
