"""Retriever with BM25 and semantic reranking."""
from rank_bm25 import BM25Okapi
import numpy as np
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Hybrid retriever combining BM25 and semantic search."""

    def __init__(
        self,
        vector_store,
        embedding_model,
        bm25_weight: float = 0.3,
        semantic_weight: float = 0.7
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            vector_store: FAISSVectorStore instance
            embedding_model: EmbeddingModel instance
            bm25_weight: Weight for BM25 scores
            semantic_weight: Weight for semantic scores
        """
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.bm25_weight = bm25_weight
        self.semantic_weight = semantic_weight
        self.bm25 = None
        self.tokenized_corpus = []

    def index_documents(self, documents: List[Dict[str, Any]]):
        """
        Index documents for BM25 retrieval.
        
        Args:
            documents: List of document dicts with 'text' field
        """
        # Tokenize corpus for BM25
        self.tokenized_corpus = [
            doc['text'].lower().split() for doc in documents
        ]
        
        # Build BM25 index
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        logger.info(f"Indexed {len(documents)} documents for BM25")

    def retrieve(
        self, 
        query: str, 
        k: int = 8,
        bm25_top_k: int = 20
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Retrieve relevant documents using hybrid approach.
        
        Args:
            query: Query text
            k: Number of final results
            bm25_top_k: Number of candidates from BM25
            
        Returns:
            List of (document, score) tuples
        """
        if not self.bm25 or self.vector_store.size == 0:
            logger.warning("Retriever not initialized properly")
            return []
        
        # 1. BM25 retrieval to get candidates
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Get top BM25 candidates
        bm25_top_indices = np.argsort(bm25_scores)[::-1][:bm25_top_k]
        
        # 2. Semantic reranking
        query_embedding = self.embedding_model.embed_query(query)
        semantic_results = self.vector_store.search(query_embedding, k=min(bm25_top_k, self.vector_store.size))
        
        # Create index to doc mapping
        semantic_doc_scores = {
            i: score for i, (doc, score) in enumerate(semantic_results)
        }
        
        # 3. Combine scores
        combined_scores = {}
        
        for idx in bm25_top_indices:
            if idx >= len(self.vector_store.documents):
                continue
            
            # Normalize BM25 score (0-1 range)
            bm25_score = bm25_scores[idx]
            max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
            norm_bm25 = bm25_score / max_bm25
            
            # Get semantic score if available
            semantic_score = semantic_doc_scores.get(idx, 0.0)
            
            # Combine scores
            combined_score = (self.bm25_weight * norm_bm25 + 
                            self.semantic_weight * semantic_score)
            
            combined_scores[idx] = combined_score
        
        # Sort by combined score
        sorted_indices = sorted(
            combined_scores.keys(),
            key=lambda x: combined_scores[x],
            reverse=True
        )[:k]
        
        # Gather results
        results = [
            (self.vector_store.documents[idx], combined_scores[idx])
            for idx in sorted_indices
        ]
        
        logger.info(f"Retrieved {len(results)} documents for query: {query[:50]}...")
        return results

    def retrieve_semantic_only(
        self, 
        query: str, 
        k: int = 8
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Retrieve using only semantic search.
        
        Args:
            query: Query text
            k: Number of results
            
        Returns:
            List of (document, score) tuples
        """
        query_embedding = self.embedding_model.embed_query(query)
        return self.vector_store.search(query_embedding, k=k)

    def retrieve_bm25_only(
        self, 
        query: str, 
        k: int = 8
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Retrieve using only BM25.
        
        Args:
            query: Query text
            k: Number of results
            
        Returns:
            List of (document, score) tuples
        """
        if not self.bm25:
            logger.warning("BM25 not initialized")
            return []
        
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        top_indices = np.argsort(bm25_scores)[::-1][:k]
        
        results = [
            (self.vector_store.documents[idx], float(bm25_scores[idx]))
            for idx in top_indices
            if idx < len(self.vector_store.documents)
        ]
        
        return results
