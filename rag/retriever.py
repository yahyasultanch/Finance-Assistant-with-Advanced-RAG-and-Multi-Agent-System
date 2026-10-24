"""Hybrid retriever: BM25 + semantic search, fused with RRF, optionally reranked."""
from rank_bm25 import BM25Okapi
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
import re

from rag.chunking import indexed_text

logger = logging.getLogger(__name__)


def tokenize(text: str) -> List[str]:
    """Lowercase word tokens, so "risks," and "risks" match."""
    return re.findall(r"[a-z0-9]+", text.lower())


class HybridRetriever:
    """Hybrid retriever combining BM25 and semantic search."""

    def __init__(
        self,
        vector_store,
        embedding_model,
        bm25_weight: float = 0.3,
        semantic_weight: float = 0.7,
        reranker=None
    ):
        """
        Initialize hybrid retriever.

        Args:
            vector_store: FAISSVectorStore instance
            embedding_model: EmbeddingModel instance
            bm25_weight: Weight for BM25 ranks in fusion
            semantic_weight: Weight for semantic ranks in fusion
            reranker: Optional CrossEncoderReranker applied to fused candidates
        """
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.bm25_weight = bm25_weight
        self.semantic_weight = semantic_weight
        self.reranker = reranker
        self.bm25 = None

    def index_documents(self, documents: List[Dict[str, Any]]):
        """
        Index documents for BM25 retrieval (header + text, like the embeddings).

        Args:
            documents: List of document dicts with 'text' (and optional 'context')
        """
        self.bm25 = BM25Okapi([tokenize(indexed_text(doc)) for doc in documents])
        logger.info(f"Indexed {len(documents)} documents for BM25")

    def retrieve(
        self,
        query: str,
        k: int = 8,
        candidates: int = 20,
        filters: Optional[Dict[str, List[str]]] = None,
        weights: Optional[Tuple[float, float]] = None,
        rerank: Optional[bool] = None
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Retrieve relevant documents using hybrid approach.

        Args:
            query: Query text
            k: Number of final results
            candidates: Number of candidates from each retriever
            filters: Optional metadata filters, e.g. {"form_type": ["10-K"],
                "section": ["Risk Factors"]}; values match case-insensitively
            weights: (bm25, semantic) fusion weights; a 0 weight skips that
                retriever (used by the evaluation harness for ablations)
            rerank: Rerank fused candidates with the cross-encoder; defaults
                to True when a reranker is configured

        Returns:
            List of (document, score) tuples
        """
        if not self.bm25 or self.vector_store.size == 0:
            logger.warning("Retriever not initialized properly")
            return []
        bm25_weight, semantic_weight = weights or (self.bm25_weight, self.semantic_weight)
        allowed = self._allowed_indices(filters)
        documents = self.vector_store.documents

        # 1. BM25 candidates (skip zero scores: no query term matched)
        bm25_ranked = []
        if bm25_weight:
            bm25_scores = self.bm25.get_scores(tokenize(query))
            bm25_ranked = [
                int(idx) for idx in np.argsort(bm25_scores)[::-1]
                if bm25_scores[idx] > 0 and idx < len(documents)
                and (allowed is None or idx in allowed)
            ][:candidates]

        # 2. Semantic candidates, as corpus indices. With filters, search the
        # whole (small, flat) index and filter after.
        # ponytail: post-filtering a flat index, move to a store with native
        # metadata filtering if indexes grow past ~100k chunks
        semantic_ranked = []
        if semantic_weight:
            query_embedding = self.embedding_model.embed_query(query)
            semantic_k = self.vector_store.size if allowed is not None else candidates
            semantic_ranked = [
                idx for idx, _ in self.vector_store.search_ids(query_embedding, k=semantic_k)
                if allowed is None or idx in allowed
            ][:candidates]

        # 3. Weighted Reciprocal Rank Fusion over the union of both lists
        fused = self._rrf([(bm25_ranked, bm25_weight), (semantic_ranked, semantic_weight)])
        ranked = sorted(fused, key=fused.get, reverse=True)

        # 4. Cross-encoder rerank: reads query and passage together, so it
        # judges relevance better than either first-stage retriever
        if (self.reranker is not None if rerank is None else rerank) and ranked:
            scores = self.reranker.score(query, [indexed_text(documents[i]) for i in ranked])
            fused = dict(zip(ranked, scores))
            ranked = sorted(fused, key=fused.get, reverse=True)

        results = [(documents[idx], float(fused[idx])) for idx in ranked[:k]]
        logger.info(f"Retrieved {len(results)} documents for query: {query[:50]}...")
        return results

    def _allowed_indices(self, filters: Optional[Dict[str, List[str]]]) -> Optional[set]:
        """Indices of documents whose metadata matches every filter, or None if no filters."""
        wanted = {
            key: {str(v).lower() for v in values}
            for key, values in (filters or {}).items() if values
        }
        if not wanted:
            return None
        return {
            i for i, doc in enumerate(self.vector_store.documents)
            if all(str(doc.get("metadata", {}).get(key, "")).lower() in values
                   for key, values in wanted.items())
        }

    @staticmethod
    def _rrf(
        ranked_lists: List[Tuple[List[int], float]],
        k: int = 60
    ) -> Dict[int, float]:
        """
        Weighted Reciprocal Rank Fusion: score(d) = sum(w / (k + rank)).

        Args:
            ranked_lists: List of (doc indices best-first, weight) pairs
            k: RRF damping constant (60 is the standard value)

        Returns:
            Dict mapping doc index to fused score
        """
        scores: Dict[int, float] = {}
        for ranked, weight in ranked_lists:
            for rank, idx in enumerate(ranked, 1):
                scores[idx] = scores.get(idx, 0.0) + weight / (k + rank)
        return scores
