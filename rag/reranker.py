"""Cross-encoder reranker (runs locally via transformers)."""
import logging
from typing import List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Scores (query, passage) pairs jointly. First-stage retrievers embed query
    and passage separately (or just count words); a cross-encoder attends
    across both, which ranks better but is too slow for a whole corpus, so it
    only sees the ~40 fused candidates.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", batch_size: int = 16):
        logger.info(f"Loading reranker: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).eval()
        self.batch_size = batch_size

    @torch.no_grad()
    def score(self, query: str, passages: List[str]) -> List[float]:
        """Relevance logits, higher is more relevant."""
        scores: List[float] = []
        for start in range(0, len(passages), self.batch_size):
            batch = passages[start:start + self.batch_size]
            encoded = self.tokenizer(
                [query] * len(batch), batch,
                truncation="only_second", max_length=512, padding=True, return_tensors="pt",
            )
            scores.extend(self.model(**encoded).logits[:, 0].tolist())
        return scores
