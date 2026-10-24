"""Checks for hybrid retrieval and the Phase 0 ingestion fixes. Run: pytest -q"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.vector_store import FAISSVectorStore
from rag.retriever import HybridRetriever
from rag.chunking import DocumentChunker
from agents.ingestion_agent import IngestionAgent
from utils.news_client import NewsClient
from utils.predictive_model import PredictiveEngine

DIM = 6


def _vec(*weights):
    v = np.zeros(DIM, dtype=np.float32)
    v[: len(weights)] = weights
    return v


class FakeEmbedder:
    """Returns a fixed vector for the query; no API calls."""

    def __init__(self, query_vec):
        self.query_vec = query_vec

    def embed_query(self, query):
        return self.query_vec


def _retriever(docs, vectors, query_vec):
    store = FAISSVectorStore(dimension=DIM)
    store.add_documents(np.array(vectors), docs)
    retriever = HybridRetriever(store, FakeEmbedder(query_vec))
    retriever.index_documents(docs)
    return retriever


def test_hybrid_keeps_semantic_only_hit_and_uses_corpus_indices():
    docs = [
        {"text": "revenue grew strongly in fiscal 2025", "id": 0},
        {"text": "the board approved a dividend", "id": 1},
        {"text": "litigation remains ongoing in europe", "id": 2},
        # No word overlap with the query: only semantic search can find it
        {"text": "sales increased across all segments", "id": 3},
        {"text": "headcount was flat year over year", "id": 4},
        {"text": "new office lease signed in austin", "id": 5},
    ]
    vectors = [
        _vec(0.6, 0.8),       # 0: somewhat close to query
        _vec(0, 0, 1),
        _vec(0, 0, 0, 1),
        _vec(0, 1),           # 3: closest to query
        _vec(0, 0, 0, 0, 1),
        _vec(0, 0, 0, 0, 0, 1),
    ]
    retriever = _retriever(docs, vectors, query_vec=_vec(0, 1))

    results = retriever.retrieve("revenue", k=2)
    ids = [doc["id"] for doc, _ in results]

    # Doc 0 is first in BM25 and second in semantic; doc 3 is semantic-only
    assert ids == [0, 3]


def test_rrf_scores_union_of_lists():
    scores = HybridRetriever._rrf([([1, 2], 1.0), ([3, 1], 1.0)], k=60)
    assert set(scores) == {1, 2, 3}
    assert scores[1] == 1 / 61 + 1 / 62
    assert max(scores, key=scores.get) == 1


def test_sections_skip_toc_and_cross_references():
    business = "Our business designs chips. " * 20
    risks = "Competition is intense. " * 20
    mdna = "Revenue rose. " * 20
    text = "\n".join([
        "Table of Contents", "Item 1.", "Business", "4", "Item 1A.", "Risk Factors", "13",
        "Item 1. Business", business,
        "Refer to “Item 1A. Risk Factors” for details.",  # cross-ref mid-line
        "Item 1A. Risk Factors", risks,
        "Item 1B. Unresolved Staff Comments", "None.",
        "Item 2. Properties", "Santa Clara.",
        "Item 7. Management’s Discussion and Analysis", mdna,
        "Read with “Item 1A. Risk Factors,” our financial statements. " * 50,
        "Item 8. Financial Statements",
    ])
    sections = DocumentChunker().extract_sections_from_10k(text)
    assert sections["Business"].startswith("Our business") and "Refer to" in sections["Business"]
    assert sections["Risk Factors"].startswith("Competition") and "Revenue" not in sections["Risk Factors"]
    assert sections["MD&A"].startswith("and Analysis")


def test_sections_fall_back_to_unanchored_headers():
    body = "Our business designs chips. " * 20
    text = f"TABLE OF CONTENTS Item 1. Business 4 Item 1A. Risk Factors 12 Item 1. Business {body} Item 2."
    assert "designs chips" in DocumentChunker().extract_sections_from_10k(text)["Business"]


def test_select_filings_keeps_10k_despite_many_8ks():
    filings = [{"form_type": "8-K", "n": i} for i in range(15)]
    filings += [{"form_type": "10-Q", "n": 15}, {"form_type": "10-K", "n": 16}]
    selected = IngestionAgent._select_filings(filings, limit=10)
    forms = [f["form_type"] for f in selected]
    assert len(selected) == 10 and "10-K" in forms and "10-Q" in forms


def test_parse_yahoo_item_new_and_old_formats():
    new = NewsClient._parse_yahoo_item({
        "id": "x",
        "content": {
            "title": "NVDA beats estimates",
            "summary": "Revenue up 20%",
            "pubDate": "2026-10-01T12:00:00Z",
            "provider": {"displayName": "Reuters"},
            "canonicalUrl": {"url": "https://example.com/a"},
        },
    })
    assert new == {
        "title": "NVDA beats estimates", "date": "2026-10-01", "source": "Reuters",
        "url": "https://example.com/a", "summary": "Revenue up 20%",
    }
    old = NewsClient._parse_yahoo_item({
        "title": "Old style", "publisher": "Yahoo", "link": "u", "providerPublishTime": 0,
    })
    assert old["title"] == "Old style" and old["summary"] == "Old style - Yahoo"


def test_mentions_whole_words_and_case():
    m = PredictiveEngine._mentions
    assert m("the ceo resigned on monday", ["CEO"])
    assert m("company acquired a startup", ["acquire"])
    assert not m("tech sector rallied", ["sec"])
    assert not m("supply chain risk", ["up"])


class ReverseReranker:
    """Scores later candidates higher, to prove rerank order wins."""

    def score(self, query, passages):
        return list(range(len(passages)))


def test_weights_ablate_retrievers_and_reranker_reorders():
    docs = [{"text": "revenue grew", "id": 0}, {"text": "sales increased", "id": 1},
            {"text": "board meeting", "id": 2}]
    vectors = [_vec(1), _vec(0, 1), _vec(0, 0, 1)]
    retriever = _retriever(docs, vectors, query_vec=_vec(0, 1))

    assert [d["id"] for d, _ in retriever.retrieve("revenue", k=3, weights=(1.0, 0.0))] == [0]
    assert [d["id"] for d, _ in retriever.retrieve("revenue", k=1, weights=(0.0, 1.0))] == [1]

    retriever.reranker = ReverseReranker()
    plain = [d["id"] for d, _ in retriever.retrieve("revenue", k=3, rerank=False)]
    reranked = [d["id"] for d, _ in retriever.retrieve("revenue", k=3)]
    assert reranked == plain[::-1]


def test_contextual_header_and_indexed_text():
    from rag.chunking import contextual_header, indexed_text
    meta = {"ticker": "NVDA", "form_type": "10-K", "filing_date": "2026-02-25", "section": "Risk Factors"}
    assert contextual_header(meta) == "NVDA 10-K (filed 2026-02-25) · Risk Factors"
    assert indexed_text({"text": "Competition is intense.", "context": "NVDA 10-K"}) == \
        "NVDA 10-K\nCompetition is intense."
