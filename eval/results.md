# Evaluation results

Tickers: NVDA, AAPL · 24 answerable question-ticker pairs (12 questions) · run 2026-11-25

## Retrieval ablation (silver labels, top 10)

| Setup | hit@5 | MRR@10 | P@5 |
|-------|-------|--------|-----|
| BM25 only | 0.92 | 0.82 | 0.71 |
| Semantic only | 0.92 | 0.93 | 0.82 |
| Hybrid (RRF) | 0.92 | 0.88 | 0.82 |
| Hybrid + contextual headers | 0.96 | 0.96 | 0.90 |
| Hybrid + headers + reranker (cross-encoder/ms-marco-MiniLM-L-6-v2) | 0.96 | 0.92 | 0.84 |

## End-to-end briefs

| Ticker | Claims | Verified | First draft | Final | Event recall | Revisions | LLM calls | Tokens | Seconds |
|--------|--------|----------|-------------|-------|--------------|-----------|-----------|--------|---------|
| NVDA | 17 | 16 | 75% | 94% | 3/3 | 1 | 27 | 67,625 | 49 |
| AAPL | 18 | 17 | 94% | 94% | 1/1 | 1 | 28 | 65,660 | 51 |

Event recall: recent 8-Ks with a press release (last 90 days) that a verified claim cites.
