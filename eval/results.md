# Evaluation results

Tickers: NVDA, AAPL · 24 answerable question-ticker pairs (12 questions) · run 2026-10-24

## Retrieval ablation (silver labels, top 10)

| Setup | hit@5 | MRR@10 | P@5 |
|-------|-------|--------|-----|
| BM25 only | 0.88 | 0.78 | 0.62 |
| Semantic only | 0.96 | 0.93 | 0.85 |
| Hybrid (RRF) | 0.96 | 0.91 | 0.83 |
| Hybrid + contextual headers | 0.96 | 0.94 | 0.89 |
| Hybrid + headers + reranker (cross-encoder/ms-marco-MiniLM-L-6-v2) | 0.96 | 0.90 | 0.82 |

## End-to-end briefs

| Ticker | Claims | Verified | First draft | Final | Revisions | LLM calls | Tokens | Seconds |
|--------|--------|----------|-------------|-------|-----------|-----------|--------|---------|
| NVDA | 17 | 15 | 88% | 88% | 0 | 18 | 34,223 | 27 |
| AAPL | 17 | 15 | 88% | 88% | 0 | 17 | 27,876 | 29 |
