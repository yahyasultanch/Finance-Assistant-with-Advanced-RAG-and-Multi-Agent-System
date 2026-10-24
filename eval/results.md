# Evaluation results

Tickers: NVDA, AAPL · 24 answerable question-ticker pairs (12 questions) · run 2026-10-24

## Retrieval ablation (silver labels, top 10)

| Setup | hit@5 | MRR@10 | P@5 |
|-------|-------|--------|-----|
| BM25 only | 0.92 | 0.81 | 0.65 |
| Semantic only | 0.92 | 0.92 | 0.83 |
| Hybrid (RRF) | 0.92 | 0.90 | 0.82 |
| Hybrid + contextual headers | 0.96 | 0.94 | 0.89 |
| Hybrid + headers + reranker (cross-encoder/ms-marco-MiniLM-L-6-v2) | 0.96 | 0.90 | 0.82 |

## End-to-end briefs

| Ticker | Claims | Verified | First draft | Final | Revisions | LLM calls | Tokens | Seconds |
|--------|--------|----------|-------------|-------|-----------|-----------|--------|---------|
| NVDA | 15 | 14 | 93% | 93% | 0 | 16 | 26,973 | 32 |
| AAPL | 19 | 17 | 89% | 89% | 0 | 16 | 30,578 | 27 |
