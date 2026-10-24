"""Retrieval evaluation (ablation) and optional end-to-end brief evaluation.

    python eval/run_eval.py NVDA AAPL              # retrieval ablation
    python eval/run_eval.py NVDA AAPL --briefs     # plus a full brief per ticker

Relevance uses silver labels: a chunk counts as relevant to a question if it
comes from the expected form/section AND matches the question's keyword
pattern. That is cheaper than hand labeling and good enough to compare
retrieval setups on the same corpus, but it is not a human judgment.

Results are written to eval/results.md.
"""
import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from agents.ingestion_agent import IngestionAgent  # noqa: E402
from rag.reranker import CrossEncoderReranker  # noqa: E402
from rag.retriever import HybridRetriever  # noqa: E402
from rag.vector_store import FAISSVectorStore  # noqa: E402

SEC = {"10-K", "10-Q", "8-K"}
RISKS = {"Risk Factors"}
RESULTS = {"MD&A", "Exhibit 99.1", "Exhibit 99.2", "Financial Statements"}

# (question, forms, sections or None for any, keyword pattern)
QUESTIONS = [
    ("What does the company do and what are its main products?",
     {"10-K"}, {"Business"}, r"products?|services?|platform"),
    ("How does the company describe competition in its markets?",
     {"10-K", "10-Q"}, {"Business"} | RISKS, r"compet"),
    ("What supply chain or manufacturing risks does the company face?",
     {"10-K", "10-Q"}, RISKS, r"supply|supplier|manufactur"),
    ("What regulatory or legal risks could affect the business?",
     {"10-K", "10-Q"}, RISKS, r"regulat|legal|litigation|government"),
    ("What risks come from international operations, trade restrictions or export controls?",
     {"10-K", "10-Q"}, RISKS, r"export|tariff|international|trade"),
    ("What were revenue and net income in the most recent quarter?",
     SEC, RESULTS, r"revenue|net sales"),
    ("How did gross margin change and why?",
     SEC, RESULTS, r"gross margin"),
    ("What is the company's liquidity and capital resources position?",
     {"10-K", "10-Q"}, {"MD&A"}, r"liquidity|capital resources|cash and cash equivalents"),
    ("Does the company return capital through share repurchases or dividends?",
     SEC, None, r"repurchase|dividend"),
    ("What outlook or guidance did management give for the next quarter?",
     SEC, RESULTS, r"outlook|expect|guidance"),
    ("How much does the company invest in research and development?",
     {"10-K", "10-Q"}, {"MD&A", "Business", "Financial Statements"}, r"research and development"),
    ("What cybersecurity risks does the company describe?",
     {"10-K", "10-Q"}, RISKS | {"Business"}, r"cyber|security breach|attack"),
]

# Setup name -> (use contextual headers, (bm25, semantic) weights, rerank)
SETUPS = {
    "BM25 only": (False, (1.0, 0.0), False),
    "Semantic only": (False, (0.0, 1.0), False),
    "Hybrid (RRF)": (False, (0.3, 0.7), False),
    "Hybrid + contextual headers": (True, (0.3, 0.7), False),
    "Hybrid + headers + reranker": (True, (0.3, 0.7), True),
}
RERANKER = config.RERANKER_MODEL or "cross-encoder/ms-marco-MiniLM-L-6-v2"


def is_relevant(doc, forms, sections, pattern) -> bool:
    meta = doc.get("metadata", {})
    return (meta.get("form_type") in forms
            and (sections is None or meta.get("section") in sections)
            and re.search(pattern, doc["text"], re.IGNORECASE) is not None)


def metrics(flags):
    """hit@5, MRR@10, precision@5 for one ranked list of relevance flags."""
    first = next((i for i, rel in enumerate(flags[:10], 1) if rel), None)
    return {"hit@5": float(any(flags[:5])), "MRR@10": 1 / first if first else 0.0,
            "P@5": sum(flags[:5]) / 5}


def retrievers_for(agent: IngestionAgent, ticker: str, reranker):
    """The v3 index (headers) and a raw-text twin built from the same chunks."""
    loaded = agent.load_index(ticker)
    if loaded is None:
        print(f"  {ticker}: no current index, ingesting (SEC + embeddings)...")
        agent.ingest_ticker(ticker)
        loaded = agent.load_index(ticker)
    with_headers, retriever = loaded
    retriever.reranker = reranker

    raw_docs = [{k: v for k, v in d.items() if k != "context"} for d in with_headers.documents]
    cache = agent._index_path(ticker).with_name(agent._index_path(ticker).name + "_raw.npy")
    if cache.exists():
        embeddings = np.load(cache)
    else:
        print(f"  {ticker}: embedding raw chunks for the no-header baseline...")
        embeddings = agent.embedding_model.embed_documents([d["text"] for d in raw_docs])
        np.save(cache, embeddings)
    raw_store = FAISSVectorStore(dimension=agent.embedding_model.dimension)
    raw_store.add_documents(embeddings, raw_docs)
    raw = HybridRetriever(raw_store, agent.embedding_model, reranker=reranker)
    raw.index_documents(raw_docs)
    return {True: retriever, False: raw}


def retrieval_eval(tickers):
    agent = IngestionAgent(
        config.SEC_USER_AGENT, config.DATA_DIR, config.INDEX_DIR,
        config.AZURE_OPENAI_ENDPOINT, config.AZURE_EMBEDDING_KEY,
        config.AZURE_EMBEDDING_DEPLOYMENT, config.AZURE_OPENAI_API_VERSION,
    )
    reranker = CrossEncoderReranker(RERANKER)
    scores = {name: [] for name in SETUPS}
    answerable = 0
    for ticker in tickers:
        retrievers = retrievers_for(agent, ticker, reranker)
        corpus = retrievers[True].vector_store.documents
        for question, forms, sections, pattern in QUESTIONS:
            # Skip questions this ticker's corpus has no relevant chunk for
            if not any(is_relevant(d, forms, sections, pattern) for d in corpus):
                continue
            answerable += 1
            for name, (headers, weights, rerank) in SETUPS.items():
                results = retrievers[headers].retrieve(question, k=10, weights=weights, rerank=rerank)
                flags = [is_relevant(doc, forms, sections, pattern) for doc, _ in results]
                scores[name].append(metrics(flags))
    table = {name: {m: float(np.mean([r[m] for r in rows])) for m in ("hit@5", "MRR@10", "P@5")}
             for name, rows in scores.items()}
    return table, answerable


def brief_eval(tickers):
    from agents.orchestrator import Orchestrator
    orc = Orchestrator(
        sec_user_agent=config.SEC_USER_AGENT, data_dir=config.DATA_DIR, index_dir=config.INDEX_DIR,
        reports_dir=config.REPORTS_DIR, embedding_model_name="",
        azure_endpoint=config.AZURE_OPENAI_ENDPOINT, azure_key=config.AZURE_OPENAI_KEY,
        azure_deployment=config.AZURE_OPENAI_DEPLOYMENT,
        azure_embedding_deployment=config.AZURE_EMBEDDING_DEPLOYMENT,
        azure_embedding_key=config.AZURE_EMBEDDING_KEY, api_version=config.AZURE_OPENAI_API_VERSION,
        finbert_model=config.FINBERT_MODEL, reranker_model=config.RERANKER_MODEL,
    )
    rows = []
    for ticker in tickers:
        start = time.time()
        result = orc.generate_brief(ticker)
        v, usage = result["verification"], result["usage"]
        rows.append({
            "ticker": ticker, "claims": v["total"], "verified": v["verified"],
            "first_pass": v["first_pass_coverage"], "coverage": v["coverage"],
            "revisions": v["revisions"], "calls": usage["calls"],
            "tokens": usage["prompt_tokens"] + usage["completion_tokens"],
            "seconds": time.time() - start,
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tickers", nargs="+")
    parser.add_argument("--briefs", action="store_true", help="also run a full brief per ticker")
    args = parser.parse_args()
    tickers = [t.upper() for t in args.tickers]

    table, answerable = retrieval_eval(tickers)
    lines = [
        "# Evaluation results", "",
        f"Tickers: {', '.join(tickers)} · {answerable} answerable question-ticker pairs "
        f"({len(QUESTIONS)} questions) · run {time.strftime('%Y-%m-%d')}", "",
        "## Retrieval ablation (silver labels, top 10)", "",
        "| Setup | hit@5 | MRR@10 | P@5 |", "|-------|-------|--------|-----|",
    ]
    lines += [f"| {name}{f' ({RERANKER})' if 'reranker' in name else ''} | {m['hit@5']:.2f} | "
              f"{m['MRR@10']:.2f} | {m['P@5']:.2f} |" for name, m in table.items()]

    if args.briefs:
        rows = brief_eval(tickers)
        lines += [
            "", "## End-to-end briefs", "",
            "| Ticker | Claims | Verified | First draft | Final | Revisions | LLM calls | Tokens | Seconds |",
            "|--------|--------|----------|-------------|-------|-----------|-----------|--------|---------|",
        ]
        lines += [
            f"| {r['ticker']} | {r['claims']} | {r['verified']} | {r['first_pass']:.0%} | "
            f"{r['coverage']:.0%} | {r['revisions']} | {r['calls']} | {r['tokens']:,} | {r['seconds']:.0f} |"
            for r in rows
        ]

    report = "\n".join(lines) + "\n"
    (ROOT / "eval" / "results.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
