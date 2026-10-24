"""Orchestrator: owns models, data and tools, and runs the LangGraph graphs."""
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import sqlite3
import uuid

import numpy as np

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.sqlite import SqliteStore
from langgraph.types import Command

from agents import memory as mem
from agents.graph import build_brief_graph, build_chat_graph
from agents.ingestion_agent import IngestionAgent
from agents.llm import LLM, Tool
from agents.reporter_agent import ReporterAgent
from agents.sentiment_agent import SentimentAgent
from agents.tools import ResearchContext, build_tools
from rag.reranker import CrossEncoderReranker
from utils.prices import PriceClient
from utils.rating_engine import RatingEngine
from utils.predictive_model import PredictiveEngine
from utils.xbrl import key_metrics

logger = logging.getLogger(__name__)


def _plain(obj: Any) -> Any:
    """Convert numpy scalars (from the scoring models) so the checkpointer can serialize state."""
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_plain(v) for v in obj)
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


class Orchestrator:
    """Main orchestrator coordinating all agents."""

    def __init__(
        self,
        sec_user_agent: str,
        data_dir: Path,
        index_dir: Path,
        reports_dir: Path,
        embedding_model_name: str,
        azure_endpoint: str,
        azure_key: str,
        azure_deployment: str,
        azure_embedding_deployment: str,
        azure_embedding_key: str,
        api_version: str = "2024-12-01-preview",
        finbert_model: str = "ProsusAI/finbert",
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        top_k: int = 8,
        reranker_model: str = ""
    ):
        """
        Initialize orchestrator.

        Args:
            sec_user_agent: User agent for SEC requests
            data_dir: Data directory (also holds the graph checkpoint database)
            index_dir: Index directory
            reports_dir: Reports directory
            embedding_model_name: Embedding model name (deprecated, kept for compatibility)
            azure_endpoint: Azure OpenAI endpoint
            azure_key: Azure OpenAI API key
            azure_deployment: Azure OpenAI deployment name
            azure_embedding_deployment: Azure embedding deployment name
            azure_embedding_key: Azure embedding API key
            api_version: Azure OpenAI API version
            finbert_model: FinBERT model name
            chunk_size: Chunk size
            chunk_overlap: Chunk overlap
            top_k: Top K retrieval (deprecated: agents choose k per search)
            reranker_model: Cross-encoder for reranking retrieval candidates; empty disables
        """
        self.ingestion_agent = IngestionAgent(
            sec_user_agent,
            data_dir,
            index_dir,
            azure_endpoint,
            azure_embedding_key,
            azure_embedding_deployment,
            api_version,
            chunk_size,
            chunk_overlap,
            reranker=CrossEncoderReranker(reranker_model) if reranker_model else None
        )
        self.sentiment_agent = SentimentAgent(finbert_model)
        self.reporter_agent = ReporterAgent(reports_dir)
        self.price_client = PriceClient()
        self.rating_engine = RatingEngine()
        self.predictive_engine = PredictiveEngine()
        self.llm = LLM(azure_endpoint, azure_key, azure_deployment, api_version)

        self._contexts: Dict[str, ResearchContext] = {}
        self._tools: Dict[str, Dict[str, Tool]] = {}
        self._usage: Dict[str, Dict[str, int]] = {}  # thread_id -> LLM usage

        # Short-term memory: every brief run (resumable, reviewable) and chat thread
        data_dir.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(data_dir / "checkpoints.sqlite"), check_same_thread=False)
        self.checkpointer = SqliteSaver(connection)
        # Long-term memory across runs: earlier briefs and the prediction ledger
        store_connection = sqlite3.connect(str(data_dir / "memory.sqlite"), check_same_thread=False,
                                           isolation_level=None)
        self.store = SqliteStore(store_connection)
        self.store.setup()
        self.brief_graph = build_brief_graph(self, self.checkpointer, self.store)
        self.chat_graph = build_chat_graph(self, self.checkpointer, self.store)

    # Data and tools, looked up by graph nodes

    def load_context(
        self,
        ticker: str,
        use_cache: bool = True,
        days_back: int = 365,
        max_cache_age_days: float = 7
    ) -> ResearchContext:
        """Ingest (or load the cached index for) a ticker and build its research context."""
        cached = self.ingestion_agent.load_index(ticker, max_cache_age_days) if use_cache else None
        if cached:
            logger.info("Using cached index")
            vector_store, retriever = cached
            news_items = self._news_from_index(vector_store)
            company_info = self.price_client.get_company_info(ticker)
        else:
            logger.info("Performing full ingestion")
            ingestion_results = self.ingestion_agent.ingest_ticker(ticker, days_back=days_back)
            loaded = self.ingestion_agent.load_index(ticker)
            if not loaded:
                raise ValueError(f"No SEC filings or news found for {ticker}")
            vector_store, retriever = loaded
            news_items = ingestion_results["news"]
            company_info = ingestion_results["price_data"]

        sec_client = self.ingestion_agent.sec_client
        cik = sec_client.get_cik(ticker) or ""
        facts = sec_client.get_company_facts(cik) if cik else None
        previous = self._contexts.get(ticker)

        ctx = ResearchContext(
            ticker=ticker,
            retriever=retriever,
            vector_store=vector_store,
            news_items=news_items,
            company_info=company_info,
            price_metrics=self.price_client.get_key_metrics(ticker),
            metrics=key_metrics(facts) if facts else [],
            cik=cik,
            report=previous.report if previous else "",
        )
        self._contexts[ticker] = ctx
        self._tools[ticker] = build_tools(ctx, sec_client)
        return ctx

    def context(self, ticker: str) -> ResearchContext:
        """Research context for a ticker, loading the cached index if needed."""
        return self._contexts.get(ticker) or self.load_context(ticker)

    def tools(self, ticker: str) -> Dict[str, Tool]:
        if ticker not in self._tools:
            self.context(ticker)
        return self._tools[ticker]

    @staticmethod
    def _news_from_index(vector_store) -> List[Dict[str, Any]]:
        """
        Rebuild news items from cached news chunks (news and 8-K events).

        Indexes built before title/summary were stored in metadata fall back
        to the chunk text, which is "title. summary".
        """
        news_items = []
        for doc in vector_store.documents:
            meta = doc.get("metadata", {})
            if meta.get("form_type") != "NEWS" and meta.get("source") != "SEC 8-K":
                continue
            news_items.append({
                "title": meta.get("title", doc.get("text", "")),
                "summary": meta.get("summary", ""),
                "date": meta.get("filing_date", ""),
                "source": meta.get("source", ""),
                "url": meta.get("url", ""),
            })
        return news_items

    # Steps called by graph nodes

    def score(self, ticker: str, summaries: Dict[str, Any]) -> Dict[str, Any]:
        """Sentiment, rating and predictions (deterministic models, no LLM)."""
        ctx = self.context(ticker)
        sentiment_results = self.sentiment_agent.analyze_news(ctx.news_items)
        risk_texts = [c["text"] for c in summaries["risks"]["contexts"][:10]]
        if risk_texts:
            sentiment_results["risk_sentiment"] = self.sentiment_agent.analyze_risk_factors(risk_texts)

        price_data = {
            "current_price": ctx.price_metrics.get("current_price", 0),
            "ytd_return": ctx.price_metrics.get("ytd_return", 0),
            "high_52w": ctx.price_metrics.get("high_52w", 0),
            "low_52w": ctx.price_metrics.get("low_52w", 0),
            "volatility": ctx.price_metrics.get("volatility", 30)
        }
        rating_results = self.rating_engine.calculate_rating(
            ticker=ticker,
            price_data=price_data,
            company_info=ctx.company_info,
            sentiment_results=sentiment_results,
            summaries=summaries
        )
        prediction_results = self.predictive_engine.generate_predictions(
            ticker=ticker,
            price_data=price_data,
            company_info=ctx.company_info,
            sentiment_results=sentiment_results,
            news_items=ctx.news_items,
            rating_results=rating_results,
            summaries=summaries
        )
        return _plain({"sentiment": sentiment_results, "rating": rating_results,
                       "prediction": prediction_results})

    def price_closes(self, ticker: str, period: str = "2y") -> List[tuple]:
        """(iso date, close) pairs, oldest first, for scoring the prediction ledger."""
        df = self.price_client.get_price_data(ticker, period=period)
        if df is None or df.empty:
            return []
        return [(idx.strftime("%Y-%m-%d"), float(close)) for idx, close in df["Close"].items()]

    def track_record(self, ticker: str) -> Dict[str, Any]:
        """Scored ledger, summary and earlier briefs for a ticker (for the UI)."""
        scored = mem.score_entries(mem.ledger(self.store, ticker), self.price_closes(ticker))
        return {"entries": scored, "summary": mem.summarize(scored),
                "briefs": mem.recent_briefs(self.store, ticker, limit=20)}

    def finish_report(self, state: Dict[str, Any], track_record: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Assemble and save the brief; returns the result dict the UI renders."""
        ticker = state["ticker"]
        ctx = self.context(ticker)
        scores = state["scores"]

        report_content = self.reporter_agent.generate_brief(
            ticker,
            ctx.company_info,
            ctx.price_metrics,
            state["summaries"],
            scores["sentiment"],
            ctx.news_items,
            scores["rating"],
            scores["prediction"],
            verification=state["verification"],
            review_note=state.get("review_note", ""),
            track_record=track_record
        )
        report_path = self.reporter_agent.save_brief(
            ticker, report_content, format=state.get("save_format", "markdown")
        )
        ctx.report = report_content

        # Price history for charts
        price_df = self.price_client.get_price_data(ticker, period="1y")
        price_history = []
        if price_df is not None and not price_df.empty:
            for idx, row in price_df.iterrows():
                price_history.append({
                    'date': idx.strftime('%Y-%m-%d'),
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close']),
                    'volume': int(row['Volume'])
                })

        return _plain({
            "ticker": ticker,
            "report_path": report_path,
            "report_content": report_content,
            "summaries": state["summaries"],
            "sentiment": scores["sentiment"],
            "rating": scores["rating"],
            "prediction_results": scores["prediction"],
            "company_info": ctx.company_info,
            "price_metrics": ctx.price_metrics,
            "price_data": {
                "current_price": ctx.price_metrics.get("current_price", 0),
                "history": price_history
            },
            "news_data": ctx.news_items,
            "plan": state.get("plan", []),
            "verification": state["verification"],
            "trace": state.get("trace", []),
        })

    # Public API

    def generate_brief(
        self,
        ticker: str,
        use_cache: bool = True,
        days_back: int = 365,
        save_format: str = "markdown",
        review: bool = False
    ) -> Dict[str, Any]:
        """
        Generate a complete analyst brief by running the agent graph.

        Args:
            ticker: Stock ticker symbol
            use_cache: Whether to use cached index if available
            days_back: Days to look back for filings
            save_format: Output format (markdown, html)
            review: Pause for human approval before the report is written

        Returns:
            Result dict with status "done" (plus report data), "needs_review"
            (plus review payload; call resume_brief) or "rejected"
        """
        thread_id = f"brief-{ticker}-{uuid.uuid4().hex[:8]}"
        state = {"ticker": ticker, "use_cache": use_cache, "days_back": days_back,
                 "save_format": save_format, "review": review}
        return self._run_brief(state, thread_id)

    def resume_brief(self, thread_id: str, approved: bool = True, note: str = "") -> Dict[str, Any]:
        """Resume a brief paused for review with the reviewer's decision."""
        return self._run_brief(Command(resume={"approved": approved, "note": note}), thread_id)

    def _run_brief(self, graph_input: Any, thread_id: str) -> Dict[str, Any]:
        usage_before = dict(self.llm.usage)
        config = {"configurable": {"thread_id": thread_id}}
        logger.info(f"Running brief graph, thread {thread_id}")
        out = self.brief_graph.invoke(graph_input, config)
        # Accumulate per thread, so a resumed run reports the whole brief's usage
        usage = self._usage.setdefault(thread_id, dict.fromkeys(self.llm.usage, 0))
        for k in usage:
            usage[k] += self.llm.usage[k] - usage_before.get(k, 0)

        if out.get("__interrupt__"):
            return {"status": "needs_review", "thread_id": thread_id,
                    "review": out["__interrupt__"][0].value, "usage": usage}
        if out.get("result"):
            return {**out["result"], "status": "done", "thread_id": thread_id, "usage": usage}
        return {"status": "rejected", "thread_id": thread_id, "usage": usage}

    def chat(self, ticker: str, question: str, thread_id: str) -> Dict[str, Any]:
        """Answer a follow-up question; history persists per thread_id."""
        config = {"configurable": {"thread_id": thread_id}}
        out = self.chat_graph.invoke(
            {"ticker": ticker, "messages": [{"role": "user", "content": question}]}, config
        )
        return out["messages"][-1]

    def chat_history(self, thread_id: str) -> List[Dict[str, Any]]:
        state = self.chat_graph.get_state({"configurable": {"thread_id": thread_id}})
        return state.values.get("messages", []) if state else []
