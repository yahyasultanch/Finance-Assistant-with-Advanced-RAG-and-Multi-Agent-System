"""
AI Investment Research Assistant

A comprehensive system for generating analyst briefs using:
- RAG (Retrieval-Augmented Generation) with verified citations
- Multi-agent orchestration with LangGraph (planner, tool-calling researchers, verifier)
- Financial sentiment analysis with FinBERT
- SEC EDGAR filing analysis

Usage:
    from agents.orchestrator import Orchestrator
    
    orchestrator = Orchestrator(...)
    result = orchestrator.generate_brief("AAPL")
    print(result['report_content'])

For UI:
    streamlit run app/ui_app.py

"""

__version__ = "1.0.0"
__author__ = "AI Finance Assistant Team"

# Main exports
from agents.orchestrator import Orchestrator
from agents.ingestion_agent import IngestionAgent
from agents.sentiment_agent import SentimentAgent
from agents.reporter_agent import ReporterAgent

__all__ = [
    "Orchestrator",
    "IngestionAgent",
    "SentimentAgent",
    "ReporterAgent",
]
