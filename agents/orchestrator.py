"""Orchestrator to coordinate all agents."""
from pathlib import Path
from typing import Dict, Any, Optional
import logging

from agents.ingestion_agent import IngestionAgent
from agents.summarizer_agent import SummarizerAgent
from agents.sentiment_agent import SentimentAgent
from agents.reporter_agent import ReporterAgent
from utils.prices import PriceClient
from utils.rating_engine import RatingEngine
from utils.predictive_model import PredictiveEngine

logger = logging.getLogger(__name__)


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
        top_k: int = 8
    ):
        """
        Initialize orchestrator.
        
        Args:
            sec_user_agent: User agent for SEC requests
            data_dir: Data directory
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
            top_k: Top K retrieval
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
            chunk_overlap
        )
        self.sentiment_agent = SentimentAgent(finbert_model)
        self.reporter_agent = ReporterAgent(reports_dir)
        self.price_client = PriceClient()
        self.rating_engine = RatingEngine()
        self.predictive_engine = PredictiveEngine()
        
        self.azure_endpoint = azure_endpoint
        self.azure_key = azure_key
        self.azure_deployment = azure_deployment
        self.api_version = api_version
        self.top_k = top_k
        
        self.summarizer_agent = None  # Initialized per ticker

    def generate_brief(
        self,
        ticker: str,
        use_cache: bool = True,
        days_back: int = 365,
        save_format: str = "markdown"
    ) -> Dict[str, Any]:
        """
        Generate complete analyst brief for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            use_cache: Whether to use cached index if available
            days_back: Days to look back for filings
            save_format: Output format (markdown, html)
            
        Returns:
            Dict with report path and metadata
        """
        logger.info(f"Starting brief generation for {ticker}")
        
        # Step 1: Ingest data (or load from cache)
        logger.info("Step 1: Data ingestion...")
        
        if use_cache:
            cached = self.ingestion_agent.load_index(ticker)
            if cached:
                logger.info("Using cached index")
                vector_store, retriever = cached
                
                # Still fetch latest price data
                company_info = self.price_client.get_company_info(ticker)
                price_metrics = self.price_client.get_key_metrics(ticker)
                
                # Get news from cache metadata
                news_items = [
                    doc["metadata"] for doc in vector_store.documents
                    if doc.get("metadata", {}).get("form_type") == "NEWS"
                ]
            else:
                logger.info("No cache found, performing full ingestion")
                ingestion_results = self.ingestion_agent.ingest_ticker(
                    ticker, days_back=days_back
                )
                
                # Load the newly created index
                vector_store, retriever = self.ingestion_agent.load_index(ticker)
                company_info = ingestion_results["price_data"]
                price_metrics = self.price_client.get_key_metrics(ticker)
                news_items = ingestion_results["news"]
        else:
            logger.info("Performing fresh ingestion")
            ingestion_results = self.ingestion_agent.ingest_ticker(
                ticker, days_back=days_back
            )
            vector_store, retriever = self.ingestion_agent.load_index(ticker)
            company_info = ingestion_results["price_data"]
            price_metrics = self.price_client.get_key_metrics(ticker)
            news_items = ingestion_results["news"]
        
        # Step 2: Initialize summarizer with retriever
        logger.info("Step 2: Initializing summarizer...")
        self.summarizer_agent = SummarizerAgent(
            retriever,
            self.azure_endpoint,
            self.azure_key,
            self.azure_deployment,
            self.api_version,
            self.top_k
        )
        
        # Step 3: Generate summaries
        logger.info("Step 3: Generating RAG summaries...")
        summaries = {}
        
        summaries["overview"] = self.summarizer_agent.get_company_overview(ticker)
        summaries["drivers"] = self.summarizer_agent.get_growth_drivers(ticker)
        summaries["risks"] = self.summarizer_agent.get_risk_factors(ticker)
        summaries["events"] = self.summarizer_agent.get_recent_events(ticker)
        summaries["outlook"] = self.summarizer_agent.get_financial_outlook(ticker)
        
        # Step 4: Sentiment analysis
        logger.info("Step 4: Analyzing sentiment...")
        sentiment_results = self.sentiment_agent.analyze_news(news_items)
        
        # Also analyze risk sections if available
        risk_contexts = summaries["risks"].get("contexts", [])
        if risk_contexts:
            risk_texts = [ctx["text"] for ctx in risk_contexts[:10]]
            risk_sentiment = self.sentiment_agent.analyze_risk_factors(risk_texts)
            sentiment_results["risk_sentiment"] = risk_sentiment
        
        # Step 5: Calculate investment rating
        logger.info("Step 5: Calculating investment rating...")
        rating_results = self.rating_engine.calculate_rating(
            ticker=ticker,
            price_data={
                "current_price": price_metrics.get("current_price", 0),
                "ytd_return": price_metrics.get("ytd_return", 0),
                "high_52w": price_metrics.get("high_52w", 0),
                "low_52w": price_metrics.get("low_52w", 0),
                "volatility": price_metrics.get("volatility", 30)
            },
            company_info=company_info,
            sentiment_results=sentiment_results,
            summaries=summaries
        )
        
        # Step 6: Generate predictions
        logger.info("Step 6: Generating predictive forecasts...")
        prediction_results = self.predictive_engine.generate_predictions(
            ticker=ticker,
            price_data={
                "current_price": price_metrics.get("current_price", 0),
                "ytd_return": price_metrics.get("ytd_return", 0),
                "high_52w": price_metrics.get("high_52w", 0),
                "low_52w": price_metrics.get("low_52w", 0),
                "volatility": price_metrics.get("volatility", 30)
            },
            company_info=company_info,
            sentiment_results=sentiment_results,
            news_items=news_items,
            rating_results=rating_results,
            summaries=summaries
        )
        
        # Step 7: Generate report
        logger.info("Step 7: Generating analyst brief...")
        report_content = self.reporter_agent.generate_brief(
            ticker,
            company_info,
            price_metrics,
            summaries,
            sentiment_results,
            news_items,
            rating_results,
            prediction_results
        )
        
        # Step 8: Save report
        logger.info("Step 8: Saving report...")
        report_path = self.reporter_agent.save_brief(
            ticker,
            report_content,
            format=save_format
        )
        
        logger.info(f"Brief generation complete for {ticker}")
        
        return {
            "ticker": ticker,
            "report_path": report_path,
            "report_content": report_content,
            "summaries": summaries,
            "sentiment": sentiment_results,
            "rating": rating_results,
            "predictions": prediction_results,
            "company_info": company_info,
            "price_metrics": price_metrics
        }

    def get_quick_summary(self, ticker: str, query: str) -> Dict[str, Any]:
        """
        Get a quick summary for a specific query.
        
        Args:
            ticker: Stock ticker
            query: Query to answer
            
        Returns:
            Summary result
        """
        # Load index
        cached = self.ingestion_agent.load_index(ticker)
        if not cached:
            return {"error": f"No index found for {ticker}. Run ingestion first."}
        
        vector_store, retriever = cached
        
        # Initialize summarizer
        summarizer = SummarizerAgent(
            retriever,
            self.azure_endpoint,
            self.azure_key,
            self.azure_deployment,
            self.api_version,
            self.top_k
        )
        
        # Generate summary
        result = summarizer.generate_summary(query, ticker)
        return result
