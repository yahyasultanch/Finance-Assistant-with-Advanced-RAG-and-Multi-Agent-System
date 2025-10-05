"""Ingestion agent for fetching and indexing financial data."""
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

from utils.sec_client import SECClient
from utils.news_client import NewsClient
from utils.prices import PriceClient
from rag.chunking import DocumentChunker
from rag.embed import EmbeddingModel
from rag.vector_store import FAISSVectorStore
from rag.retriever import HybridRetriever

logger = logging.getLogger(__name__)


class IngestionAgent:
    """Agent responsible for data ingestion and indexing."""

    def __init__(
        self,
        sec_user_agent: str,
        data_dir: Path,
        index_dir: Path,
        azure_endpoint: str,
        azure_key: str,
        azure_deployment: str = "text-embedding-ada-002",
        api_version: str = "2024-12-01-preview",
        chunk_size: int = 1000,
        chunk_overlap: int = 100
    ):
        """
        Initialize ingestion agent.
        
        Args:
            sec_user_agent: User agent for SEC requests
            data_dir: Directory for caching data
            index_dir: Directory for vector indices
            azure_endpoint: Azure OpenAI endpoint
            azure_key: Azure OpenAI API key
            azure_deployment: Azure embedding deployment name
            api_version: Azure OpenAI API version
            chunk_size: Target chunk size in characters
            chunk_overlap: Chunk overlap in characters
        """
        self.sec_client = SECClient(sec_user_agent, data_dir / "raw")
        self.news_client = NewsClient()
        self.price_client = PriceClient()
        self.chunker = DocumentChunker(chunk_size, chunk_overlap)
        self.embedding_model = EmbeddingModel(
            azure_endpoint,
            azure_key,
            azure_deployment,
            api_version
        )
        self.data_dir = data_dir
        self.index_dir = index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)

    def ingest_ticker(
        self,
        ticker: str,
        days_back: int = 365,
        form_types: List[str] = ["10-K", "10-Q", "8-K"]
    ) -> Dict[str, Any]:
        """
        Ingest all data for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            days_back: Number of days to look back for filings
            form_types: List of SEC form types to fetch
            
        Returns:
            Dict with ingestion results and metadata
        """
        logger.info(f"Starting ingestion for ticker: {ticker}")
        
        results = {
            "ticker": ticker,
            "timestamp": datetime.now().isoformat(),
            "filings": [],
            "news": [],
            "price_data": {},
            "chunks_created": 0,
            "index_path": None
        }
        
        # 1. Fetch SEC filings
        logger.info("Fetching SEC filings...")
        filings = self.sec_client.get_recent_filings(ticker, form_types, days_back)
        results["filings"] = filings
        logger.info(f"Found {len(filings)} filings")
        
        # 2. Fetch news
        logger.info("Fetching news...")
        news = self.news_client.aggregate_news(ticker, filings, use_yahoo=True, days_back=90)
        results["news"] = news
        logger.info(f"Found {len(news)} news items")
        
        # 3. Fetch price data
        logger.info("Fetching price data...")
        price_data = self.price_client.get_key_metrics(ticker)
        company_info = self.price_client.get_company_info(ticker)
        results["price_data"] = {**price_data, **company_info}
        
        # 4. Process filings and create chunks
        logger.info("Processing filings and creating chunks...")
        all_chunks = []
        
        for filing in filings[:10]:  # Process top 10 most recent filings
            # Download filing
            filepath = self.sec_client.download_filing(filing)
            if not filepath:
                continue
            
            # Parse text
            text = self.sec_client.parse_filing_text(filepath)
            if not text or len(text) < 100:
                continue
            
            # Extract sections based on form type
            sections = {}
            if filing["form_type"] == "10-K":
                sections = self.chunker.extract_sections_from_10k(text)
            elif filing["form_type"] == "10-Q":
                sections = self.chunker.extract_sections_from_10q(text)
            
            # Create chunks
            filing_metadata = {
                "ticker": ticker,
                "form_type": filing["form_type"],
                "filing_date": filing["filing_date"],
                "source": "SEC EDGAR",
                "url": filing["document_url"]
            }
            
            if sections:
                chunks = self.chunker.chunk_filing(text, filing_metadata, sections)
            else:
                chunks = self.chunker.chunk_text(text, filing_metadata)
            
            all_chunks.extend(chunks)
        
        # Add news as chunks
        for news_item in news:
            news_chunk = {
                "text": f"{news_item['title']}. {news_item['summary']}",
                "chunk_id": len(all_chunks),
                "metadata": {
                    "ticker": ticker,
                    "form_type": "NEWS",
                    "filing_date": news_item["date"],
                    "source": news_item["source"],
                    "url": news_item["url"]
                }
            }
            all_chunks.append(news_chunk)
        
        results["chunks_created"] = len(all_chunks)
        logger.info(f"Created {len(all_chunks)} chunks")
        
        # 5. Build vector index
        if all_chunks:
            logger.info("Building vector index...")
            index_path = self.index_dir / f"{ticker.lower()}"
            vector_store = self._build_index(all_chunks, index_path)
            results["index_path"] = str(index_path)
            logger.info(f"Index saved to: {index_path}")
        
        logger.info(f"Ingestion complete for {ticker}")
        return results

    def _build_index(
        self,
        chunks: List[Dict[str, Any]],
        index_path: Path
    ) -> FAISSVectorStore:
        """
        Build and save vector index.
        
        Args:
            chunks: List of document chunks
            index_path: Path to save index
            
        Returns:
            FAISSVectorStore instance
        """
        # Initialize vector store
        vector_store = FAISSVectorStore(
            dimension=self.embedding_model.dimension,
            index_path=index_path
        )
        
        # Generate embeddings in batches
        texts = [chunk["text"] for chunk in chunks]
        logger.info(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.embedding_model.embed_documents(texts)
        
        # Add to vector store
        vector_store.add_documents(embeddings, chunks)
        
        # Save index
        vector_store.save()
        
        return vector_store

    def load_index(self, ticker: str) -> Optional[tuple]:
        """
        Load existing index for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Tuple of (vector_store, retriever) or None
        """
        index_path = self.index_dir / f"{ticker.lower()}"
        
        if not index_path.with_suffix('.faiss').exists():
            logger.warning(f"No index found for {ticker}")
            return None
        
        # Load vector store
        vector_store = FAISSVectorStore(
            dimension=self.embedding_model.dimension,
            index_path=index_path
        )
        vector_store.load()
        
        # Create retriever
        retriever = HybridRetriever(vector_store, self.embedding_model)
        retriever.index_documents(vector_store.documents)
        
        logger.info(f"Loaded index for {ticker} with {vector_store.size} documents")
        return vector_store, retriever
