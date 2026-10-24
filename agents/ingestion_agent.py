"""Ingestion agent for fetching and indexing financial data."""
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
import time
from datetime import datetime

from utils.sec_client import SECClient
from utils.news_client import NewsClient
from utils.prices import PriceClient
from rag.chunking import DocumentChunker, contextual_header, indexed_text
from rag.embed import EmbeddingModel
from rag.vector_store import FAISSVectorStore
from rag.retriever import HybridRetriever

logger = logging.getLogger(__name__)

# v2: section-aware chunks from real headers, 8-K exhibits, news title/summary metadata
# v3: contextual chunk headers in embeddings and BM25
# v4: safe-harbor disclaimers dropped, Yahoo news filtered to the company
INDEX_VERSION = 4


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
        chunk_overlap: int = 100,
        reranker=None
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
            reranker: Optional cross-encoder used by loaded retrievers
        """
        self.sec_client = SECClient(sec_user_agent, data_dir / "raw")
        self.news_client = NewsClient(sec_client=self.sec_client)  # Pass SEC client for 8-K parsing
        self.price_client = PriceClient()
        self.chunker = DocumentChunker(chunk_size, chunk_overlap)
        self.embedding_model = EmbeddingModel(
            azure_endpoint,
            azure_key,
            azure_deployment,
            api_version
        )
        self.data_dir = data_dir
        self.reranker = reranker
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
        
        # 2. Download 8-K filings early for news parsing
        logger.info("Pre-downloading 8-K filings for event parsing...")
        for filing in filings:
            if filing["form_type"] == "8-K":
                filepath = self.sec_client.download_filing(filing)
                if filepath:
                    filing['filepath'] = filepath
        
        # 3. Fetch news (now 8-Ks have filepaths for detailed parsing)
        logger.info("Fetching news from all sources...")
        news = self.news_client.aggregate_news(
            ticker, 
            filings, 
            use_yahoo=True, 
            use_rss=True, 
            use_reddit=True, 
            days_back=90
        )
        results["news"] = news
        logger.info(f"Found {len(news)} total news items")
        
        # 4. Fetch price data
        logger.info("Fetching price data...")
        price_data = self.price_client.get_key_metrics(ticker)
        company_info = self.price_client.get_company_info(ticker)
        results["price_data"] = {**price_data, **company_info}
        
        # 5. Process filings and create chunks
        logger.info("Processing filings and creating chunks...")
        all_chunks = []
        
        for filing in self._select_filings(filings, limit=10):
            # Download filing
            filepath = self.sec_client.download_filing(filing)
            if not filepath:
                continue
            
            # Store filepath in filing metadata for later use (e.g., 8-K parsing)
            filing['filepath'] = filepath
            
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

            # 8-K cover pages just point to EX-99.x, where the content is
            if filing["form_type"] == "8-K":
                for exhibit in self.sec_client.get_exhibits(filing):
                    exhibit_text = self.sec_client.parse_filing_text(exhibit["filepath"])
                    if len(exhibit_text) < 100:
                        continue
                    exhibit_metadata = {
                        **filing_metadata,
                        "url": exhibit["url"],
                        "section": f"Exhibit {exhibit['type'][3:]}",  # "Exhibit 99.1"
                    }
                    all_chunks.extend(self.chunker.chunk_text(exhibit_text, exhibit_metadata))
        
        # Add news as chunks
        for news_item in news:
            # Determine form_type based on source (8-K events vs regular news)
            form_type = "8-K" if news_item["source"] == "SEC 8-K" else "NEWS"
            
            news_chunk = {
                "text": f"{news_item['title']}. {news_item['summary']}",
                "chunk_id": len(all_chunks),
                "metadata": {
                    "ticker": ticker,
                    "form_type": form_type,
                    "filing_date": news_item["date"],
                    "source": news_item["source"],
                    "url": news_item["url"],
                    "title": news_item["title"],
                    "summary": news_item["summary"],
                    "section": "Document"  # Add section for citation consistency
                }
            }
            all_chunks.append(news_chunk)
        
        results["chunks_created"] = len(all_chunks)
        logger.info(f"Created {len(all_chunks)} chunks")
        
        # 6. Build vector index
        if all_chunks:
            logger.info("Building vector index...")
            index_path = self._index_path(ticker)
            vector_store = self._build_index(all_chunks, index_path)
            results["index_path"] = str(index_path)
            logger.info(f"Index saved to: {index_path}")
        
        logger.info(f"Ingestion complete for {ticker}")
        return results

    @staticmethod
    def _select_filings(
        filings: List[Dict[str, Any]],
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Pick filings to index: latest 10-K and two latest 10-Qs first, then
        the most recent remaining filings (mostly 8-Ks) up to the limit, so
        frequent 8-Ks can't push the annual report out.

        Args:
            filings: Filing metadata, newest first
            limit: Maximum number of filings

        Returns:
            Selected filings
        """
        core = (
            [f for f in filings if f["form_type"] == "10-K"][:1]
            + [f for f in filings if f["form_type"] == "10-Q"][:2]
        )
        rest = [f for f in filings if f not in core]
        return (core + rest)[:limit]

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
        
        # Contextual headers: embed (and BM25-index) each chunk with where it came from
        for chunk in chunks:
            chunk["context"] = contextual_header(chunk.get("metadata", {}))
        texts = [indexed_text(chunk) for chunk in chunks]
        logger.info(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.embedding_model.embed_documents(texts)
        
        # Add to vector store
        vector_store.add_documents(embeddings, chunks)
        
        # Save index
        vector_store.save()
        
        return vector_store

    def _index_path(self, ticker: str) -> Path:
        """
        Versioned index path: bump INDEX_VERSION when chunking or metadata
        changes so indexes built by older code are rebuilt, not reused. Dots
        are replaced so with_suffix() can't eat part of tickers like BRK.B.
        """
        return self.index_dir / f"{ticker.lower().replace('.', '-')}_v{INDEX_VERSION}"

    def load_index(self, ticker: str, max_age_days: Optional[float] = None) -> Optional[tuple]:
        """
        Load existing index for a ticker.

        Args:
            ticker: Stock ticker symbol
            max_age_days: Treat older indexes as missing (filings and news go stale)

        Returns:
            Tuple of (vector_store, retriever) or None
        """
        index_path = self._index_path(ticker)
        index_file = index_path.with_suffix('.faiss')

        if not index_file.exists():
            logger.warning(f"No index found for {ticker}")
            return None
        age_days = (time.time() - index_file.stat().st_mtime) / 86400
        if max_age_days is not None and age_days > max_age_days:
            logger.info(f"Index for {ticker} is {age_days:.0f} days old, rebuilding")
            return None
        
        # Load vector store
        vector_store = FAISSVectorStore(
            dimension=self.embedding_model.dimension,
            index_path=index_path
        )
        vector_store.load()
        
        # Create retriever
        retriever = HybridRetriever(vector_store, self.embedding_model, reranker=self.reranker)
        retriever.index_documents(vector_store.documents)
        
        logger.info(f"Loaded index for {ticker} with {vector_store.size} documents")
        return vector_store, retriever
