"""News client for fetching financial headlines."""
import requests
import feedparser
from typing import List, Dict, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class NewsClient:
    """Client for fetching financial news headlines."""

    def __init__(self):
        """Initialize news client."""
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

    def get_sec_news_from_8k(self, filings: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Extract news events from 8-K filings.
        
        Args:
            filings: List of filing metadata from SEC
            
        Returns:
            List of news items extracted from 8-K filings
        """
        news_items = []
        
        for filing in filings:
            if filing["form_type"] == "8-K":
                news_items.append({
                    "title": f"8-K Filing: {filing['filing_date']}",
                    "date": filing["filing_date"],
                    "source": "SEC 8-K",
                    "url": filing["document_url"],
                    "summary": f"Material event reported via 8-K on {filing['filing_date']}"
                })
        
        return news_items

    def get_rss_headlines(
        self, 
        ticker: str, 
        rss_urls: List[str] = None,
        days_back: int = 90
    ) -> List[Dict[str, str]]:
        """
        Fetch headlines from RSS feeds.
        
        Args:
            ticker: Stock ticker symbol
            rss_urls: List of RSS feed URLs (optional)
            days_back: Number of days to look back
            
        Returns:
            List of news headline dicts
        """
        if rss_urls is None:
            # Default: use generic financial news feeds
            # Note: Customize these based on available/allowed feeds
            rss_urls = []
        
        headlines = []
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        for feed_url in rss_urls:
            try:
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries:
                    # Check if ticker mentioned in title or summary
                    title = entry.get("title", "")
                    summary = entry.get("summary", entry.get("description", ""))
                    
                    if ticker.upper() in title.upper() or ticker.upper() in summary.upper():
                        pub_date = entry.get("published_parsed")
                        if pub_date:
                            pub_datetime = datetime(*pub_date[:6])
                            if pub_datetime >= cutoff_date:
                                headlines.append({
                                    "title": title,
                                    "date": pub_datetime.strftime("%Y-%m-%d"),
                                    "source": feed.feed.get("title", "RSS Feed"),
                                    "url": entry.get("link", ""),
                                    "summary": summary[:500]
                                })
            except Exception as e:
                logger.error(f"Error fetching RSS feed {feed_url}: {e}")
        
        return headlines

    def get_yahoo_finance_news(self, ticker: str, max_items: int = 20) -> List[Dict[str, str]]:
        """
        Fetch news from Yahoo Finance (via yfinance).
        
        Args:
            ticker: Stock ticker symbol
            max_items: Maximum number of news items to fetch
            
        Returns:
            List of news headline dicts
        """
        try:
            import yfinance as yf
            
            stock = yf.Ticker(ticker)
            news = stock.news
            
            headlines = []
            for item in news[:max_items]:
                headlines.append({
                    "title": item.get("title", ""),
                    "date": datetime.fromtimestamp(item.get("providerPublishTime", 0)).strftime("%Y-%m-%d"),
                    "source": item.get("publisher", "Yahoo Finance"),
                    "url": item.get("link", ""),
                    "summary": item.get("title", "")  # Yahoo doesn't always provide summary
                })
            
            return headlines
        except Exception as e:
            logger.error(f"Error fetching Yahoo Finance news for {ticker}: {e}")
            return []

    def aggregate_news(
        self, 
        ticker: str, 
        sec_filings: List[Dict[str, Any]] = None,
        use_yahoo: bool = True,
        days_back: int = 90
    ) -> List[Dict[str, str]]:
        """
        Aggregate news from all available sources.
        
        Args:
            ticker: Stock ticker symbol
            sec_filings: SEC filings to extract 8-K events from
            use_yahoo: Whether to include Yahoo Finance news
            days_back: Number of days to look back
            
        Returns:
            Aggregated and deduplicated news items
        """
        all_news = []
        
        # Get 8-K events from SEC
        if sec_filings:
            sec_news = self.get_sec_news_from_8k(sec_filings)
            all_news.extend(sec_news)
        
        # Get Yahoo Finance news
        if use_yahoo:
            yahoo_news = self.get_yahoo_finance_news(ticker, max_items=20)
            all_news.extend(yahoo_news)
        
        # Sort by date (newest first)
        all_news.sort(key=lambda x: x["date"], reverse=True)
        
        # Simple deduplication by title
        seen_titles = set()
        unique_news = []
        for item in all_news:
            title_lower = item["title"].lower()
            if title_lower not in seen_titles:
                seen_titles.add(title_lower)
                unique_news.append(item)
        
        return unique_news[:30]  # Limit to 30 most recent unique items
