"""News client for fetching financial headlines."""
import re
import requests
import feedparser
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


def hacker_news_stories(query: str, days_back: int = 30, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Recent Hacker News stories matching a query (Algolia HN API, no key needed),
    most relevant first. A tech-community channel, so treat it as social, not news.
    """
    since = int((datetime.now() - timedelta(days=days_back)).timestamp())
    try:
        response = requests.get(
            "https://hn.algolia.com/api/v1/search",
            params={"query": query, "tags": "story", "hitsPerPage": limit,
                    "numericFilters": f"created_at_i>{since}"},
            timeout=10,
        )
        response.raise_for_status()
        hits = response.json().get("hits", [])
    except Exception as e:
        logger.warning(f"Hacker News search failed for {query!r}: {e}")
        return []
    return [{
        "title": h.get("title") or "",
        "date": (h.get("created_at") or "")[:10],
        "source": "Hacker News",
        "url": f"https://news.ycombinator.com/item?id={h['objectID']}",
        "summary": f"{h.get('points') or 0} points, {h.get('num_comments') or 0} comments. "
                   f"Links to {h.get('url') or 'a text post'}.",
    } for h in hits if h.get("title")]


class NewsClient:
    """Client for fetching financial news headlines."""

    def __init__(self, sec_client=None):
        """
        Initialize news client.
        
        Args:
            sec_client: Optional SECClient instance for parsing 8-K details
        """
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        self.sec_client = sec_client

    def get_sec_news_from_8k(self, filings: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Extract news events from 8-K filings with actual event descriptions.
        
        Args:
            filings: List of filing metadata from SEC (should include 'filepath' if downloaded)
            
        Returns:
            List of news items extracted from 8-K filings with event details
        """
        news_items = []
        
        for filing in filings:
            if filing["form_type"] == "8-K":
                # Try to get detailed event summary if SEC client is available
                if self.sec_client and filing.get('filepath'):
                    try:
                        event_summary = self.sec_client.extract_8k_event_summary(filing)
                        title = f"8-K: {event_summary[:100]}"
                        summary = event_summary
                    except Exception as e:
                        logger.warning(f"Could not parse 8-K details: {e}")
                        title = f"8-K Filing: {filing['filing_date']}"
                        summary = f"Material event reported via 8-K on {filing['filing_date']}"
                else:
                    # Fallback to generic description
                    title = f"8-K Filing: {filing['filing_date']}"
                    summary = f"Material event reported via 8-K on {filing['filing_date']}"
                
                news_items.append({
                    "title": title,
                    "date": filing["filing_date"],
                    "source": "SEC 8-K",
                    "url": filing["document_url"],
                    "summary": summary
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
            # Ticker.news can come back empty when Yahoo changes its feed;
            # the search endpoint has stayed reliable, but it also returns
            # market roundups that merely list the ticker, so filter those
            news = stock.news or yf.Search(ticker, news_count=max_items * 2).news
            name = (stock.info.get("shortName") or ticker).replace(",", " ").split()[0]

            headlines = []
            for item in news:
                parsed = self._parse_yahoo_item(item)
                if parsed["title"] and self._is_about(f"{parsed['title']} {parsed['summary']}", ticker, name):
                    headlines.append(parsed)
            return headlines[:max_items]
        except Exception as e:
            logger.error(f"Error fetching Yahoo Finance news for {ticker}: {e}")
            return []

    @staticmethod
    def _is_about(text: str, ticker: str, name: str) -> bool:
        """The story names the company or ticker (Yahoo's relatedTickers tags are too loose)."""
        return re.search(rf"\b({re.escape(ticker)}|{re.escape(name)})\b", text, re.IGNORECASE) is not None

    @staticmethod
    def _parse_yahoo_item(item: Dict[str, Any]) -> Dict[str, str]:
        """
        Normalize a yfinance news item.

        yfinance >= 0.2.50 nests articles under "content" (title, summary,
        pubDate, provider, canonicalUrl); older versions used a flat dict
        (title, publisher, link, providerPublishTime). Both are handled.
        """
        content = item.get("content")
        if content:
            title = content.get("title") or ""
            summary = content.get("summary") or content.get("description") or ""
            source = (content.get("provider") or {}).get("displayName") or "Yahoo Finance"
            url = ((content.get("canonicalUrl") or {}).get("url")
                   or (content.get("clickThroughUrl") or {}).get("url") or "")
            date = (content.get("pubDate") or "")[:10]
        else:
            title = item.get("title", "")
            summary = ""
            source = item.get("publisher", "Yahoo Finance")
            url = item.get("link", "")
            published = item.get("providerPublishTime")
            date = datetime.fromtimestamp(published).strftime("%Y-%m-%d") if published else ""

        return {
            "title": title,
            "date": date,
            "source": source,
            "url": url,
            # Fall back to title + source so sentiment has some context
            "summary": (summary or f"{title} - {source}")[:500]
        }

    def get_financial_rss_feeds(self, ticker: str, days_back: int = 90) -> List[Dict[str, str]]:
        """
        Fetch news from free financial RSS feeds.
        
        Args:
            ticker: Stock ticker symbol
            days_back: Number of days to look back
            
        Returns:
            List of news headline dicts from RSS feeds
        """
        # Free RSS feeds (no API key required)
        rss_feeds = [
            f"https://finance.yahoo.com/rss/headline?s={ticker}",
            "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^GSPC&region=US&lang=en-US",
            "https://www.marketwatch.com/rss/topstories",
        ]
        
        return self.get_rss_headlines(ticker, rss_feeds, days_back)

    def get_reddit_discussions(self, ticker: str, max_posts: int = 10) -> List[Dict[str, str]]:
        """
        Fetch Reddit discussions about a ticker from financial subreddits.
        
        Requires Reddit API credentials (free). To set up:
        1. Go to https://www.reddit.com/prefs/apps
        2. Click "Create App" or "Create Another App"
        3. Select "script" type
        4. Note your client_id and client_secret
        5. Add to .env file:
           REDDIT_CLIENT_ID=your_client_id_here
           REDDIT_CLIENT_SECRET=your_client_secret_here
        
        Args:
            ticker: Stock ticker symbol
            max_posts: Maximum number of posts to fetch
            
        Returns:
            List of Reddit post dicts
        """
        try:
            import praw
            import os
            
            # Get credentials from environment
            client_id = os.getenv("REDDIT_CLIENT_ID", "")
            client_secret = os.getenv("REDDIT_CLIENT_SECRET", "")
            
            if not client_id or not client_secret:
                logger.info("Reddit credentials not configured. Skipping Reddit integration. See documentation for setup instructions.")
                return []
            
            # Initialize Reddit with credentials
            reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent="FinanceAnalyst/1.0 (by /u/YourRedditUsername)"
            )
            
            discussions = []
            subreddits = ["wallstreetbets", "stocks", "investing"]
            
            for subreddit_name in subreddits:
                try:
                    subreddit = reddit.subreddit(subreddit_name)
                    
                    # Search for ticker mentions
                    query = f"${ticker} OR {ticker}"
                    for submission in subreddit.search(query, limit=max_posts // len(subreddits), time_filter="week"):
                        # Calculate a pseudo-date from timestamp
                        post_date = datetime.fromtimestamp(submission.created_utc).strftime("%Y-%m-%d")
                        
                        discussions.append({
                            "title": submission.title,
                            "date": post_date,
                            "source": f"Reddit r/{subreddit_name}",
                            "url": f"https://reddit.com{submission.permalink}",
                            "summary": submission.selftext[:300] if submission.selftext else submission.title,
                            "score": submission.score
                        })
                except Exception as e:
                    logger.warning(f"Error fetching from r/{subreddit_name}: {e}")
                    continue
            
            # Sort by score (upvotes) to get most popular discussions
            discussions.sort(key=lambda x: x.get("score", 0), reverse=True)
            return discussions[:max_posts]
            
        except ImportError:
            logger.error("PRAW not installed. Run: pip install praw")
            return []
        except Exception as e:
            logger.error(f"Error fetching Reddit discussions for {ticker}: {e}")
            return []

    def aggregate_news(
        self, 
        ticker: str, 
        sec_filings: List[Dict[str, Any]] = None,
        use_yahoo: bool = True,
        use_rss: bool = True,
        use_reddit: bool = True,
        days_back: int = 90
    ) -> List[Dict[str, str]]:
        """
        Aggregate news from all available sources.
        
        Args:
            ticker: Stock ticker symbol
            sec_filings: SEC filings to extract 8-K events from
            use_yahoo: Whether to include Yahoo Finance news
            use_rss: Whether to include RSS feeds
            use_reddit: Whether to include Reddit discussions
            days_back: Number of days to look back
            
        Returns:
            Aggregated and deduplicated news items
        """
        all_news = []
        
        # Get 8-K events from SEC
        if sec_filings:
            sec_news = self.get_sec_news_from_8k(sec_filings)
            all_news.extend(sec_news)
            logger.info(f"Found {len(sec_news)} 8-K events")
        
        # Get Yahoo Finance news
        if use_yahoo:
            yahoo_news = self.get_yahoo_finance_news(ticker, max_items=20)
            all_news.extend(yahoo_news)
            logger.info(f"Found {len(yahoo_news)} Yahoo Finance articles")
        
        # Get RSS feed news
        if use_rss:
            rss_news = self.get_financial_rss_feeds(ticker, days_back)
            all_news.extend(rss_news)
            logger.info(f"Found {len(rss_news)} RSS feed articles")
        
        # Get Reddit discussions
        if use_reddit:
            reddit_news = self.get_reddit_discussions(ticker, max_posts=10)
            all_news.extend(reddit_news)
            logger.info(f"Found {len(reddit_news)} Reddit discussions")
        
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
        
        logger.info(f"Total unique news items after deduplication: {len(unique_news)}")
        return unique_news[:40]  # Increased limit to 40 for more diverse sources
