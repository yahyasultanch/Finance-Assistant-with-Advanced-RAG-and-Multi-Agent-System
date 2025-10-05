"""Price data utilities using yfinance."""
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class PriceClient:
    """Client for fetching and analyzing price data."""

    def __init__(self):
        """Initialize price client."""
        pass

    def get_price_data(
        self, 
        ticker: str, 
        period: str = "1y",
        interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical price data.
        
        Args:
            ticker: Stock ticker symbol
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            
        Returns:
            DataFrame with OHLCV data or None
        """
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period, interval=interval)
            
            if df.empty:
                logger.error(f"No price data found for {ticker}")
                return None
            
            return df
        except Exception as e:
            logger.error(f"Error fetching price data for {ticker}: {e}")
            return None

    def get_company_info(self, ticker: str) -> dict:
        """
        Get company information.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with company info
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            return {
                "name": info.get("longName", ticker),
                "sector": info.get("sector", "N/A"),
                "industry": info.get("industry", "N/A"),
                "market_cap": info.get("marketCap", 0),
                "current_price": info.get("currentPrice", 0),
                "52_week_high": info.get("fiftyTwoWeekHigh", 0),
                "52_week_low": info.get("fiftyTwoWeekLow", 0),
                "pe_ratio": info.get("trailingPE", 0),
                "dividend_yield": info.get("dividendYield", 0),
                "beta": info.get("beta", 0),
            }
        except Exception as e:
            logger.error(f"Error fetching company info for {ticker}: {e}")
            return {"name": ticker, "sector": "N/A", "industry": "N/A"}

    def calculate_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate returns from price data.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with added return columns
        """
        df = df.copy()
        df["daily_return"] = df["Close"].pct_change()
        df["cumulative_return"] = (1 + df["daily_return"]).cumprod() - 1
        return df

    def calculate_volatility(
        self, 
        df: pd.DataFrame, 
        window: int = 30
    ) -> float:
        """
        Calculate rolling volatility.
        
        Args:
            df: DataFrame with price data
            window: Rolling window size
            
        Returns:
            Annualized volatility
        """
        if "daily_return" not in df.columns:
            df = self.calculate_returns(df)
        
        # Annualized volatility (assuming 252 trading days)
        volatility = df["daily_return"].std() * np.sqrt(252)
        return volatility

    def generate_sparkline_data(
        self, 
        ticker: str, 
        period: str = "1mo"
    ) -> Optional[list]:
        """
        Generate sparkline data for visualization.
        
        Args:
            ticker: Stock ticker symbol
            period: Time period
            
        Returns:
            List of closing prices or None
        """
        df = self.get_price_data(ticker, period=period)
        if df is None:
            return None
        
        return df["Close"].tolist()

    def get_key_metrics(self, ticker: str) -> dict:
        """
        Calculate key performance metrics.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with key metrics
        """
        df = self.get_price_data(ticker, period="1y")
        if df is None:
            return {}
        
        df = self.calculate_returns(df)
        
        # Calculate metrics
        current_price = df["Close"].iloc[-1]
        year_ago_price = df["Close"].iloc[0]
        ytd_return = (current_price - year_ago_price) / year_ago_price
        
        max_price = df["Close"].max()
        min_price = df["Close"].min()
        
        avg_volume = df["Volume"].mean()
        
        volatility = self.calculate_volatility(df)
        
        return {
            "current_price": round(current_price, 2),
            "ytd_return": round(ytd_return * 100, 2),
            "52_week_high": round(max_price, 2),
            "52_week_low": round(min_price, 2),
            "avg_volume": int(avg_volume),
            "volatility": round(volatility * 100, 2),
        }

    def calculate_event_return(
        self, 
        ticker: str, 
        event_date: str,
        window_days: int = 5
    ) -> Tuple[float, float]:
        """
        Calculate returns around an event date.
        
        Args:
            ticker: Stock ticker symbol
            event_date: Event date (YYYY-MM-DD)
            window_days: Days before and after event
            
        Returns:
            Tuple of (pre_event_return, post_event_return)
        """
        try:
            event_dt = datetime.strptime(event_date, "%Y-%m-%d")
            start_date = event_dt - timedelta(days=window_days * 2)
            end_date = event_dt + timedelta(days=window_days * 2)
            
            df = self.get_price_data(
                ticker,
                period=None,  # Use start/end instead
                interval="1d"
            )
            
            # Filter to window
            df = df.loc[start_date:end_date]
            df = self.calculate_returns(df)
            
            # Split pre/post event
            pre_event = df.loc[:event_date]
            post_event = df.loc[event_date:]
            
            pre_return = pre_event["daily_return"].sum() if len(pre_event) > 0 else 0
            post_return = post_event["daily_return"].sum() if len(post_event) > 0 else 0
            
            return pre_return, post_return
        except Exception as e:
            logger.error(f"Error calculating event return: {e}")
            return 0.0, 0.0
