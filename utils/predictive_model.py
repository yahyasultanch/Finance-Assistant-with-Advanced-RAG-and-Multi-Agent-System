"""Predictive model engine for stock price forecasting and event impact analysis."""
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta
import logging
import re
from collections import defaultdict

logger = logging.getLogger(__name__)


class PredictiveEngine:
    """
    Multi-horizon predictive model for stock forecasting.
    
    Combines:
    - Sentiment momentum analysis (short-term signals)
    - Event impact scoring (catalyst detection)
    - Technical trend analysis (price momentum)
    - Macro factor correlation (geopolitical/economic)
    """
    
    def __init__(self):
        """Initialize predictive engine."""
        # Prediction confidence thresholds (recalibrated)
        self.confidence_thresholds = {
            "high": 0.70,    # >70% confidence (strong data support)
            "medium": 0.50,  # 50-70% confidence (moderate data)
            "low": 0.30      # 30-50% confidence (weak data)
        }
        
        # Event impact weights - conservative estimates for stock prediction
        # Note: Actual impact varies by company size, sector, and market conditions
        self.event_weights = {
            "earnings_beat": 0.04,        # +4% realistic move (not 8%)
            "earnings_miss": -0.05,       # -5% realistic move
            "product_launch": 0.03,       # +3% for major launch (conservative)
            "partnership": 0.04,          # +4% for strategic deal (reduced from 7%)
            "regulatory_approval": 0.08,  # +8% (pharma/biotech higher)
            "lawsuit": -0.04,             # -4% for litigation
            "executive_change": -0.02,    # -2% for CEO departure
            "acquisition": 0.06,          # +6% for announced M&A (conservative)
            "downgrade": -0.03,           # -3% for analyst downgrade
            "upgrade": 0.02               # +2% for analyst upgrade
        }
        
        # Volatility dampening factors (prevent overly aggressive predictions)
        self.volatility_multipliers = {
            "low": 0.7,      # Reduce prediction range for low-vol stocks
            "moderate": 1.0,  # Normal prediction range
            "high": 1.3       # Expand range for high-vol stocks
        }
    
    def generate_predictions(
        self,
        ticker: str,
        price_data: Dict[str, Any],
        company_info: Dict[str, Any],
        sentiment_results: Dict[str, Any],
        news_items: List[Dict[str, str]],
        rating_results: Dict[str, Any],
        summaries: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate multi-horizon predictions.
        
        Args:
            ticker: Stock ticker
            price_data: Price metrics and history
            company_info: Company fundamentals
            sentiment_results: Sentiment analysis results
            news_items: Recent news with timestamps
            rating_results: Current rating and scores
            summaries: RAG summaries from filings
            
        Returns:
            Dict with predictions for 3 time horizons and supporting analysis
        """
        logger.info(f"Generating predictions for {ticker}")
        
        current_price = price_data.get("current_price", 0)
        
        # 1. Analyze sentiment momentum
        sentiment_momentum = self._analyze_sentiment_momentum(news_items, sentiment_results)
        
        # 2. Detect upcoming catalysts
        catalysts = self._detect_catalysts(ticker, news_items, summaries)
        
        # 3. Analyze technical trends
        technical_trends = self._analyze_technical_trends(price_data)
        
        # 4. Detect macro factors
        macro_factors = self._detect_macro_factors(news_items, summaries)
        
        # 5. Generate predictions for 2 horizons (removed medium-term)
        predictions = {
            "short_term": self._predict_short_term(
                current_price, sentiment_momentum, technical_trends, price_data
            ),
            "long_term": self._predict_long_term(
                current_price, rating_results, macro_factors, company_info, catalysts
            )
        }
        
        return {
            "predictions": predictions,
            "sentiment_momentum": sentiment_momentum,
            "catalysts": catalysts,
            "technical_trends": technical_trends,
            "macro_factors": macro_factors,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def _analyze_sentiment_momentum(
        self,
        news_items: List[Dict[str, str]],
        sentiment_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze sentiment velocity and acceleration.
        
        Tracks how sentiment is changing over time:
        - Velocity: Rate of change in sentiment
        - Acceleration: Rate of change in velocity
        - Inflection points: When sentiment reverses direction
        """
        if not news_items:
            return {
                "velocity": 0.0,
                "acceleration": 0.0,
                "trend": "stable",
                "confidence": 0.0
            }
        
        # Sort news by date (most recent first)
        sorted_news = sorted(
            news_items,
            key=lambda x: x.get("date", ""),
            reverse=True
        )
        
        # Group into time windows (last 7 days, 7-14 days, 14-30 days)
        now = datetime.now()
        windows = {
            "recent": [],  # Last 7 days
            "previous": [], # 7-14 days ago
            "older": []     # 14-30 days ago
        }
        
        for item in sorted_news:
            try:
                # Parse date from news item
                date_str = item.get("date", "")
                if not date_str:
                    continue
                
                # Try to parse date (handle various formats)
                item_date = self._parse_date(date_str)
                if not item_date:
                    continue
                
                days_ago = (now - item_date).days
                
                if days_ago <= 7:
                    windows["recent"].append(item)
                elif days_ago <= 14:
                    windows["previous"].append(item)
                elif days_ago <= 30:
                    windows["older"].append(item)
            except Exception as e:
                logger.warning(f"Error parsing date: {e}")
                continue
        
        # Calculate sentiment for each window
        window_sentiments = {}
        for window_name, items in windows.items():
            if items:
                # Use simple heuristic: check if sentiment is mentioned in text
                positive_count = sum(1 for item in items if self._is_positive_sentiment(item))
                negative_count = sum(1 for item in items if self._is_negative_sentiment(item))
                total = len(items)
                
                sentiment_score = (positive_count - negative_count) / total if total > 0 else 0
                window_sentiments[window_name] = sentiment_score
            else:
                window_sentiments[window_name] = 0.0
        
        # Calculate velocity (recent vs previous)
        recent_sentiment = window_sentiments.get("recent", 0.0)
        previous_sentiment = window_sentiments.get("previous", 0.0)
        older_sentiment = window_sentiments.get("older", 0.0)
        
        velocity = recent_sentiment - previous_sentiment
        previous_velocity = previous_sentiment - older_sentiment
        acceleration = velocity - previous_velocity
        
        # Determine trend
        if velocity > 0.1:
            trend = "improving"
        elif velocity < -0.1:
            trend = "deteriorating"
        else:
            trend = "stable"
        
        # Calculate confidence based on sample size
        total_items = sum(len(items) for items in windows.values())
        confidence = min(1.0, total_items / 20.0)  # Full confidence at 20+ items
        
        return {
            "velocity": round(velocity, 3),
            "acceleration": round(acceleration, 3),
            "trend": trend,
            "confidence": round(confidence, 2),
            "recent_sentiment": round(recent_sentiment, 2),
            "window_counts": {k: len(v) for k, v in windows.items()}
        }
    
    def _detect_catalysts(
        self,
        ticker: str,
        news_items: List[Dict[str, str]],
        summaries: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Detect upcoming catalysts from news and filings.
        
        Looks for:
        - Earnings dates
        - Product launches
        - Regulatory events
        - Partnerships/acquisitions
        - Executive changes
        """
        catalysts = []
        
        # Keywords for different catalyst types
        catalyst_keywords = {
            "earnings": ["earnings", "quarterly results", "fiscal", "Q1", "Q2", "Q3", "Q4"],
            "product_launch": ["launch", "release", "unveil", "introduce", "new product"],
            "partnership": ["partnership", "deal", "agreement", "collaboration", "joint venture"],
            "acquisition": ["acquisition", "acquire", "merger", "purchase", "buy"],
            "regulatory": ["FDA approval", "regulatory", "clearance", "authorization"],
            "executive_change": ["CEO", "CFO", "executive", "appoint", "resign", "depart"]
        }
        
        # Search in news items
        for item in news_items:
            text = self._news_text(item)
            
            for catalyst_type, keywords in catalyst_keywords.items():
                if self._mentions(text, keywords):
                    # Estimate impact
                    impact = self._estimate_catalyst_impact(catalyst_type, text)
                    
                    catalysts.append({
                        "type": catalyst_type,
                        "source": "news",
                        "title": item.get("title", ""),
                        "date": item.get("date", ""),
                        "impact_estimate": impact,
                        "confidence": 0.6  # Medium confidence from news
                    })
        
        # Search in SEC filings summaries
        events_summary = summaries.get("events", {}).get("summary", "")
        outlook_summary = summaries.get("outlook", {}).get("summary", "")
        
        combined_filing_text = (events_summary + " " + outlook_summary).lower()
        
        for catalyst_type, keywords in catalyst_keywords.items():
            if self._mentions(combined_filing_text, keywords):
                impact = self._estimate_catalyst_impact(catalyst_type, combined_filing_text)
                
                catalysts.append({
                    "type": catalyst_type,
                    "source": "sec_filing",
                    "title": f"{catalyst_type.replace('_', ' ').title()} mentioned in filings",
                    "date": "upcoming",
                    "impact_estimate": impact,
                    "confidence": 0.8  # High confidence from SEC filings
                })
        
        # Remove duplicates and sort by impact
        unique_catalysts = []
        seen_types = set()
        
        for catalyst in sorted(catalysts, key=lambda x: abs(x["impact_estimate"]), reverse=True):
            key = (catalyst["type"], catalyst["source"])
            if key not in seen_types:
                unique_catalysts.append(catalyst)
                seen_types.add(key)
        
        return unique_catalysts[:5]  # Top 5 catalysts
    
    def _analyze_technical_trends(self, price_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze technical indicators and price trends.
        
        Simple technical analysis:
        - Price position in 52-week range
        - YTD momentum
        - Volatility trend
        """
        current_price = price_data.get("current_price", 0)
        high_52w = price_data.get("high_52w", current_price)
        low_52w = price_data.get("low_52w", current_price)
        ytd_return = price_data.get("ytd_return", 0)
        volatility = price_data.get("volatility", 30)
        
        # Calculate price position
        if high_52w > low_52w:
            price_position = (current_price - low_52w) / (high_52w - low_52w)
        else:
            price_position = 0.5
        
        # Determine trend
        if ytd_return > 15:
            trend = "strong_uptrend"
            trend_score = 0.8
        elif ytd_return > 5:
            trend = "uptrend"
            trend_score = 0.6
        elif ytd_return > -5:
            trend = "sideways"
            trend_score = 0.5
        elif ytd_return > -15:
            trend = "downtrend"
            trend_score = 0.4
        else:
            trend = "strong_downtrend"
            trend_score = 0.2
        
        # Volatility assessment
        if volatility < 20:
            volatility_regime = "low"
        elif volatility < 35:
            volatility_regime = "moderate"
        else:
            volatility_regime = "high"
        
        return {
            "trend": trend,
            "trend_score": trend_score,
            "price_position": round(price_position, 2),
            "ytd_return": ytd_return,
            "volatility_regime": volatility_regime,
            "momentum_signal": "bullish" if ytd_return > 10 else "bearish" if ytd_return < -10 else "neutral"
        }
    
    def _detect_macro_factors(
        self,
        news_items: List[Dict[str, str]],
        summaries: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Detect macro and geopolitical factors mentioned in news/filings.
        
        Looks for:
        - Interest rate mentions
        - Inflation/deflation
        - Supply chain issues
        - Geopolitical tensions
        - Regulatory changes
        """
        macro_keywords = {
            "interest_rates": ["interest rate", "fed", "federal reserve", "rate cut", "rate hike"],
            "inflation": ["inflation", "deflation", "price increase", "cost pressure"],
            "supply_chain": ["supply chain", "shortage", "logistics", "procurement"],
            "geopolitical": ["china", "trade war", "tariff", "sanctions", "conflict"],
            "regulatory": ["regulation", "compliance", "sec", "antitrust", "policy"]
        }
        
        factor_counts = defaultdict(int)
        factor_sentiment = defaultdict(list)
        
        # Analyze news
        for item in news_items:
            text = self._news_text(item)
            
            for factor_type, keywords in macro_keywords.items():
                if self._mentions(text, keywords):
                    factor_counts[factor_type] += 1
                    
                    # Determine if mention is positive or negative
                    if self._is_negative_sentiment(item):
                        factor_sentiment[factor_type].append(-1)
                    elif self._is_positive_sentiment(item):
                        factor_sentiment[factor_type].append(1)
                    else:
                        factor_sentiment[factor_type].append(0)
        
        # Analyze filings
        risk_summary = summaries.get("risks", {}).get("summary", "").lower()
        outlook_summary = summaries.get("outlook", {}).get("summary", "").lower()
        
        combined_text = risk_summary + " " + outlook_summary
        
        for factor_type, keywords in macro_keywords.items():
            if self._mentions(combined_text, keywords):
                factor_counts[factor_type] += 2  # Weight SEC filings higher
                factor_sentiment[factor_type].append(-0.5)  # Assume cautious
        
        # Build factor analysis
        detected_factors = []
        
        for factor_type, count in factor_counts.items():
            if count > 0:
                sentiments = factor_sentiment[factor_type]
                avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
                
                detected_factors.append({
                    "type": factor_type,
                    "mentions": count,
                    "sentiment": "positive" if avg_sentiment > 0.2 else "negative" if avg_sentiment < -0.2 else "neutral",
                    "impact": "high" if count >= 3 else "medium" if count >= 2 else "low"
                })
        
        return {
            "detected_factors": detected_factors,
            "factor_count": len(detected_factors),
            "overall_macro_risk": "elevated" if len(detected_factors) >= 3 else "moderate" if len(detected_factors) >= 2 else "low"
        }
    
    def _predict_short_term(
        self,
        current_price: float,
        sentiment_momentum: Dict[str, Any],
        technical_trends: Dict[str, Any],
        price_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Short-term prediction (1-7 days) - REFINED FOR REALISM.
        
        Based on:
        - Sentiment momentum (velocity and acceleration)
        - Technical momentum
        - Volatility regime (dampens aggressive predictions)
        """
        if current_price <= 0:
            return self._empty_prediction("short_term")
        
        # Get volatility regime
        volatility = price_data.get("volatility", 30)
        if volatility < 20:
            vol_regime = "low"
        elif volatility < 35:
            vol_regime = "moderate"
        else:
            vol_regime = "high"
        
        vol_multiplier = self.volatility_multipliers.get(vol_regime, 1.0)
        
        # Base prediction on sentiment velocity (REDUCED impact)
        velocity = sentiment_momentum.get("velocity", 0)
        acceleration = sentiment_momentum.get("acceleration", 0)
        trend_score = technical_trends.get("trend_score", 0.5)
        
        # CONSERVATIVE short-term impact (1-7 days is very short!)
        sentiment_impact = velocity * 0.015 * vol_multiplier  # 1.5% per unit velocity (was 3%)
        momentum_impact = (trend_score - 0.5) * 0.01 * vol_multiplier  # ±0.5% (was ±1%)
        acceleration_impact = acceleration * 0.005 * vol_multiplier  # 0.5% per unit (was 1%)
        
        expected_change_pct = (sentiment_impact + momentum_impact + acceleration_impact) * 100
        
        # Cap short-term moves at ±5% (realistic for 1 week)
        expected_change_pct = max(-5.0, min(5.0, expected_change_pct))
        
        # Calculate target price
        target_price = current_price * (1 + expected_change_pct / 100)
        
        # Calculate confidence based on data quality
        sentiment_confidence = sentiment_momentum.get("confidence", 0)
        news_count = sum(sentiment_momentum.get("window_counts", {}).values())
        
        # Confidence scoring (recalibrated)
        data_quality_score = min(1.0, news_count / 15.0)  # Need 15+ news items for high confidence
        base_confidence = (sentiment_confidence * 0.6 + data_quality_score * 0.4)
        
        # Penalize low volatility scenarios (less predictable)
        if vol_regime == "high":
            base_confidence *= 0.85  # Reduce confidence in volatile markets
        
        # Determine direction
        if expected_change_pct > 0.5:
            direction = "upward"
        elif expected_change_pct < -0.5:
            direction = "downward"
        else:
            direction = "sideways"
        
        return {
            "horizon": "1-7 days",
            "target_price": round(target_price, 2),
            "expected_change_pct": round(expected_change_pct, 2),
            "direction": direction,
            "confidence": self._get_confidence_level(base_confidence),
            "confidence_score": round(base_confidence, 2),
            "key_drivers": [
                f"Sentiment momentum: {sentiment_momentum.get('trend', 'stable')}",
                f"Technical trend: {technical_trends.get('momentum_signal', 'neutral')}",
                f"Volatility regime: {vol_regime}"
            ]
        }
    
    def _predict_long_term(
        self,
        current_price: float,
        rating_results: Dict[str, Any],
        macro_factors: Dict[str, Any],
        company_info: Dict[str, Any],
        catalysts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Long-term prediction (3-12 months) - ENHANCED WITH CATALYSTS.
        
        Based on:
        - Rating engine's 12-month price target (fundamental baseline)
        - Identified catalysts with timing consideration
        - Macro factor adjustments
        - Sector trends
        """
        if current_price <= 0:
            return self._empty_prediction("long_term")
        
        # Use rating engine's price target as baseline (already well-calibrated)
        base_target = rating_results.get("price_target", current_price)
        
        # Integrate high-confidence catalysts (from SEC filings primarily)
        catalyst_adjustment = 0.0
        high_confidence_catalysts = []
        
        for catalyst in catalysts[:3]:  # Top 3
            confidence = catalyst.get("confidence", 0.5)
            source = catalyst.get("source", "")
            
            # Weight SEC filings higher than news
            if source == "sec_filing" and confidence >= 0.7:
                impact = catalyst.get("impact_estimate", 0)
                # Discount impact for long-term (not all materializes immediately)
                catalyst_adjustment += impact * 0.6  # 60% of estimated impact
                high_confidence_catalysts.append(catalyst)
            elif source == "news" and confidence >= 0.8:
                impact = catalyst.get("impact_estimate", 0)
                catalyst_adjustment += impact * 0.4  # 40% of impact (news less reliable)
                high_confidence_catalysts.append(catalyst)
        
        # Cap catalyst impact at ±10% (realistic for 12 months)
        catalyst_adjustment = max(-0.10, min(0.10, catalyst_adjustment))
        
        # Macro factor adjustment (conservative)
        macro_risk = macro_factors.get("overall_macro_risk", "moderate")
        macro_adjustment = {
            "low": 0.015,      # +1.5% in benign macro (reduced from 2%)
            "moderate": 0.0,   # No adjustment
            "elevated": -0.02  # -2% in challenging macro (reduced from -3%)
        }.get(macro_risk, 0.0)
        
        # Sector adjustment (small, already in rating target)
        sector = company_info.get("sector", "Technology")
        sector_trend_adjustment = {
            "Technology": 0.01,
            "Healthcare": 0.01,
            "Energy": 0.0,
            "Utilities": -0.01,
            "Financial Services": 0.005
        }.get(sector, 0.0)
        
        # Apply adjustments to rating target
        adjusted_target = base_target * (1 + catalyst_adjustment + macro_adjustment + sector_trend_adjustment)
        expected_change_pct = ((adjusted_target - current_price) / current_price) * 100
        
        # Direction
        if expected_change_pct > 8:
            direction = "upward"
        elif expected_change_pct < -8:
            direction = "downward"
        else:
            direction = "sideways"
        
        # Confidence (moderate for 12-month horizon)
        rating_confidence = rating_results.get("composite_score", 50) / 100  # Convert to 0-1
        catalyst_conf = min(1.0, len(high_confidence_catalysts) / 2.0)
        base_confidence = (rating_confidence * 0.6 + catalyst_conf * 0.4) * 0.75  # Max 75% for long-term
        
        # Key drivers
        drivers = []
        rating = rating_results.get("rating", "HOLD")
        drivers.append(f"Current rating: {rating}")
        
        # Add catalyst drivers if significant
        if high_confidence_catalysts:
            for cat in high_confidence_catalysts[:2]:  # Top 2
                cat_type = cat.get("type", "").replace("_", " ").title()
                drivers.append(f"Catalyst: {cat_type}")
        
        if macro_factors.get("factor_count", 0) > 0:
            drivers.append(f"Macro environment: {macro_risk} risk")
        
        drivers.append(f"Sector outlook: {sector}")
        
        # Add upside/downside range (±15% uncertainty band for 12 months)
        uncertainty_band = 0.15
        upside_target = adjusted_target * (1 + uncertainty_band)
        downside_target = adjusted_target * (1 - uncertainty_band)
        
        return {
            "horizon": "3-12 months",
            "target_price": round(adjusted_target, 2),
            "upside_target": round(upside_target, 2),
            "downside_target": round(downside_target, 2),
            "expected_change_pct": round(expected_change_pct, 2),
            "direction": direction,
            "confidence": self._get_confidence_level(base_confidence),
            "confidence_score": round(base_confidence, 2),
            "key_drivers": drivers[:4],
            "catalysts_identified": len(high_confidence_catalysts)
        }
    
    # Helper methods
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string in various formats."""
        formats = [
            "%Y-%m-%d",
            "%Y-%m-%d %H:%M:%S",
            "%B %d, %Y",
            "%b %d, %Y",
            "%m/%d/%Y"
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except:
                continue
        
        return None
    
    def _is_positive_sentiment(self, item: Dict[str, str]) -> bool:
        """Simple heuristic for positive sentiment."""
        positive_words = ["surge", "soar", "beat", "growth", "profit", "up", "gain", "positive", "strong"]
        return self._mentions(self._news_text(item), positive_words)
    
    def _is_negative_sentiment(self, item: Dict[str, str]) -> bool:
        """Simple heuristic for negative sentiment."""
        negative_words = ["fall", "drop", "miss", "loss", "down", "decline", "negative", "weak", "concern"]
        return self._mentions(self._news_text(item), negative_words)

    @staticmethod
    def _news_text(item: Dict[str, str]) -> str:
        """Lowercased title + summary of a news item."""
        return f"{item.get('title', '')} {item.get('summary', '')}".lower()

    @staticmethod
    def _mentions(text: str, keywords: List[str]) -> bool:
        """
        Case-insensitive whole-word keyword match, allowing simple suffixes
        (s/es/d/ed/ing) so "acquired" matches "acquire" but "sector" does
        not match "sec".
        """
        # ponytail: crude suffix list, swap for a stemmer or LLM tagging if recall matters
        return any(
            re.search(rf"\b{re.escape(kw.lower())}(?:s|es|d|ed|ing)?\b", text, re.IGNORECASE)
            for kw in keywords
        )
    
    def _estimate_catalyst_impact(self, catalyst_type: str, text: str) -> float:
        """Estimate price impact of catalyst based on type and context."""
        base_impact = self.event_weights.get(catalyst_type, 0.03)
        
        # Adjust based on context
        if "major" in text or "significant" in text:
            base_impact *= 1.5
        elif "minor" in text or "small" in text:
            base_impact *= 0.5
        
        return base_impact
    
    def _get_confidence_level(self, score: float) -> str:
        """Convert confidence score to level."""
        if score >= self.confidence_thresholds["high"]:
            return "high"
        elif score >= self.confidence_thresholds["medium"]:
            return "medium"
        else:
            return "low"
    
    def _empty_prediction(self, horizon: str) -> Dict[str, Any]:
        """Return empty prediction template."""
        return {
            "horizon": horizon,
            "target_price": 0.0,
            "expected_change_pct": 0.0,
            "direction": "unknown",
            "confidence": "low",
            "confidence_score": 0.0,
            "key_drivers": []
        }
