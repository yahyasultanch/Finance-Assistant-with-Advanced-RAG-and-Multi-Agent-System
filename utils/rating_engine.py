"""Investment rating engine for buy/hold/sell recommendations."""
from typing import Dict, Any, List, Tuple
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class RatingEngine:
    """
    Multi-factor investment rating engine.
    
    Generates BUY/HOLD/SELL ratings based on:
    - Fundamental metrics (P/E, growth, profitability)
    - Sentiment analysis (FinBERT scores)
    - Technical indicators (momentum, volatility)
    - Risk assessment (from SEC filings)
    - Recent material events
    """
    
    # Sector average P/E ratios (approximate benchmarks)
    SECTOR_PE_RATIOS = {
        "Technology": 28.0,
        "Healthcare": 22.0,
        "Financial Services": 12.0,
        "Consumer Cyclical": 20.0,
        "Consumer Defensive": 18.0,
        "Industrials": 20.0,
        "Energy": 15.0,
        "Utilities": 18.0,
        "Real Estate": 25.0,
        "Communication Services": 16.0,
        "Basic Materials": 15.0
    }
    
    def __init__(self):
        """Initialize rating engine."""
        self.weights = {
            "fundamentals": 0.35,    # 35% - Financial metrics
            "sentiment": 0.25,       # 25% - News & market sentiment
            "technical": 0.20,       # 20% - Price momentum
            "risk": 0.20             # 20% - Risk factors
        }
    
    def calculate_rating(
        self,
        ticker: str,
        price_data: Dict[str, Any],
        company_info: Dict[str, Any],
        sentiment_results: Dict[str, Any],
        summaries: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive investment rating.
        
        Args:
            ticker: Stock ticker symbol
            price_data: Price metrics (current_price, ytd_return, etc.)
            company_info: Company info (sector, industry, market_cap, pe_ratio, etc.)
            sentiment_results: Sentiment analysis results
            summaries: RAG summaries (risks, growth drivers, outlook)
            
        Returns:
            Dict with rating, score, price_target, recommendation, and reasoning
        """
        logger.info(f"Calculating investment rating for {ticker}")
        
        # Calculate component scores
        fundamental_score = self._score_fundamentals(price_data, company_info)
        sentiment_score = self._score_sentiment(sentiment_results)
        technical_score = self._score_technical(price_data)
        risk_score = self._score_risk(summaries.get("risks", {}))
        
        # Composite score (0-100)
        composite_score = (
            fundamental_score * self.weights["fundamentals"] +
            sentiment_score * self.weights["sentiment"] +
            technical_score * self.weights["technical"] +
            risk_score * self.weights["risk"]
        )
        
        # Determine rating
        rating, rating_color = self._get_rating_from_score(composite_score)
        
        # Calculate price target
        current_price = price_data.get("current_price", 0)
        price_target = self._calculate_price_target(
            current_price, 
            company_info, 
            composite_score
        )
        upside = ((price_target - current_price) / current_price * 100) if current_price > 0 else 0
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            rating,
            fundamental_score,
            sentiment_score,
            technical_score,
            risk_score,
            price_data,
            company_info,
            sentiment_results,
            summaries
        )
        
        return {
            "rating": rating,
            "rating_color": rating_color,
            "composite_score": round(composite_score, 1),
            "component_scores": {
                "fundamentals": round(fundamental_score, 1),
                "sentiment": round(sentiment_score, 1),
                "technical": round(technical_score, 1),
                "risk": round(risk_score, 1)
            },
            "price_target": round(price_target, 2),
            "current_price": current_price,
            "upside_percent": round(upside, 2),
            "reasoning": reasoning,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def _score_fundamentals(self, price_data: Dict, company_info: Dict) -> float:
        """Score based on fundamental metrics (0-100)."""
        score = 50  # Start at neutral
        
        # P/E Ratio vs sector average
        pe_ratio = company_info.get("pe_ratio")
        sector = company_info.get("sector", "Technology")
        sector_pe = self.SECTOR_PE_RATIOS.get(sector, 20.0)
        
        if pe_ratio and pe_ratio > 0:
            pe_ratio_score = max(0, min(100, 100 - ((pe_ratio / sector_pe - 1) * 50)))
            score += (pe_ratio_score - 50) * 0.3
        
        # YTD Return (momentum)
        ytd_return = price_data.get("ytd_return", 0)
        if ytd_return > 20:
            score += 15
        elif ytd_return > 10:
            score += 10
        elif ytd_return > 0:
            score += 5
        elif ytd_return < -20:
            score -= 15
        elif ytd_return < -10:
            score -= 10
        
        # Market cap (liquidity preference for large caps)
        market_cap = company_info.get("market_cap", 0)
        if market_cap > 1_000_000_000_000:  # > $1T
            score += 5
        elif market_cap > 100_000_000_000:  # > $100B
            score += 3
        
        return max(0, min(100, score))
    
    def _score_sentiment(self, sentiment_results: Dict) -> float:
        """Score based on sentiment analysis (0-100)."""
        if not sentiment_results:
            return 50  # Neutral default
        
        sentiment_index = sentiment_results.get("sentiment_index", 0)
        
        # Convert sentiment index (-1 to +1) to score (0-100)
        # -1 = 0, 0 = 50, +1 = 100
        score = (sentiment_index + 1) * 50
        
        # Adjust based on predominant sentiment
        predominant = sentiment_results.get("predominant_sentiment", "neutral").lower()
        if predominant == "positive":
            score += 5
        elif predominant == "negative":
            score -= 5
        
        return max(0, min(100, score))
    
    def _score_technical(self, price_data: Dict) -> float:
        """Score based on technical indicators (0-100)."""
        score = 50  # Start at neutral
        
        # Price vs 52-week high/low (momentum)
        current_price = price_data.get("current_price", 0)
        high_52w = price_data.get("high_52w", current_price)
        low_52w = price_data.get("low_52w", current_price)
        
        if high_52w > low_52w:
            price_position = (current_price - low_52w) / (high_52w - low_52w)
            score += (price_position - 0.5) * 40  # ±20 points based on position
        
        # Volatility (prefer lower volatility for stability)
        volatility = price_data.get("volatility", 30)
        if volatility < 20:
            score += 10
        elif volatility < 30:
            score += 5
        elif volatility > 50:
            score -= 10
        elif volatility > 40:
            score -= 5
        
        return max(0, min(100, score))
    
    def _score_risk(self, risk_summary: Dict) -> float:
        """Score based on risk assessment (0-100, higher = lower risk)."""
        score = 60  # Start slightly above neutral
        
        # Check if risks were found
        summary_text = risk_summary.get("summary", "").lower()
        
        # Negative keywords (reduce score)
        negative_keywords = [
            "significant risk", "material risk", "uncertainty", "volatile",
            "litigation", "regulatory", "competition", "supply chain",
            "cybersecurity", "geopolitical"
        ]
        
        negative_count = sum(1 for keyword in negative_keywords if keyword in summary_text)
        score -= negative_count * 5  # -5 points per negative keyword
        
        # Positive keywords (increase score)
        positive_keywords = [
            "diversified", "strong balance", "market leader", "competitive advantage",
            "intellectual property", "brand value"
        ]
        
        positive_count = sum(1 for keyword in positive_keywords if keyword in summary_text)
        score += positive_count * 3  # +3 points per positive keyword
        
        return max(0, min(100, score))
    
    def _get_rating_from_score(self, score: float) -> Tuple[str, str]:
        """Convert composite score to rating."""
        if score >= 70:
            return "BUY", "🟢"
        elif score >= 45:
            return "HOLD", "🟡"
        else:
            return "SELL", "🔴"
    
    def _calculate_price_target(
        self,
        current_price: float,
        company_info: Dict,
        composite_score: float
    ) -> float:
        """Calculate 12-month price target."""
        if current_price <= 0:
            return 0
        
        # Base target on composite score
        # Score 50 = 0% change, Score 70 = +10%, Score 90 = +20%
        # Score 30 = -10%, Score 10 = -20%
        target_change = (composite_score - 50) / 40 * 0.20  # ±20% max
        
        # Adjust for sector growth expectations
        sector = company_info.get("sector", "Technology")
        sector_adjustment = {
            "Technology": 0.05,  # Tech tends to have higher growth
            "Healthcare": 0.03,
            "Energy": 0.02,
            "Utilities": 0.01,
        }.get(sector, 0.02)
        
        total_change = target_change + sector_adjustment
        price_target = current_price * (1 + total_change)
        
        return price_target
    
    def _generate_reasoning(
        self,
        rating: str,
        fundamental_score: float,
        sentiment_score: float,
        technical_score: float,
        risk_score: float,
        price_data: Dict,
        company_info: Dict,
        sentiment_results: Dict,
        summaries: Dict
    ) -> List[str]:
        """Generate reasoning bullets for the rating."""
        reasoning = []
        
        # Fundamental reasoning
        if fundamental_score >= 70:
            pe_ratio = company_info.get("pe_ratio", 0)
            sector = company_info.get("sector", "")
            reasoning.append(f"**Strong fundamentals**: Attractive valuation metrics with P/E ratio of {pe_ratio:.2f} in {sector} sector")
        elif fundamental_score <= 40:
            reasoning.append("**Valuation concerns**: Current metrics suggest potential overvaluation relative to sector peers")
        
        # Growth momentum
        ytd_return = price_data.get("ytd_return", 0)
        if ytd_return > 20:
            reasoning.append(f"**Strong momentum**: YTD return of {ytd_return:.1f}% demonstrates robust market performance")
        elif ytd_return < -10:
            reasoning.append(f"**Underperformance**: YTD return of {ytd_return:.1f}% raises concerns about near-term prospects")
        
        # Sentiment reasoning
        if sentiment_score >= 65:
            reasoning.append("**Positive sentiment**: Market and news sentiment indicates bullish outlook from analysts and investors")
        elif sentiment_score <= 40:
            reasoning.append("**Negative sentiment**: Recent news and market sentiment suggest cautious positioning")
        
        # Technical reasoning
        current_price = price_data.get("current_price", 0)
        high_52w = price_data.get("high_52w", 0)
        if high_52w > 0:
            pct_from_high = ((current_price - high_52w) / high_52w * 100)
            if pct_from_high > -5:
                reasoning.append(f"**Near 52-week highs**: Trading at {abs(pct_from_high):.1f}% from peak suggests continued strength")
            elif pct_from_high < -20:
                reasoning.append(f"**Below recent highs**: Price is {abs(pct_from_high):.1f}% off 52-week high, potential for recovery")
        
        # Risk reasoning
        if risk_score >= 70:
            reasoning.append("**Manageable risk profile**: Risk factors appear well-contained with strong competitive positioning")
        elif risk_score <= 40:
            reasoning.append("**Elevated risks**: Significant headwinds including regulatory, competitive, or operational challenges")
        
        # Add specific insights from summaries
        growth_summary = summaries.get("drivers", {}).get("summary", "")
        if "strong" in growth_summary.lower() or "growth" in growth_summary.lower():
            reasoning.append("**Growth catalysts identified**: SEC filings highlight clear drivers for future revenue expansion")
        
        return reasoning[:5]  # Limit to top 5 reasoning points
