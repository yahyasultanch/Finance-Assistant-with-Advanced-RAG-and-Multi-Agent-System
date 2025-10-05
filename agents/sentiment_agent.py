"""Sentiment agent using FinBERT."""
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import numpy as np
from typing import List, Dict, Any, Tuple
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class SentimentAgent:
    """Agent for financial sentiment analysis using FinBERT."""

    def __init__(self, model_name: str = "ProsusAI/finbert"):
        """
        Initialize sentiment agent.
        
        Args:
            model_name: Name of FinBERT model
        """
        self.model_name = model_name
        self.sentiment_pipeline = None
        self._load_model()

    def _load_model(self):
        """Load FinBERT model and tokenizer."""
        try:
            logger.info(f"Loading sentiment model: {self.model_name}")
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                tokenizer=self.model_name,
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info("Sentiment model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading sentiment model: {e}")
            raise

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of a single text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dict with label, score, and confidence
        """
        # Truncate if too long (FinBERT has 512 token limit)
        text = text[:512]
        
        try:
            result = self.sentiment_pipeline(text)[0]
            label = result['label'].lower()
            score = result['score']
            
            # Map to standardized labels
            label_map = {
                'positive': 'positive',
                'negative': 'negative',
                'neutral': 'neutral'
            }
            
            return {
                "label": label_map.get(label, 'neutral'),
                "confidence": score,
                "raw_label": result['label']
            }
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {"label": "neutral", "confidence": 0.0, "raw_label": "error"}

    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze sentiment for multiple texts.
        
        Args:
            texts: List of texts
            
        Returns:
            List of sentiment results
        """
        # Truncate texts
        truncated = [text[:512] for text in texts]
        
        try:
            results = self.sentiment_pipeline(truncated)
            
            processed = []
            for result in results:
                label = result['label'].lower()
                score = result['score']
                
                processed.append({
                    "label": label,
                    "confidence": score,
                    "raw_label": result['label']
                })
            
            return processed
        except Exception as e:
            logger.error(f"Error in batch sentiment analysis: {e}")
            return [{"label": "neutral", "confidence": 0.0, "raw_label": "error"}] * len(texts)

    def analyze_news(self, news_items: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Analyze sentiment of news headlines.
        
        Args:
            news_items: List of news item dicts with 'title' and 'summary'
            
        Returns:
            Aggregated sentiment analysis
        """
        if not news_items:
            return self._empty_sentiment_result()
        
        # Extract texts to analyze
        texts = []
        for item in news_items:
            title = item.get('title', '')
            summary = item.get('summary', '')
            combined = f"{title}. {summary}"
            texts.append(combined)
        
        # Analyze
        results = self.analyze_batch(texts)
        
        # Aggregate
        return self._aggregate_sentiments(results, "news")

    def analyze_risk_factors(self, risk_texts: List[str]) -> Dict[str, Any]:
        """
        Analyze sentiment of risk factor excerpts.
        
        Args:
            risk_texts: List of risk factor text excerpts
            
        Returns:
            Aggregated sentiment analysis
        """
        if not risk_texts:
            return self._empty_sentiment_result()
        
        results = self.analyze_batch(risk_texts)
        return self._aggregate_sentiments(results, "risk_factors")

    def analyze_filing_sections(
        self,
        sections: Dict[str, str],
        max_excerpts_per_section: int = 5
    ) -> Dict[str, Any]:
        """
        Analyze sentiment of filing sections.
        
        Args:
            sections: Dict of section_name -> section_text
            max_excerpts_per_section: Number of excerpts to sample per section
            
        Returns:
            Aggregated sentiment analysis with per-section breakdown
        """
        all_results = []
        section_sentiments = {}
        
        for section_name, section_text in sections.items():
            # Extract excerpts (split into chunks)
            excerpts = self._extract_excerpts(section_text, max_excerpts_per_section)
            
            # Analyze
            results = self.analyze_batch(excerpts)
            all_results.extend(results)
            
            # Store section-level sentiment
            section_sentiments[section_name] = self._aggregate_sentiments(
                results, 
                f"section_{section_name}"
            )
        
        # Overall aggregation
        overall = self._aggregate_sentiments(all_results, "filing_sections")
        overall["section_breakdown"] = section_sentiments
        
        return overall

    def _aggregate_sentiments(
        self,
        results: List[Dict[str, Any]],
        source: str
    ) -> Dict[str, Any]:
        """
        Aggregate sentiment results.
        
        Args:
            results: List of individual sentiment results
            source: Source label
            
        Returns:
            Aggregated sentiment metrics
        """
        if not results:
            return self._empty_sentiment_result()
        
        # Count labels
        label_counts = {"positive": 0, "negative": 0, "neutral": 0}
        for result in results:
            label = result["label"]
            label_counts[label] = label_counts.get(label, 0) + 1
        
        total = len(results)
        
        # Calculate percentages
        percentages = {
            label: (count / total) * 100
            for label, count in label_counts.items()
        }
        
        # Calculate sentiment index: (positive - negative) / total
        sentiment_index = (label_counts["positive"] - label_counts["negative"]) / total
        
        # Bootstrap confidence interval for sentiment index
        ci_lower, ci_upper = self._bootstrap_ci(
            [1 if r["label"] == "positive" else -1 if r["label"] == "negative" else 0 
             for r in results]
        )
        
        return {
            "source": source,
            "total_analyzed": total,
            "counts": label_counts,
            "percentages": percentages,
            "sentiment_index": sentiment_index,
            "confidence_interval_95": (ci_lower, ci_upper),
            "predominant_sentiment": max(label_counts, key=label_counts.get)
        }

    def _bootstrap_ci(
        self,
        data: List[float],
        n_iterations: int = 1000,
        ci: float = 0.95
    ) -> Tuple[float, float]:
        """
        Calculate bootstrap confidence interval.
        
        Args:
            data: List of sentiment scores
            n_iterations: Number of bootstrap iterations
            ci: Confidence interval level
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if len(data) < 2:
            return (0.0, 0.0)
        
        means = []
        data_array = np.array(data)
        
        for _ in range(n_iterations):
            sample = np.random.choice(data_array, size=len(data), replace=True)
            means.append(np.mean(sample))
        
        alpha = 1 - ci
        lower = np.percentile(means, alpha / 2 * 100)
        upper = np.percentile(means, (1 - alpha / 2) * 100)
        
        return (float(lower), float(upper))

    def _extract_excerpts(self, text: str, max_excerpts: int) -> List[str]:
        """
        Extract representative excerpts from long text.
        
        Args:
            text: Long text
            max_excerpts: Maximum number of excerpts
            
        Returns:
            List of text excerpts
        """
        # Split into sentences
        sentences = text.split('. ')
        
        if len(sentences) <= max_excerpts:
            return sentences
        
        # Sample evenly
        step = len(sentences) // max_excerpts
        excerpts = [sentences[i * step] for i in range(max_excerpts)]
        
        return excerpts

    def _empty_sentiment_result(self) -> Dict[str, Any]:
        """Return empty sentiment result."""
        return {
            "source": "empty",
            "total_analyzed": 0,
            "counts": {"positive": 0, "negative": 0, "neutral": 0},
            "percentages": {"positive": 0.0, "negative": 0.0, "neutral": 0.0},
            "sentiment_index": 0.0,
            "confidence_interval_95": (0.0, 0.0),
            "predominant_sentiment": "neutral"
        }
