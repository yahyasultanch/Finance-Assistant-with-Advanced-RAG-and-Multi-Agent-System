"""Reporter agent for generating analyst briefs."""
from typing import Dict, Any, List
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ReporterAgent:
    """Agent for assembling analyst briefs."""

    def __init__(self, output_dir: Path):
        """
        Initialize reporter agent.
        
        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_brief(
        self,
        ticker: str,
        company_info: Dict[str, Any],
        price_metrics: Dict[str, Any],
        summaries: Dict[str, Dict[str, Any]],
        sentiment_results: Dict[str, Any],
        news_items: List[Dict[str, str]],
        rating_results: Dict[str, Any] = None,
        prediction_results: Dict[str, Any] = None,
        verification: Dict[str, Any] = None,
        review_note: str = "",
        track_record: Dict[str, Any] = None
    ) -> str:
        """
        Generate analyst brief in markdown.
        
        Args:
            ticker: Stock ticker
            company_info: Company information
            price_metrics: Price and performance metrics
            summaries: Dict of query -> summary results
            sentiment_results: Sentiment analysis results
            news_items: List of news items
            rating_results: Investment rating results (optional)
            prediction_results: Predictive forecasts (optional)
            verification: Claim verification results from the verifier (optional)
            review_note: Human reviewer's note (optional)
            track_record: Prediction ledger summary (optional)
            
        Returns:
            Markdown text of the brief
        """
        # Build report sections
        report_sections = []
        
        # Header
        report_sections.append(self._generate_header(ticker, company_info))
        
        # Executive Summary
        report_sections.append(self._generate_executive_summary(ticker, summaries))
        
        if review_note:
            report_sections.append(f"## Reviewer Note\n\n{review_note}\n")
        
        # Investment Recommendation (new - after executive summary)
        if rating_results:
            report_sections.append(self._generate_recommendation_section(rating_results))
        
        # Company Snapshot
        report_sections.append(self._generate_snapshot(company_info, price_metrics))
        
        # Business Overview
        if "overview" in summaries:
            report_sections.append(self._generate_section(
                "Business Overview",
                summaries["overview"]
            ))
        
        # Growth Drivers
        if "drivers" in summaries:
            report_sections.append(self._generate_section(
                "Key Growth Drivers",
                summaries["drivers"]
            ))
        
        # Risk Factors
        if "risks" in summaries:
            report_sections.append(self._generate_section(
                "Risk Factors",
                summaries["risks"]
            ))
        
        # Recent Events
        if "events" in summaries:
            report_sections.append(self._generate_section(
                "Recent Material Events",
                summaries["events"]
            ))
        
        # Outlook
        if "outlook" in summaries:
            report_sections.append(self._generate_section(
                "Financial Outlook",
                summaries["outlook"]
            ))

        # Changes vs the previous brief (only when memory had one and claims verified)
        if summaries.get("changes", {}).get("citations"):
            report_sections.append(self._generate_section(
                "Changes Since Last Brief",
                summaries["changes"]
            ))

        # Sentiment Analysis
        report_sections.append(self._generate_sentiment_section(sentiment_results, news_items))

        # Forward Outlook & Predictions (new)
        if prediction_results:
            report_sections.append(self._generate_predictions_section(prediction_results))

        if track_record:
            report_sections.append(self._generate_track_record_section(track_record))

        if verification:
            report_sections.append(self._generate_verification_section(verification))
        
        # Appendix
        report_sections.append(self._generate_appendix(summaries))
        
        # Footer
        report_sections.append(self._generate_footer())
        
        # Combine
        full_report = "\n\n---\n\n".join(report_sections)
        
        return full_report

    def save_brief(
        self,
        ticker: str,
        report_content: str,
        format: str = "markdown"
    ) -> str:
        """
        Save brief to file.
        
        Args:
            ticker: Stock ticker
            report_content: Report markdown content
            format: Output format (markdown, html, pdf)
            
        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == "markdown":
            filename = f"{ticker}_analyst_brief_{timestamp}.md"
            filepath = self.output_dir / filename
            filepath.write_text(report_content, encoding='utf-8')
            logger.info(f"Saved markdown report to: {filepath}")
            return str(filepath)
        
        elif format == "html":
            from utils.md_to_pdf import MarkdownToPDF
            converter = MarkdownToPDF()
            filename = f"{ticker}_analyst_brief_{timestamp}.html"
            filepath = self.output_dir / filename
            converter.convert(
                report_content,
                str(filepath).replace('.html', '.pdf'),
                title=f"{ticker} Analyst Brief"
            )
            return str(filepath)
        
        return ""

    def _generate_header(self, ticker: str, company_info: Dict[str, Any]) -> str:
        """Generate report header."""
        company_name = company_info.get("name", ticker)
        sector = company_info.get("sector", "N/A")
        industry = company_info.get("industry", "N/A")
        date = datetime.now().strftime("%B %d, %Y")
        
        return f"""# {ticker.upper()} - Analyst Brief

**Company:** {company_name}  
**Sector:** {sector} | **Industry:** {industry}  
**Report Date:** {date}  
**Report Type:** AI-Generated Investment Research Brief

---
"""

    def _generate_executive_summary(
        self,
        ticker: str,
        summaries: Dict[str, Dict[str, Any]]
    ) -> str:
        """Generate executive summary."""
        summary_text = f"## Executive Summary\n\n"
        summary_text += f"This brief provides an overview of **{ticker}** based on recent SEC filings and news analysis. "
        summary_text += "Key insights cover business operations, growth drivers, risk factors, and sentiment analysis.\n"
        
        return summary_text
    
    def _generate_recommendation_section(
        self,
        rating_results: Dict[str, Any]
    ) -> str:
        """Generate investment recommendation section."""
        section = "## Investment Recommendation\n\n"
        
        # Rating badge and key metrics
        rating = rating_results.get("rating", "HOLD")
        rating_color = rating_results.get("rating_color", "🟡")
        current_price = rating_results.get("current_price", 0)
        price_target = rating_results.get("price_target", 0)
        upside = rating_results.get("upside_percent", 0)
        composite_score = rating_results.get("composite_score", 50)
        
        section += f"### {rating_color} **{rating}**\n\n"
        
        # Key metrics table
        section += "| Metric | Value |\n"
        section += "|--------|-------|\n"
        section += f"| **Rating** | {rating} |\n"
        section += f"| **Current Price** | ${current_price:.2f} |\n"
        section += f"| **12M Price Target** | ${price_target:.2f} |\n"
        
        if upside >= 0:
            section += f"| **Upside Potential** | +{upside:.1f}% |\n"
        else:
            section += f"| **Downside Risk** | {upside:.1f}% |\n"
        
        section += f"| **Composite Score** | {composite_score:.1f}/100 |\n\n"
        
        # Component scores breakdown
        component_scores = rating_results.get("component_scores", {})
        if component_scores:
            section += "**Rating Components:**\n\n"
            section += "| Factor | Score | Weight |\n"
            section += "|--------|-------|--------|\n"
            section += f"| Fundamentals | {component_scores.get('fundamentals', 50):.1f}/100 | 35% |\n"
            section += f"| Sentiment | {component_scores.get('sentiment', 50):.1f}/100 | 25% |\n"
            section += f"| Technical | {component_scores.get('technical', 50):.1f}/100 | 20% |\n"
            section += f"| Risk | {component_scores.get('risk', 50):.1f}/100 | 20% |\n\n"
        
        # Reasoning bullets
        reasoning = rating_results.get("reasoning", [])
        if reasoning:
            section += "**Investment Rationale:**\n\n"
            for point in reasoning:
                section += f"- {point}\n"
            section += "\n"
        
        section += "---\n\n"
        
        return section

    def _generate_snapshot(
        self,
        company_info: Dict[str, Any],
        price_metrics: Dict[str, Any]
    ) -> str:
        """Generate company snapshot table."""
        snapshot = "## Company Snapshot\n\n"
        
        # Build KPI table
        snapshot += "| Metric | Value |\n"
        snapshot += "|--------|-------|\n"
        
        # Add metrics
        if "current_price" in price_metrics:
            snapshot += f"| **Current Price** | ${price_metrics['current_price']:.2f} |\n"
        
        if "market_cap" in company_info:
            market_cap = company_info['market_cap']
            if market_cap > 1e9:
                snapshot += f"| **Market Cap** | ${market_cap/1e9:.2f}B |\n"
            elif market_cap > 1e6:
                snapshot += f"| **Market Cap** | ${market_cap/1e6:.2f}M |\n"
        
        if "ytd_return" in price_metrics:
            snapshot += f"| **YTD Return** | {price_metrics['ytd_return']:.2f}% |\n"
        
        if "52_week_high" in price_metrics:
            snapshot += f"| **52-Week High** | ${price_metrics['52_week_high']:.2f} |\n"
        
        if "52_week_low" in price_metrics:
            snapshot += f"| **52-Week Low** | ${price_metrics['52_week_low']:.2f} |\n"
        
        if "pe_ratio" in company_info and company_info["pe_ratio"]:
            snapshot += f"| **P/E Ratio** | {company_info['pe_ratio']:.2f} |\n"
        
        if "volatility" in price_metrics:
            snapshot += f"| **Volatility (Ann.)** | {price_metrics['volatility']:.2f}% |\n"
        
        if "beta" in company_info and company_info["beta"]:
            snapshot += f"| **Beta** | {company_info['beta']:.2f} |\n"
        
        return snapshot

    def _generate_section(
        self,
        title: str,
        summary_result: Dict[str, Any]
    ) -> str:
        """Generate a report section with citations."""
        section = f"## {title}\n\n"
        
        # Add summary text
        summary_text = summary_result.get("summary", "No information available.")
        section += summary_text + "\n\n"
        
        # Add citations if present
        citations = summary_result.get("citations", [])
        if citations:
            section += "\n**Sources:**\n"  # Single newline after summary, then Sources header
            for citation in citations:
                ref = citation.get("reference", "")
                source = citation.get("source", "")
                # Only add citation if both reference and source are non-empty
                if ref and source:
                    # Escape brackets in reference to prevent markdown link interpretation
                    ref_escaped = ref.replace('[', '\\[').replace(']', '\\]')
                    section += f"- {ref_escaped}: {source}\n"
                else:
                    logger.warning(f"Empty citation in {title}: ref={ref}, source={source}")
        
        return section

    def _generate_sentiment_section(
        self,
        sentiment_results: Dict[str, Any],
        news_items: List[Dict[str, str]]
    ) -> str:
        """Generate sentiment analysis section."""
        section = "## Sentiment Analysis\n\n"
        
        if not sentiment_results or sentiment_results.get("total_analyzed", 0) == 0:
            section += "*No sentiment data available.*\n"
            return section
        
        # Summary
        total = sentiment_results.get("total_analyzed", 0)
        percentages = sentiment_results.get("percentages", {})
        sentiment_index = sentiment_results.get("sentiment_index", 0)
        predominant = sentiment_results.get("predominant_sentiment", "neutral")
        
        section += f"**Analysis Summary** ({total} items analyzed)\n\n"
        
        # Sentiment distribution
        section += "| Sentiment | Count | Percentage |\n"
        section += "|-----------|-------|------------|\n"
        
        counts = sentiment_results.get("counts", {})
        for label in ["positive", "neutral", "negative"]:
            count = counts.get(label, 0)
            pct = percentages.get(label, 0.0)
            emoji = "🟢" if label == "positive" else "🟡" if label == "neutral" else "🔴"
            section += f"| {emoji} {label.capitalize()} | {count} | {pct:.1f}% |\n"
        
        # Sentiment index
        section += f"\n**Sentiment Index:** {sentiment_index:.3f} (95% CI: "
        ci = sentiment_results.get("confidence_interval_95", (0, 0))
        section += f"[{ci[0]:.3f}, {ci[1]:.3f}])\n\n"
        
        section += f"**Predominant Sentiment:** {predominant.capitalize()}\n\n"
        
        # Interpretation - fixed logic to handle zero correctly
        if sentiment_index > 0.3:
            interpretation = "**Strong positive** sentiment across news and filings."
        elif sentiment_index > 0.05:
            interpretation = "**Moderately positive** sentiment."
        elif sentiment_index >= -0.05:
            interpretation = "**Neutral** sentiment overall."
        elif sentiment_index > -0.3:
            interpretation = "**Moderately negative** sentiment."
        else:
            interpretation = "**Strong negative** sentiment."
        
        section += f"**Interpretation:** {interpretation}\n"
        
        return section
    
    def _generate_predictions_section(
        self,
        prediction_results: Dict[str, Any]
    ) -> str:
        """Generate forward outlook & predictions section."""
        section = "## Forward Outlook & Predictions\n\n"
        
        predictions = prediction_results.get("predictions", {})
        
        # Short-term prediction
        short_term = predictions.get("short_term", {})
        if short_term:
            section += "### 📅 Short-Term Forecast (1-7 Days)\n\n"
            section += f"**Target Price:** ${short_term.get('target_price', 0):.2f}\n\n"
            section += f"**Expected Move:** {short_term.get('expected_change_pct', 0):+.2f}%\n\n"
            section += f"**Direction:** {short_term.get('direction', 'sideways').capitalize()}\n\n"
            
            confidence = short_term.get('confidence', 'low').upper()
            confidence_emoji = "🟢" if confidence == "HIGH" else "🟡" if confidence == "MEDIUM" else "🔴"
            section += f"**Confidence:** {confidence_emoji} {confidence}\n\n"
            
            drivers = short_term.get('key_drivers', [])
            if drivers:
                section += "**Key Drivers:**\n"
                for driver in drivers:
                    section += f"- {driver}\n"
                section += "\n"
        
        # Long-term prediction
        long_term = predictions.get("long_term", {})
        if long_term:
            section += "### � Long-Term Forecast (3-12 Months)\n\n"
            section += f"**Target Price:** ${long_term.get('target_price', 0):.2f}\n\n"
            
            # Show risk-adjusted range if available
            upside = long_term.get('upside_target')
            downside = long_term.get('downside_target')
            if upside and downside:
                section += f"**Risk-Adjusted Range:** ${downside:.2f} - ${upside:.2f}\n\n"
            
            section += f"**Expected Move:** {long_term.get('expected_change_pct', 0):+.2f}%\n\n"
            section += f"**Direction:** {long_term.get('direction', 'sideways').capitalize()}\n\n"
            
            confidence = long_term.get('confidence', 'low').upper()
            confidence_emoji = "🟢" if confidence == "HIGH" else "🟡" if confidence == "MEDIUM" else "🔴"
            section += f"**Confidence:** {confidence_emoji} {confidence}  \n\n"
            
            # Show catalysts if integrated
            catalysts_count = long_term.get('catalysts_identified', 0)
            if catalysts_count > 0:
                section += f"**High-Confidence Catalysts:** {catalysts_count}  \n\n"
            
            drivers = long_term.get('key_drivers', [])
            if drivers:
                section += "**Key Drivers:**\n"
                for driver in drivers:
                    section += f"- {driver}\n"
                section += "\n"
        
        # Sentiment Momentum Analysis
        sentiment_momentum = prediction_results.get("sentiment_momentum", {})
        if sentiment_momentum:
            section += "### 📈 Sentiment Momentum Analysis\n\n"
            
            trend = sentiment_momentum.get("trend", "stable")
            velocity = sentiment_momentum.get("velocity", 0)
            
            trend_emoji = "📈" if trend == "improving" else "📉" if trend == "deteriorating" else "➡️"
            section += f"**Trend:** {trend_emoji} {trend.capitalize()}  \n"
            section += f"**Velocity:** {velocity:+.3f} (rate of sentiment change)  \n\n"
        
        # Detected Catalysts
        catalysts = prediction_results.get("catalysts", [])
        if catalysts:
            section += "### 🎯 Upcoming Catalysts\n\n"
            section += "| Event Type | Source | Impact Estimate | Confidence |\n"
            section += "|------------|--------|-----------------|------------|\n"
            
            for catalyst in catalysts[:5]:  # Top 5
                event_type = catalyst.get("type", "").replace("_", " ").title()
                source = catalyst.get("source", "").replace("_", " ").title()
                impact = catalyst.get("impact_estimate", 0)
                confidence = catalyst.get("confidence", 0)
                
                impact_str = f"{impact*100:+.1f}%"
                confidence_str = f"{confidence*100:.0f}%"
                
                section += f"| {event_type} | {source} | {impact_str} | {confidence_str} |\n"
            
            section += "\n"
        
        # Macro Factors
        macro_factors = prediction_results.get("macro_factors", {})
        detected_factors = macro_factors.get("detected_factors", [])
        if detected_factors:
            section += "### 🌍 Macro & Geopolitical Factors\n\n"
            
            for factor in detected_factors:
                factor_type = factor.get("type", "").replace("_", " ").title()
                mentions = factor.get("mentions", 0)
                sentiment_label = factor.get("sentiment", "neutral").capitalize()
                impact = factor.get("impact", "low").upper()
                
                sentiment_emoji = "🟢" if sentiment_label == "Positive" else "🔴" if sentiment_label == "Negative" else "🟡"
                
                section += f"- **{factor_type}**: {mentions} mentions | {sentiment_emoji} {sentiment_label} sentiment | Impact: {impact}  \n"
            
            section += "\n"
            
            macro_risk = macro_factors.get("overall_macro_risk", "moderate").upper()
            section += f"**Overall Macro Risk Level:** {macro_risk}  \n\n"
        
        # Disclaimer
        section += "---\n\n"
        section += "**Prediction Disclaimer:** Forecasts are probabilistic estimates based on historical data, sentiment analysis, "
        section += "and identified catalysts. Actual outcomes may vary significantly due to unforeseen events, market volatility, "
        section += "and other factors beyond the model's scope. These predictions should be used as directional guidance only.\n"
        
        return section

    def _generate_appendix(self, summaries: Dict[str, Dict[str, Any]]) -> str:
        """Generate appendix with source links."""
        section = "## Appendix: Data Sources\n\n"
        
        # Collect all unique citations
        all_citations = []
        for summary_result in summaries.values():
            citations = summary_result.get("citations", [])
            for citation in citations:
                if citation not in all_citations:
                    all_citations.append(citation)
        
        if all_citations:
            section += "### SEC Filings & Sources\n\n"
            tiers = {1: "SEC", 2: "Market data", 3: "News", 4: "Social", 5: "Prior brief"}
            for i, citation in enumerate(all_citations, 1):
                number = citation.get("number", i)
                source = citation.get("source", "Unknown")
                url = citation.get("url", "#")
                date = citation.get("date", "N/A")
                tier = tiers.get(citation.get("tier"), "")
                tier = f" *({tier})*" if tier else ""
                section += f"{number}. [{source} - {date}]({url}){tier}\n"
        else:
            section += "*No sources available.*\n"
        
        return section

    def _generate_track_record_section(self, track_record: Dict[str, Any]) -> str:
        """How earlier price calls for this ticker turned out (prediction ledger)."""
        section = "## Prediction Track Record\n\n"
        if not track_record.get("scored"):
            return section + (f"No earlier calls have reached their horizon yet "
                              f"({track_record.get('pending', 0)} pending).\n")
        section += (
            f"{track_record['scored']} earlier calls scored against actual prices: direction "
            f"correct **{track_record['direction_hit_rate']:.0%}**, mean absolute error "
            f"**{track_record['mean_abs_error_pct']:.1f}%** ({track_record['pending']} pending).\n\n"
            "| Call | Scored | Direction correct | Mean abs. error |\n"
            "|------|--------|-------------------|-----------------|\n"
        )
        names = {"7d": "7-day", "12m": "12-month", "rating_target": "Rating price target"}
        for kind, stats in track_record["by_kind"].items():
            if stats["scored"]:
                section += (f"| {names[kind]} | {stats['scored']} | {stats['direction_hit_rate']:.0%} "
                            f"| {stats['mean_abs_error_pct']:.1f}% |\n")
        return section

    def _generate_verification_section(self, verification: Dict[str, Any]) -> str:
        """Claim verification summary: coverage and what was removed."""
        section = "## Citation Verification\n\n"
        section += (
            f"**{verification['verified']} of {verification['total']} drafted claims verified "
            f"({verification['coverage']:.0%})**; first draft: {verification['first_pass_coverage']:.0%}, "
            f"revision rounds: {verification.get('revisions', 0)}. Each claim's cited source ids must "
            "exist, its figures must appear in those sources, and an LLM judge must confirm the "
            "sources support it. Claims that failed were removed from this report:\n\n"
        )
        failed = [c for c in verification["claims"] if c["status"] != "verified"]
        for claim in failed:
            section += f"- ~~{claim['text']}~~ ({claim['status']}: {claim['reason']})\n"
        if not failed:
            section += "*None.*\n"
        return section

    def _generate_footer(self) -> str:
        """Generate report footer."""
        return """---

## Disclaimer

This report is generated using artificial intelligence and public data sources. It is for informational and educational purposes only and should not be considered as investment advice. Always conduct your own due diligence and consult with financial professionals before making investment decisions.

**Data Sources:** SEC EDGAR, Yahoo Finance, AI-generated analysis  
**Report Generated:** """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
