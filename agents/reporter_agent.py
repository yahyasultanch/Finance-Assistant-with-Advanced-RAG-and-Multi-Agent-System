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
        news_items: List[Dict[str, str]]
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
            
        Returns:
            Markdown text of the brief
        """
        # Build report sections
        report_sections = []
        
        # Header
        report_sections.append(self._generate_header(ticker, company_info))
        
        # Executive Summary
        report_sections.append(self._generate_executive_summary(ticker, summaries))
        
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
        
        # Sentiment Analysis
        report_sections.append(self._generate_sentiment_section(sentiment_results, news_items))
        
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
            for i, citation in enumerate(all_citations, 1):
                source = citation.get("source", "Unknown")
                url = citation.get("url", "#")
                date = citation.get("date", "N/A")
                section += f"{i}. [{source} - {date}]({url})\n"
        else:
            section += "*No sources available.*\n"
        
        return section

    def _generate_footer(self) -> str:
        """Generate report footer."""
        return """---

## Disclaimer

This report is generated using artificial intelligence and public data sources. It is for informational and educational purposes only and should not be considered as investment advice. Always conduct your own due diligence and consult with financial professionals before making investment decisions.

**Data Sources:** SEC EDGAR, Yahoo Finance, AI-generated analysis  
**Report Generated:** """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
