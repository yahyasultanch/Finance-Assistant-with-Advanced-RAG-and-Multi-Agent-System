"""Research tools the agents call, bound to one ticker's data.

Each tool returns text for the model plus registry sources; the text shows
every source with its [S...] id so the model can cite it.
"""
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field

from agents.citations import make_source, render_sources
from agents.llm import Tool, ToolOutput
from utils.xbrl import format_value

FormType = Literal["10-K", "10-Q", "8-K", "NEWS"]


@dataclass
class ResearchContext:
    """Everything the tools need for one ticker (kept out of graph state)."""
    ticker: str
    retriever: Any
    vector_store: Any
    news_items: List[Dict[str, Any]]
    company_info: Dict[str, Any]
    price_metrics: Dict[str, Any]
    metrics: List[Dict[str, Any]]          # utils.xbrl.key_metrics output
    cik: str = ""
    report: str = ""                       # latest brief, for the chat agent
    extra: Dict[str, Any] = field(default_factory=dict)

    def inventory(self) -> Dict[str, int]:
        """Chunk counts by form#section, so planners know what's indexed."""
        return dict(Counter(
            f"{d['metadata'].get('form_type')}#{d['metadata'].get('section', 'Document')}"
            for d in self.vector_store.documents
        ))


class SearchFilingsArgs(BaseModel):
    query: str = Field(description="What to look for, in natural language")
    form_types: List[FormType] = Field(default=[], description="Restrict to these forms; empty = all")
    sections: List[str] = Field(
        default=[],
        description="Restrict to sections, e.g. 'Business', 'Risk Factors', 'MD&A', "
                    "'Financial Statements', 'Exhibit 99.1' (8-K press release)",
    )
    k: int = Field(default=6, ge=1, le=10, description="Number of passages")


class LimitArgs(BaseModel):
    limit: int = Field(default=10, ge=1, le=30)


class NoArgs(BaseModel):
    pass


def _filing_source(doc: Dict[str, Any]) -> Dict[str, Any]:
    meta = doc.get("metadata", {})
    form = meta.get("form_type", "Doc")
    section = meta.get("section", "Document")
    is_news = form == "NEWS"
    return make_source(
        kind="news" if is_news else "filing",
        label=f"{form}#{meta.get('source', 'News')}" if is_news else f"{form}#{section}",
        title=f"{meta.get('ticker', '')} {form} {section}".strip(),
        url=meta.get("url", ""),
        date=meta.get("filing_date", ""),
        text=doc["text"],
        tier="social" if "reddit" in str(meta.get("source", "")).lower() else ("news" if is_news else "sec"),
    )


def build_tools(ctx: ResearchContext, sec_client) -> Dict[str, Tool]:
    """Create the tool set for one ticker."""

    def search_filings(args: SearchFilingsArgs) -> ToolOutput:
        filters = {}
        if args.form_types:
            filters["form_type"] = args.form_types
        if args.sections:
            filters["section"] = args.sections
        results = ctx.retriever.retrieve(args.query, k=args.k, filters=filters)
        if not results:
            # Tell the model why and what exists, so it can retry instead of giving up
            return ToolOutput(
                f"No passages matched filters {filters or 'none'} for this query. "
                f"Indexed chunks by form#section: {ctx.inventory()}. "
                "Retry with filters that exist above, or none."
            )
        sources = [_filing_source(doc) for doc, _ in results]
        return ToolOutput(render_sources(sources), sources)

    def get_financial_metrics(args: NoArgs) -> ToolOutput:
        revenue = next((m for m in ctx.metrics if m["name"] == "Revenue"), {})
        sources = []
        for metric in ctx.metrics:
            for period in ("annual", "quarter", "instant"):
                series = metric.get(period)
                if not series:
                    continue
                latest, prior = series["latest"], series["prior"]
                text = (
                    f"{metric['name']} ({metric['concept']}), {period} period "
                    f"{latest['frame']} ending {latest['end']}, from {latest['form']} "
                    f"filed {latest['filed']}: {format_value(latest['value'], metric['unit'])}."
                )
                if prior:
                    text += f" Same period a year earlier: {format_value(prior['value'], metric['unit'])}."
                if series["yoy_pct"] is not None:
                    text += f" Year-over-year change: {series['yoy_pct']:+.1f}%."
                # Margins, from revenue for the same frame
                rev = revenue.get(period, {}).get("latest", {})
                if (metric["name"] in ("Gross Profit", "Operating Income", "Net Income")
                        and rev.get("frame") == latest["frame"] and rev.get("value")):
                    text += f" Margin: {latest['value'] / rev['value'] * 100:.1f}% of revenue."
                url = (sec_client.filing_index_url(ctx.cik, latest["accession_number"])
                       if latest["accession_number"] else "")
                sources.append(make_source(
                    kind="xbrl", label=f"XBRL#{metric['name']}",
                    title=f"{ctx.ticker} {metric['name']} ({latest['form']}, {latest['frame']})",
                    url=url, date=latest["filed"], text=text,
                ))
        if not sources:
            return ToolOutput("No XBRL financial data available for this company.")
        return ToolOutput(render_sources(sources), sources)

    def get_recent_events(args: LimitArgs) -> ToolOutput:
        events = [n for n in ctx.news_items if n.get("source") == "SEC 8-K"][:args.limit]
        sources = [make_source(
            kind="event", label="8-K#Event", title=n["title"], url=n["url"],
            date=n["date"], text=f"{n['title']}. {n['summary']}",
        ) for n in events]
        return ToolOutput(render_sources(sources), sources)

    def get_news(args: LimitArgs) -> ToolOutput:
        news = [n for n in ctx.news_items if n.get("source") != "SEC 8-K"][:args.limit]
        sources = [make_source(
            kind="news", label=f"NEWS#{n['source']}", title=n["title"], url=n["url"],
            date=n["date"], text=f"{n['title']}. {n['summary']}",
            tier="social" if "reddit" in n["source"].lower() else "news",
        ) for n in news]
        return ToolOutput(render_sources(sources), sources)

    def get_price_summary(args: NoArgs) -> ToolOutput:
        p, c = ctx.price_metrics, ctx.company_info
        text = (
            f"{c.get('name', ctx.ticker)} ({c.get('sector', 'N/A')} / {c.get('industry', 'N/A')}). "
            f"Price {p.get('current_price', 0):.2f}; YTD return {p.get('ytd_return', 0):.1f}%; "
            f"52-week range {p.get('low_52w', 0):.2f}-{p.get('high_52w', 0):.2f}; "
            f"annualized volatility {p.get('volatility', 0):.1f}%; "
            f"trailing P/E {c.get('pe_ratio') or 'N/A'}; beta {c.get('beta') or 'N/A'}; "
            f"market cap {c.get('market_cap', 0):,.0f}."
        )
        source = make_source(
            kind="market_data", label="MARKET#Yahoo Finance", title=f"{ctx.ticker} market data",
            url=f"https://finance.yahoo.com/quote/{ctx.ticker}", date="live", text=text,
            tier="market_data",
        )
        return ToolOutput(render_sources([source]), [source])

    tools = [
        Tool("search_filings",
             "Hybrid (BM25 + semantic) search over indexed SEC filings, 8-K press releases "
             "and news. Filter by form type and section to target the right document.",
             SearchFilingsArgs, search_filings),
        Tool("get_financial_metrics",
             "Reported financials from SEC XBRL: revenue, gross/operating/net income with "
             "margins, EPS, R&D, cash flow, cash, debt, equity; latest annual and quarter "
             "with year-over-year change.",
             NoArgs, get_financial_metrics),
        Tool("get_recent_events",
             "Recent 8-K material events (earnings releases, agreements, executive changes).",
             LimitArgs, get_recent_events),
        Tool("get_news", "Recent news headlines and summaries from publishers.", LimitArgs, get_news),
        Tool("get_price_summary",
             "Current price, YTD return, 52-week range, volatility, P/E, beta, market cap.",
             NoArgs, get_price_summary),
    ]
    return {tool.name: tool for tool in tools}


# Researcher roles: what each one is for, and which tools it gets
ROLES = {
    "filings": (
        "Reads 10-K/10-Q/8-K text: business model, segments, strategy, risk factors, MD&A, guidance.",
        ["search_filings", "get_recent_events"],
    ),
    "financials": (
        "Reported numbers: growth, margins, profitability, cash and debt, plus valuation and price data.",
        ["get_financial_metrics", "get_price_summary", "search_filings"],
    ),
    "news": (
        "Recent news narrative and material events, including 8-K press releases.",
        ["get_news", "get_recent_events", "search_filings"],
    ),
}
