"""MCP server listing, safe-harbor filtering, news relevance, price tool keys."""
import json

import mcp_server
from agents.tools import ResearchContext, build_tools
from rag.chunking import is_safe_harbor
from utils.news_client import NewsClient


def test_mcp_lists_every_research_tool_with_ticker_required():
    tools = {t.name: t for t in mcp_server.list_tool_definitions()}
    assert set(mcp_server.RESEARCH_TOOLS) | {"generate_brief", "get_track_record"} == set(tools)
    search = tools["search_filings"].input_schema
    assert search["required"][:2] == ["ticker", "query"]
    assert "sections" in search["properties"]


def test_safe_harbor_disclaimers_are_dropped_but_guidance_is_kept():
    disclaimer = ("Certain statements in this press release including statements as to growth are "
                  "forward-looking statements within the meaning of Section 27A of the Securities Act.")
    guidance = "Revenue for the third quarter is expected to be $54.0 billion, plus or minus 2%."
    assert is_safe_harbor(disclaimer)
    assert not is_safe_harbor(guidance)


def test_news_relevance_needs_the_company_or_ticker():
    assert NewsClient._is_about("Can Amazon take share from Nvidia?", "NVDA", "NVIDIA")
    assert NewsClient._is_about("NVDA slips after earnings", "NVDA", "NVIDIA")
    assert not NewsClient._is_about("5 Halloween jump scares in your retirement account", "NVDA", "NVIDIA")


def test_price_tool_reads_the_52_week_keys_prices_produce():
    ctx = ResearchContext(
        ticker="NVDA", retriever=None, vector_store=None, news_items=[],
        company_info={"name": "NVIDIA"}, metrics=[],
        price_metrics={"current_price": 210.0, "ytd_return": 13.5,
                       "52_week_high": 239.24, "52_week_low": 164.79, "volatility": 37.7},
    )
    text = build_tools(ctx, None)["get_price_summary"].invoke(json.dumps({})).text
    assert "52-week range 164.79-239.24" in text
