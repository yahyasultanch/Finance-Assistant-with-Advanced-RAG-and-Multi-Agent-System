"""MCP server: the agents' research tools, served to any MCP client.

    python mcp_server.py        # stdio transport

The same Tool objects the LangGraph researchers call are listed here, with
their Pydantic argument schemas plus a `ticker` argument, so Claude Desktop,
Claude Code or any other MCP client can research a company with them, run a
full verified brief, or read the prediction track record.

Written on the low-level MCP server API (list_tools / call_tool handlers)
rather than decorated functions, because the tools already carry their own
JSON schemas: one generic handler serves all of them.
"""
import contextlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import anyio
import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

sys.path.insert(0, str(Path(__file__).parent))

from agents.memory import format_track_record  # noqa: E402
from agents.tools import build_tools  # noqa: E402

# stdout carries the protocol, so logs go to stderr
logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("mcp_server")

TICKER = {"ticker": {"type": "string", "description": "Stock ticker, e.g. NVDA"}}

# Tool objects are closures over a ticker's data; building them for no ticker
# gives the names, descriptions and schemas without loading anything
RESEARCH_TOOLS = build_tools(None, None)

EXTRA_TOOLS = {
    "generate_brief": (
        "Run the full multi-agent research graph for a ticker (planner, parallel "
        "researchers, analyst, citation verifier) and return the verified brief. "
        "Slow: 30-120 seconds, and indexes the company on first use.",
        {},
    ),
    "get_track_record": (
        "Earlier briefs remembered for a ticker and how their price forecasts turned "
        "out against actual prices.",
        {},
    ),
}

_orchestrator = None


def orchestrator():
    """Created on first call: loading models takes a while and listing tools doesn't need them."""
    global _orchestrator
    if _orchestrator is None:
        from agents.orchestrator import Orchestrator
        _orchestrator = Orchestrator.from_config()
    return _orchestrator


def with_ticker(schema: Dict[str, Any]) -> Dict[str, Any]:
    properties = {**TICKER, **schema.get("properties", {})}
    return {**schema, "type": "object", "properties": properties,
            "required": ["ticker", *schema.get("required", [])]}


def list_tool_definitions():
    tools = [types.Tool(name=t.name, description=t.description,
                        inputSchema=with_ticker(t.args.model_json_schema()))
             for t in RESEARCH_TOOLS.values()]
    tools += [types.Tool(name=name, description=desc, inputSchema=with_ticker({"properties": props}))
              for name, (desc, props) in EXTRA_TOOLS.items()]
    return tools


def run_tool(name: str, arguments: Dict[str, Any]) -> str:
    """Blocking tool execution (called in a worker thread)."""
    arguments = dict(arguments)
    ticker = str(arguments.pop("ticker", "")).upper().strip()
    if not ticker:
        raise ValueError("ticker is required")
    orc = orchestrator()

    if name == "generate_brief":
        result = orc.generate_brief(ticker)
        v = result["verification"]
        return (f"{v['verified']}/{v['total']} claims verified ({v['coverage']:.0%}). "
                f"Saved to {result['report_path']}\n\n{result['report_content']}")
    if name == "get_track_record":
        record = orc.track_record(ticker)
        return json.dumps({"summary": format_track_record(record["summary"]),
                           "ledger": record["entries"], "briefs": record["briefs"]},
                          indent=1, default=str)
    if name not in RESEARCH_TOOLS:
        raise ValueError(f"Unknown tool: {name}")
    return orc.tools(ticker)[name].invoke(json.dumps(arguments)).text


async def on_list_tools(ctx, params) -> types.ListToolsResult:
    return types.ListToolsResult(tools=list_tool_definitions())


async def on_call_tool(ctx, params: types.CallToolRequestParams) -> types.CallToolResult:
    try:
        # Model loaders print to stdout, which would corrupt the protocol stream
        with contextlib.redirect_stdout(sys.stderr):
            text = await anyio.to_thread.run_sync(run_tool, params.name, params.arguments or {})
        return types.CallToolResult(content=[types.TextContent(text=text)])
    except Exception as e:
        # Errors go back to the model as a tool result it can react to
        logger.exception(f"{params.name} failed")
        return types.CallToolResult(content=[types.TextContent(text=f"Error: {e}")], isError=True)


server = Server(
    "finance-analyst", version="1.0.0",
    instructions="Equity research tools over SEC filings (10-K/10-Q/8-K), XBRL financials, "
                 "news and prices. Every result carries [S...] source ids to cite.",
    on_list_tools=on_list_tools,
    on_call_tool=on_call_tool,
)


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    anyio.run(main)
