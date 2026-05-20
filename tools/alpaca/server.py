"""Alpaca MCP server — exposes paper-trading tools gated by Risk_Officer.

Tools (added incrementally per the Phase A punchlist):
  - ping_alpaca         [A1] health check
  - get_paper_account   [A2] equity/cash/buying_power from the broker
  - get_paper_positions [A3] open positions with local-vs-broker reconciliation
  - risk_check_proposed_trade [A4] Risk_Officer.evaluate() — no broker call
  - place_paper_order   [A5] risk-checked order placement
  - close_paper_position [A6] opposing market order to flatten

Entry: python -m tools.alpaca.server server
"""
from typing import Any, Dict, List
import asyncio
import json
import sys
from datetime import date, datetime

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, ServerCapabilities
from mcp.server.models import InitializationOptions


def _json_default(obj):
  if isinstance(obj, (date, datetime)):
    return obj.isoformat()
  raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def _safe_dumps(obj) -> str:
  return json.dumps(obj, default=_json_default)


class AlpacaServer:
  """MCP server that bridges Claude Code to the existing Execution_Agent
  and Risk_Officer. Tool handlers are added incrementally across A2-A6."""

  def __init__(self):
    self.server = Server("nemo_alpaca")
    self._setup_handlers()

  def _setup_handlers(self):
    parent = self

    @self.server.list_tools()
    async def list_tools() -> List[Tool]:
      return [
        Tool(
          name="ping_alpaca",
          description=(
            "Health check for the Alpaca MCP server. Returns "
            "{\"status\": \"pong\"}. Use to verify server is reachable."
          ),
          inputSchema={
            "type": "object",
            "properties": {},
            "required": [],
          },
        ),
      ]

    @self.server.call_tool()
    async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
      if name == "ping_alpaca":
        return [TextContent(type="text", text=_safe_dumps({"status": "pong"}))]
      return [TextContent(
        type="text",
        text=_safe_dumps({"error": f"unknown tool: {name}"}),
      )]

  async def run_server(self):
    try:
      async with stdio_server() as (read_stream, write_stream):
        await self.server.run(
          read_stream,
          write_stream,
          InitializationOptions(
            server_name="nemo_alpaca",
            server_version="0.1.0",
            capabilities=ServerCapabilities(),
          ),
        )
        print("Successfully created alpaca process", file=sys.stderr, flush=True)
    except Exception:
      import traceback
      traceback.print_exc(file=sys.stderr)
      raise


if __name__ == "__main__":
  if len(sys.argv) < 2:
    print("Usage: python -m tools.alpaca.server server", file=sys.stderr)
    sys.exit(1)

  if sys.argv[1] == "server":
    print("Starting alpaca process", file=sys.stderr, flush=True)
    try:
      server = AlpacaServer()
      asyncio.run(server.run_server())
    except Exception as e:
      print(f"SERVER: Exception in main: {e}", file=sys.stderr, flush=True)
      import traceback
      traceback.print_exc(file=sys.stderr)
      sys.exit(1)
  else:
    print(f"Unknown argument: {sys.argv[1]}", file=sys.stderr, flush=True)
    print("Usage: python -m tools.alpaca.server server", file=sys.stderr)
    sys.exit(1)
