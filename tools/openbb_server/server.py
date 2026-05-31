"""OpenBB MCP server -- wraps the OpenBB Python SDK as request/response MCP
tools that any Claude Code skill can call.

NOT a daemon; not in the auto-start list. Started on demand via the MCP
stdio transport, like the other tool servers.

Tool catalog (4 tools, all backed by free default providers):

  obb_insider_trading        -- alternative insider Form 4 view (different
                                ingestion path than Finnhub)
  obb_news_company           -- aggregated news across ~10 OpenBB sources
  obb_options_chain          -- full options chain with greeks (yfinance)
  obb_analyst_consensus      -- forward consensus estimates (yfinance)

To register with Claude Code:

  claude mcp add -s user nemo_openbb -- \\
    "<repo>/.venv/Scripts/python.exe" -m tools.openbb_server.server

Additional FMP/Intrinio/Benzinga-gated tools (government_trades, esg_score,
management_compensation, forward_pe, etc.) are deferred to a follow-up PR
that ships once those credentials are set in the environment.
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Each tool call runs in a fresh child process (runner.py) so the OpenBB SDK
# never shares an asyncio event loop with the MCP framework.  This avoids the
# Windows asyncio/thread deadlock that caused indefinite hangs when the SDK
# was invoked via asyncio.to_thread inside the MCP server process.
_RUNNER = os.path.join(os.path.dirname(__file__), 'runner.py')

# The MCP manager may spawn this server with the uv-managed Python rather than
# the project venv, and that Python may not have OpenBB installed.  Always
# invoke runner.py with the venv Python so OpenBB is guaranteed to be present.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_VENV_PYTHON = os.path.join(_REPO_ROOT, '.venv', 'Scripts', 'python.exe')
if not os.path.isfile(_VENV_PYTHON):
    # Non-Windows fallback (bin/python)
    _VENV_PYTHON = os.path.join(_REPO_ROOT, '.venv', 'bin', 'python')
if not os.path.isfile(_VENV_PYTHON):
    _VENV_PYTHON = sys.executable  # last resort

# subprocess.run timeout (slightly shorter than the asyncio.wait_for cap so
# the inner timeout fires first and we get a clean error rather than a
# TimeoutExpired exception propagating up)
_OBB_SUBPROCESS_TIMEOUT_S = 43.0
_OBB_CALL_TIMEOUT_S = 45.0


def _run_subprocess(tool_name: str, kwargs: dict) -> dict:
    """Run an OpenBB tool in runner.py as a fresh child process.

    Returns a dict: {"success": bool, "data": any}  on success
                 or {"success": bool, "error": str}  on failure.
    """
    try:
        proc = subprocess.run(
            [_VENV_PYTHON, _RUNNER, tool_name, json.dumps(kwargs)],
            capture_output=True,
            stdin=subprocess.DEVNULL,   # prevent openbb interactive prompts from blocking
            text=True,
            timeout=_OBB_SUBPROCESS_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return {'success': False,
                'error': f'subprocess timed out after {_OBB_SUBPROCESS_TIMEOUT_S}s'}

    stdout = (proc.stdout or '').strip()
    if not stdout:
        stderr_snippet = (proc.stderr or '').strip()[:300]
        return {'success': False,
                'error': (f'subprocess produced no output '
                          f'(exit {proc.returncode}): {stderr_snippet}')}

    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        return {'success': False,
                'error': f'subprocess output not valid JSON: {stdout[:300]}'}


# ---------------------------------------------------------------------------
# Envelope
# ---------------------------------------------------------------------------

def build_envelope(
  data: Any,
  ticker: str,
  tool: str,
  errors: Optional[List[str]] = None,
) -> Dict[str, Any]:
  return {
    "domain": "alt_data",
    "ticker": ticker,
    "tool": tool,
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "success": not bool(errors),
    "data": data,
    "metadata": {
      "errors": errors or [],
      "provider": "openbb",
    },
  }


def _serialize_safe(obj):
  if isinstance(obj, datetime):
    return obj.isoformat()
  if hasattr(obj, 'isoformat'):
    return obj.isoformat()
  return str(obj)


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

class OpenBBServer:
  def __init__(self):
    self.server = Server("openbb")
    self._setup_handlers()

  def _setup_handlers(self):
    parent = self

    @self.server.list_tools()
    async def list_tools() -> List[Tool]:
      return [
        Tool(
          name="obb_insider_trading",
          description=(
            "Alternative insider Form 4 view via OpenBB. Returns recent "
            "insider transactions for a ticker. Complements get_insider_"
            "transactions (Finnhub-backed) -- different ingestion path so "
            "cross-checking the two surfaces discrepancies or filings one "
            "source missed. Returns a list of rows with transaction date, "
            "insider name, change in shares, and transaction code."
          ),
          inputSchema={
            "type": "object",
            "required": ["ticker"],
            "properties": {
              "ticker": {"type": "string",
                         "description": "Stock ticker symbol (e.g. AAPL)"},
              "limit":  {"type": "integer",
                         "description": "Max rows to return (default 50)",
                         "default": 50},
            },
          },
        ),
        Tool(
          name="obb_news_company",
          description=(
            "Aggregated company news across OpenBB's default provider set "
            "(~10 sources). Use when you want broader coverage than the "
            "single-source Finnhub get_company_news call. Returns title, "
            "date, source, URL per article."
          ),
          inputSchema={
            "type": "object",
            "required": ["ticker"],
            "properties": {
              "ticker": {"type": "string"},
              "limit":  {"type": "integer", "default": 30,
                         "description": "Max articles to return"},
            },
          },
        ),
        Tool(
          name="obb_options_chain",
          description=(
            "Full options chain (yfinance-backed) including greeks where "
            "available. Returns all strikes across all listed expiries "
            "for the ticker. Use for skew / put-call / implied-vol surface "
            "analysis in /equity-deep-research Step 11 positioning."
          ),
          inputSchema={
            "type": "object",
            "required": ["ticker"],
            "properties": {
              "ticker": {"type": "string"},
            },
          },
        ),
        Tool(
          name="obb_analyst_consensus",
          description=(
            "Forward consensus estimates (yfinance-backed). Returns "
            "consensus EPS, revenue, and other forward metrics for the "
            "ticker. Use to cross-check the Finnhub-backed "
            "get_forward_estimates call."
          ),
          inputSchema={
            "type": "object",
            "required": ["ticker"],
            "properties": {
              "ticker": {"type": "string"},
            },
          },
        ),
      ]

    @self.server.call_tool()
    async def call_tool(name: str, args: Dict[str, Any]):
      if name == "obb_insider_trading":
        return await parent.insider_trading(args)
      if name == "obb_news_company":
        return await parent.news_company(args)
      if name == "obb_options_chain":
        return await parent.options_chain(args)
      if name == "obb_analyst_consensus":
        return await parent.analyst_consensus(args)
      raise ValueError(f"unknown tool: {name}")

  # -- tool methods ----------------------------------------------------------
  # Each spawns runner.py as a child process via asyncio.to_thread so the
  # blocking subprocess.run call doesn't stall the MCP server's event loop.

  async def insider_trading(self, args: Dict[str, Any]) -> List[TextContent]:
    ticker = (args.get('ticker') or '').upper()
    limit = int(args.get('limit') or 50)
    if not ticker:
      return self._error('obb_insider_trading', ticker, 'ticker is required')
    return await self._dispatch(
      'obb_insider_trading', ticker,
      {'ticker': ticker, 'limit': limit},
    )

  async def news_company(self, args: Dict[str, Any]) -> List[TextContent]:
    ticker = (args.get('ticker') or '').upper()
    limit = int(args.get('limit') or 30)
    if not ticker:
      return self._error('obb_news_company', ticker, 'ticker is required')
    return await self._dispatch(
      'obb_news_company', ticker,
      {'ticker': ticker, 'limit': limit},
    )

  async def options_chain(self, args: Dict[str, Any]) -> List[TextContent]:
    ticker = (args.get('ticker') or '').upper()
    if not ticker:
      return self._error('obb_options_chain', ticker, 'ticker is required')
    return await self._dispatch('obb_options_chain', ticker, {'ticker': ticker})

  async def analyst_consensus(self, args: Dict[str, Any]) -> List[TextContent]:
    ticker = (args.get('ticker') or '').upper()
    if not ticker:
      return self._error('obb_analyst_consensus', ticker, 'ticker is required')
    return await self._dispatch('obb_analyst_consensus', ticker, {'ticker': ticker})

  async def _dispatch(
    self, tool: str, ticker: str, kwargs: dict,
  ) -> List[TextContent]:
    """Spawn runner.py in a thread and return the MCP TextContent response."""
    try:
      result = await asyncio.wait_for(
        asyncio.to_thread(_run_subprocess, tool, kwargs),
        timeout=_OBB_CALL_TIMEOUT_S,
      )
    except asyncio.TimeoutError:
      return self._error(tool, ticker,
                         f'openbb_timeout: subprocess did not return within '
                         f'{_OBB_CALL_TIMEOUT_S}s')
    except Exception as exc:
      return self._error(tool, ticker,
                         f"{type(exc).__name__}: {str(exc)[:200]}")

    if not result.get('success'):
      return self._error(tool, ticker, result.get('error', 'unknown error'))

    env = build_envelope(result['data'], ticker, tool)
    return [TextContent(type='text',
                        text=json.dumps(env, default=_serialize_safe))]

  def _error(self, tool: str, ticker: str, msg: str) -> List[TextContent]:
    env = build_envelope([], ticker, tool, errors=[msg])
    return [TextContent(type='text',
                        text=json.dumps(env, default=_serialize_safe))]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main():
  srv = OpenBBServer()
  async with stdio_server() as (read, write):
    await srv.server.run(read, write,
                         srv.server.create_initialization_options())


if __name__ == "__main__":
  asyncio.run(main())
