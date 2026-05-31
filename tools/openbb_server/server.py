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
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# OpenBB is NOT imported at module level.  The SDK auto-discovers ~50
# extensions at import time which takes ~6s on cold start.  That delay
# caused the MCP server process to be SIGTERMed (exit 143) by Claude Code's
# MCP manager before the `initialize` handshake could complete.
# _get_obb() below imports lazily on first tool call and caches the instance.
_obb = None


def _get_obb():
    global _obb
    if _obb is None:
        from openbb import obb as _loaded  # noqa: PLC0415
        _obb = _loaded
        pat = os.getenv('OPENBB_PAT')
        if pat:
            try:
                _obb.account.login(pat=pat)
            except Exception as exc:
                print(f"[openbb] OPENBB_PAT login failed: {exc}",
                      file=sys.stderr, flush=True)
    return _obb


# ---------------------------------------------------------------------------
# Envelope (matches the shape used by other Nemo MCP tools so the analyst
# skills get a consistent response shape)
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


def _to_records(obbject) -> List[Dict[str, Any]]:
  """Unwrap an OBBject into a list of plain dicts (for JSON-serializable
  responses). Handles list-of-models, single-model, and weirdly-shaped
  results gracefully."""
  if obbject is None:
    return []
  d = None
  if hasattr(obbject, 'model_dump'):
    d = obbject.model_dump()
  elif isinstance(obbject, dict):
    d = obbject
  if not d:
    return []
  results = d.get('results')
  if isinstance(results, list):
    out = []
    for r in results:
      if hasattr(r, 'model_dump'):
        out.append(r.model_dump())
      elif isinstance(r, dict):
        out.append(r)
    return out
  if isinstance(results, dict):
    return [results]
  return []


def _serialize_safe(obj):
  """JSON-safe serializer: date/datetime -> isoformat; everything else
  passes through json.dumps default-handling."""
  if isinstance(obj, (datetime,)):
    return obj.isoformat()
  if hasattr(obj, 'isoformat'):
    return obj.isoformat()
  return str(obj)


# Per-call timeout for OpenBB SDK calls.  The underlying yfinance / provider
# calls typically complete in 7-8s.  45s gives plenty of headroom while still
# preventing an indefinitely hung stdio loop.
_OBB_CALL_TIMEOUT_S = 45.0


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

class OpenBBServer:
  def __init__(self):
    self.server = Server("openbb")
    # PAT login deferred to _get_obb() on first tool call.
    self._setup_handlers()

  # -- handlers -----------------------------------------------------------

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

  # -- tool methods (lazy OpenBB import, bounded by _OBB_CALL_TIMEOUT_S) ---

  async def insider_trading(self, args: Dict[str, Any]) -> List[TextContent]:
    ticker = (args.get('ticker') or '').upper()
    limit = int(args.get('limit') or 50)
    if not ticker:
      return self._error('obb_insider_trading', ticker, 'ticker is required')
    try:
      def _call():
        return _get_obb().equity.ownership.insider_trading(
          symbol=ticker, limit=limit,
        )
      r = await asyncio.wait_for(asyncio.to_thread(_call),
                                 timeout=_OBB_CALL_TIMEOUT_S)
      records = _to_records(r)
      env = build_envelope(records, ticker, 'obb_insider_trading')
      return [TextContent(type='text',
                          text=json.dumps(env, default=_serialize_safe))]
    except asyncio.TimeoutError:
      return self._error('obb_insider_trading', ticker,
                         f'openbb_timeout: call did not return within {_OBB_CALL_TIMEOUT_S}s')
    except Exception as exc:
      return self._error('obb_insider_trading', ticker,
                         f"{type(exc).__name__}: {str(exc)[:200]}")

  async def news_company(self, args: Dict[str, Any]) -> List[TextContent]:
    ticker = (args.get('ticker') or '').upper()
    limit = int(args.get('limit') or 30)
    if not ticker:
      return self._error('obb_news_company', ticker, 'ticker is required')
    try:
      def _call():
        return _get_obb().news.company(symbol=ticker, limit=limit)
      r = await asyncio.wait_for(asyncio.to_thread(_call),
                                 timeout=_OBB_CALL_TIMEOUT_S)
      records = _to_records(r)
      env = build_envelope(records, ticker, 'obb_news_company')
      return [TextContent(type='text',
                          text=json.dumps(env, default=_serialize_safe))]
    except asyncio.TimeoutError:
      return self._error('obb_news_company', ticker,
                         f'openbb_timeout: call did not return within {_OBB_CALL_TIMEOUT_S}s')
    except Exception as exc:
      return self._error('obb_news_company', ticker,
                         f"{type(exc).__name__}: {str(exc)[:200]}")

  async def options_chain(self, args: Dict[str, Any]) -> List[TextContent]:
    ticker = (args.get('ticker') or '').upper()
    if not ticker:
      return self._error('obb_options_chain', ticker, 'ticker is required')
    try:
      def _call():
        return _get_obb().derivatives.options.chains(symbol=ticker)
      r = await asyncio.wait_for(asyncio.to_thread(_call),
                                 timeout=_OBB_CALL_TIMEOUT_S)
      records = _to_records(r)
      # Truncate by default -- options chains are huge (2000+ rows for
      # large-cap names). Keep the first 200 rows; downstream skills can
      # filter further via separate calls if they need the full chain.
      truncated = records[:200]
      env = build_envelope(
        {'rows': truncated, 'total_rows': len(records), 'returned': len(truncated)},
        ticker, 'obb_options_chain',
      )
      return [TextContent(type='text',
                          text=json.dumps(env, default=_serialize_safe))]
    except asyncio.TimeoutError:
      return self._error('obb_options_chain', ticker,
                         f'openbb_timeout: call did not return within {_OBB_CALL_TIMEOUT_S}s')
    except Exception as exc:
      return self._error('obb_options_chain', ticker,
                         f"{type(exc).__name__}: {str(exc)[:200]}")

  async def analyst_consensus(self, args: Dict[str, Any]) -> List[TextContent]:
    ticker = (args.get('ticker') or '').upper()
    if not ticker:
      return self._error('obb_analyst_consensus', ticker, 'ticker is required')
    try:
      def _call():
        return _get_obb().equity.estimates.consensus(symbol=ticker)
      r = await asyncio.wait_for(asyncio.to_thread(_call),
                                 timeout=_OBB_CALL_TIMEOUT_S)
      records = _to_records(r)
      env = build_envelope(records, ticker, 'obb_analyst_consensus')
      return [TextContent(type='text',
                          text=json.dumps(env, default=_serialize_safe))]
    except asyncio.TimeoutError:
      return self._error('obb_analyst_consensus', ticker,
                         f'openbb_timeout: call did not return within {_OBB_CALL_TIMEOUT_S}s')
    except Exception as exc:
      return self._error('obb_analyst_consensus', ticker,
                         f"{type(exc).__name__}: {str(exc)[:200]}")

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
