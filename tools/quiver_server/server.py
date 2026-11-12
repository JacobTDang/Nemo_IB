"""nemo_quiver MCP server — QuiverQuant web data tools.

Reverse-engineered from quiverquant.com's public web application (a
server-rendered Django site): datasets arrive as embedded JavaScript
literals, Plotly chart payloads, and HTML tables, plus a family of
GET JSON endpoints. No API key is required; the data is what the site
serves anonymous visitors. An optional logged-in session cookie
(QUIVER_SESSION_COOKIE) unlocks session-gated sections such as
bulls/bears analyst counts.

9 tools:
  get_quiver_congress_trades      -- ~300 most recent congressional trades,
                                     filterable by ticker/politician/party/
                                     chamber
  get_quiver_politician_trades    -- one politician's full disclosed trade
                                     history
  get_quiver_wsb_sentiment        -- WallStreetBets mention counts, sentiment
                                     and mention time series per ticker
  get_quiver_lobbying             -- recent federal lobbying filings + top
                                     spenders
  get_quiver_gov_contracts        -- recent federal contract awards
  get_quiver_ticker_snapshot      -- per-ticker JSON endpoints (price,
                                     insiders, institutions, forecast, ETF
                                     holders)
  get_quiver_insider_trades       -- ~1,800 corporate insider purchases and
                                     sales from the union of the dashboard's
                                     two embedded lists
  get_quiver_congress_net_worth   -- disclosed net worth per sitting member,
                                     ranked
  get_quiver_bills                -- ~33,000-bill legislation title index,
                                     searchable by title

Failure vocabulary, shared with tools/altdata_server/server.py:

  success: false  -- the lookup could not be performed at all (site
                     unreachable, page layout changed, unknown entity)
  coverage        -- "full" | "partial" | "not_covered"
  reason          -- machine-readable cause (provider_unavailable,
                     rate_limited, parse_failed, unknown_politician, ...)
  degraded        -- named degradations that narrowed an answer without
                     failing it

Honesty notes baked into the tools:
  * The WSB dashboard's mention time series currently ENDS 2025-02-21 on
    Quiver's side; every response carries `data_horizon` so a stale
    series is never mistaken for current sentiment.
  * Congress pages embed the most recent ~300 trades (all recent) and a
    politician page embeds that member's full history — both are exactly
    what the public pages show, no more.
  * These are public disclosure datasets: a trade filed today happened
    up to 45 days ago (disclosure_lag warning).

Register:
  claude mcp add -s user nemo_quiver -e PYTHONPATH=<repo> -- \\
    "<repo>/.venv/bin/python" -m tools.quiver_server.server

Optional env vars:
  QUIVER_SESSION_COOKIE -- Django sessionid cookie of a logged-in
                          quiverquant.com browser session (raises rate
                          ceilings, unlocks bulls/bears)
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
from mcp.types import TextContent, Tool

from tools.response_meta import annotating, warning

from tools.quiver_server.client import (
    QuiverError,
    QuiverWebClient,
)

_CLIENT: Optional[QuiverWebClient] = None


def _client() -> QuiverWebClient:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = QuiverWebClient(
            session_cookie=os.environ.get("QUIVER_SESSION_COOKIE") or None)
    return _CLIENT


# ---------------------------------------------------------------------------
# Envelope (mirrors altdata's contract)
# ---------------------------------------------------------------------------

def _envelope(data: Any, tool: str, ticker: str = "",
              errors: Optional[List[str]] = None,
              warnings: Optional[List[Dict[str, Any]]] = None
              ) -> Dict[str, Any]:
    payload = {
        "domain": "quiver",
        "ticker": ticker,
        "tool": tool,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "success": not bool(errors),
        "data": data,
        "metadata": {"errors": errors or []},
    }
    if warnings:
        payload["warnings"] = warnings
    if errors:
        payload["error"] = errors[0]
    return payload


def _ok(tool: str, data: Any, ticker: str = "",
        warnings: Optional[List[Dict[str, Any]]] = None
        ) -> List[TextContent]:
    return [TextContent(
        type="text",
        text=json.dumps(_envelope(data, tool, ticker, warnings=warnings),
                        default=str))]


def _fail(tool: str, data: Any, msg: str, ticker: str = "",
          warnings: Optional[List[Dict[str, Any]]] = None
          ) -> List[TextContent]:
    return [TextContent(
        type="text",
        text=json.dumps(_envelope(data, tool, ticker, errors=[msg],
                                  warnings=warnings), default=str))]


def _run(tool: str, fn, *, ticker: str = "",
         warnings: Optional[List[Dict[str, Any]]] = None
         ) -> List[TextContent]:
    """Run a client call; map QuiverError to the failure envelope."""
    try:
        result = fn()
    except QuiverError as exc:
        payload = {"success": False, "reason": exc.reason,
                   "error": exc.message}
        return _fail(tool, payload, exc.message, ticker, warnings=warnings)
    except Exception as exc:  # network/parse surprises
        payload = {"success": False, "reason": "unexpected",
                   "error": str(exc)}
        return _fail(tool, payload, str(exc), ticker, warnings=warnings)
    if isinstance(result, dict):
        result.setdefault("success", True)
    return _ok(tool, result, ticker, warnings=warnings)


# ---------------------------------------------------------------------------
# Shared warnings
# ---------------------------------------------------------------------------

_DISCLOSURE_LAG = warning(
    "disclosure_lag",
    "Members file up to 45 days after trading, so transaction_date and "
    "filed_date differ. Sort and filter on filed_date for freshness.")

_WSB_HORIZON = warning(
    "data_horizon",
    "Quiver's WSB mention time series is dated, not live: every response "
    "carries data_horizon, the last date Quiver's Reddit scrape actually "
    "covers. Compare it to today before calling the history 'recent'.")

_QUIVER_WEB = warning(
    "scraped_source",
    "Data is parsed from QuiverQuant's public web pages, not a contracted "
    "API. A page layout change degrades the parser; failures carry "
    "reason=parse_failed. Page fetches are cached 15 minutes.")

_INSIDER_LAG = warning(
    "insider_filing_lag",
    "A Form 4 is due within two business days of the transaction, so "
    "filed_date trails transaction_date by a few days. The list is also a "
    "window, not full history: about 1,800 distinct transactions across the "
    "latest month.")

_NET_WORTH_ESTIMATE = warning(
    "net_worth_is_an_estimate",
    "Net worth is Quiver's estimate from disclosed ranges, not a filed "
    "total. Disclosures are banded and often partial, so treat the figure "
    "as an order of magnitude, not a balance sheet.")


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

class QuiverServer:
    def __init__(self):
        self.server = Server("quiver")
        self._setup_handlers()

    def _setup_handlers(self):
        parent = self

        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            return [
                Tool(
                    name="get_quiver_congress_trades",
                    description=(
                        "The ~300 most recent US congressional stock trades "
                        "from QuiverQuant's congress-trading dashboard "
                        "(House PTRs + Senate eFD filings, parsed from the "
                        "page's embedded data). Filter by ticker, politician "
                        "name, party, chamber or transaction type; days "
                        "filters on filed_date. Returns ticker, security, "
                        "asset type, transaction type, disclosed amount "
                        "range, politician, chamber, party, filed/traded "
                        "dates. Complements Nemo's primary-source congress "
                        "store with a fast, pre-normalized cross-check."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "ticker": {
                                "type": "string",
                                "description": "Filter to one ticker (e.g. NVDA).",
                            },
                            "politician": {
                                "type": "string",
                                "description": "Substring match on politician name or bioguide ID.",
                            },
                            "tx_type": {
                                "type": "string",
                                "description": "Substring match, e.g. 'Purchase' or 'Sale'.",
                            },
                            "chamber": {
                                "type": "string",
                                "enum": ["House", "Senate"],
                            },
                            "party": {
                                "type": "string",
                                "enum": ["D", "R", "I"],
                            },
                            "days": {
                                "type": "integer",
                                "description": "Keep trades filed within the last N days.",
                            },
                            "limit": {
                                "type": "integer",
                                "default": 50,
                                "description": "Max trades to return (data is newest-first).",
                            },
                        },
                    },
                ),
                Tool(
                    name="get_quiver_politician_trades",
                    description=(
                        "One politician's full disclosed trade history from "
                        "QuiverQuant (per-politician page, parsed from "
                        "embedded data). Accepts a name ('Nancy Pelosi'), a "
                        "bioguide ID ('P000197') or the page slug. Returns "
                        "ticker, transaction type, filed/traded dates, "
                        "description, disclosed amount range and midpoint, "
                        "asset type, sector. Use get_quiver_congress_trades first "
                        "to find who traded a ticker, then this for depth."
                    ),
                    inputSchema={
                        "type": "object",
                        "required": ["politician"],
                        "properties": {
                            "politician": {
                                "type": "string",
                                "description": "Name, bioguide ID, or 'Name-ID' slug.",
                            },
                            "limit": {
                                "type": "integer",
                                "default": 50,
                                "description": "Max trades (data is newest-first).",
                            },
                        },
                    },
                ),
                Tool(
                    name="get_quiver_wsb_sentiment",
                    description=(
                        "WallStreetBets daily-discussion data from "
                        "QuiverQuant: mention counts per ticker (~600 "
                        "tickers), a per-ticker sentiment score and mention "
                        "change (top charted tickers only), and — with "
                        "ticker + include_history — a daily mention-count "
                        "time series for that ticker. IMPORTANT: the series "
                        "carries data_horizon, the last date Quiver's "
                        "Reddit scrape actually covers; treat it as dated "
                        "cross-sectional sentiment, not live."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "ticker": {
                                "type": "string",
                                "description": "Focus one ticker (enables sentiment + history).",
                            },
                            "include_history": {
                                "type": "boolean",
                                "default": False,
                                "description": "Include the daily mention time series for the ticker.",
                            },
                            "limit": {
                                "type": "integer",
                                "default": 50,
                                "description": "Max mention rows when no ticker is given.",
                            },
                        },
                    },
                ),
                Tool(
                    name="get_quiver_lobbying",
                    description=(
                        "Federal corporate lobbying data from QuiverQuant: "
                        "the most recent ~100 lobbying filings (ticker, "
                        "disclosed amount, filing date) plus the top-spender "
                        "leaderboard (~200 companies). Filter by ticker. "
                        "Signal: sustained or rising lobbying spend by an "
                        "acquirer/regulatee often precedes policy or M&A "
                        "action; new filings by a company not recently "
                        "active in Washington is the interesting delta."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "ticker": {
                                "type": "string",
                                "description": "Filter filings and spenders to one ticker.",
                            },
                            "limit": {
                                "type": "integer",
                                "default": 50,
                                "description": "Max filings (newest-first).",
                            },
                        },
                    },
                ),
                Tool(
                    name="get_quiver_gov_contracts",
                    description=(
                        "Recent federal contract awards from QuiverQuant's "
                        "government-contracts dashboard (recipient ticker, "
                        "date awarded, amount, funding agency, description). "
                        "Distinct from the USASpending-based "
                        "get_government_contracts tool: that one measures "
                        "period obligation FLOW per company; this one lists "
                        "individual recent awards with agency and "
                        "description, useful for award-level color."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "ticker": {
                                "type": "string",
                                "description": "Filter to one recipient ticker.",
                            },
                            "days": {
                                "type": "integer",
                                "description": "Keep awards within the last N days.",
                            },
                            "limit": {
                                "type": "integer",
                                "default": 50,
                                "description": "Max awards (newest-first).",
                            },
                        },
                    },
                ),
                Tool(
                    name="get_quiver_ticker_snapshot",
                    description=(
                        "Per-ticker data from QuiverQuant's JSON endpoints: "
                        "current price, insider trades (most recent N of "
                        "1,600+ for busy tickers), institutional holdings "
                        "heatmap (fund, holdings, quarter-over-quarter "
                        "change), analyst forecast/price targets, top ETF "
                        "holders. Sections degrade independently — a failed "
                        "endpoint is named in degraded_sections, never "
                        "dropping the snapshot. bulls_bears requires a "
                        "session cookie (QUIVER_SESSION_COOKIE)."
                    ),
                    inputSchema={
                        "type": "object",
                        "required": ["ticker"],
                        "properties": {
                            "ticker": {"type": "string"},
                            "trade_limit": {
                                "type": "integer",
                                "default": 25,
                                "description": "Max insider trades to return.",
                            },
                        },
                    },
                ),
                Tool(
                    name="get_quiver_insider_trades",
                    description=(
                        "Corporate insider purchases and sales from "
                        "QuiverQuant's insider dashboard. The page embeds two "
                        "1,000-row lists — latest filings and the month's "
                        "largest — whose union is ~1,800 distinct "
                        "transactions, each with owner, title, ticker, "
                        "transaction type, shares, price per share, estimated "
                        "value, transaction and filing dates, and an "
                        "estimate of the owner's net worth. Filter by ticker, "
                        "owner, purchase/sale, minimum value or recency. "
                        "Distinct from get_insider_transactions (SEC "
                        "primary-source Form 4 parsing): this is Quiver's "
                        "pre-normalized view with values computed for you."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "ticker": {
                                "type": "string",
                                "description": "Exact ticker (e.g. NVDA).",
                            },
                            "owner": {
                                "type": "string",
                                "description": "Substring of the insider's name.",
                            },
                            "tx_type": {
                                "type": "string",
                                "description": "Purchase or Sale (substring match).",
                            },
                            "min_value": {
                                "type": "number",
                                "description": "Keep trades worth at least this many USD.",
                            },
                            "days": {
                                "type": "integer",
                                "description": "Keep trades in the last N days.",
                            },
                            "limit": {
                                "type": "integer",
                                "default": 50,
                                "description": "Max trades (newest-first).",
                            },
                        },
                    },
                ),
                Tool(
                    name="get_quiver_congress_net_worth",
                    description=(
                        "Disclosed net worth per sitting US member, ranked, "
                        "from the array Quiver embeds on its congress-trading "
                        "dashboard (494 members). Returns rank, name, net "
                        "worth in USD, bioguide id, chamber, party and state. "
                        "Filter by name, chamber, party, state or a minimum "
                        "threshold. This is a disclosed-estimate ranking, not "
                        "a filing-level holdings ledger — for positions and "
                        "transactions use get_quiver_politician_trades."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "name": {"type": "string",
                                     "description": "Substring of the member's name."},
                            "chamber": {"type": "string",
                                        "description": "House or Senate."},
                            "party": {"type": "string",
                                      "description": "Democratic, Republican or Independent."},
                            "state": {"type": "string",
                                      "description": "State name (e.g. California)."},
                            "min_net_worth": {
                                "type": "number",
                                "description": "Keep members at or above this net worth in USD.",
                            },
                            "limit": {
                                "type": "integer",
                                "default": 50,
                                "description": "Max members (richest first).",
                            },
                        },
                    },
                ),
                Tool(
                    name="get_quiver_bills",
                    description=(
                        "Search Quiver's legislation index: ~33,000 bills "
                        "from the 118th and 119th Congress as {congress, "
                        "number, type, title}. `query` matches the title "
                        "case-insensitively, which turns a theme into the "
                        "bill numbers to follow up on congress.gov. This is "
                        "a title index, not bill text or status — pair it "
                        "with the SEC/government tools for the economic side."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Substring of the bill title (e.g. semiconductor).",
                            },
                            "bill_type": {
                                "type": "string",
                                "description": "hr or s.",
                            },
                            "congress": {
                                "type": "string",
                                "description": "118 or 119.",
                            },
                            "limit": {
                                "type": "integer",
                                "default": 50,
                                "description": "Max bills to return.",
                            },
                            "offset": {
                                "type": "integer",
                                "default": 0,
                                "description": "Skip this many matches (for paging).",
                            },
                        },
                    },
                ),
            ]

        @self.server.call_tool()
        @annotating(
            "quiver",
            per_tool={
                "get_quiver_congress_trades": "QuiverQuant web (congress dashboard)",
                "get_quiver_politician_trades": "QuiverQuant web (politician page)",
                "get_quiver_wsb_sentiment": "QuiverQuant web (WSB dashboard)",
                "get_quiver_lobbying": "QuiverQuant web (lobbying dashboard)",
                "get_quiver_gov_contracts": "QuiverQuant web (govcontracts dashboard)",
                "get_quiver_ticker_snapshot": "QuiverQuant web (ticker JSON endpoints)",
                "get_quiver_insider_trades": "QuiverQuant web (insider dashboard)",
                "get_quiver_congress_net_worth": "QuiverQuant web (congress dashboard net-worth array)",
                "get_quiver_bills": "QuiverQuant web (legislation index)",
            },
            warnings_per_tool={
                "get_quiver_congress_trades": [_DISCLOSURE_LAG, _QUIVER_WEB],
                "get_quiver_politician_trades": [_DISCLOSURE_LAG, _QUIVER_WEB],
                "get_quiver_wsb_sentiment": [_WSB_HORIZON, _QUIVER_WEB],
                "get_quiver_lobbying": [_QUIVER_WEB],
                "get_quiver_gov_contracts": [_QUIVER_WEB],
                "get_quiver_ticker_snapshot": [_QUIVER_WEB],
                "get_quiver_insider_trades": [_INSIDER_LAG, _QUIVER_WEB],
                "get_quiver_congress_net_worth": [_DISCLOSURE_LAG,
                                                  _NET_WORTH_ESTIMATE,
                                                  _QUIVER_WEB],
                "get_quiver_bills": [_QUIVER_WEB],
            },
        )
        async def call_tool(name: str, args: Dict[str, Any]):
            c = _client()
            ticker = (args.get("ticker") or "").upper()
            if name == "get_quiver_congress_trades":
                return _run("get_quiver_congress_trades", lambda: c.get_congress_trades(
                    ticker=args.get("ticker"),
                    politician=args.get("politician"),
                    tx_type=args.get("tx_type"),
                    chamber=args.get("chamber"),
                    party=args.get("party"),
                    days=args.get("days"),
                    limit=args.get("limit", 50),
                ), ticker=ticker)
            if name == "get_quiver_politician_trades":
                return _run("get_quiver_politician_trades", lambda: c.get_politician_trades(
                    args["politician"], limit=args.get("limit", 50),
                ))
            if name == "get_quiver_wsb_sentiment":
                return _run("get_quiver_wsb_sentiment", lambda: c.get_wsb_sentiment(
                    ticker=args.get("ticker"),
                    include_history=bool(args.get("include_history")),
                    limit=args.get("limit", 50),
                ), ticker=ticker)
            if name == "get_quiver_lobbying":
                return _run("get_quiver_lobbying", lambda: c.get_lobbying_filings(
                    ticker=args.get("ticker"), limit=args.get("limit", 50),
                ), ticker=ticker)
            if name == "get_quiver_gov_contracts":
                return _run("get_quiver_gov_contracts",
                            lambda: c.get_gov_contract_awards(
                    ticker=args.get("ticker"), days=args.get("days"),
                    limit=args.get("limit", 50),
                ), ticker=ticker)
            if name == "get_quiver_ticker_snapshot":
                return _run("get_quiver_ticker_snapshot", lambda: c.get_ticker_snapshot(
                    args["ticker"], trade_limit=args.get("trade_limit", 25),
                ), ticker=ticker)
            if name == "get_quiver_insider_trades":
                return _run("get_quiver_insider_trades",
                            lambda: c.get_insider_trades(
                    ticker=args.get("ticker"),
                    owner=args.get("owner"),
                    tx_type=args.get("tx_type"),
                    min_value=args.get("min_value"),
                    days=args.get("days"),
                    limit=args.get("limit", 50),
                ), ticker=ticker)
            if name == "get_quiver_congress_net_worth":
                return _run("get_quiver_congress_net_worth",
                            lambda: c.get_congress_net_worth(
                    name=args.get("name"),
                    chamber=args.get("chamber"),
                    party=args.get("party"),
                    state=args.get("state"),
                    min_net_worth=args.get("min_net_worth"),
                    limit=args.get("limit", 50),
                ))
            if name == "get_quiver_bills":
                return _run("get_quiver_bills", lambda: c.get_bills(
                    query=args.get("query"),
                    bill_type=args.get("bill_type"),
                    congress=args.get("congress"),
                    limit=args.get("limit", 50),
                    offset=args.get("offset", 0),
                ))
            return _fail(name, {"success": False,
                                "reason": "unknown_tool"},
                         f"unknown tool: {name}")

    async def run_server(self):
        async with stdio_server() as (read, write):
            await self.server.run(read, write,
                                  self.server.create_initialization_options())


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "http":
        # Streamable HTTP, for a host a client connects to rather than one
        # that spawns it. stdio stays the default for local use.
        from tools.mcp_http import run_http
        print("[quiver] starting over streamable HTTP", file=sys.stderr,
              flush=True)
        # QUIVER_SESSION_COOKIE is optional; its absence only degrades the
        # bulls/bears section, so it must not mark a working container
        # unhealthy.
        run_http(QuiverServer().server, required_env=())
    else:
        print("[quiver] starting", file=sys.stderr, flush=True)
        server = QuiverServer()
        try:
            asyncio.run(server.run_server())
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] not in ("server", "http"):
        print("Usage: python -m tools.quiver_server.server [server|http]",
              file=sys.stderr)
        sys.exit(1)
    main()
