"""nemo_altdata MCP server — alternative data tools for pre-earnings research.

5 tools:
  get_google_trends          -- pytrends wrapper; YoY demand signal
  get_finbert_sentiment      -- FinBERT financial sentiment on news headlines
  get_taiwan_monthly_revenue -- MOPS scraper for TSMC/Foxconn/MediaTek/ASE
  get_job_postings_count     -- Lever / Greenhouse public job listing JSON
  get_options_implied_move   -- ATM straddle implied move + put/call skew

Heavy tools (pytrends, FinBERT) run in isolated subprocesses via the same
pattern as tools/openbb_server/server.py to avoid asyncio conflicts on Windows.
Light tools (MOPS, Greenhouse, options math) run directly in async handlers.

Register:
  claude mcp add -s user nemo_altdata -e PYTHONPATH=<repo> -- \\
    "<repo>/.venv/Scripts/python.exe" -m tools.altdata_server.server server
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

# ---------------------------------------------------------------------------
# Subprocess runner paths + Python executable
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
_VENV_PYTHON = os.path.join(_REPO_ROOT, ".venv", "Scripts", "python.exe")
if not os.path.isfile(_VENV_PYTHON):
    _VENV_PYTHON = os.path.join(_REPO_ROOT, ".venv", "bin", "python")
if not os.path.isfile(_VENV_PYTHON):
    _VENV_PYTHON = sys.executable

_TRENDS_RUNNER = os.path.join(_HERE, "trends_runner.py")
_FINBERT_RUNNER = os.path.join(_HERE, "finbert_runner.py")

_TRENDS_TIMEOUT_S = 30.0
_FINBERT_TIMEOUT_S = 120.0   # first call downloads 440MB model
_SUBPROCESS_TIMEOUT_S = 25.0  # inner subprocess.run timeout for trends
_FINBERT_SUB_TIMEOUT_S = 115.0


# ---------------------------------------------------------------------------
# Subprocess dispatch helper (mirrors openbb_server pattern)
# ---------------------------------------------------------------------------

def _run_subprocess(runner_path: str, tool_name: str, kwargs: dict,
                    sub_timeout: float) -> dict:
    """Spawn a runner script in a fresh child process; return parsed JSON."""
    try:
        proc = subprocess.run(
            [_VENV_PYTHON, runner_path, tool_name, json.dumps(kwargs)],
            capture_output=True,
            stdin=subprocess.DEVNULL,
            text=True,
            timeout=sub_timeout,
        )
    except subprocess.TimeoutExpired:
        return {"success": False,
                "error": f"subprocess timed out after {sub_timeout}s"}

    stdout = (proc.stdout or "").strip()
    if not stdout:
        stderr_snippet = (proc.stderr or "").strip()[-300:]
        return {"success": False,
                "error": f"no output (exit {proc.returncode}): {stderr_snippet}"}
    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        return {"success": False,
                "error": f"output not valid JSON: {stdout[:300]}"}


# ---------------------------------------------------------------------------
# Envelope
# ---------------------------------------------------------------------------

def _envelope(data: Any, tool: str, ticker: str = "",
               errors: Optional[List[str]] = None) -> Dict[str, Any]:
    return {
        "domain": "alt_data",
        "ticker": ticker,
        "tool": tool,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "success": not bool(errors),
        "data": data,
        "metadata": {"errors": errors or []},
    }


def _ok(tool: str, data: Any, ticker: str = "") -> List[TextContent]:
    return [TextContent(type="text",
                        text=json.dumps(_envelope(data, tool, ticker), default=str))]


def _err(tool: str, msg: str, ticker: str = "") -> List[TextContent]:
    return [TextContent(type="text",
                        text=json.dumps(_envelope(None, tool, ticker, errors=[msg]),
                                        default=str))]


# ---------------------------------------------------------------------------
# Options implied move (pure math — no subprocess needed)
# ---------------------------------------------------------------------------

def compute_implied_move(spot: float, atm_call_ask: float,
                         atm_put_ask: float) -> Dict[str, Any]:
    """ATM straddle cost / spot = implied move %."""
    straddle = atm_call_ask + atm_put_ask
    implied_move_pct = straddle / spot if spot > 0 else 0.0
    return {"implied_move_pct": round(implied_move_pct, 4), "straddle_cost": straddle}


def _find_atm_options(rows: List[Dict], spot: float,
                      target_expiry: Optional[str] = None):
    """Find the nearest ATM call and put from options chain rows."""
    if not rows:
        return None, None, None

    # Group expiries
    expiries = sorted({r.get("expiration") or r.get("expiration_date", "") for r in rows
                       if r.get("expiration") or r.get("expiration_date")})
    if not expiries:
        return None, None, None

    # Pick the nearest expiry after today (front month)
    today = datetime.now(timezone.utc).date().isoformat()
    future = [e for e in expiries if e > today]
    chosen_expiry = future[0] if future else expiries[-1]
    if target_expiry:
        chosen_expiry = target_expiry

    chain = [r for r in rows
             if (r.get("expiration") or r.get("expiration_date", "")) == chosen_expiry]
    calls = [r for r in chain if (r.get("option_type") or r.get("optionType", "")).lower() == "call"]
    puts  = [r for r in chain if (r.get("option_type") or r.get("optionType", "")).lower() == "put"]

    def nearest_atm(options):
        if not options:
            return None
        return min(options, key=lambda r: abs(float(r.get("strike", 0)) - spot))

    atm_call = nearest_atm(calls)
    atm_put  = nearest_atm(puts)
    return atm_call, atm_put, chosen_expiry


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

class AltDataServer:
    def __init__(self):
        self.server = Server("altdata")
        self._setup_handlers()

    def _setup_handlers(self):
        parent = self

        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            return [
                Tool(
                    name="get_google_trends",
                    description=(
                        "Google Trends interest-over-time for up to 5 search keywords. "
                        "Returns weekly index values (0-100) and a YoY ratio comparing "
                        "the last 13 weeks vs the same period 1 year prior. "
                        "Use to gauge consumer demand before a company's earnings: "
                        "ratio > 1.10 = bullish demand signal, < 0.90 = bearish. "
                        "Best for consumer-facing companies (tech, automotive, retail, streaming)."
                    ),
                    inputSchema={
                        "type": "object",
                        "required": ["keywords"],
                        "properties": {
                            "keywords": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Search terms, max 5 (e.g. ['iPhone', 'buy iPhone'])",
                            },
                            "timeframe": {
                                "type": "string",
                                "default": "today 12-m",
                                "description": "pytrends timeframe string (e.g. 'today 12-m', 'today 3-m'). Use 12-m for YoY ratio.",
                            },
                            "geo": {
                                "type": "string",
                                "default": "US",
                                "description": "Country code ('US', 'CN', 'GB') or '' for worldwide.",
                            },
                        },
                    },
                ),
                Tool(
                    name="get_finbert_sentiment",
                    description=(
                        "Score a list of news headlines or article summaries using "
                        "FinBERT (ProsusAI/finbert), a financial domain BERT model. "
                        "Returns per-article positive/negative/neutral labels and an "
                        "aggregated net_score (-1.0 = all negative, +1.0 = all positive). "
                        "Signal: net_score > 0.15 = bullish, < -0.15 = bearish. "
                        "Pipe in headlines from get_company_news or obb_news_company. "
                        "First call downloads ~440MB model; subsequent calls use cache."
                    ),
                    inputSchema={
                        "type": "object",
                        "required": ["texts", "ticker"],
                        "properties": {
                            "texts": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "News headlines or article summaries (max 50).",
                            },
                            "ticker": {"type": "string"},
                        },
                    },
                ),
                Tool(
                    name="get_taiwan_monthly_revenue",
                    description=(
                        "Fetch monthly revenue for Taiwan-listed companies from MOPS "
                        "(Market Observation Post System). Published by the 10th of "
                        "the following month — typically 3-4 weeks before downstream "
                        "US companies report. Key company codes: TSMC=2330, "
                        "Foxconn=2317, MediaTek=2454, ASE Group=3711. "
                        "Returns NTD millions per month + YoY% for each company."
                    ),
                    inputSchema={
                        "type": "object",
                        "required": ["company_codes"],
                        "properties": {
                            "company_codes": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Taiwan stock codes (e.g. ['2330', '2317']). Max 5.",
                            },
                            "months": {
                                "type": "integer",
                                "default": 6,
                                "description": "Number of recent months to return.",
                            },
                        },
                    },
                ),
                Tool(
                    name="get_job_postings_count",
                    description=(
                        "Count open job postings for a company via Greenhouse or Lever "
                        "ATS public APIs (no auth required). Returns total count and "
                        "breakdown by department. A surge in engineering / sales postings "
                        "signals investment ahead of revenue growth; a freeze signals "
                        "cost discipline or guidance cut risk. "
                        "Find the company slug at boards.greenhouse.io/<slug> or "
                        "jobs.lever.co/<slug>."
                    ),
                    inputSchema={
                        "type": "object",
                        "required": ["company_slug"],
                        "properties": {
                            "company_slug": {
                                "type": "string",
                                "description": "Company identifier in the ATS URL (e.g. 'nvidia', 'apple', 'openai').",
                            },
                            "ats": {
                                "type": "string",
                                "enum": ["greenhouse", "lever"],
                                "default": "greenhouse",
                                "description": "Which ATS to query first. Falls back to the other if not found.",
                            },
                            "department_filter": {
                                "type": "string",
                                "description": "Optional: filter to postings containing this string in department name.",
                            },
                        },
                    },
                ),
                Tool(
                    name="get_options_implied_move",
                    description=(
                        "Compute the options-implied earnings move from an ATM straddle "
                        "price. Pass the raw options chain from obb_options_chain plus "
                        "the current spot price. Returns: implied_move_pct (market's "
                        "own expected move magnitude), put_call_skew (widening puts = "
                        "institutional hedging = bearish lean), and the front expiry used. "
                        "Rule: implied_move_pct > 0.15 = high binary risk, do not size large."
                    ),
                    inputSchema={
                        "type": "object",
                        "required": ["ticker", "spot_price", "options_chain_rows"],
                        "properties": {
                            "ticker": {"type": "string"},
                            "spot_price": {
                                "type": "number",
                                "description": "Current stock price.",
                            },
                            "options_chain_rows": {
                                "type": "array",
                                "description": "The 'rows' array from obb_options_chain data field.",
                            },
                            "target_expiry": {
                                "type": "string",
                                "description": "Optional YYYY-MM-DD: use this expiry instead of front-month.",
                            },
                        },
                    },
                ),
            ]

        @self.server.call_tool()
        async def call_tool(name: str, args: Dict[str, Any]):
            if name == "get_google_trends":
                return await parent.google_trends(args)
            if name == "get_finbert_sentiment":
                return await parent.finbert_sentiment(args)
            if name == "get_taiwan_monthly_revenue":
                return await parent.taiwan_monthly_revenue(args)
            if name == "get_job_postings_count":
                return await parent.job_postings_count(args)
            if name == "get_options_implied_move":
                return await parent.options_implied_move(args)
            return _err(name, f"unknown tool: {name}")

    # -----------------------------------------------------------------------
    # Tool implementations
    # -----------------------------------------------------------------------

    async def google_trends(self, args: Dict[str, Any]) -> List[TextContent]:
        keywords = args.get("keywords")
        if not keywords:
            return _err("get_google_trends", "keywords is required")
        kwargs = {
            "keywords": keywords,
            "timeframe": args.get("timeframe", "today 12-m"),
            "geo": args.get("geo", "US"),
        }
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    _run_subprocess, _TRENDS_RUNNER, "get_google_trends",
                    kwargs, _SUBPROCESS_TIMEOUT_S,
                ),
                timeout=_TRENDS_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            return _err("get_google_trends",
                        f"timeout: pytrends did not return within {_TRENDS_TIMEOUT_S}s")
        if not result.get("success"):
            return _err("get_google_trends", result.get("error", "unknown error"))
        return _ok("get_google_trends", result["data"])

    async def finbert_sentiment(self, args: Dict[str, Any]) -> List[TextContent]:
        texts = args.get("texts")
        ticker = str(args.get("ticker", "")).upper()
        if not texts:
            return _err("get_finbert_sentiment", "texts is required", ticker)
        kwargs = {"texts": texts, "ticker": ticker}
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    _run_subprocess, _FINBERT_RUNNER, "get_finbert_sentiment",
                    kwargs, _FINBERT_SUB_TIMEOUT_S,
                ),
                timeout=_FINBERT_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            return _err("get_finbert_sentiment",
                        f"timeout: finbert did not return within {_FINBERT_TIMEOUT_S}s",
                        ticker)
        if not result.get("success"):
            return _err("get_finbert_sentiment", result.get("error", "unknown error"), ticker)
        return _ok("get_finbert_sentiment", result["data"], ticker)

    async def taiwan_monthly_revenue(self, args: Dict[str, Any]) -> List[TextContent]:
        codes = args.get("company_codes", [])
        if not codes:
            return _err("get_taiwan_monthly_revenue", "company_codes is required")
        months = int(args.get("months", 6))
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(_fetch_mops_revenue, codes[:5], months),
                timeout=30.0,
            )
        except asyncio.TimeoutError:
            return _err("get_taiwan_monthly_revenue", "MOPS request timed out after 30s")
        except Exception as exc:
            return _err("get_taiwan_monthly_revenue",
                        f"{type(exc).__name__}: {str(exc)[:200]}")
        return _ok("get_taiwan_monthly_revenue", result)

    async def job_postings_count(self, args: Dict[str, Any]) -> List[TextContent]:
        slug = str(args.get("company_slug", "")).strip().lower()
        if not slug:
            return _err("get_job_postings_count", "company_slug is required")
        ats = str(args.get("ats", "greenhouse"))
        dept_filter = args.get("department_filter")
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(_fetch_job_postings, slug, ats, dept_filter),
                timeout=20.0,
            )
        except asyncio.TimeoutError:
            return _err("get_job_postings_count", "job postings request timed out after 20s")
        except Exception as exc:
            return _err("get_job_postings_count",
                        f"{type(exc).__name__}: {str(exc)[:200]}")
        return _ok("get_job_postings_count", result, slug.upper())

    async def options_implied_move(self, args: Dict[str, Any]) -> List[TextContent]:
        ticker = str(args.get("ticker", "")).upper()
        spot = float(args.get("spot_price", 0))
        rows = args.get("options_chain_rows", [])
        target_expiry = args.get("target_expiry")
        if spot <= 0:
            return _err("get_options_implied_move", "spot_price must be > 0", ticker)
        if not rows:
            return _err("get_options_implied_move", "options_chain_rows is required", ticker)
        try:
            atm_call, atm_put, expiry = _find_atm_options(rows, spot, target_expiry)
            if not atm_call or not atm_put:
                return _err("get_options_implied_move",
                            "could not find ATM call and put in provided chain", ticker)

            call_ask = float(atm_call.get("ask") or atm_call.get("ask_price") or 0)
            put_ask  = float(atm_put.get("ask")  or atm_put.get("ask_price")  or 0)
            call_iv = float(atm_call.get("implied_volatility") or atm_call.get("impliedVolatility") or 0)
            put_iv  = float(atm_put.get("implied_volatility")  or atm_put.get("impliedVolatility")  or 0)
            strike  = float(atm_call.get("strike", 0))

            move = compute_implied_move(spot, call_ask, put_ask)
            skew_diff = put_iv - call_iv

            skew_label = (
                "put_heavy" if skew_diff > 0.03
                else "call_heavy" if skew_diff < -0.03
                else "balanced"
            )

            data = {
                "ticker": ticker,
                "spot_price": spot,
                "front_expiry": expiry,
                "atm_strike": strike,
                "atm_call_ask": call_ask,
                "atm_put_ask": put_ask,
                "implied_move_pct": move["implied_move_pct"],
                "straddle_cost": move["straddle_cost"],
                "call_iv": round(call_iv, 4),
                "put_iv": round(put_iv, 4),
                "put_call_skew": round(skew_diff, 4),
                "skew_label": skew_label,
                "risk_flag": (
                    "HIGH VOLATILITY WARNING: implied move >15%"
                    if move["implied_move_pct"] > 0.15 else None
                ),
            }
        except Exception as exc:
            return _err("get_options_implied_move",
                        f"{type(exc).__name__}: {str(exc)[:200]}", ticker)
        return _ok("get_options_implied_move", data, ticker)

    async def run_server(self):
        async with stdio_server() as (read, write):
            await self.server.run(read, write,
                                  self.server.create_initialization_options())


# ---------------------------------------------------------------------------
# Taiwan MOPS scraper (sync, runs in thread)
# ---------------------------------------------------------------------------

def _fetch_mops_revenue(company_codes: List[str], months: int) -> Dict[str, Any]:
    """Scrape MOPS monthly revenue for a list of Taiwan stock codes.

    MOPS endpoint: POST https://mops.twse.com.tw/mops/web/ajax_t05st10_ifrs
    Returns HTML table; parse with BeautifulSoup.
    """
    import requests
    from bs4 import BeautifulSoup
    from datetime import datetime

    now = datetime.now()
    results = {}

    for code in company_codes:
        try:
            resp = requests.post(
                "https://mops.twse.com.tw/mops/web/ajax_t05st10_ifrs",
                data={
                    "encodeURIComponent": "1",
                    "step": "1",
                    "firstin": "1",
                    "off": "1",
                    "keyword4": "",
                    "code1": "",
                    "TYPEK2": "",
                    "checkbtn": "",
                    "queryName": "co_id",
                    "inpuType": "co_id",
                    "TYPEK": "all",
                    "isnew": "false",
                    "co_id": code,
                    "year": str(now.year - 1911),  # ROC year
                },
                timeout=15,
                headers={
                    "User-Agent": "Mozilla/5.0",
                    "Referer": "https://mops.twse.com.tw/mops/web/t05st10_ifrs",
                },
            )
            resp.raise_for_status()
        except Exception as exc:
            results[code] = {"error": f"{type(exc).__name__}: {str(exc)[:150]}"}
            continue

        soup = BeautifulSoup(resp.text, "html.parser")
        tables = soup.find_all("table")
        if not tables:
            results[code] = {"error": "no table found in MOPS response"}
            continue

        rows_data = []
        for table in tables:
            trs = table.find_all("tr")
            for tr in trs[1:]:  # skip header
                tds = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
                if len(tds) >= 4:
                    rows_data.append(tds)

        if not rows_data:
            results[code] = {"error": "table parsed but no data rows found"}
            continue

        # Parse month rows: typically [year_month, revenue, MoM%, YoY%]
        parsed = []
        for row in rows_data[-months:]:
            try:
                parsed.append({
                    "period": row[0],
                    "revenue_ntd_m": float(row[1].replace(",", "")) if row[1] else None,
                    "mom_pct": row[2] if len(row) > 2 else None,
                    "yoy_pct": row[3] if len(row) > 3 else None,
                })
            except (ValueError, IndexError):
                continue

        results[code] = {
            "company_code": code,
            "months_returned": len(parsed),
            "months": parsed,
        }

    return {"companies": results, "codes_requested": company_codes}


# ---------------------------------------------------------------------------
# Job postings (Greenhouse / Lever public JSON APIs)
# ---------------------------------------------------------------------------

def _fetch_job_postings(slug: str, ats: str,
                        dept_filter: Optional[str]) -> Dict[str, Any]:
    """Fetch open job postings from Greenhouse or Lever public endpoints."""
    import requests

    def _try_greenhouse(s: str):
        url = f"https://boards-api.greenhouse.io/v1/boards/{s}/jobs"
        resp = requests.get(url, timeout=10,
                            headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data = resp.json()
        jobs = data.get("jobs", [])
        return jobs, "greenhouse", url

    def _try_lever(s: str):
        url = f"https://api.lever.co/v0/postings/{s}?mode=json"
        resp = requests.get(url, timeout=10,
                            headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        jobs = resp.json()
        return jobs, "lever", url

    jobs = None
    source = None
    source_url = None

    if ats == "greenhouse":
        result = _try_greenhouse(slug)
        if result is None:
            result = _try_lever(slug)
    else:
        result = _try_lever(slug)
        if result is None:
            result = _try_greenhouse(slug)

    if result is None:
        return {"error": f"company '{slug}' not found on Greenhouse or Lever",
                "slug": slug}

    jobs, source, source_url = result

    # Normalize department field across ATS formats
    by_dept: Dict[str, int] = {}
    for job in jobs:
        if source == "greenhouse":
            dept = (job.get("departments") or [{}])[0].get("name", "Unknown")
        else:
            dept = job.get("categories", {}).get("department", "Unknown")
        by_dept[dept] = by_dept.get(dept, 0) + 1

    # Optional department filter
    if dept_filter:
        jobs = [
            j for j in jobs
            if dept_filter.lower() in str(
                (j.get("departments") or [{}])[0].get("name", "") or
                j.get("categories", {}).get("department", "")
            ).lower()
        ]

    return {
        "slug": slug,
        "ats": source,
        "source_url": source_url,
        "total": len(jobs),
        "total_all_depts": sum(by_dept.values()),
        "by_department": dict(sorted(by_dept.items(), key=lambda x: -x[1])[:15]),
        "dept_filter_applied": dept_filter,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] != "server":
        print("Usage: python -m tools.altdata_server.server server", file=sys.stderr)
        sys.exit(1)
    print("[altdata] starting", file=sys.stderr, flush=True)
    srv = AltDataServer()
    asyncio.run(srv.run_server())
