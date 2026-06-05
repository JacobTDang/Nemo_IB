"""Subprocess runner for yfinance options-chain fetches.

Invoked by altdata_server/server.py as an isolated child process. yfinance
lazy-initializes its Yahoo session/cookie/cache on first use; inside the
long-running async MCP server that cold start can stall well past the handler
timeout (and a hung asyncio.to_thread cannot be killed, leaking a thread). A
fresh subprocess does the same fetch in ~1s, and subprocess.run(timeout=...)
hard-kills it on hang — no leak.

Usage (internal):
  python options_runner.py get_options_chain <json_args>

Writes exactly one JSON line to stdout:
  {"success": true, "data": {"rows": [...], "row_count": N}}   on success
  {"success": false, "error": "..."}                            on failure
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, List, Dict


def _ok(data: Any) -> None:
    print(json.dumps({"success": True, "data": data}, default=str), flush=True)


def _fail(msg: str) -> None:
    print(json.dumps({"success": False, "error": msg}), flush=True)


def _safe_float(v: Any, default: float = 0.0) -> float:
    """float() mapping None/non-numeric/NaN to default (NaN != NaN)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return f if f == f else default


def fetch_options_chain(ticker: str, near_days: int = 60) -> List[Dict]:
    """Fetch a complete options chain via yfinance (no row-count cap).

    Captures ask, bid, last_price, and IV so the caller can fall back to
    last_price/bid when ask is 0 (market closed)."""
    import yfinance as yf

    t = yf.Ticker(ticker)
    exps = t.options  # tuple of 'YYYY-MM-DD' strings, sorted
    if not exps:
        return []

    today = datetime.now(timezone.utc).date()
    cutoff = today + timedelta(days=near_days)
    target_exps = [
        e for e in exps
        if today < datetime.strptime(e, "%Y-%m-%d").date() <= cutoff
    ][:4]
    if not target_exps:
        target_exps = [exps[0]]  # nearest available expiry

    rows: List[Dict] = []
    for exp in target_exps:
        try:
            chain = t.option_chain(exp)
        except Exception:
            continue
        for df, otype in ((chain.calls, "call"), (chain.puts, "put")):
            for _, row in df.iterrows():
                rows.append({
                    "expiration": exp, "option_type": otype,
                    "strike": _safe_float(row.get("strike")),
                    "ask": _safe_float(row.get("ask")),
                    "bid": _safe_float(row.get("bid")),
                    "last_price": _safe_float(row.get("lastPrice")),
                    "implied_volatility": _safe_float(row.get("impliedVolatility")),
                })
    return rows


def main() -> int:
    if len(sys.argv) < 3:
        _fail("usage: options_runner.py <tool_name> <json_args>")
        return 1

    tool_name = sys.argv[1]
    if tool_name != "get_options_chain":
        _fail(f"unknown tool: {tool_name}")
        return 1

    try:
        args = json.loads(sys.argv[2])
    except json.JSONDecodeError as exc:
        _fail(f"invalid json args: {exc}")
        return 1

    ticker = str(args.get("ticker", "")).strip()
    if not ticker:
        _fail("ticker is required")
        return 1
    near_days = int(args.get("near_days", 60))

    try:
        from importlib import import_module
        import_module("yfinance")
    except ImportError as exc:
        _fail(f"yfinance not installed: {exc}")
        return 1

    try:
        rows = fetch_options_chain(ticker, near_days)
    except Exception as exc:
        _fail(f"{type(exc).__name__}: {str(exc)[:200]}")
        return 1

    _ok({"rows": rows, "row_count": len(rows)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
