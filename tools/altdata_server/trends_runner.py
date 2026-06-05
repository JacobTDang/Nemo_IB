"""Subprocess runner for Google Trends (pytrends) calls.

Invoked by altdata_server/server.py as an isolated child process so pytrends'
session state never contends with the MCP framework's asyncio event loop.

Usage (internal):
  python trends_runner.py get_google_trends <json_args>

Writes exactly one JSON line to stdout:
  {"success": true, "data": {...}}    on success
  {"success": false, "error": "..."}  on failure
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from typing import Any


def _ok(data: Any) -> None:
    print(json.dumps({"success": True, "data": data}, default=str), flush=True)


def _fail(msg: str) -> None:
    print(json.dumps({"success": False, "error": msg}), flush=True)


def _compute_yoy_ratio(df, keywords: list) -> float | None:
    """Compute average of last 13 weeks vs same 13 weeks 1 year prior."""
    try:
        import pandas as pd
        if df is None or df.empty or not keywords:
            return None
        col = keywords[0]
        if col not in df.columns:
            return None
        series = df[col].astype(float)
        if len(series) < 26:
            return None
        recent_13 = series.iloc[-13:].mean()
        prior_13 = series.iloc[-26:-13].mean()
        if prior_13 == 0:
            return None
        return round(recent_13 / prior_13, 3)
    except Exception:
        return None


def main() -> int:
    if len(sys.argv) < 3:
        _fail("usage: trends_runner.py <tool_name> <json_args>")
        return 1

    tool_name = sys.argv[1]
    try:
        args = json.loads(sys.argv[2])
    except json.JSONDecodeError as exc:
        _fail(f"invalid json args: {exc}")
        return 1

    if tool_name != "get_google_trends":
        _fail(f"unknown tool: {tool_name}")
        return 1

    keywords = args.get("keywords")
    if not keywords or not isinstance(keywords, list):
        _fail("keywords must be a non-empty list of strings")
        return 1

    keywords = [str(k) for k in keywords[:5]]  # pytrends max 5 terms
    timeframe = str(args.get("timeframe", "today 12-m"))
    geo = str(args.get("geo", "US"))

    try:
        from pytrends.request import TrendReq
    except ImportError as exc:
        _fail(f"pytrends not installed: {exc}")
        return 1

    try:
        pytrends = TrendReq(hl="en-US", tz=360, timeout=(10, 25))
        pytrends.build_payload(keywords, cat=0, timeframe=timeframe, geo=geo)
        df = pytrends.interest_over_time()
    except Exception as exc:
        _fail(f"{type(exc).__name__}: {str(exc)[:300]}")
        return 1

    if df is None or df.empty:
        _fail("pytrends returned empty data — possible rate limit or invalid keywords")
        return 1

    # Drop the isPartial column Google appends
    if "isPartial" in df.columns:
        df = df.drop(columns=["isPartial"])

    # Convert to list of {date: ISO string, keyword: value} dicts
    df.index = df.index.astype(str)
    records = df.reset_index().rename(columns={"date": "date"}).to_dict(orient="records")

    yoy_ratio = _compute_yoy_ratio(df, keywords)

    _ok({
        "keywords": keywords,
        "timeframe": timeframe,
        "geo": geo,
        "record_count": len(records),
        "records": records,
        "yoy_ratio": yoy_ratio,
        "yoy_signal": (
            "bullish" if yoy_ratio and yoy_ratio > 1.10
            else "bearish" if yoy_ratio and yoy_ratio < 0.90
            else "neutral"
        ),
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
