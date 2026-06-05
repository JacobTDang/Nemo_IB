"""Subprocess runner for Google Trends (pytrends) calls.

Invoked by altdata_server/server.py as an isolated child process so pytrends'
session state never contends with the MCP framework's asyncio event loop.

Usage (internal):
  python trends_runner.py get_google_trends <json_args>

Writes exactly one JSON line to stdout:
  {"success": true, "data": {...}}    on success
  {"success": false, "error": "..."}  on failure

Caching (Fix B):
  Results are cached in SQLite at db_cache/trends_cache.db for 12 hours
  to eliminate repeated 429 rate-limit errors from pytrends. Cache key is
  SHA-256(sorted_keywords + timeframe + geo).

Retry (Fix B):
  On TooManyRequests / 429, the runner backs off 10s then 20s before
  giving up. Total extra latency: up to 30s. Server timeout is 45s.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from typing import Any


# ---------------------------------------------------------------------------
# Cache helpers (SQLite, 12h TTL)
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CACHE_DB = os.path.join(_REPO_ROOT, "db_cache", "trends_cache.db")
_CACHE_TTL_SECONDS = 12 * 3600


def _ensure_cache_db() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(_CACHE_DB), exist_ok=True)
    conn = sqlite3.connect(_CACHE_DB, timeout=5)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trends_cache (
            cache_key   TEXT PRIMARY KEY,
            payload     TEXT NOT NULL,
            cached_at   REAL NOT NULL
        )
    """)
    conn.commit()
    return conn


def _cache_key(keywords: list, timeframe: str, geo: str) -> str:
    blob = json.dumps(
        {"keywords": sorted(keywords), "timeframe": timeframe, "geo": geo},
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()


def _cache_get(key: str) -> Any | None:
    try:
        conn = _ensure_cache_db()
        row = conn.execute(
            "SELECT payload, cached_at FROM trends_cache WHERE cache_key = ?", (key,)
        ).fetchone()
        conn.close()
        if row is None:
            return None
        payload, cached_at = row
        age = time.time() - cached_at
        if age > _CACHE_TTL_SECONDS:
            return None
        return json.loads(payload)
    except Exception:
        return None


def _cache_put(key: str, data: Any) -> None:
    try:
        conn = _ensure_cache_db()
        conn.execute(
            "INSERT OR REPLACE INTO trends_cache (cache_key, payload, cached_at) VALUES (?, ?, ?)",
            (key, json.dumps(data, default=str), time.time()),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass  # Cache write failure is non-fatal


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _ok(data: Any) -> None:
    print(json.dumps({"success": True, "data": data}, default=str), flush=True)


def _fail(msg: str) -> None:
    print(json.dumps({"success": False, "error": msg}), flush=True)


# ---------------------------------------------------------------------------
# YoY computation
# ---------------------------------------------------------------------------

def _compute_yoy_ratio(df, keywords: list) -> float | None:
    """Compute average of last 13 weeks vs same 13 weeks 1 year prior."""
    try:
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


# ---------------------------------------------------------------------------
# Retry-aware pytrends fetch
# ---------------------------------------------------------------------------

_RETRY_DELAYS = [10, 20]   # seconds between attempts on 429


def _fetch_with_retry(keywords: list, timeframe: str, geo: str):
    """Call pytrends up to 3 times (1 + 2 retries). Returns (df, error_str)."""
    from pytrends.request import TrendReq

    last_error: str = ""
    for attempt, delay in enumerate([0] + _RETRY_DELAYS):
        if delay:
            time.sleep(delay)
        try:
            pt = TrendReq(hl="en-US", tz=360, timeout=(10, 25))
            pt.build_payload(keywords, cat=0, timeframe=timeframe, geo=geo)
            df = pt.interest_over_time()
            if df is None or df.empty:
                last_error = "pytrends returned empty data — possible rate limit or invalid keywords"
                continue
            return df, None
        except Exception as exc:
            cls = type(exc).__name__
            msg = str(exc)[:300]
            last_error = f"{cls}: {msg}"
            is_429 = (
                "429" in msg
                or "TooManyRequests" in cls
                or "ResponseError" in cls
                or "rate" in msg.lower()
            )
            if not is_429:
                return None, last_error

    return None, last_error


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------

def main() -> int:
    if len(sys.argv) < 3:
        _fail("usage: trends_runner.py <tool_name> <json_args>")
        return 1

    tool_name = sys.argv[1]
    if tool_name != "get_google_trends":
        _fail(f"unknown tool: {tool_name}")
        return 1

    try:
        args = json.loads(sys.argv[2])
    except json.JSONDecodeError as exc:
        _fail(f"invalid json args: {exc}")
        return 1

    keywords = args.get("keywords")
    if not keywords or not isinstance(keywords, list):
        _fail("keywords must be a non-empty list of strings")
        return 1

    keywords = [str(k) for k in keywords[:5]]   # pytrends max 5 terms
    timeframe = str(args.get("timeframe", "today 12-m"))
    geo = str(args.get("geo", "US"))

    # ---- cache check ----
    key = _cache_key(keywords, timeframe, geo)
    cached = _cache_get(key)
    if cached is not None:
        cached["cached"] = True
        _ok(cached)
        return 0

    # ---- pytrends import check ----
    try:
        from pytrends.request import TrendReq  # noqa: F401
    except ImportError as exc:
        _fail(f"pytrends not installed: {exc}")
        return 1

    # ---- fetch with retry ----
    df, error = _fetch_with_retry(keywords, timeframe, geo)
    if error:
        _fail(error)
        return 1

    # Drop Google's isPartial column
    if "isPartial" in df.columns:
        df = df.drop(columns=["isPartial"])

    df.index = df.index.astype(str)
    records = df.reset_index().rename(columns={"date": "date"}).to_dict(orient="records")

    yoy_ratio = _compute_yoy_ratio(df, keywords)
    yoy_signal = (
        "bullish" if yoy_ratio and yoy_ratio > 1.10
        else "bearish" if yoy_ratio and yoy_ratio < 0.90
        else "neutral"
    )

    result = {
        "keywords": keywords,
        "timeframe": timeframe,
        "geo": geo,
        "record_count": len(records),
        "records": records,
        "yoy_ratio": yoy_ratio,
        "yoy_signal": yoy_signal,
        "cached": False,
    }

    # ---- write to cache ----
    _cache_put(key, result)

    _ok(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
