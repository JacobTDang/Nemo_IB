"""Dividends and splits.

Neither had any coverage, and the absence of split data makes historical
per-share comparison silently wrong. Comparing NVDA's FY2023 EPS with FY2025
without knowing about the 10:1 split in June 2024 yields a conclusion off by an
order of magnitude, and nothing in the data signals the error.

Split ratios are reported as None when no split occurred, never as 1.0 — a
ratio of one would read as "a split happened that changed nothing", which is a
different claim from "no split happened".
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
import yfinance as yf


def _ticker(symbol: str):
    """Indirection so tests can substitute a fake without touching the network."""
    return yf.Ticker(symbol)


def _unresolved(symbol: str, handle: Any) -> Optional[Dict[str, Any]]:
    """A symbol the provider does not know is not a company that pays nothing.

    Without this, a typo answered `pays_dividend: false`, `split_count: 0`,
    `ttm_dividend: 0.0` and `success: true` -- concrete claims about a security
    that does not exist.
    """
    from .utils import unresolved_symbol_error
    return unresolved_symbol_error(symbol, handle)


def _rows_since(series: "pd.Series", cutoff: datetime) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if series is None or len(series) == 0:
        return rows
    for stamp, value in series.items():
        try:
            moment = pd.Timestamp(stamp)
            if moment.tzinfo is None:
                moment = moment.tz_localize("UTC")
        except (TypeError, ValueError):
            continue
        if moment.to_pydatetime() < cutoff:
            continue
        try:
            amount = float(value)
        except (TypeError, ValueError):
            continue
        if amount != amount:  # NaN
            continue
        rows.append({"date": moment.isoformat(), "value": amount})
    rows.sort(key=lambda r: r["date"], reverse=True)
    return rows


def get_corporate_actions(ticker: str, years: int = 10) -> Dict[str, Any]:
    """Dividend and split history within the window, newest first.

    `ttm_dividend` sums the trailing twelve months, which is the figure that
    matters for yield. `latest_split_ratio` is None when no split falls inside
    the window.
    """
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=365 * years)

    try:
        handle = _ticker(ticker)
        unresolved = _unresolved(ticker, handle)
        if unresolved is not None:
            return unresolved
        dividend_rows = _rows_since(getattr(handle, "dividends", None), cutoff)
        split_rows = _rows_since(getattr(handle, "splits", None), cutoff)
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        return {
            "ticker": ticker,
            "success": False,
            "error": f"{type(exc).__name__}: {exc}",
            "dividends": [],
            "splits": [],
        }

    ttm_cutoff = (now - timedelta(days=365)).isoformat()
    ttm_dividend = float(sum(
        row["value"] for row in dividend_rows if row["date"] >= ttm_cutoff))

    latest_split: Optional[Dict[str, Any]] = split_rows[0] if split_rows else None

    return {
        "ticker": ticker,
        "success": True,
        "years": years,
        "dividends": [{"date": r["date"], "amount": r["value"]}
                      for r in dividend_rows],
        "splits": [{"date": r["date"], "ratio": r["value"]} for r in split_rows],
        "dividend_count": len(dividend_rows),
        "split_count": len(split_rows),
        "ttm_dividend": ttm_dividend,
        "pays_dividend": ttm_dividend > 0,
        "latest_split_ratio": latest_split["value"] if latest_split else None,
        "latest_split_date": latest_split["date"] if latest_split else None,
    }
