"""Dividends and splits.

Neither had any coverage, and the absence of split data makes historical
per-share comparison silently wrong. Comparing NVDA's FY2023 EPS with FY2025
without knowing about the 10:1 split in June 2024 yields a conclusion off by an
order of magnitude, and nothing in the data signals the error.

Split ratios are reported as None when no split occurred, never as 1.0 — a
ratio of one would read as "a split happened that changed nothing", which is a
different claim from "no split happened".

The dividends carry the same hazard one level down. yfinance restates every
historical payment into today's share units, so AAPL's 2020-08-07 dividend
comes back as 0.205 against an as-declared $0.82 — a 4x error against an
as-filed share count from that quarter, and 4x in the *opposite* direction to
get_price_history, whose closes are back-adjusted the other way. Both bases are
reported: `amount` as the provider states it, `amount_as_declared` as the
company declared it, and `split_factor_since` as the multiple between them.
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


def _split_events(series: "pd.Series") -> List[Dict[str, Any]]:
    """Every split the provider knows about, oldest first, as {date, ratio}.

    Unfiltered by the reporting window on purpose: yfinance adjusts the whole
    dividend series for the whole split history, so a split that falls outside
    the window still sits inside the numbers reported within it.
    """
    events: List[Dict[str, Any]] = []
    if series is None or len(series) == 0:
        return events
    for stamp, value in series.items():
        try:
            ratio = float(value)
            moment = pd.Timestamp(stamp)
        except (TypeError, ValueError):
            continue
        if ratio != ratio or ratio <= 0 or ratio == 1.0:
            continue
        events.append({"date": moment.strftime("%Y-%m-%d"), "ratio": ratio})
    events.sort(key=lambda event: event["date"])
    return events


def as_declared_factor(dividend_date: str,
                       splits: List[Dict[str, Any]]) -> float:
    """Multiple that puts a split-adjusted dividend back on its as-declared basis.

    yfinance restates every historical dividend into today's share units
    without saying so. AAPL declared $0.82 a share on 2020-08-07 and the
    provider reports 0.205, because the 4-for-1 of 2020-08-31 has been divided
    into it. Multiplied by an as-filed share count from the same quarter that
    is out by exactly 4x -- and in the opposite direction to every price in
    this toolset, which is back-adjusted the other way.

    Strictly after, matching the rebasing in get_share_count_series: a payment
    on the split date is already stated in post-split units, and applying the
    ratio there too would invent the same error in reverse.
    """
    day = str(dividend_date)[:10]
    factor = 1.0
    for event in splits:
        if event["date"] > day:
            factor *= event["ratio"]
    return factor


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
            # The refusal carries the two collections this function documents
            # as always present, the same way the exception path below does.
            # Without them a caller that reached for result["splits"] after a
            # delisted symbol got a KeyError rather than the refusal, and one
            # writing .get("splits", []) read "no splits on record" out of a
            # lookup that never resolved the symbol. They stay empty and stay
            # beside success=False; the fields that would be claims about the
            # security -- pays_dividend, ttm_dividend, the counts -- are still
            # absent, because an unresolved lookup has nothing to say about
            # them.
            return {**unresolved, "dividends": [], "splits": []}
        raw_splits = getattr(handle, "splits", None)
        dividend_rows = _rows_since(getattr(handle, "dividends", None), cutoff)
        split_rows = _rows_since(raw_splits, cutoff)
        all_splits = _split_events(raw_splits)
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

    dividends = []
    restated: List[Dict[str, Any]] = []
    for row in dividend_rows:
        factor = as_declared_factor(row["date"], all_splits)
        dividends.append({
            "date": row["date"],
            "amount": row["value"],
            "amount_as_declared": row["value"] * factor,
            "split_factor_since": factor,
        })
        if factor != 1.0:
            restated.append(dividends[-1])

    warnings: List[Dict[str, Any]] = []
    if restated:
        described = ", ".join(
            f"{e['ratio']:g}-for-1 on {e['date']}" if e["ratio"] >= 1.0
            else f"1-for-{1.0 / e['ratio']:g} on {e['date']}"
            for e in all_splits
            if e["date"] > str(restated[-1]["date"])[:10])
        widest = max(r["split_factor_since"] for r in restated)
        warnings.append({
            "code": "dividends_split_adjusted",
            "message": (
                f"{ticker}: `amount` restates every dividend into today's "
                f"share units, so {len(restated)} of the payments here are not "
                f"the figures the company declared -- they have been divided "
                f"through by the splits since ({described}). "
                f"`amount_as_declared` carries those figures. Multiplying "
                f"`amount` by an as-filed share count from the same quarter is "
                f"out by up to {widest:g}x, and in the opposite direction to "
                f"get_price_history, whose closes are back-adjusted for the "
                f"same splits."),
            "restated_count": len(restated),
            "widest_split_factor": widest,
        })

    return {
        "ticker": ticker,
        "success": True,
        "years": years,
        "dividends": dividends,
        "dividend_basis": "split_adjusted",
        "splits": [{"date": r["date"], "ratio": r["value"]} for r in split_rows],
        "warnings": warnings,
        "dividend_count": len(dividend_rows),
        "split_count": len(split_rows),
        # The trailing twelve months are stated on the current share basis,
        # which is the basis today's quoted price is on -- so the yield built
        # from the two is right. It is labelled anyway: an unlabelled figure
        # beside a labelled one reads as a third basis.
        "ttm_dividend": ttm_dividend,
        "ttm_dividend_basis": "split_adjusted",
        "pays_dividend": ttm_dividend > 0,
        "latest_split_ratio": latest_split["value"] if latest_split else None,
        "latest_split_date": latest_split["date"] if latest_split else None,
    }
