"""Relative volume, average dollar volume, and average true range.

These are execution inputs, not company research. They answer how large a
position can be, how long it takes to get out of it, and how wide a stop has
to be to survive ordinary noise. None of them says anything about whether the
business is any good.

The bars were already being pulled by get_price_history; only the derived
metrics were missing, so both tools now read the same fetch.

Three deliberate choices, each of which the obvious implementation gets wrong:

* ATR uses Wilder's smoothing over the true range, and true range includes the
  overnight gap. On a gap day high-minus-low can understate the real move by
  an order of magnitude, which is exactly the day a stop matters.
* ADV is dollars. Share counts do not tell you whether a $50m position can be
  exited; a million shares is $2m of a penny stock and $700m of BRK-A.
* RVOL measures the latest session against the sessions *before* it. Folding
  the session into its own baseline drags every reading toward one.

A metric that cannot be computed from the available history is None with a
note saying what was missing, never a short-window average dressed up as the
requested one.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import pandas as pd

from .utils import PRICE_BASIS, fetch_daily_bars

_EASTERN = ZoneInfo("America/New_York")
_US_EQUITY_CLOSE_HOUR = 16
_REQUIRED_COLUMNS = ("Open", "High", "Low", "Close", "Volume")
_DEFAULT_ADV_WINDOWS: Tuple[int, ...] = (20, 60)


def _now_eastern() -> datetime:
    """Indirection so tests can freeze the session clock."""
    return datetime.now(timezone.utc).astimezone(_EASTERN)


def _is_partial_session(stamp: Any) -> bool:
    """True when the newest bar is the session currently in progress.

    yfinance hands back a bar for today from the opening print onward. Its
    volume is whatever has traded so far, which is not comparable with a
    baseline of completed sessions.
    """
    try:
        moment = pd.Timestamp(stamp)
    except (TypeError, ValueError):
        return False
    if moment.tzinfo is not None:
        moment = moment.tz_convert(_EASTERN)
    now = _now_eastern()
    return moment.date() == now.date() and now.hour < _US_EQUITY_CLOSE_HOUR


def _true_range(frame: "pd.DataFrame") -> "pd.Series":
    """max(high-low, |high - prev close|, |prev close - low|), gaps included.

    The first bar has no prior close and therefore no true range; it is
    dropped rather than filled with high-low, which would seed the average
    with a value that is too small on any gap.
    """
    previous_close = frame["Close"].shift(1)
    spans = pd.concat(
        [
            frame["High"] - frame["Low"],
            (frame["High"] - previous_close).abs(),
            (previous_close - frame["Low"]).abs(),
        ],
        axis=1,
    )
    return spans.max(axis=1).iloc[1:]


def _wilder_average(values: "pd.Series", period: int) -> Optional[float]:
    """Wilder's smoothing: seed on the first `period` values, then decay.

    ATR_t = (ATR_(t-1) * (period - 1) + value_t) / period. This is not a
    rolling mean -- it never drops an old observation, it decays it -- and the
    two diverge sharply after a volatility break.
    """
    if len(values) < period:
        return None
    running = float(values.iloc[:period].mean())
    for value in values.iloc[period:]:
        running = (running * (period - 1) + float(value)) / period
    return running


def _atr_block(complete: "pd.DataFrame", period: int) -> Dict[str, Any]:
    ranges = _true_range(complete)
    atr = _wilder_average(ranges, period) if period > 0 else None
    if atr is None:
        return {
            "period": period,
            "method": "wilder",
            "atr": None,
            "atr_pct_of_price": None,
            "note": (f"needs {period} true ranges ({period + 1} complete bars); "
                     f"{len(complete)} bar(s) available"),
        }
    last_close = float(complete["Close"].iloc[-1])
    pct = round(atr / last_close * 100, 3) if last_close else None
    return {
        "period": period,
        "method": "wilder",
        "atr": round(atr, 4),
        "atr_pct_of_price": pct,
        "note": None,
    }


def _rvol_block(bars: "pd.DataFrame", lookback: int, partial: bool) -> Dict[str, Any]:
    latest_volume = float(bars["Volume"].iloc[-1])
    baseline = bars["Volume"].iloc[-(lookback + 1):-1]
    notes: List[str] = []
    if partial:
        notes.append(
            "latest bar is a partial session -- its volume is understated "
            "against a baseline of completed sessions")

    if len(baseline) < lookback:
        notes.append(f"needs {lookback} prior sessions for the baseline; "
                     f"{len(baseline)} available")
        return {
            "lookback_days": lookback,
            "latest_volume": int(latest_volume),
            "average_volume": None,
            "ratio": None,
            "note": " | ".join(notes),
        }

    average = float(baseline.mean())
    if average <= 0:
        notes.append("baseline average volume is zero -- no ratio is defined")
        return {
            "lookback_days": lookback,
            "latest_volume": int(latest_volume),
            "average_volume": int(average),
            "ratio": None,
            "note": " | ".join(notes),
        }

    return {
        "lookback_days": lookback,
        "latest_volume": int(latest_volume),
        "average_volume": int(round(average)),
        "ratio": round(latest_volume / average, 3),
        "note": " | ".join(notes) or None,
    }


def _adv_block(complete: "pd.DataFrame", windows: Tuple[int, ...],
               excluded_partial_session: bool) -> Dict[str, Any]:
    """Mean dollar volume per session, each session priced at its own close.

    Pricing the whole window at the latest close would rewrite history after
    any large move, which is when the exit question actually gets asked.
    """
    dollars = complete["Close"] * complete["Volume"]
    block: Dict[str, Any] = {
        "windows_days": list(windows),
        "excluded_partial_session": excluded_partial_session,
        "note": None,
    }
    missing: List[str] = []
    for window in windows:
        key = f"{window}d"
        if len(complete) < window:
            block[key] = {"dollar_volume": None, "share_volume": None}
            missing.append(str(window))
            continue
        block[key] = {
            "dollar_volume": round(float(dollars.iloc[-window:].mean()), 2),
            "share_volume": round(float(complete["Volume"].iloc[-window:].mean()), 1),
        }
    if missing:
        block["note"] = (f"history is {len(complete)} complete session(s); "
                         f"window(s) {', '.join(missing)}d could not be filled")
    return block


# The blocks the success path always publishes. A refusal that drops them
# hands a caller a KeyError instead of the refusal it was given, and a caller
# writing .get("atr", {}) reads "no ATR for this name" out of a fetch that
# never happened. None is not a claim about the security; an
# atr_pct_of_price of 0.0 would be.
_REFUSAL_BLOCKS = ("rvol", "adv", "atr")


def _refusal(ticker: str, error: str) -> Dict[str, Any]:
    """A failed lookup, in the shape a successful one has."""
    return {"ticker": ticker.upper(), "success": False, "error": error,
            **{block: None for block in _REFUSAL_BLOCKS}}


def get_trading_metrics(ticker: str, period: str = '1y',
                        rvol_lookback: int = 20,
                        atr_period: int = 14) -> Dict[str, Any]:
    """RVOL, average dollar volume and ATR from one daily-bar fetch.

    `period` only has to be long enough to cover the widest window asked for;
    the default year comfortably covers a 60-day ADV.
    """
    try:
        frame = fetch_daily_bars(ticker, period)
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        return _refusal(
            ticker,
            f"yfinance history fetch failed: {type(exc).__name__}: {exc}")

    if frame is None or frame.empty:
        return _refusal(ticker, "no price history returned")

    absent = [column for column in _REQUIRED_COLUMNS if column not in frame.columns]
    if absent:
        return _refusal(
            ticker, f"price history is missing column(s): {', '.join(absent)}")

    bars = frame.dropna(subset=list(_REQUIRED_COLUMNS))
    if bars.empty:
        return _refusal(
            ticker, "no price history returned with complete OHLCV")

    partial = _is_partial_session(bars.index[-1])
    complete = bars.iloc[:-1] if partial else bars
    if complete.empty:
        return {"ticker": ticker.upper(), "success": False,
                "error": "the only bar returned is an in-progress session"}

    return {
        "ticker": ticker.upper(),
        "success": True,
        "error": None,
        "period_requested": period,
        "bars_returned": len(bars),
        # Same bars as get_price_history, so the same basis. ADV and ATR are
        # same-day ratios and survive the adjustment intact; `last_close` is a
        # price like any other here, and an unlabelled price beside labelled
        # ones reads as a third basis.
        "price_basis": PRICE_BASIS,
        "as_of": pd.Timestamp(bars.index[-1]).strftime("%Y-%m-%d"),
        "last_close": round(float(bars["Close"].iloc[-1]), 4),
        "latest_bar_is_partial_session": partial,
        "rvol": _rvol_block(bars, rvol_lookback, partial),
        "adv": _adv_block(complete, _DEFAULT_ADV_WINDOWS, partial),
        "atr": _atr_block(complete, atr_period),
    }
