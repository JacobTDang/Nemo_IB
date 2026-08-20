"""Options math helpers, post-merge into the financial engine.

These encode four bugs documented in docs/known_issues.md: put-call parity
violation, after-hours last_price fallback, yfinance NaN/sentinel handling,
and the zero-spot guard. They are the safety net for the merge.
"""
from __future__ import annotations

import os
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from tools.financial_modeling_engine.utils import (  # noqa: E402
    _find_atm_options,
    _leg_price,
    _safe_float,
    compute_implied_move,
)


def test_safe_float_handles_nan():
    assert _safe_float(float("nan")) == 0.0
    assert _safe_float(float("nan"), default=1.5) == 1.5
    assert _safe_float(None) == 0.0
    assert _safe_float("abc") == 0.0
    assert _safe_float("2.5") == 2.5


def test_leg_price_prefers_live_ask():
    price, stale = _leg_price({"ask": 3.0, "last_price": 9.0, "bid": 1.0})
    assert price == 3.0
    assert stale is False


def test_leg_price_falls_back_to_last_when_ask_zero():
    price, stale = _leg_price({"ask": 0.0, "last_price": 4.25, "bid": 1.0})
    assert price == 4.25
    assert stale is True


def test_leg_price_falls_back_to_bid_when_no_last():
    price, stale = _leg_price({"ask": 0.0, "last_price": 0.0, "bid": 2.0})
    assert price == 2.0
    assert stale is True


def test_leg_price_all_zero():
    price, stale = _leg_price({"ask": 0.0, "last_price": 0.0, "bid": 0.0})
    assert price == 0.0
    assert stale is True


def test_leg_price_nan_ask_does_not_mask_ask_price():
    # A NaN ask is truthy; `ask or ask_price` would swallow a valid ask_price.
    price, stale = _leg_price({"ask": float("nan"), "ask_price": 5.0})
    assert price == 5.0
    assert stale is False


def test_compute_implied_move_basic_math():
    out = compute_implied_move(spot=100.0, atm_call_ask=3.0, atm_put_ask=2.0)
    assert out["straddle_cost"] == 5.0
    assert out["implied_move_pct"] == 0.05


def test_compute_implied_move_zero_spot():
    out = compute_implied_move(spot=0.0, atm_call_ask=3.0, atm_put_ask=2.0)
    assert out["implied_move_pct"] == 0.0


def test_compute_implied_move_nan_ask_no_nan_output():
    out = compute_implied_move(spot=100.0, atm_call_ask=float("nan"), atm_put_ask=2.0)
    assert out["straddle_cost"] == 2.0
    assert out["implied_move_pct"] == out["implied_move_pct"]  # not NaN


def _rows(expiry: str = "2099-01-15"):
    return [
        {"expiration": expiry, "option_type": "call", "strike": 95.0, "ask": 7.0},
        {"expiration": expiry, "option_type": "call", "strike": 100.0, "ask": 3.0},
        {"expiration": expiry, "option_type": "put", "strike": 100.0, "ask": 2.0},
        {"expiration": expiry, "option_type": "put", "strike": 105.0, "ask": 8.0},
    ]


def test_find_atm_options_selects_nearest_strike():
    call, put, expiry = _find_atm_options(_rows(), spot=100.0)
    assert call["strike"] == 100.0
    assert put["strike"] == 100.0
    assert expiry == "2099-01-15"


def test_find_atm_options_empty_returns_none():
    assert _find_atm_options([], spot=100.0) == (None, None, None)


def test_find_atm_uses_last_price_after_hours():
    # ask == 0 everywhere (market closed); selection must still find the strike.
    rows = [
        {"expiration": "2099-01-15", "option_type": "call", "strike": 100.0,
         "ask": 0.0, "last_price": 3.0},
        {"expiration": "2099-01-15", "option_type": "put", "strike": 100.0,
         "ask": 0.0, "last_price": 2.0},
    ]
    call, put, _ = _find_atm_options(rows, spot=100.0)
    assert call is not None and put is not None
    assert call["strike"] == 100.0


def test_skew_classification():
    rows = [
        {"expiration": "2099-08-15", "option_type": "call", "strike": 100,
         "ask": 3.0, "implied_volatility": 0.30},
        {"expiration": "2099-08-15", "option_type": "put", "strike": 100,
         "ask": 4.0, "implied_volatility": 0.38},
    ]
    call, put, _ = _find_atm_options(rows, 100.0, "2099-08-15")
    skew_diff = float(put["implied_volatility"]) - float(call["implied_volatility"])
    assert skew_diff > 0.03  # put_heavy


def test_chain_to_rows_normalizes_yfinance_columns():
    import pandas as pd

    from tools.financial_modeling_engine.utils import _chain_to_rows

    class _Chain:
        calls = pd.DataFrame([{"strike": 100.0, "ask": 3.0, "bid": 2.8,
                               "lastPrice": 2.9, "impliedVolatility": 0.35}])
        puts = pd.DataFrame([{"strike": 100.0, "ask": 2.0, "bid": 1.8,
                              "lastPrice": 1.9, "impliedVolatility": 0.33}])

    rows = _chain_to_rows(_Chain(), "2099-01-15")

    assert len(rows) == 2
    call = next(r for r in rows if r["option_type"] == "call")
    assert call["expiration"] == "2099-01-15"
    assert call["strike"] == 100.0
    assert call["ask"] == 3.0
    assert call["last_price"] == 2.9          # camelCase -> snake_case
    assert call["implied_volatility"] == 0.35
    assert {r["option_type"] for r in rows} == {"call", "put"}


def test_straddle_legs_uses_ask_when_parity_holds():
    from tools.financial_modeling_engine.utils import _straddle_legs

    # C - P = 3 - 2 = 1; S - K = 100 - 99 = 1. Parity holds, asks are trusted.
    call = {"strike": 99.0, "ask": 3.0, "last_price": 2.5}
    put = {"strike": 99.0, "ask": 2.0, "last_price": 1.5}
    call_px, put_px, stale = _straddle_legs(call, put, spot=100.0)
    assert (call_px, put_px) == (3.0, 2.0)
    assert stale is False


def test_straddle_legs_rebuilds_on_parity_violation():
    """Live ORCL case: call 6.75 / put 28.35 at strike 237.5, spot 236.34 -- a
    $21 parity violation from junk wide quotes left at the close. A nonzero ask
    is not necessarily a sane ask."""
    from tools.financial_modeling_engine.utils import _straddle_legs

    call = {"strike": 237.5, "ask": 6.75, "last_price": 5.10}
    put = {"strike": 237.5, "ask": 28.35, "last_price": 6.40}
    call_px, put_px, stale = _straddle_legs(call, put, spot=236.34)

    assert (call_px, put_px) == (5.10, 6.40), "should rebuild both legs off last_price"
    assert stale is True


def test_straddle_legs_keeps_ask_when_no_fallback_available():
    from tools.financial_modeling_engine.utils import _straddle_legs

    # Parity violated but no last_price/bid to fall back to: keep the asks
    # rather than fabricating a number.
    call = {"strike": 237.5, "ask": 6.75}
    put = {"strike": 237.5, "ask": 28.35}
    call_px, put_px, stale = _straddle_legs(call, put, spot=236.34)
    assert (call_px, put_px) == (6.75, 28.35)
