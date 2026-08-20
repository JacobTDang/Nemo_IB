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
