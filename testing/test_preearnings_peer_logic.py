"""Phase B unit tests — deterministic peer-readthrough logic (no network)."""
from __future__ import annotations

import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.preearnings.peer_logic import (
    quarter_window,
    reported_this_quarter,
    rank_peer_relevance,
    select_peers_for_fanout,
    aggregate_readthroughs,
)


# ---------------------------------------------------------------------------
# quarter_window — inferred from the target's own cadence
# ---------------------------------------------------------------------------

def test_quarter_window_normal_uses_last_earnings():
    start, end = quarter_window("2026-03-01", "2026-06-01")
    assert start == date(2026, 3, 1) and end == date(2026, 6, 1)


def test_quarter_window_missing_last_earnings_defaults():
    start, end = quarter_window(None, "2026-06-01")
    assert end == date(2026, 6, 1)
    assert (end - start).days == 95


def test_quarter_window_clamps_stale_last_earnings():
    # last earnings a year ago -> clamped to max_span_days
    start, end = quarter_window("2025-06-01", "2026-06-01", max_span_days=130)
    assert (end - start).days == 130


def test_quarter_window_future_last_earnings_falls_back():
    start, end = quarter_window("2026-09-01", "2026-06-01")
    assert start < end and (end - start).days == 95


# ---------------------------------------------------------------------------
# reported_this_quarter
# ---------------------------------------------------------------------------

def test_reported_this_quarter_inside_and_boundary():
    w = (date(2026, 3, 1), date(2026, 6, 1))
    assert reported_this_quarter("2026-04-15", w) is True
    assert reported_this_quarter("2026-03-01", w) is True   # inclusive
    assert reported_this_quarter("2026-06-01", w) is True   # inclusive


def test_reported_this_quarter_outside_and_none():
    w = (date(2026, 3, 1), date(2026, 6, 1))
    assert reported_this_quarter("2026-02-28", w) is False
    assert reported_this_quarter("2026-06-02", w) is False
    assert reported_this_quarter(None, w) is False
    assert reported_this_quarter("", w) is False


# ---------------------------------------------------------------------------
# rank_peer_relevance — generic relationship types, not companies
# ---------------------------------------------------------------------------

def test_relevance_by_relationship_type():
    assert rank_peer_relevance("supplier") == 1.0
    assert rank_peer_relevance("customer") == 1.0
    assert rank_peer_relevance("competitor") == 0.7
    assert rank_peer_relevance("Peer") == 0.7        # case-insensitive
    assert rank_peer_relevance("adjacent") == 0.4


def test_relevance_unknown_and_none_default():
    assert rank_peer_relevance("something_else") == 0.5
    assert rank_peer_relevance(None) == 0.5


# ---------------------------------------------------------------------------
# select_peers_for_fanout
# ---------------------------------------------------------------------------

def test_select_filters_to_reported_and_ranks():
    w = (date(2026, 3, 1), date(2026, 6, 1))
    peers = [
        {"ticker": "P1", "relationship": "adjacent",   "report_date": "2026-04-01"},
        {"ticker": "P2", "relationship": "supplier",   "report_date": "2026-05-20"},
        {"ticker": "P3", "relationship": "competitor", "report_date": "2026-01-01"},  # outside window
        {"ticker": "P4", "relationship": "customer",   "report_date": "2026-05-25"},
    ]
    out = select_peers_for_fanout(peers, w, max_n=6)
    tickers = [p["ticker"] for p in out]
    assert "P3" not in tickers                 # filtered (reported last quarter)
    assert tickers[0] in {"P2", "P4"}          # supplier/customer rank first
    assert out[0]["relevance"] == 1.0


def test_select_caps_at_max_n():
    w = (date(2026, 3, 1), date(2026, 6, 1))
    peers = [{"ticker": f"P{i}", "relationship": "competitor", "report_date": "2026-04-01"}
             for i in range(10)]
    out = select_peers_for_fanout(peers, w, max_n=3)
    assert len(out) == 3


def test_select_zero_eligible_returns_empty():
    w = (date(2026, 3, 1), date(2026, 6, 1))
    peers = [{"ticker": "P1", "relationship": "peer", "report_date": "2025-01-01"}]
    assert select_peers_for_fanout(peers, w) == []


# ---------------------------------------------------------------------------
# aggregate_readthroughs — relevance-weighted
# ---------------------------------------------------------------------------

def test_aggregate_bullish_majority():
    items = [
        {"ticker": "A", "direction": "bullish", "magnitude": 0.8, "relevance": 1.0},
        {"ticker": "B", "direction": "bullish", "magnitude": 0.6, "relevance": 0.7},
        {"ticker": "C", "direction": "neutral", "magnitude": 0.0, "relevance": 0.7},
    ]
    out = aggregate_readthroughs(items)
    assert out["direction"] == "bullish"
    assert out["n"] == 3 and out["score"] > 0.15


def test_aggregate_high_relevance_bearish_outweighs_low_bullish():
    items = [
        {"ticker": "SUP", "direction": "bearish", "magnitude": 0.9, "relevance": 1.0},
        {"ticker": "ADJ", "direction": "bullish", "magnitude": 0.5, "relevance": 0.4},
    ]
    out = aggregate_readthroughs(items)
    assert out["direction"] == "bearish", out


def test_aggregate_empty_is_na():
    out = aggregate_readthroughs([])
    assert out["direction"] == "na" and out["n"] == 0


def test_aggregate_mixed_cancels_to_neutral():
    items = [
        {"ticker": "A", "direction": "bullish", "magnitude": 0.5, "relevance": 1.0},
        {"ticker": "B", "direction": "bearish", "magnitude": 0.5, "relevance": 1.0},
    ]
    out = aggregate_readthroughs(items)
    assert out["direction"] == "neutral"


# ---------------------------------------------------------------------------
# No-hardcoding audit
# ---------------------------------------------------------------------------

def test_no_hardcoded_tickers_in_peer_logic():
    import re
    import tools.preearnings.peer_logic as mod
    src = open(mod.__file__, encoding="utf-8").read()
    # No embedded company tickers / peer lists. Allow generic words; flag
    # well-known tickers specifically.
    banned = ["NVDA", "AAPL", "MSFT", "TSMC", "AMD", "GOOGL", "META", "ORCL", "JPM"]
    present = [b for b in banned if re.search(rf"\b{b}\b", src)]
    assert not present, f"hardcoded tickers found in peer_logic: {present}"
