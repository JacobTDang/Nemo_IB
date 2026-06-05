"""Phase G unit tests — Tier 2 web-traffic signal (SimilarWeb).

The signal math is pure/unit-tested. The fetch is key-gated: without
SIMILARWEB_API_KEY the tool returns a clean error (verified), and live fetch is
only exercised when a key is configured.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.preearnings.web_traffic import (
    compute_traffic_signal,
    web_traffic_signal,
)


# ---------------------------------------------------------------------------
# compute_traffic_signal — pure
# ---------------------------------------------------------------------------

def test_traffic_signal_bullish():
    out = compute_traffic_signal(120.0, 100.0)
    assert out["signal"] == "bullish" and out["yoy_pct"] == 20.0


def test_traffic_signal_bearish():
    out = compute_traffic_signal(80.0, 100.0)
    assert out["signal"] == "bearish" and out["yoy_pct"] == -20.0


def test_traffic_signal_neutral():
    out = compute_traffic_signal(105.0, 100.0)
    assert out["signal"] == "neutral"


def test_traffic_signal_data_gap_on_missing():
    assert compute_traffic_signal(None, 100.0)["signal"] == "data_gap"
    assert compute_traffic_signal(100.0, 0)["signal"] == "data_gap"
    assert compute_traffic_signal(100.0, None)["signal"] == "data_gap"


# ---------------------------------------------------------------------------
# Key gating — clean error without the paid key
# ---------------------------------------------------------------------------

def test_web_traffic_clean_error_without_key(monkeypatch):
    monkeypatch.delenv("SIMILARWEB_API_KEY", raising=False)
    out = web_traffic_signal("NVDA")
    assert "error" in out
    assert out.get("tier") == 2
    assert "SIMILARWEB_API_KEY" in out["error"]
    assert "Traceback" not in out["error"]


# ---------------------------------------------------------------------------
# Live (only when a key is present)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not os.getenv("SIMILARWEB_API_KEY"),
                    reason="SIMILARWEB_API_KEY not set (tier-2 paid data)")
def test_web_traffic_live():
    out = web_traffic_signal("AMZN")
    assert "error" not in out, out.get("error")
    assert out["source"] == "similarweb"
    assert out["signal"] in {"bullish", "bearish", "neutral", "data_gap"}


# ---------------------------------------------------------------------------
# No-hardcoding audit
# ---------------------------------------------------------------------------

def test_no_hardcoded_domains():
    import re
    import tools.preearnings.web_traffic as mod
    src = open(mod.__file__, encoding="utf-8").read()
    # No embedded company domains/tickers in logic (docstring examples aside,
    # assert no real company ticker symbols appear).
    banned = ["NVDA", "AAPL", "AMZN", "MSFT", "TSLA"]
    present = [b for b in banned if re.search(rf"\b{b}\b", src)]
    assert not present, f"hardcoded tickers found: {present}"
