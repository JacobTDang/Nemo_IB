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

from datetime import date

from tools.preearnings.web_traffic import (
    compute_traffic_signal,
    web_traffic_signal,
    align_yoy_months,
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
# align_yoy_months — date-matched YoY, never index-based
# ---------------------------------------------------------------------------

def _months(seq):
    return [{"date": f"{ym}-01", "visits": v} for ym, v in seq]


def test_align_handles_vendor_lag():
    """SimilarWeb lags 1-2 months: latest complete month is 2026-04; comparator
    must be 2025-04 by date match (index-based selection made this data_gap)."""
    months = _months([(f"2025-{m:02d}", 100 + m) for m in range(2, 13)]
                     + [(f"2026-{m:02d}", 200 + m) for m in range(1, 5)])
    out = align_yoy_months(months, today=date(2026, 6, 5))
    assert out["current"]["date"][:7] == "2026-04"
    assert out["prior"]["date"][:7] == "2025-04"


def test_align_excludes_partial_current_month():
    """The in-progress calendar month must never be 'current' (false-bearish)."""
    months = _months([("2025-05", 500), ("2025-06", 510)]
                     + [(f"2026-{m:02d}", 600 + m) for m in range(1, 7)])  # incl 2026-06
    out = align_yoy_months(months, today=date(2026, 6, 5))
    assert out["current"]["date"][:7] == "2026-05"
    assert out["prior"]["date"][:7] == "2025-05"


def test_align_prior_by_date_not_index():
    """Prior must be the date-matched year-ago month, never months[0]."""
    months = _months([(f"2025-{m:02d}", 100) for m in range(4, 13)]      # 2025-04..12
                     + [(f"2026-{m:02d}", 120) for m in range(1, 6)])    # 2026-01..05
    out = align_yoy_months(months, today=date(2026, 6, 15))
    assert out["current"]["date"][:7] == "2026-05"
    assert out["prior"]["date"][:7] == "2025-05"   # date-matched, not index 0 (2025-04)


def test_align_exactly_12_buckets_missing_comparator_is_honest_gap():
    """The old index-based pick compared adjacent months here; the honest
    answer is prior=None (comparator outside the window)."""
    months = _months([(f"2025-{m:02d}", 100) for m in range(6, 13)]
                     + [(f"2026-{m:02d}", 120) for m in range(1, 6)])
    out = align_yoy_months(months, today=date(2026, 6, 15))
    assert out["current"]["date"][:7] == "2026-05"
    assert out["prior"] is None


def test_align_missing_comparator_returns_none_prior():
    months = _months([("2026-03", 100), ("2026-04", 110)])
    out = align_yoy_months(months, today=date(2026, 6, 5))
    assert out["current"]["date"][:7] == "2026-04"
    assert out["prior"] is None     # -> compute_traffic_signal gives data_gap


def test_align_empty_months():
    out = align_yoy_months([], today=date(2026, 6, 5))
    assert out["current"] is None and out["prior"] is None


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
