"""Phase D unit tests — positioning + reaction-profile primitives (no network)."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.preearnings.asymmetry_logic import (
    reaction_profile,
    classify_positioning,
    implied_vs_realized,
)


# ---------------------------------------------------------------------------
# reaction_profile
# ---------------------------------------------------------------------------

def test_reaction_clean_repricer():
    events = [
        {"surprise_pct": 5, "next_day_return": 4},
        {"surprise_pct": -3, "next_day_return": -5},
        {"surprise_pct": 2, "next_day_return": 3},
        {"surprise_pct": -1, "next_day_return": -2},
    ]
    out = reaction_profile(events)
    assert out["pattern"] == "clean_repricer"
    assert out["consistency"] >= 0.6
    assert out["n"] == 4


def test_reaction_beats_fade():
    events = [
        {"surprise_pct": 5, "next_day_return": -2},
        {"surprise_pct": 4, "next_day_return": -1},
        {"surprise_pct": 3, "next_day_return": 0},
        {"surprise_pct": 6, "next_day_return": 1},
    ]
    out = reaction_profile(events)
    assert out["pattern"] == "beats_fade"
    assert out["beat_fade_rate"] >= 0.5


def test_reaction_unknown_when_empty():
    out = reaction_profile([])
    assert out["pattern"] == "unknown" and out["n"] == 0


def test_reaction_avg_abs_move():
    events = [
        {"surprise_pct": 1, "next_day_return": 4},
        {"surprise_pct": -1, "next_day_return": -6},
    ]
    out = reaction_profile(events)
    assert out["avg_abs_move"] == 5.0


# ---------------------------------------------------------------------------
# classify_positioning
# ---------------------------------------------------------------------------

def test_positioning_crowded_short_high():
    out = classify_positioning(short_interest_pct_float=25, days_to_cover=7)
    assert out["positioning"] == "crowded_short"
    assert out["squeeze_risk"] == "high"


def test_positioning_crowded_short_elevated():
    out = classify_positioning(short_interest_pct_float=12, days_to_cover=2)
    assert out["positioning"] == "crowded_short"
    assert out["squeeze_risk"] == "elevated"


def test_positioning_crowded_long():
    out = classify_positioning(short_interest_pct_float=2, put_call_iv_skew=-0.05)
    assert out["positioning"] == "crowded_long"


def test_positioning_neutral_default():
    out = classify_positioning(short_interest_pct_float=5)
    assert out["positioning"] == "neutral"
    assert out["squeeze_risk"] == "low"


def test_positioning_handles_missing_inputs():
    out = classify_positioning()
    assert out["positioning"] == "neutral"


# ---------------------------------------------------------------------------
# implied_vs_realized
# ---------------------------------------------------------------------------

def test_implied_rich():
    out = implied_vs_realized(0.10, [0.05, 0.06, 0.05])
    assert out["verdict"] == "rich"


def test_implied_cheap():
    out = implied_vs_realized(0.04, [0.08, 0.09, 0.07])
    assert out["verdict"] == "cheap"


def test_implied_fair():
    out = implied_vs_realized(0.06, [0.055, 0.06, 0.065])
    assert out["verdict"] == "fair"


def test_implied_unknown_without_history():
    out = implied_vs_realized(0.06, [])
    assert out["verdict"] == "unknown"


# ---------------------------------------------------------------------------
# No-hardcoding audit
# ---------------------------------------------------------------------------

def test_no_hardcoded_tickers():
    import re
    import tools.preearnings.asymmetry_logic as mod
    src = open(mod.__file__, encoding="utf-8").read()
    banned = ["NVDA", "AAPL", "MSFT", "ORCL", "TSLA", "GME"]
    present = [b for b in banned if re.search(rf"\b{b}\b", src)]
    assert not present, f"hardcoded tickers found: {present}"
