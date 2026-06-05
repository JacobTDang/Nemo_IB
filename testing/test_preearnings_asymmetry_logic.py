"""Phase D unit tests — positioning + reaction-profile primitives (no network)."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.preearnings.asymmetry_logic import (
    reaction_profile,
    classify_positioning,
    implied_vs_realized,
    pair_surprises_with_reactions,
    score_reaction,
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


def test_positioning_crowded_long_needs_two_pieces_of_evidence():
    # skew alone (1 piece) is NOT enough anymore
    one = classify_positioning(short_interest_pct_float=2, put_call_iv_skew=-0.05)
    assert one["positioning"] == "neutral"
    # skew + call-heavy volume (2 pieces) -> crowded_long
    two = classify_positioning(short_interest_pct_float=2, put_call_iv_skew=-0.05,
                               put_call_volume_ratio=0.3)
    assert two["positioning"] == "crowded_long"


def test_positioning_orcl_live_case_is_crowded_long():
    """Red-green for the gap surfaced live: SI 2.12%, P/C volume 0.31, +53% 3M,
    81.6% analyst bullish read 'neutral' under the old skew-only rule."""
    out = classify_positioning(
        short_interest_pct_float=2.12, days_to_cover=1.28, put_call_iv_skew=0.0,
        put_call_volume_ratio=0.31, momentum_3m_pct=53.22, analyst_pct_bullish=81.6,
    )
    assert out["positioning"] == "crowded_long", out


def test_positioning_momentum_alone_not_crowded():
    out = classify_positioning(short_interest_pct_float=2, momentum_3m_pct=40)
    assert out["positioning"] == "neutral"


def test_positioning_crowded_short_overrides_long_evidence():
    out = classify_positioning(short_interest_pct_float=22,
                               put_call_volume_ratio=0.3, momentum_3m_pct=50)
    assert out["positioning"] == "crowded_short"


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
# pair_surprises_with_reactions — the reaction data path
# ---------------------------------------------------------------------------

def _bars(seq):
    return [{"date": d, "close": c} for d, c in seq]


def test_pairing_matches_event_after_period_and_computes_return():
    surprises = [{"period": "2026-03-31", "surprise_pct": 5.0}]
    events = ["2026-04-10"]
    bars = _bars([("2026-04-09", 100.0), ("2026-04-10", 102.0), ("2026-04-13", 110.0)])
    out = pair_surprises_with_reactions(surprises, events, bars)
    assert len(out) == 1
    # report bar = 04-10 close 102; next bar 04-13 close 110 -> +7.84%
    assert out[0]["next_day_return"] == 7.84
    assert out[0]["event_date"] == "2026-04-10"


def test_pairing_event_on_weekend_uses_prior_trading_day():
    surprises = [{"period": "2026-03-31", "surprise_pct": -2.0}]
    events = ["2026-04-11"]   # a Saturday — no bar that day
    bars = _bars([("2026-04-09", 100.0), ("2026-04-10", 105.0), ("2026-04-13", 99.75)])
    out = pair_surprises_with_reactions(surprises, events, bars)
    assert len(out) == 1
    # nearest prior bar = 04-10 (105); next = 04-13 (99.75) -> -5.0%
    assert out[0]["next_day_return"] == -5.0


def test_pairing_skips_surprise_with_no_event_in_window():
    surprises = [{"period": "2026-03-31", "surprise_pct": 3.0}]
    events = ["2026-09-01"]   # far outside max_gap_days
    bars = _bars([("2026-04-01", 100.0), ("2026-04-02", 101.0)])
    assert pair_surprises_with_reactions(surprises, events, bars) == []


def test_pairing_skips_event_at_last_bar():
    surprises = [{"period": "2026-03-31", "surprise_pct": 3.0}]
    events = ["2026-04-02"]
    bars = _bars([("2026-04-01", 100.0), ("2026-04-02", 101.0)])  # no t+1 bar
    assert pair_surprises_with_reactions(surprises, events, bars) == []


def test_pairing_fiscal_offset_event_before_period_label():
    """Red-green from the live ADBE run: fiscal-offset companies report BEFORE
    the vendor's calendar-quarter period label (8-K 2026-03-12 for period
    2026-03-31). The pairing must pick the nearest event, not skip to a later
    non-earnings 8-K."""
    surprises = [{"period": "2026-03-31", "surprise_pct": 1.17}]
    events = ["2026-03-12", "2026-04-21"]   # earnings 8-K, then unrelated 8-K
    bars = _bars([("2026-03-12", 100.0), ("2026-03-13", 95.0),
                  ("2026-04-21", 110.0), ("2026-04-22", 111.0)])
    out = pair_surprises_with_reactions(surprises, events, bars)
    assert len(out) == 1
    assert out[0]["event_date"] == "2026-03-12"     # nearest to period end
    assert out[0]["next_day_return"] == -5.0


def test_pairing_event_too_far_before_period_excluded():
    """An event older than max_back_days (e.g. the PRIOR quarter's report)
    must not be matched."""
    surprises = [{"period": "2026-03-31", "surprise_pct": 2.0}]
    events = ["2026-01-30"]    # 60 days before the label — prior quarter
    bars = _bars([("2026-01-30", 100.0), ("2026-01-31", 101.0)])
    assert pair_surprises_with_reactions(surprises, events, bars) == []


def test_pairing_two_quarters_use_distinct_events():
    surprises = [
        {"period": "2025-12-31", "surprise_pct": 35.0},
        {"period": "2026-03-31", "surprise_pct": 3.0},
    ]
    events = ["2026-01-10", "2026-04-10"]
    bars = _bars([
        ("2026-01-10", 100.0), ("2026-01-12", 114.0),
        ("2026-04-10", 200.0), ("2026-04-13", 196.0),
    ])
    out = pair_surprises_with_reactions(surprises, events, bars)
    assert len(out) == 2
    assert out[0]["next_day_return"] == 14.0    # Dec quarter -> Jan report
    assert out[1]["next_day_return"] == -2.0    # Mar quarter -> Apr report
    # feeds straight into reaction_profile
    prof = reaction_profile([{k: o[k] for k in ("surprise_pct", "next_day_return")} for o in out])
    assert prof["n"] == 2


def test_pairing_earlier_quarter_does_not_steal_later_quarters_only_event():
    """Global assignment: Q1's 8-K is missing from a truncated filing history;
    Q2 (fiscal offset) reported 2026-06-10. The old per-quarter greedy let Q1
    claim 06-10 (inside its +75d window) and drop Q2 entirely."""
    surprises = [
        {"period": "2026-03-31", "surprise_pct": 2.0},   # its event is missing
        {"period": "2026-06-30", "surprise_pct": 5.0},   # reported 06-10
    ]
    events = ["2026-06-10"]
    bars = _bars([("2026-06-10", 100.0), ("2026-06-11", 108.0)])
    out = pair_surprises_with_reactions(surprises, events, bars)
    assert len(out) == 1
    assert out[0]["period"] == "2026-06-30"     # nearest claim wins globally
    assert out[0]["surprise_pct"] == 5.0


def test_pairing_skips_nan_close_bars():
    surprises = [{"period": "2026-03-31", "surprise_pct": 1.0}]
    events = ["2026-04-10"]
    bars = _bars([("2026-04-09", 100.0), ("2026-04-10", float("nan")),
                  ("2026-04-13", 105.0), ("2026-04-14", 110.5)])
    out = pair_surprises_with_reactions(surprises, events, bars)
    # NaN bar dropped: report bar falls back to 04-09 (nearest prior valid),
    # next valid bar 04-13 -> +5.0%
    assert len(out) == 1
    assert out[0]["next_day_return"] == 5.0


# ---------------------------------------------------------------------------
# score_reaction unit guard + reaction precedence + positioning SI requirement
# ---------------------------------------------------------------------------

def test_score_reaction_rejects_percent_shaped_implied():
    import pytest
    with pytest.raises(ValueError):
        score_reaction("crowded_long", "beat", 2.0, 12.8, "likely_beat")


def test_implied_vs_realized_detects_mixed_units():
    out = implied_vs_realized(0.128, [5.2, 6.1, 4.8])   # fraction vs percents
    assert out["verdict"] == "unknown"
    assert "unit mismatch" in out.get("note", "")


def test_reaction_beats_fade_takes_precedence_over_clean_repricer():
    """All beats fade while misses track down: for a likely_beat setup,
    'your beats get sold' is the dangerous property — it must win."""
    events = [
        {"surprise_pct": 4.0, "next_day_return": -2.0},
        {"surprise_pct": 3.0, "next_day_return": -1.0},
        {"surprise_pct": -2.0, "next_day_return": -4.0},
        {"surprise_pct": -3.0, "next_day_return": -5.0},
        {"surprise_pct": -1.0, "next_day_return": -2.0},
    ]
    out = reaction_profile(events)
    assert out["beat_fade_rate"] == 1.0
    assert out["pattern"] == "beats_fade"


def test_reaction_single_fading_beat_is_not_a_pattern():
    events = [{"surprise_pct": 4.0, "next_day_return": -2.0}]
    out = reaction_profile(events)
    assert out["pattern"] != "beats_fade"


def test_positioning_si_unknown_never_crowded_long():
    """With short interest unknown, a heavily-shorted name could be classified
    exactly backwards — crowded_long requires KNOWN low SI."""
    out = classify_positioning(short_interest_pct_float=None,
                               put_call_volume_ratio=0.3, momentum_3m_pct=50)
    assert out["positioning"] == "neutral"


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
