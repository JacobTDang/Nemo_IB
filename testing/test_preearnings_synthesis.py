"""Phase E unit tests — direction x asymmetry synthesis (no network)."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.preearnings.synthesis import (
    coverage,
    direction_score,
    agreement,
    predict_from_score,
    base_confidence,
    asymmetry_adjust,
    final_verdict,
    DEFAULT_DIRECTION_WEIGHTS,
)


def _sig(name, direction, magnitude=1.0):
    return {"name": name, "direction": direction, "magnitude": magnitude}


# ---------------------------------------------------------------------------
# coverage — na excluded, data_gap kept in denominator
# ---------------------------------------------------------------------------

def test_coverage_full_when_all_have_data():
    sigs = [_sig(n, "neutral") for n in DEFAULT_DIRECTION_WEIGHTS]
    assert coverage(sigs) == 1.0


def test_coverage_na_redistributed_not_penalized():
    # supplier_mops na (software) -> excluded from denominator
    sigs = [
        _sig("guidance", "bullish"), _sig("peer_readthrough", "bullish"),
        _sig("kpi_vs_consensus", "neutral"), _sig("revision_velocity", "neutral"),
        _sig("supplier_mops", "na"), _sig("thin_altdata", "neutral"),
    ]
    # applicable = all but supplier_mops (0.90); all have data -> coverage 1.0
    assert coverage(sigs) == 1.0


def test_coverage_data_gap_lowers_coverage():
    sigs = [
        _sig("guidance", "bullish"),            # 0.24 data
        {"name": "peer_readthrough", "direction": "data_gap"},   # 0.22 missing
        {"name": "kpi_vs_consensus", "direction": "data_gap"},   # 0.20 missing
        _sig("revision_velocity", "neutral"),   # 0.14 data
        _sig("supplier_mops", "na"),            # excluded
        {"name": "thin_altdata", "direction": "data_gap"},       # 0.10 missing
    ]
    # applicable = 0.24+0.22+0.20+0.14+0.10 = 0.90 ; have = 0.24+0.14 = 0.38
    assert coverage(sigs) == round(0.38 / 0.90, 3)


# ---------------------------------------------------------------------------
# direction_score
# ---------------------------------------------------------------------------

def test_direction_score_all_bullish():
    sigs = [_sig(n, "bullish") for n in DEFAULT_DIRECTION_WEIGHTS]
    assert direction_score(sigs) == 1.0


def test_direction_score_renormalizes_over_data():
    # only guidance has data (bullish) -> score should be +1 (renormalized)
    sigs = [
        _sig("guidance", "bullish"),
        {"name": "peer_readthrough", "direction": "data_gap"},
        {"name": "kpi_vs_consensus", "direction": "na"},
    ]
    assert direction_score(sigs) == 1.0


def test_direction_score_mixed():
    sigs = [_sig("guidance", "bullish", 1.0), _sig("peer_readthrough", "bearish", 1.0)]
    # (0.24*1 - 0.22*1) / (0.24+0.22)
    assert direction_score(sigs) == round((0.24 - 0.22) / 0.46, 4)


# ---------------------------------------------------------------------------
# agreement / prediction / confidence
# ---------------------------------------------------------------------------

def test_agreement_full_and_split():
    full = [_sig("guidance", "bullish"), _sig("peer_readthrough", "bullish")]
    assert agreement(full) == 1.0
    split = [_sig("guidance", "bullish"), _sig("peer_readthrough", "bearish"),
             _sig("kpi_vs_consensus", "bullish")]
    assert 0.0 < agreement(split) < 1.0


def test_predict_thresholds():
    assert predict_from_score(0.3) == "likely_beat"
    assert predict_from_score(-0.3) == "likely_miss"
    assert predict_from_score(0.1) == "in_line"


def test_low_coverage_pulls_confidence_down():
    high = base_confidence(0.5, cov=1.0, agree=1.0)
    low = base_confidence(0.5, cov=0.4, agree=1.0)
    assert low < high


# ---------------------------------------------------------------------------
# asymmetry adjustment
# ---------------------------------------------------------------------------

def test_asymmetry_crowded_long_trims_beat():
    base = asymmetry_adjust("likely_beat", 0.70)
    crowded = asymmetry_adjust("likely_beat", 0.70, positioning="crowded_long")
    assert crowded["confidence"] < base["confidence"]
    assert any("priced in" in n for n in crowded["notes"])


def test_asymmetry_high_implied_move_forces_no_position():
    out = asymmetry_adjust("likely_beat", 0.80, implied_move_pct=0.25)
    assert out["sizing"] == "no_position"


def test_asymmetry_low_confidence_no_position():
    out = asymmetry_adjust("likely_beat", 0.50)
    assert out["sizing"] == "no_position"


def test_asymmetry_inline_never_sizes():
    out = asymmetry_adjust("in_line", 0.80)
    assert out["sizing"] == "no_position"


# ---------------------------------------------------------------------------
# final_verdict — end to end
# ---------------------------------------------------------------------------

def test_final_verdict_software_keeps_coverage_without_mops():
    """The ORCL fix: software (no MOPS) still gets real coverage via guidance,
    peer, KPI, revision."""
    sigs = [
        _sig("guidance", "bullish", 0.7),
        _sig("peer_readthrough", "bullish", 0.6),
        _sig("kpi_vs_consensus", "bullish", 0.5),
        _sig("revision_velocity", "neutral"),
        _sig("supplier_mops", "na"),
        _sig("thin_altdata", "bullish", 0.3),
    ]
    out = final_verdict(sigs)
    assert out["coverage"] >= 0.60, out
    assert out["prediction"] == "likely_beat"
    assert out["low_confidence"] is False


def test_final_verdict_thin_coverage_is_low_confidence():
    sigs = [
        _sig("guidance", "bullish", 0.5),
        {"name": "peer_readthrough", "direction": "data_gap"},
        {"name": "kpi_vs_consensus", "direction": "data_gap"},
        {"name": "revision_velocity", "direction": "data_gap"},
        _sig("supplier_mops", "na"),
        {"name": "thin_altdata", "direction": "data_gap"},
    ]
    out = final_verdict(sigs)
    assert out["low_confidence"] is True
    assert out["sizing"] == "no_position"


def test_agreement_zero_magnitude_same_direction_is_full():
    sigs = [_sig("guidance", "bullish", 0.0), _sig("peer_readthrough", "bullish", 0.0)]
    assert agreement(sigs) == 1.0


def test_agreement_weighted_by_magnitude():
    """Two near-zero dissenters must not trim like a real disagreement."""
    sigs = [_sig("guidance", "bullish", 0.9),
            _sig("peer_readthrough", "bearish", 0.05),
            _sig("kpi_vs_consensus", "bearish", 0.05)]
    assert agreement(sigs) >= 0.8


def test_agreement_ignores_zero_weight_signals():
    """An unregistered signal name can't move the score — it must not be able
    to silently change sizing via agreement either."""
    sigs = [_sig("guidance", "bullish", 0.9), _sig("made_up_signal", "bearish", 0.9)]
    assert agreement(sigs) == 1.0
    plain = final_verdict([_sig("guidance", "bullish", 0.9)])
    with_stray = final_verdict(sigs)
    assert with_stray["confidence"] == plain["confidence"]


def test_direction_score_none_magnitude_is_data_gap():
    """A directional signal with unreported strength must not count as neutral."""
    sigs = [{"name": "guidance", "direction": "bullish", "magnitude": None},
            _sig("peer_readthrough", "bearish", 0.5)]
    # guidance excluded entirely -> score is pure bearish renormalized
    assert direction_score(sigs) == -0.5


def test_coverage_all_na_is_zero():
    sigs = [_sig(n, "na") for n in DEFAULT_DIRECTION_WEIGHTS]
    assert coverage(sigs) == 0.0


def test_predict_boundary_exact_threshold_is_inline():
    assert predict_from_score(0.25) == "in_line"
    assert predict_from_score(-0.25) == "in_line"


def test_base_confidence_max_is_080():
    assert base_confidence(1.0, 1.0, 1.0) == 0.80


def test_final_verdict_crowded_long_beat_trims_size():
    sigs = [_sig("guidance", "bullish", 0.9), _sig("peer_readthrough", "bullish", 0.9),
            _sig("kpi_vs_consensus", "bullish", 0.9)]
    plain = final_verdict(sigs)
    crowded = final_verdict(sigs, positioning="crowded_long")
    assert crowded["confidence"] <= plain["confidence"]
