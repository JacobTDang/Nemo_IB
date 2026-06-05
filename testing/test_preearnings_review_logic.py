"""Unit tests — /preearnings-review deterministic checks + reaction scoring.

All pure / no network. Includes red-green cases for the two bugs found live:
the stale eval-row prediction (check_db_consistency) and the Finnhub-vs-yfinance
consensus gap (check_estimate_dispersion).
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.preearnings.review_logic import (
    check_citations,
    check_freshness,
    check_completeness,
    check_contradictions,
    check_stale_flags,
    check_hard_rules,
    check_db_consistency,
    check_estimate_dispersion,
    run_review,
    run_manifest,
)
from tools.preearnings.asymmetry_logic import score_reaction


_NOW = datetime.now(timezone.utc)


def _layer(component, direction=None, magnitude=None, payload=None,
           sources=None, age_hours=0.0):
    return {
        "component": component,
        "direction": direction,
        "magnitude": magnitude,
        "payload": payload or {},
        "sources": sources if sources is not None
                   else [{"claim": "x=1", "tool": "some_tool"}],
        "created_at": (_NOW - timedelta(hours=age_hours)).isoformat(),
    }


def _full_layers(**overrides):
    layers = [
        _layer("peer_readthrough", "bullish", 0.5),
        _layer("guidance", "bullish", 0.5),
        _layer("kpi:growth", "bullish", 0.6),
        _layer("positioning", "neutral"),
        _layer("reaction", None),
        _layer("synthesis", "bullish", 0.46, payload={
            "prediction": "likely_beat", "confidence": 0.6, "coverage": 1.0,
            "sizing": "cautious", "low_confidence": False,
        }),
    ]
    return layers


def _eval_row(prediction="likely_beat", confidence=0.6, implied=0.10):
    return {"prediction": prediction, "confidence": confidence,
            "implied_move_pct": implied, "outcome": None}


# ---------------------------------------------------------------------------
# citations
# ---------------------------------------------------------------------------

def test_citations_pass_when_all_cited():
    out = check_citations(_full_layers())
    assert all(c["status"] == "pass" for c in out)


def test_citations_fail_on_uncited_direction_critical():
    layers = [_layer("peer_readthrough", "bullish", 0.5, sources=[])]
    out = check_citations(layers)
    assert any(c["status"] == "fail" for c in out)


def test_citations_warn_on_uncited_noncritical():
    layers = [_layer("positioning", "neutral", sources=[])]
    out = check_citations(layers)
    assert any(c["status"] == "warn" for c in out)
    assert not any(c["status"] == "fail" for c in out)


def test_citations_warn_on_malformed_entries():
    layers = [_layer("guidance", "bullish", 0.5,
                     sources=[{"claim": "ok", "tool": "t"}, {"claim": "", "tool": ""}])]
    out = check_citations(layers)
    assert any(c["status"] == "warn" for c in out)


# ---------------------------------------------------------------------------
# freshness
# ---------------------------------------------------------------------------

def test_freshness_pass_fresh():
    out = check_freshness(_full_layers(), now=_NOW)
    assert all(c["status"] == "pass" for c in out)


def test_freshness_warn_when_old():
    out = check_freshness([_layer("positioning", age_hours=30)], now=_NOW)
    assert any(c["status"] == "warn" for c in out)


def test_freshness_fail_when_very_old():
    out = check_freshness([_layer("positioning", age_hours=100)], now=_NOW)
    assert any(c["status"] == "fail" for c in out)


def test_freshness_slow_components_get_longer_windows():
    """Reaction profile (years of history) and guidance archaeology (changes
    once a quarter) must not force expensive sub-agent re-runs at T-1."""
    out = check_freshness([_layer("reaction", age_hours=100),    # < 168h
                           _layer("guidance", age_hours=80)],    # < 96h
                          now=_NOW)
    assert all(c["status"] == "pass" for c in out), out


def test_freshness_slow_components_still_expire():
    out = check_freshness([_layer("reaction", age_hours=600)], now=_NOW)  # > 3*168
    assert any(c["status"] == "fail" for c in out)


# ---------------------------------------------------------------------------
# completeness
# ---------------------------------------------------------------------------

def test_completeness_pass_full():
    out = check_completeness(_full_layers(), _eval_row())
    assert all(c["status"] == "pass" for c in out)


def test_completeness_fail_without_synthesis():
    layers = [l for l in _full_layers() if l["component"] != "synthesis"]
    out = check_completeness(layers, _eval_row())
    assert any(c["status"] == "fail" for c in out)


def test_completeness_fail_without_eval_row():
    out = check_completeness(_full_layers(), None)
    assert any(c["status"] == "fail" for c in out)


def test_completeness_warn_missing_reaction():
    layers = [l for l in _full_layers() if l["component"] != "reaction"]
    out = check_completeness(layers, _eval_row())
    assert any(c["status"] == "warn" and "reaction" in c["detail"] for c in out)


# ---------------------------------------------------------------------------
# contradictions / stale flags
# ---------------------------------------------------------------------------

def test_contradictions_warn_on_high_mag_disagreement():
    layers = [_layer("peer_readthrough", "bullish", 0.7),
              _layer("guidance", "bearish", 0.6)]
    out = check_contradictions(layers)
    assert out[0]["status"] == "warn"


def test_contradictions_pass_when_low_magnitude():
    layers = [_layer("peer_readthrough", "bullish", 0.7),
              _layer("guidance", "bearish", 0.2)]
    assert check_contradictions(layers)[0]["status"] == "pass"


def test_stale_flags_warn_on_quotes_stale():
    layers = [_layer("positioning", payload={"implied": {"quotes_stale": True}})]
    out = check_stale_flags(layers)
    assert any(c["status"] == "warn" for c in out)


def test_stale_flags_pass_clean():
    out = check_stale_flags([_layer("guidance", payload={"v": 1})])
    assert out[0]["status"] == "pass"


# ---------------------------------------------------------------------------
# hard rules
# ---------------------------------------------------------------------------

def test_hard_rules_fail_inline_with_size():
    out = check_hard_rules({"prediction": "in_line", "sizing": "cautious",
                            "confidence": 0.6})
    assert any(c["status"] == "fail" for c in out)


def test_hard_rules_fail_low_conf_with_size():
    out = check_hard_rules({"prediction": "likely_beat", "confidence": 0.50,
                            "sizing": "cautious"})
    assert any(c["status"] == "fail" for c in out)


def test_hard_rules_fail_high_implied_with_size():
    out = check_hard_rules({"prediction": "likely_beat", "confidence": 0.7,
                            "sizing": "normal", "implied_move_pct": 0.25})
    assert any(c["status"] == "fail" for c in out)


def test_hard_rules_fail_inconsistent_low_confidence_flag():
    out = check_hard_rules({"prediction": "likely_beat", "confidence": 0.45,
                            "coverage": 0.9, "sizing": "no_position",
                            "low_confidence": False})
    assert any(c["status"] == "fail" for c in out)


def test_hard_rules_pass_clean():
    out = check_hard_rules({"prediction": "likely_beat", "confidence": 0.6,
                            "coverage": 1.0, "sizing": "cautious",
                            "low_confidence": False, "implied_move_pct": 0.12})
    assert out[0]["status"] == "pass"


# ---------------------------------------------------------------------------
# db consistency — red-green for the live ORCL stale-row bug
# ---------------------------------------------------------------------------

def test_db_consistency_catches_stale_prediction_row():
    """The exact live bug: eval row in_line/0.42 while synthesis said
    likely_beat/0.52 — the reviewer must fail this."""
    eval_row = {"prediction": "in_line", "confidence": 0.42}
    synthesis = {"prediction": "likely_beat", "confidence": 0.52}
    out = check_db_consistency(eval_row, synthesis)
    assert out[0]["status"] == "fail"


def test_db_consistency_pass_when_matching():
    row = {"prediction": "likely_beat", "confidence": 0.52}
    syn = {"prediction": "likely_beat", "confidence": 0.52}
    assert check_db_consistency(row, syn)[0]["status"] == "pass"


# ---------------------------------------------------------------------------
# estimate dispersion — red-green for the live Finnhub/yfinance gap
# ---------------------------------------------------------------------------

def test_dispersion_warns_on_live_orcl_gap():
    """yfinance 1.96163 vs Finnhub 1.9985 = 1.9% apart -> must warn."""
    out = check_estimate_dispersion(1.96163, 1.9985, "yfinance", "finnhub")
    assert out[0]["status"] == "warn"
    assert "bar differs" in out[0]["detail"]


def test_dispersion_pass_within_tolerance():
    out = check_estimate_dispersion(2.00, 2.005, "a", "b")
    assert out[0]["status"] == "pass"


def test_dispersion_warn_when_source_missing():
    out = check_estimate_dispersion(2.0, None)
    assert out[0]["status"] == "warn"


# ---------------------------------------------------------------------------
# run_review aggregation
# ---------------------------------------------------------------------------

def test_run_review_sound_when_all_pass():
    layers = _full_layers()
    row = _eval_row(prediction="likely_beat", confidence=0.6)
    out = run_review(layers, row,
                     dispersion={"eps_a": 2.0, "eps_b": 2.001,
                                 "label_a": "a", "label_b": "b"}, now=_NOW)
    assert out["verdict"] == "sound", out


def test_run_review_not_actionable_on_db_mismatch():
    layers = _full_layers()
    row = _eval_row(prediction="in_line", confidence=0.42)
    out = run_review(layers, row, now=_NOW)
    assert out["verdict"] == "not_actionable"
    assert out["fixes"]


def test_run_review_warnings_on_stale_quotes():
    layers = _full_layers()
    layers[3] = _layer("positioning", "neutral",
                       payload={"quotes_stale": True})
    out = run_review(layers, _eval_row(), now=_NOW)
    assert out["verdict"] == "sound_with_warnings"


def test_stale_flags_pass_when_quotes_stale_false():
    """A payload explicitly recording FRESH quotes must not be flagged."""
    layers = [_layer("implied_move", payload={"quotes_stale": False})]
    out = check_stale_flags(layers)
    assert out[0]["status"] == "pass"


def test_stale_flags_ignores_marker_in_long_prose():
    layers = [_layer("guidance", payload={
        "headline": "CFO suspected of leaking the stale negotiation details to press"})]
    out = check_stale_flags(layers)
    assert out[0]["status"] == "pass"


def test_stale_flags_catches_enum_like_value():
    layers = [_layer("positioning", payload={"iv_status": "suspect_iv_sentinel"})]
    out = check_stale_flags(layers)
    assert any(c["status"] == "warn" for c in out)


def test_run_review_uses_eval_implied_when_synthesis_has_none():
    """A failed options fetch persisting implied_move_pct: None must not mask
    the eval row's real implied from the binary-event hard rule."""
    layers = _full_layers()
    layers[-1] = _layer("synthesis", "bullish", 0.46, payload={
        "prediction": "likely_beat", "confidence": 0.7, "coverage": 1.0,
        "sizing": "normal", "low_confidence": False, "implied_move_pct": None,
    })
    row = {"prediction": "likely_beat", "confidence": 0.7,
           "implied_move_pct": 0.25, "outcome": None}
    out = run_review(layers, row, now=_NOW)
    assert out["verdict"] == "not_actionable"
    assert any("implied" in f["detail"] for f in out["fails"])


# ---------------------------------------------------------------------------
# run_manifest — context-decay-proof resume state
# ---------------------------------------------------------------------------

def test_manifest_complete_run():
    layers = _full_layers() + [_layer("implied_move")]
    m = run_manifest(layers, now=_NOW)
    assert m["complete"] is True
    assert m["missing"] == []
    assert m["resumable"] is True


def test_manifest_partial_run_lists_missing():
    layers = [_layer("guidance"), _layer("positioning")]
    m = run_manifest(layers, now=_NOW)
    assert m["resumable"] is True and m["complete"] is False
    assert "peer_readthrough" in m["missing"]
    assert "kpi" in m["missing"]
    assert "synthesis" in m["missing"]


def test_manifest_kpi_wildcard_matches():
    layers = _full_layers() + [_layer("implied_move")]
    m = run_manifest(layers, now=_NOW)        # _full_layers has kpi:growth
    assert "kpi" not in m["missing"]


def test_manifest_flags_stale_components():
    layers = [_layer("positioning", age_hours=30)]
    m = run_manifest(layers, now=_NOW)
    assert "positioning" in m["stale"]


def test_manifest_empty_not_resumable():
    m = run_manifest([], now=_NOW)
    assert m["resumable"] is False and m["complete"] is False


# ---------------------------------------------------------------------------
# score_reaction — the grader's asymmetry scorer
# ---------------------------------------------------------------------------

def test_reaction_crowded_long_beat_muted_is_correct():
    # implied 12.8% -> half = 6.4; +2% move on a beat = muted reward
    out = score_reaction("crowded_long", "beat", 2.0, 0.128, "likely_beat")
    assert out["asymmetry_correct"] == 1
    assert out["price_direction_match"] == 1


def test_reaction_crowded_long_beat_full_reward_is_wrong():
    out = score_reaction("crowded_long", "beat", 10.0, 0.128, "likely_beat")
    assert out["asymmetry_correct"] == 0


def test_reaction_crowded_long_miss_punished_hard_is_correct():
    out = score_reaction("crowded_long", "miss", -8.0, 0.128, "likely_beat")
    assert out["asymmetry_correct"] == 1
    assert out["price_direction_match"] == 0   # predicted beat, price fell


def test_reaction_crowded_short_beat_squeeze_is_correct():
    out = score_reaction("crowded_short", "beat", 8.0, 0.128, "likely_miss")
    assert out["asymmetry_correct"] == 1


def test_reaction_crowded_short_miss_cushioned_is_correct():
    out = score_reaction("crowded_short", "miss", -2.0, 0.128, "likely_miss")
    assert out["asymmetry_correct"] == 1
    assert out["price_direction_match"] == 1


def test_reaction_neutral_positioning_not_scored():
    out = score_reaction("neutral", "beat", 5.0, 0.128, "likely_beat")
    assert out["asymmetry_correct"] is None
    assert out["price_direction_match"] == 1


def test_reaction_inline_prediction_no_direction_match():
    out = score_reaction("neutral", "in_line", 1.0, 0.10, "in_line")
    assert out["price_direction_match"] is None


def test_reaction_missing_data_not_scored():
    out = score_reaction("crowded_long", "beat", None, 0.128, "likely_beat")
    assert out["asymmetry_correct"] is None


# ---------------------------------------------------------------------------
# No-hardcoding audit
# ---------------------------------------------------------------------------

def test_no_hardcoded_tickers_in_review_logic():
    import re
    import tools.preearnings.review_logic as mod
    src = open(mod.__file__, encoding="utf-8").read()
    banned = ["NVDA", "AAPL", "MSFT", "ORCL", "CRM", "HPE"]
    present = [b for b in banned if re.search(rf"\b{b}\b", src)]
    assert not present, f"hardcoded tickers found: {present}"
