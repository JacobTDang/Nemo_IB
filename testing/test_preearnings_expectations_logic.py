"""Phase C unit tests — guidance archaeology + dynamic KPI logic (no network)."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.preearnings.expectations_logic import (
    classify_guide_style,
    bar_position,
    guidance_direction,
    rank_kpis,
    kpi_vs_consensus,
    select_scoring_bar,
)


# ---------------------------------------------------------------------------
# Guidance style
# ---------------------------------------------------------------------------

def test_guide_style_sandbag():
    pairs = [
        {"guided_low": 1.0, "guided_high": 1.1, "actual": 1.3},
        {"guided_low": 1.2, "guided_high": 1.3, "actual": 1.5},
        {"guided_low": 1.4, "guided_high": 1.5, "actual": 1.6},
    ]
    assert classify_guide_style(pairs) == "sandbag"


def test_guide_style_aggressive():
    pairs = [
        {"guided_low": 2.0, "guided_high": 2.2, "actual": 1.7},
        {"guided_low": 2.1, "guided_high": 2.3, "actual": 1.9},
    ]
    assert classify_guide_style(pairs) == "aggressive"


def test_guide_style_inline():
    pairs = [
        {"guided_low": 1.0, "guided_high": 1.2, "actual": 1.1},
        {"guided_low": 1.1, "guided_high": 1.3, "actual": 1.2},
    ]
    assert classify_guide_style(pairs) == "inline"


def test_guide_style_unknown_when_no_pairs():
    assert classify_guide_style([]) == "unknown"
    assert classify_guide_style([{"guided_low": None, "guided_high": 1, "actual": 1}]) == "unknown"


def test_guide_style_handles_reversed_bounds():
    pairs = [{"guided_low": 1.3, "guided_high": 1.0, "actual": 1.5}]  # low/high swapped
    assert classify_guide_style(pairs) == "sandbag"


# ---------------------------------------------------------------------------
# Bar position + combined direction
# ---------------------------------------------------------------------------

def test_bar_position():
    assert bar_position(0.9, 1.0, 1.2) == "easy"
    assert bar_position(1.1, 1.0, 1.2) == "normal"
    assert bar_position(1.3, 1.0, 1.2) == "hard"


def test_guidance_direction_combinations():
    assert guidance_direction("sandbag", "easy") == "bullish"
    assert guidance_direction("aggressive", "hard") == "bearish"
    assert guidance_direction("inline", "normal") == "neutral"
    # sandbag but consensus already hard -> cancels to neutral
    assert guidance_direction("sandbag", "hard") == "neutral"


# ---------------------------------------------------------------------------
# Dynamic KPI ranking — derived, not hardcoded
# ---------------------------------------------------------------------------

def test_rank_kpis_combines_materiality_and_attention():
    segments = [
        {"name": "Cloud", "revenue": 100.0},
        {"name": "Licensing", "revenue": 40.0},
        {"name": "Hardware", "revenue": 10.0},
    ]
    qa = {"Cloud": 12, "RPO": 8, "Hardware": 1}
    out = rank_kpis(segments, qa, top_n=3)
    kpis = [o["kpi"] for o in out]
    assert kpis[0] == "Cloud"          # high materiality + high attention
    assert "RPO" in kpis               # surfaces from Q&A even with 0 materiality


def test_rank_kpis_software_vs_consumer_differ():
    # Proves KPIs are derived from inputs, not a fixed map.
    sw = rank_kpis([{"name": "Cloud", "revenue": 100}], {"RPO": 10, "Cloud": 9}, top_n=2)
    consumer = rank_kpis([{"name": "Footwear", "revenue": 100}],
                         {"units": 10, "same store sales": 9}, top_n=2)
    assert {o["kpi"] for o in sw} != {o["kpi"] for o in consumer}


def test_rank_kpis_empty_inputs():
    assert rank_kpis([], {}) == []


# ---------------------------------------------------------------------------
# KPI vs consensus
# ---------------------------------------------------------------------------

def test_kpi_vs_consensus_relative():
    assert kpi_vs_consensus(0.22, 0.18) == "bullish"      # growth above consensus
    assert kpi_vs_consensus(0.15, 0.18) == "bearish"
    assert kpi_vs_consensus(0.180, 0.180) == "neutral"


def test_kpi_vs_consensus_zero_consensus():
    assert kpi_vs_consensus(5, 0) == "bullish"
    assert kpi_vs_consensus(-3, 0) == "bearish"
    assert kpi_vs_consensus(0, 0) == "neutral"


def test_rank_kpis_tied_scores_deterministic():
    """Tied scores must break alphabetically — set-iteration order varies per
    process, which previously selected different KPIs (and spawned different
    sub-agents) across runs."""
    out = rank_kpis([{"name": n, "revenue": 0} for n in ("Gamma", "Alpha", "Beta")],
                    {}, top_n=2)
    assert [o["kpi"] for o in out] == ["Alpha", "Beta"]


def test_guide_style_skips_unparseable_pair():
    pairs = [{"guided_low": "n/a", "guided_high": 1.2, "actual": 1.1}]
    assert classify_guide_style(pairs) == "unknown"


def test_bar_position_exact_bounds_are_normal():
    assert bar_position(1.0, 1.0, 1.2) == "normal"
    assert bar_position(1.2, 1.0, 1.2) == "normal"


def test_guide_style_tie_above_below_is_inline():
    """Documents the tie semantics: 1 above + 1 below = erratic guiding, which
    the classifier reports as inline (no systematic bias either way)."""
    pairs = [
        {"guided_low": 1.0, "guided_high": 1.1, "actual": 1.3},   # above
        {"guided_low": 1.2, "guided_high": 1.3, "actual": 1.0},   # below
    ]
    assert classify_guide_style(pairs) == "inline"


# ---------------------------------------------------------------------------
# select_scoring_bar — the GAAP/adjusted basis trap (live CHWY red-green)
# ---------------------------------------------------------------------------

def test_scoring_bar_chwy_live_case_flags_alt_basis():
    """The live trap: Finnhub 0.2548 vs yfinance 0.42617 (67% apart). The
    year-ago actual from the scoring source's own history (0.16) shows the
    scoring bar implies sane 1.6x growth while the alt implies 2.7x — the alt
    is on a different (adjusted) basis. Bar stays the scoring source."""
    out = select_scoring_bar(0.2548, 0.42617, 0.16)
    assert out["bar"] == 0.2548
    assert out["basis_flag"] == "divergent_alt_basis_suspect"
    assert out["divergence_pct"] > 60


def test_scoring_bar_agreeing_sources_ok():
    """RH live case: -2.1829 vs -2.04601 (6.7% apart) -> ok, no basis check."""
    out = select_scoring_bar(-2.1829, -2.04601, 0.13)
    assert out["basis_flag"] == "ok"
    assert out["bar"] == -2.1829


def test_scoring_bar_scoring_source_suspect():
    """When the SCORING series itself looks basis-shifted vs its own year-ago
    actual, the bar stays (eval grades against it) but the risk is flagged."""
    out = select_scoring_bar(0.42617, 0.2548, 0.16)
    assert out["bar"] == 0.42617              # policy: always the scoring source
    assert out["basis_flag"] == "divergent_scoring_basis_suspect"


def test_scoring_bar_sign_flip_unverifiable():
    out = select_scoring_bar(-2.18, 1.5, 0.13)
    assert out["basis_flag"] == "divergent_unverifiable"


def test_scoring_bar_single_and_missing_sources():
    assert select_scoring_bar(None, 1.5, 1.0)["basis_flag"] == "single_source"
    assert select_scoring_bar(None, 1.5, 1.0)["bar"] == 1.5
    assert select_scoring_bar(None, None, 1.0)["basis_flag"] == "no_consensus"


def test_scoring_bar_both_implausible():
    out = select_scoring_bar(1.0, 2.0, 0.1)   # 10x and 20x year-ago
    assert out["basis_flag"] == "divergent_both_implausible"


# ---------------------------------------------------------------------------
# No-hardcoding audit
# ---------------------------------------------------------------------------

def test_no_hardcoded_tickers_or_kpis():
    import re
    import tools.preearnings.expectations_logic as mod
    src = open(mod.__file__, encoding="utf-8").read()
    banned = ["NVDA", "AAPL", "MSFT", "ORCL", "CRM", "iPhone", "Azure", "OCI"]
    present = [b for b in banned if re.search(rf"\b{re.escape(b)}\b", src)]
    assert not present, f"hardcoded company/KPI tokens found: {present}"
