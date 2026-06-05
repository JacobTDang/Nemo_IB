"""Deterministic expectations logic (Phase C): guidance archaeology + dynamic
KPI identification.

The orchestration (reading transcripts via get_earnings_transcripts, extracting
guided/actual pairs, counting analyst Q&A mentions, spawning a KPI sub-agent per
metric) lives in the pre-earnings skill. This module is the pure core.

No company/ticker/KPI is hardcoded — the KPIs that matter are derived from the
company's own segment materiality and the metrics analysts actually ask about.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Guidance archaeology
# ---------------------------------------------------------------------------

def _position(actual: float, low: float, high: float) -> str:
    if actual > high:
        return "above"
    if actual < low:
        return "below"
    return "within"


def classify_guide_style(pairs: List[Dict[str, Any]]) -> str:
    """Classify a company's guidance style from its own history.

    pairs: [{guided_low, guided_high, actual}] over recent quarters.
    - sandbag    : actual lands above the guided high most quarters (lowballs)
    - aggressive : actual lands below the guided low most quarters (overpromises)
    - inline     : mostly within the guide
    - unknown    : no usable pairs
    """
    valid = []
    for p in pairs:
        lo, hi, act = p.get("guided_low"), p.get("guided_high"), p.get("actual")
        if lo is None or hi is None or act is None:
            continue
        lo, hi = (lo, hi) if lo <= hi else (hi, lo)
        valid.append(_position(float(act), float(lo), float(hi)))
    if not valid:
        return "unknown"
    n = len(valid)
    above = valid.count("above")
    below = valid.count("below")
    if above / n >= 0.5 and above > below:
        return "sandbag"
    if below / n >= 0.5 and below > above:
        return "aggressive"
    return "inline"


def bar_position(consensus: float, guided_low: float, guided_high: float) -> str:
    """Where consensus sits vs the company's guide.
    easy   : consensus below the guide (low bar to clear)
    hard   : consensus above the guide (consensus already ahead of guidance)
    normal : consensus inside the guide
    """
    lo, hi = (guided_low, guided_high) if guided_low <= guided_high else (guided_high, guided_low)
    if consensus < lo:
        return "easy"
    if consensus > hi:
        return "hard"
    return "normal"


def guidance_direction(guide_style: str, bar: str) -> str:
    """Combine guide style + bar position into a direction lean for the surprise.

    A sandbagger facing an easy/normal bar tends to beat; an aggressive guider
    facing a hard bar tends to miss.
    """
    score = 0
    score += {"sandbag": 1, "inline": 0, "aggressive": -1, "unknown": 0}.get(guide_style, 0)
    score += {"easy": 1, "normal": 0, "hard": -1}.get(bar, 0)
    if score >= 1:
        return "bullish"
    if score <= -1:
        return "bearish"
    return "neutral"


# ---------------------------------------------------------------------------
# Dynamic KPI identification — the metrics that actually move the stock
# ---------------------------------------------------------------------------

def _normalize(values: Dict[str, float]) -> Dict[str, float]:
    if not values:
        return {}
    mx = max(values.values())
    if mx <= 0:
        return {k: 0.0 for k in values}
    return {k: v / mx for k, v in values.items()}


def rank_kpis(
    segments: List[Dict[str, Any]],
    qa_mentions: Dict[str, int],
    top_n: int = 3,
    materiality_weight: float = 0.5,
) -> List[Dict[str, Any]]:
    """Rank the KPIs that matter, derived from the company's own data.

    segments:   [{name, revenue}] from get_segment_financials (materiality)
    qa_mentions: {metric_name: count} from transcript Q&A (analyst attention)

    Candidate KPIs = union of segment names and metrics analysts asked about.
    Score = materiality_weight * normalized_materiality
          + (1 - materiality_weight) * normalized_mentions.
    No hardcoded KPI list — candidates come entirely from the inputs.
    """
    materiality: Dict[str, float] = {}
    for s in segments or []:
        name = s.get("name")
        if not name:
            continue
        materiality[name] = max(float(s.get("revenue") or 0.0), 0.0)

    mentions: Dict[str, float] = {str(k): float(v or 0) for k, v in (qa_mentions or {}).items()}

    candidates = set(materiality) | set(mentions)
    if not candidates:
        return []

    norm_mat = _normalize({k: materiality.get(k, 0.0) for k in candidates})
    norm_men = _normalize({k: mentions.get(k, 0.0) for k in candidates})

    scored = []
    for k in candidates:
        score = (materiality_weight * norm_mat.get(k, 0.0)
                 + (1 - materiality_weight) * norm_men.get(k, 0.0))
        scored.append({"kpi": k, "score": round(score, 3),
                       "materiality": round(norm_mat.get(k, 0.0), 3),
                       "attention": round(norm_men.get(k, 0.0), 3)})
    scored.sort(key=lambda x: -x["score"])
    return scored[:top_n]


def kpi_vs_consensus(actual_or_trend: float, consensus: float,
                     tolerance: float = 0.02) -> str:
    """Direction of a single KPI vs its consensus.
    Relative comparison when consensus != 0, absolute otherwise.
    """
    a, c = float(actual_or_trend), float(consensus)
    if c != 0:
        rel = (a - c) / abs(c)
        if rel > tolerance:
            return "bullish"
        if rel < -tolerance:
            return "bearish"
        return "neutral"
    if a > 0:
        return "bullish"
    if a < 0:
        return "bearish"
    return "neutral"
