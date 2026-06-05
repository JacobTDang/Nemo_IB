"""Direction x asymmetry synthesis (Phase E).

DIRECTION signals answer "will they clear the bar?" and are weighted/averaged
over the signals that actually have data (N/A weight redistributed). ASYMMETRY
inputs (positioning, reaction profile, implied move) do NOT vote on direction —
they adjust confidence and sizing, because they govern how a surprise is
punished/rewarded.

Pure and unit-tested. No company-specific constants.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

# DIRECTION weights (sum to 1.0). A signal marked "na" is not applicable to the
# company (e.g. Taiwan MOPS for a bank) and its weight is redistributed by
# renormalizing over the signals that have data. Tier-2 panels, when present,
# replace the matching proxy weight at the call site.
DEFAULT_DIRECTION_WEIGHTS: Dict[str, float] = {
    "guidance":          0.24,
    "peer_readthrough":  0.22,
    "kpi_vs_consensus":  0.20,
    "revision_velocity": 0.14,
    "supplier_mops":     0.10,
    "thin_altdata":      0.10,
}

_DIR_SIGN = {"bullish": 1.0, "bearish": -1.0, "neutral": 0.0}
_HAS_DATA = set(_DIR_SIGN)            # bullish/bearish/neutral count as data
# "na" -> not applicable (excluded from denominator); "data_gap"/missing ->
# applicable but unobserved (kept in the applicable denominator).


def _weight(weights: Dict[str, float], name: str) -> float:
    return float(weights.get(name, 0.0))


def coverage(signals: List[Dict[str, Any]],
             weights: Optional[Dict[str, float]] = None) -> float:
    """Fraction of APPLICABLE weight that actually has data."""
    w = weights or DEFAULT_DIRECTION_WEIGHTS
    applicable = 0.0
    have = 0.0
    for s in signals:
        wt = _weight(w, s.get("name", ""))
        if wt <= 0:
            continue
        direction = s.get("direction")
        if direction == "na":
            continue                      # not applicable -> out of denominator
        applicable += wt                  # data_gap/missing stays applicable
        if direction in _HAS_DATA:
            have += wt
    return round(have / applicable, 3) if applicable > 0 else 0.0


def direction_score(signals: List[Dict[str, Any]],
                    weights: Optional[Dict[str, float]] = None) -> float:
    """Weighted lean in [-1, 1], renormalized over signals that have data.

    A bullish/bearish signal with magnitude=None is treated as a data gap
    (excluded from numerator AND denominator) — a stated direction with an
    unreported strength must not silently count as exactly neutral."""
    w = weights or DEFAULT_DIRECTION_WEIGHTS
    num = 0.0
    den = 0.0
    for s in signals:
        if s.get("direction") not in _HAS_DATA:
            continue
        wt = _weight(w, s.get("name", ""))
        if wt <= 0:
            continue
        mag = s.get("magnitude")
        if mag is None and s["direction"] != "neutral":
            continue  # directional but strength unknown -> data gap
        num += wt * _DIR_SIGN[s["direction"]] * float(mag or 0.0)
        den += wt
    return round(num / den, 4) if den > 0 else 0.0


def agreement(signals: List[Dict[str, Any]],
              weights: Optional[Dict[str, float]] = None) -> float:
    """Magnitude-weighted alignment of directional signals with the net lean.

    Weighted by the same magnitudes that drive the score, so two near-zero
    dissenters cannot trim confidence the way a real disagreement does. Signals
    with names outside the weight dict are ignored (mirrors direction_score),
    so a stray unregistered signal cannot silently change sizing."""
    w = weights or DEFAULT_DIRECTION_WEIGHTS
    directional = [s for s in signals
                   if s.get("direction") in ("bullish", "bearish")
                   and _weight(w, s.get("name", "")) > 0]
    if len(directional) < 2:
        return 1.0
    masses = [(_DIR_SIGN[s["direction"]], float(s.get("magnitude") or 0.0))
              for s in directional]
    total = sum(m for _, m in masses)
    if total <= 0:
        # no magnitude information: unanimous direction is full agreement
        return 1.0 if len({d for d, _ in masses}) == 1 else 0.5
    net = sum(d * m for d, m in masses)
    if net == 0:
        return 0.5
    sign = 1.0 if net > 0 else -1.0
    agree_mass = sum(m for d, m in masses if d == sign)
    return round(agree_mass / total, 3)


def predict_from_score(score: float) -> str:
    if score > 0.25:
        return "likely_beat"
    if score < -0.25:
        return "likely_miss"
    return "in_line"


def base_confidence(score: float, cov: float, agree: float) -> float:
    """Confidence from signal strength, scaled by coverage and agreement.
    Honest: thin coverage or internal disagreement pulls confidence down."""
    raw = 0.45 + 0.35 * min(abs(score), 1.0)
    if cov < 0.5:
        cov_factor = 0.70
    elif cov < 0.7:
        cov_factor = 0.85
    else:
        cov_factor = 1.0
    agree_factor = 0.7 + 0.3 * agree
    # NOTE: raw maxes at 0.80 (0.45 + 0.35), so 0.85 is headroom for a future
    # slope increase, not a binding ceiling today.
    return round(min(raw * cov_factor * agree_factor, 0.85), 3)


def _sizing(conf: float, prediction: str) -> str:
    if prediction == "in_line":
        return "no_position"
    if conf >= 0.65:
        return "normal"
    if conf >= 0.55:
        return "cautious"
    return "no_position"


def asymmetry_adjust(
    prediction: str,
    confidence: float,
    positioning: Optional[str] = None,
    squeeze_risk: Optional[str] = None,
    reaction_pattern: Optional[str] = None,
    implied_verdict: Optional[str] = None,
    implied_move_pct: Optional[float] = None,
) -> Dict[str, Any]:
    """Adjust confidence + sizing for reaction asymmetry. Returns the adjusted
    confidence, sizing, and human-readable notes."""
    conf = confidence
    notes: List[str] = []

    if positioning == "crowded_long" and prediction == "likely_beat":
        conf *= 0.85
        notes.append("beat partly priced in (crowded long) — reaction asymmetric to downside")
    if positioning == "crowded_short" and prediction == "likely_miss":
        conf *= 0.85
        notes.append("miss partly priced in (crowded short) — squeeze risk on any beat")
    if squeeze_risk == "high" and prediction == "likely_miss":
        conf *= 0.85
        notes.append("high squeeze risk can overwhelm a miss")
    if reaction_pattern == "beats_fade" and prediction == "likely_beat":
        conf *= 0.9
        notes.append("history shows beats tend to fade")
    if implied_verdict == "rich":
        conf *= 0.92
        notes.append("options-implied move is rich vs realized — expensive entry")

    conf = round(conf, 3)
    sizing = _sizing(conf, prediction)

    # Hard binary-event discipline.
    if implied_move_pct is not None and implied_move_pct > 0.20:
        sizing = "no_position"
        notes.append("implied move > 20% — binary event, not a research-edge trade")
    if conf < 0.55:
        sizing = "no_position"

    return {"confidence": conf, "sizing": sizing, "notes": notes}


def final_verdict(
    signals: List[Dict[str, Any]],
    *,
    positioning: Optional[str] = None,
    squeeze_risk: Optional[str] = None,
    reaction_pattern: Optional[str] = None,
    implied_verdict: Optional[str] = None,
    implied_move_pct: Optional[float] = None,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Full synthesis: direction (weighted, coverage-renormalized) x asymmetry."""
    score = direction_score(signals, weights)
    cov = coverage(signals, weights)
    agree = agreement(signals, weights)
    prediction = predict_from_score(score)
    conf0 = base_confidence(score, cov, agree)
    adj = asymmetry_adjust(prediction, conf0, positioning, squeeze_risk,
                           reaction_pattern, implied_verdict, implied_move_pct)
    return {
        "prediction": prediction,
        "direction_score": score,
        "coverage": cov,
        "agreement": agree,
        "base_confidence": conf0,
        "confidence": adj["confidence"],
        "sizing": adj["sizing"],
        "low_confidence": bool(cov < 0.5 or adj["confidence"] < 0.5),
        "asymmetry_notes": adj["notes"],
    }
