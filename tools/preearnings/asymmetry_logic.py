"""Deterministic asymmetry primitives (Phase D).

These measure HOW a stock reacts given a result and how it's positioned — they do
NOT predict direction. The orchestration (pulling short interest, options metrics,
past surprises + next-day returns) lives in the skill; this module is the pure
math. Everything is derived from the ticker's own data — nothing hardcoded.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import math


def _sign(x: float) -> int:
    return 1 if x > 0 else (-1 if x < 0 else 0)


def reaction_profile(events: List[Dict[str, Any]], surprise_tol: float = 0.0) -> Dict[str, Any]:
    """Characterize the ticker's own earnings-reaction behavior.

    events: [{surprise_pct, next_day_return}] from past prints.
    Returns pattern + stats:
      - clean_repricer : reaction sign tracks surprise sign (>=60% consistent)
      - beats_fade     : positive surprises frequently fade to flat/negative
      - noisy          : no reliable relationship
    """
    pairs = [
        (float(e["surprise_pct"]), float(e["next_day_return"]))
        for e in events
        if e.get("surprise_pct") is not None and e.get("next_day_return") is not None
    ]
    if not pairs:
        return {"pattern": "unknown", "n": 0, "avg_abs_move": None,
                "consistency": None, "beat_fade_rate": None}

    n = len(pairs)
    consistent = sum(1 for s, r in pairs if _sign(s) == _sign(r) and _sign(s) != 0)
    directional = sum(1 for s, r in pairs if _sign(s) != 0)
    consistency = consistent / directional if directional else 0.0

    beats = [(s, r) for s, r in pairs if s > surprise_tol]
    beat_fade_rate = (sum(1 for _, r in beats if r <= 0) / len(beats)) if beats else None

    avg_abs_move = round(sum(abs(r) for _, r in pairs) / n, 4)

    if consistency >= 0.6:
        pattern = "clean_repricer"
    elif beat_fade_rate is not None and beat_fade_rate >= 0.5:
        pattern = "beats_fade"
    else:
        pattern = "noisy"

    return {
        "pattern": pattern,
        "n": n,
        "avg_abs_move": avg_abs_move,
        "consistency": round(consistency, 3),
        "beat_fade_rate": round(beat_fade_rate, 3) if beat_fade_rate is not None else None,
    }


def classify_positioning(
    short_interest_pct_float: Optional[float] = None,
    days_to_cover: Optional[float] = None,
    put_call_iv_skew: Optional[float] = None,
) -> Dict[str, Any]:
    """Classify crowding + squeeze risk from positioning inputs.

    short_interest_pct_float: % of float sold short
    days_to_cover:            short interest / avg daily volume
    put_call_iv_skew:         put_iv - call_iv (positive = hedging/bearish skew)
    """
    si = short_interest_pct_float
    skew = put_call_iv_skew

    positioning = "neutral"
    squeeze_risk = "low"

    if si is not None:
        if si >= 20:
            positioning, squeeze_risk = "crowded_short", "high"
        elif si >= 10:
            positioning, squeeze_risk = "crowded_short", "elevated"
        elif si < 3 and skew is not None and skew < -0.02:
            positioning = "crowded_long"   # low short + call-heavy skew
    if squeeze_risk in ("elevated", "high") and days_to_cover and days_to_cover >= 5:
        squeeze_risk = "high"

    return {"positioning": positioning, "squeeze_risk": squeeze_risk}


def implied_vs_realized(
    implied_move: float, realized_abs_moves: List[float]
) -> Dict[str, Any]:
    """Is the options-implied move rich or cheap vs the ticker's realized history?"""
    moves = [abs(float(m)) for m in (realized_abs_moves or []) if m is not None]
    if implied_move is None or not moves:
        return {"verdict": "unknown", "implied": implied_move, "avg_realized": None}
    avg_realized = sum(moves) / len(moves)
    if avg_realized <= 0:
        return {"verdict": "unknown", "implied": implied_move, "avg_realized": avg_realized}
    ratio = implied_move / avg_realized
    if ratio > 1.15:
        verdict = "rich"
    elif ratio < 0.85:
        verdict = "cheap"
    else:
        verdict = "fair"
    return {"verdict": verdict, "implied": round(implied_move, 4),
            "avg_realized": round(avg_realized, 4), "ratio": round(ratio, 2)}
