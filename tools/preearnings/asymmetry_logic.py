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

    # beats_fade takes precedence: for a likely_beat setup, "your beats get
    # sold" is the dangerous property even when misses track down cleanly
    # enough to clear the consistency bar. Requires >=2 beats (one fading beat
    # is an anecdote, not a pattern).
    if beat_fade_rate is not None and beat_fade_rate >= 0.5 and len(beats) >= 2:
        pattern = "beats_fade"
    elif consistency >= 0.6:
        pattern = "clean_repricer"
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
    put_call_volume_ratio: Optional[float] = None,
    momentum_3m_pct: Optional[float] = None,
    analyst_pct_bullish: Optional[float] = None,
) -> Dict[str, Any]:
    """Classify crowding + squeeze risk from positioning inputs.

    short_interest_pct_float: % of float sold short
    days_to_cover:            short interest / avg daily volume
    put_call_iv_skew:         put_iv - call_iv (positive = hedging/bearish skew)
    put_call_volume_ratio:    put volume / call volume (< 0.5 = call-heavy speculation)
    momentum_3m_pct:          3-month price return % (a big run-up = late longs)
    analyst_pct_bullish:      % of covering analysts at buy/strong-buy

    crowded_long requires LOW short interest plus >=2 independent pieces of
    crowding evidence — a single noisy indicator (IV skew alone) is not enough.
    (Gap surfaced live: a name with SI 2.1%, P/C volume 0.31, +53% 3M and 81.6%
    bullish analysts read "neutral" under the old skew-only rule.)
    """
    si = short_interest_pct_float
    skew = put_call_iv_skew

    positioning = "neutral"
    squeeze_risk = "low"

    if si is not None and si >= 20:
        positioning, squeeze_risk = "crowded_short", "high"
    elif si is not None and si >= 10:
        positioning, squeeze_risk = "crowded_short", "elevated"
    else:
        evidence = 0
        if put_call_volume_ratio is not None and put_call_volume_ratio < 0.5:
            evidence += 1          # call-heavy speculative flow
        if skew is not None and skew < -0.02:
            evidence += 1          # calls priced at an IV premium
        if momentum_3m_pct is not None and momentum_3m_pct > 30:
            evidence += 1          # large run-up = late-money longs
        if analyst_pct_bullish is not None and analyst_pct_bullish > 75:
            evidence += 1          # sell-side consensus crowd
        # crowded_long REQUIRES known-low short interest — with SI unknown, a
        # heavily-shorted name could be classified exactly backwards.
        if si is not None and si < 5 and evidence >= 2:
            positioning = "crowded_long"

    if squeeze_risk in ("elevated", "high") and days_to_cover and days_to_cover >= 5:
        squeeze_risk = "high"

    return {"positioning": positioning, "squeeze_risk": squeeze_risk}


def _parse_d(value: Any):
    from datetime import date, datetime
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    s = str(value).strip()[:10]
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError:
        return None


def pair_surprises_with_reactions(
    surprises: List[Dict[str, Any]],
    event_dates: List[Any],
    bars: List[Dict[str, Any]],
    max_gap_days: int = 75,
    max_back_days: int = 45,
) -> List[Dict[str, Any]]:
    """Build the (surprise, next-day reaction) pairs that reaction_profile needs.

    surprises:   [{period: quarter-end date, surprise_pct}] (get_earnings_surprises)
    event_dates: report/8-K filing dates (get_company_filings_history)
    bars:        [{date, close}] daily bars (get_price_history)

    Each surprise quarter is matched to the UNUSED event date NEAREST its period
    end, within [period_end - max_back_days, period_end + max_gap_days]. Events
    BEFORE the labeled period end are allowed because vendors label fiscal
    quarters with calendar-quarter ends — fiscal-offset companies (Nov/May/Jan
    year-ends) report 2-3 weeks BEFORE their label. The reaction approximates
    close[t+1] vs close[t], where t is the report date's bar (or nearest prior
    trading day) — most prints land after the close, so t+1 carries the
    reaction. BMO reporters are off by a session; sign/magnitude survive for
    material moves.
    """
    def _valid_close(b) -> bool:
        c = b.get("close")
        try:
            f = float(c)
        except (TypeError, ValueError):
            return False
        return f == f and f > 0          # rejects NaN and non-positive

    parsed_bars = sorted(
        ((d, float(b["close"])) for b in bars
         if _valid_close(b) and (d := _parse_d(b.get("date"))) is not None),
        key=lambda x: x[0],
    )
    if len(parsed_bars) < 2:
        return []
    bar_dates = [d for d, _ in parsed_bars]

    events = sorted({d for e in event_dates if (d := _parse_d(e)) is not None})

    import bisect
    from datetime import timedelta

    # OPTIMAL assignment (maximize matched quarters, then minimize total
    # distance). Per-quarter greedy let an earlier quarter steal a later
    # quarter's only event; distance-greedy let a late filer take the PRIOR
    # quarter's report (44d cross-match beating two correct 47d matches).
    # Sizes are tiny (<=12 surprises x <=12 events), so exhaustive search with
    # a skip option is exact and instant.
    parsed_surprises = []
    for i, s in enumerate(sorted(surprises, key=lambda x: str(x.get("period") or ""))):
        period_end = _parse_d(s.get("period"))
        if period_end is None or s.get("surprise_pct") is None:
            continue
        parsed_surprises.append((i, period_end, s))

    cand_lists: List[tuple] = []   # (period_end, s, [(dist, ev), ...])
    for i, period_end, s in parsed_surprises:
        window_start = period_end - timedelta(days=max_back_days)
        window_end = period_end + timedelta(days=max_gap_days)
        opts = sorted(
            (abs((ev - period_end).days), ev)
            for ev in events if window_start <= ev <= window_end
        )
        if opts:
            cand_lists.append((period_end, s, opts))

    best: Dict[str, Any] = {"count": -1, "dist": float("inf"), "pairs": []}

    def _search(k: int, used: set, pairs: List[tuple], total: int) -> None:
        # upper bound prune: even matching everything left can't beat best
        if len(pairs) + (len(cand_lists) - k) < best["count"]:
            return
        if k == len(cand_lists):
            if (len(pairs) > best["count"]
                    or (len(pairs) == best["count"] and total < best["dist"])):
                best.update(count=len(pairs), dist=total, pairs=list(pairs))
            return
        period_end, s, opts = cand_lists[k]
        for dist, ev in opts:
            if ev in used:
                continue
            used.add(ev)
            pairs.append((period_end, ev, s))
            _search(k + 1, used, pairs, total + dist)
            pairs.pop()
            used.remove(ev)
        _search(k + 1, used, pairs, total)   # leave this quarter unmatched

    _search(0, set(), [], 0)
    assigned = best["pairs"]

    out: List[Dict[str, Any]] = []
    for period_end, ev, s in sorted(assigned, key=lambda x: x[0]):
        # bar at the report date, or nearest prior trading day
        idx = bisect.bisect_right(bar_dates, ev) - 1
        if idx < 0 or idx + 1 >= len(parsed_bars):
            continue
        c0, c1 = parsed_bars[idx][1], parsed_bars[idx + 1][1]
        out.append({
            "surprise_pct": float(s["surprise_pct"]),
            "next_day_return": round((c1 - c0) / c0 * 100, 2),
            "period": str(s.get("period")),
            "event_date": ev.isoformat(),
        })
    return out


def score_reaction(
    positioning: Optional[str],
    outcome: Optional[str],
    price_move_1d_pct: Optional[float],
    implied_move_pct: Optional[float],
    prediction: Optional[str] = None,
) -> Dict[str, Any]:
    """Score the ASYMMETRY call separately from the EPS-direction call
    (post-earnings, used by /earnings-eval).

    price_direction_match: did the 1-day price move agree with the prediction?
      likely_beat -> up, likely_miss -> down. (None for in_line / missing data.)

    asymmetry_correct: did the crowding thesis hold? Falsifiable claims:
      crowded_long  + beat -> reward muted   (move <  +0.5 * implied) or negative
      crowded_long  + miss -> punished hard  (move <= -0.5 * implied)
      crowded_short + beat -> squeeze        (move >= +0.5 * implied)
      crowded_short + miss -> cushioned      (move >  -0.5 * implied)
      neutral positioning -> no asymmetry call was made -> not scored (None).
    """
    move = None if price_move_1d_pct is None else float(price_move_1d_pct)
    # UNIT CONTRACT: implied_move_pct is a FRACTION (0.128 = 12.8%); the move is
    # a PERCENT (-6.4 = -6.4%). Mixing them silently corrupts every grade, so a
    # percent-shaped implied is rejected loudly.
    if implied_move_pct is not None and not (0 <= abs(float(implied_move_pct)) < 1.0):
        raise ValueError(
            f"implied_move_pct must be a fraction in [0,1) e.g. 0.128, got {implied_move_pct} "
            "(looks like a percent — divide by 100)")
    implied = None if implied_move_pct is None else abs(float(implied_move_pct)) * 100.0

    price_direction_match: Optional[int] = None
    if move is not None and prediction in ("likely_beat", "likely_miss"):
        if prediction == "likely_beat":
            price_direction_match = 1 if move > 0 else 0
        else:
            price_direction_match = 1 if move < 0 else 0

    asymmetry_correct: Optional[int] = None
    basis = "no asymmetry call (neutral positioning)"
    if (positioning in ("crowded_long", "crowded_short")
            and outcome in ("beat", "miss")
            and move is not None and implied is not None and implied > 0):
        half = 0.5 * implied
        if positioning == "crowded_long" and outcome == "beat":
            asymmetry_correct = 1 if move < half else 0
            basis = f"crowded_long+beat: muted reward expected (move {move} vs +{half:.1f} threshold)"
        elif positioning == "crowded_long" and outcome == "miss":
            asymmetry_correct = 1 if move <= -half else 0
            basis = f"crowded_long+miss: hard punishment expected (move {move} vs -{half:.1f})"
        elif positioning == "crowded_short" and outcome == "beat":
            asymmetry_correct = 1 if move >= half else 0
            basis = f"crowded_short+beat: squeeze expected (move {move} vs +{half:.1f})"
        else:  # crowded_short + miss
            asymmetry_correct = 1 if move > -half else 0
            basis = f"crowded_short+miss: cushioned downside expected (move {move} vs -{half:.1f})"
    elif positioning in ("crowded_long", "crowded_short"):
        basis = "asymmetry call made but outcome/move/implied unavailable"

    return {
        "asymmetry_correct": asymmetry_correct,
        "price_direction_match": price_direction_match,
        "basis": basis,
    }


def implied_vs_realized(
    implied_move: float, realized_abs_moves: List[float]
) -> Dict[str, Any]:
    """Is the options-implied move rich or cheap vs the ticker's realized history?

    UNIT CONTRACT: both inputs in the SAME unit (both fractions, e.g. 0.128 and
    [0.053, 0.076], or both percents). pair_surprises_with_reactions emits
    next_day_return in PERCENT — divide by 100 before passing here when implied
    is a fraction. Mixed units are detected and returned as unknown rather than
    a silently wrong rich/cheap verdict."""
    moves = [abs(float(m)) for m in (realized_abs_moves or []) if m is not None]
    if implied_move is None or not moves:
        return {"verdict": "unknown", "implied": implied_move, "avg_realized": None}
    avg_realized = sum(moves) / len(moves)
    if avg_realized <= 0:
        return {"verdict": "unknown", "implied": implied_move, "avg_realized": avg_realized}
    if implied_move < 1.0 and avg_realized > 2.0:
        return {"verdict": "unknown", "implied": implied_move, "avg_realized": avg_realized,
                "note": "unit mismatch suspected: implied looks like a fraction, "
                        "realized like percents — convert to the same unit"}
    ratio = implied_move / avg_realized
    if ratio > 1.15:
        verdict = "rich"
    elif ratio < 0.85:
        verdict = "cheap"
    else:
        verdict = "fair"
    return {"verdict": verdict, "implied": round(implied_move, 4),
            "avg_realized": round(avg_realized, 4), "ratio": round(ratio, 2)}
