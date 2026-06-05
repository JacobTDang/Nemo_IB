"""Deterministic peer-readthrough logic (Phase B).

The orchestration (deriving the peer universe via get_company_peers /
get_supply_chain, spawning a /cross-company-readthrough sub-agent per reported
peer) lives in the /peer-readthrough-fanout skill. This module is the pure core:
which peers count as "reported this quarter", how to weight a relationship type,
and how to aggregate the per-peer readthroughs into one signal.

No company/ticker is hardcoded — callers pass derived data in.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

_DIR_SIGN = {"bullish": 1.0, "bearish": -1.0, "neutral": 0.0, "na": 0.0}

# Relationship TYPE -> readthrough weight. These are generic relationship
# categories (not companies): a supplier/customer print reads through harder to
# the target than a same-sector peer, which reads harder than a loose adjacency.
_RELEVANCE = {
    "supplier": 1.0,
    "customer": 1.0,
    "competitor": 0.7,
    "peer": 0.7,
    "adjacent": 0.4,
}
_RELEVANCE_DEFAULT = 0.5


def _to_date(value: Any) -> Optional[date]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    s = str(value).strip()
    if not s:
        return None
    # Accept full ISO timestamps or bare YYYY-MM-DD
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).date()
    except ValueError:
        try:
            return datetime.strptime(s[:10], "%Y-%m-%d").date()
        except ValueError:
            return None


def quarter_window(
    target_last_earnings: Any,
    today: Any,
    max_span_days: int = 130,
    default_span_days: int = 95,
) -> Tuple[date, date]:
    """Infer the current reporting window for the target.

    Peers that reported between the target's previous print and now are reporting
    on the same calendar quarter, so their results read through to the target's
    upcoming print. Derived from the target's own cadence — no hardcoded dates.
    The span is clamped so a missing/stale last-earnings can't produce an absurd
    window.
    """
    end = _to_date(today) or date.today()
    start = _to_date(target_last_earnings)
    if start is None or start >= end:
        start = end - timedelta(days=default_span_days)
    if (end - start).days > max_span_days:
        start = end - timedelta(days=max_span_days)
    return start, end


def reported_this_quarter(peer_report_date: Any, window: Tuple[date, date]) -> bool:
    """True if the peer's most recent report falls inside the window (inclusive)."""
    d = _to_date(peer_report_date)
    if d is None:
        return False
    start, end = window
    return start <= d <= end


def rank_peer_relevance(relationship: Optional[str]) -> float:
    """Map a generic relationship type to a readthrough weight in [0, 1]."""
    if not relationship:
        return _RELEVANCE_DEFAULT
    return _RELEVANCE.get(str(relationship).strip().lower(), _RELEVANCE_DEFAULT)


def select_peers_for_fanout(
    peers: List[Dict[str, Any]],
    window: Tuple[date, date],
    max_n: int = 6,
) -> List[Dict[str, Any]]:
    """From a derived peer universe, keep those that reported this quarter, ranked
    by relevance (then recency), capped at max_n for cost.

    Each peer dict should carry: ticker, relationship (optional), report_date.
    Returns the same dicts annotated with `relevance`, ready for fan-out.
    """
    eligible = []
    for p in peers:
        if not reported_this_quarter(p.get("report_date"), window):
            continue
        rel = rank_peer_relevance(p.get("relationship"))
        d = _to_date(p.get("report_date")) or window[0]
        eligible.append({**p, "relevance": rel, "_rank_date": d})
    eligible.sort(key=lambda x: (-x["relevance"], x["_rank_date"]), reverse=False)
    # primary sort: relevance desc; tie-break: more recent report first
    eligible.sort(key=lambda x: (x["relevance"], x["_rank_date"]), reverse=True)
    for e in eligible:
        e.pop("_rank_date", None)
    return eligible[:max_n]


def aggregate_readthroughs(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Relevance-weighted aggregation of per-peer readthroughs.

    items: [{ticker, direction, magnitude (0..1), relevance (0..1)}]
    Returns {direction, magnitude, score, n, detail}. Empty -> neutral/na, n=0.
    """
    usable = [
        it for it in items
        if it.get("direction") in _DIR_SIGN and it.get("relevance", 0) > 0
    ]
    if not usable:
        return {"direction": "na", "magnitude": 0.0, "score": 0.0,
                "n": 0, "detail": []}

    wsum = sum(it["relevance"] for it in usable)
    score = sum(
        it["relevance"] * _DIR_SIGN[it["direction"]] * float(it.get("magnitude") or 0.0)
        for it in usable
    ) / wsum if wsum else 0.0

    direction = "neutral"
    if score > 0.15:
        direction = "bullish"
    elif score < -0.15:
        direction = "bearish"

    return {
        "direction": direction,
        "magnitude": round(min(abs(score), 1.0), 3),
        "score": round(score, 3),
        "n": len(usable),
        "detail": [
            {"ticker": it.get("ticker"), "direction": it["direction"],
             "magnitude": it.get("magnitude"), "relevance": it["relevance"]}
            for it in usable
        ],
    }
