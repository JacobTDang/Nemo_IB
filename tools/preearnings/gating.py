"""Cost gate for deep pre-earnings research (Phase F).

Layer 0 (cheap structured signals) always runs. The expensive Layer 1+ sub-agent
fan-out runs only when this gate passes, so the Sentry loop never blanket-fans-out
across a watchlist. Pure and unit-tested.
"""
from __future__ import annotations

from typing import Any, Dict, Optional


def should_deep_research(
    days_to_earnings: Optional[int],
    liquid: bool,
    has_peers: bool,
    has_options: bool = True,
    deep_window_days: int = 10,
) -> Dict[str, Any]:
    """Decide whether to run the deep (sub-agent) layers.

    Gate: earnings within the window and not already passed, the name is liquid
    enough to trade, and there is at least one deep input available (peers for
    readthrough, or options for implied move). Returns {deep, reason}.
    """
    if days_to_earnings is None:
        return {"deep": False, "reason": "no confirmed earnings date"}
    if days_to_earnings < 0:
        return {"deep": False, "reason": "earnings already passed"}
    if days_to_earnings > deep_window_days:
        return {"deep": False,
                "reason": f"earnings {days_to_earnings}d out (> {deep_window_days}d window)"}
    if not liquid:
        return {"deep": False, "reason": "insufficient liquidity to trade"}
    if not (has_peers or has_options):
        return {"deep": False, "reason": "no deep inputs (no peers and no options)"}
    return {"deep": True, "reason": "within window, liquid, deep inputs available"}
