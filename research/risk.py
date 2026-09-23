"""The limits the paper book enforces on itself (issue #117).

Numbers set by the owner on 2026-09-23, before any forward result was read:

- No position larger than 5% of the gross target. The scan spreads the gross
  over twenty names, which is 5% each, but only by arithmetic; this states it,
  so a change to the name count or the sizing rule cannot move it silently.
- A drawdown switch. Each night, before deciding, the scan records what the
  book is worth: closed trades at their net, open ones marked at the last
  close. A loss of 10% of the gross target from the peak trips the switch, and
  a tripped switch stops the scan filing orders until a person resets it with
  a reason. It latches: a switch that re-arms when equity bounces trades
  through the drawdown it exists to stop.

The go/no-go gate is in `scoring`, beside the statistics it reads.

    python -m research.risk                          # the switch, as JSON
    python -m research.risk --reset --reason "why"   # re-arm a tripped switch
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from research import pit_store, scoring

MAX_POSITION_FRACTION = 0.05
DRAWDOWN_HALT_FRACTION = 0.10


class RiskHalted(RuntimeError):
    """The drawdown switch is tripped, so nothing may be filed."""


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _stamp(as_of: str) -> str:
    # The decision's own time, like every other row the scan writes, so a
    # replayed date sees its own switch and not today's.
    return f"{as_of}T21:00:00Z"


def _mark(order: Dict[str, Any], as_of: str) -> Optional[float]:
    """An open position's worth in dollars at the last close, or None.

    Entered at the open of its intended session, rolled one session when the
    exchange was shut, the same way scoring fills it. Charged its whole round
    trip already: a mark that ignores the exit's cost flatters every open book.
    """
    bars = pit_store.adjusted_bars(order["ticker"], as_of)
    session = order["intended_session"]
    entry = next((b for b in bars if b["trade_date"] == session), None)
    if entry is None and scoring._exchange_shut(session, as_of):
        rolled = scoring._next_open_session(session, as_of)
        entry = next((b for b in bars if b["trade_date"] == rolled),
                     None) if rolled else None
    last = next((b for b in reversed(bars) if b.get("close")), None)
    if entry is None or not entry.get("open") or last is None:
        return None
    move = last["close"] / entry["open"] - 1.0
    if order.get("side") == "short":
        move = -move
    charges = ((order.get("cost_bps") or 0.0)
               + (order.get("borrow_bps") or 0.0)) / 10_000
    return (move - charges) * (order.get("target_dollars") or 0.0)


def book_equity(as_of: str,
                horizon_days: int = scoring.DEFAULT_HORIZON_DAYS
                ) -> Dict[str, Any]:
    """What the paper book is worth on `as_of`, in dollars.

    Closed trades at the net scoring gives them, open ones at `_mark`. An open
    position with no price to mark it at is counted in `unmarked_trades`, not
    valued at zero: its exposure is unknown, which is a different fact.
    """
    out = {"equity_dollars": 0.0, "realized_dollars": 0.0,
           "open_dollars": 0.0, "closed_trades": 0, "open_trades": 0,
           "unmarked_trades": 0}
    orders = pit_store.paper_orders_as_of(as_of, accepted_only=True)
    if not orders:
        return out

    book = scoring.evaluate_book(as_of, horizon_days)
    settled = {(r["ticker"], r["as_of_date"])
               for r in book["scored"] + book["unfilled"]}
    out["realized_dollars"] = sum(r["net_bps"] / 10_000
                                  * (r["target_dollars"] or 0.0)
                                  for r in book["scored"])
    out["closed_trades"] = len(book["scored"])

    for order in orders:
        session = order["intended_session"]
        if (order["ticker"], order["as_of_date"]) in settled:
            continue
        if not session or session > as_of:
            continue  # not entered yet: no exposure
        mark = _mark(order, as_of)
        if mark is None:
            out["unmarked_trades"] += 1
            continue
        out["open_dollars"] += mark
        out["open_trades"] += 1

    out["equity_dollars"] = out["realized_dollars"] + out["open_dollars"]
    return out


def check_and_record(as_of: str, gross_target: float,
                     equity: Optional[Dict[str, Any]] = None,
                     horizon_days: int = scoring.DEFAULT_HORIZON_DAYS
                     ) -> Dict[str, Any]:
    """Record the book's equity, trip the switch if it is due, report both.

    `equity` is for tests; the scan leaves it to `book_equity`.
    """
    limit = DRAWDOWN_HALT_FRACTION * gross_target
    equity = equity if equity is not None else book_equity(as_of, horizon_days)
    pit_store.record_book_equity(as_of, equity, limit_dollars=limit,
                                 recorded_at=_stamp(as_of))
    state = pit_store.risk_state(as_of)
    drawdown = state["drawdown_dollars"]

    if not state["halted"] and drawdown is not None and drawdown >= limit:
        reason = (
            f"risk switch tripped on {as_of}: book equity "
            f"{state['latest']['equity_dollars']:+,.0f} is {drawdown:,.0f} "
            f"below its peak of {state['peak_dollars']:+,.0f}, at or past the "
            f"limit of {limit:,.0f}, {DRAWDOWN_HALT_FRACTION:.0%} of the "
            f"{gross_target:,.0f} gross target. The scan files no orders until "
            f"a person resets it with a reason")
        pit_store.record_risk_event(as_of, "tripped", reason,
                                    drawdown_dollars=drawdown,
                                    recorded_at=_stamp(as_of))
        state = pit_store.risk_state(as_of)

    return {"as_of": as_of, "halted": state["halted"],
            "reason": state["tripped"]["reason"] if state["tripped"] else None,
            "equity_dollars": (state["latest"] or {}).get("equity_dollars"),
            "peak_dollars": state["peak_dollars"],
            "drawdown_dollars": state["drawdown_dollars"],
            "limit_dollars": limit}


def reset(as_of: str, reason: str) -> Dict[str, Any]:
    """Re-arm a tripped switch, measuring the peak from here on.

    Refused when the switch is armed: a reset then would only lower the peak,
    and the one use for that is to stop the switch tripping when it should.
    """
    if not reason or not reason.strip():
        raise ValueError("a reset needs a reason; it is kept on the record")
    state = pit_store.risk_state(as_of)
    if not state["halted"]:
        raise ValueError(f"the risk switch is not tripped on {as_of}, so there "
                         f"is nothing to reset")
    baseline = (state["latest"] or {}).get("equity_dollars", 0.0)
    pit_store.record_risk_event(as_of, "reset", reason.strip(),
                                baseline_dollars=baseline,
                                recorded_at=_stamp(as_of))
    return {"as_of": as_of, "reset": True, "baseline_dollars": baseline,
            "reason": reason.strip()}


def main(argv=None) -> int:
    import argparse
    import json

    pit_store.init_schema()
    parser = argparse.ArgumentParser(
        prog="risk", description="The paper book's drawdown switch.")
    parser.add_argument("--as-of", dest="as_of", default=None,
                        help="date to act as (default: today, UTC)")
    parser.add_argument("--reset", action="store_true",
                        help="re-arm a tripped switch; needs --reason")
    parser.add_argument("--reason", default=None,
                        help="why it is safe to trade again; kept on record")
    args = parser.parse_args(argv)
    as_of = args.as_of or _today()

    if args.reset:
        if not args.reason:
            parser.error("--reset needs --reason: the record keeps why the "
                         "book was allowed to trade again")
        try:
            print(json.dumps(reset(as_of, args.reason), indent=2))
        except ValueError as exc:
            print(json.dumps({"reset": False, "error": str(exc)}, indent=2))
            return 1
        return 0

    state = pit_store.risk_state(as_of)
    print(json.dumps(state, indent=2, default=str))
    return 1 if state["halted"] else 0


if __name__ == "__main__":  # pragma: no cover - exercised via main()
    raise SystemExit(main())
