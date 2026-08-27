"""What the filed orders actually did, and what that implies about the drift.

The scanner logs an intention -- a ticker, a side, a size, and the session the
order is for -- and deliberately no price, because that session has not
happened when the row is written. This is where the two meet. Everything
upstream measures something; nothing upstream is ever checked against an
outcome, and measurement that is never checked is arithmetic.

It is also the only route to the one number the scanner admits it assumed.
`DRIFT_BPS_PER_SUE` is declared, reported uncalibrated on every scan, and every
net edge is proportional to it. The honest replacement is the coefficient this
book's own outcomes imply, which is what `drift_bps_per_sue` here reports --
measured from realised returns, never echoed back from the assumption.

The failure modes of a scorer all point the same way, toward flattering the
strategy:

  Scoring an unfinished horizon. Not a small error: the trades still open are
  disproportionately the ones that have not gone anywhere, so counting them
  early drags every average toward nothing having happened.

  Filling an order that never filled. A name that did not trade on its intended
  session is not a position bought at the next available price; it is a
  position nobody has. Sliding to the next print is how a study buys the day
  after bad news.

  Reading a split as a return. The store keeps as-traded prices on purpose, so
  a 10-for-1 is a real -90% print sitting in the middle of a holding period.
  Everything here reads `adjusted_bars`, which rebuilds from actions the
  scorer could have known.

  Dropping the cost that was already measured and written down at decision
  time. It is on the order row; there is no excuse for scoring gross.
"""
from __future__ import annotations

import statistics
from typing import Any, Dict, List, Optional

from research import pit_store

# How many scored trades before the realised coefficient is worth quoting. One
# trade is not a coefficient, and replacing a declared assumption with a
# one-sample estimate is trading a stated guess for an unstated one.
MIN_CALIBRATION_SAMPLE = 30

DEFAULT_HORIZON_DAYS = 20


def _today() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).date().isoformat()


def score_orders(as_of: Optional[str] = None,
                 horizon_days: int = DEFAULT_HORIZON_DAYS) -> Dict[str, Any]:
    """Score every filed order whose holding period has finished by `as_of`.

    Reads only what a reader standing on `as_of` could have seen, like
    everything else here -- so a scorer run today cannot use a bar that was
    back-filled tomorrow, and re-running it for a past date gives that date's
    answer rather than today's.
    """
    as_of = as_of or _today()
    orders = [o for o in pit_store.paper_orders_as_of(as_of, accepted_only=True)]

    scored: List[Dict[str, Any]] = []
    pending: List[Dict[str, Any]] = []
    unfilled: List[Dict[str, Any]] = []

    for order in orders:
        entry_session = order["intended_session"]
        if not entry_session:
            unfilled.append({**_stub(order),
                             "reason": "no intended session on the order"})
            continue

        bars = pit_store.adjusted_bars(order["ticker"], as_of)
        forward = [b for b in bars if b["trade_date"] >= entry_session]

        entry = next((b for b in forward if b["trade_date"] == entry_session),
                     None)
        if entry is None:
            # Two different nothings. If the horizon has not elapsed yet the
            # session may simply not have been recorded; if it has, the name
            # did not trade and the order never filled.
            later = [b for b in bars if b["trade_date"] > entry_session]
            if len(later) < horizon_days:
                pending.append({**_stub(order),
                                "reason": f"{entry_session} not yet recorded"})
            else:
                unfilled.append({
                    **_stub(order),
                    "reason": (f"{order['ticker']} did not trade on "
                               f"{entry_session}, so the order never filled")})
            continue

        if len(forward) <= horizon_days:
            pending.append({
                **_stub(order),
                "reason": (f"{len(forward) - 1} of {horizon_days} sessions "
                           f"elapsed since {entry_session}")})
            continue

        exit_bar = forward[horizon_days]
        entry_price = entry["open"]
        exit_price = exit_bar["open"]
        if not entry_price or not exit_price:
            unfilled.append({**_stub(order),
                             "reason": "a price on the path is missing"})
            continue

        move = (exit_price / entry_price) - 1.0
        if order["side"] == "short":
            move = -move
        gross_bps = move * 10_000
        cost_bps = order["cost_bps"] or 0.0

        scored.append({
            **_stub(order),
            "entry_session": entry_session,
            "entry_price": entry_price,
            "exit_session": exit_bar["trade_date"],
            "exit_price": exit_price,
            "gross_bps": gross_bps,
            "cost_bps": cost_bps,
            "net_bps": gross_bps - cost_bps,
            "expected_edge_bps": order["expected_edge_bps"],
        })

    return {"as_of": as_of, "horizon_days": horizon_days,
            "scored": scored, "pending": pending, "unfilled": unfilled,
            **_summarise(scored)}


def _stub(order: Dict[str, Any]) -> Dict[str, Any]:
    return {"ticker": order["ticker"], "as_of_date": order["as_of_date"],
            "side": order["side"], "sue": order["sue"],
            "target_dollars": order["target_dollars"]}


def _summarise(scored: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The realised numbers, and an honest label on the coefficient.

    `drift_bps_per_sue` is gross of cost on purpose. Cost is a property of how
    the position was traded; drift is a property of the surprise, and mixing
    them gives a coefficient that changes with the size of the book.
    """
    if not scored:
        return {"sample": 0, "hit_rate": None, "mean_net_bps": None,
                "mean_gross_bps": None, "drift_bps_per_sue": None,
                "calibrated": False,
                "calibration_note": "no finished trades to measure"}

    nets = [r["net_bps"] for r in scored]
    grosses = [r["gross_bps"] for r in scored]
    usable = [(r["gross_bps"], abs(r["sue"])) for r in scored
              if r["sue"] not in (None, 0)]
    per_sue = [g / s for g, s in usable] if usable else []

    sample = len(scored)
    enough = sample >= MIN_CALIBRATION_SAMPLE
    return {
        "sample": sample,
        "hit_rate": sum(1 for n in nets if n > 0) / sample,
        "mean_net_bps": statistics.fmean(nets),
        "mean_gross_bps": statistics.fmean(grosses),
        "drift_bps_per_sue": statistics.fmean(per_sue) if per_sue else None,
        "calibrated": enough,
        "calibration_note": (
            f"{sample} finished trades"
            if enough else
            f"sample of {sample} is under the {MIN_CALIBRATION_SAMPLE} this "
            f"will quote a coefficient from; replacing a declared assumption "
            f"with a handful of trades is trading a stated guess for an "
            f"unstated one"),
    }


# ------------------------------------------------------------- entry point

def main(argv: Optional[List[str]] = None) -> int:
    """`python -m research.scoring`. Reports; changes nothing."""
    import argparse
    import json

    parser = argparse.ArgumentParser(
        prog="scoring",
        description="Score filed paper orders whose horizon has finished.")
    parser.add_argument("--as-of", dest="as_of", default=None)
    parser.add_argument("--horizon-days", dest="horizon_days", type=int,
                        default=DEFAULT_HORIZON_DAYS)
    args = parser.parse_args(argv)

    result = score_orders(as_of=args.as_of, horizon_days=args.horizon_days)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via main()
    raise SystemExit(main())
