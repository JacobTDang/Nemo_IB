"""Paper fills: the scan's orders at a paper broker, and what it filled (#116).

Every result the study has produced is net of modeled costs: a spread
estimated from daily bars plus an impact term. The release-timed arm's median
trade is about zero, so a cost error of a few tens of basis points changes its
sign, and the names that carry the edge are the thin ones where a spread from
daily bars is least reliable. This job replaces the assumption with a
measurement.

Each weekday morning, before the opening auction closes to new orders, it
sends the legs due that session to an Alpaca PAPER account as opening-auction
market orders: entries for orders the scan filed for today, exits for entries
the holding period has run out on. After the open it collects what filled. The
report sets each fill against the open the scoring assumes.

It runs from its own image (`docker build --target fills`). The data image
serves market data and cannot place an order, and this module is not in it.
It never constructs a live broker.

Paper fills are simulated from quotes. They measure the quoted spread at the
open, not the price impact of the order itself; only small live orders measure
that, and that is the owner's decision.

    python -m tools.alpaca.fills submit     # 13:15 UTC, weekdays
    python -m tools.alpaca.fills collect    # 14:45 UTC, weekdays
    python -m tools.alpaca.fills report
"""
from __future__ import annotations

import asyncio
import math
import statistics
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from research import pit_store, scoring
from tools.alpaca.async_broker import AsyncBroker, AsyncBrokerError

HORIZON_DAYS = scoring.DEFAULT_HORIZON_DAYS
SUBMIT_JOB = "fill_submit"
COLLECT_JOB = "fill_collect"

_ENTRY_SIDE = {"long": "buy", "short": "sell"}
_EXIT_SIDE = {"long": "sell", "short": "buy"}

NOTE = ("Paper fills are simulated from quotes: they measure the quoted spread "
        "at the open, not the price impact of the order itself. Only small "
        "live orders measure impact.")


def client_order_id(order_as_of: str, ticker: str, leg: str) -> str:
    """Deterministic, so a second run cannot send a leg twice: the store
    refuses the row and the broker refuses the id."""
    return f"nemo-{order_as_of}-{ticker}-{leg}"


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _stamp(today: str) -> str:
    return f"{today}T{datetime.now(timezone.utc).strftime('%H:%M:%S')}Z"


def _last_close(ticker: str, as_of: str) -> Optional[float]:
    """The close the scan could see when it decided, to size whole shares."""
    bars = pit_store.bars_as_of(ticker, as_of)
    last = next((b for b in reversed(bars) if b.get("close")), None)
    return last["close"] if last else None


def plan(orders: List[Dict[str, Any]], recorded: Dict[tuple, Dict[str, Any]],
         calendar: List[str], today: str, horizon_days: int,
         last_close: Callable[[str, str], Optional[float]] = _last_close
         ) -> Tuple[List[Dict[str, Any]], int]:
    """The legs due at today's open, and how many entries were missed.

    An entry is due on the first trading day on or after the session the scan
    named, which is how scoring rolls a holiday. One the job did not send that
    morning is not sent late: entered a day later it is a different trade from
    the one the book scores. An exit is due `horizon_days` trading days after
    its entry filled; one sent late is still sent, to flatten the position, and
    is marked so it is not measured.
    """
    legs: List[Dict[str, Any]] = []
    missed = 0
    for order in orders:
        key = (order["as_of_date"], order["ticker"])
        side = order.get("side") or "long"
        entry = recorded.get((*key, "entry"))

        if entry is None:
            session = order.get("intended_session")
            if not session:
                continue
            scheduled = next((d for d in calendar if d >= session), None)
            if scheduled is None or scheduled > today:
                continue
            if scheduled < today:
                missed += 1
                continue
            target = order.get("target_dollars") or 0.0
            price = last_close(order["ticker"], order["as_of_date"])
            qty = math.floor(target / price) if price else 0
            reason = None
            if not price:
                reason = "no close on record to size it in whole shares"
            elif qty <= 0:
                reason = (f"a target of {target:,.0f} is below one share at "
                          f"{price:,.2f}; the opening auction takes whole "
                          f"shares only")
            legs.append({"order_as_of": key[0], "ticker": key[1],
                         "leg": "entry", "scheduled_session": today,
                         "session": today, "side": _ENTRY_SIDE[side],
                         "qty": qty, "reason": reason,
                         "client_order_id": client_order_id(*key, "entry")})
            continue

        if recorded.get((*key, "exit")) is not None:
            continue
        # Exited once the entry has its final answer and holds shares. An
        # auction order that part-filled and had the rest cancelled still
        # left a position, and a position nobody exits is a different book.
        if (entry["status"] not in pit_store.FILL_TERMINAL
                or not entry.get("filled_qty")):
            continue
        if entry["session"] not in calendar or today not in calendar:
            continue
        start = calendar.index(entry["session"])
        elapsed = calendar.index(today) - start
        if elapsed < horizon_days:
            continue
        late = elapsed - horizon_days
        legs.append({"order_as_of": key[0], "ticker": key[1], "leg": "exit",
                     "scheduled_session": calendar[start + horizon_days],
                     "session": today, "side": _EXIT_SIDE[side],
                     "qty": int(entry["filled_qty"]),
                     "reason": (f"sent {late} session(s) late to flatten the "
                                f"position; not measured" if late else None),
                     "client_order_id": client_order_id(*key, "exit")})
    return legs, missed


async def submit(broker, today: Optional[str] = None,
                 horizon_days: int = HORIZON_DAYS) -> Dict[str, Any]:
    """Send today's due legs to the opening auction and record each one."""
    today = today or _today()
    run_id = pit_store.start_run(SUBMIT_JOB, as_of_date=today)
    out = {"as_of": today, "status": "ok", "submitted": 0, "skipped": 0,
           "rejected": 0, "missed": 0}
    try:
        orders = pit_store.paper_orders_as_of(today, accepted_only=True)
        recorded = {(r["order_as_of"], r["ticker"], r["leg"]): r
                    for r in pit_store.paper_fills_as_of(today)}
        sessions = ([o["intended_session"] for o in orders
                     if o.get("intended_session")]
                    + [r["session"] for r in recorded.values()])
        if not orders:
            pit_store.finish_run(rows_written=0, status="ok",
                                 error="no filed orders", run_id=run_id)
            return out

        calendar = sorted(await broker.get_calendar(min(sessions + [today]),
                                                    today))
        if today not in calendar:
            out["status"] = "closed"
            pit_store.finish_run(rows_written=0, status="closed",
                                 error=f"the exchange is shut on {today}",
                                 run_id=run_id)
            return out

        legs, out["missed"] = plan(orders, recorded, calendar, today,
                                   horizon_days)
        for leg in legs:
            row = {**leg, "broker_order_id": None,
                   "submitted_at": _stamp(today)}
            if leg["qty"] <= 0:
                pit_store.record_fill_submission({**row, "status": "skipped"})
                out["skipped"] += 1
                continue
            try:
                sent = await broker.submit_market_order(
                    leg["ticker"], leg["qty"], leg["side"],
                    client_order_id=leg["client_order_id"],
                    time_in_force="opg")
            except AsyncBrokerError as exc:
                pit_store.record_fill_submission(
                    {**row, "status": "rejected", "reason": str(exc)[:300]})
                out["rejected"] += 1
                continue
            pit_store.record_fill_submission(
                {**row, "status": sent.get("status") or "submitted",
                 "broker_order_id": sent.get("id")})
            out["submitted"] += 1

        out["status"] = "partial" if out["rejected"] else "ok"
        notes = [f"{out[k]} {k}" for k in ("rejected", "skipped", "missed")
                 if out[k]]
        pit_store.finish_run(
            rows_written=out["submitted"] + out["skipped"] + out["rejected"],
            status=out["status"], error="; ".join(notes) or None,
            run_id=run_id)
        return out
    except Exception as exc:
        pit_store.finish_run(rows_written=0, status="failed",
                             error=f"{type(exc).__name__}: {exc}"[:300],
                             run_id=run_id)
        raise


async def collect(broker, today: Optional[str] = None) -> Dict[str, Any]:
    """Ask the broker about every leg not yet at a terminal answer."""
    today = today or _today()
    run_id = pit_store.start_run(COLLECT_JOB, as_of_date=today)
    try:
        open_legs = [r for r in pit_store.paper_fills_as_of(today)
                     if r["status"] not in pit_store.FILL_TERMINAL
                     and r["broker_order_id"]]
        updated = 0
        for row in open_legs:
            order = await broker.get_order_by_id(row["broker_order_id"])
            price = order.get("filled_avg_price")
            updated += pit_store.update_paper_fill(
                row["client_order_id"], order.get("status") or "unknown",
                float(order.get("filled_qty") or 0),
                float(price) if price else None, order.get("filled_at"))
        pit_store.finish_run(rows_written=updated, status="ok",
                             error=None if open_legs else "nothing open",
                             run_id=run_id)
        return {"as_of": today, "status": "ok", "asked": len(open_legs),
                "updated": updated}
    except Exception as exc:
        pit_store.finish_run(rows_written=0, status="failed",
                             error=f"{type(exc).__name__}: {exc}"[:300],
                             run_id=run_id)
        raise


def _open_on(ticker: str, session: str, as_of: str) -> Optional[float]:
    bar = next((b for b in pit_store.bars_as_of(ticker, as_of)
                if b["trade_date"] == session), None)
    return bar["open"] if bar and bar.get("open") else None


def report(as_of: Optional[str] = None) -> Dict[str, Any]:
    """Measured round-trip cost against the modeled one, per trade and overall.

    Cost is what the fills paid relative to the opens the scoring uses: bought
    above the open or sold below it is a positive cost, on each leg.
    """
    as_of = as_of or _today()
    trips, unpriced = [], 0
    for trip in pit_store.measured_round_trips(as_of):
        entry_open = _open_on(trip["ticker"], trip["entry_session"], as_of)
        exit_open = _open_on(trip["ticker"], trip["exit_session"], as_of)
        if not entry_open or not exit_open:
            unpriced += 1
            continue
        sign = -1.0 if trip["side"] == "short" else 1.0
        entry_cost = sign * (trip["entry_price"] / entry_open - 1.0) * 10_000
        exit_cost = -sign * (trip["exit_price"] / exit_open - 1.0) * 10_000
        measured = entry_cost + exit_cost
        modeled = trip.get("cost_bps") or 0.0
        trips.append({**trip, "measured_cost_bps": measured,
                      "modeled_cost_bps": modeled,
                      "excess_bps": measured - modeled})

    out = {"as_of": as_of, "round_trips": len(trips),
           "unpriced": unpriced, "note": NOTE,
           "mean_measured_cost_bps": None, "mean_modeled_cost_bps": None,
           "mean_excess_bps": None, "median_excess_bps": None,
           "trips": trips}
    if trips:
        out.update({
            "mean_measured_cost_bps": statistics.fmean(
                t["measured_cost_bps"] for t in trips),
            "mean_modeled_cost_bps": statistics.fmean(
                t["modeled_cost_bps"] for t in trips),
            "mean_excess_bps": statistics.fmean(t["excess_bps"] for t in trips),
            "median_excess_bps": statistics.median(
                t["excess_bps"] for t in trips)})
    return out


def main(argv=None) -> int:
    import argparse
    import json

    pit_store.init_schema()
    parser = argparse.ArgumentParser(
        prog="fills", description="Paper fills against the filed book.")
    parser.add_argument("action", choices=("submit", "collect", "report"))
    parser.add_argument("--as-of", dest="as_of", default=None,
                        help="date to act as (default: today, UTC)")
    parser.add_argument("--horizon-days", dest="horizon_days", type=int,
                        default=HORIZON_DAYS)
    args = parser.parse_args(argv)

    if args.action == "report":
        print(json.dumps(report(args.as_of), indent=2, default=str))
        return 0

    async def run():
        # Paper, always. The live broker needs a different pair of keys and
        # this job never asks for it.
        async with AsyncBroker(paper=True) as broker:
            if args.action == "submit":
                return await submit(broker, args.as_of, args.horizon_days)
            return await collect(broker, args.as_of)

    try:
        result = asyncio.run(run())
    except Exception as exc:  # noqa: BLE001 - reported, then non-zero
        print(json.dumps({"action": args.action, "status": "failed",
                          "error": f"{type(exc).__name__}: {exc}"}, indent=2))
        return 1
    print(json.dumps(result, indent=2, default=str))
    return 0 if result["status"] in ("ok", "closed") else 1


if __name__ == "__main__":  # pragma: no cover - exercised via main()
    raise SystemExit(main())
