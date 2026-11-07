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

Issue #190 adds the congressional copy book beside the earnings book in the
same run. Copy legs use their own client ids and land in `copy_fill`, so the
earnings gate and risk book never see them. The decision job still sends no
order; this job is what sends the copy book's legs.

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
from research.congress_copy import Settings as CopySettings
from tools.alpaca.async_broker import AsyncBroker, AsyncBrokerError

HORIZON_DAYS = scoring.DEFAULT_HORIZON_DAYS
SUBMIT_JOB = "fill_submit"
COLLECT_JOB = "fill_collect"

_ENTRY_SIDE = {"long": "buy", "short": "sell"}
_EXIT_SIDE = {"long": "sell", "short": "buy"}

NOTE = ("Paper fills are simulated from quotes: they measure the quoted spread "
        "at the open, not the price impact of the order itself. Only small "
        "live orders measure impact.")


def client_order_id(order_as_of: str, ticker: str, leg: str,
                    attempt: int = 1, book: str = "earnings") -> str:
    """Deterministic per attempt, so a second run cannot send the same
    attempt twice: the store refuses the row and the broker refuses the id.

    Attempt 1 keeps the original id shape so rows already at the broker stay
    addressable; later remainder exits append the attempt number (issue #189).
    The copy book's prefix is `nemo-copy-` so its ids can never collide with
    the earnings book's `nemo-` ids (issue #190).
    """
    if book == "copy":
        base = f"nemo-copy-{order_as_of}-{ticker}-{leg}"
    elif book == "earnings":
        base = f"nemo-{order_as_of}-{ticker}-{leg}"
    else:
        raise ValueError(f"unknown fill book: {book!r}")
    if attempt <= 1:
        return base
    return f"{base}-{attempt}"


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _stamp(today: str) -> str:
    return f"{today}T{datetime.now(timezone.utc).strftime('%H:%M:%S')}Z"


def _last_close(ticker: str, as_of: str) -> Optional[float]:
    """The close the scan could see when it decided, to size whole shares."""
    bars = pit_store.bars_as_of(ticker, as_of)
    last = next((b for b in reversed(bars) if b.get("close")), None)
    return last["close"] if last else None


def _held_for_exit(position: Optional[Dict[str, Any]], exit_side: str) -> float:
    """Shares the broker holds that an exit of `exit_side` would close.

    A sell flattens a long; a buy flattens a short. The opposite side, or no
    position, is zero held — never treat that as a reason to open a short.
    """
    if not position:
        return 0.0
    side = (position.get("side") or "long").lower()
    qty = float(position.get("qty") or 0)
    if exit_side == "sell" and side == "long":
        return qty
    if exit_side == "buy" and side == "short":
        return qty
    return 0.0


def _exit_filled_total(exits: List[Dict[str, Any]]) -> float:
    """Shares already closed across every terminal exit attempt."""
    total = 0.0
    for row in exits:
        if row["status"] in pit_store.FILL_TERMINAL:
            total += float(row.get("filled_qty") or 0)
    return total


def _empty_book() -> Dict[str, Any]:
    return {"status": "ok", "submitted": 0, "skipped": 0,
            "rejected": 0, "missed": 0}


def plan(orders: List[Dict[str, Any]], fills: List[Dict[str, Any]],
         calendar: List[str], today: str, horizon_days: int,
         last_close: Callable[[str, str], Optional[float]] = _last_close,
         book: str = "earnings"
         ) -> Tuple[List[Dict[str, Any]], int]:
    """The legs due at today's open, and how many entries were missed.

    An entry is due on the first trading day on or after the session the scan
    named, which is how scoring rolls a holiday. One the job did not send that
    morning is not sent late: entered a day later it is a different trade from
    the one the book scores. An exit is due `horizon_days` trading days after
    its entry filled; one sent late is still sent, to flatten the position, and
    is marked so it is not measured.

    An exit that expired unfilled or part-filled still holds shares. The next
    session sends the remainder (entry filled minus every exit fill so far),
    as a new attempt marked late and not measured, until the sum of exit fills
    equals the entry (issue #189).
    """
    by_key: Dict[tuple, Dict[str, Any]] = {}
    for row in fills:
        slot = by_key.setdefault(
            (row["order_as_of"], row["ticker"]),
            {"entry": None, "exits": []})
        if row["leg"] == "entry":
            slot["entry"] = row
        else:
            slot["exits"].append(row)

    legs: List[Dict[str, Any]] = []
    missed = 0
    for order in orders:
        key = (order["as_of_date"], order["ticker"])
        side = order.get("side") or "long"
        slot = by_key.get(key, {"entry": None, "exits": []})
        entry = slot["entry"]
        exits = slot["exits"]

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
                         "leg": "entry", "attempt": 1,
                         "scheduled_session": today,
                         "session": today, "side": _ENTRY_SIDE[side],
                         "qty": qty, "reason": reason,
                         "client_order_id": client_order_id(
                             *key, "entry", book=book)})
            continue

        # Exited once the entry has its final answer and holds shares. An
        # auction order that part-filled and had the rest cancelled still
        # left a position, and a position nobody exits is a different book.
        # A prior exit that expired the same way still holds a remainder, so
        # the check is on shares closed, not on whether an exit row exists.
        if (entry["status"] not in pit_store.FILL_TERMINAL
                or not entry.get("filled_qty")):
            continue
        if any(e["status"] not in pit_store.FILL_TERMINAL for e in exits):
            continue
        entry_qty = float(entry["filled_qty"])
        remainder = int(entry_qty - _exit_filled_total(exits))
        if remainder <= 0:
            continue
        if entry["session"] not in calendar or today not in calendar:
            continue
        start = calendar.index(entry["session"])
        elapsed = calendar.index(today) - start
        if elapsed < horizon_days:
            continue
        late = elapsed - horizon_days
        attempt = max((int(e.get("attempt") or 1) for e in exits), default=0) + 1
        # First exit on the due session is on-time and measured; every later
        # attempt, and any first exit after the due session, is a flatten.
        reason = None
        if late or attempt > 1:
            reason = (f"sent {late} session(s) late to flatten the "
                      f"position; not measured")
        legs.append({"order_as_of": key[0], "ticker": key[1], "leg": "exit",
                     "attempt": attempt,
                     "scheduled_session": calendar[start + horizon_days],
                     "session": today, "side": _EXIT_SIDE[side],
                     "qty": remainder, "reason": reason,
                     "client_order_id": client_order_id(
                         *key, "exit", attempt=attempt, book=book)})
    return legs, missed


async def _submit_legs(broker, legs: List[Dict[str, Any]], today: str,
                       table: str) -> Dict[str, Any]:
    """Send one book's planned legs and record each one in `table`."""
    out = _empty_book()
    for leg in legs:
        row = {**leg, "broker_order_id": None,
               "submitted_at": _stamp(today)}
        if leg["qty"] <= 0:
            pit_store.record_fill_submission(
                {**row, "status": "skipped"}, table=table)
            out["skipped"] += 1
            continue
        # A remainder exit must not sell shares another trade (or a
        # leftover like the 2.5 F) is holding. Ask the broker first; a
        # lookup error is a refusal, never a silent "not held".
        if leg["leg"] == "exit" and int(leg.get("attempt") or 1) > 1:
            try:
                position = await broker.get_open_position(leg["ticker"])
            except AsyncBrokerError as exc:
                pit_store.record_fill_submission(
                    {**row, "status": "rejected",
                     "reason": str(exc)[:300]}, table=table)
                out["rejected"] += 1
                continue
            held = _held_for_exit(position, leg["side"])
            if held < leg["qty"]:
                pit_store.record_fill_submission({
                    **row, "status": "skipped",
                    "reason": (f"broker holds {held:g} shares; need "
                               f"{leg['qty']} to flatten this trade "
                               f"without opening a short")}, table=table)
                out["skipped"] += 1
                continue
        try:
            sent = await broker.submit_market_order(
                leg["ticker"], leg["qty"], leg["side"],
                client_order_id=leg["client_order_id"],
                time_in_force="opg")
        except AsyncBrokerError as exc:
            pit_store.record_fill_submission(
                {**row, "status": "rejected",
                 "reason": str(exc)[:300]}, table=table)
            out["rejected"] += 1
            continue
        pit_store.record_fill_submission(
            {**row, "status": sent.get("status") or "submitted",
             "broker_order_id": sent.get("id")}, table=table)
        out["submitted"] += 1
    out["status"] = "partial" if out["rejected"] else "ok"
    return out


async def _collect_legs(broker, today: str, table: str) -> Dict[str, Any]:
    """Ask the broker about every open leg in `table`."""
    open_legs = [r for r in pit_store.fills_as_of(today, table=table)
                 if r["status"] not in pit_store.FILL_TERMINAL
                 and r["broker_order_id"]]
    updated = 0
    for row in open_legs:
        order = await broker.get_order_by_id(row["broker_order_id"])
        price = order.get("filled_avg_price")
        updated += pit_store.update_paper_fill(
            row["client_order_id"], order.get("status") or "unknown",
            float(order.get("filled_qty") or 0),
            float(price) if price else None, order.get("filled_at"),
            table=table)
    return {"status": "ok", "asked": len(open_legs), "updated": updated}


def _copy_orders_for_submit(today: str, fills: List[Dict[str, Any]],
                            settings: CopySettings) -> List[Dict[str, Any]]:
    """Accepted copy decisions due today. Off still exits held positions."""
    orders = [o for o in pit_store.copy_orders_as_of(today) if o.get("accepted")]
    if settings.enabled:
        return orders
    # Turning the flag off must never leave shares nobody exits: keep only
    # orders that already have an entry leg on file, so plan can flatten them
    # and cannot open a new name.
    held = {(r["order_as_of"], r["ticker"]) for r in fills if r["leg"] == "entry"}
    return [o for o in orders if (o["as_of_date"], o["ticker"]) in held]


async def submit(broker, today: Optional[str] = None,
                 horizon_days: int = HORIZON_DAYS) -> Dict[str, Any]:
    """Send today's due legs for both books to the opening auction."""
    today = today or _today()
    run_id = pit_store.start_run(SUBMIT_JOB, as_of_date=today)
    out: Dict[str, Any] = {
        "as_of": today, "status": "ok", "submitted": 0, "skipped": 0,
        "rejected": 0, "missed": 0, "copy": _empty_book()}
    # The same notes the run log keeps, so Discord can say what went wrong.
    notes: List[str] = []
    out["notes"] = notes
    try:
        earn_orders = pit_store.paper_orders_as_of(today, accepted_only=True)
        earn_fills = pit_store.paper_fills_as_of(today)
        # The copy book's inputs are read inside its own guard and raised again
        # in the copy section below: a typo in a copy setting fails that book by
        # name, and must not stop the earnings legs.
        copy_load_error: Optional[Exception] = None
        copy_settings: Optional[CopySettings] = None
        copy_fills: List[Dict[str, Any]] = []
        copy_orders: List[Dict[str, Any]] = []
        try:
            copy_settings = CopySettings.from_env()
            copy_fills = pit_store.fills_as_of(today, table="copy_fill")
            copy_orders = _copy_orders_for_submit(today, copy_fills,
                                                  copy_settings)
        except Exception as exc:  # noqa: BLE001 - re-raised in the copy section
            copy_load_error = exc

        if not earn_orders and not copy_orders and copy_load_error is None:
            pit_store.finish_run(rows_written=0, status="ok",
                                 error="no filed orders", run_id=run_id)
            return out

        sessions = ([o["intended_session"] for o in earn_orders + copy_orders
                     if o.get("intended_session")]
                    + [r["session"] for r in earn_fills + copy_fills])
        calendar = sorted(await broker.get_calendar(min(sessions + [today]),
                                                    today))
        if today not in calendar:
            out["status"] = "closed"
            out["copy"]["status"] = "closed"
            pit_store.finish_run(rows_written=0, status="closed",
                                 error=f"the exchange is shut on {today}",
                                 run_id=run_id)
            return out

        # Each book is its own try: a failure in one must still send the other
        # book's legs, and the run names the failure rather than swallow it.
        try:
            legs, out["missed"] = plan(
                earn_orders, earn_fills, calendar, today, horizon_days,
                book="earnings")
            earn = await _submit_legs(broker, legs, today, table="paper_fill")
            for key in ("submitted", "skipped", "rejected", "status"):
                out[key] = earn[key]
            if earn["rejected"]:
                notes.append(f"{earn['rejected']} rejected")
            if earn["skipped"]:
                notes.append(f"{earn['skipped']} skipped")
            if out["missed"]:
                notes.append(f"{out['missed']} missed")
        except Exception as exc:
            out["status"] = "partial"
            msg = f"earnings: {type(exc).__name__}: {exc}"[:300]
            notes.append(msg)
            out["error"] = msg

        try:
            if copy_load_error is not None:
                raise copy_load_error
            copy_legs, copy_missed = plan(
                copy_orders, copy_fills, calendar, today,
                copy_settings.hold_days, book="copy")
            copy = await _submit_legs(
                broker, copy_legs, today, table="copy_fill")
            copy["missed"] = copy_missed
            out["copy"] = copy
            copy_bits = [
                f"{copy[k]} {k}" for k in
                ("submitted", "rejected", "skipped", "missed") if copy.get(k)]
            if copy_bits or copy["status"] != "ok":
                notes.append("copy: " + ("; ".join(copy_bits) or copy["status"]))
            if copy["status"] == "partial":
                out["status"] = "partial"
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"[:300]
            out["copy"] = {**_empty_book(), "status": "failed", "error": msg}
            out["status"] = "partial"
            notes.append(f"copy: {msg}")

        rows = (out["submitted"] + out["skipped"] + out["rejected"]
                + out["copy"]["submitted"] + out["copy"]["skipped"]
                + out["copy"]["rejected"])
        pit_store.finish_run(
            rows_written=rows, status=out["status"],
            error="; ".join(notes) or None, run_id=run_id)
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
    out: Dict[str, Any] = {
        "as_of": today, "status": "ok", "asked": 0, "updated": 0,
        "copy": {"status": "ok", "asked": 0, "updated": 0}}
    notes: List[str] = []
    out["notes"] = notes
    try:
        try:
            earn = await _collect_legs(broker, today, table="paper_fill")
            out["asked"] = earn["asked"]
            out["updated"] = earn["updated"]
        except Exception as exc:
            msg = f"earnings: {type(exc).__name__}: {exc}"[:300]
            out["status"] = "partial"
            out["error"] = msg
            notes.append(msg)

        try:
            copy = await _collect_legs(broker, today, table="copy_fill")
            out["copy"] = copy
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"[:300]
            out["copy"] = {"status": "failed", "asked": 0, "updated": 0,
                           "error": msg}
            out["status"] = "partial"
            notes.append(f"copy: {msg}")

        if out["status"] == "ok" and not out["asked"] and not out["copy"]["asked"]:
            notes.append("nothing open")
        pit_store.finish_run(
            rows_written=out["updated"] + out["copy"]["updated"],
            status=out["status"],
            error="; ".join(notes) or None, run_id=run_id)
        return out
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
    above the open or sold below it is a positive cost, on each leg. The copy
    book is reported beside that measurement and never mixed into it.
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
           "trips": trips,
           "copy": pit_store.copy_book_summary(as_of)}
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

    from research import discord_post
    try:
        result = asyncio.run(run())
    except Exception as exc:  # noqa: BLE001 - reported, then non-zero
        error = f"{type(exc).__name__}: {exc}"
        print(json.dumps({"action": args.action, "status": "failed",
                          "error": error}, indent=2))
        return discord_post.finish(
            "research-fills", 1,
            {"action": args.action, "status": "failed", "error": error})
    print(json.dumps(result, indent=2, default=str))
    code = 0 if result["status"] in ("ok", "closed") else 1
    return discord_post.finish(
        "research-fills", code, {**result, "action": args.action})


if __name__ == "__main__":  # pragma: no cover - exercised via main()
    raise SystemExit(main())
