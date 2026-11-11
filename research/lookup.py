"""What the store holds for the Discord bot's /orders and /ticker.

Read only, and every read goes through the store's point-in-time readers, so
a reply says what Nemo knew on the date it reads as of.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Optional

from research import pit_store

ORDERS_SHOWN = 3
RELEASES_SHOWN = 3
FILINGS_SHOWN = 3
CONGRESS_SHOWN = 3

_SYMBOL = re.compile(r"[A-Z][A-Z0-9.\-]{0,9}")


def symbol(raw: str) -> str:
    """A ticker in capitals, or ValueError naming what was given."""
    text = (raw or "").strip().upper()
    if not _SYMBOL.fullmatch(text):
        raise ValueError(f"{(raw or '').strip()!r} is not a ticker")
    return text


def _legs(table: str, as_of: str) -> Dict[tuple, Dict[str, Any]]:
    """Each leg's latest attempt, by (order date, ticker, leg)."""
    latest: Dict[tuple, Dict[str, Any]] = {}
    for row in pit_store.fills_as_of(as_of, table=table):
        key = (row["order_as_of"], row["ticker"], row["leg"])
        if key not in latest or row["attempt"] > latest[key]["attempt"]:
            latest[key] = row
    return latest


def _earnings(as_of: str) -> Optional[Dict[str, Any]]:
    accepted = pit_store.paper_orders_as_of(as_of, accepted_only=True)
    if not accepted:
        return None
    decided = max(row["as_of_date"] for row in accepted)
    rows = [row for row in accepted if row["as_of_date"] == decided]
    legs = _legs("paper_fill", as_of)
    session = rows[0]["intended_session"]
    exits = sorted((leg for (_, _, kind), leg in legs.items()
                    if kind == "exit" and leg["session"] == session),
                   key=lambda leg: leg["ticker"])
    return {"decided": decided, "session": session,
            "orders": [{**row, "entry": legs.get(
                (decided, row["ticker"], "entry"))} for row in rows],
            "exits": exits}


def _copy(as_of: str) -> Optional[Dict[str, Any]]:
    decisions = pit_store.copy_orders_as_of(as_of)
    if not decisions:
        return None
    decided = max(row["as_of_date"] for row in decisions)
    rows = [row for row in decisions if row["as_of_date"] == decided]
    legs = _legs("copy_fill", as_of)
    return {"decided": decided, "seen": len(rows),
            "orders": [{**row, "entry": legs.get(
                (decided, row["ticker"], "entry"))}
                for row in rows if row["accepted"]],
            "refused": dict(Counter(row["reason"] for row in rows
                                    if not row["accepted"]))}


def orders(as_of: str) -> Dict[str, Any]:
    """Each book's latest decision on or before `as_of`, with its fills.

    The earnings book also carries the exits sent for the session its new
    orders are for. A book with no decision yet is None.
    """
    return {"as_of": as_of, "earnings": _earnings(as_of),
            "copy": _copy(as_of)}


def ticker(name: str, as_of: str) -> Dict[str, Any]:
    """Everything the store holds on one ticker, newest first.

    `signal` is the latest scan's row for it, traded or not, because the
    reason a name was not traded is the answer more often than an order is.
    """
    decisions = pit_store.paper_orders_as_of(as_of, ticker=name)
    legs = _legs("paper_fill", as_of)
    traded = [row for row in reversed(decisions) if row["accepted"]]
    return {
        "symbol": name, "as_of": as_of,
        "signal": decisions[-1] if decisions else None,
        "orders": [{**row,
                    "entry": legs.get((row["as_of_date"], name, "entry")),
                    "exit": legs.get((row["as_of_date"], name, "exit"))}
                   for row in traded[:ORDERS_SHOWN]],
        "releases": list(reversed(
            pit_store.announcements_as_of(name, as_of)))[:RELEASES_SHOWN],
        "filings": pit_store.activist_filings_as_of(
            as_of, ticker=name)[:FILINGS_SHOWN],
        "congress": [row for row in reversed(pit_store.copy_orders_as_of(as_of))
                     if row["ticker"] == name][:CONGRESS_SHOWN],
    }
