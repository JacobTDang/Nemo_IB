"""A nightly record of what QuiverQuant publishes.

Quiver's pages publish rolling windows: the latest 300 congressional trades,
the latest 100 contract awards, about 2,000 WARN notices. A row that leaves the
window is gone from the source, and nothing else records when it first
appeared. This job writes each feed into pit.db every night and three
slow-moving lists once per ISO week. It never overwrites a row.

Each feed fails alone and by name. A feed that parses no rows is a failure,
never a quiet night: "nothing new" means rows were read and none were new.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from research import pit_store
from tools.quiver_server.client import QuiverError, QuiverWebClient

JOB = "research-quiver"
FEEDS_STAGE = "quiver_feeds"
LISTS_STAGE = "quiver_lists"


@dataclass(frozen=True)
class Feed:
    dataset: str
    method: str                       # QuiverWebClient method, called with no arguments
    rows_key: str                     # the list of rows in that method's result
    identity: Tuple[str, ...] = ()    # fields hashed into row_key
    id_field: Optional[str] = None    # or Quiver's own id, used as row_key
    ticker_field: Optional[str] = "ticker"
    date_field: Optional[str] = None
    volatile: Tuple[str, ...] = ()    # stored, ignored when comparing


# Identity fields leave out anything Quiver may fill in later (a ticker match,
# its quality), so filling it in is a revision of the row and not a new row.
FEEDS: Tuple[Feed, ...] = (
    Feed("congress_trade", "get_congress_trades", "trades",
         id_field="tx_id", date_field="filed_date"),
    # The fields the client already deduplicates its two insider lists on.
    Feed("insider_trade", "get_insider_trades", "trades",
         identity=("owner", "ticker", "transaction_date", "shares", "tx_type"),
         date_field="filed_date",
         volatile=("list", "net_worth_estimate")),
    Feed("layoff", "get_layoffs", "notices",
         identity=("employer", "notice_date", "city", "state", "workers",
                   "kind"),
         date_field="notice_date"),
    Feed("contract_award", "get_gov_contract_awards", "awards",
         identity=("recipient", "date_awarded", "amount", "agency",
                   "description"),
         ticker_field=None, date_field="date_awarded"),
    Feed("gov_spending", "get_gov_spending", "transactions",
         identity=("award_id", "date", "amount_usd", "recipient", "agency"),
         date_field="date"),
    Feed("lobbying_filing", "get_lobbying_filings", "filings",
         identity=("ticker", "amount", "date"), date_field="date"),
    Feed("revolving_door", "get_revolving_door", "registrations",
         identity=("lobbyist", "client", "firm", "filed_date", "filing_type",
                   "period"),
         date_field="filed_date", volatile=("is_new",)),
)

LISTS: Tuple[Feed, ...] = (
    Feed("congress_net_worth", "get_congress_net_worth", "members",
         identity=("bioguide_id", "name"), ticker_field=None),
    Feed("election_contribution", "get_election_contributions",
         "contributions", identity=("ticker", "company", "source", "cycle")),
    Feed("lobbying_top_spender", "get_lobbying_filings", "top_spenders",
         identity=("ticker",)),
)


def keyed_rows(feed: Feed, rows: Sequence[Dict[str, Any]]
               ) -> List[Dict[str, Any]]:
    """Each row with its key, ticker and event date, ready for pit_store.

    A hashed key carries an occurrence index: the n-th row in one fetch with
    the same identity gets n in its hash, so two identical lobbying filings
    stay two rows. The rows are identical, so the index is stable across
    nights whichever of them Quiver lists first.
    """
    seen: Dict[str, int] = {}
    out: List[Dict[str, Any]] = []
    for row in rows:
        if feed.id_field:
            raw = row.get(feed.id_field)
            if not raw:
                raise ValueError(
                    f"{feed.dataset}: a row has no {feed.id_field}, so it "
                    f"cannot be told apart from any other: {row!r}")
            key = str(raw)
        else:
            identity = json.dumps([row.get(f) for f in feed.identity],
                                  default=str)
            occurrence = seen.get(identity, 0)
            seen[identity] = occurrence + 1
            key = hashlib.sha256(
                f"{identity}#{occurrence}".encode()).hexdigest()[:32]
        ticker = row.get(feed.ticker_field) if feed.ticker_field else None
        out.append({
            "row_key": key,
            "ticker": str(ticker).upper() if ticker else None,
            "event_date": row.get(feed.date_field) if feed.date_field else None,
            "payload": row,
        })
    return out
