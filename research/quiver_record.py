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
    # Price and value join the volatile fields because, for recent trades,
    # Quiver fills them from the market: two fetches minutes apart disagreed on
    # 10 of 1,733 rows (measured 2026-11-12). The first copy is kept.
    Feed("insider_trade", "get_insider_trades", "trades",
         identity=("owner", "ticker", "transaction_date", "shares", "tx_type"),
         date_field="filed_date",
         volatile=("list", "net_worth_estimate", "price_per_share",
                   "value_usd")),
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


def iso_week(now: datetime) -> str:
    year, week, _ = now.isocalendar()
    return f"{year}-W{week:02d}"


def _fetch(client: Any, feed: Feed) -> List[Dict[str, Any]]:
    result = getattr(client, feed.method)()
    rows = result.get(feed.rows_key) if isinstance(result, dict) else None
    if not isinstance(rows, list):
        raise QuiverError(
            "parse_failed",
            f"{feed.method} returned no {feed.rows_key!r} list")
    if not rows:
        raise QuiverError(
            "empty",
            f"{feed.method} parsed 0 {feed.rows_key}. An empty feed is a "
            f"failure, not a night with nothing new.")
    return rows


def _reason(exc: Exception) -> str:
    if isinstance(exc, QuiverError):
        return f"{exc.reason}: {exc.message}"
    return f"{type(exc).__name__}: {exc}"


def _stage_status(failures: Dict[str, str], succeeded: int) -> str:
    if not failures:
        return "ok"
    return "partial" if succeeded else "failed"


def _finish(run_id: int, written: int, status: str,
            failures: Dict[str, str]) -> None:
    error = "; ".join(f"{name}: {reason}" for name, reason in failures.items())
    pit_store.finish_run(rows_written=written, status=status,
                         error=error or None, run_id=run_id)


def record_feeds(client: Any, now: datetime) -> Dict[str, Any]:
    """The seven feeds into quiver_record. Each fails alone, by name."""
    stamp = pit_store.iso_utc(now)
    run_id = pit_store.start_run(FEEDS_STAGE, as_of_date=now.date().isoformat())
    counts: Dict[str, Dict[str, int]] = {}
    failures: Dict[str, str] = {}
    for feed in FEEDS:
        try:
            rows = keyed_rows(feed, _fetch(client, feed))
            new, revised = pit_store.record_quiver_rows(
                feed.dataset, rows, volatile=feed.volatile, recorded_at=stamp)
        except Exception as exc:  # noqa: BLE001 - named in the run and the exit
            failures[feed.dataset] = _reason(exc)
            continue
        counts[feed.dataset] = {"read": len(rows), "new": new,
                                "revised": revised}
    written = sum(c["new"] + c["revised"] for c in counts.values())
    status = _stage_status(failures, len(counts))
    _finish(run_id, written, status, failures)
    return {"status": status, "written": written, "feeds": counts,
            "failed": failures}


def record_lists(client: Any, now: datetime) -> Dict[str, Any]:
    """The weekly lists into quiver_snapshot, once per ISO week.

    Checked every night, so a night whose fetch failed is retried by the next
    one instead of leaving the week empty.
    """
    week = iso_week(now)
    stamp = pit_store.iso_utc(now)
    run_id = pit_store.start_run(LISTS_STAGE, as_of_date=now.date().isoformat())
    counts: Dict[str, Dict[str, int]] = {}
    held: List[str] = []
    failures: Dict[str, str] = {}
    for feed in LISTS:
        if pit_store.has_quiver_snapshot(feed.dataset, week):
            held.append(feed.dataset)
            continue
        try:
            rows = keyed_rows(feed, _fetch(client, feed))
            written = pit_store.record_quiver_snapshot(
                feed.dataset, week, rows, recorded_at=stamp)
        except Exception as exc:  # noqa: BLE001 - named in the run and the exit
            failures[feed.dataset] = _reason(exc)
            continue
        counts[feed.dataset] = {"read": len(rows), "written": written}
    written = sum(c["written"] for c in counts.values())
    status = _stage_status(failures, len(counts) + len(held))
    _finish(run_id, written, status, failures)
    return {"status": status, "week": week, "written": written,
            "lists": counts, "already_held": held, "failed": failures}


def run(client: Any = None, now: Optional[datetime] = None) -> Dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    pit_store.init_schema()
    owned = client is None
    client = client or QuiverWebClient()
    try:
        feeds = record_feeds(client, now)
        lists = record_lists(client, now)
    finally:
        if owned:
            client.close()
    failed = {f"{stage}/{name}": reason
              for stage, result in ((FEEDS_STAGE, feeds), (LISTS_STAGE, lists))
              for name, reason in result["failed"].items()}
    statuses = {feeds["status"], lists["status"]}
    if statuses == {"ok"}:
        status = "ok"
    elif statuses == {"failed"}:
        status = "failed"
    else:
        status = "partial"
    return {"status": status, "as_of": now.date().isoformat(),
            "written": feeds["written"] + lists["written"],
            FEEDS_STAGE: feeds, LISTS_STAGE: lists, "failed": failed}


def main(argv: Optional[Sequence[str]] = None) -> int:
    argparse.ArgumentParser(
        description="Record QuiverQuant's published feeds into pit.db."
    ).parse_args(argv)
    from research import discord_post
    elapsed = discord_post.stopwatch()
    pit_store.init_schema()
    try:
        result = run()
    except Exception as exc:  # noqa: BLE001 - reported and exits non-zero
        traceback.print_exc()
        result = {"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                  "written": 0, "failed": {}}
    print(json.dumps(result, indent=2))
    code = 0 if result["status"] == "ok" else 1
    return discord_post.finish(JOB, code, result, seconds=elapsed())


if __name__ == "__main__":
    raise SystemExit(main())
