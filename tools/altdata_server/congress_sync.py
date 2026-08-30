"""Ingestion for congressional disclosures: index, fetch, parse, store.

Runs are resumable by construction. The index is cheap and lists everything;
the expensive part is one HTTP round trip and a PDF parse per filing, so the
store is asked which filings it still needs and only those are fetched. A
second run over an unchanged year fetches nothing.

Two failures are kept apart, because conflating them is what makes a pipeline
quietly wrong:

* A **scan** of a paper filing has no extractable text and never will. It is
  recorded once and never retried, or every future run would spend its whole
  budget re-reading filings that cannot be parsed.
* A **failed fetch** is transient. It is recorded, and offered again next run.

Neither is dropped. A filing absent from the store and a filing that could not
be read look identical to a query and mean opposite things.

Usage:
    python -m tools.altdata_server.congress_sync --house           # this year and last
    python -m tools.altdata_server.congress_sync --house 2026
    python -m tools.altdata_server.congress_sync --house 2024 2025 2026
    python -m tools.altdata_server.congress_sync --senate --days 90
    python -m tools.altdata_server.congress_sync --senate-annual
    python -m tools.altdata_server.congress_sync --status
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import requests

from . import congress_store as store
from .congress_trades import (
    DisclosureBlocked,
    DisclosureUnavailable,
    _throttle,
    fetch_house_index,
    fetch_house_ptr,
    fetch_senate_ptr,
    search_senate_ptrs,
    senate_session,
)
from .senate_annual import (
    fetch_senate_annual,
    latest_amendments,
    search_senate_annuals,
)

# How many refusals in a row before the run stops asking. A source that has
# turned down five consecutive requests is not going to answer the sixth, and
# working through the rest of the budget deepens the block while reporting it
# as N filings that happened to fail.
MAX_CONSECUTIVE_FAILURES = 5


def _is_scan(exc: Exception) -> bool:
    return "no extractable text" in str(exc)


def _source_has_stopped_answering(exc: Exception, consecutive: int) -> bool:
    """Whether this is the source refusing rather than one filing failing."""
    return (isinstance(exc, DisclosureBlocked)
            or consecutive >= MAX_CONSECUTIVE_FAILURES)


def _house_docid_is_paper(doc_id: str) -> bool:
    """A 7-digit House DocID is a scan of a paper filing.

    Length is the discriminator, not the leading digit. Measured over 1,330
    House filings in the store: every one of the 166 seven-digit ids was a
    scan and every one of the 1,164 eight-digit ids parsed, with no
    exceptions either way. An earlier version of this also required the id to
    begin with 9, which is how the published descriptions put it -- that
    caught 51 of the 166 and quietly downloaded the other 115 to discover the
    same thing the slow way.

    Recognising a scan from the index costs nothing; fetching it costs a
    request and several megabytes -- one runs to 353 pages of images -- for an
    identical answer. Paper was 6.8% of 2025 annual filings and 30.3% of
    2014's, so this matters more the further back a backfill reaches.
    """
    return len((doc_id or "").strip()) == 7


def _iso_from_us(text: str) -> Optional[str]:
    for fmt in ("%m/%d/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(text.strip(), fmt).date().isoformat()
        except (ValueError, AttributeError):
            continue
    return None


def _log(message: str, quiet: bool = False) -> None:
    if not quiet:
        print(message, file=sys.stderr, flush=True)


# -------------------------------------------------------------------- House

def sync_house_ptrs(year: int, max_filings: Optional[int] = None,
                    quiet: bool = False,
                    recheck_days: Optional[int] = None) -> Dict[str, Any]:
    """Ingest every House Periodic Transaction Report filed in `year`.

    An index failure propagates rather than returning an empty run: a year in
    which the Clerk was unreachable must never be recorded as a year in which
    nobody filed.
    """
    store.init_schema()
    index = fetch_house_index(year)          # raises DisclosureUnavailable

    ptrs = [f for f in index if f.get("filing_type") == "P"]
    by_id: Dict[str, Dict[str, Any]] = {}
    for filing in ptrs:
        member = store.member_id("house", filing["last"], filing["first"],
                                 (filing.get("state_district") or "")[:2])
        store.upsert_member({
            "member_id": member, "chamber": "house",
            "first": filing["first"], "last": filing["last"],
            "full_name": f"{filing['first']} {filing['last']}".strip(),
            "state": (filing.get("state_district") or "")[:2],
            "district": filing.get("state_district"), "office": None,
            "first_seen": _iso_from_us(filing.get("filing_date", "")),
            "last_seen": _iso_from_us(filing.get("filing_date", "")),
        })
        by_id[f"house:{filing['doc_id']}"] = {**filing, "member_id": member}

    # What the Clerk's index says today. A filing whose index row has moved
    # was re-posted under the same DocID, and the store is holding the
    # numbers it superseded.
    published = {fid: _iso_from_us(f.get("filing_date", ""))
                 for fid, f in by_id.items()}
    pending = store.unparsed_filing_ids(list(by_id), index_filed_dates=published,
                                        recheck_days=recheck_days)
    already_held = len(by_id) - len(pending)
    budget = pending if max_filings is None else pending[:max_filings]
    remaining = len(pending) - len(budget)

    parsed = failed = scanned = 0
    consecutive = 0
    blocked: Optional[str] = None
    for position, filing_id in enumerate(budget):
        filing = by_id[filing_id]
        record = {
            "filing_id": filing_id, "chamber": "house",
            "doc_id": filing["doc_id"], "member_id": filing["member_id"],
            "filing_type": "ptr", "raw_filing_type": filing.get("filing_type"),
            "filed_date": _iso_from_us(filing.get("filing_date", "")),
            "year": year,
            "source_url": (f"https://disclosures-clerk.house.gov/public_disc/"
                           f"ptr-pdfs/{year}/{filing['doc_id']}.pdf"),
        }
        if _house_docid_is_paper(filing["doc_id"]):
            record["parse_status"] = "scanned"
            record["parse_error"] = (
                "filed on paper; the PDF is page images with no extractable "
                "text (identified from the DocID, not downloaded)")
            store.upsert_filing(record)
            scanned += 1
            continue

        _throttle()
        try:
            parsed_filing = fetch_house_ptr(filing["doc_id"], year)
        except DisclosureUnavailable as exc:
            record["parse_status"] = "scanned" if _is_scan(exc) else "error"
            record["parse_error"] = str(exc)[:400]
            store.upsert_filing(record)
            scanned += _is_scan(exc)
            failed += not _is_scan(exc)
            # A scan is a document this parser cannot read; the Clerk answered
            # perfectly well. Only a refusal counts towards giving up.
            consecutive = 0 if _is_scan(exc) else consecutive + 1
            if _source_has_stopped_answering(exc, consecutive):
                blocked = str(exc)[:400]
                remaining += len(budget) - position - 1
                break
            continue

        if (not parsed_filing["transactions"]
                and not parsed_filing.get("no_reportable_transactions")):
            # `fetch_house_ptr` refuses these, but this is where 'parsed'
            # becomes permanent, so the rule is enforced here rather than
            # trusted to whatever produced the filing.
            record["parse_status"] = "error"
            record["parse_error"] = (
                "parsed to zero transactions without stating it has none; a "
                "PTR is filed in order to report a trade, so the table was "
                "not read")
            store.upsert_filing(record)
            failed += 1
            continue

        record["parse_status"] = store.PARSED
        record["parsed_at"] = datetime.now(timezone.utc).isoformat(
            timespec="seconds")
        record["content_hash"] = parsed_filing.get("content_hash")
        # Status and rows in one transaction: a filing durably `parsed` before
        # its rows exist is never offered for reading again.
        store.record_parsed_filing(record,
                                   transactions=parsed_filing["transactions"])
        consecutive = 0
        parsed += 1

    store.record_sync(f"house_ptr_{year}", filings_seen=len(by_id),
                      filings_parsed=parsed, filings_failed=failed + scanned,
                      cursor=str(year))
    _log(f"[house {year}] seen={len(by_id)} held={already_held} "
         f"parsed={parsed} scans={scanned} errors={failed} "
         f"remaining={remaining}"
         + (f" BLOCKED: {blocked}" if blocked else ""), quiet)

    return {"chamber": "house", "year": year, "filings_seen": len(by_id),
            "already_held": already_held, "filings_parsed": parsed,
            "scans": scanned, "errors": failed, "remaining": remaining,
            "blocked": blocked, "complete": remaining == 0 and blocked is None}


# ------------------------------------------------------------------- Senate

def sync_senate_ptrs(days: int = 90, max_filings: Optional[int] = None,
                     quiet: bool = False,
                     recheck_days: Optional[int] = None) -> Dict[str, Any]:
    """Ingest Senate PTRs filed in the last `days`."""
    store.init_schema()
    since = (datetime.now() - timedelta(days=days)).strftime("%m/%d/%Y")

    session = senate_session()               # raises DisclosureUnavailable
    # No limit: the search pages at 100 a time and the walk must reach
    # recordsTotal, or the ingest silently sees only the newest page.
    found = search_senate_ptrs(session, since)

    by_id: Dict[str, Dict[str, Any]] = {}
    for filing in found:
        if not filing.get("uuid"):
            continue
        member = store.member_id("senate", filing["last"], filing["first"])
        store.upsert_member({
            "member_id": member, "chamber": "senate",
            "first": filing["first"], "last": filing["last"],
            "full_name": f"{filing['first']} {filing['last']}".strip(),
            "state": None, "district": None, "office": filing.get("office"),
            "first_seen": _iso_from_us(filing.get("filed_date", "")),
            "last_seen": _iso_from_us(filing.get("filed_date", "")),
        })
        by_id[f"senate:{filing['uuid']}"] = {**filing, "member_id": member}

    published = {fid: _iso_from_us(f.get("filed_date", ""))
                 for fid, f in by_id.items()}
    pending = store.unparsed_filing_ids(list(by_id), index_filed_dates=published,
                                        recheck_days=recheck_days)
    already_held = len(by_id) - len(pending)
    budget = pending if max_filings is None else pending[:max_filings]
    remaining = len(pending) - len(budget)

    parsed = failed = scanned = 0
    consecutive = 0
    blocked: Optional[str] = None
    for position, filing_id in enumerate(budget):
        filing = by_id[filing_id]
        record = {
            "filing_id": filing_id, "chamber": "senate",
            "doc_id": filing["uuid"], "member_id": filing["member_id"],
            "filing_type": "ptr", "raw_filing_type": filing.get("kind"),
            "filed_date": _iso_from_us(filing.get("filed_date", "")),
            "year": None,
            "source_url": (f"https://efdsearch.senate.gov/search/view/ptr/"
                           f"{filing['uuid']}/"),
        }
        # A paper filing is a scan behind a different route and carries no
        # table to read.
        if filing.get("kind") != "ptr":
            record["parse_status"] = "scanned"
            record["parse_error"] = f"filed on paper ({filing.get('kind')})"
            store.upsert_filing(record)
            scanned += 1
            continue

        _throttle()
        try:
            # Raises rather than returning [] for the agreement page, an
            # expired session, a refusal, or a report with no rows in it --
            # all of which eFD serves with status 200.
            report = fetch_senate_ptr(session, filing["uuid"])
        except DisclosureUnavailable as exc:
            record["parse_status"] = "error"
            record["parse_error"] = str(exc)[:400]
            store.upsert_filing(record)
            failed += 1
            consecutive += 1
            if _source_has_stopped_answering(exc, consecutive):
                blocked = str(exc)[:400]
                remaining += len(budget) - position - 1
                break
            continue

        record["parse_status"] = store.PARSED
        record["parsed_at"] = datetime.now(timezone.utc).isoformat(
            timespec="seconds")
        record["content_hash"] = report.get("content_hash")
        store.record_parsed_filing(record, transactions=report["transactions"])
        consecutive = 0
        parsed += 1

    store.record_sync("senate_ptr", filings_seen=len(by_id),
                      filings_parsed=parsed, filings_failed=failed + scanned,
                      cursor=since)
    _log(f"[senate {days}d] seen={len(by_id)} held={already_held} "
         f"parsed={parsed} scans={scanned} errors={failed} "
         f"remaining={remaining}"
         + (f" BLOCKED: {blocked}" if blocked else ""), quiet)

    return {"chamber": "senate", "days": days, "filings_seen": len(by_id),
            "already_held": already_held, "filings_parsed": parsed,
            "scans": scanned, "errors": failed, "remaining": remaining,
            "blocked": blocked, "complete": remaining == 0 and blocked is None}


# ---------------------------------------------------------------------- CLI

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ingest congressional STOCK Act disclosures into the "
                    "local store. Safe to re-run: nothing already parsed is "
                    "fetched twice.")
    parser.add_argument("--house", nargs="*", type=int, metavar="YEAR",
                        help="House PTR years to ingest, e.g. --house 2025 2026. "
                             "With no years: the current calendar year and the "
                             "one before it")
    parser.add_argument("--senate", action="store_true",
                        help="Ingest recent Senate PTRs (trades)")
    parser.add_argument("--senate-annual", action="store_true",
                        help="Ingest Senate annual reports (holdings)")
    parser.add_argument("--since", default="01/01/2026", metavar="MM/DD/YYYY",
                        help="Earliest submission date for annual reports")
    parser.add_argument("--days", type=int, default=90,
                        help="Senate window in days (default 90)")
    parser.add_argument("--max-filings", type=int, default=None,
                        help="Cap filings fetched per source this run")
    parser.add_argument("--recheck-days", type=int, default=None, metavar="N",
                        help="Also re-read filings last fetched more than N "
                             "days ago, so a correction re-posted under the "
                             "same id is eventually noticed")
    parser.add_argument("--status", action="store_true",
                        help="Print what the store holds and exit")
    args = parser.parse_args(argv)

    store.init_schema()

    if args.status:
        overall = store.coverage()
        print(f"database: {store.current_db_path()}")
        print(f"filings:  {overall['total']} "
              f"({overall['parsed']} parsed, {overall['unparsed']} unparsed)")
        for status, count in sorted(overall["by_status"].items()):
            print(f"   {status:10} {count}")
        for chamber in ("house", "senate"):
            print(f"   {chamber}: {store.coverage(chamber)}")
        with store.connect() as conn:
            for table in ("members", "transactions", "holdings"):
                count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                print(f"{table+':':10} {count}")
        return 0

    # `--house` with no years, which is what the nightly cron line runs. A
    # literal year here is a pipeline with an expiry date: the Clerk's PTR
    # archives are per calendar year, so a hard-coded 2026 stops fetching on
    # 2027-01-01 while coverage() goes on reporting complete -- every filing
    # the job knows about was parsed, and it no longer knows about any.
    #
    # The prior year is asked for as well, not out of caution. A PTR for a
    # December trade is filed in January and lands in the *previous* year's
    # archive, so a run that asked only for the current year would miss every
    # January carry-over. Re-reading a year already ingested costs one index
    # fetch: nothing already parsed is fetched twice.
    #
    # `None` is the flag absent; `[]` is the flag present with no years. Only
    # the second gets a default, so `--senate` alone still means the Senate
    # alone.
    if args.house == []:
        this_year = datetime.now(timezone.utc).year
        args.house = [this_year - 1, this_year]

    if not args.house and not args.senate and not args.senate_annual:
        parser.error("nothing to do: pass --house YEAR..., --senate, "
                     "--senate-annual, or --status")

    # Before anything is fetched, because a transaction report recorded as
    # read with no rows in it is a parse that failed, and nothing else would
    # ever offer it again. Requeued here, this run re-reads it.
    requeued = store.requeue_empty_transaction_reports()
    if requeued:
        _log(f"[repair] requeued {requeued} transaction report(s) recorded as "
             f"parsed with no transactions in them")

    failures = blocked = 0
    for year in args.house or []:
        try:
            result = sync_house_ptrs(year, max_filings=args.max_filings,
                                     recheck_days=args.recheck_days)
            blocked += bool(result["blocked"])
        except DisclosureUnavailable as exc:
            print(f"[house {year}] FAILED: {exc}", file=sys.stderr)
            failures += 1

    if args.senate:
        try:
            result = sync_senate_ptrs(days=args.days,
                                      max_filings=args.max_filings,
                                      recheck_days=args.recheck_days)
            blocked += bool(result["blocked"])
        except DisclosureUnavailable as exc:
            print(f"[senate] FAILED: {exc}", file=sys.stderr)
            failures += 1

    if args.senate_annual:
        try:
            result = sync_senate_annuals(since=args.since,
                                         max_filings=args.max_filings,
                                         recheck_days=args.recheck_days)
            blocked += bool(result["blocked"])
        except DisclosureUnavailable as exc:
            print(f"[senate annual] FAILED: {exc}", file=sys.stderr)
            failures += 1

    # Idempotent and cheap. The normalised key stops new duplicates forming,
    # but a run that ingested filings written under an older spelling of a
    # name would otherwise leave that person split in two.
    merged = store.merge_duplicate_members()
    if merged:
        _log(f"[members] merged {merged} duplicate record(s)")

    # Also idempotent and cheap. replace_transactions refuses impossible rows
    # on the way in, but rows ingested before it did are still held, and a
    # parser fixed later does not go back and correct what it already wrote.
    repaired = store.repair_impossible_rows()
    if repaired["amounts_cleared"] or repaired["dates_cleared"]:
        _log(f"[repair] cleared {repaired['amounts_cleared']} impossible "
             f"amount range(s) and {repaired['dates_cleared']} trade date(s) "
             f"that fell after their own filing")

    # A blocked source is a non-zero exit too: the run did not cover what it
    # was asked to cover, and a caller reading only the status code would
    # otherwise treat a refusal as a completed backfill.
    return 1 if failures or blocked else 0




# ------------------------------------------------- Senate annual (holdings)

def sync_senate_annuals(since: str = "01/01/2026",
                        max_filings: Optional[int] = None,
                        quiet: bool = False,
                        recheck_days: Optional[int] = None) -> Dict[str, Any]:
    """Ingest Senate annual reports -- the holdings, not the trades.

    Only the highest amendment per (senator, calendar year) is fetched.
    Amendments are full restatements rather than deltas, so ingesting a base
    report alongside the amendment that replaced it would double every
    unchanged holding and keep figures the filer has since corrected.

    Paper filings are recorded as scans without being fetched: they are
    galleries of GIF page images, and no parser reads them.
    """
    store.init_schema()
    session = senate_session()               # raises DisclosureUnavailable
    found = search_senate_annuals(session, since)

    electronic = [f for f in found if f.get("kind") == "annual" and f.get("uuid")]
    paper = [f for f in found if f.get("kind") != "annual" and f.get("uuid")]
    wanted = latest_amendments(electronic)

    by_id: Dict[str, Dict[str, Any]] = {}
    for filing in wanted + paper:
        member = store.member_id("senate", filing["last"], filing["first"])
        store.upsert_member({
            "member_id": member, "chamber": "senate",
            "first": filing["first"], "last": filing["last"],
            "full_name": f"{filing['first']} {filing['last']}".strip(),
            "state": None, "district": None, "office": filing.get("office"),
            "first_seen": _iso_from_us(filing.get("filed_date", "")),
            "last_seen": _iso_from_us(filing.get("filed_date", "")),
        })
        by_id[f"senate:annual:{filing['uuid']}"] = {**filing, "member_id": member}

    published = {fid: _iso_from_us(f.get("filed_date", ""))
                 for fid, f in by_id.items()}
    pending = store.unparsed_filing_ids(list(by_id), index_filed_dates=published,
                                        recheck_days=recheck_days)
    already_held = len(by_id) - len(pending)
    budget = pending if max_filings is None else pending[:max_filings]
    remaining = len(pending) - len(budget)

    parsed = failed = scanned = 0
    consecutive = 0
    blocked: Optional[str] = None
    for position, filing_id in enumerate(budget):
        filing = by_id[filing_id]
        year = filing.get("calendar_year")
        record = {
            "filing_id": filing_id, "chamber": "senate",
            "doc_id": filing["uuid"], "member_id": filing["member_id"],
            "filing_type": "annual", "raw_filing_type": filing.get("kind"),
            "filed_date": _iso_from_us(filing.get("filed_date", "")),
            "year": year,
            "source_url": (f"https://efdsearch.senate.gov/search/view/"
                           f"{filing.get('kind')}/{filing['uuid']}/"),
        }
        if filing.get("kind") != "annual":
            record["parse_status"] = "scanned"
            record["parse_error"] = "filed on paper; the report is page images"
            store.upsert_filing(record)
            scanned += 1
            continue

        _throttle()
        try:
            report = fetch_senate_annual(session, filing["uuid"])
        except DisclosureUnavailable as exc:
            record["parse_status"] = "error"
            record["parse_error"] = str(exc)[:400]
            store.upsert_filing(record)
            failed += 1
            consecutive += 1
            if _source_has_stopped_answering(exc, consecutive):
                blocked = str(exc)[:400]
                remaining += len(budget) - position - 1
                break
            continue

        # The parser decides both, because only it has seen the heading.
        # `report_types=[7]` is an umbrella and New Filer, Candidate and
        # Termination reports arrive through it: they carry real holdings but
        # no calendar year, and dating them to a year end would invent a
        # period they never covered.
        record["filing_type"] = report.get("report_kind") or "annual"
        record["year"] = report.get("calendar_year") or year
        record["parse_status"] = store.PARSED
        record["parsed_at"] = datetime.now(timezone.utc).isoformat(
            timespec="seconds")
        record["content_hash"] = report.get("content_hash")

        # Last resort: the date it was filed. A holding with no as-of at all
        # cannot be aged against the trades that came after it, and an
        # approximate date carries that caveat where an empty one does not.
        as_of = report.get("as_of") or record["filed_date"]
        for row in report["holdings"]:
            row["as_of"] = row.get("as_of") or as_of
        store.record_parsed_filing(record, holdings=report["holdings"])
        consecutive = 0
        parsed += 1

    store.record_sync("senate_annual", filings_seen=len(by_id),
                      filings_parsed=parsed, filings_failed=failed + scanned,
                      cursor=since)
    _log(f"[senate annual] seen={len(by_id)} held={already_held} "
         f"parsed={parsed} scans={scanned} errors={failed} "
         f"remaining={remaining}"
         + (f" BLOCKED: {blocked}" if blocked else ""), quiet)

    return {"chamber": "senate", "kind": "annual", "filings_seen": len(by_id),
            "already_held": already_held, "filings_parsed": parsed,
            "scans": scanned, "errors": failed, "remaining": remaining,
            "blocked": blocked, "complete": remaining == 0 and blocked is None}


if __name__ == "__main__":
    raise SystemExit(main())
