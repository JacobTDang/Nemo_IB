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
    python -m tools.altdata_server.congress_sync --house 2026
    python -m tools.altdata_server.congress_sync --house 2024 2025 2026
    python -m tools.altdata_server.congress_sync --senate --days 90
    python -m tools.altdata_server.congress_sync --senate-annual
    python -m tools.altdata_server.congress_sync --status
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import requests

from . import congress_store as store
from .congress_trades import (
    DisclosureUnavailable,
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

# The House Clerk and the Senate publish no rate limit. This is deliberate
# politeness rather than a measured ceiling: the whole backfill is a few
# hundred requests and there is nothing to gain by going faster.
_REQUEST_INTERVAL_S = 0.8
_last_request = 0.0


def _throttle() -> None:
    global _last_request
    wait = _REQUEST_INTERVAL_S - (time.monotonic() - _last_request)
    if wait > 0:
        time.sleep(wait)
    _last_request = time.monotonic()


def _is_scan(exc: Exception) -> bool:
    return "no extractable text" in str(exc)


def _house_docid_is_paper(doc_id: str) -> bool:
    """A 7-digit House DocID beginning with 9 is a scan of a paper filing.

    Electronic filings carry an 8-digit id beginning with 1. Recognising a
    scan from the index costs nothing; downloading it to discover the same
    thing costs a request and several megabytes -- Khanna's runs to 353 pages
    of images -- and the answer is identical either way. Paper was 6.8% of
    2025 annual filings and 30.3% of 2014's, so this matters more the further
    back a backfill reaches.
    """
    doc_id = (doc_id or "").strip()
    return len(doc_id) == 7 and doc_id.startswith("9")


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
                    quiet: bool = False) -> Dict[str, Any]:
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

    pending = store.unparsed_filing_ids(list(by_id))
    already_held = len(by_id) - len(pending)
    budget = pending if max_filings is None else pending[:max_filings]
    remaining = len(pending) - len(budget)

    parsed = failed = scanned = 0
    for filing_id in budget:
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
            continue

        record["parse_status"] = store.PARSED
        record["parsed_at"] = datetime.now(timezone.utc).isoformat(
            timespec="seconds")
        store.upsert_filing(record)
        store.replace_transactions(filing_id, filing["member_id"],
                                   parsed_filing["transactions"])
        parsed += 1

    store.record_sync(f"house_ptr_{year}", filings_seen=len(by_id),
                      filings_parsed=parsed, filings_failed=failed + scanned,
                      cursor=str(year))
    _log(f"[house {year}] seen={len(by_id)} held={already_held} "
         f"parsed={parsed} scans={scanned} errors={failed} "
         f"remaining={remaining}", quiet)

    return {"chamber": "house", "year": year, "filings_seen": len(by_id),
            "already_held": already_held, "filings_parsed": parsed,
            "scans": scanned, "errors": failed, "remaining": remaining,
            "complete": remaining == 0}


# ------------------------------------------------------------------- Senate

def sync_senate_ptrs(days: int = 90, max_filings: Optional[int] = None,
                     quiet: bool = False) -> Dict[str, Any]:
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

    pending = store.unparsed_filing_ids(list(by_id))
    already_held = len(by_id) - len(pending)
    budget = pending if max_filings is None else pending[:max_filings]
    remaining = len(pending) - len(budget)

    parsed = failed = scanned = 0
    for filing_id in budget:
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
            rows = fetch_senate_ptr(session, filing["uuid"])
        except DisclosureUnavailable as exc:
            record["parse_status"] = "error"
            record["parse_error"] = str(exc)[:400]
            store.upsert_filing(record)
            failed += 1
            continue

        record["parse_status"] = store.PARSED
        record["parsed_at"] = datetime.now(timezone.utc).isoformat(
            timespec="seconds")
        store.upsert_filing(record)
        store.replace_transactions(filing_id, filing["member_id"], rows)
        parsed += 1

    store.record_sync("senate_ptr", filings_seen=len(by_id),
                      filings_parsed=parsed, filings_failed=failed + scanned,
                      cursor=since)
    _log(f"[senate {days}d] seen={len(by_id)} held={already_held} "
         f"parsed={parsed} scans={scanned} errors={failed} "
         f"remaining={remaining}", quiet)

    return {"chamber": "senate", "days": days, "filings_seen": len(by_id),
            "already_held": already_held, "filings_parsed": parsed,
            "scans": scanned, "errors": failed, "remaining": remaining,
            "complete": remaining == 0}


# ---------------------------------------------------------------------- CLI

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ingest congressional STOCK Act disclosures into the "
                    "local store. Safe to re-run: nothing already parsed is "
                    "fetched twice.")
    parser.add_argument("--house", nargs="*", type=int, metavar="YEAR",
                        help="House PTR years to ingest, e.g. --house 2025 2026")
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

    if not args.house and not args.senate and not args.senate_annual:
        parser.error("nothing to do: pass --house YEAR..., --senate, "
                     "--senate-annual, or --status")

    failures = 0
    for year in args.house or []:
        try:
            sync_house_ptrs(year, max_filings=args.max_filings)
        except DisclosureUnavailable as exc:
            print(f"[house {year}] FAILED: {exc}", file=sys.stderr)
            failures += 1

    if args.senate:
        try:
            sync_senate_ptrs(days=args.days, max_filings=args.max_filings)
        except DisclosureUnavailable as exc:
            print(f"[senate] FAILED: {exc}", file=sys.stderr)
            failures += 1

    if args.senate_annual:
        try:
            sync_senate_annuals(since=args.since, max_filings=args.max_filings)
        except DisclosureUnavailable as exc:
            print(f"[senate annual] FAILED: {exc}", file=sys.stderr)
            failures += 1

    return 1 if failures else 0




# ------------------------------------------------- Senate annual (holdings)

def sync_senate_annuals(since: str = "01/01/2026",
                        max_filings: Optional[int] = None,
                        quiet: bool = False) -> Dict[str, Any]:
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

    pending = store.unparsed_filing_ids(list(by_id))
    already_held = len(by_id) - len(pending)
    budget = pending if max_filings is None else pending[:max_filings]
    remaining = len(pending) - len(budget)

    parsed = failed = scanned = 0
    for filing_id in budget:
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
        store.upsert_filing(record)

        # Last resort: the date it was filed. A holding with no as-of at all
        # cannot be aged against the trades that came after it, and an
        # approximate date carries that caveat where an empty one does not.
        as_of = report.get("as_of") or record["filed_date"]
        for row in report["holdings"]:
            row["as_of"] = row.get("as_of") or as_of
        store.replace_holdings(filing_id, filing["member_id"], report["holdings"])
        parsed += 1

    store.record_sync("senate_annual", filings_seen=len(by_id),
                      filings_parsed=parsed, filings_failed=failed + scanned,
                      cursor=since)
    _log(f"[senate annual] seen={len(by_id)} held={already_held} "
         f"parsed={parsed} scans={scanned} errors={failed} "
         f"remaining={remaining}", quiet)

    return {"chamber": "senate", "kind": "annual", "filings_seen": len(by_id),
            "already_held": already_held, "filings_parsed": parsed,
            "scans": scanned, "errors": failed, "remaining": remaining,
            "complete": remaining == 0}


if __name__ == "__main__":
    raise SystemExit(main())
