"""Local store for congressional STOCK Act disclosures.

Fetching one House PTR costs an HTTP round trip and a PDF parse. On demand
that bought roughly twenty filings per call, so every answer carried partial
coverage and "no NVDA trades" never meant anything. Parsing each filing once
into this store is what makes complete coverage affordable.

Which puts the burden on the store being honest about what it holds. A filing
nobody has fetched and a filing that turned out to be a scan of a paper form
look identical from a query -- both simply absent -- and they mean opposite
things. Every filing is therefore recorded with a `parse_status`, including
the ones that could not be read, and `coverage()` reports the split.

The two failure modes are kept apart on purpose. A scan will not become
readable on a retry, so requeueing it forever would spend an entire sync
budget on filings that cannot be parsed. A fetch that failed is transient and
is retried.

This database is rebuildable from the public record, so it lives apart from
the book state in `state/schema.py`. It is not disposable, though: the
container's `db_cache` is a RAM-backed tmpfs, and re-parsing thousands of PDFs
on every restart is not a pipeline. Point `NEMO_CONGRESS_DB` at a real volume.
"""
from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
_DEFAULT_DB_PATH = os.path.join(_REPO_ROOT, "db_cache", "congress.db")

# A scan will never parse; retrying it spends the budget on nothing. A failed
# fetch is transient and is offered again.
PERMANENTLY_UNREADABLE = ("scanned",)
PARSED = "parsed"


def current_db_path() -> str:
    """Resolve the path on every call, never at import.

    A default argument is evaluated once, at function-definition time, so
    `def connect(path=DB_PATH)` freezes whatever the environment held when this
    module first loaded and silently ignores every later override -- which
    surfaces as an empty result rather than an error.
    """
    return os.environ.get("NEMO_CONGRESS_DB", _DEFAULT_DB_PATH)


CREATE_SCHEMA = [
    """CREATE TABLE IF NOT EXISTS members(
        member_id    TEXT PRIMARY KEY,
        chamber      TEXT NOT NULL,
        first        TEXT,
        last         TEXT,
        full_name    TEXT,
        state        TEXT,
        district     TEXT,
        office       TEXT,
        first_seen   TEXT,
        last_seen    TEXT
    )""",

    # One row per disclosure document, readable or not.
    """CREATE TABLE IF NOT EXISTS filings(
        filing_id        TEXT PRIMARY KEY,
        chamber          TEXT NOT NULL,
        doc_id           TEXT NOT NULL,
        member_id        TEXT,
        filing_type      TEXT,          -- ptr | annual | amendment | other
        raw_filing_type  TEXT,          -- the chamber's own code
        filed_date       TEXT,
        year             INTEGER,
        source_url       TEXT,
        parse_status     TEXT NOT NULL, -- parsed | scanned | error | pending
        parse_error      TEXT,
        transaction_count INTEGER DEFAULT 0,
        holding_count    INTEGER DEFAULT 0,
        fetched_at       TEXT,
        parsed_at        TEXT
    )""",

    """CREATE TABLE IF NOT EXISTS transactions(
        txn_id            TEXT PRIMARY KEY,
        filing_id         TEXT NOT NULL,
        member_id         TEXT,
        row_index         INTEGER,
        ticker            TEXT,
        cusip             TEXT,
        asset_name        TEXT,
        asset_type_code   TEXT,
        owner             TEXT,
        transaction_type  TEXT,
        transaction_date  TEXT,
        notification_date TEXT,
        amount_min        INTEGER,
        amount_max        INTEGER
    )""",

    # Holdings come from annual reports and are a snapshot, so each carries the
    # date it was true. Without that a stale position cannot be aged.
    """CREATE TABLE IF NOT EXISTS holdings(
        holding_id       TEXT PRIMARY KEY,
        filing_id        TEXT NOT NULL,
        member_id        TEXT,
        row_index        INTEGER,
        ticker           TEXT,
        cusip            TEXT,
        asset_name       TEXT,
        asset_type_code  TEXT,
        owner            TEXT,
        value_min        INTEGER,
        value_max        INTEGER,
        income_min       INTEGER,
        income_max       INTEGER,
        income_type      TEXT,
        as_of            TEXT
    )""",

    """CREATE TABLE IF NOT EXISTS sync_state(
        source          TEXT PRIMARY KEY,
        last_synced_at  TEXT,
        last_cursor     TEXT,
        filings_seen    INTEGER,
        filings_parsed  INTEGER,
        filings_failed  INTEGER
    )""",

    "CREATE INDEX IF NOT EXISTS idx_txn_ticker ON transactions(ticker)",
    "CREATE INDEX IF NOT EXISTS idx_txn_member ON transactions(member_id)",
    "CREATE INDEX IF NOT EXISTS idx_txn_date ON transactions(transaction_date)",
    "CREATE INDEX IF NOT EXISTS idx_txn_filing ON transactions(filing_id)",
    "CREATE INDEX IF NOT EXISTS idx_hold_ticker ON holdings(ticker)",
    "CREATE INDEX IF NOT EXISTS idx_hold_member ON holdings(member_id)",
    "CREATE INDEX IF NOT EXISTS idx_hold_filing ON holdings(filing_id)",
    "CREATE INDEX IF NOT EXISTS idx_filings_status ON filings(parse_status)",
    "CREATE INDEX IF NOT EXISTS idx_filings_member ON filings(member_id)",
]


@contextmanager
def connect():
    """A connection to the store, committing on clean exit."""
    path = current_db_path()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_schema() -> None:
    with connect() as conn:
        for statement in CREATE_SCHEMA:
            conn.execute(statement)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------------ members

def member_id(chamber: str, last: str, first: str, state: str = "") -> str:
    """A stable key. Names repeat across chambers and across states."""
    parts = [chamber.lower(), (last or "").strip().lower(),
             (first or "").strip().lower(), (state or "").strip().upper()]
    return ":".join(p.replace(" ", "_") for p in parts)


def upsert_member(member: Dict[str, Any]) -> None:
    with connect() as conn:
        conn.execute(
            """INSERT INTO members(member_id, chamber, first, last, full_name,
                                   state, district, office, first_seen, last_seen)
               VALUES(:member_id, :chamber, :first, :last, :full_name,
                      :state, :district, :office, :first_seen, :last_seen)
               ON CONFLICT(member_id) DO UPDATE SET
                 full_name = COALESCE(excluded.full_name, members.full_name),
                 office    = COALESCE(excluded.office, members.office),
                 district  = COALESCE(excluded.district, members.district),
                 last_seen = MAX(COALESCE(excluded.last_seen, ''),
                                 COALESCE(members.last_seen, ''))""",
            {"member_id": member["member_id"], "chamber": member["chamber"],
             "first": member.get("first"), "last": member.get("last"),
             "full_name": member.get("full_name"), "state": member.get("state"),
             "district": member.get("district"), "office": member.get("office"),
             "first_seen": member.get("first_seen"),
             "last_seen": member.get("last_seen")})


# ------------------------------------------------------------------ filings

def upsert_filing(filing: Dict[str, Any]) -> None:
    payload = {
        "filing_id": filing["filing_id"], "chamber": filing["chamber"],
        "doc_id": filing["doc_id"], "member_id": filing.get("member_id"),
        "filing_type": filing.get("filing_type"),
        "raw_filing_type": filing.get("raw_filing_type"),
        "filed_date": filing.get("filed_date"), "year": filing.get("year"),
        "source_url": filing.get("source_url"),
        "parse_status": filing.get("parse_status", "pending"),
        "parse_error": filing.get("parse_error"),
        "transaction_count": filing.get("transaction_count", 0),
        "holding_count": filing.get("holding_count", 0),
        "fetched_at": filing.get("fetched_at") or _now(),
        "parsed_at": filing.get("parsed_at"),
    }
    with connect() as conn:
        conn.execute(
            """INSERT INTO filings(filing_id, chamber, doc_id, member_id,
                    filing_type, raw_filing_type, filed_date, year, source_url,
                    parse_status, parse_error, transaction_count,
                    holding_count, fetched_at, parsed_at)
               VALUES(:filing_id, :chamber, :doc_id, :member_id, :filing_type,
                      :raw_filing_type, :filed_date, :year, :source_url,
                      :parse_status, :parse_error, :transaction_count,
                      :holding_count, :fetched_at, :parsed_at)
               ON CONFLICT(filing_id) DO UPDATE SET
                 member_id        = excluded.member_id,
                 filing_type      = excluded.filing_type,
                 raw_filing_type  = excluded.raw_filing_type,
                 filed_date       = excluded.filed_date,
                 year             = excluded.year,
                 source_url       = excluded.source_url,
                 parse_status     = excluded.parse_status,
                 parse_error      = excluded.parse_error,
                 transaction_count= excluded.transaction_count,
                 holding_count    = excluded.holding_count,
                 parsed_at        = excluded.parsed_at""", payload)


def unparsed_filing_ids(candidates: Sequence[str]) -> List[str]:
    """Which of `candidates` still need fetching.

    A filing already parsed is skipped. So is one recorded as a scan: it will
    not become readable on a retry, and requeueing it every run would spend
    the whole budget on filings that cannot be parsed. A failed fetch is
    transient and is offered again.
    """
    if not candidates:
        return []
    with connect() as conn:
        placeholders = ",".join("?" * len(candidates))
        settled = {
            row[0] for row in conn.execute(
                f"""SELECT filing_id FROM filings
                    WHERE filing_id IN ({placeholders})
                      AND parse_status IN ({','.join('?' * (1 + len(PERMANENTLY_UNREADABLE)))})""",
                (*candidates, PARSED, *PERMANENTLY_UNREADABLE))}
    return [c for c in candidates if c not in settled]


# ------------------------------------------------------- rows within filings

def replace_transactions(filing_id: str, member: Optional[str],
                         rows: Iterable[Dict[str, Any]]) -> int:
    """Write a filing's transactions, replacing anything held for it.

    Replace rather than append: an amendment is re-parsed under the same
    filing id, and appending would stack a corrected filing on top of the one
    it corrects.
    """
    rows = list(rows)
    with connect() as conn:
        conn.execute("DELETE FROM transactions WHERE filing_id = ?", (filing_id,))
        conn.executemany(
            """INSERT INTO transactions(txn_id, filing_id, member_id, row_index,
                    ticker, cusip, asset_name, asset_type_code, owner,
                    transaction_type, transaction_date, notification_date,
                    amount_min, amount_max)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            [(f"{filing_id}#{i}", filing_id, member, i, r.get("ticker"),
              r.get("cusip"), r.get("asset_name"), r.get("asset_type_code"),
              r.get("owner"), r.get("transaction_type"),
              r.get("transaction_date"), r.get("notification_date"),
              r.get("amount_min"), r.get("amount_max"))
             for i, r in enumerate(rows)])
        conn.execute("UPDATE filings SET transaction_count = ? WHERE filing_id = ?",
                     (len(rows), filing_id))
    return len(rows)


def replace_holdings(filing_id: str, member: Optional[str],
                     rows: Iterable[Dict[str, Any]]) -> int:
    rows = list(rows)
    with connect() as conn:
        conn.execute("DELETE FROM holdings WHERE filing_id = ?", (filing_id,))
        conn.executemany(
            """INSERT INTO holdings(holding_id, filing_id, member_id, row_index,
                    ticker, cusip, asset_name, asset_type_code, owner,
                    value_min, value_max, income_min, income_max, income_type,
                    as_of)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            [(f"{filing_id}#{i}", filing_id, member, i, r.get("ticker"),
              r.get("cusip"), r.get("asset_name"), r.get("asset_type_code"),
              r.get("owner"), r.get("value_min"), r.get("value_max"),
              r.get("income_min"), r.get("income_max"), r.get("income_type"),
              r.get("as_of"))
             for i, r in enumerate(rows)])
        conn.execute("UPDATE filings SET holding_count = ? WHERE filing_id = ?",
                     (len(rows), filing_id))
    return len(rows)


# ----------------------------------------------------------------- coverage

def coverage(chamber: Optional[str] = None) -> Dict[str, Any]:
    """What the store holds, split by whether it could be read.

    `complete` is the only honest basis for reading an empty query result as
    an absence rather than as a gap.
    """
    clause, params = ("WHERE chamber = ?", (chamber,)) if chamber else ("", ())
    with connect() as conn:
        rows = conn.execute(
            f"SELECT parse_status, COUNT(*) FROM filings {clause} "
            f"GROUP BY parse_status", params).fetchall()
    by_status = {status: count for status, count in rows}
    total = sum(by_status.values())
    parsed = by_status.get(PARSED, 0)
    return {
        "total": total,
        "by_status": by_status,
        "parsed": parsed,
        "unparsed": total - parsed,
        "complete": total > 0 and parsed == total,
    }


# -------------------------------------------------------------- sync state

def record_sync(source: str, filings_seen: int, filings_parsed: int,
                filings_failed: int, cursor: Optional[str] = None) -> None:
    with connect() as conn:
        conn.execute(
            """INSERT INTO sync_state(source, last_synced_at, last_cursor,
                    filings_seen, filings_parsed, filings_failed)
               VALUES(?,?,?,?,?,?)
               ON CONFLICT(source) DO UPDATE SET
                 last_synced_at = excluded.last_synced_at,
                 last_cursor    = excluded.last_cursor,
                 filings_seen   = excluded.filings_seen,
                 filings_parsed = excluded.filings_parsed,
                 filings_failed = excluded.filings_failed""",
            (source, _now(), cursor, filings_seen, filings_parsed, filings_failed))


def sync_state(source: str) -> Optional[Dict[str, Any]]:
    with connect() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM sync_state WHERE source = ?",
                           (source,)).fetchone()
    return dict(row) if row else None
