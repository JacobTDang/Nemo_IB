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
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
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
        -- What was actually read. The filing id alone cannot notice a
        -- correction re-posted under the same DocID; the bytes can.
        content_hash     TEXT,
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


# Bumped whenever _MIGRATIONS grows, and stamped into PRAGMA user_version so
# a database can say which shape it is rather than being inspected for it.
SCHEMA_VERSION = 1

# Columns added after the first store shipped. CREATE TABLE IF NOT EXISTS
# leaves an existing table exactly as it was, so a column added to
# CREATE_SCHEMA never reaches the deployed volume -- and the first write that
# mentions it raises `no such column` for every filing in the run.
_MIGRATIONS = (
    ("filings", "content_hash", "TEXT"),
)

# The database paths this process has already brought up to the current
# schema. The sync calls init_schema(); the MCP server never does, so before
# the first sync every congress tool answered `no such table: filings` in
# place of the empty-store guidance written for exactly that moment.
_INITIALISED: set = set()


def _missing_columns(conn: sqlite3.Connection) -> List[Sequence[str]]:
    missing = []
    for table, column, declaration in _MIGRATIONS:
        present = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        if column not in present:
            missing.append((table, column, declaration))
    return missing


def _apply_schema(conn: sqlite3.Connection) -> None:
    for statement in CREATE_SCHEMA:
        conn.execute(statement)
    if not _missing_columns(conn):
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        return

    # The sync and the server both open the store, and on the first run after
    # a column is added they both find it missing. Without the write lock they
    # both issue the ALTER and the second one fails on a column that is by
    # then already there, so the check and the change are taken together.
    conn.execute("BEGIN IMMEDIATE")
    try:
        for table, column, declaration in _missing_columns(conn):
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        conn.commit()
    except Exception:
        conn.rollback()
        raise


@contextmanager
def connect():
    """A connection to the store, committing on clean exit.

    Everything written inside one `with` block lands together or not at all,
    which is what keeps a filing's status from being durable before the rows
    it describes.
    """
    path = current_db_path()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        if path not in _INITIALISED:
            _apply_schema(conn)
            _INITIALISED.add(path)
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_schema() -> None:
    """Create the schema, and add to it what an older database lacks."""
    with connect() as conn:
        _apply_schema(conn)
        _INITIALISED.add(current_db_path())


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------------ members

# Honorifics the House index sometimes folds into the first-name field.
_HONORIFICS = {"mr", "mrs", "ms", "dr", "hon", "honorable", "rep", "sen"}


def _given_name(first: str) -> str:
    """The first given name, without middle initials or honorifics.

    The House index writes the same person several ways across filings --
    "Rudy C. Yakym" and "Rudy Yakym", "Thomas Suozzi" and "Thomas R. Suozzi",
    "Laurel Lee" and "Laurel Mrs Lee". Keyed literally, each variant became
    its own member and split that person's history.

    Only the leading given name is kept, which is enough to separate Rick
    Scott from Tim Scott and Adam Smith from Adrian Smith while collapsing
    the punctuation variants of one person.
    """
    tokens = [t for t in re.split(r"[\s.]+", (first or "").strip().lower()) if t]
    for token in tokens:
        if token in _HONORIFICS or len(token) == 1:
            continue
        return token
    return tokens[0] if tokens else ""


def member_id(chamber: str, last: str, first: str, state: str = "") -> str:
    """A stable key. Names repeat across chambers and across states."""
    parts = [chamber.lower(), (last or "").strip().lower(),
             _given_name(first), (state or "").strip().upper()]
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

def _upsert_filing(conn: sqlite3.Connection, filing: Dict[str, Any]) -> None:
    payload = {
        "filing_id": filing["filing_id"], "chamber": filing["chamber"],
        "doc_id": filing["doc_id"], "member_id": filing.get("member_id"),
        "filing_type": filing.get("filing_type"),
        "raw_filing_type": filing.get("raw_filing_type"),
        "filed_date": filing.get("filed_date"), "year": filing.get("year"),
        "source_url": filing.get("source_url"),
        "parse_status": filing.get("parse_status", "pending"),
        "parse_error": filing.get("parse_error"),
        # Left as None when the caller does not say, so a re-check that
        # failed does not report zero rows over the rows still stored.
        "transaction_count": filing.get("transaction_count"),
        "holding_count": filing.get("holding_count"),
        "content_hash": filing.get("content_hash"),
        "fetched_at": filing.get("fetched_at") or _now(),
        "parsed_at": filing.get("parsed_at"),
    }
    conn.execute(
        """INSERT INTO filings(filing_id, chamber, doc_id, member_id,
                filing_type, raw_filing_type, filed_date, year, source_url,
                parse_status, parse_error, transaction_count,
                holding_count, content_hash, fetched_at, parsed_at)
           VALUES(:filing_id, :chamber, :doc_id, :member_id, :filing_type,
                  :raw_filing_type, :filed_date, :year, :source_url,
                  :parse_status, :parse_error,
                  COALESCE(:transaction_count, 0),
                  COALESCE(:holding_count, 0), :content_hash,
                  :fetched_at, :parsed_at)
           ON CONFLICT(filing_id) DO UPDATE SET
             member_id        = excluded.member_id,
             filing_type      = excluded.filing_type,
             raw_filing_type  = excluded.raw_filing_type,
             filed_date       = excluded.filed_date,
             year             = excluded.year,
             source_url       = excluded.source_url,
             parse_status     = excluded.parse_status,
             parse_error      = excluded.parse_error,
             transaction_count= COALESCE(:transaction_count,
                                         filings.transaction_count),
             holding_count    = COALESCE(:holding_count,
                                         filings.holding_count),
             -- A re-check that failed says nothing about the bytes already
             -- read, so the hash of those bytes survives it.
             content_hash     = COALESCE(excluded.content_hash,
                                         filings.content_hash),
             -- When the filing was last looked at, which is a different fact
             -- from when it was first seen and the only way to tell a filing
             -- checked this morning from one nobody has opened since 2024.
             fetched_at       = excluded.fetched_at,
             parsed_at        = excluded.parsed_at""", payload)


def upsert_filing(filing: Dict[str, Any]) -> None:
    with connect() as conn:
        _upsert_filing(conn, filing)


def unparsed_filing_ids(candidates: Sequence[str],
                        index_filed_dates: Optional[Dict[str, str]] = None,
                        recheck_days: Optional[int] = None) -> List[str]:
    """Which of `candidates` still need fetching.

    A filing already parsed is skipped. So is one recorded as a scan: it will
    not become readable on a retry, and requeueing it every run would spend
    the whole budget on filings that cannot be parsed. A failed fetch is
    transient and is offered again.

    Two things reopen a settled filing, because the id alone cannot notice a
    document that changed underneath it. `index_filed_dates` is what the
    chamber's own index says today: a filing whose index row has moved has
    been re-posted, and the store is holding the superseded numbers.
    `recheck_days` re-offers anything last looked at longer ago than that,
    which is the only way a correction that left the index row alone -- or a
    paper filing since re-posted with a text layer -- is ever seen again.
    """
    if not candidates:
        return []
    with connect() as conn:
        placeholders = ",".join("?" * len(candidates))
        settled = {
            row[0]: (row[1], row[2]) for row in conn.execute(
                f"""SELECT filing_id, filed_date, fetched_at FROM filings
                    WHERE filing_id IN ({placeholders})
                      AND parse_status IN ({','.join('?' * (1 + len(PERMANENTLY_UNREADABLE)))})""",
                (*candidates, PARSED, *PERMANENTLY_UNREADABLE))}

    cutoff = None
    if recheck_days is not None:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=recheck_days)
                  ).isoformat(timespec="seconds")

    pending: List[str] = []
    for candidate in candidates:
        if candidate not in settled:
            pending.append(candidate)
            continue
        filed_date, fetched_at = settled[candidate]
        published = (index_filed_dates or {}).get(candidate)
        if published and filed_date and published != filed_date:
            pending.append(candidate)
        elif cutoff is not None and (fetched_at or "") < cutoff:
            pending.append(candidate)
    return pending


# ------------------------------------------------------- rows within filings

def _sane_transaction(row: Dict[str, Any],
                      filed_date: Optional[str]) -> Dict[str, Any]:
    """Drop the fields of a row that cannot be true, keeping the rest.

    The last place a false figure can be stopped. Parsers get better and
    parsers get replaced, but every one of them writes through here, so a row
    that cannot be true should not depend on which produced it.

    Both shapes below were found in the live store. Neither is repaired, only
    refused: a trade dated 2026-12-26 and disclosed on 2026-02-09 is almost
    certainly December 2025, and "almost certainly" is how a parsing bug
    becomes a fact in a database.
    """
    row = dict(row)

    # An amount bracket whose floor is above its ceiling. Found 24 times, and
    # every one had amount_max = 200 -- the PTR's own "Cap. Gains > $200?"
    # header, read as a figure when the entry spanned a page break. The floor
    # was right in every case, so the floor stays.
    low, high = row.get("amount_min"), row.get("amount_max")
    if low is not None and high is not None and low > high:
        row["amount_max"] = None

    # A trade disclosed before it happened. Found 7 times, one by ten months.
    txn, filed = row.get("transaction_date"), filed_date
    if txn and filed and txn > filed:
        row["transaction_date"] = None

    return row


def _replace_transactions(conn: sqlite3.Connection, filing_id: str,
                          member: Optional[str],
                          rows: Iterable[Dict[str, Any]]) -> int:
    rows = list(rows)
    filed_row = conn.execute(
        "SELECT filed_date FROM filings WHERE filing_id = ?",
        (filing_id,)).fetchone()
    # Absence of a filing date is not evidence a trade date is wrong.
    filed_date = filed_row[0] if filed_row else None
    rows = [_sane_transaction(r, filed_date) for r in rows]
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


def replace_transactions(filing_id: str, member: Optional[str],
                         rows: Iterable[Dict[str, Any]]) -> int:
    """Write a filing's transactions, replacing anything held for it.

    Replace rather than append: an amendment is re-parsed under the same
    filing id, and appending would stack a corrected filing on top of the one
    it corrects.
    """
    with connect() as conn:
        return _replace_transactions(conn, filing_id, member, rows)


def _replace_holdings(conn: sqlite3.Connection, filing_id: str,
                      member: Optional[str],
                      rows: Iterable[Dict[str, Any]]) -> int:
    rows = list(rows)
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


def replace_holdings(filing_id: str, member: Optional[str],
                     rows: Iterable[Dict[str, Any]]) -> int:
    with connect() as conn:
        return _replace_holdings(conn, filing_id, member, rows)


def record_parsed_filing(filing: Dict[str, Any],
                         transactions: Optional[Iterable[Dict[str, Any]]] = None,
                         holdings: Optional[Iterable[Dict[str, Any]]] = None
                         ) -> int:
    """Write a filing and its rows as one indivisible act.

    Written separately, the status committed first and the rows second, so
    anything in between -- a cron timeout, a host reboot, an OOM kill, a
    raising row write -- left the filing durably `parse_status='parsed'` with
    nothing in it. `unparsed_filing_ids` never offers a parsed filing again,
    so that filing is silently and permanently empty while `coverage.complete`
    goes on reporting true.
    """
    with connect() as conn:
        _upsert_filing(conn, filing)
        written = 0
        if transactions is not None:
            written += _replace_transactions(
                conn, filing["filing_id"], filing.get("member_id"), transactions)
        if holdings is not None:
            written += _replace_holdings(
                conn, filing["filing_id"], filing.get("member_id"), holdings)
    return written


# ----------------------------------------------------------------- coverage

def repair_impossible_rows() -> Dict[str, int]:
    """Apply the write-time sanity rules to rows already stored.

    `replace_transactions` refuses these on the way in, but rows written before
    it did are still held. Idempotent, so it is safe to run on every sync.
    Returns what it changed rather than logging it, because a repair that
    reports nothing is indistinguishable from one that did nothing.
    """
    with connect() as conn:
        inverted = conn.execute(
            """UPDATE transactions SET amount_max = NULL
               WHERE amount_min IS NOT NULL AND amount_max IS NOT NULL
                 AND amount_min > amount_max""").rowcount
        undated = conn.execute(
            """UPDATE transactions SET transaction_date = NULL
               WHERE transaction_date IS NOT NULL
                 AND filing_id IN (SELECT filing_id FROM filings
                                   WHERE filed_date IS NOT NULL)
                 AND transaction_date > (SELECT filed_date FROM filings
                                         WHERE filings.filing_id =
                                               transactions.filing_id)""").rowcount
    return {"amounts_cleared": inverted, "dates_cleared": undated}


def requeue_empty_transaction_reports() -> int:
    """Offer again the transaction reports recorded as read but holding nothing.

    A PTR exists to report a trade, so zero rows is a parse that failed rather
    than a member who did not trade -- a table header that shifted, an
    extraction that moved a column, an interstitial served with status 200.
    Three House filings reached the live store this way (20025111, 20025152,
    20033695) and `unparsed_filing_ids` will never offer a parsed filing
    again, so nothing else would ever look at them.

    Transaction reports only: an annual report holds holdings, and having no
    transactions is the normal shape of one. Idempotent -- a filing requeued
    here is re-read on the next run and settles either way. Returns how many
    were reopened, because a repair that reports nothing cannot be told from
    one that did nothing.
    """
    with connect() as conn:
        return conn.execute(
            """UPDATE filings
                  SET parse_status = 'error', parse_error = ?
                WHERE parse_status = ?
                  AND filing_type = 'ptr'
                  AND NOT EXISTS (SELECT 1 FROM transactions t
                                  WHERE t.filing_id = filings.filing_id)""",
            ("recorded as parsed with no transactions; a PTR is filed to "
             "report a trade, so this was a parse that failed",
             PARSED)).rowcount


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


def merge_duplicate_members() -> int:
    """Reunite members whose key was written before `_given_name` normalised it.

    The House index writes one person several ways across filings, so the
    store accumulated "Rudy C. Yakym" beside "Rudy Yakym" and
    "Thomas Suozzi" beside "Thomas R. Suozzi". Normalising the key stops new
    duplicates; this repairs the ones already recorded, which otherwise keep
    that person's history split in two and invisible to any query.

    Every dependent row is repointed rather than deleted -- the point is to
    reunite a history, not halve it. Returns how many records were merged.
    Safe to re-run.
    """
    merged = 0
    with connect() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT member_id, chamber, last, first, full_name, state FROM members"
        ).fetchall()

        def key_of(row) -> str:
            # The state is part of the key and must be carried through, or the
            # recomputed key matches neither stored id and the merge keeps an
            # arbitrary one.
            return member_id(row["chamber"], row["last"] or "",
                             row["first"] or "", row["state"] or "")

        canonical: Dict[str, str] = {}
        for row in rows:
            key = key_of(row)
            # Prefer the id that already equals the normalised key; failing
            # that, keep the first seen so the choice is deterministic.
            if key not in canonical or row["member_id"] == key:
                canonical[key] = row["member_id"]

        for row in rows:
            keep = canonical[key_of(row)]
            if row["member_id"] == keep:
                continue
            for table in ("transactions", "holdings", "filings"):
                conn.execute(
                    f"UPDATE {table} SET member_id = ? WHERE member_id = ?",
                    (keep, row["member_id"]))
            conn.execute("DELETE FROM members WHERE member_id = ?",
                         (row["member_id"],))
            merged += 1

    return merged
