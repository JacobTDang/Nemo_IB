"""The local store behind the congressional disclosure pipeline.

Fetching and parsing a House PTR costs an HTTP round trip and a PDF parse, so
the on-demand tool could only afford to open twenty filings per call and had
to report partial coverage every time. Parsing each filing once into a store
is what makes complete coverage affordable, and complete coverage is what
makes "no trades in NVDA" mean anything.

The store therefore has to be honest about what it holds. A filing that could
not be parsed -- a scan of a paper filing, a fetch that failed -- is recorded
as such rather than left absent, because an absent filing and an unreadable
one look identical from a query and mean opposite things.
"""
import os
import sqlite3

import pytest

from tools.altdata_server import congress_store as store


@pytest.fixture
def db(tmp_path, monkeypatch):
    path = tmp_path / "congress.db"
    monkeypatch.setenv("NEMO_CONGRESS_DB", str(path))
    store.init_schema()
    return path


def _filing(doc_id="1", chamber="house", status="parsed", **kw):
    base = dict(filing_id=f"{chamber}:{doc_id}", chamber=chamber, doc_id=doc_id,
                member_id="house:allen:richard:GA12", filing_type="ptr",
                raw_filing_type="P", filed_date="2025-01-16", year=2025,
                source_url="https://example.invalid/1.pdf", parse_status=status)
    base.update(kw)
    return base


def test_the_schema_is_created_idempotently(db):
    store.init_schema()
    store.init_schema()
    with store.connect() as conn:
        names = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"members", "filings", "transactions", "holdings",
            "sync_state"} <= names


def test_the_database_path_is_resolved_per_call(tmp_path, monkeypatch):
    """A default argument would freeze the path at import.

    state/schema.py carries this same lesson: `def get_connection(db_path=DB_PATH)`
    bound the value once, at function-definition time, and silently ignored
    every later override.
    """
    first = tmp_path / "one.db"
    monkeypatch.setenv("NEMO_CONGRESS_DB", str(first))
    assert store.current_db_path() == str(first)

    second = tmp_path / "two.db"
    monkeypatch.setenv("NEMO_CONGRESS_DB", str(second))
    assert store.current_db_path() == str(second)


def test_reingesting_a_filing_does_not_duplicate_it(db):
    store.upsert_filing(_filing())
    store.upsert_filing(_filing())
    with store.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM filings").fetchone()[0] == 1


def test_reingesting_replaces_rows_rather_than_appending(db):
    """An amended filing is re-parsed; its rows must not stack up."""
    store.upsert_filing(_filing())
    rows = [{"ticker": "AAPL", "asset_name": "Apple Inc",
             "transaction_type": "purchase", "transaction_date": "2025-01-05",
             "amount_min": 1001, "amount_max": 15000, "owner": "self"}]
    store.replace_transactions("house:1", "house:allen:richard:GA12", rows)
    store.replace_transactions("house:1", "house:allen:richard:GA12", rows)

    with store.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0] == 1


def test_an_unreadable_filing_is_recorded_not_skipped(db):
    """A scan and a filing nobody fetched look identical unless one is stored."""
    store.upsert_filing(_filing(doc_id="9", status="scanned",
                                parse_error="no extractable text"))
    coverage = store.coverage()

    assert coverage["by_status"]["scanned"] == 1
    assert coverage["unparsed"] == 1
    assert coverage["complete"] is False


def test_coverage_is_complete_only_when_everything_parsed(db):
    for i in range(3):
        store.upsert_filing(_filing(doc_id=str(i)))
    coverage = store.coverage()

    assert coverage["total"] == 3
    assert coverage["by_status"]["parsed"] == 3
    assert coverage["unparsed"] == 0
    assert coverage["complete"] is True


def test_already_parsed_filings_are_not_offered_again(db):
    """Incremental sync must not re-download what it already read."""
    store.upsert_filing(_filing(doc_id="1", status="parsed"))
    store.upsert_filing(_filing(doc_id="2", status="scanned"))

    pending = store.unparsed_filing_ids(["house:1", "house:2", "house:3"])
    assert "house:1" not in pending, "a parsed filing was queued for refetch"
    assert "house:3" in pending, "an unseen filing must be queued"
    assert "house:2" not in pending, (
        "a scan will not become readable on a retry; requeueing it forever "
        "spends the whole budget on filings that cannot be parsed")


def test_a_failed_fetch_is_retried_but_a_scan_is_not(db):
    """The two failures are different: one is transient, one is permanent."""
    store.upsert_filing(_filing(doc_id="4", status="error",
                                parse_error="connection reset"))
    pending = store.unparsed_filing_ids(["house:4"])
    assert "house:4" in pending, (
        "a transient fetch failure was recorded as permanently unreadable")


def test_sync_state_records_what_a_run_covered(db):
    store.record_sync("house_2025", filings_seen=100, filings_parsed=97,
                      filings_failed=3, cursor="2025-12-31")
    state = store.sync_state("house_2025")

    assert state["filings_seen"] == 100
    assert state["filings_parsed"] == 97
    assert state["filings_failed"] == 3
    assert state["last_cursor"] == "2025-12-31"
    assert state["last_synced_at"]


def test_amount_brackets_survive_a_round_trip(db):
    store.upsert_filing(_filing())
    store.replace_transactions("house:1", "house:allen:richard:GA12", [
        {"ticker": "NVDA", "asset_name": "Nvidia", "transaction_type": "sale",
         "transaction_date": "2025-03-01", "amount_min": 1_000_001,
         "amount_max": 5_000_000, "owner": "spouse"}])

    with store.connect() as conn:
        row = conn.execute(
            "SELECT amount_min, amount_max, owner FROM transactions").fetchone()
    assert row[0] == 1_000_001 and row[1] == 5_000_000 and row[2] == "spouse"


def test_holdings_carry_their_own_as_of_date(db):
    """A holding is a snapshot; without its date it cannot be aged."""
    store.upsert_filing(_filing(doc_id="7", filing_type="annual",
                                raw_filing_type="O"))
    store.replace_holdings("house:7", "house:allen:richard:GA12", [
        {"ticker": "MSFT", "asset_name": "Microsoft Corp", "owner": "joint",
         "value_min": 15001, "value_max": 50000, "as_of": "2024-12-31"}])

    with store.connect() as conn:
        row = conn.execute(
            "SELECT ticker, value_min, value_max, as_of FROM holdings").fetchone()
    assert row == ("MSFT", 15001, 50000, "2024-12-31")


# ------------------------------------------------- one filing, one transaction

def test_a_filing_and_its_rows_are_written_in_one_transaction(db):
    """A crash between the two leaves the filing permanently marked read.

    `unparsed_filing_ids` never offers a `parsed` filing again, so a status
    that commits before its rows do is a filing that is silently empty for
    good -- with `coverage.complete` still reporting true.
    """
    def rows_that_die():
        yield {"ticker": "AAPL", "asset_name": "Apple Inc",
               "transaction_type": "purchase", "transaction_date": "2025-01-05",
               "amount_min": 1001, "amount_max": 15000, "owner": "self"}
        raise RuntimeError("the container went away mid-write")

    with pytest.raises(RuntimeError):
        store.record_parsed_filing(_filing(), transactions=rows_that_die())

    with store.connect() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM filings WHERE parse_status = 'parsed'"
        ).fetchone()[0] == 0, (
            "the filing is recorded as read and holds no rows; it will never "
            "be offered for reading again")


def test_a_filing_written_whole_holds_both_halves(db):
    store.record_parsed_filing(_filing(), transactions=[
        {"ticker": "NVDA", "asset_name": "Nvidia", "transaction_type": "sale",
         "transaction_date": "2025-01-05", "amount_min": 1001,
         "amount_max": 15000, "owner": "self"}])

    with store.connect() as conn:
        status, count = conn.execute(
            "SELECT parse_status, transaction_count FROM filings").fetchone()
        rows = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
    assert (status, count, rows) == ("parsed", 1, 1)


def test_holdings_are_written_with_their_filing_too(db):
    store.record_parsed_filing(
        _filing(doc_id="7", filing_type="annual"),
        holdings=[{"ticker": "MSFT", "asset_name": "Microsoft Corp",
                   "owner": "self", "value_min": 15001, "value_max": 50000,
                   "as_of": "2024-12-31"}])

    with store.connect() as conn:
        assert conn.execute(
            "SELECT holding_count FROM filings").fetchone()[0] == 1


# ------------------------------------------------------------- schema changes

_PRE_MIGRATION_FILINGS = """CREATE TABLE filings(
    filing_id        TEXT PRIMARY KEY,
    chamber          TEXT NOT NULL,
    doc_id           TEXT NOT NULL,
    member_id        TEXT,
    filing_type      TEXT,
    raw_filing_type  TEXT,
    filed_date       TEXT,
    year             INTEGER,
    source_url       TEXT,
    parse_status     TEXT NOT NULL,
    parse_error      TEXT,
    transaction_count INTEGER DEFAULT 0,
    holding_count    INTEGER DEFAULT 0,
    fetched_at       TEXT,
    parsed_at        TEXT
)"""


def test_a_column_added_later_reaches_a_database_that_predates_it(tmp_path,
                                                                  monkeypatch):
    """CREATE TABLE IF NOT EXISTS leaves an existing table exactly as it was.

    This is the deployed `congress-data` volume: the table is there, so the
    new column never arrives, and the first write that mentions it raises
    `no such column` for every filing in the run.
    """
    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.execute(_PRE_MIGRATION_FILINGS)
    old.commit()
    old.close()
    monkeypatch.setenv("NEMO_CONGRESS_DB", str(path))

    store.init_schema()

    with store.connect() as conn:
        columns = {r[1] for r in conn.execute("PRAGMA table_info(filings)")}
    missing = {c for _, c, _ in store._MIGRATIONS} - columns
    assert not missing, f"{missing} never reached an existing database"


def test_an_old_database_still_takes_a_write(tmp_path, monkeypatch):
    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.execute(_PRE_MIGRATION_FILINGS)
    old.commit()
    old.close()
    monkeypatch.setenv("NEMO_CONGRESS_DB", str(path))
    store.init_schema()

    store.upsert_filing(_filing(content_hash="abc"))

    with store.connect() as conn:
        assert conn.execute(
            "SELECT content_hash FROM filings").fetchone()[0] == "abc"


def test_the_schema_version_is_stamped(db):
    with store.connect() as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == \
            store.SCHEMA_VERSION


# --------------------------------------------------- noticing a republication

def test_fetched_at_records_when_the_filing_was_last_checked(db):
    store.upsert_filing(_filing(fetched_at="2026-01-01T00:00:00+00:00"))
    store.upsert_filing(_filing(fetched_at="2026-06-01T00:00:00+00:00"))

    with store.connect() as conn:
        assert conn.execute("SELECT fetched_at FROM filings").fetchone()[0] == \
            "2026-06-01T00:00:00+00:00", (
                "fetched_at is not in the DO UPDATE SET list, so there is no "
                "record of when the filing was last looked at")


def test_a_content_hash_records_what_produced_the_stored_rows(db):
    store.upsert_filing(_filing(content_hash="deadbeef"))
    with store.connect() as conn:
        assert conn.execute(
            "SELECT content_hash FROM filings").fetchone()[0] == "deadbeef"


def test_an_error_does_not_erase_the_hash_of_what_was_last_read(db):
    """A failed re-check says nothing about the bytes already parsed."""
    store.upsert_filing(_filing(content_hash="deadbeef"))
    store.upsert_filing(_filing(status="error", parse_error="connection reset"))

    with store.connect() as conn:
        assert conn.execute(
            "SELECT content_hash FROM filings").fetchone()[0] == "deadbeef"


def test_a_filing_whose_index_row_changed_is_read_again(db):
    """A corrected republish keeps its DocID; only the index row moves."""
    store.upsert_filing(_filing(doc_id="1", filed_date="2025-01-16"))

    pending = store.unparsed_filing_ids(
        ["house:1"], index_filed_dates={"house:1": "2025-03-02"})
    assert "house:1" in pending, (
        "the Clerk re-posted the filing and the store went on serving the "
        "superseded numbers")


def test_an_unchanged_index_row_is_still_not_refetched(db):
    store.upsert_filing(_filing(doc_id="1", filed_date="2025-01-16"))

    pending = store.unparsed_filing_ids(
        ["house:1"], index_filed_dates={"house:1": "2025-01-16"})
    assert pending == [], "a re-run refetched a filing that had not changed"


def test_a_stale_filing_is_offered_again_on_a_rolling_recheck(db):
    store.upsert_filing(_filing(doc_id="1",
                                fetched_at="2020-01-01T00:00:00+00:00"))
    store.upsert_filing(_filing(doc_id="2"))

    pending = store.unparsed_filing_ids(["house:1", "house:2"], recheck_days=30)
    assert pending == ["house:1"], (
        f"expected only the stale filing to be re-offered, got {pending}")


def test_a_scan_is_re_examined_on_a_rolling_recheck(db):
    """A paper filing re-posted with a text layer stays 'scanned' forever."""
    store.upsert_filing(_filing(doc_id="9", status="scanned",
                                fetched_at="2020-01-01T00:00:00+00:00"))
    assert store.unparsed_filing_ids(["house:9"], recheck_days=30) == ["house:9"]


# ------------------------------------------------------- a store never synced

def test_a_never_synced_database_reads_as_an_empty_store(tmp_path, monkeypatch):
    """On a fresh volume the tables do not exist yet.

    `OperationalError: no such table: filings` reaches the caller verbatim,
    in place of the empty-store guidance that exists for exactly this moment.
    """
    monkeypatch.setenv("NEMO_CONGRESS_DB", str(tmp_path / "never-synced.db"))

    coverage = store.coverage()

    assert coverage["total"] == 0
    assert coverage["complete"] is False


def test_a_filing_can_be_requeued_when_it_parsed_to_nothing(db):
    """The three House filings already recorded 'parsed' with zero rows."""
    store.upsert_filing(_filing(doc_id="20025111", status="parsed"))
    store.upsert_filing(_filing(doc_id="20025152", status="parsed",
                                filing_type="annual"))
    store.upsert_filing(_filing(doc_id="2", status="parsed"))
    store.replace_transactions("house:2", "house:allen:richard:GA12", [
        {"ticker": "AAPL", "asset_name": "Apple Inc",
         "transaction_type": "purchase", "amount_min": 1001,
         "amount_max": 15000}])

    assert store.requeue_empty_transaction_reports() == 1

    pending = store.unparsed_filing_ids(["house:20025111", "house:20025152",
                                         "house:2"])
    assert pending == ["house:20025111"], (
        f"requeued {pending}: an annual report holds holdings rather than "
        f"transactions, and a filing with rows was already read")


def test_a_failed_recheck_does_not_report_zero_over_the_rows_still_held(db):
    """The rows from the last successful read are still there to be queried."""
    store.record_parsed_filing(_filing(), transactions=[
        {"ticker": "NVDA", "asset_name": "Nvidia", "transaction_type": "sale",
         "transaction_date": "2025-01-05", "amount_min": 1001,
         "amount_max": 15000, "owner": "self"}])
    store.upsert_filing(_filing(status="error", parse_error="connection reset"))

    with store.connect() as conn:
        count = conn.execute(
            "SELECT transaction_count FROM filings").fetchone()[0]
        rows = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
    assert count == rows == 1
