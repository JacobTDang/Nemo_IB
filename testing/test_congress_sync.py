"""Ingestion: index -> fetch -> parse -> store, resumably.

The pipeline's job is to be re-runnable. A sync that re-downloads everything
it already read is unusable at 500 filings a year, and one that silently
gives up on the filings it could not read is worse than useless -- it reports
an absence it did not verify.

So: already-parsed filings are never refetched, scans are recorded once and
not retried, transient failures are retried, and every run writes down what
it covered.
"""
import pytest

from tools.altdata_server import congress_store as store
from tools.altdata_server import congress_sync as sync
from tools.altdata_server.congress_trades import DisclosureUnavailable


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("NEMO_CONGRESS_DB", str(tmp_path / "c.db"))
    store.init_schema()


@pytest.fixture
def house(monkeypatch):
    """A House index of PTRs, with controllable per-filing outcomes."""
    state = {"fetched": [], "index": [], "outcomes": {}}

    def install(n, scans=(), errors=()):
        state["index"] = [
            {"last": f"Member{i}", "first": "A", "filing_type": "P",
             "state_district": "XX01", "filing_date": "01/15/2026",
             "doc_id": str(i), "year": "2026"} for i in range(n)]
        state["outcomes"] = {**{str(i): "scan" for i in scans},
                             **{str(i): "error" for i in errors}}

    def fetch_index(year, session=None):
        return state["index"]

    def fetch_ptr(doc_id, year, session=None):
        state["fetched"].append(doc_id)
        outcome = state["outcomes"].get(doc_id)
        if outcome == "scan":
            raise DisclosureUnavailable(
                f"House PTR {doc_id} carries no extractable text; it is most "
                f"likely a scan of a paper filing.")
        if outcome == "error":
            raise DisclosureUnavailable(f"House PTR {doc_id}: connection reset")
        return {"doc_id": doc_id, "member": f"Hon. Member{doc_id}",
                "state_district": "XX01", "chamber": "house",
                "source_url": f"https://example.invalid/{doc_id}.pdf",
                "transactions": [
                    {"ticker": "AAPL", "asset_name": "Apple Inc", "owner": "self",
                     "transaction_type": "purchase",
                     "transaction_date": "2026-01-05",
                     "notification_date": "2026-01-15",
                     "amount_min": 1001, "amount_max": 15000}]}

    monkeypatch.setattr(sync, "fetch_house_index", fetch_index)
    monkeypatch.setattr(sync, "fetch_house_ptr", fetch_ptr)
    monkeypatch.setattr(sync, "_throttle", lambda: None)
    install.state = state
    return install


def test_a_backfill_stores_every_readable_filing(db, house):
    house(5)
    result = sync.sync_house_ptrs(2026)

    assert result["filings_seen"] == 5
    assert result["filings_parsed"] == 5
    assert store.coverage()["complete"] is True
    with store.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0] == 5


def test_a_second_run_refetches_nothing(db, house):
    house(4)
    sync.sync_house_ptrs(2026)
    house.state["fetched"].clear()

    second = sync.sync_house_ptrs(2026)
    assert house.state["fetched"] == [], (
        f"refetched {house.state['fetched']} that were already parsed")
    assert second["filings_parsed"] == 0
    assert second["already_held"] == 4


def test_a_scan_is_recorded_once_and_never_retried(db, house):
    house(3, scans=[1])
    sync.sync_house_ptrs(2026)
    assert store.coverage()["by_status"]["scanned"] == 1

    house.state["fetched"].clear()
    sync.sync_house_ptrs(2026)
    assert "1" not in house.state["fetched"], (
        "a scan cannot become readable; retrying it every run spends the "
        "budget on filings that will never parse")


def test_a_transient_failure_is_retried_next_run(db, house):
    house(3, errors=[2])
    sync.sync_house_ptrs(2026)
    assert store.coverage()["by_status"]["error"] == 1

    house.state["fetched"].clear()
    house(3)                       # the outage has passed
    sync.sync_house_ptrs(2026)
    assert "2" in house.state["fetched"], "a transient failure was never retried"
    assert store.coverage()["complete"] is True


def test_the_run_is_bounded_and_says_what_it_left(db, house):
    """A capped run must report the remainder, not imply it finished."""
    house(10)
    result = sync.sync_house_ptrs(2026, max_filings=4)

    assert result["filings_parsed"] == 4
    assert result["remaining"] == 6, (
        "six filings went unread and the run did not say so")
    assert result["complete"] is False


def test_sync_state_is_written_for_the_run(db, house):
    house(3)
    sync.sync_house_ptrs(2026)
    state = store.sync_state("house_ptr_2026")

    assert state is not None
    assert state["filings_seen"] == 3
    assert state["filings_parsed"] == 3


def test_members_are_registered_from_the_index(db, house):
    house(2)
    sync.sync_house_ptrs(2026)

    with store.connect() as conn:
        rows = conn.execute("SELECT member_id, chamber, last FROM members").fetchall()
    assert len(rows) == 2
    assert all(r[1] == "house" for r in rows)


def test_an_index_failure_is_raised_not_swallowed(db, monkeypatch):
    """An outage must never look like a year in which nobody filed."""
    def boom(year, session=None):
        raise DisclosureUnavailable("House index for 2026 could not be read: 503")

    monkeypatch.setattr(sync, "fetch_house_index", boom)
    with pytest.raises(DisclosureUnavailable):
        sync.sync_house_ptrs(2026)


# --------------------------------------------------- annual reports/holdings

@pytest.fixture
def senate_annuals(monkeypatch):
    """Senate annual reports, with amendments and paper filings mixed in."""
    state = {"fetched": []}

    def install(filings, holdings=2):
        state["filings"] = filings
        state["holdings"] = holdings

    monkeypatch.setattr(sync, "senate_session", lambda: object())
    monkeypatch.setattr(sync, "search_senate_annuals",
                        lambda session, since, limit=200: state["filings"])
    monkeypatch.setattr(sync, "_throttle", lambda: None)

    def fetch(session, uuid):
        state["fetched"].append(uuid)
        return {"member": "A B", "calendar_year": 2025, "amendment": 0,
                "filed_date": "2026-05-15", "has_assets_table": True,
                "source_url": f"https://example.invalid/{uuid}",
                "rows": [], "holdings": [
                    {"ticker": "MSFT", "asset_name": "Microsoft Corp",
                     "owner": "self", "value_min": 15001, "value_max": 50000,
                     "income_min": 201, "income_max": 1000,
                     "income_type": "Dividends", "asset_type": "Corporate Securities"}
                ] * state["holdings"]}

    monkeypatch.setattr(sync, "fetch_senate_annual", fetch)
    install.state = state
    return install


def _annual(uuid, last, amendment=0, kind="annual", year=2025):
    return {"first": "A", "last": last, "office": f"{last} (Senator)",
            "filed_date": "05/15/2026", "label": f"Annual Report for CY {year}",
            "kind": kind, "uuid": uuid, "calendar_year": year,
            "amendment": amendment}


def test_only_the_highest_amendment_is_ingested(db, senate_annuals):
    """Amendments restate in full; ingesting both doubles every holding."""
    senate_annuals([_annual("u1", "Boozman", amendment=0),
                    _annual("u2", "Boozman", amendment=2)])
    sync.sync_senate_annuals()

    assert senate_annuals.state["fetched"] == ["u2"], (
        f"fetched {senate_annuals.state['fetched']}; a superseded base report "
        f"was ingested alongside the amendment that replaced it")


def test_paper_annual_reports_are_recorded_as_scans(db, senate_annuals):
    senate_annuals([_annual("u3", "Durbin", kind="paper")])
    result = sync.sync_senate_annuals()

    assert result["scans"] == 1
    assert store.coverage()["by_status"]["scanned"] == 1
    assert "u3" not in senate_annuals.state["fetched"]


def test_holdings_are_stored_against_the_filing(db, senate_annuals):
    senate_annuals([_annual("u4", "Britt")], holdings=3)
    sync.sync_senate_annuals()

    with store.connect() as conn:
        rows = conn.execute(
            "SELECT ticker, value_min, value_max, as_of FROM holdings").fetchall()
    assert len(rows) == 3
    assert rows[0][0] == "MSFT" and rows[0][1] == 15001
    assert rows[0][3] == "2025-12-31", (
        "a holding without its as-of date cannot be aged, and CY2025 assets "
        "are reported as of the end of that year")


def test_an_annual_filing_is_typed_as_annual(db, senate_annuals):
    senate_annuals([_annual("u5", "Cruz")])
    sync.sync_senate_annuals()

    with store.connect() as conn:
        row = conn.execute(
            "SELECT filing_type, holding_count FROM filings").fetchone()
    assert row[0] == "annual"
    assert row[1] == 2
