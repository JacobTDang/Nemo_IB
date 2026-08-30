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
from tools.altdata_server.congress_trades import (
    DisclosureBlocked,
    DisclosureUnavailable,
)


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("NEMO_CONGRESS_DB", str(tmp_path / "c.db"))
    store.init_schema()


@pytest.fixture
def house(monkeypatch):
    """A House index of PTRs, with controllable per-filing outcomes."""
    state = {"fetched": [], "index": [], "outcomes": {}}

    def install(n, scans=(), errors=(), empty=(), blocked=(), filed="01/15/2026"):
        state["index"] = [
            {"last": f"Member{i}", "first": "A", "filing_type": "P",
             "state_district": "XX01", "filing_date": filed,
             "doc_id": str(i), "year": "2026"} for i in range(n)]
        state["outcomes"] = {**{str(i): "scan" for i in scans},
                             **{str(i): "error" for i in errors},
                             **{str(i): "empty" for i in empty},
                             **{str(i): "blocked" for i in blocked}}

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
        if outcome == "blocked":
            raise DisclosureBlocked(
                f"House PTR {doc_id}: the Clerk refused the request (429)")
        # A table the parser never located: the metadata reads, the rows do
        # not, and nothing about the filing says so.
        empty = outcome == "empty"
        return {"doc_id": doc_id, "member": f"Hon. Member{doc_id}",
                "state_district": "XX01", "chamber": "house",
                "source_url": f"https://example.invalid/{doc_id}.pdf",
                "table_found": not empty,
                "no_reportable_transactions": False,
                "content_hash": f"hash-of-{doc_id}",
                "transactions": [] if empty else [
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
        # Mirrors the real parser: it resolves report_kind and as_of itself,
        # because only it has seen the heading.
        return {"member": "A B", "calendar_year": 2025, "amendment": 0,
                "report_kind": "annual", "as_of": "2025-12-31",
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


def test_a_new_filer_report_keeps_its_own_kind_and_date(db, senate_annuals, monkeypatch):
    """The umbrella search returns these; they must not be filed as annual."""
    senate_annuals([_annual("u9", "Armstrong")])

    def fetch(session, uuid):
        return {"member": "Alan Armstrong", "calendar_year": None,
                "amendment": 0, "report_kind": "new_filer",
                "as_of": "2026-03-24", "filed_date": "2026-07-21",
                "has_assets_table": True, "source_url": "https://example.invalid",
                "rows": [], "holdings": [
                    {"ticker": "MSFT", "asset_name": "Microsoft Corp",
                     "owner": "self", "value_min": 15001, "value_max": 50000}]}

    monkeypatch.setattr(sync, "fetch_senate_annual", fetch)
    sync.sync_senate_annuals()

    with store.connect() as conn:
        kind = conn.execute("SELECT filing_type FROM filings").fetchone()[0]
        as_of = conn.execute("SELECT as_of FROM holdings").fetchone()[0]
    assert kind == "new_filer", f"filed as {kind!r}"
    assert as_of == "2026-03-24", (
        "a New Filer snapshot was dated to a calendar year end it never covered")


def test_no_holding_is_ever_stored_without_an_as_of(db, senate_annuals, monkeypatch):
    """A holding with no date cannot be aged against later trades."""
    senate_annuals([_annual("u10", "Nodate")])

    def fetch(session, uuid):
        return {"member": "No Date", "calendar_year": None, "amendment": 0,
                "report_kind": "annual", "as_of": None,
                "filed_date": "2026-05-15", "has_assets_table": True,
                "source_url": "https://example.invalid", "rows": [],
                "holdings": [{"ticker": "X", "asset_name": "X Corp",
                              "owner": "self", "value_min": 1, "value_max": 2}]}

    monkeypatch.setattr(sync, "fetch_senate_annual", fetch)
    sync.sync_senate_annuals()

    with store.connect() as conn:
        as_of = conn.execute("SELECT as_of FROM holdings").fetchone()[0]
    assert as_of is not None, (
        "the parser gave no as-of and the sync stored the holding anyway; "
        "it must fall back to the filing date rather than leave it empty")


@pytest.mark.parametrize("doc_id,is_paper", [
    ("8221263", True),    # McCaul 2025 -- a real scan beginning with 8
    ("9116162", True),    # Bilirakis -- a real scan beginning with 9
    ("20026537", False),  # Allen PTR -- electronic
    ("10077643", False),  # Adams annual -- electronic
])
def test_paper_filings_are_recognised_by_length_not_prefix(doc_id, is_paper):
    """Every 7-digit id in the store was a scan; every 8-digit one parsed.

    Requiring the id to begin with 9, as the published descriptions put it,
    matched 51 of 166 scans and downloaded the other 115 to learn the same
    thing from several megabytes of page images.
    """
    assert sync._house_docid_is_paper(doc_id) is is_paper


# ------------------------------------------------- zero rows is not "no trades"

def test_a_ptr_that_parsed_to_nothing_is_retried_rather_than_believed(db, house):
    """A PTR is filed to report a trade. Zero rows is a failure to read it."""
    house(3, empty=[1])
    result = sync.sync_house_ptrs(2026)

    assert store.coverage()["by_status"].get("parsed") == 2
    assert result["errors"] == 1, (
        "a filing whose transaction table was never located was recorded as "
        "read, and unparsed_filing_ids will never offer it again")

    house.state["fetched"].clear()
    house(3)                       # the layout, or the extraction, recovered
    sync.sync_house_ptrs(2026)
    assert "1" in house.state["fetched"], "the empty parse was never retried"


def test_the_filing_that_says_it_has_nothing_is_taken_at_its_word(db, house,
                                                                  monkeypatch):
    def fetch(doc_id, year, session=None):
        return {"doc_id": doc_id, "member": "Hon. Member", "chamber": "house",
                "state_district": "XX01", "table_found": True,
                "no_reportable_transactions": True, "content_hash": "h",
                "source_url": "https://example.invalid", "transactions": []}

    house(1)
    monkeypatch.setattr(sync, "fetch_house_ptr", fetch)
    result = sync.sync_house_ptrs(2026)

    assert result["filings_parsed"] == 1
    assert result["errors"] == 0


# ------------------------------------------------- status and rows together

def test_a_crash_writing_rows_leaves_no_filing_marked_read(db, house,
                                                           monkeypatch):
    """The filing's status must not commit before the rows it describes.

    Before the fix the two were separate transactions, so a failure between
    them left `parse_status='parsed'` with zero rows -- and a parsed filing is
    never offered again.
    """
    def boom(*args, **kwargs):
        raise RuntimeError("the container went away mid-write")

    house(1)
    monkeypatch.setattr(store, "replace_transactions", boom)
    try:
        sync.sync_house_ptrs(2026)
    except RuntimeError:
        pass

    with store.connect() as conn:
        phantom = conn.execute(
            """SELECT COUNT(*) FROM filings f
               WHERE f.parse_status = 'parsed'
                 AND NOT EXISTS (SELECT 1 FROM transactions t
                                 WHERE t.filing_id = f.filing_id)"""
        ).fetchone()[0]
    assert phantom == 0, (
        "a filing is recorded as read and holds no rows; it is permanently "
        "and silently empty")


# ------------------------------------------------------------ being refused

def test_a_blocked_source_stops_the_run_rather_than_deepening_the_block(db,
                                                                        house):
    house(20, blocked=range(20))
    result = sync.sync_house_ptrs(2026)

    assert result["blocked"], "the run marched through every filing while blocked"
    assert len(house.state["fetched"]) <= sync.MAX_CONSECUTIVE_FAILURES
    assert result["complete"] is False
    assert result["remaining"] > 0, (
        "the filings the run never attempted were not reported as remaining")


def test_scattered_failures_do_not_abort_a_run_that_is_working(db, house):
    """Only a run that is failing consecutively is a blocked run."""
    house(12, errors=[0, 4, 9])
    result = sync.sync_house_ptrs(2026)

    assert not result["blocked"]
    assert result["filings_parsed"] == 9
    assert result["errors"] == 3


# ------------------------------------------------------- noticing a republish

def test_a_corrected_republish_is_read_again(db, house):
    house(1)
    sync.sync_house_ptrs(2026)
    house.state["fetched"].clear()

    house(1, filed="03/02/2026")    # the Clerk re-posted it under the same id
    sync.sync_house_ptrs(2026)

    assert house.state["fetched"] == ["0"], (
        "the filing was re-posted and the store went on serving the "
        "superseded numbers")


def test_what_was_read_is_recorded_beside_the_filing(db, house):
    house(1)
    sync.sync_house_ptrs(2026)

    with store.connect() as conn:
        content_hash, fetched_at = conn.execute(
            "SELECT content_hash, fetched_at FROM filings").fetchone()
    assert content_hash == "hash-of-0"
    assert fetched_at


# ------------------------------------------------- the years the nightly run asks for
#
# The nightly cron line passes no arguments, so whatever `main()` defaults to
# is what actually runs. It defaulted to the literal 2026 written into
# `deploy/docker-compose.yml`, which means every House PTR filed from
# 2027-01-01 lands in an annual ZIP nobody fetches -- while `coverage.complete`
# stays true, because every filing the job *knows about* was parsed. A store
# that is frozen and reports itself healthy is the failure these two tests
# exist to prevent.

@pytest.fixture
def years_asked_for(monkeypatch):
    """Record the House years `main()` asks for, fetching nothing."""
    asked = []

    def sync_house(year, **kwargs):
        asked.append(year)
        return {"blocked": False}

    monkeypatch.setattr(sync, "sync_house_ptrs", sync_house)
    monkeypatch.setattr(sync, "sync_senate_ptrs",
                        lambda **kwargs: {"blocked": False})
    monkeypatch.setattr(sync, "sync_senate_annuals",
                        lambda **kwargs: {"blocked": False})
    return asked


def test_house_with_no_years_asks_for_this_year_and_the_one_before(
        db, years_asked_for):
    """`--house` with no years is what the cron line runs.

    This year because that is where today's filings are; last year because the
    Clerk's ZIPs are per calendar year and a PTR for a December trade is filed
    in January, into the *previous* year's ZIP. Dropping the prior year would
    lose every January carry-over.
    """
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    assert sync.main(["--house"]) == 0
    assert years_asked_for == [now.year - 1, now.year]


def test_house_with_years_given_asks_for_exactly_those(db, years_asked_for):
    """The default must not be added to an explicit backfill."""
    assert sync.main(["--house", "2024", "2025"]) == 0
    assert years_asked_for == [2024, 2025]


def test_a_run_with_no_house_flag_asks_for_no_house_year(db, years_asked_for):
    """`--senate` alone still means the Senate alone."""
    assert sync.main(["--senate"]) == 0
    assert years_asked_for == []


def test_a_run_with_nothing_to_do_is_still_refused(db, years_asked_for):
    """Defaulting the years must not turn an argument-less run into a backfill
    of everything: a bare invocation is a mistake and has always said so."""
    with pytest.raises(SystemExit):
        sync.main([])


# ------------------------------------------------- what the nightly line runs
#
# `main()`'s default is only the nightly default if the compose command lets it
# be one. `command: ["--house", "2026", "--senate", "--days", "90"]` overrode it
# with the same expiring literal, and the cron line passes no arguments of its
# own, so the compose file is the last place the year could be pinned.

def test_the_composed_default_pins_no_calendar_year():
    """A year written into the compose command is a pipeline with an expiry
    date, and it expires silently -- see main()'s default above."""
    import re

    command = _composed_congress_sync_command()
    years = [word for word in command if re.fullmatch(r"(19|20)\d\d", str(word))]
    assert not years, (
        f"the congress-sync command pins {years}; House filings stop being "
        f"fetched the January after")


def test_the_composed_default_asks_for_holdings_as_well_as_trades():
    """Without `--senate-annual` the holdings are whatever the backfill left:
    the nightly run refreshes the trades and never the positions behind them.
    """
    command = _composed_congress_sync_command()
    assert "--senate-annual" in command, (
        "the nightly run ingests no annual reports, so holdings never refresh "
        "after the initial backfill")
    assert "--senate" in command and "--house" in command


def _composed_congress_sync_command():
    import pathlib

    import yaml

    compose = pathlib.Path(__file__).resolve().parent.parent \
        / "deploy" / "docker-compose.yml"
    service = yaml.safe_load(compose.read_text())["services"]["congress-sync"]
    return [str(word) for word in service["command"]]
