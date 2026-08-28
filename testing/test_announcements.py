"""When the market learned the number, from the filing that told it.

The store dates a quarter by its 10-Q, because that is where the XBRL lives.
The market learns the figure earlier, in the Item 2.02 8-K -- "Results of
Operations and Financial Condition", which is the earnings release itself.

Measured across 60 filings from 20 large caps, the gap is a median of 2 days
and a mean of 6.1, which sounds ignorable until the tail: JPM 31 days, TGT 28,
LOW 26, WMT 22, HD 22. Banks and retailers, systematically, and the drift being
measured is largest in the first days. A study entered off the 10-Q misses most
of it for exactly the names where the lag is worst.

Two things come out of the 8-K and neither is a guess. The filing date is when
the release went out, and `acceptance_datetime` says at what hour -- before the
open, during the session, or after the close -- which decides which bar is the
reaction. AMAT's 13 August print landed after the close, so its gap is the
14th, and dating it to the announcement gives -2.48% instead of -6.57%.
"""
from datetime import datetime, timezone

import pytest

from research import announcements, pit_store


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("NEMO_PIT_DB", str(tmp_path / "pit.db"))
    pit_store.init_schema()
    return pit_store


class Filing:
    def __init__(self, date_, items, accepted_utc):
        self.filing_date = date_
        self.items = items
        self.acceptance_datetime = (
            datetime.fromisoformat(accepted_utc).replace(tzinfo=timezone.utc)
            if accepted_utc else None)
        self.accession_no = f"acc-{date_}"


# --- which filing is the earnings release -----------------------------------

def test_only_an_item_2_02_filing_is_an_earnings_release(monkeypatch):
    monkeypatch.setattr(announcements, "_fetch_8k", lambda t, **kw: [
        Filing("2026-02-10", "5.02", "2026-02-10T14:00:00"),      # a director left
        Filing("2026-02-12", "2.02,9.01", "2026-02-12T21:05:00"),  # earnings
        Filing("2026-02-20", "8.01", "2026-02-20T14:00:00"),      # other events
    ])

    got = announcements.earnings_releases("AAA", as_of="2026-03-01")
    assert [r["announced_date"] for r in got] == ["2026-02-12"]


def test_a_filing_with_no_items_is_not_assumed_to_be_earnings(monkeypatch):
    monkeypatch.setattr(announcements, "_fetch_8k", lambda t, **kw: [
        Filing("2026-02-12", None, "2026-02-12T21:05:00")])
    assert announcements.earnings_releases("AAA", as_of="2026-03-01") == []


# --- what hour it landed ----------------------------------------------------

@pytest.mark.parametrize("utc,expected", [
    ("2026-02-12T11:30:00", "bmo"),   # 06:30 ET, before the open
    ("2026-02-12T14:29:00", "bmo"),   # 09:29 ET, a minute before
    ("2026-02-12T14:31:00", "dmh"),   # 09:31 ET, in the session
    ("2026-02-12T20:59:00", "dmh"),   # 15:59 ET, a minute before the close
    ("2026-02-12T21:05:00", "amc"),   # 16:05 ET, after the close
    ("2026-02-13T01:00:00", "amc"),   # 20:00 ET, well after
])
def test_the_hour_decides_which_session_reacts(monkeypatch, utc, expected):
    monkeypatch.setattr(announcements, "_fetch_8k", lambda t, **kw: [
        Filing("2026-02-12", "2.02", utc)])
    assert announcements.earnings_releases("AAA", "2026-03-01")[0]["timing"] \
        == expected


def test_the_hour_is_read_in_new_york_not_in_utc(monkeypatch):
    """Summer and winter differ by an hour, and the boundary is at 16:00 local.
    21:05 UTC is 16:05 in January -- after the close -- and 17:05 in July."""
    monkeypatch.setattr(announcements, "_fetch_8k", lambda t, **kw: [
        Filing("2026-01-15", "2.02", "2026-01-15T20:30:00")])   # 15:30 ET, in session
    assert announcements.earnings_releases("AAA", "2026-02-01")[0]["timing"] \
        == "dmh"

    monkeypatch.setattr(announcements, "_fetch_8k", lambda t, **kw: [
        Filing("2026-07-15", "2.02", "2026-07-15T20:30:00")])    # 16:30 ET, after
    assert announcements.earnings_releases("AAA", "2026-08-01")[0]["timing"] \
        == "amc"


def test_a_missing_acceptance_time_is_unknown_not_a_guess(monkeypatch):
    """Half the value of the record is knowing which bar reacted. A default
    would answer that question with a coin flip."""
    monkeypatch.setattr(announcements, "_fetch_8k", lambda t, **kw: [
        Filing("2026-02-12", "2.02", None)])
    assert announcements.earnings_releases("AAA", "2026-03-01")[0]["timing"] \
        == "unknown"


# --- matching a release to the quarter it reports ---------------------------

QUARTERS = {
    "2026Q1": {"period_end": "2026-01-31", "known_at": "2026-03-11"},
    "2025Q4": {"period_end": "2025-10-31", "known_at": "2025-12-03"},
}


def test_a_release_is_matched_to_the_quarter_it_falls_between(monkeypatch):
    """After the quarter closed and no later than the filing that reported it.
    Anything else is a different quarter's release."""
    monkeypatch.setattr(announcements, "_fetch_8k", lambda t, **kw: [
        Filing("2026-02-11", "2.02", "2026-02-11T21:05:00"),
        Filing("2025-11-19", "2.02", "2025-11-19T12:00:00"),
    ])
    monkeypatch.setattr(announcements, "_quarters",
                        lambda t, as_of=None: QUARTERS)

    got = announcements.for_quarters("AAA", as_of="2026-03-20")
    assert got["2026Q1"]["announced_date"] == "2026-02-11"
    assert got["2026Q1"]["timing"] == "amc"
    assert got["2025Q4"]["announced_date"] == "2025-11-19"
    assert got["2025Q4"]["timing"] == "bmo"


def test_a_quarter_with_no_release_is_absent_rather_than_dated_by_its_filing(
        monkeypatch):
    """Falling back to the 10-Q would put the announcement weeks late and
    nothing would say so -- which is the error this module exists to remove."""
    monkeypatch.setattr(announcements, "_fetch_8k", lambda t, **kw: [])
    monkeypatch.setattr(announcements, "_quarters",
                        lambda t, as_of=None: QUARTERS)
    assert announcements.for_quarters("AAA", as_of="2026-03-20") == {}


def test_a_release_after_the_filing_belongs_to_the_next_quarter(monkeypatch):
    monkeypatch.setattr(announcements, "_fetch_8k", lambda t, **kw: [
        Filing("2026-05-20", "2.02", "2026-05-20T21:05:00")])
    monkeypatch.setattr(announcements, "_quarters",
                        lambda t, as_of=None: QUARTERS)
    assert announcements.for_quarters("AAA", as_of="2026-06-01") == {}


# --- recording it -----------------------------------------------------------

def test_a_backfilled_announcement_is_known_on_its_own_date(store, monkeypatch):
    monkeypatch.setattr(announcements, "_fetch_8k", lambda t, **kw: [
        Filing("2026-02-11", "2.02", "2026-02-11T21:05:00")])
    monkeypatch.setattr(announcements, "_quarters",
                        lambda t, as_of=None: QUARTERS)

    announcements.backfill(["AAA"], as_of="2026-03-20")

    assert pit_store.announcements_as_of("AAA", "2026-02-10") == []
    got = pit_store.announcements_as_of("AAA", "2026-02-12")
    assert [(a["fiscal_period"], a["announced_date"], a["timing"])
            for a in got] == [("2026Q1", "2026-02-11", "amc")]


def test_backfilling_twice_writes_nothing_the_second_time(store, monkeypatch):
    monkeypatch.setattr(announcements, "_fetch_8k", lambda t, **kw: [
        Filing("2026-02-11", "2.02", "2026-02-11T21:05:00")])
    monkeypatch.setattr(announcements, "_quarters",
                        lambda t, as_of=None: QUARTERS)

    assert announcements.backfill(["AAA"], as_of="2026-03-20")["written"] == 1
    assert announcements.backfill(["AAA"], as_of="2026-03-20")["written"] == 0


def test_a_name_edgar_refuses_is_reported_not_swallowed(store, monkeypatch):
    def boom(ticker, **kw):
        raise ConnectionError("EDGAR returned 503")

    monkeypatch.setattr(announcements, "_fetch_8k", boom)
    out = announcements.backfill(["AAA"], as_of="2026-03-20")
    assert out["written"] == 0
    assert "503" in out["failed"][0]


# --- the awkward filings ----------------------------------------------------

def test_a_company_that_files_2_02_for_something_else(monkeypatch):
    """2.02 is 'Results of Operations and Financial Condition', which is
    normally the earnings release and occasionally a mid-quarter update or a
    restatement. Two inside one window means the earlier one is the release
    the market reacted to; a later correction does not move the announcement.
    """
    monkeypatch.setattr(announcements, "_fetch_8k", lambda t, **kw: [
        Filing("2026-02-11", "2.02", "2026-02-11T21:05:00"),
        Filing("2026-02-26", "2.02,4.02", "2026-02-26T14:00:00"),
    ])
    monkeypatch.setattr(announcements, "_quarters",
                        lambda t, as_of=None: QUARTERS)

    got = announcements.for_quarters("AAA", as_of="2026-03-20")
    assert got["2026Q1"]["announced_date"] == "2026-02-11"


def test_a_release_on_the_period_end_itself_is_not_this_quarters(monkeypatch):
    """A quarter cannot be reported on the day it closes. Anything dated at
    the close belongs to the quarter before it."""
    monkeypatch.setattr(announcements, "_fetch_8k", lambda t, **kw: [
        Filing("2026-01-31", "2.02", "2026-01-31T21:05:00")])
    monkeypatch.setattr(announcements, "_quarters",
                        lambda t, as_of=None: QUARTERS)
    assert "2026Q1" not in announcements.for_quarters("AAA", "2026-03-20")


def test_a_release_on_the_filing_date_itself_counts(monkeypatch):
    """Plenty of companies file the 8-K and the 10-Q the same day -- MSFT,
    NVDA. Excluding the boundary would lose them entirely."""
    monkeypatch.setattr(announcements, "_fetch_8k", lambda t, **kw: [
        Filing("2026-03-11", "2.02", "2026-03-11T21:05:00")])
    monkeypatch.setattr(announcements, "_quarters",
                        lambda t, as_of=None: QUARTERS)
    got = announcements.for_quarters("AAA", "2026-03-20")
    assert got["2026Q1"]["announced_date"] == "2026-03-11"


def test_a_release_after_the_read_date_is_invisible(monkeypatch):
    """The lookahead rule, at the one place this module could break it."""
    monkeypatch.setattr(announcements, "_fetch_8k", lambda t, **kw: [
        Filing("2026-02-11", "2.02", "2026-02-11T21:05:00"),
        Filing("2026-05-20", "2.02", "2026-05-20T21:05:00"),
    ])
    got = announcements.earnings_releases("AAA", as_of="2026-03-01")
    assert [r["announced_date"] for r in got] == ["2026-02-11"]


def test_the_item_string_is_matched_on_the_whole_code(monkeypatch):
    """'12.02' and '2.021' are not item 2.02, and a substring test would take
    them. No such item exists today, which is exactly why it is worth pinning
    before one does."""
    monkeypatch.setattr(announcements, "_fetch_8k", lambda t, **kw: [
        Filing("2026-02-11", "12.02", "2026-02-11T21:05:00"),
        Filing("2026-02-12", "2.021", "2026-02-12T21:05:00"),
    ])
    assert announcements.earnings_releases("AAA", "2026-03-01") == []


def test_items_arriving_as_a_list_are_handled(monkeypatch):
    """edgartools returns a comma-joined string today. A list would silently
    match nothing, and 'not one release anywhere' is indistinguishable from a
    company that files none."""
    monkeypatch.setattr(announcements, "_fetch_8k", lambda t, **kw: [
        Filing("2026-02-11", ["2.02", "9.01"], "2026-02-11T21:05:00")])
    got = announcements.earnings_releases("AAA", "2026-03-01")
    assert [r["announced_date"] for r in got] == ["2026-02-11"]


def test_a_naive_acceptance_time_is_not_assumed_to_be_utc(monkeypatch):
    """A timestamp with no zone cannot be placed on the clock that matters, and
    guessing puts the boundary an hour or five out."""
    class Naive:
        filing_date = "2026-02-11"
        items = "2.02"
        accession_no = "x"
        acceptance_datetime = __import__("datetime").datetime(2026, 2, 11, 21, 5)

    monkeypatch.setattr(announcements, "_fetch_8k", lambda t, **kw: [Naive()])
    assert announcements.earnings_releases("AAA", "2026-03-01")[0]["timing"] \
        == "unknown"
