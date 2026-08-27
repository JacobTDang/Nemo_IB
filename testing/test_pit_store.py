"""The record you cannot go back and make.

A backtest can be run later on history bought later. A point-in-time record
cannot be back-filled: what analysts expected on a Tuesday is unrecoverable by
Thursday, and a company delisted in March is gone from the vendor by June. Every
day this store is not running is data that no amount of money buys back.

So the store has one job and one discipline. The job is to append what was
known, when it was known. The discipline is that nothing is ever overwritten,
because the moment a row can change, the record stops being evidence of what we
believed and becomes evidence of what we believe now.

Three failure modes this is built against, all of which have already been
observed in this project's own data:

  Survivorship -- the vendor drops delisted tickers, so a study of "every name
  that reported" silently becomes "every name that reported and survived".
  Recording forward is the only fix that costs nothing.

  Mutating adjustment -- yfinance's adjusted close changes as new splits and
  dividends land, so the same backtest returns different numbers in six months.
  Raw OHLC is immutable; adjustment factors are stored separately and applied
  at read time, so any past adjustment can be reconstructed exactly.

  Lookahead -- the subtle one. A bar for 3 March that we back-filled on 10 May
  must not be visible to a simulation standing on 4 March. Filtering on the
  trade date alone does not prevent this; only `recorded_at` does.
"""
import sqlite3
from datetime import datetime, timezone

import pytest

from research import pit_store


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("NEMO_PIT_DB", str(tmp_path / "pit.db"))
    pit_store.init_schema()
    return pit_store


# --- append-only ------------------------------------------------------------

def test_a_recorded_bar_is_never_overwritten(store):
    store.record_bars("AAPL", [
        {"trade_date": "2026-03-03", "open": 1.0, "high": 2.0,
         "low": 0.5, "close": 1.5, "volume": 1000},
    ], recorded_at="2026-03-03T21:00:00Z")

    # The vendor returns a different close for the same session.
    store.record_bars("AAPL", [
        {"trade_date": "2026-03-03", "open": 1.0, "high": 2.0,
         "low": 0.5, "close": 9.9, "volume": 1000},
    ], recorded_at="2026-03-10T21:00:00Z")

    bars = store.bars_as_of("AAPL", as_of="2026-06-01")
    assert len(bars) == 1
    assert bars[0]["close"] == 1.5, "the later value overwrote the original"


def test_a_vendor_revision_is_recorded_rather_than_swallowed(store):
    """A vendor changing history under us is itself information."""
    store.record_bars("AAPL", [
        {"trade_date": "2026-03-03", "open": 1.0, "high": 2.0,
         "low": 0.5, "close": 1.5, "volume": 1000},
    ], recorded_at="2026-03-03T21:00:00Z")
    store.record_bars("AAPL", [
        {"trade_date": "2026-03-03", "open": 1.0, "high": 2.0,
         "low": 0.5, "close": 9.9, "volume": 1000},
    ], recorded_at="2026-03-10T21:00:00Z")

    revisions = store.revisions(ticker="AAPL")
    assert len(revisions) == 1
    r = revisions[0]
    assert r["field"] == "close"
    assert r["old_value"] == 1.5
    assert r["new_value"] == 9.9


def test_recording_the_same_bar_twice_is_not_a_revision(store):
    """Idempotent re-runs are normal and must stay quiet."""
    row = [{"trade_date": "2026-03-03", "open": 1.0, "high": 2.0,
            "low": 0.5, "close": 1.5, "volume": 1000}]
    store.record_bars("AAPL", row, recorded_at="2026-03-03T21:00:00Z")
    store.record_bars("AAPL", row, recorded_at="2026-03-04T21:00:00Z")

    assert store.revisions(ticker="AAPL") == []
    assert len(store.bars_as_of("AAPL", as_of="2026-06-01")) == 1


# --- the anti-lookahead property -------------------------------------------

def test_a_query_cannot_see_a_row_recorded_after_its_date(store):
    """The whole point. A bar back-filled in May is invisible to a simulation
    standing in March, even though the bar's own date is in March."""
    store.record_bars("AAPL", [
        {"trade_date": "2026-03-03", "open": 1, "high": 1, "low": 1,
         "close": 1, "volume": 1},
    ], recorded_at="2026-05-10T21:00:00Z")

    assert store.bars_as_of("AAPL", as_of="2026-03-04") == []
    assert len(store.bars_as_of("AAPL", as_of="2026-05-11")) == 1


def test_a_future_trade_date_is_never_returned(store):
    """Each session recorded on the evening it happened, as the live job does."""
    store.record_bars("AAPL", [
        {"trade_date": "2026-03-03", "open": 1, "high": 1, "low": 1,
         "close": 1, "volume": 1},
    ], recorded_at="2026-03-03T21:00:00Z")
    store.record_bars("AAPL", [
        {"trade_date": "2026-03-04", "open": 2, "high": 2, "low": 2,
         "close": 2, "volume": 1},
    ], recorded_at="2026-03-04T21:00:00Z")

    dates = [b["trade_date"] for b in store.bars_as_of("AAPL", as_of="2026-03-03")]
    assert dates == ["2026-03-03"], "tomorrow's session leaked into today"


def test_consensus_is_read_as_it_stood_not_as_it_ended(store):
    """The field that cannot be reconstructed. An estimate revised upward in
    April must not leak into a March simulation."""
    store.record_consensus("2026-03-01", "AAPL", "2026Q2",
                           eps_estimate=1.10, analyst_count=8)
    store.record_consensus("2026-04-01", "AAPL", "2026Q2",
                           eps_estimate=1.45, analyst_count=11)

    march = store.consensus_as_of("AAPL", "2026Q2", as_of="2026-03-15")
    assert march["eps_estimate"] == 1.10
    assert march["analyst_count"] == 8

    april = store.consensus_as_of("AAPL", "2026Q2", as_of="2026-04-15")
    assert april["eps_estimate"] == 1.45


def test_consensus_before_any_snapshot_is_none_not_zero(store):
    store.record_consensus("2026-03-01", "AAPL", "2026Q2", eps_estimate=1.10)
    assert store.consensus_as_of("AAPL", "2026Q2", as_of="2026-02-01") is None


# --- raw prices, reconstructible adjustment ---------------------------------

def test_bars_are_stored_raw(store):
    """Adjusted prices mutate as new actions land; raw ones do not. Storing
    raw is what makes a result reproducible six months later."""
    with sqlite3.connect(store.db_path()) as conn:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(daily_bar)")}
    assert "adj_close" not in cols, (
        "an adjusted column in the store is a value that will change under you")
    assert {"open", "high", "low", "close", "volume"} <= cols


def test_a_split_is_recorded_so_adjustment_can_be_rebuilt(store):
    store.record_corporate_action("AAPL", "2026-06-10", "split", 10.0,
                                  recorded_at="2026-06-10T12:00:00Z")
    actions = store.corporate_actions_as_of("AAPL", as_of="2026-07-01")
    assert actions[0]["action_type"] == "split"
    assert actions[0]["value"] == 10.0


def test_an_action_recorded_later_does_not_alter_an_earlier_view(store):
    """The mutation bug, prevented: a split announced in June must not change
    what a May simulation computed."""
    store.record_corporate_action("AAPL", "2026-06-10", "split", 10.0,
                                  recorded_at="2026-06-10T12:00:00Z")
    assert store.corporate_actions_as_of("AAPL", as_of="2026-05-01") == []


# --- universe membership ----------------------------------------------------

def test_universe_membership_is_a_daily_fact(store):
    store.record_universe("2026-03-02", [
        {"ticker": "AAPL", "cik": "320193", "eligible": True},
        {"ticker": "TINY", "cik": "999", "eligible": False,
         "exclusion_reason": "below $500k median dollar volume"},
    ])
    members = store.universe_as_of("2026-03-02")
    assert [m["ticker"] for m in members if m["eligible"]] == ["AAPL"]
    excluded = [m for m in members if not m["eligible"]][0]
    assert "dollar volume" in excluded["exclusion_reason"]


def test_a_name_that_leaves_the_universe_stays_in_the_record(store):
    """Survivorship, prevented. A delisted name must remain visible on the
    dates it was eligible, or a later study silently drops it."""
    store.record_universe("2026-03-02", [{"ticker": "GONE", "cik": "1",
                                          "eligible": True}])
    store.record_universe("2026-04-02", [{"ticker": "AAPL", "cik": "320193",
                                          "eligible": True}])

    assert [m["ticker"] for m in store.universe_as_of("2026-03-02")] == ["GONE"]
    assert [m["ticker"] for m in store.universe_as_of("2026-04-02")] == ["AAPL"]


# --- knowing when the record has a hole -------------------------------------

def test_a_day_the_job_did_not_run_is_visible(store):
    """A gap you cannot see is worse than one you can: the first becomes a
    conclusion, the second becomes a caveat."""
    store.start_run("daily_bars", as_of_date="2026-03-02")
    store.finish_run(rows_written=500, status="ok")
    store.start_run("daily_bars", as_of_date="2026-03-04")
    store.finish_run(rows_written=500, status="ok")

    missing = store.missing_days("daily_bars", "2026-03-02", "2026-03-04")
    assert missing == ["2026-03-03"]


def test_a_failed_run_does_not_count_as_coverage(store):
    store.start_run("daily_bars", as_of_date="2026-03-03")
    store.finish_run(rows_written=0, status="failed", error="SEC returned 503")

    assert store.missing_days("daily_bars", "2026-03-03", "2026-03-03") == \
        ["2026-03-03"]


def test_an_unfinished_run_does_not_count_as_coverage(store):
    """A process killed mid-run leaves a started row and no finish."""
    store.start_run("daily_bars", as_of_date="2026-03-03")
    assert store.missing_days("daily_bars", "2026-03-03", "2026-03-03") == \
        ["2026-03-03"]


# --- announcements ----------------------------------------------------------

def test_an_announcement_is_keyed_on_fiscal_identity(store):
    """Not on the vendor's calendar bucket. get_earnings_surprises labels
    AMAT's 13 August print 2026-09-30; joining on that returns nothing."""
    store.record_announcement("AMAT", fiscal_period="2026Q3",
                              announced_date="2026-08-13", timing="amc")
    got = store.announcements_as_of("AMAT", as_of="2026-08-31")
    assert got[0]["fiscal_period"] == "2026Q3"
    assert got[0]["announced_date"] == "2026-08-13"
    assert got[0]["timing"] == "amc"


def test_timing_defaults_to_unknown_not_to_a_guess(store):
    """BMO vs AMC decides which bar is the gap. AMAT reported after the close,
    so its gap is 14 August; using the filing date gives -2.48% instead of
    -6.57%. A guess here is a wrong number, so the field says it does not know."""
    store.record_announcement("XYZ", fiscal_period="2026Q1",
                              announced_date="2026-05-01")
    today = datetime.now(timezone.utc).date().isoformat()
    assert store.announcements_as_of("XYZ", as_of=today)[0]["timing"] == "unknown"


def test_a_row_recorded_today_is_invisible_to_a_past_date(store):
    """The same discipline, stated directly. Anything written without an
    explicit stamp is known as of now -- so back-filling cannot silently
    become history."""
    store.record_announcement("XYZ", fiscal_period="2026Q1",
                              announced_date="2026-05-01")
    assert store.announcements_as_of("XYZ", as_of="2026-05-30") == []
