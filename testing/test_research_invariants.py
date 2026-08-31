"""The two rules the whole package rests on, checked against every function.

Both have been tested one place at a time, which is how `consensus_as_of` came
to be the only reader in the store that did not filter on `recorded_at` -- for
months, on the one field where the leak matters most. A sweep catches the next
one on the day it is added rather than the day someone reads a result and
wonders.

**Nothing is visible before it was written.** Every reader takes an `as_of` and
must hide a row recorded after it, not merely a row dated after it. The two
come apart exactly when a backfill happens, which is when it matters.

**A refusal is empty.** A function that cannot answer returns its numbers as
None and says why. Leaving a stale value beside `success: False` is worse than
raising, because the caller that forgets to check gets a number rather than a
crash.
"""
import inspect

import pytest

from research import (activist_watch, pit_store, scanner, scoring, spread,
                      sue, sue_cs)


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("NEMO_PIT_DB", str(tmp_path / "pit.db"))
    pit_store.init_schema()
    return pit_store


LATER = "2026-09-01T21:00:00Z"   # everything below is written "later"
PAST = "2026-03-03"              # ...and read as of here


def _write_everything_late(store):
    """One row in every table, all recorded well after the date read below."""
    store.record_bars("AAA", [{"trade_date": "2026-03-02", "open": 10.0,
                               "high": 10.0, "low": 10.0, "close": 10.0,
                               "volume": 1000}], recorded_at=LATER)
    store.record_corporate_action("AAA", "2026-03-02", "split", 2.0,
                                  recorded_at=LATER)
    store.record_universe("2026-03-02",
                          [{"ticker": "AAA", "cik": "1", "eligible": True}],
                          recorded_at=LATER)
    store.record_announcement("AAA", fiscal_period="2026Q1",
                              announced_date="2026-03-02", recorded_at=LATER)
    store.record_consensus("2026-03-02", "AAA", "2026Q1", eps_estimate=1.0,
                           eps_actual=1.2, recorded_at=LATER)
    store.record_paper_orders(
        "2026-03-02",
        [{"ticker": "AAA", "fiscal_period": "2026Q1", "side": "long",
          "sue": 2.0, "target_dollars": 100.0,
          "intended_session": "2026-03-03"}], recorded_at=LATER)
    store.record_activist_filings([{
        "accession": "0001-25-000001", "subject_ticker": "AAA",
        "filer": "Someone LP", "form": "SC 13D",
        "filing_date": "2026-03-02", "detected_at": LATER}],
        recorded_at=LATER)
    store.record_borrow_rates("2026-03-02",
                              [{"ticker": "AAA", "annual_rate": 0.03}],
                              recorded_at=LATER)


# --- nothing is visible before it was written -------------------------------

@pytest.mark.parametrize("name,call", [
    ("bars_as_of", lambda: pit_store.bars_as_of("AAA", PAST)),
    ("adjusted_bars", lambda: pit_store.adjusted_bars("AAA", PAST)),
    ("corporate_actions_as_of",
     lambda: pit_store.corporate_actions_as_of("AAA", PAST)),
    ("universe_as_of", lambda: pit_store.universe_as_of(PAST)),
    ("announcements_as_of", lambda: pit_store.announcements_as_of("AAA", PAST)),
    ("paper_orders_as_of", lambda: pit_store.paper_orders_as_of(PAST)),
    ("activist_filings_as_of",
     lambda: pit_store.activist_filings_as_of(PAST)),
    ("reporters_since",
     lambda: pit_store.reporters_since("2026-01-01", PAST)),
    ("filed_periods", lambda: sorted(pit_store.filed_periods(PAST))),
    ("cohort", lambda: sue_cs.cohort(as_of=PAST)),
])
def test_no_reader_sees_a_row_written_after_its_date(store, name, call):
    _write_everything_late(store)
    assert call() in ([], set(), ()), (
        f"{name} returned a row recorded on {LATER} to a reader standing on "
        f"{PAST}")


def test_the_scalar_readers_hide_it_too(store):
    _write_everything_late(store)
    assert pit_store.consensus_as_of("AAA", "2026Q1", PAST) is None
    assert pit_store.actual_as_of("AAA", "2026Q1", PAST) is None
    assert pit_store.has_consensus_history(PAST) is False
    # Borrow is the number most likely to be loaded in after the fact -- a
    # broker file for last month arrives next month -- so a leak here would
    # price every historical short with a rate nobody could have known.
    assert pit_store.borrow_rate_as_of("AAA", PAST) is None


def test_the_same_readers_do_see_it_afterwards(store):
    """The other half: the guard must hide a late write, not lose it."""
    _write_everything_late(store)
    later = "2026-09-02"
    assert pit_store.bars_as_of("AAA", later)
    assert pit_store.corporate_actions_as_of("AAA", later)
    assert pit_store.universe_as_of(later)
    assert pit_store.announcements_as_of("AAA", later)
    assert pit_store.paper_orders_as_of(later)
    assert pit_store.activist_filings_as_of(later)
    assert pit_store.consensus_as_of("AAA", "2026Q1", later)
    assert pit_store.actual_as_of("AAA", "2026Q1", later) == 1.2
    assert pit_store.borrow_rate_as_of("AAA", later)["annual_rate"] == 0.03


def test_every_as_of_reader_in_the_store_is_covered_here():
    """So a reader added later is not silently left out of the sweep above."""
    swept = {
        "bars_as_of", "adjusted_bars", "corporate_actions_as_of",
        "universe_as_of", "announcements_as_of", "paper_orders_as_of",
        "activist_filings_as_of", "consensus_as_of", "actual_as_of",
        "reporters_since", "filed_periods", "has_consensus_history",
        "borrow_rate_as_of",
        # Reads the run log, which records what the process did rather than
        # what the market did; see the test below for why it is keyed on the
        # date a run was FOR rather than on when it finished.
        "last_successful_run",
    }
    found = {n for n, fn in vars(pit_store).items()
             if callable(fn) and not n.startswith("_")
             and inspect.getmodule(fn) is pit_store
             and ("as_of" in inspect.signature(fn).parameters
                  or n.endswith("_as_of"))}
    missing = found - swept
    assert not missing, (
        f"these readers take an as_of and are not in the lookahead sweep: "
        f"{sorted(missing)}")


def test_the_run_log_reader_is_keyed_on_the_date_a_run_was_for(store):
    """And is deliberately outside the sweep above.

    Every other reader answers "what did the market look like on this date",
    where a row learned later is a leak. This one answers "was anything
    watching on this date", and a night recorded late still covers the night it
    was recorded for -- `missing_days` has counted it that way since it was
    written. Keying it on `finished_at` instead would report a backfilled night
    as a hole forever.
    """
    store.start_run("daily_bars", as_of_date="2026-03-02")
    store.finish_run(rows_written=1, status="ok")

    assert pit_store.last_successful_run("daily_bars", PAST) == "2026-03-02"
    assert pit_store.missing_days("daily_bars", "2026-03-02",
                                  "2026-03-02") == []


# --- a refusal is empty -----------------------------------------------------

def test_a_refused_time_series_surprise_carries_no_number(store, monkeypatch):
    monkeypatch.setattr(sue, "_fetch_cik_map", lambda: {})
    out = sue.sue_ts("NOPE", as_of=PAST)
    assert out["success"] is False and out["error"]
    for field in ("sue", "sigma", "eps", "eps_year_ago", "delta"):
        assert out[field] is None, f"{field} survived a refusal"


def test_a_refused_analyst_surprise_carries_no_number(store, monkeypatch):
    monkeypatch.setattr(sue, "_fetch_cik_map", lambda: {})
    out = sue.sue_af("NOPE", as_of=PAST)
    assert out["success"] is False and out["error"]
    for field in ("sue", "sigma", "surprise", "consensus"):
        assert out[field] is None, f"{field} survived a refusal"


def test_a_refused_cross_section_carries_no_number(store):
    out = sue_cs.surprise_rank("NOPE", as_of=PAST)
    assert out["success"] is False and out["error"]
    for field in ("z", "robust_z", "percentile", "surprise", "scaled_surprise"):
        assert out[field] is None, f"{field} survived a refusal"


def test_a_refused_spread_carries_no_number(store):
    out = spread.estimate_spread("NOPE", PAST, window=252)
    assert out["reason"]
    assert out["spread"] is None and out["spread_upper"] is None


def test_a_refused_cost_carries_no_number(store):
    out = spread.round_trip_cost("NOPE", PAST, 1000.0, window=252)
    assert out["reason"]
    for field in ("cost", "spread_cost", "impact_cost", "spread_used"):
        assert out[field] is None, f"{field} survived a refusal"


def test_a_refused_participation_is_not_within_the_limit(store):
    out = spread.participation_rate("NOPE", PAST, 1000.0, window=252)
    assert out["reason"]
    assert out["within_limit"] is False
    assert out["participation"] is None


def test_a_scan_with_nothing_to_scan_returns_empty_lists_not_none(store):
    out = scanner.scan(as_of=PAST)
    assert out["candidates"] == [] and out["rejected"] == []
    assert out["undetermined"] == []
    assert out["gross_target"] > 0


def test_scoring_an_empty_book_returns_the_empty_summary(store):
    out = scoring.score_orders(as_of=PAST)
    assert out["scored"] == [] and out["sample"] == 0
    assert out["calibrated"] is False
    assert out["mean_net_bps"] is None and out["hit_rate"] is None


def test_a_watch_pass_over_nothing_is_ok_with_no_findings(store, monkeypatch):
    monkeypatch.setattr(activist_watch, "_subject_filings",
                        lambda *a, **k: [], raising=False)
    out = activist_watch.watch_pass(tickers=[], as_of=PAST)
    assert out["status"] in ("ok", "failed")
    assert out.get("recorded", 0) == 0
