"""Direct tests for the helpers everything else is built on.

Each of these had no test of its own. Most were exercised somewhere -- through
a scan, through a recorder -- which catches a crash and not a wrong answer. A
helper that returns the wrong set quietly makes the thing above it wrong in a
way that reads as a market condition: a narrowing that drops a name looks like
a company that did not report, and a dedup that misses looks like a fresh
signal.

So these assert on exact outputs, including the boundaries, rather than on the
call not raising.
"""
from datetime import date, timedelta

import pytest

from research import daily_job, pit_store, scoring, sue_cs


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("NEMO_PIT_DB", str(tmp_path / "pit.db"))
    pit_store.init_schema()
    return pit_store


def _con(store, ticker, period, day, estimate=1.0, actual=None, source="recorded"):
    store.record_consensus(day, ticker, period, eps_estimate=estimate,
                           eps_actual=actual, recorded_at=f"{day}T21:00:00Z",
                           source=source)


def _bar(store, ticker, day, close=100.0, volume=1_000_000):
    store.record_bars(ticker, [{"trade_date": day, "open": close,
                                "high": close, "low": close, "close": close,
                                "volume": volume}],
                      recorded_at=f"{day}T21:00:00Z")


# --- coverage_status --------------------------------------------------------

@pytest.mark.parametrize("covered,requested,expected", [
    (0, 0, "ok"),        # asked for nothing, got nothing
    (0, 10, "failed"),   # asked and got none
    (1, 10, "partial"),
    (9, 10, "partial"),
    (10, 10, "ok"),
    (11, 10, "ok"),      # more than asked: still complete, never "partial"
])
def test_coverage_status_boundaries(covered, requested, expected):
    assert daily_job.coverage_status(covered, requested) == expected


# --- reporters_since --------------------------------------------------------

def test_reporters_since_returns_only_names_with_an_actual(store):
    _con(store, "HAS", "2026Q1", "2026-03-02", actual=1.2)
    _con(store, "NONE", "2026Q1", "2026-03-02", actual=None)

    got = pit_store.reporters_since("2026-02-01", "2026-03-03")
    assert [r["ticker"] for r in got] == ["HAS"]


def test_reporters_since_respects_both_ends_of_the_window(store):
    _con(store, "EARLY", "2025Q4", "2026-01-05", actual=1.0)
    _con(store, "INSIDE", "2026Q1", "2026-02-15", actual=1.0)
    _con(store, "LATE", "2026Q2", "2026-04-01", actual=1.0)

    got = {r["ticker"] for r in
           pit_store.reporters_since("2026-02-01", "2026-03-03")}
    assert got == {"INSIDE"}


def test_reporters_since_hides_a_row_recorded_after_the_read(store):
    store.record_consensus("2026-02-15", "LATER", "2026Q1", eps_actual=1.0,
                           recorded_at="2026-06-01T21:00:00Z")
    assert pit_store.reporters_since("2026-02-01", "2026-03-03") == []


def test_reporters_since_collapses_a_name_reported_on_many_days(store):
    """The recorder writes a row a day while a print sits in the calendar
    window. That is one reporter, not five, or the scan considers it five
    times and the cohort it is ranked against is five times too big."""
    for day in ("2026-03-02", "2026-03-03", "2026-03-04"):
        _con(store, "AAA", "2026Q1", day, actual=1.2)

    got = pit_store.reporters_since("2026-02-01", "2026-03-05")
    assert len(got) == 1
    assert got[0]["as_of_date"] == "2026-03-04"


def test_reporters_since_keeps_two_quarters_of_one_name_apart(store):
    _con(store, "AAA", "2026Q1", "2026-03-02", actual=1.2)
    _con(store, "AAA", "2026Q2", "2026-06-02", actual=1.4)

    got = pit_store.reporters_since("2026-01-01", "2026-07-01")
    assert {r["fiscal_period"] for r in got} == {"2026Q1", "2026Q2"}


# --- has_consensus_history --------------------------------------------------

def test_has_consensus_history_is_false_on_an_empty_store(store):
    assert pit_store.has_consensus_history("2026-03-03") is False


def test_has_consensus_history_is_true_once_anything_is_written(store):
    _con(store, "AAA", "2026Q1", "2026-03-02")
    assert pit_store.has_consensus_history("2026-03-03") is True


def test_has_consensus_history_is_false_before_the_first_write(store):
    """The difference between a quiet season and a store that was not yet
    watching, which is the whole reason the function exists."""
    _con(store, "AAA", "2026Q1", "2026-03-02")
    assert pit_store.has_consensus_history("2026-03-01") is False


# --- filed_periods ----------------------------------------------------------

def _order(store, ticker, period, decided, accepted=True):
    row = {"ticker": ticker, "fiscal_period": period, "side": "long",
           "sue": 2.0, "expected_edge_bps": 30.0, "cost_bps": 5.0,
           "net_edge_bps": 25.0, "target_dollars": 5000.0,
           "intended_session": decided, "rank": 1}
    store.record_paper_orders(decided, [row] if accepted else [],
                              [] if accepted else [{**row, "reason": "x"}],
                              recorded_at=f"{decided}T21:00:00Z")


def test_filed_periods_returns_pairs_not_tickers(store):
    _order(store, "AAA", "2026Q1", "2026-03-02")
    assert pit_store.filed_periods("2026-03-03") == {("AAA", "2026Q1")}


def test_filed_periods_excludes_the_day_being_asked_about(store):
    _order(store, "AAA", "2026Q1", "2026-03-03")
    assert pit_store.filed_periods("2026-03-03") == set()
    assert pit_store.filed_periods("2026-03-04") == {("AAA", "2026Q1")}


def test_filed_periods_ignores_rejections(store):
    """A name considered and turned down was not acted on, and must stay
    eligible for the next session."""
    _order(store, "AAA", "2026Q1", "2026-03-02", accepted=False)
    assert pit_store.filed_periods("2026-03-04") == set()


def test_filed_periods_hides_orders_recorded_after_the_read(store):
    store.record_paper_orders(
        "2026-03-02",
        [{"ticker": "AAA", "fiscal_period": "2026Q1", "side": "long",
          "sue": 2.0, "target_dollars": 1.0, "intended_session": "2026-03-03"}],
        recorded_at="2026-06-01T21:00:00Z")
    assert pit_store.filed_periods("2026-03-04") == set()


# --- iso_utc ----------------------------------------------------------------

@pytest.mark.parametrize("value,expected", [
    (None, None),
    ("", None),
    ("2026-03-02", "2026-03-02T00:00:00Z"),
    ("2026-03-02T14:30:00", "2026-03-02T14:30:00Z"),
    ("2026-03-02T14:30:00Z", "2026-03-02T14:30:00Z"),
    ("2026-03-02 14:30:00", "2026-03-02T14:30:00Z"),
])
def test_iso_utc_normalises(value, expected):
    assert pit_store.iso_utc(value) == expected


def test_iso_utc_converts_an_offset_to_utc(store):
    got = pit_store.iso_utc("2026-03-02T09:30:00-05:00")
    assert got == "2026-03-02T14:30:00Z"


def test_iso_utc_refuses_nonsense_rather_than_inventing_a_time():
    """It raises rather than returning None, which is the stronger answer: a
    null here would say SEC supplied no acceptance time, when what happened is
    that we could not read the one it did supply."""
    with pytest.raises(ValueError, match="unparseable"):
        pit_store.iso_utc("not a date")


# --- known_activist_accessions ---------------------------------------------

def test_known_accessions_starts_empty_and_records_what_was_seen(store):
    assert pit_store.known_activist_accessions("AAPL") == set()

    pit_store.record_activist_filings([{
        "accession": "0001-25-000001", "subject_ticker": "AAPL",
        "filer": "Someone LP", "form": "SC 13D",
        "filing_date": "2026-03-02", "detected_at": "2026-03-02T15:00:00Z"}])

    assert pit_store.known_activist_accessions("AAPL") == {"0001-25-000001"}


def test_known_accessions_are_scoped_to_the_subject(store):
    pit_store.record_activist_filings([{
        "accession": "0001-25-000001", "subject_ticker": "AAPL",
        "filer": "Someone LP", "form": "SC 13D",
        "filing_date": "2026-03-02", "detected_at": "2026-03-02T15:00:00Z"}])
    assert pit_store.known_activist_accessions("MSFT") == set()


# --- scoring.fill -----------------------------------------------------------

def _b(day, open_):
    return {"trade_date": day, "open": open_, "high": open_, "low": open_,
            "close": open_, "volume": 1}


def test_fill_computes_a_long_from_open_to_open():
    row = scoring.fill({"ticker": "A", "side": "long", "sue": 2.0,
                        "cost_bps": 10.0, "as_of_date": "2026-03-01",
                        "target_dollars": 100.0},
                       _b("2026-03-02", 100.0), _b("2026-03-20", 110.0))
    assert row["gross_bps"] == pytest.approx(1000.0)
    assert row["net_bps"] == pytest.approx(990.0)
    assert row["entry_session"] == "2026-03-02"
    assert row["exit_session"] == "2026-03-20"


def test_fill_inverts_a_short():
    row = scoring.fill({"ticker": "A", "side": "short", "sue": -2.0,
                        "cost_bps": 0.0, "as_of_date": "2026-03-01",
                        "target_dollars": 100.0},
                       _b("2026-03-02", 100.0), _b("2026-03-20", 90.0))
    assert row["gross_bps"] == pytest.approx(1000.0)


def test_fill_refuses_a_missing_price():
    assert scoring.fill({"ticker": "A", "side": "long", "sue": 1.0,
                         "cost_bps": 0.0, "as_of_date": "x",
                         "target_dollars": 1.0},
                        _b("2026-03-02", 0.0), _b("2026-03-20", 110.0)) is None


def test_fill_treats_a_missing_cost_as_zero_not_as_a_crash():
    row = scoring.fill({"ticker": "A", "side": "long", "sue": 1.0,
                        "as_of_date": "x", "target_dollars": 1.0},
                       _b("2026-03-02", 100.0), _b("2026-03-20", 100.0))
    assert row["cost_bps"] == 0.0
    assert row["net_bps"] == pytest.approx(0.0)


# --- sue_cs.cohort ----------------------------------------------------------

def test_cohort_needs_both_legs_and_a_price(store):
    _con(store, "FULL", "2026Q1", "2026-03-02", estimate=1.0, actual=1.2)
    _bar(store, "FULL", "2026-03-02")
    _con(store, "NOPRICE", "2026Q1", "2026-03-02", estimate=1.0, actual=1.2)
    _con(store, "NOACTUAL", "2026Q1", "2026-03-02", estimate=1.0)
    _bar(store, "NOACTUAL", "2026-03-02")

    got = {c["ticker"] for c in sue_cs.cohort(as_of="2026-03-03")}
    assert got == {"FULL"}


def test_cohort_scales_by_the_price_on_the_read_date(store):
    _con(store, "AAA", "2026Q1", "2026-03-02", estimate=1.0, actual=1.5)
    _bar(store, "AAA", "2026-03-02", close=50.0)

    row = sue_cs.cohort(as_of="2026-03-03")[0]
    assert row["surprise"] == pytest.approx(0.5)
    assert row["scaled_surprise"] == pytest.approx(0.5 / 50.0)


def test_cohort_window_is_measured_from_the_read_date(store):
    _con(store, "OLD", "2025Q3", "2025-10-01", estimate=1.0, actual=1.2)
    _bar(store, "OLD", "2025-10-01")
    _con(store, "NEW", "2026Q1", "2026-03-02", estimate=1.0, actual=1.2)
    _bar(store, "NEW", "2026-03-02")

    assert {c["ticker"] for c in
            sue_cs.cohort(as_of="2026-03-03", window_days=45)} == {"NEW"}
    assert {c["ticker"] for c in
            sue_cs.cohort(as_of="2025-10-02", window_days=45)} == {"OLD"}


def test_cohort_is_empty_rather_than_raising_on_an_empty_store(store):
    assert sue_cs.cohort(as_of="2026-03-03") == []


# --- the network seams ------------------------------------------------------
#
# build_signals and prints_from_filings each make one EDGAR pass per name and
# had no test at all. They are the two places a replay's whole input is built,
# so a shape change upstream turns into an empty run rather than an error.

def test_build_signals_keeps_only_quarters_that_have_a_surprise(monkeypatch):
    from research import replay, sue

    monkeypatch.setattr(sue, "sue_ts_history", lambda t, as_of=None: {
        "signals": [
            {"fiscal_period": "2026Q1", "known_at": "2026-04-30", "sue": 1.5},
            {"fiscal_period": "2025Q4", "known_at": "2026-01-30", "sue": None},
            {"fiscal_period": "2025Q3", "known_at": "2025-10-30", "sue": -2.0},
        ]})

    out = replay.build_signals(["AAA"], as_of="2026-06-01")

    assert out["tickers"] == 1
    assert out["signals"] == 2
    assert replay._signal_for("AAA", "2026-06-01")["fiscal_period"] == "2026Q1"


def test_build_signals_tolerates_the_other_key_name(monkeypatch):
    """sue_ts_history has been seen returning its rows under `quarters`.
    Reading only `signals` would leave the table silently empty."""
    from research import replay, sue

    monkeypatch.setattr(sue, "sue_ts_history", lambda t, as_of=None: {
        "quarters": [{"fiscal_period": "2026Q1", "known_at": "2026-04-30",
                      "sue": 1.5}]})

    assert replay.build_signals(["AAA"], as_of="2026-06-01")["signals"] == 1


def test_build_signals_replaces_rather_than_accumulating(monkeypatch):
    from research import replay, sue

    monkeypatch.setattr(sue, "sue_ts_history", lambda t, as_of=None: {
        "signals": [{"fiscal_period": "2026Q1", "known_at": "2026-04-30",
                     "sue": 1.5}]})
    replay.build_signals(["AAA"], as_of="2026-06-01")
    replay.build_signals(["BBB"], as_of="2026-06-01")

    assert replay._signal_for("AAA", "2026-06-01")["success"] is False
    assert replay._signal_for("BBB", "2026-06-01")["success"] is True


def test_prints_from_filings_carries_the_filing_date_not_the_period_end(
        monkeypatch):
    from research import replay, sue

    monkeypatch.setattr(sue, "eps_series", lambda t, as_of=None: {
        "success": True,
        "quarters": [{"fiscal_period": "2026Q1", "eps": 1.25,
                      "period_end": "2026-03-31",
                      "known_at": "2026-04-30T00:00:00Z"}]})

    out = replay.prints_from_filings(["AAA"], as_of="2026-06-01")
    assert out == [{"ticker": "AAA", "fiscal_period": "2026Q1",
                    "known_at": "2026-04-30", "eps": 1.25}]


def test_prints_from_filings_skips_a_name_with_no_series(monkeypatch):
    from research import replay, sue

    monkeypatch.setattr(sue, "eps_series", lambda t, as_of=None: {
        "success": False, "error": "no filings"})
    assert replay.prints_from_filings(["AAA"], as_of="2026-06-01") == []


def test_prints_from_filings_drops_a_quarter_with_no_date(monkeypatch):
    """Undated, it cannot be placed in time, and a print with no date is a
    print the scanner would narrow on at the wrong moment."""
    from research import replay, sue

    monkeypatch.setattr(sue, "eps_series", lambda t, as_of=None: {
        "success": True,
        "quarters": [{"fiscal_period": "2026Q1", "eps": 1.25,
                      "period_end": "2026-03-31", "known_at": None}]})
    assert replay.prints_from_filings(["AAA"], as_of="2026-06-01") == []


# --- the thin ones ----------------------------------------------------------

def test_latency_report_on_an_empty_store_is_empty_not_an_error(store):
    from research import activist_watch

    out = activist_watch.latency_report(as_of="2026-03-03")
    assert out["events"] == 0
    assert out["live_detections"] == 0
    assert out["backfilled"] == 0
    # None, not zero. A latency of zero would say every filing was caught
    # instantly; there were no filings.
    assert out["median_latency_seconds"] is None
    assert out["worst_latency_seconds"] is None


def test_db_path_follows_the_environment(tmp_path, monkeypatch):
    target = tmp_path / "elsewhere.db"
    monkeypatch.setenv("NEMO_PIT_DB", str(target))
    assert pit_store.db_path() == str(target)


def test_summarise_carries_the_caveats_even_with_results(store):
    from research import replay

    out = replay.summarise([{"ticker": "A", "sue": 2.0, "net_bps": 10.0,
                             "gross_bps": 12.0, "cost_bps": 2.0}])
    assert out["caveats"]
    assert out["sample"] == 1
