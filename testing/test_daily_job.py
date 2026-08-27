"""The job that starts the clock, and the ways it could lie about having run.

The store is only worth having if something feeds it every day. This is that
something, and its failure modes are not "it crashed" -- a crash is loud and
gets fixed. The dangerous ones are the quiet kind:

  A run that fetched nothing and logged success, so the gap is invisible and
  six months later a hole in the record reads as a market that did not trade.

  A ticker that returned no data being recorded as a session with no volume,
  which is the absent-is-not-zero defect this codebase has now fixed in eight
  other places.

  Back-filled history stamped as though it had been known all along, which
  quietly re-introduces the lookahead the store exists to prevent.

  A universe screen that drops a name without saying why, so a later reader
  cannot tell an excluded name from one that never existed.

Every test here is about one of those, not about the happy path.
"""
import pytest

from research import daily_job, pit_store


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("NEMO_PIT_DB", str(tmp_path / "pit.db"))
    pit_store.init_schema()
    return pit_store


def _bars(*dates, close=10.0, volume=1_000_000):
    return [{"trade_date": d, "open": close, "high": close, "low": close,
             "close": close, "volume": volume} for d in dates]


# --- a run that did not work must not look like one that did ---------------

def test_a_run_that_fetched_nothing_is_not_recorded_as_ok(store, monkeypatch):
    monkeypatch.setattr(daily_job, "_fetch_bars", lambda tickers, **kw: {})

    daily_job.record_daily_bars(["AAPL", "MSFT"], as_of="2026-03-03")

    assert store.missing_days("daily_bars", "2026-03-03", "2026-03-03") == \
        ["2026-03-03"], "an empty fetch was logged as coverage"


def test_a_partial_fetch_is_recorded_as_partial(store, monkeypatch):
    """Half the universe failing is neither success nor failure, and calling
    it either one loses the number that matters."""
    monkeypatch.setattr(daily_job, "_fetch_bars",
                        lambda tickers, **kw: {"AAPL": _bars("2026-03-03")})

    daily_job.record_daily_bars(["AAPL", "MSFT"], as_of="2026-03-03")

    run = daily_job.last_run("daily_bars")
    assert run["status"] == "partial"
    assert "1" in str(run["error"]) or run["rows_written"] == 1


def test_an_upstream_exception_is_reported_not_swallowed(store, monkeypatch):
    def boom(tickers, **kw):
        raise ConnectionError("Yahoo returned 502")

    monkeypatch.setattr(daily_job, "_fetch_bars", boom)
    daily_job.record_daily_bars(["AAPL"], as_of="2026-03-03")

    run = daily_job.last_run("daily_bars")
    assert run["status"] == "failed"
    assert "502" in str(run["error"])
    assert store.missing_days("daily_bars", "2026-03-03", "2026-03-03") == \
        ["2026-03-03"]


# --- absent is not zero -----------------------------------------------------

def test_a_ticker_with_no_data_gets_no_bar(store, monkeypatch):
    """A session with volume 0 says the market opened and nobody traded. A
    ticker the vendor did not answer for says nothing at all."""
    monkeypatch.setattr(daily_job, "_fetch_bars",
                        lambda tickers, **kw: {"AAPL": _bars("2026-03-03"),
                                               "GONE": []})

    daily_job.record_daily_bars(["AAPL", "GONE"], as_of="2026-03-03")

    assert store.bars_as_of("GONE", as_of="2026-03-04") == []
    assert len(store.bars_as_of("AAPL", as_of="2026-03-04")) == 1


# --- back-filled history must stay back-filled ------------------------------

def test_bootstrapped_history_is_stamped_now_not_backdated(store, monkeypatch):
    """Pulling two years of history on day one is fine and useful. Stamping it
    as though it had been known at the time would hand every future simulation
    the lookahead this store exists to prevent."""
    monkeypatch.setattr(daily_job, "_fetch_bars",
                        lambda tickers, **kw: {
                            "AAPL": _bars("2024-01-02", "2024-01-03")})

    daily_job.bootstrap_history(["AAPL"], lookback_days=730,
                                as_of="2026-03-03")

    assert store.bars_as_of("AAPL", as_of="2024-06-01") == [], (
        "back-filled bars were visible to a date before they were fetched")
    assert len(store.bars_as_of("AAPL", as_of="2026-03-03")) == 2


# --- the universe screen must explain itself --------------------------------

def test_an_excluded_name_carries_its_reason(store, monkeypatch):
    monkeypatch.setattr(daily_job, "_fetch_sec_tickers", lambda: [
        {"ticker": "BIG", "cik": "1", "name": "Big Co"},
        {"ticker": "THIN", "cik": "2", "name": "Thin Co"},
    ])
    # 60 sessions each: enough history that the screen judges liquidity
    # rather than coverage. Those are different exclusions and the next test
    # covers the other one.
    sessions = [f"2026-01-{d:02d}" for d in range(1, 32)] + \
               [f"2026-02-{d:02d}" for d in range(1, 30)]
    store.record_bars("BIG", _bars(*sessions, volume=1_000_000),
                      recorded_at="2026-03-02T21:00:00Z")
    store.record_bars("THIN", _bars(*sessions, volume=10),
                      recorded_at="2026-03-02T21:00:00Z")

    daily_job.refresh_universe(as_of="2026-03-03")

    members = {m["ticker"]: m for m in store.universe_as_of("2026-03-03")}
    assert members["BIG"]["eligible"] is True
    assert members["THIN"]["eligible"] is False
    assert members["THIN"]["exclusion_reason"], "excluded with no reason given"
    assert "volume" in members["THIN"]["exclusion_reason"].lower()


def test_a_name_with_no_history_is_excluded_for_that_reason(store, monkeypatch):
    """Not for illiquidity. A name we have never seen and a name that barely
    trades are different facts and a screen that conflates them is unauditable."""
    monkeypatch.setattr(daily_job, "_fetch_sec_tickers", lambda: [
        {"ticker": "NEW", "cik": "3", "name": "New Co"}])

    daily_job.refresh_universe(as_of="2026-03-03")

    member = store.universe_as_of("2026-03-03")[0]
    assert member["eligible"] is False
    assert "histor" in member["exclusion_reason"].lower()


# --- consensus: the series with a clock on it -------------------------------

def test_consensus_is_snapshotted_for_upcoming_reporters(store, monkeypatch):
    monkeypatch.setattr(daily_job, "_fetch_calendar",
                        lambda start, end: [
                            {"ticker": "AAPL", "fiscal_period": "2026Q2",
                             "eps_estimate": 1.42, "analyst_count": 12}])

    daily_job.record_consensus_snapshots(as_of="2026-03-03", horizon_days=10)

    snap = store.consensus_as_of("AAPL", "2026Q2", as_of="2026-03-03")
    assert snap["eps_estimate"] == 1.42
    assert snap["analyst_count"] == 12


def test_a_calendar_entry_without_an_estimate_is_not_a_zero(store, monkeypatch):
    monkeypatch.setattr(daily_job, "_fetch_calendar",
                        lambda start, end: [
                            {"ticker": "AAPL", "fiscal_period": "2026Q2",
                             "eps_estimate": None, "analyst_count": 0}])

    daily_job.record_consensus_snapshots(as_of="2026-03-03", horizon_days=10)

    snap = store.consensus_as_of("AAPL", "2026Q2", as_of="2026-03-03")
    assert snap is None or snap["eps_estimate"] is None, (
        "a missing estimate was stored as a number")


# --- idempotence ------------------------------------------------------------

def test_running_twice_in_a_day_changes_nothing(store, monkeypatch):
    monkeypatch.setattr(daily_job, "_fetch_bars",
                        lambda tickers, **kw: {"AAPL": _bars("2026-03-03")})

    daily_job.record_daily_bars(["AAPL"], as_of="2026-03-03")
    daily_job.record_daily_bars(["AAPL"], as_of="2026-03-03")

    assert len(store.bars_as_of("AAPL", as_of="2026-03-04")) == 1
    assert store.revisions("AAPL") == []


# --- the vendor's own actual, or the analyst signal can never be built ------
#
# SUE_af pairs a street estimate with a reported figure. A street estimate is
# non-GAAP and XBRL is GAAP, and they are not small differences: Finnhub's MSFT
# 2026Q2 actual is 4.14 against 5.16 in the 10-Q, NVDA 1.05 against 1.08.
# Pairing across the two manufactures a surprise larger than the real one.
#
# So the analyst variant needs the VENDOR's actual, not the filing's -- and
# that number is only visible in the calendar for a short window after the
# print. Not recording it today means SUE_af is unbuildable in two years no
# matter how much history has accrued, which is the same clock argument that
# put the recorder first.


def test_the_vendors_own_actual_is_recorded_beside_its_estimate(store,
                                                                monkeypatch):
    monkeypatch.setattr(daily_job, "_fetch_calendar",
                        lambda start, end: [
                            {"ticker": "MSFT", "fiscal_period": "2026Q2",
                             "eps_estimate": 4.02, "eps_actual": 4.14,
                             "analyst_count": 30}])

    daily_job.record_consensus_snapshots(as_of="2026-08-01", horizon_days=10)

    snap = store.consensus_as_of("MSFT", "2026Q2", as_of="2026-08-01")
    assert snap["eps_estimate"] == 4.02
    assert snap["eps_actual"] == 4.14, (
        "the vendor's actual was dropped; SUE_af cannot be built from a "
        "GAAP filing figure paired with a non-GAAP estimate")


def test_a_name_that_has_not_reported_has_no_actual(store, monkeypatch):
    """Absent, not zero. A quarter not yet reported has no actual, and a zero
    there would read as a company that earned nothing."""
    monkeypatch.setattr(daily_job, "_fetch_calendar",
                        lambda start, end: [
                            {"ticker": "AAPL", "fiscal_period": "2026Q4",
                             "eps_estimate": 1.55, "eps_actual": None}])

    daily_job.record_consensus_snapshots(as_of="2026-08-01", horizon_days=10)

    snap = store.consensus_as_of("AAPL", "2026Q4", as_of="2026-08-01")
    assert snap["eps_actual"] is None


def test_the_window_looks_back_so_actuals_are_caught(store, monkeypatch):
    """An actual only appears after the print. A forward-only window would
    snapshot the estimate every day and never the outcome."""
    seen = {}

    def calendar(start, end):
        seen["start"], seen["end"] = start, end
        return []

    monkeypatch.setattr(daily_job, "_fetch_calendar", calendar)
    daily_job.record_consensus_snapshots(as_of="2026-08-10", horizon_days=10)

    assert seen["start"] < "2026-08-10", (
        "the calendar window starts today, so a company that reported "
        "yesterday never has its actual recorded")
    assert seen["end"] > "2026-08-10"


# --- the fetch seam itself, which had been returning nothing ---------------
#
# `_fetch_calendar` read payload["data"]["earnings"]. The key is "events".
# `.get("earnings") or []` turned a wrong key into an empty calendar, so the
# consensus job ran green every day and recorded not one row. That is the
# swallowed-failure class this project has now closed in nine other places,
# and it is worse here than elsewhere: the estimate series is the one thing
# that cannot be back-filled, so every green-but-empty day was permanent.


def _raw(symbol, q, y, est=1.0, act=None):
    row = {"symbol": symbol, "date": "2026-08-20", "epsEstimate": est,
           "quarter": q, "year": y, "numberOfAnalysts": 7, "hour": "amc"}
    if act is not None:
        row["epsActual"] = act
    return row


def _patch_client(monkeypatch, rows):
    async def fake_get(self, endpoint, params=None):
        assert endpoint == "/calendar/earnings", endpoint
        return {"earningsCalendar": rows}

    from tools.news_agregator import finnhub_utils
    monkeypatch.setattr(finnhub_utils.FinnhubClient, "get", fake_get)


def test_the_calendar_fetch_returns_rows_at_all(monkeypatch):
    _patch_client(monkeypatch, [_raw("MSFT", 2, 2026, est=4.02, act=4.14)])

    rows = daily_job._fetch_calendar("2026-08-15", "2026-08-25")

    assert len(rows) == 1, "the fetch seam returned an empty calendar"
    assert rows[0]["ticker"] == "MSFT"
    assert rows[0]["fiscal_period"] == "2026Q2"
    assert rows[0]["eps_estimate"] == 4.02
    assert rows[0]["eps_actual"] == 4.14
    assert rows[0]["analyst_count"] == 7


def test_the_recorder_is_not_subject_to_the_display_cap(monkeypatch):
    """The MCP tool caps events at 15 to protect an LLM's context. A recorder
    has no context to protect and every dropped name is a permanent hole."""
    _patch_client(monkeypatch, [_raw(f"T{i}", 3, 2026) for i in range(40)])

    rows = daily_job._fetch_calendar("2026-08-15", "2026-08-25")

    assert len(rows) == 40, f"the recorder saw only {len(rows)} of 40 reporters"


def test_a_row_without_a_fiscal_period_is_dropped_not_keyed_on_none(monkeypatch):
    """A snapshot keyed on None collides with every other such row and
    silently overwrites a real quarter's estimate."""
    _patch_client(monkeypatch, [
        {"symbol": "WAT", "date": "2026-08-20", "epsEstimate": 1.0},
        _raw("GOOD", 3, 2026),
    ])

    rows = daily_job._fetch_calendar("2026-08-15", "2026-08-25")

    assert [r["ticker"] for r in rows] == ["GOOD"]
