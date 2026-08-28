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


def test_the_job_loads_the_repo_env_file():
    """Cron has no shell profile.

    Every other entry point in this repo picks up credentials because a human
    ran it from a shell that had already sourced them. A nightly job has no
    such shell: it would raise "SEC_EMAIL is not set" at 21:00 every day and
    work perfectly in every hand test.
    """
    from pathlib import Path

    assert daily_job._DOTENV_PATH.name == ".env"
    # Repo root, the same place the container mounts it -- not the package
    # directory, which is where a relative path would land.
    assert (daily_job._DOTENV_PATH.parent / "tools").is_dir()
    assert daily_job._DOTENV_PATH.parent == Path(
        daily_job.__file__).resolve().parent.parent


def test_a_single_ticker_fetch_returns_its_bars(monkeypatch):
    """One name is not a special case, but the frame's shape made it one.

    yfinance returns a two-level column index whether you ask for one ticker or
    fifty, so branching on the count read the multi-level frame as flat, found
    no "Close", and returned {}. The daily pass hides this -- it always asks for
    thousands -- so it would surface only when retrying a single failed name,
    or backfilling one, which is exactly when a quiet empty is most expensive.
    """
    import pandas as pd

    idx = pd.to_datetime(["2024-06-10", "2024-06-11"])
    cols = {}
    for sym in ("NVDA", daily_job.FETCH_CANARY):
        cols[(sym, "Open")] = [1.0, 2.0]
        cols[(sym, "High")] = [1.0, 2.0]
        cols[(sym, "Low")] = [1.0, 2.0]
        cols[(sym, "Close")] = [1.0, 2.0]
        cols[(sym, "Volume")] = [10, 20]
    frame = pd.DataFrame(cols, index=idx)
    frame.columns = pd.MultiIndex.from_tuples(frame.columns)

    import yfinance
    monkeypatch.setattr(yfinance, "download", lambda *a, **k: frame)

    got = daily_job._fetch_bars(["NVDA"], start="2024-06-10", end="2024-06-12")

    assert list(got) == ["NVDA"], f"single-ticker fetch returned {got!r}"
    assert [b["close"] for b in got["NVDA"]] == [1.0, 2.0]


# --- what the vendor calls unadjusted, and what it actually is -------------
#
# `auto_adjust=False` does not return as-traded prices. It returns SPLIT-
# adjusted, dividend-unadjusted ones: NVDA's close for 2024-06-07 comes back as
# 120.89 when the stock really printed 1208.88 that day, and the volume comes
# back 412,386,000 against a true 41,238,600.
#
# That breaks the store's premise in a way nothing would have surfaced. Forward
# recording is safe by accident -- a one-day window has no future split in it,
# so today's bar is today's real price -- but bootstrap pulls two years in the
# vendor's current adjustment space. The store would hold a mixture with an
# invisible seam at the bootstrap boundary, and a read-time adjustment would
# double-apply every split on one side of it.
#
# So bars are converted to as-traded on the way in, using the splits that come
# down the same stream. The conversion is exactly invertible, which means a
# round trip returns the vendor's own number even if the action list is short.


def _frame(rows, ticker="NVDA"):
    """A response shaped like the vendor's, canary included.

    Every batch goes out with a known-liquid name attached so an unanswered
    batch can be told from names that genuinely have no sessions. A fixture
    that omits it is indistinguishable from a batch that never landed.
    """
    import pandas as pd
    from research.daily_job import FETCH_CANARY
    idx = pd.to_datetime([r["d"] for r in rows])
    data = {}
    for sym in (ticker, FETCH_CANARY):
        for field, key in (("Open", "c"), ("High", "c"), ("Low", "c"),
                           ("Close", "c"), ("Volume", "v"),
                           ("Stock Splits", "s"), ("Dividends", "div")):
            src = rows if sym == ticker else [{"c": 1.0, "v": 1.0} for _ in rows]
            data[(sym, field)] = [r.get(key, 0.0) for r in src]
    f = pd.DataFrame(data, index=idx)
    f.columns = pd.MultiIndex.from_tuples(f.columns)
    return f


NVDA_WINDOW = [
    {"d": "2024-06-07", "c": 120.888, "v": 412_386_000, "s": 0.0},
    {"d": "2024-06-10", "c": 121.790, "v": 313_434_100, "s": 10.0},
    {"d": "2024-06-11", "c": 120.910, "v": 222_551_200, "s": 0.0, "div": 0.01},
]


def test_bootstrap_stores_the_price_the_stock_actually_printed(store, monkeypatch):
    import yfinance
    monkeypatch.setattr(yfinance, "download", lambda *a, **k: _frame(NVDA_WINDOW))

    daily_job.bootstrap_history(["NVDA"], as_of="2026-08-27")

    bars = {b["trade_date"]: b for b in store.bars_as_of("NVDA", "2026-08-27")}
    assert round(bars["2024-06-07"]["close"], 2) == 1208.88, (
        "the vendor's split-adjusted close was stored as if it were as-traded")
    assert round(bars["2024-06-07"]["volume"]) == 41_238_600
    # The ex-date session already trades at the new price, so it is untouched.
    assert round(bars["2024-06-10"]["close"], 2) == 121.79
    assert round(bars["2024-06-10"]["volume"]) == 313_434_100


def test_the_split_that_did_the_adjusting_is_recorded(store, monkeypatch):
    """It arrives in the same download. Not recording it leaves the store
    unable to rebuild the very adjustment it just undid."""
    import yfinance
    monkeypatch.setattr(yfinance, "download", lambda *a, **k: _frame(NVDA_WINDOW))

    daily_job.bootstrap_history(["NVDA"], as_of="2026-08-27")

    actions = store.corporate_actions_as_of("NVDA", as_of="2026-08-27")
    splits = [a for a in actions if a["action_type"] == "split"]
    assert len(splits) == 1
    assert splits[0]["ex_date"] == "2024-06-10"
    assert splits[0]["value"] == 10.0
    divs = [a for a in actions if a["action_type"] == "dividend"]
    assert [(d["ex_date"], d["value"]) for d in divs] == [("2024-06-11", 0.01)]


def test_the_round_trip_returns_the_vendors_own_number(store, monkeypatch):
    """The safety property. Un-adjusting on the way in and re-adjusting on the
    way out are exact inverses, so a short action list cannot produce a price
    that never existed -- it can only return what the vendor said."""
    import yfinance
    monkeypatch.setattr(yfinance, "download", lambda *a, **k: _frame(NVDA_WINDOW))

    daily_job.bootstrap_history(["NVDA"], as_of="2026-08-27")

    back = {b["trade_date"]: b["close"]
            for b in store.adjusted_bars("NVDA", as_of="2026-08-27")}
    for row in NVDA_WINDOW:
        assert round(back[row["d"]], 6) == round(row["c"], 6), row["d"]


def test_todays_bar_is_never_rescaled(store, monkeypatch):
    """A one-day window has no split after it by construction, so the daily
    path must come through untouched -- including on an ex-date, where the
    printed price is already the new one."""
    import yfinance
    monkeypatch.setattr(yfinance, "download",
                        lambda *a, **k: _frame([NVDA_WINDOW[1]]))

    daily_job.record_daily_bars(["NVDA"], as_of="2024-06-10")

    bar = store.bars_as_of("NVDA", "2024-06-10")[0]
    assert round(bar["close"], 2) == 121.79
    assert round(bar["volume"]) == 313_434_100


# --- the universe does not fit in one request -------------------------------
#
# yfinance starts a thread per ticker. Asked for all 10,388 SEC names at once
# it raises RuntimeError: can't start new thread, which is what the nightly job
# would have done on its first run against the real universe. Every hand test
# used a handful of tickers and passed.


def test_a_large_fetch_is_split_into_batches(monkeypatch):
    calls = []

    def fake_download(*args, **kwargs):
        import pandas as pd
        asked = kwargs["tickers"].split()
        calls.append(len(asked))
        idx = pd.to_datetime(["2026-08-26"])
        data = {}
        for t in asked:
            for field in ("Open", "High", "Low", "Close", "Volume",
                          "Stock Splits", "Dividends"):
                data[(t, field)] = [1.0]
        f = pd.DataFrame(data, index=idx)
        f.columns = pd.MultiIndex.from_tuples(f.columns)
        return f

    import yfinance
    monkeypatch.setattr(yfinance, "download", fake_download)

    tickers = [f"T{i}" for i in range(500)]
    got = daily_job._fetch_bars(tickers, start="2026-08-26", end="2026-08-27")

    assert len(calls) > 1, "500 tickers went out as a single request"
    # +1 for the canary each batch carries.
    assert max(calls) <= daily_job.FETCH_BATCH_SIZE + 1
    assert len(got) == 500, f"batching lost tickers: {len(got)} of 500"


def test_a_transient_batch_failure_is_retried(monkeypatch):
    """The first request against a cold session fails and the second works.

    Measured, not assumed: asked for the top 200 names twice in a row, the
    first call returned 0 and the second returned 200. Against the full
    universe that cost every mega-cap -- AAPL, MSFT, NVDA and 4,553 others --
    because the failing batches are simply the ones that go out first.
    """
    import pandas as pd
    state = {"n": 0}

    def fake_download(*args, **kwargs):
        state["n"] += 1
        if state["n"] == 1:
            raise RuntimeError("Invalid Crumb")
        asked = kwargs["tickers"].split()
        idx = pd.to_datetime(["2026-08-26"])
        data = {}
        for t in asked:
            for field in ("Open", "High", "Low", "Close", "Volume",
                          "Stock Splits", "Dividends"):
                data[(t, field)] = [1.0]
        f = pd.DataFrame(data, index=idx)
        f.columns = pd.MultiIndex.from_tuples(f.columns)
        return f

    import yfinance
    monkeypatch.setattr(yfinance, "download", fake_download)
    monkeypatch.setattr(daily_job, "FETCH_RETRY_BACKOFF", 0.0)

    got = daily_job._fetch_bars([f"T{i}" for i in range(500)],
                                start="2026-08-26", end="2026-08-27")

    assert len(got) == 500, f"a retried batch was lost: {len(got)} of 500"


def test_a_batch_that_keeps_failing_is_reported_not_swallowed(monkeypatch):
    """It cannot take the night down, and it cannot go unmentioned either.

    One lost batch costs each of its names a single session out of the sixty
    the screen reads, so the day is still worth keeping -- but a gap nobody
    recorded is the one that later reads as a market where those names did not
    trade. So the batch is skipped, and the run says which.
    """
    import pandas as pd

    def fake_download(*args, **kwargs):
        asked = kwargs["tickers"].split()
        if "T0" in asked:
            raise ConnectionError("Yahoo returned 502")
        data = {}
        for t in asked:
            for field in ("Open", "High", "Low", "Close", "Volume",
                          "Stock Splits", "Dividends"):
                data[(t, field)] = [1.0]
        f = pd.DataFrame(data, index=pd.to_datetime(["2026-08-26"]))
        f.columns = pd.MultiIndex.from_tuples(f.columns)
        return f

    import yfinance
    monkeypatch.setattr(yfinance, "download", fake_download)
    monkeypatch.setattr(daily_job, "FETCH_RETRY_BACKOFF", 0.0)
    monkeypatch.setattr(daily_job, "FETCH_BATCH_PAUSE", 0.0)

    failures = []
    got = daily_job._fetch_bars([f"T{i}" for i in range(500)],
                                start="2026-08-26", end="2026-08-27",
                                failures=failures)

    assert 0 < len(got) < 500
    assert len(failures) == 1
    assert "502" in failures[0]


def test_a_lost_batch_reaches_the_run_log(store, monkeypatch):
    """Where an operator will actually see it."""
    def fetch(tickers, **kw):
        kw.get("failures", []).append("batch 2 of 52: ConnectionError: 502")
        return {"AAPL": _bars("2026-03-03")}

    monkeypatch.setattr(daily_job, "_fetch_bars", fetch)
    daily_job.record_daily_bars(["AAPL", "MSFT"], as_of="2026-03-03")

    run = daily_job.last_run("daily_bars")
    assert run["status"] == "partial"
    assert "502" in str(run["error"]), (
        f"the lost batch is not in the run log: {run['error']!r}")


def test_every_batch_failing_is_raised_not_returned_empty(monkeypatch):
    """Total failure must not read as a market where nobody traded."""
    def fake_download(*args, **kwargs):
        raise ConnectionError("Yahoo returned 502")

    import yfinance
    monkeypatch.setattr(yfinance, "download", fake_download)
    monkeypatch.setattr(daily_job, "FETCH_RETRY_BACKOFF", 0.0)

    with pytest.raises(RuntimeError, match="502"):
        daily_job._fetch_bars([f"T{i}" for i in range(500)],
                              start="2026-08-26", end="2026-08-27")


# --- telling a dead batch from dead tickers ---------------------------------
#
# The failing batch does not raise. yfinance catches per-ticker errors itself
# and hands back an empty frame -- 0 rows by 1200 columns for 200 names -- so a
# rate-limited batch looks exactly like 200 companies that did not trade.
# Retrying on exceptions alone missed all of it: the full universe still lost
# every mega-cap and reported 59% coverage with no error anywhere.
#
# An empty frame cannot be read either way on its own, because a batch of 200
# delisted shells is genuinely empty and the SEC list has thousands of those.
# So each batch carries a known-liquid name. If that name comes back, the batch
# worked and the absences are real; if it does not, the vendor did not answer.


def _empty_frame():
    import pandas as pd
    return pd.DataFrame()


def test_a_batch_that_answers_nothing_is_retried_not_believed(monkeypatch):
    import pandas as pd
    state = {"n": 0}

    def fake_download(*args, **kwargs):
        state["n"] += 1
        if state["n"] == 1:
            return _empty_frame()
        asked = kwargs["tickers"].split()
        data = {}
        for t in asked:
            for field in ("Open", "High", "Low", "Close", "Volume",
                          "Stock Splits", "Dividends"):
                data[(t, field)] = [1.0]
        f = pd.DataFrame(data, index=pd.to_datetime(["2026-08-26"]))
        f.columns = pd.MultiIndex.from_tuples(f.columns)
        return f

    import yfinance
    monkeypatch.setattr(yfinance, "download", fake_download)
    monkeypatch.setattr(daily_job, "FETCH_RETRY_BACKOFF", 0.0)

    got = daily_job._fetch_bars(["AAPL", "MSFT"], start="2026-08-26",
                                end="2026-08-27")

    assert set(got) == {"AAPL", "MSFT"}, (
        "an unanswered batch was recorded as two companies that did not trade")


def test_genuinely_dead_tickers_are_absent_not_an_error(monkeypatch):
    """The other side of it. If the canary answers, the batch reached the
    vendor and the missing names really have no sessions -- which must record
    as absence, not raise and take the night down with it."""
    import pandas as pd

    def fake_download(*args, **kwargs):
        asked = kwargs["tickers"].split()
        alive = [t for t in asked if t == daily_job.FETCH_CANARY]
        data = {}
        for t in alive:
            for field in ("Open", "High", "Low", "Close", "Volume",
                          "Stock Splits", "Dividends"):
                data[(t, field)] = [1.0]
        f = pd.DataFrame(data, index=pd.to_datetime(["2026-08-26"]))
        f.columns = pd.MultiIndex.from_tuples(f.columns)
        return f

    import yfinance
    monkeypatch.setattr(yfinance, "download", fake_download)
    monkeypatch.setattr(daily_job, "FETCH_RETRY_BACKOFF", 0.0)

    got = daily_job._fetch_bars(["DEAD1", "DEAD2"], start="2026-08-26",
                                end="2026-08-27")

    assert got == {}, "a dead ticker produced a bar"


def test_the_canary_is_not_smuggled_into_the_results(monkeypatch):
    """It is scaffolding. Recording it as though the caller asked would put a
    ticker in the universe that nothing screened."""
    import pandas as pd

    def fake_download(*args, **kwargs):
        asked = kwargs["tickers"].split()
        data = {}
        for t in asked:
            for field in ("Open", "High", "Low", "Close", "Volume",
                          "Stock Splits", "Dividends"):
                data[(t, field)] = [1.0]
        f = pd.DataFrame(data, index=pd.to_datetime(["2026-08-26"]))
        f.columns = pd.MultiIndex.from_tuples(f.columns)
        return f

    import yfinance
    monkeypatch.setattr(yfinance, "download", fake_download)

    got = daily_job._fetch_bars(["AAPL"], start="2026-08-26", end="2026-08-27")
    assert set(got) == {"AAPL"}

    # ...but a caller who genuinely wants it still gets it.
    got = daily_job._fetch_bars(["AAPL", daily_job.FETCH_CANARY],
                                start="2026-08-26", end="2026-08-27")
    assert set(got) == {"AAPL", daily_job.FETCH_CANARY}


# --- the universe has to be able to grow ------------------------------------
#
# run_all fetched bars for the eligible names and screened every registrant.
# A newly listed company is not eligible, because it has no history; it then
# never gets a bar, because only eligible names are fetched; so it never
# acquires history. Nothing errors and nothing is logged -- the universe simply
# never admits another name for as long as the job runs.
#
# Asking for all 10,388 registrants nightly is the other extreme, and Yahoo
# answers it with YFRateLimitError. So the nightly ask is the eligible set plus
# a rotating slice of everything else, which bounds the request and still gets
# every name its sixty sessions inside a cycle.


def test_a_new_listing_can_reach_eligibility(store, monkeypatch):
    monkeypatch.setattr(daily_job, "_fetch_sec_tickers", lambda: [
        {"ticker": "OLD", "cik": "1", "name": "Old Co"},
        {"ticker": "IPO", "cik": "2", "name": "Newly Listed Inc"},
    ])
    sessions = [f"2026-01-{d:02d}" for d in range(1, 32)] + \
               [f"2026-02-{d:02d}" for d in range(1, 29)]
    store.record_bars("OLD", _bars(*sessions), recorded_at="2026-03-02T21:00:00Z")
    store.record_universe("2026-03-02", [
        {"ticker": "OLD", "cik": "1", "eligible": True},
        {"ticker": "IPO", "cik": "2", "eligible": False,
         "exclusion_reason": "insufficient history"},
    ],
                          recorded_at="2026-03-02T21:00:00Z")

    asked = {}
    monkeypatch.setattr(daily_job, "_fetch_bars",
                        lambda tickers, **kw: asked.update(t=list(tickers)) or {})
    monkeypatch.setattr(daily_job, "record_consensus_snapshots",
                        lambda **kw: {"status": "ok"})

    daily_job.run_all(as_of="2026-03-03")

    assert "IPO" in asked["t"], (
        "an ineligible name is never fetched, so it can never become eligible")


def test_the_nightly_ask_is_bounded(store, monkeypatch):
    """10,388 names in one night is what the rate limiter answers."""
    registrants = [{"ticker": f"T{i}", "cik": str(i), "name": f"Co {i}"}
                   for i in range(5000)]
    monkeypatch.setattr(daily_job, "_fetch_sec_tickers", lambda: registrants)
    monkeypatch.setattr(daily_job, "record_consensus_snapshots",
                        lambda **kw: {"status": "ok"})

    asked = {}
    monkeypatch.setattr(daily_job, "_fetch_bars",
                        lambda tickers, **kw: asked.update(t=list(tickers)) or {})

    daily_job.run_all(as_of="2026-03-03")

    # Minus the liveness canary, which every fetch carries and none records.
    universe = [t for t in asked["t"] if t != daily_job.FETCH_CANARY]
    assert len(universe) <= daily_job.MAX_NIGHTLY_TICKERS


def test_the_rotation_covers_everyone_eventually(store, monkeypatch):
    """A slice that is the same every night starves the rest forever."""
    registrants = [{"ticker": f"T{i}", "cik": str(i), "name": f"Co {i}"}
                   for i in range(600)]
    monkeypatch.setattr(daily_job, "_fetch_sec_tickers", lambda: registrants)
    monkeypatch.setattr(daily_job, "record_consensus_snapshots",
                        lambda **kw: {"status": "ok"})
    monkeypatch.setattr(daily_job, "MAX_NIGHTLY_TICKERS", 100)

    seen = set()
    def fetch(tickers, **kw):
        seen.update(tickers)
        return {}
    monkeypatch.setattr(daily_job, "_fetch_bars", fetch)

    for day in range(1, 15):
        daily_job.run_all(as_of=f"2026-03-{day:02d}")

    seen.discard(daily_job.FETCH_CANARY)
    assert len(seen) == 600, f"only {len(seen)} of 600 names were ever asked for"


def test_eligible_names_are_never_starved_by_the_rotation(store, monkeypatch):
    """They are the ones a signal actually reads. A rotation that pushes them
    out leaves holes in exactly the series that matter."""
    registrants = [{"ticker": f"T{i}", "cik": str(i), "name": f"Co {i}"}
                   for i in range(600)]
    monkeypatch.setattr(daily_job, "_fetch_sec_tickers", lambda: registrants)
    monkeypatch.setattr(daily_job, "record_consensus_snapshots",
                        lambda **kw: {"status": "ok"})
    monkeypatch.setattr(daily_job, "MAX_NIGHTLY_TICKERS", 100)
    store.record_universe("2026-03-02", [
        {"ticker": f"T{i}", "cik": str(i), "eligible": True} for i in range(60)],
                          recorded_at="2026-03-02T21:00:00Z")

    asked = {}
    monkeypatch.setattr(daily_job, "_fetch_bars",
                        lambda tickers, **kw: asked.update(t=set(tickers)) or {})

    daily_job.run_all(as_of="2026-03-03")

    assert {f"T{i}" for i in range(60)} <= asked["t"]


# --- the entry point cron actually calls ------------------------------------

def test_a_failed_night_exits_non_zero(store, monkeypatch):
    """The only way a scheduler ever finds out. A job that prints an error and
    exits 0 is a job whose failures nobody sees until someone reads a chart
    built on a hole."""
    monkeypatch.setattr(daily_job, "run_all",
                        lambda **kw: {"as_of": "2026-03-03",
                                      "error": "universe unavailable: 503"})
    assert daily_job.main(["--as-of", "2026-03-03"]) == 1


def test_a_partial_night_exits_zero(store, monkeypatch):
    """Partial is normal on a universe this size and must not page anyone."""
    monkeypatch.setattr(daily_job, "run_all",
                        lambda **kw: {"as_of": "2026-03-03",
                                      "daily_bars": {"status": "partial"},
                                      "universe": {"status": "ok"},
                                      "consensus": {"status": "ok"}})
    assert daily_job.main(["--as-of", "2026-03-03"]) == 0


def test_a_failed_stage_exits_non_zero(store, monkeypatch):
    monkeypatch.setattr(daily_job, "run_all",
                        lambda **kw: {"as_of": "2026-03-03",
                                      "daily_bars": {"status": "ok"},
                                      "universe": {"status": "failed"},
                                      "consensus": {"status": "ok"}})
    assert daily_job.main(["--as-of", "2026-03-03"]) == 1


def test_the_flags_reach_run_all(store, monkeypatch):
    seen = {}
    monkeypatch.setattr(daily_job, "run_all",
                        lambda **kw: seen.update(kw) or {"as_of": "x"})
    daily_job.main(["--as-of", "2026-03-03", "--bootstrap"])
    assert seen == {"as_of": "2026-03-03", "bootstrap": True}


# --- the announcement table, which nothing was filling -----------------------
#
# record_announcement and announcements_as_of exist and are tested, and no job
# ever wrote a row: zero in every store. The calendar fetch already parses the
# one field that makes the table worth having -- `hour`, which is bmo or amc --
# and then dropped it.
#
# It decides which session is the reaction. AMAT reported after the close on 13
# August, so its gap is the 14th; dating it to the announcement gives -2.48%
# instead of -6.57%. An empty table that looks populated-in-principle is how
# someone later reads "no announcements" as a quiet market.

def test_the_consensus_run_records_the_announcement(store, monkeypatch):
    monkeypatch.setattr(daily_job, "_fetch_calendar",
                        lambda start, end: [
                            {"ticker": "AMAT", "fiscal_period": "2026Q3",
                             "eps_estimate": 2.10, "eps_actual": 2.48,
                             "timing": "amc", "date": "2026-08-13"}])

    daily_job.record_consensus_snapshots(as_of="2026-08-14")

    got = store.announcements_as_of("AMAT", as_of="2026-08-15")
    assert [(a["fiscal_period"], a["announced_date"], a["timing"])
            for a in got] == [("2026Q3", "2026-08-13", "amc")]


def test_an_announcement_is_only_recorded_once_it_has_happened(store,
                                                               monkeypatch):
    """A company on next week's calendar has not announced. Recording it would
    put a reaction date in the record before there was a reaction."""
    monkeypatch.setattr(daily_job, "_fetch_calendar",
                        lambda start, end: [
                            {"ticker": "AAPL", "fiscal_period": "2026Q4",
                             "eps_estimate": 1.55, "eps_actual": None,
                             "timing": "amc", "date": "2026-09-05"}])

    daily_job.record_consensus_snapshots(as_of="2026-08-27")
    assert store.announcements_as_of("AAPL", as_of="2026-09-30") == []


def test_a_calendar_row_with_no_timing_records_unknown_not_a_guess(store,
                                                                   monkeypatch):
    monkeypatch.setattr(daily_job, "_fetch_calendar",
                        lambda start, end: [
                            {"ticker": "XYZ", "fiscal_period": "2026Q1",
                             "eps_estimate": 1.0, "eps_actual": 1.1,
                             "timing": "unknown", "date": "2026-05-01"}])

    daily_job.record_consensus_snapshots(as_of="2026-05-02")
    got = store.announcements_as_of("XYZ", as_of="2026-05-03")
    assert got[0]["timing"] == "unknown"


def test_an_announcement_carries_the_stamp_of_the_run_that_saw_it(store,
                                                                  monkeypatch):
    monkeypatch.setattr(daily_job, "_fetch_calendar",
                        lambda start, end: [
                            {"ticker": "AMAT", "fiscal_period": "2026Q3",
                             "eps_estimate": 2.10, "eps_actual": 2.48,
                             "timing": "amc", "date": "2026-08-13"}])

    daily_job.record_consensus_snapshots(as_of="2026-08-14")
    assert store.announcements_as_of("AMAT", as_of="2026-08-13") == []
    assert store.announcements_as_of("AMAT", as_of="2026-08-14")


def test_a_row_with_no_announcement_date_is_not_recorded(store, monkeypatch):
    monkeypatch.setattr(daily_job, "_fetch_calendar",
                        lambda start, end: [
                            {"ticker": "XYZ", "fiscal_period": "2026Q1",
                             "eps_estimate": 1.0, "eps_actual": 1.1,
                             "timing": "amc", "date": None}])

    daily_job.record_consensus_snapshots(as_of="2026-05-02")
    assert store.announcements_as_of("XYZ", as_of="2026-05-03") == []


# --- the first run on a fresh volume ----------------------------------------
#
# Nothing in production ever created the tables. Every test here builds them in
# a fixture, which is exactly why the suite was green while the first
# `docker compose run --rm research-daily` on a new homelab died on
# "no such table: run_log" before it fetched anything.


def test_a_fresh_store_is_created_by_the_job_itself(tmp_path, monkeypatch):
    monkeypatch.setenv("NEMO_PIT_DB", str(tmp_path / "fresh.db"))
    monkeypatch.setattr(daily_job, "_fetch_sec_tickers",
                        lambda: [{"ticker": "AAPL", "cik": "320193",
                                  "name": "Apple Inc."}])
    monkeypatch.setattr(daily_job, "_fetch_bars", lambda tickers, **kw: {
        "AAPL": _bars("2026-03-03"),
        daily_job.FETCH_CANARY: _bars("2026-03-03")})
    monkeypatch.setattr(daily_job, "_fetch_calendar", lambda start, end: [
        {"ticker": "AAPL", "fiscal_period": "2026Q1", "eps_estimate": 1.0}])

    assert daily_job.main(["--as-of", "2026-03-03"]) == 0
    assert len(pit_store.bars_as_of("AAPL", "2026-03-04")) == 1


# --- a session that has not happened is not a holiday ------------------------
#
# `_today()` is UTC and cron is host-local, so on an America/New_York host the
# 22:30 run is dated tomorrow. The canary then returns a window ending
# yesterday, no bar carries the run's date, and "the exchange was shut" is the
# wrong reading of it: the job records nothing, exits 0, and `missing_days`
# counts the night as covered. A year of that reads as a year of holidays.
#
# The two cases are identical in the data -- a holiday leaves the canary's last
# bar on the previous session too -- so only the clock separates them, and both
# tests here run the job the way cron does, letting it date itself.


def _clock(monkeypatch, moment):
    from datetime import datetime, timezone
    monkeypatch.setattr(daily_job, "_utc_now",
                        lambda: datetime.fromisoformat(moment).replace(
                            tzinfo=timezone.utc))


def test_a_session_that_has_not_happened_is_not_a_holiday(store, monkeypatch):
    # 22:30 on Monday in New York, which is 02:30 Tuesday in UTC.
    _clock(monkeypatch, "2026-03-03T02:30:00")
    monkeypatch.setattr(daily_job, "_fetch_bars", lambda tickers, **kw: {
        daily_job.FETCH_CANARY: _bars("2026-03-02"),
        "AAPL": _bars("2026-03-02")})

    result = daily_job.record_daily_bars(["AAPL"])

    assert result["status"] == "failed", (
        "a run dated ahead of the market was logged as a holiday")
    run = daily_job.last_run("daily_bars")
    assert run["status"] == "failed"
    assert "2026-03-02" in str(run["error"]), (
        f"the error does not say what the last session was: {run['error']!r}")
    assert store.missing_days("daily_bars", "2026-03-03", "2026-03-03") == \
        ["2026-03-03"], "a night that recorded nothing counted as coverage"


def test_a_real_holiday_is_still_a_complete_run_over_nothing(store,
                                                             monkeypatch):
    """The other side of it. The exchange shuts about ten weekdays a year, and
    listing those as permanent holes is how the one real gap stops being
    noticed."""
    # Thanksgiving evening, after the session that did not happen would have
    # closed. Nothing is pending: the day is over and the exchange was shut.
    _clock(monkeypatch, "2026-11-26T22:30:00")
    monkeypatch.setattr(daily_job, "_fetch_bars", lambda tickers, **kw: {
        daily_job.FETCH_CANARY: _bars("2026-11-24", "2026-11-25")})

    result = daily_job.record_daily_bars(["AAPL"])

    assert result["status"] == "closed"
    assert store.missing_days("daily_bars", "2026-11-26", "2026-11-26") == []


# --- the write loop, and what one bad ticker costs ---------------------------
#
# The try/except covered the fetch and stopped there. A single ticker whose
# write raised -- a locked database at 22:40, a vendor NaN reaching a validator
# -- took the process down mid-list, left the run log open, and cost the
# universe screen and the consensus snapshot that had not run yet. Consensus is
# the one series that cannot be refetched later.


def test_one_ticker_that_cannot_be_written_does_not_cost_the_night(
        store, monkeypatch):
    import sqlite3

    monkeypatch.setattr(daily_job, "_fetch_bars", lambda tickers, **kw: {
        "AAPL": _bars("2026-03-03"), "MSFT": _bars("2026-03-03"),
        daily_job.FETCH_CANARY: _bars("2026-03-03")})

    real = daily_job._record_ticker

    def flaky(ticker, rows, stamp, keep=None):
        if ticker == "MSFT":
            raise sqlite3.OperationalError("database is locked")
        return real(ticker, rows, stamp, keep=keep)

    monkeypatch.setattr(daily_job, "_record_ticker", flaky)

    result = daily_job.record_daily_bars(["AAPL", "MSFT"], as_of="2026-03-03")

    assert result["status"] == "partial"
    assert len(store.bars_as_of("AAPL", "2026-03-04")) == 1, (
        "the ticker before the bad one was lost with it")
    run = daily_job.last_run("daily_bars")
    assert run["finished_at"], "the run log was left open"
    assert "MSFT" in str(run["error"]) and "locked" in str(run["error"]), (
        f"the ticker that failed is not in the run log: {run['error']!r}")


def test_a_stage_that_dies_does_not_cost_the_consensus_snapshot(
        store, monkeypatch):
    import sqlite3
    ran = []

    def boom(*args, **kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(daily_job, "_fetch_sec_tickers",
                        lambda: [{"ticker": "AAPL", "cik": "1", "name": "A"}])
    monkeypatch.setattr(daily_job, "record_daily_bars", boom)
    monkeypatch.setattr(
        daily_job, "refresh_universe",
        lambda **kw: ran.append("universe") or {"status": "ok"})
    monkeypatch.setattr(
        daily_job, "record_consensus_snapshots",
        lambda **kw: ran.append("consensus") or {"status": "ok"})

    result = daily_job.run_all(as_of="2026-03-03")

    assert ran == ["universe", "consensus"], (
        "one stage's exception cost the series that cannot be refetched")
    assert result["daily_bars"]["status"] == "failed"
    assert "locked" in result["daily_bars"]["error"]


def test_a_vendor_nan_is_not_a_corporate_action(store, monkeypatch):
    """NaN is truthy, so `NaN or 0.0` is NaN, which passes the `if split`
    guard and then fails `record_corporate_action`'s `value > 0` check. The
    as-traded conversion beside it already ignores the same value, so the two
    paths disagreed about one number."""
    import pandas as pd

    fields = {"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0,
              "Volume": 100.0, "Stock Splits": float("nan"),
              "Dividends": float("nan")}
    frame = pd.DataFrame({("AAPL", f): [v] for f, v in fields.items()},
                         index=pd.to_datetime(["2026-08-26"]))
    frame.columns = pd.MultiIndex.from_tuples(frame.columns)

    rows = daily_job._rows_from_frame(frame, ["AAPL"])
    assert rows["AAPL"][0]["split"] == 0.0
    assert rows["AAPL"][0]["dividend"] == 0.0

    daily_job._record_ticker("AAPL", rows["AAPL"], "2026-08-26T21:00:00Z")
    assert store.corporate_actions_as_of("AAPL", "2026-08-27") == []


def test_a_split_and_the_bars_it_explains_land_together(store, monkeypatch):
    """The order is documented as an invariant: a reader that finds bars
    without the split that shaped them cannot tell an as-traded series from an
    adjusted one. Written in two transactions, a crash between them leaves
    exactly the state the invariant forbids."""
    import sqlite3

    rows = [{"trade_date": "2026-06-10", "open": 10.0, "high": 10.0,
             "low": 10.0, "close": 10.0, "volume": 1000.0, "split": 10.0,
             "dividend": 0.0}]

    def boom(*args, **kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(pit_store, "record_bars", boom)
    with pytest.raises(sqlite3.OperationalError):
        daily_job._record_ticker("NVDA", rows, "2026-06-10T21:00:00Z")

    assert store.corporate_actions_as_of("NVDA", "2026-06-11") == [], (
        "the split outlived the bars it exists to explain")


# --- the canary belongs to the run, not to one batch -------------------------
#
# It is appended to every batch's request and kept only from the batch that
# owned it -- which, since it is appended to the end of the ask, is a batch of
# exactly one name. Fetched sixteen times, discarded fifteen, and the copy
# retained comes from the single most fragile request in the run. Lose it and
# the market-closed check is skipped and the regime series gains a hole.


def test_the_canary_survives_the_batch_that_owned_it_failing(monkeypatch):
    import pandas as pd

    def fake_download(*args, **kwargs):
        asked = kwargs["tickers"].split()
        if asked == [daily_job.FETCH_CANARY]:
            raise ConnectionError("Yahoo returned 502")
        data = {}
        for t in asked:
            for field in ("Open", "High", "Low", "Close", "Volume",
                          "Stock Splits", "Dividends"):
                data[(t, field)] = [1.0]
        frame = pd.DataFrame(data, index=pd.to_datetime(["2026-08-26"]))
        frame.columns = pd.MultiIndex.from_tuples(frame.columns)
        return frame

    import yfinance
    monkeypatch.setattr(yfinance, "download", fake_download)
    monkeypatch.setattr(daily_job, "FETCH_RETRY_BACKOFF", 0.0)
    monkeypatch.setattr(daily_job, "FETCH_BATCH_PAUSE", 0.0)

    asked = [f"T{i}" for i in range(daily_job.FETCH_BATCH_SIZE)]
    got = daily_job._fetch_bars([*asked, daily_job.FETCH_CANARY],
                                start="2026-08-26", end="2026-08-27")

    assert daily_job.FETCH_CANARY in got, (
        "the canary was answered by every other batch and kept from none")
    assert got[daily_job.FETCH_CANARY][0]["trade_date"] == "2026-08-26"


def test_a_holiday_is_still_recognised_when_the_canarys_batch_failed(
        store, monkeypatch):
    """The determination is about the exchange, not about which request
    happened to succeed."""
    def fetch(tickers, **kw):
        kw.get("failures", []).append("batch 2 of 2 failed: 502")
        return {daily_job.FETCH_CANARY: _bars("2026-03-02", "2026-03-04")}

    monkeypatch.setattr(daily_job, "_fetch_bars", fetch)
    result = daily_job.record_daily_bars(["AAPL"], as_of="2026-03-03")

    assert result["status"] == "closed"


# --- a rerun must not report rows it did not write ---------------------------

def test_a_rerun_reports_the_rows_it_actually_wrote(store, monkeypatch):
    """`rows_written` is the only evidence a night was captured, and consensus
    is the series where that matters most. A replay that changed nothing must
    not look like a capture."""
    monkeypatch.setattr(daily_job, "_fetch_calendar", lambda start, end: [
        {"ticker": "MSFT", "fiscal_period": "2026Q2", "eps_estimate": 4.02,
         "eps_actual": None, "analyst_count": 30}])

    first = daily_job.record_consensus_snapshots(as_of="2026-08-01")
    second = daily_job.record_consensus_snapshots(as_of="2026-08-01")

    assert first["written"] == 1
    assert second["written"] == 0, (
        f"the rerun reported {second['written']} rows it did not write")
    assert second["seen"] == 1, "the rows it saw are still worth reporting"
    assert daily_job.last_run("consensus")["rows_written"] == 0
