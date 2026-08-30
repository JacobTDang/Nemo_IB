"""What the happy-path tests never ask.

Each test here is a specific suspicion about a code path that works on the data
it was written against and has never seen anything else. The store runs
unattended for months; every one of these arrives eventually.
"""
from datetime import date, timedelta

import pytest

from research import daily_job, pit_store, scanner


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("NEMO_PIT_DB", str(tmp_path / "pit.db"))
    pit_store.init_schema()
    return pit_store


def _bar(d, close, volume=1_000_000):
    return {"trade_date": d, "open": close, "high": close * 1.01,
            "low": close * 0.99, "close": close, "volume": volume}


# --- the rotation, once the universe grows up -------------------------------

def test_a_full_universe_still_admits_new_names(store, monkeypatch):
    """The deadlock, reintroduced by its own fix.

    `room = MAX_NIGHTLY_TICKERS - len(eligible)` goes to zero the moment the
    eligible set reaches the cap, and the function then returns only the
    eligible names. A new listing is never fetched, so it never gets history,
    so it is never eligible -- which is exactly the starvation the rotation was
    added to prevent, arriving a few weeks later instead of on day one.

    The real eligible universe is a few thousand names, so this triggers in
    normal operation rather than in some extreme.
    """
    monkeypatch.setattr(daily_job, "MAX_NIGHTLY_TICKERS", 100)
    store.record_universe("2026-03-02", [
        {"ticker": f"OLD{i}", "cik": str(i), "eligible": True}
        for i in range(100)],
                          recorded_at="2026-03-02T21:00:00Z")

    asked = daily_job.nightly_tickers("2026-03-03",
                                      [f"OLD{i}" for i in range(100)] + ["IPO"])

    assert "IPO" in asked, (
        "with the universe at its cap, no new name is ever asked for again")


def test_a_full_universe_never_drops_an_eligible_name(store, monkeypatch):
    """Whatever room is made for newcomers cannot come out of the names a
    signal actually reads."""
    monkeypatch.setattr(daily_job, "MAX_NIGHTLY_TICKERS", 100)
    store.record_universe("2026-03-02", [
        {"ticker": f"OLD{i}", "cik": str(i), "eligible": True}
        for i in range(100)],
                          recorded_at="2026-03-02T21:00:00Z")

    asked = set(daily_job.nightly_tickers(
        "2026-03-03", [f"OLD{i}" for i in range(100)] + [f"NEW{j}" for j in range(50)]))

    assert {f"OLD{i}" for i in range(100)} <= asked


def test_newcomers_still_rotate_when_the_universe_is_full(store, monkeypatch):
    monkeypatch.setattr(daily_job, "MAX_NIGHTLY_TICKERS", 100)
    # Recorded before the window so every simulated day sees a full universe;
    # dated inside it, the first day would find no snapshot, screen nothing as
    # eligible and rotate freely -- passing for a reason the test is not about.
    store.record_universe("2026-02-01", [
        {"ticker": f"OLD{i}", "cik": str(i), "eligible": True}
        for i in range(100)],
                          recorded_at="2026-02-01T21:00:00Z")
    registrants = [f"OLD{i}" for i in range(100)] + [f"NEW{j}" for j in range(60)]

    seen = set()
    for day in range(1, 30):
        seen.update(daily_job.nightly_tickers(f"2026-03-{day:02d}", registrants))

    assert {f"NEW{j}" for j in range(60)} <= seen, (
        f"only {len({s for s in seen if s.startswith('NEW')})} of 60 newcomers "
        f"were ever asked for")


# --- adjustment arithmetic that the ordinary cases never reach --------------

def test_a_dividend_larger_than_the_price_is_not_swallowed(store):
    """Special dividends do exceed the price -- a fund distributing most of its
    NAV, a company paying out a sale. The factor goes negative or zero, and the
    code `continue`s past it, so the series is quietly left unadjusted with
    nothing saying so."""
    store.record_bars("X", [_bar("2024-05-01", 10.0), _bar("2024-05-02", 1.0)],
                      recorded_at="2024-05-02T21:00:00Z")
    store.record_corporate_action("X", "2024-05-02", "dividend", 12.0,
                                  recorded_at="2024-05-02T21:00:00Z")

    with pytest.raises(ValueError, match="dividend"):
        pit_store.adjusted_bars("X", as_of="2024-06-01", total_return=True)


def test_a_split_before_every_known_bar_changes_nothing(store):
    """The factor is built from actions *after* each bar, so a split older than
    the whole series must not scale it."""
    store.record_corporate_action("X", "2020-01-02", "split", 4.0,
                                  recorded_at="2020-01-02T21:00:00Z")
    store.record_bars("X", [_bar("2024-05-01", 10.0)],
                      recorded_at="2024-05-01T21:00:00Z")

    bars = pit_store.adjusted_bars("X", as_of="2024-06-01")
    assert bars[0]["close"] == 10.0
    assert bars[0]["adj_factor"] == 1.0


def test_a_zero_or_negative_split_ratio_is_refused(store):
    """A 0 ratio divides by zero and a negative one flips the sign of every
    price before it."""
    for bad in (0.0, -2.0):
        with pytest.raises(ValueError, match="split"):
            pit_store.record_corporate_action("X", "2024-05-02", "split", bad,
                                              recorded_at="2024-05-02T21:00:00Z")


def test_two_actions_on_one_day_both_apply(store):
    """A split and a dividend can share an ex-date, and keying a dict on the
    date rather than accumulating would keep only one."""
    store.record_bars("X", [_bar("2024-05-01", 100.0), _bar("2024-05-02", 49.5)],
                      recorded_at="2024-05-02T21:00:00Z")
    store.record_corporate_action("X", "2024-05-02", "split", 2.0,
                                  recorded_at="2024-05-02T21:00:00Z")
    store.record_corporate_action("X", "2024-05-02", "dividend", 1.0,
                                  recorded_at="2024-05-02T21:00:00Z")

    bars = pit_store.adjusted_bars("X", as_of="2024-06-01", total_return=True)
    # 100 -> halved for the split -> 1% off for the dividend against the 100 close
    assert round(bars[0]["close"], 4) == 49.5


def test_as_traded_conversion_survives_a_split_on_the_first_bar(store,
                                                               monkeypatch):
    """Nothing precedes it, so the factor must apply to nothing."""
    rows = [{"trade_date": "2024-06-10", "open": 10.0, "high": 10.0,
             "low": 10.0, "close": 10.0, "volume": 100, "split": 10.0,
             "dividend": 0.0}]
    out = daily_job._to_as_traded(rows)
    assert out[0]["close"] == 10.0
    assert out[0]["volume"] == 100


# --- one ticker must not take down a run ------------------------------------

def test_a_signal_that_raises_does_not_kill_the_scan(store, monkeypatch):
    """EDGAR times out on one name and the other four hundred are fine."""
    for t in ("GOOD", "BAD"):
        store.record_bars(t, [_bar(f"2026-02-{d:02d}", 100.0)
                              for d in range(1, 29)],
                          recorded_at="2026-02-28T21:00:00Z")
    store.record_bars(scanner.REGIME_TICKER,
                      [_bar(f"2026-02-{d:02d}", 100.0) for d in range(1, 29)],
                      recorded_at="2026-02-28T21:00:00Z")
    store.record_universe("2026-03-02", [
        {"ticker": "GOOD", "cik": "1", "eligible": True},
        {"ticker": "BAD", "cik": "2", "eligible": True}],
                          recorded_at="2026-03-02T21:00:00Z")

    def signal(ticker, as_of):
        if ticker == "BAD":
            raise ConnectionError("EDGAR timed out")
        return {"ticker": "GOOD", "success": True, "sue": 0.1,
                "known_at": "2026-03-02", "sigma_quarters": 8,
                "sigma_periods": ["2026Q1"], "basis_changes": [],
                "fiscal_period": "2026Q1"}

    monkeypatch.setattr(scanner, "_signal_for", signal)

    result = scanner.scan(as_of="2026-03-03")

    tickers = {r["ticker"] for r in result["rejected"]}
    assert "BAD" in tickers, "one ticker's outage took the whole scan down"
    reason = next(r["reason"] for r in result["rejected"] if r["ticker"] == "BAD")
    assert "timed out" in reason or "ConnectionError" in reason


# --- a run over many days keeps its invariants ------------------------------

def test_a_month_of_runs_never_breaks_the_anti_lookahead_rule(store,
                                                             monkeypatch):
    """The property that matters, checked over a whole simulated month rather
    than on one hand-built pair of rows."""
    import pandas as pd

    days = []
    d = date(2026, 3, 2)
    while len(days) < 20:
        if d.weekday() < 5:
            days.append(d.isoformat())
        d += timedelta(days=1)

    def frame_for(day, tickers):
        data = {}
        for t in tickers:
            for field in ("Open", "High", "Low", "Close", "Volume",
                          "Stock Splits", "Dividends"):
                data[(t, field)] = [100.0 if field != "Volume" else 1e6]
                if field in ("Stock Splits", "Dividends"):
                    data[(t, field)] = [0.0]
        f = pd.DataFrame(data, index=pd.to_datetime([day]))
        f.columns = pd.MultiIndex.from_tuples(f.columns)
        return f

    import yfinance
    for day in days:
        monkeypatch.setattr(
            yfinance, "download",
            lambda *a, _d=day, **k: frame_for(_d, k["tickers"].split()))
        daily_job.record_daily_bars(["AAA", "BBB"], as_of=day)

    for i, day in enumerate(days):
        seen = pit_store.bars_as_of("AAA", as_of=day)
        assert len(seen) == i + 1, (
            f"standing on {day}, {len(seen)} sessions visible, expected {i + 1}")
        assert all(b["trade_date"] <= day for b in seen)


def test_asking_for_nothing_is_not_a_failed_run(store, monkeypatch):
    """A run with an empty ticker list reached everything it was asked for."""
    monkeypatch.setattr(daily_job, "_fetch_bars", lambda tickers, **kw: {})
    result = daily_job.record_daily_bars([], as_of="2026-03-03")
    assert result["status"] != "failed", (
        "requesting zero tickers and getting zero back was called a failure")


# --- the property the whole storage design rests on -------------------------
#
# Prices are stored as-traded so they never move. That claim only holds if the
# conversion is invariant to WHEN the data was fetched: the vendor re-adjusts
# its history the day a split lands, and the un-adjustment multiplies by that
# same split, so the two changes have to cancel exactly. If they do not, every
# re-fetch after a corporate action files a revision against every historical
# bar and the store's central promise is false.


def _vendor_frame(sessions, splits_after, ticker="X", start=None, end=None):
    """What the vendor would return for a window, as of a later date: history
    divided by every split since, with the split column marking each ex-date
    that falls INSIDE the requested window -- which is the crux. A split after
    the window has already been applied to the prices and is nowhere in the
    response."""
    import pandas as pd

    rows = [s for s in sessions
            if (not start or s[0] >= start) and (not end or s[0] < end)]
    data = {}
    for sym in (ticker, daily_job.FETCH_CANARY):
        closes, vols, sp = [], [], []
        for day, raw_close, raw_vol in rows:
            factor = 1.0
            for ex, ratio in splits_after:
                if ex > day:
                    factor *= ratio
            closes.append(raw_close / factor)
            vols.append(raw_vol * factor)
            sp.append(next((r for e, r in splits_after
                            if e == day and (not start or e >= start)
                            and (not end or e < end)), 0.0))
        for field, series in (("Open", closes), ("High", closes),
                              ("Low", closes), ("Close", closes),
                              ("Volume", vols), ("Stock Splits", sp),
                              ("Dividends", [0.0] * len(closes))):
            data[(sym, field)] = list(series)
    if not rows:
        return pd.DataFrame()
    f = pd.DataFrame(data, index=pd.to_datetime([d for d, _, _ in rows]))
    f.columns = pd.MultiIndex.from_tuples(f.columns)
    return f


SESSIONS = [("2024-06-06", 1209.98, 66_469_600),
            ("2024-06-07", 1208.88, 41_238_600),
            ("2024-06-10", 121.79, 313_434_100),
            ("2025-03-03", 40.00, 900_000_000),
            ("2025-06-02", 44.00, 800_000_000)]
SPLIT = [("2024-06-10", 10.0)]
LATER = SPLIT + [("2025-03-03", 4.0)]


def test_backfilling_a_past_session_after_a_split_stores_the_wrong_price(store,
                                                                        monkeypatch):
    """The reachable form of the invariance problem, and `--as-of` invites it.

    record_daily_bars asks for one day. If a split has gone ex between that day
    and now, the vendor has already divided that day's price by it and the
    one-day window contains no split column to undo it with. The session is
    then stored 4x too small, in a table whose entire premise is that its
    numbers are what the stock actually printed.
    """
    import yfinance
    monkeypatch.setattr(yfinance, "download",
                        lambda *a, **k: _vendor_frame(
                            SESSIONS, LATER, start=k.get("start"),
                            end=k.get("end")))

    daily_job.record_daily_bars(["X"], as_of="2024-06-07")

    bar = pit_store.bars_as_of("X", "2024-06-08")[0]
    assert round(bar["close"], 2) == 1208.88, (
        f"backfilled 2024-06-07 at {bar['close']:.2f}; it printed 1208.88")


def test_a_bootstrap_run_after_a_split_agrees_with_one_before_it(store,
                                                                 monkeypatch):
    """Bootstrap windows do reach today, so this one should already hold --
    and it is the property that makes the store's central claim true."""
    import yfinance
    monkeypatch.setattr(yfinance, "download",
                        lambda *a, **k: _vendor_frame(
                            SESSIONS, SPLIT, start=k.get("start"),
                            end=k.get("end")))
    daily_job.bootstrap_history(["X"], as_of="2024-06-30")
    first = {b["trade_date"]: b["close"]
             for b in pit_store.bars_as_of("X", "2024-06-30")}

    monkeypatch.setattr(yfinance, "download",
                        lambda *a, **k: _vendor_frame(
                            SESSIONS, LATER, start=k.get("start"),
                            end=k.get("end")))
    daily_job.bootstrap_history(["X"], as_of="2025-06-30")

    assert pit_store.revisions("X") == [], (
        f"re-fetching after a split filed {len(pit_store.revisions('X'))} "
        f"revisions against history that did not change")
    second = {b["trade_date"]: b["close"]
              for b in pit_store.bars_as_of("X", "2025-06-30")}
    for day in first:
        assert round(first[day], 6) == round(second[day], 6), day
    assert round(first["2024-06-07"], 2) == 1208.88


def test_the_adjusted_view_moves_even_though_the_stored_one_does_not(store,
                                                                     monkeypatch):
    """The other half. Raw stays put; the adjusted view is supposed to change
    as new actions land, and to change only for readers standing after them."""
    import yfinance

    monkeypatch.setattr(yfinance, "download",
                        lambda *a, **k: _vendor_frame(SESSIONS, SPLIT))
    daily_job.bootstrap_history(["X"], as_of="2024-06-30")
    before = pit_store.adjusted_bars("X", "2024-12-01")

    pit_store.record_corporate_action("X", "2025-03-03", "split", 4.0,
                                      recorded_at="2025-03-03T21:00:00Z")

    after = pit_store.adjusted_bars("X", "2025-06-30")
    assert round(before[0]["close"], 4) == round(after[0]["close"] * 4, 4)
    # ...and the December reader is untouched by a March action.
    assert pit_store.adjusted_bars("X", "2024-12-01")[0]["close"] == \
        before[0]["close"]


# --- does the whole thing actually converge ---------------------------------

def test_a_cold_store_reaches_a_tradeable_universe(store, monkeypatch):
    """Bootstrap, then screen, then scan -- end to end over simulated days,
    checking that the pipeline produces a universe rather than starving."""
    import pandas as pd

    names = [f"N{i}" for i in range(40)]
    monkeypatch.setattr(daily_job, "_fetch_sec_tickers",
                        lambda: [{"ticker": t, "cik": str(i), "name": t}
                                 for i, t in enumerate(names)])
    monkeypatch.setattr(daily_job, "MAX_NIGHTLY_TICKERS", 20)
    monkeypatch.setattr(daily_job, "MIN_NEWCOMER_SLOTS", 10)
    monkeypatch.setattr(daily_job, "record_consensus_snapshots",
                        lambda **kw: {"status": "ok"})

    hist = []
    d = date(2025, 12, 1)
    while len(hist) < 90:
        if d.weekday() < 5:
            hist.append(d.isoformat())
        d += timedelta(days=1)

    def frame(*a, **k):
        asked = k["tickers"].split()
        start, end = k.get("start"), k.get("end")
        days = [x for x in hist if (not start or x >= start) and (not end or x < end)]
        if not days:
            return pd.DataFrame()
        data = {}
        for t in asked:
            for field in ("Open", "High", "Low", "Close"):
                data[(t, field)] = [50.0] * len(days)
            data[(t, "Volume")] = [1_000_000.0] * len(days)
            data[(t, "Stock Splits")] = [0.0] * len(days)
            data[(t, "Dividends")] = [0.0] * len(days)
        f = pd.DataFrame(data, index=pd.to_datetime(days))
        f.columns = pd.MultiIndex.from_tuples(f.columns)
        return f

    import yfinance
    monkeypatch.setattr(yfinance, "download", frame)

    # Four bootstrap nights, as the compose comment instructs.
    for day in hist[85:89]:
        daily_job.run_all(as_of=day, bootstrap=True)

    eligible = daily_job.eligible_tickers(hist[88])
    assert len(eligible) == 40, (
        f"after four bootstrap nights only {len(eligible)} of 40 names are "
        f"eligible; the universe is not filling")

    # And the ask stays bounded once it has.
    asked = daily_job.nightly_tickers(hist[88], names)
    assert len(asked) >= 40


def test_a_past_dated_bootstrap_has_the_same_problem(store, monkeypatch):
    """`--bootstrap --as-of <past>` is allowed, and its window ends at `as_of`
    too -- so every split since is baked into the prices with nothing in the
    response to undo it. Same defect as the one-day backfill, over two years of
    sessions instead of one."""
    import yfinance
    monkeypatch.setattr(yfinance, "download",
                        lambda *a, **k: _vendor_frame(
                            SESSIONS, LATER, start=k.get("start"),
                            end=k.get("end")))

    daily_job.bootstrap_history(["X"], as_of="2024-06-30")

    stored = {b["trade_date"]: b["close"]
              for b in pit_store.bars_as_of("X", "2024-06-30")}
    assert round(stored["2024-06-07"], 2) == 1208.88, (
        f"bootstrapped 2024-06-07 at {stored['2024-06-07']:.2f}")
    assert "2025-03-03" not in stored, (
        "a session after the bootstrap date was recorded as history")


# --- the market is shut about ten times a year ------------------------------
#
# On a holiday no US equity has a bar, so the canary is missing from every
# batch -- and a missing canary is how a rate-limited batch announces itself.
# The job cannot tell them apart from a one-day request, so it treats
# Thanksgiving as a total vendor outage: raises, exits non-zero, pages
# whoever is on call, and leaves the day permanently listed as a gap.
#
# Ten false alarms a year is how an alert stops being read.


def _holiday_frame(sessions, closed_on, **kw):
    """A vendor response for a window in which one weekday has no sessions."""
    import pandas as pd

    start, end = kw.get("start"), kw.get("end")
    days = [d for d in sessions
            if d != closed_on
            and (not start or d >= start) and (not end or d < end)]
    if not days:
        return pd.DataFrame()
    data = {}
    for t in kw["tickers"].split():
        for field in ("Open", "High", "Low", "Close"):
            data[(t, field)] = [100.0] * len(days)
        data[(t, "Volume")] = [1e6] * len(days)
        data[(t, "Stock Splits")] = [0.0] * len(days)
        data[(t, "Dividends")] = [0.0] * len(days)
    f = pd.DataFrame(data, index=pd.to_datetime(days))
    f.columns = pd.MultiIndex.from_tuples(f.columns)
    return f


WEEK = ["2026-11-23", "2026-11-24", "2026-11-25", "2026-11-26", "2026-11-27"]


def test_a_market_holiday_is_not_a_vendor_outage(store, monkeypatch):
    import yfinance
    monkeypatch.setattr(yfinance, "download",
                        lambda *a, **k: _holiday_frame(WEEK, "2026-11-26", **k))
    monkeypatch.setattr(daily_job, "FETCH_RETRY_BACKOFF", 0.0)
    monkeypatch.setattr(daily_job, "FETCH_BATCH_PAUSE", 0.0)

    result = daily_job.record_daily_bars(["AAA", "BBB"], as_of="2026-11-26")

    assert result["status"] != "failed", (
        f"the market being shut was reported as {result['status']}")
    assert pit_store.bars_as_of("AAA", "2026-11-26") == [], (
        "a bar was invented for a day the exchange did not open")


def test_a_closed_day_is_not_a_hole_in_the_record(store, monkeypatch):
    """It must not sit in missing_days forever, or the one real gap is lost
    among ten holidays."""
    import yfinance
    monkeypatch.setattr(yfinance, "download",
                        lambda *a, **k: _holiday_frame(WEEK, "2026-11-26", **k))
    monkeypatch.setattr(daily_job, "FETCH_RETRY_BACKOFF", 0.0)
    monkeypatch.setattr(daily_job, "FETCH_BATCH_PAUSE", 0.0)

    daily_job.record_daily_bars(["AAA"], as_of="2026-11-26")

    assert pit_store.missing_days("daily_bars", "2026-11-26", "2026-11-26") == []


def test_a_closed_day_exits_zero(store, monkeypatch):
    """A scheduler that pages on Thanksgiving is a scheduler nobody reads."""
    monkeypatch.setattr(daily_job, "run_all",
                        lambda **kw: {"as_of": "2026-11-26",
                                      "daily_bars": {"status": "closed"},
                                      "universe": {"status": "ok"},
                                      "consensus": {"status": "ok"}})
    assert daily_job.main(["--as-of", "2026-11-26"]) == 0


def test_a_genuine_outage_is_still_loud(store, monkeypatch):
    """The other side. If the canary has no bar anywhere in the window, the
    vendor did not answer -- and that is not a holiday."""
    import pandas as pd
    import yfinance
    monkeypatch.setattr(yfinance, "download", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(daily_job, "FETCH_RETRY_BACKOFF", 0.0)
    monkeypatch.setattr(daily_job, "FETCH_BATCH_PAUSE", 0.0)

    result = daily_job.record_daily_bars(["AAA"], as_of="2026-11-26")
    assert result["status"] == "failed"


# --- two writers ------------------------------------------------------------
#
# The recorder takes minutes over a real universe and the scanner reads the
# same file half an hour later. Those windows overlap the moment a run goes
# long or someone re-scans by hand.
#
# This holds today for a reason worth writing down: python's sqlite3 connects
# with timeout=5.0, so a blocked writer waits rather than raising, and writes
# here are per-call transactions that clear in milliseconds. Both facts are
# defaults nobody chose deliberately, which is exactly the kind of thing a
# later `connect(..., timeout=0)` or a long-held transaction would undo
# silently. These two pin the behaviour rather than the mechanism.


def test_a_second_writer_waits_rather_than_dying(store, tmp_path):
    import sqlite3
    import threading

    blocker = pit_store.connect()
    blocker.execute("BEGIN IMMEDIATE")
    blocker.execute(
        "INSERT INTO daily_bar (trade_date, ticker, close, recorded_at) "
        "VALUES ('2026-03-03','LOCK',1.0,'2026-03-03T21:00:00Z')")

    outcome = {}

    def writer():
        try:
            pit_store.record_bars("OTHER", [_bar("2026-03-03", 10.0)],
                                  recorded_at="2026-03-03T21:00:00Z")
            outcome["ok"] = True
        except sqlite3.OperationalError as exc:
            outcome["error"] = str(exc)

    thread = threading.Thread(target=writer)
    thread.start()
    thread.join(timeout=0.4)
    blocker.commit()
    blocker.close()
    thread.join(timeout=5)

    assert "error" not in outcome, (
        f"the second writer gave up instead of waiting: {outcome.get('error')}")
    assert outcome.get("ok")


def test_a_reader_is_not_blocked_by_a_writer(store):
    """WAL, in other words. Without it the scanner's reads stall behind the
    recorder's whole transaction."""
    blocker = pit_store.connect()
    blocker.execute("BEGIN IMMEDIATE")
    blocker.execute(
        "INSERT INTO daily_bar (trade_date, ticker, close, recorded_at) "
        "VALUES ('2026-03-03','LOCK',1.0,'2026-03-03T21:00:00Z')")
    try:
        assert pit_store.bars_as_of("ANY", "2026-03-03") == []
    finally:
        blocker.commit()
        blocker.close()


# --- a decision re-made ------------------------------------------------------

def test_rerunning_a_scan_with_a_different_answer_says_so(store, monkeypatch):
    """Filing is INSERT OR IGNORE, so the first decision of a day stands -- as
    it should, since it is the one that would have been acted on. But `scan()`
    returns the new answer while the store keeps the old one, and nothing in
    between says they disagree. A parameter changed at lunchtime and re-run
    would leave the operator reading candidates the record does not contain.
    """
    from research import scanner as sc

    for t in ("AAA", "BBB"):
        store.record_bars(t, [_bar(f"2026-02-{d:02d}", 100.0)
                              for d in range(1, 29)],
                          recorded_at="2026-02-28T21:00:00Z")
    store.record_bars(sc.REGIME_TICKER,
                      [_bar(f"2026-02-{d:02d}", 100.0) for d in range(1, 29)],
                      recorded_at="2026-02-28T21:00:00Z")
    store.record_universe("2026-03-02", [
        {"ticker": "AAA", "cik": "1", "eligible": True},
        {"ticker": "BBB", "cik": "2", "eligible": True}],
                          recorded_at="2026-03-02T21:00:00Z")

    def sig(t, as_of, _v={"AAA": 3.0, "BBB": 0.2}):
        return {"ticker": t, "success": True, "sue": _v[t],
                "known_at": "2026-03-02", "sigma_quarters": 8,
                "sigma_periods": ["2026Q1"], "basis_changes": [],
                "fiscal_period": "2026Q1"}

    monkeypatch.setattr(sc, "_signal_for", sig)
    monkeypatch.setattr(sc, "_cost_for", lambda t, a, d: {
        "cost": 0.0001, "cost_floor": 0.00002, "reason": None,
        "spread": 0.00005, "resolved": True})

    first = sc.record_scan(as_of="2026-03-03")
    assert [c["ticker"] for c in first["candidates"]] == ["AAA"]

    # Threshold relaxed at lunchtime; BBB now qualifies.
    monkeypatch.setattr(sc, "MIN_ABS_SUE", 0.1)
    second = sc.record_scan(as_of="2026-03-03")

    assert [c["ticker"] for c in second["candidates"]] == ["AAA", "BBB"]
    stored = {o["ticker"] for o in
              pit_store.paper_orders_as_of("2026-03-03", accepted_only=True)}
    assert stored == {"AAA"}, "the earlier decision was overwritten"
    assert second.get("superseded"), (
        "the store kept a different decision and the result did not mention it")


# --- the contract between the recorder and the estimator --------------------
#
# spread.py excludes split sessions by looking them up in corporate_action --
# not by guessing from the size of the move, which is the right design. It
# therefore depends entirely on the recorder having written them. Nothing did:
# record_actions was never wired into run_all, so the table stayed empty and
# the protection never fired once. With as-traded storage a 10-for-1 is a
# genuine -89.9% print between two adjacent bars, and EDGE compares only
# adjacent bars.


def test_the_estimator_sees_the_splits_the_recorder_writes(store, monkeypatch):
    from research import spread as sp
    import pandas as pd

    sessions = []
    d = date(2026, 1, 1)
    while len(sessions) < 60:
        if d.weekday() < 5:
            sessions.append(d.isoformat())
        d += timedelta(days=1)
    ex = sessions[40]

    def frame(*a, **k):
        rows = [s for s in sessions
                if (not k.get("start") or s >= k["start"])
                and (not k.get("end") or s < k["end"])]
        data = {}
        for t in k["tickers"].split():
            closes = [100.0 if s < ex else 10.0 for s in rows]
            data[(t, "Open")] = closes
            data[(t, "High")] = [c * 1.01 for c in closes]
            data[(t, "Low")] = [c * 0.99 for c in closes]
            data[(t, "Close")] = closes
            data[(t, "Volume")] = [5e6] * len(rows)
            data[(t, "Stock Splits")] = [10.0 if s == ex else 0.0 for s in rows]
            data[(t, "Dividends")] = [0.0] * len(rows)
        f = pd.DataFrame(data, index=pd.to_datetime(rows))
        f.columns = pd.MultiIndex.from_tuples(f.columns)
        return f

    import yfinance
    monkeypatch.setattr(yfinance, "download", frame)
    daily_job.bootstrap_history(["X"], as_of=sessions[-1])

    splits = [a for a in pit_store.corporate_actions_as_of("X", sessions[-1])
              if a["action_type"] == "split"]
    assert [(a["ex_date"], a["value"]) for a in splits] == [(ex, 10.0)], (
        "the recorder did not write the split the estimator needs")

    est = sp.estimate_spread("X", sessions[-1], window=55)
    assert est["splits_excluded"] == 1, (
        "the split session was fed to EDGE, which only ever compares adjacent "
        "bars")
    # And the result is a spread rather than a measurement of the split.
    bound = est["spread_upper"]
    assert bound is None or bound < 0.5, f"spread_upper came back {bound}"


# --- degenerate market data --------------------------------------------------
#
# The screen keeps names above $500k of median dollar volume, but the scanner
# sizes against `participation_rate` directly and nothing guarantees the two
# see the same window. A name whose median session traded nothing divides by
# zero.


def _flat(ticker, n, close=10.0, volume=1_000_000, store=None):
    days = []
    d = date(2026, 1, 1)
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d.isoformat())
        d += timedelta(days=1)
    store.record_bars(ticker, [
        {"trade_date": x, "open": close, "high": close, "low": close,
         "close": close, "volume": volume} for x in days],
        recorded_at=f"{days[-1]}T21:00:00Z")
    return days


def test_a_name_that_traded_nothing_does_not_divide_by_zero(store):
    from research import spread as sp
    days = _flat("DEAD", 300, volume=0.0, store=store)

    out = sp.participation_rate("DEAD", days[-1], 5000.0,
                                window=sp.RESOLVING_WINDOW)
    assert out["reason"], "a zero-volume name came back sizeable"
    assert out["within_limit"] is False


def test_a_zero_volume_name_is_refused_by_the_scanner(store, monkeypatch):
    from research import scanner as sc
    days = _flat("DEAD", 300, volume=0.0, store=store)
    _flat(sc.REGIME_TICKER, 300, store=store)
    store.record_universe(days[-2], [
        {"ticker": "DEAD", "cik": "1", "eligible": True}],
                          recorded_at=f"{days[-2]}T21:00:00Z")
    monkeypatch.setattr(sc, "_signal_for", lambda t, a: {
        "ticker": t, "success": True, "sue": 3.0, "known_at": a,
        "sigma_quarters": 8, "sigma_periods": ["2026Q1"],
        "basis_changes": [], "fiscal_period": "2026Q1"})

    result = sc.scan(as_of=days[-1])
    assert result["candidates"] == []
    assert result["rejected"], "no candidate and no reason"


def test_a_perfectly_flat_price_series_is_refused_not_priced(store):
    """Every high equals every low, so there is no range for EDGE to read a
    spread out of. Returning zero would say trading is free."""
    from research import spread as sp
    days = _flat("FLAT", 300, store=store)

    out = sp.estimate_spread("FLAT", days[-1], window=sp.RESOLVING_WINDOW)
    assert out["spread"] is None or out["spread"] > 0
    if out["spread"] is None:
        assert out["reason"]


# --- the regime input is nobody's responsibility ----------------------------
#
# _regime_scale reads the index's own sessions and returns "unknown" at full
# book size when it has fewer than twenty. Nothing guarantees it has any. The
# index is fetched every night as the liveness canary and then deliberately
# dropped before writing, so it lands in the store only if it happens to be in
# the universe -- and a name that is not eligible is only fetched on its turn
# in the rotation, which is a bar every few weeks.
#
# The failure is silent and it fails open: every scan runs the full book and
# reports "unknown", which reads like a calm market rather than a missing one.


def test_the_index_is_recorded_even_though_it_is_not_a_candidate(store,
                                                                 monkeypatch):
    import pandas as pd
    import yfinance

    days = ["2026-03-02", "2026-03-03"]

    def frame(*a, **k):
        rows = [d for d in days
                if (not k.get("start") or d >= k["start"])
                and (not k.get("end") or d < k["end"])]
        data = {}
        for t in k["tickers"].split():
            for f in ("Open", "High", "Low", "Close"):
                data[(t, f)] = [100.0] * len(rows)
            data[(t, "Volume")] = [1e6] * len(rows)
            data[(t, "Stock Splits")] = [0.0] * len(rows)
            data[(t, "Dividends")] = [0.0] * len(rows)
        f_ = pd.DataFrame(data, index=pd.to_datetime(rows))
        f_.columns = pd.MultiIndex.from_tuples(f_.columns)
        return f_

    monkeypatch.setattr(yfinance, "download", frame)
    monkeypatch.setattr(daily_job, "FETCH_RETRY_BACKOFF", 0.0)
    monkeypatch.setattr(daily_job, "FETCH_BATCH_PAUSE", 0.0)

    daily_job.record_daily_bars(["AAA"], as_of="2026-03-03")

    assert pit_store.bars_as_of(scanner.REGIME_TICKER, "2026-03-03"), (
        "the index the regime is measured from was fetched and thrown away")


def test_the_index_is_not_smuggled_into_the_universe(store, monkeypatch):
    """It is recorded for its prices, not proposed as a trade."""
    monkeypatch.setattr(daily_job, "_fetch_sec_tickers", lambda: [
        {"ticker": "AAA", "cik": "1", "name": "A Co"}])
    daily_job.refresh_universe(as_of="2026-03-03")
    members = {m["ticker"] for m in pit_store.universe_as_of("2026-03-03")}
    assert scanner.REGIME_TICKER not in members


def test_the_bootstrap_records_the_index_history_too(store, monkeypatch):
    """The nightly path asks for the canary by name so it survives the strip;
    the bootstrap does not, so the index's HISTORY was never written -- and the
    regime needs 252 sessions of it. Live, that left every scan reporting
    "unknown" at full book size after a complete bootstrap of 2,435 names."""
    import pandas as pd
    import yfinance

    days = _weekdays_for(60)

    def frame(*a, **k):
        rows = [d for d in days
                if (not k.get("start") or d >= k["start"])
                and (not k.get("end") or d < k["end"])]
        data = {}
        for t in k["tickers"].split():
            for f in ("Open", "High", "Low", "Close"):
                data[(t, f)] = [100.0] * len(rows)
            data[(t, "Volume")] = [1e6] * len(rows)
            data[(t, "Stock Splits")] = [0.0] * len(rows)
            data[(t, "Dividends")] = [0.0] * len(rows)
        f_ = pd.DataFrame(data, index=pd.to_datetime(rows))
        f_.columns = pd.MultiIndex.from_tuples(f_.columns)
        return f_

    monkeypatch.setattr(yfinance, "download", frame)
    monkeypatch.setattr(daily_job, "_today", lambda: days[-1])

    daily_job.bootstrap_history(["AAA"], as_of=days[-1])

    # Like for like against an ordinary name. The assertion is a count
    # against a count, so it holds whether or not the window reaches the
    # as_of session -- and it now does: bootstrap_history ends at
    # max(as_of, today) + 1 because a newcomer's first rotation night is
    # that call, and stopping a session short left a permanent hole.
    got = pit_store.bars_as_of(scanner.REGIME_TICKER, days[-1])
    peer = pit_store.bars_as_of("AAA", days[-1])
    assert peer, "the fixture recorded nothing at all"
    assert len(got) == len(peer), (
        f"the index has {len(got)} sessions where a name bootstrapped "
        f"alongside it has {len(peer)}")


def _weekdays_for(n):
    out, d = [], date(2026, 1, 1)
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += timedelta(days=1)
    return out
