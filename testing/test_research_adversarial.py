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
        for i in range(100)])

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
        for i in range(100)])

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
        for i in range(100)])
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
        {"ticker": "BAD", "cik": "2", "eligible": True}])

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
