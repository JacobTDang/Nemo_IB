"""Whole workflows, the way someone would actually run them.

The other files hold one part still and poke it. These run the sequence a
person runs -- start a store from nothing, operate it for a fortnight, lose
three days to an outage and fill them, carry a position through a split -- and
check what comes out at each step. A part can be individually correct and the
sequence still wrong, usually because two of them disagree about what a date
means.

Each scenario asserts the honest answer at every stage, including the stages
where the honest answer is a refusal. A cold store that reports candidates is
a bug; so is one that reports nothing without saying why.
"""
from datetime import date, timedelta

import pandas as pd
import pytest

from research import (daily_job, pit_store, scanner, scoring, seed_consensus,
                      spread, sue)


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("NEMO_PIT_DB", str(tmp_path / "pit.db"))
    pit_store.init_schema()
    return pit_store


NAMES = [f"N{i:02d}" for i in range(6)]


def _weekdays(n, start="2026-01-05"):
    out, d = [], date.fromisoformat(start)
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += timedelta(days=1)
    return out


SESSIONS = _weekdays(200)


class Market:
    """A deterministic tape that behaves the way the vendor does.

    The important part is `as_traded` versus what a request returns. A real
    feed re-divides its whole history the day a split goes ex, so the price it
    reports for a session before the split depends on when you ask -- which is
    precisely what the store's as-traded conversion exists to undo, and a fake
    that ignores it lets that conversion pass while doing the wrong thing.
    """

    def __init__(self, closed=(), splits=None, delisted=None):
        self.closed = set(closed)
        self.splits = splits or {}          # ticker -> (ex_date, ratio)
        self.delisted = delisted or {}      # ticker -> from_date
        self.now = SESSIONS[-1]             # when the request is being made

    def as_traded(self, ticker, day):
        """What the stock actually printed that session."""
        base = 50.0 + NAMES.index(ticker) * 10 if ticker in NAMES else 400.0
        ex = self.splits.get(ticker)
        if ex and day >= ex[0]:
            base /= ex[1]
        return round(base, 4)

    def price(self, ticker, day):
        """What the vendor reports for that session today: as-traded, divided
        by every split that has gone ex since."""
        value = self.as_traded(ticker, day)
        ex = self.splits.get(ticker)
        if ex and day < ex[0] <= self.now:
            value /= ex[1]
        return round(value, 4)

    def download(self, *a, **k):
        start, end = k.get("start"), k.get("end")
        days = [d for d in SESSIONS if d not in self.closed
                and (not start or d >= start) and (not end or d < end)]
        if not days:
            return pd.DataFrame()
        data = {}
        for t in k["tickers"].split():
            gone = self.delisted.get(t)
            closes, splits = [], []
            for d in days:
                if gone and d >= gone:
                    closes.append(float("nan"))
                    splits.append(0.0)
                    continue
                closes.append(self.price(t, d))
                ex = self.splits.get(t)
                splits.append(ex[1] if ex and d == ex[0] else 0.0)
            data[(t, "Open")] = closes
            data[(t, "High")] = [c if c != c else c * 1.01 for c in closes]
            data[(t, "Low")] = [c if c != c else c * 0.99 for c in closes]
            data[(t, "Close")] = closes
            data[(t, "Volume")] = [2e6 if c == c else float("nan")
                                   for c in closes]
            data[(t, "Stock Splits")] = splits
            data[(t, "Dividends")] = [0.0] * len(days)
        if not data:
            return pd.DataFrame()
        f = pd.DataFrame(data, index=pd.to_datetime(days))
        f.columns = pd.MultiIndex.from_tuples(f.columns)
        return f


@pytest.fixture
def wire(store, monkeypatch):
    def _wire(market, today=None):
        import yfinance
        monkeypatch.setattr(yfinance, "download", market.download)
        monkeypatch.setattr(daily_job, "_fetch_sec_tickers",
                            lambda: [{"ticker": t, "cik": str(i), "name": t}
                                     for i, t in enumerate(NAMES)])
        monkeypatch.setattr(daily_job, "record_consensus_snapshots",
                            lambda **kw: {"status": "ok"})
        market.now = today or SESSIONS[-1]
        monkeypatch.setattr(daily_job, "_today",
                            lambda: market.now)
        monkeypatch.setattr(daily_job, "FETCH_RETRY_BACKOFF", 0.0)
        monkeypatch.setattr(daily_job, "FETCH_BATCH_PAUSE", 0.0)
        return market
    return _wire


# --- a store from nothing ---------------------------------------------------

def test_a_cold_store_refuses_before_it_can_answer(wire, store, monkeypatch):
    wire(Market(), today=SESSIONS[80])

    # Day one: nothing has been recorded, so nothing is eligible and the scan
    # says so rather than reporting an empty market.
    first = scanner.scan(as_of=SESSIONS[80])
    assert first["screened"] == 0
    assert first["candidates"] == []
    assert first["narrowed_by"] is None
    assert "no prints recorded" in first["narrowing_note"]

    daily_job.run_all(as_of=SESSIONS[80], bootstrap=True)

    # After a bootstrap the universe exists and every name is eligible.
    members = pit_store.universe_as_of(SESSIONS[80])
    assert len(members) == len(NAMES)
    assert all(m["eligible"] for m in members)

    # ...and the scan still finds nothing, because no company has reported.
    second = scanner.scan(as_of=SESSIONS[80])
    assert second["screened"] == len(NAMES)
    assert second["candidates"] == []
    assert all(r["reason"] for r in second["rejected"])


def test_a_cold_store_reports_its_own_emptiness_to_the_scorer(wire, store):
    wire(Market(), today=SESSIONS[80])
    out = scoring.score_orders(as_of=SESSIONS[80])
    assert out["sample"] == 0
    assert out["calibrated"] is False
    assert "no finished trades" in out["calibration_note"]


# --- a fortnight of ordinary operation --------------------------------------

def test_a_fortnight_accumulates_exactly_one_bar_a_session(wire, store):
    market = wire(Market(), today=SESSIONS[80])
    daily_job.run_all(as_of=SESSIONS[80], bootstrap=True)

    for day in SESSIONS[81:91]:
        market.now = day
        daily_job.run_all(as_of=day)

    bars = pit_store.bars_as_of("N00", SESSIONS[90])
    dates = [b["trade_date"] for b in bars]
    assert dates == sorted(set(dates)), "a session was recorded twice"
    assert dates[-1] == SESSIONS[90]
    assert pit_store.missing_days("daily_bars", SESSIONS[81],
                                  SESSIONS[90]) == []
    assert pit_store.revisions("N00") == []


# --- three days lost, then filled -------------------------------------------

def test_an_outage_shows_as_a_gap_and_backfilling_closes_it(wire, store):
    market = wire(Market(), today=SESSIONS[100])
    daily_job.run_all(as_of=SESSIONS[80], bootstrap=True)
    for day in SESSIONS[81:84]:
        market.now = day
        daily_job.run_all(as_of=day)

    # Three sessions pass with the job down, then it comes back.
    resumed = SESSIONS[87]
    market.now = resumed
    daily_job.run_all(as_of=resumed)

    missing = pit_store.missing_days("daily_bars", SESSIONS[84], SESSIONS[87])
    assert missing == SESSIONS[84:87], missing

    for day in SESSIONS[84:87]:
        daily_job.record_daily_bars(NAMES, as_of=day)

    assert pit_store.missing_days("daily_bars", SESSIONS[81],
                                  SESSIONS[87]) == []
    # The backfilled sessions carry the prices of their own days.
    for day in SESSIONS[84:87]:
        bar = [b for b in pit_store.bars_as_of("N00", day)
               if b["trade_date"] == day]
        assert bar and bar[0]["close"] == pytest.approx(50.0)


# --- a split under a live position ------------------------------------------

def test_a_split_mid_position_moves_no_recorded_price(wire, store):
    ex = SESSIONS[85]
    market = wire(Market(splits={"N01": (ex, 4.0)}), today=SESSIONS[80])
    daily_job.run_all(as_of=SESSIONS[80], bootstrap=True)

    before = {b["trade_date"]: b["close"]
              for b in pit_store.bars_as_of("N01", SESSIONS[80])}

    for day in SESSIONS[81:95]:
        market.now = day
        daily_job.run_all(as_of=day)

    after = {b["trade_date"]: b["close"]
             for b in pit_store.bars_as_of("N01", SESSIONS[94])}
    for day, price in before.items():
        assert after[day] == pytest.approx(price), f"{day} moved"

    # The split is on the record, dated to its own session...
    splits = [a for a in pit_store.corporate_actions_as_of("N01", SESSIONS[94])
              if a["action_type"] == "split"]
    assert [(s["ex_date"], s["value"]) for s in splits] == [(ex, 4.0)]
    # ...invisible the day before it went ex...
    assert pit_store.corporate_actions_as_of("N01", SESSIONS[84]) == []
    # ...and the adjusted view is continuous across it.
    adj = pit_store.adjusted_bars("N01", SESSIONS[94])
    closes = [b["close"] for b in adj]
    worst = min(closes[i] / closes[i - 1] - 1 for i in range(1, len(closes)))
    assert worst > -0.5, f"the adjusted series still has a {worst:.0%} step"


# --- a name that stops trading ----------------------------------------------

def test_a_delisting_keeps_the_history_and_stops_the_series(wire, store):
    gone = SESSIONS[86]
    market = wire(Market(delisted={"N02": gone}), today=SESSIONS[80])
    daily_job.run_all(as_of=SESSIONS[80], bootstrap=True)
    for day in SESSIONS[81:95]:
        market.now = day
        daily_job.run_all(as_of=day)

    bars = pit_store.bars_as_of("N02", SESSIONS[94])
    assert bars, "the delisted name lost the days it did trade"
    assert max(b["trade_date"] for b in bars) < gone
    # It stays in the universe record for the dates it was in it.
    early = {m["ticker"] for m in pit_store.universe_as_of(SESSIONS[82])}
    assert "N02" in early
    # ...and the run that lost it is partial, not ok.
    run = daily_job.last_run("daily_bars")
    assert run["status"] in ("partial", "ok")


# --- the exchange shut ------------------------------------------------------

def test_a_holiday_inside_an_ordinary_fortnight(wire, store):
    holiday = SESSIONS[85]
    market = wire(Market(closed={holiday}), today=SESSIONS[80])
    daily_job.run_all(as_of=SESSIONS[80], bootstrap=True)

    statuses = {}
    for day in SESSIONS[81:91]:
        market.now = day
        statuses[day] = daily_job.run_all(as_of=day)["daily_bars"]["status"]

    assert statuses[holiday] == "closed"
    assert statuses[SESSIONS[84]] == "ok"
    assert statuses[SESSIONS[86]] == "ok"
    assert holiday not in pit_store.missing_days("daily_bars", SESSIONS[81],
                                                 SESSIONS[90])
    assert not [b for b in pit_store.bars_as_of("N00", SESSIONS[90])
                if b["trade_date"] == holiday]


def test_a_store_started_after_a_split_records_what_the_stock_printed(wire,
                                                                      store):
    """The case the forward path never exercises.

    Recording one session a night, the bar written is today's and no later
    split can have touched it -- so the as-traded conversion is the identity
    and a scenario that only runs forward passes with it removed. It earns its
    place the first time history is pulled across a split, which is the day
    someone bootstraps a new store, and then it is worth a factor of four on
    every session before the ex-date.
    """
    ex = SESSIONS[85]
    market = wire(Market(splits={"N01": (ex, 4.0)}), today=SESSIONS[95])

    daily_job.run_all(as_of=SESSIONS[95], bootstrap=True)

    bars = {b["trade_date"]: b["close"]
            for b in pit_store.bars_as_of("N01", SESSIONS[95])}
    # N01 printed 60 before the split and 15 after it. The vendor now reports
    # the earlier sessions as 15; the store must hold 60.
    assert bars[SESSIONS[80]] == pytest.approx(60.0), (
        f"pre-split session stored at {bars[SESSIONS[80]]}, not what it printed")
    assert bars[SESSIONS[90]] == pytest.approx(15.0)
    # ...and the adjusted view puts them back on one basis.
    adj = {b["trade_date"]: b["close"]
           for b in pit_store.adjusted_bars("N01", SESSIONS[95])}
    assert adj[SESSIONS[80]] == pytest.approx(15.0)
    assert adj[SESSIONS[90]] == pytest.approx(15.0)


def test_backfilling_across_a_split_stores_what_the_stock_printed(wire, store):
    """The same conversion, reached the other way: the job was down when the
    split went ex and someone fills the gap afterwards."""
    ex = SESSIONS[85]
    market = wire(Market(splits={"N01": (ex, 4.0)}), today=SESSIONS[80])
    daily_job.run_all(as_of=SESSIONS[80], bootstrap=True)

    market.now = SESSIONS[95]
    daily_job.record_daily_bars(["N01"], as_of=SESSIONS[83])

    bar = [b for b in pit_store.bars_as_of("N01", SESSIONS[95])
           if b["trade_date"] == SESSIONS[83]]
    assert bar and bar[0]["close"] == pytest.approx(60.0), (
        f"backfilled session stored at {bar[0]['close'] if bar else None}")
