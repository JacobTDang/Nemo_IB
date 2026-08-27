"""A month of the whole thing, with the awkward days left in.

The unit tests each hold one part still and poke it. This runs the parts
against each other over simulated time -- bootstrap, nightly recording, the
screen, the scan -- with a split, a market holiday, a delisting and a re-run in
the middle, and checks the properties that are supposed to survive all of it.

Those properties are the reason the store exists, so they are stated as
assertions rather than left to be true:

  Nothing is ever visible before it was known.
  A day the exchange was shut is not a hole, and a day the job missed is.
  Prices are what the stock printed, and the adjusted view is rebuilt from
  actions the reader could have seen.
  Running any of it twice changes nothing.
"""
from datetime import date, timedelta

import pytest

from research import daily_job, pit_store, scanner


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("NEMO_PIT_DB", str(tmp_path / "pit.db"))
    pit_store.init_schema()
    return pit_store


NAMES = [f"N{i:02d}" for i in range(12)]
SPLIT_NAME, SPLIT_RATIO = "N03", 4.0
DELISTED = "N11"


def _weekdays(start, count):
    out, d = [], date.fromisoformat(start)
    while len(out) < count:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += timedelta(days=1)
    return out


SESSIONS = _weekdays("2026-01-05", 140)
SPLIT_DAY = SESSIONS[100]
HOLIDAY = SESSIONS[120]
DELIST_DAY = SESSIONS[110]


def _price(ticker, day):
    """Deterministic, and stepped down for the one name that splits.

    The canary is priced too: every fetch carries it, and a vendor that has no
    quote for it is how this code reports an outage.
    """
    rank = NAMES.index(ticker) if ticker in NAMES else len(NAMES)
    base = 50.0 + rank * 10 + SESSIONS.index(day) * 0.05
    if ticker == SPLIT_NAME and day >= SPLIT_DAY:
        base /= SPLIT_RATIO
    return round(base, 4)


def _vendor(*a, **k):
    """Prices as the vendor reports them today: already divided by any split
    that has gone ex, with the split column marking the ex-date if it is in
    the requested window."""
    import pandas as pd

    start, end = k.get("start"), k.get("end")
    days = [s for s in SESSIONS
            if s != HOLIDAY
            and (not start or s >= start) and (not end or s < end)]
    if not days:
        return pd.DataFrame()

    data = {}
    for t in k["tickers"].split():
        rows = days
        if t == DELISTED:
            rows = [d for d in days if d < DELIST_DAY]
            if not rows:
                continue
        closes, splits, idx = [], [], []
        for day in days:
            if t == DELISTED and day >= DELIST_DAY:
                closes.append(float("nan"))
                splits.append(0.0)
                idx.append(day)
                continue
            closes.append(_price(t, day) if t != SPLIT_NAME
                          else _price(t, day) if day >= SPLIT_DAY
                          else _price(t, day) / SPLIT_RATIO)
            splits.append(SPLIT_RATIO if (t == SPLIT_NAME and day == SPLIT_DAY)
                          else 0.0)
            idx.append(day)
        data[(t, "Open")] = closes
        data[(t, "High")] = [c if c != c else c * 1.01 for c in closes]
        data[(t, "Low")] = [c if c != c else c * 0.99 for c in closes]
        data[(t, "Close")] = closes
        data[(t, "Volume")] = [2_000_000.0 if c == c else float("nan")
                               for c in closes]
        data[(t, "Stock Splits")] = splits
        data[(t, "Dividends")] = [0.0] * len(days)
    if not data:
        return pd.DataFrame()
    f = pd.DataFrame(data, index=pd.to_datetime(days))
    f.columns = pd.MultiIndex.from_tuples(f.columns)
    return f


@pytest.fixture
def pipeline(store, monkeypatch):
    import yfinance
    monkeypatch.setattr(yfinance, "download", _vendor)
    monkeypatch.setattr(daily_job, "_fetch_sec_tickers",
                        lambda: [{"ticker": t, "cik": str(i), "name": t}
                                 for i, t in enumerate(NAMES)])
    monkeypatch.setattr(daily_job, "record_consensus_snapshots",
                        lambda **kw: {"status": "ok"})
    monkeypatch.setattr(daily_job, "_today", lambda: SESSIONS[-1])
    monkeypatch.setattr(daily_job, "FETCH_RETRY_BACKOFF", 0.0)
    monkeypatch.setattr(daily_job, "FETCH_BATCH_PAUSE", 0.0)

    def signal(ticker, as_of):
        return {"ticker": ticker, "success": True, "error": None,
                "sue": 2.0 + NAMES.index(ticker) * 0.1,
                "fiscal_period": "2026Q1", "known_at": as_of,
                "sigma": 0.1, "sigma_quarters": 8,
                "sigma_periods": ["2026Q1"], "basis_changes": []}

    monkeypatch.setattr(scanner, "_signal_for", signal)
    return store


def _run_days(days, bootstrap_first=0):
    for i, day in enumerate(days):
        daily_job.run_all(as_of=day, bootstrap=i < bootstrap_first)


# --- the properties ---------------------------------------------------------

def test_nothing_is_visible_before_it_was_known(pipeline):
    days = SESSIONS[95:125]
    _run_days(days, bootstrap_first=1)

    for day in days:
        for ticker in ("N00", SPLIT_NAME):
            for bar in pit_store.bars_as_of(ticker, as_of=day):
                assert bar["trade_date"] <= day, (
                    f"standing on {day}, {ticker} shows {bar['trade_date']}")
            for action in pit_store.corporate_actions_as_of(ticker, as_of=day):
                assert action["ex_date"] <= day, (
                    f"standing on {day}, {ticker} shows an action dated "
                    f"{action['ex_date']}")


def test_a_split_is_invisible_until_it_goes_ex(pipeline):
    _run_days(SESSIONS[95:110], bootstrap_first=1)

    before = SESSIONS[99]
    after = SESSIONS[105]
    assert pit_store.corporate_actions_as_of(SPLIT_NAME, before) == []
    assert [a["value"] for a in
            pit_store.corporate_actions_as_of(SPLIT_NAME, after)] == [SPLIT_RATIO]

    # The as-traded price never moves; only the adjusted view does.
    raw_before = {b["trade_date"]: b["close"]
                  for b in pit_store.bars_as_of(SPLIT_NAME, before)}
    raw_after = {b["trade_date"]: b["close"]
                 for b in pit_store.bars_as_of(SPLIT_NAME, after)}
    for day, value in raw_before.items():
        assert raw_after[day] == value, f"{day} moved after the split"

    adj = {b["trade_date"]: b["close"]
           for b in pit_store.adjusted_bars(SPLIT_NAME, after)}
    assert adj[SESSIONS[99]] == pytest.approx(raw_after[SESSIONS[99]]
                                              / SPLIT_RATIO)


def test_the_holiday_is_not_a_gap_and_the_skipped_day_is(pipeline):
    days = [d for d in SESSIONS[115:125] if d != SESSIONS[118]]
    _run_days(days, bootstrap_first=1)

    missing = pit_store.missing_days("daily_bars", SESSIONS[115], SESSIONS[124])
    assert SESSIONS[118] in missing, "a day the job never ran is not reported"
    assert HOLIDAY not in missing, "the exchange being shut is reported as a gap"
    assert pit_store.bars_as_of("N00", HOLIDAY)[-1]["trade_date"] < HOLIDAY


def test_a_delisted_name_keeps_the_days_it_traded(pipeline):
    _run_days(SESSIONS[105:118], bootstrap_first=1)

    bars = pit_store.bars_as_of(DELISTED, SESSIONS[117])
    assert bars, "the delisted name lost its history"
    assert max(b["trade_date"] for b in bars) < DELIST_DAY
    # And it is still in the universe record for the dates it was in it.
    members = {m["ticker"] for m in pit_store.universe_as_of(SESSIONS[106])}
    assert DELISTED in members


def test_running_the_whole_day_twice_changes_nothing(pipeline):
    day = SESSIONS[121]
    _run_days(SESSIONS[118:122], bootstrap_first=1)

    before_bars = pit_store.bars_as_of("N00", day)
    before_actions = pit_store.corporate_actions_as_of(SPLIT_NAME, day)

    daily_job.run_all(as_of=day)

    assert pit_store.bars_as_of("N00", day) == before_bars
    assert pit_store.corporate_actions_as_of(SPLIT_NAME, day) == before_actions
    assert pit_store.revisions("N00") == []


def test_the_scan_only_ever_sees_the_record_of_its_own_day(pipeline):
    _run_days(SESSIONS[115:125], bootstrap_first=1)

    early = scanner.record_scan(as_of=SESSIONS[118])
    late = scanner.record_scan(as_of=SESSIONS[123])

    assert pit_store.paper_orders_as_of(SESSIONS[117]) == []
    filed_early = {o["ticker"] for o in
                   pit_store.paper_orders_as_of(SESSIONS[118],
                                                accepted_only=True)}
    assert filed_early == {c["ticker"] for c in early["candidates"]}
    # The later scan's orders are invisible to the earlier date.
    assert not any(o["as_of_date"] == SESSIONS[123] for o in
                   pit_store.paper_orders_as_of(SESSIONS[118]))
    assert late["candidates"] or late["rejected"]


def test_every_order_is_for_a_session_that_had_not_happened(pipeline):
    _run_days(SESSIONS[115:125], bootstrap_first=1)
    scanner.record_scan(as_of=SESSIONS[123])

    for order in pit_store.paper_orders_as_of(SESSIONS[124],
                                              accepted_only=True):
        assert order["intended_session"] > order["as_of_date"]


def test_the_universe_fills_and_then_holds(pipeline):
    _run_days(SESSIONS[100:112], bootstrap_first=2)

    eligible = daily_job.eligible_tickers(SESSIONS[111])
    assert len(eligible) >= len(NAMES) - 2, (
        f"only {len(eligible)} of {len(NAMES)} names became eligible")
    # And the ask does not run away once it has.
    asked = daily_job.nightly_tickers(SESSIONS[111], NAMES)
    assert set(eligible) <= set(asked)
