"""Measuring the drift coefficient before there is forward history to measure.

The scanner declares `DRIFT_BPS_PER_SUE` and reports it uncalibrated on every
scan. Every net edge is proportional to it, so until it is measured the whole
stack is a study of an assumption. The recorder will answer it eventually and
cannot answer it now.

Replay answers it from history instead, and the honest way to do that is not an
escape hatch in the store. `bars_as_of` filters on `recorded_at` for a reason,
and a flag that turns that off is exactly the kind of thing that gets left on.
So replay builds its own store, in which each bar is stamped at the session it
describes -- an explicit claim that a day's prices were known that evening,
which is true of prices and is the one thing being assumed.

Two biases follow from it and neither is fixable here, so both are stated on
every result:

  Survivorship. The vendor only sells history for names that still exist, so a
  replay universe is the set of companies that made it. This inflates results
  and the direction is not in doubt.

  No consensus. What analysts expected on a past Tuesday is unrecoverable, so
  the analyst variant of the surprise cannot be replayed at all -- only the
  time-series one, computed from filings whose dates are on the record.
"""
import pytest

from research import pit_store, replay


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("NEMO_PIT_DB", str(tmp_path / "pit.db"))
    pit_store.init_schema()
    return pit_store


def _weekdays(n, start="2026-01-05"):
    from datetime import date, timedelta
    out, d = [], date.fromisoformat(start)
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += timedelta(days=1)
    return out


DAYS = _weekdays(120)


def _frame(tickers, days, price=100.0):
    import pandas as pd
    data = {}
    for t in tickers:
        data[(t, "Open")] = [price] * len(days)
        data[(t, "High")] = [price * 1.01] * len(days)
        data[(t, "Low")] = [price * 0.99] * len(days)
        data[(t, "Close")] = [price] * len(days)
        data[(t, "Volume")] = [5e6] * len(days)
        data[(t, "Stock Splits")] = [0.0] * len(days)
        data[(t, "Dividends")] = [0.0] * len(days)
    f = pd.DataFrame(data, index=pd.to_datetime(days))
    f.columns = pd.MultiIndex.from_tuples(f.columns)
    return f


# --- the store replay builds ------------------------------------------------

def test_a_replay_bar_is_known_on_the_evening_of_its_own_session(store,
                                                                 monkeypatch):
    """The whole difference from a bootstrap, which stamps everything today and
    is therefore invisible to every past date -- correctly, and uselessly for
    this purpose."""
    import yfinance
    monkeypatch.setattr(yfinance, "download",
                        lambda *a, **k: _frame(k["tickers"].split(), DAYS[:30]))

    replay.build_store(["AAA"], start=DAYS[0], end=DAYS[29])

    mid = DAYS[10]
    seen = pit_store.bars_as_of("AAA", mid)
    assert seen, "a replay store is still invisible to its own dates"
    assert max(b["trade_date"] for b in seen) == mid
    assert all(b["trade_date"] <= mid for b in seen)


def test_a_replay_store_still_refuses_the_future(store, monkeypatch):
    """Stamping at the session is a claim about prices, not permission to see
    tomorrow's."""
    import yfinance
    monkeypatch.setattr(yfinance, "download",
                        lambda *a, **k: _frame(k["tickers"].split(), DAYS[:30]))

    replay.build_store(["AAA"], start=DAYS[0], end=DAYS[29])

    dates = [b["trade_date"] for b in pit_store.bars_as_of("AAA", DAYS[5])]
    assert dates == DAYS[:6]


def test_the_index_is_built_too(store, monkeypatch):
    """The regime and the cost model both read it."""
    import yfinance
    monkeypatch.setattr(yfinance, "download",
                        lambda *a, **k: _frame(k["tickers"].split(), DAYS[:30]))

    replay.build_store(["AAA"], start=DAYS[0], end=DAYS[29])

    from research import spread
    assert pit_store.bars_as_of(spread.REFERENCE_TICKER, DAYS[29])


# --- prints, from filings rather than from a vendor calendar ----------------

def test_a_filing_date_becomes_the_print_the_scanner_narrows_on(store):
    """The consensus recorder cannot be replayed -- what the street expected
    on a past Tuesday is gone. A filing's own date is on the record, though,
    and it is when the reported figure became known."""
    replay.record_prints([
        {"ticker": "AAA", "fiscal_period": "2026Q1", "known_at": DAYS[10],
         "eps": 1.25},
    ])

    snap = pit_store.consensus_as_of("AAA", "2026Q1", as_of=DAYS[11])
    assert snap["eps_actual"] == 1.25
    assert pit_store.consensus_as_of("AAA", "2026Q1", as_of=DAYS[9]) is None


def test_a_print_carries_no_estimate_because_there_was_none(store):
    """Filling in an estimate would manufacture the analyst surprise this
    cannot compute."""
    replay.record_prints([
        {"ticker": "AAA", "fiscal_period": "2026Q1", "known_at": DAYS[10],
         "eps": 1.25}])

    snap = pit_store.consensus_as_of("AAA", "2026Q1", as_of=DAYS[11])
    assert snap["eps_estimate"] is None


# --- the result, and what has to be said about it ---------------------------

def test_every_result_carries_its_biases(store):
    out = replay.summarise([], caveats_only=True)
    text = " ".join(out["caveats"]).lower()
    assert "survivor" in text
    assert "consensus" in text or "analyst" in text


def test_a_replay_result_is_never_filed_as_a_real_decision(store, monkeypatch):
    """A replay order and a paper order must not share a table. One is a
    decision that was made and the other is a decision that was imagined, and
    a scoring run that mixed them would report the imagined ones as results."""
    import yfinance
    monkeypatch.setattr(yfinance, "download",
                        lambda *a, **k: _frame(k["tickers"].split(), DAYS[:60]))
    replay.build_store(["AAA"], start=DAYS[0], end=DAYS[59])
    replay.record_prints([
        {"ticker": "AAA", "fiscal_period": "2026Q1", "known_at": DAYS[20],
         "eps": 1.25}])

    monkeypatch.setattr(replay, "_signal_for", lambda t, a: {
        "ticker": t, "success": True, "sue": 2.0, "known_at": DAYS[20],
        "fiscal_period": "2026Q1", "sigma_quarters": 8,
        "sigma_periods": ["2026Q1"], "basis_changes": []})

    replay.run([DAYS[21]], horizon_days=5)

    assert pit_store.paper_orders_as_of(DAYS[59]) == [], (
        "a replayed decision was filed in the live paper book")


def test_the_universe_refresh_is_scoped_to_the_replay_names(store, monkeypatch):
    """run() screened every SEC registrant on every decision date -- ten
    thousand names times fifty dates, and a network call for the list each
    time. A replay's universe is the set it built a store for."""
    import yfinance
    monkeypatch.setattr(yfinance, "download",
                        lambda *a, **k: _frame(k["tickers"].split(), DAYS[:80]))

    def boom():
        raise AssertionError("run() went to EDGAR for the registrant list")

    monkeypatch.setattr(replay.daily_job, "_fetch_sec_tickers", boom)
    replay.build_store(["AAA", "BBB"], start=DAYS[0], end=DAYS[79])
    monkeypatch.setattr(replay, "_signal_for", lambda t, a: {
        "ticker": t, "success": False, "error": "no filings", "sue": None})

    replay.run([DAYS[70]], horizon_days=5, tickers=["AAA", "BBB"])

    members = {m["ticker"] for m in pit_store.universe_as_of(DAYS[70])}
    assert members == {"AAA", "BBB"}


def test_a_replay_screens_each_date_on_its_own_history(store, monkeypatch):
    """A name is eligible on a date because of what it had done by then, not
    because of what it did later."""
    import yfinance
    monkeypatch.setattr(yfinance, "download",
                        lambda *a, **k: _frame(k["tickers"].split(), DAYS[:80]))
    monkeypatch.setattr(replay.daily_job, "_fetch_sec_tickers",
                        lambda: [{"ticker": "AAA", "cik": "1", "name": "A"}])
    replay.build_store(["AAA"], start=DAYS[0], end=DAYS[79])
    monkeypatch.setattr(replay, "_signal_for", lambda t, a: {
        "ticker": t, "success": False, "error": "no filings", "sue": None})

    replay.run([DAYS[10], DAYS[70]], horizon_days=5, tickers=["AAA"])

    early = {m["ticker"]: m for m in pit_store.universe_as_of(DAYS[10])}["AAA"]
    late = {m["ticker"]: m for m in pit_store.universe_as_of(DAYS[70])}["AAA"]
    assert early["eligible"] is False and "histor" in early["exclusion_reason"]
    assert late["eligible"] is True


def test_replay_and_live_scoring_agree_on_the_same_trade(store, monkeypatch):
    """The two paths differ only in where the orders come from -- the live
    table or a replay run. The arithmetic between them was copied, which is how
    a fix to one silently stops applying to the other."""
    import yfinance
    from research import scoring

    monkeypatch.setattr(yfinance, "download",
                        lambda *a, **k: _frame(k["tickers"].split(), DAYS[:40]))
    replay.build_store(["AAA"], start=DAYS[0], end=DAYS[39])

    # Make the path go somewhere, so agreement is not agreement on zero.
    pit_store.record_bars("AAA", [
        {"trade_date": DAYS[25], "open": 130.0, "high": 131.0, "low": 129.0,
         "close": 130.0, "volume": 5e6}], recorded_at=f"{DAYS[25]}T21:00:00Z")

    order = {"ticker": "AAA", "side": "long", "sue": 2.0,
             "fiscal_period": "2026Q1", "expected_edge_bps": 30.0,
             "cost_bps": 7.0, "net_edge_bps": 23.0, "target_dollars": 5000.0,
             "participation": 0.001, "spread": 0.0005,
             "spread_resolved": True, "rank": 1,
             "intended_session": DAYS[20]}

    pit_store.record_paper_orders(DAYS[19], [order], regime="calm",
                                 gross_target=100_000.0,
                                 recorded_at=f"{DAYS[19]}T21:00:00Z")

    live = scoring.score_orders(as_of=DAYS[39], horizon_days=5)["scored"]
    replayed = replay._score([{**order, "as_of_date": DAYS[19]}],
                             horizon_days=5)["scored"]

    assert len(live) == 1 and len(replayed) == 1
    for field in ("entry_price", "exit_price", "gross_bps", "net_bps"):
        assert live[0][field] == pytest.approx(replayed[0][field]), field


# --- signals, precomputed once ----------------------------------------------
#
# A replay over 180 decision dates calling sue_ts per name per date is
# thousands of EDGAR requests for a series that does not change. sue_ts_history
# returns every quarter in one pass, and it agrees with a point-in-time sue_ts
# on each quarter's own filing date to fifteen digits -- verified live on MSFT.

def test_a_precomputed_signal_is_only_visible_after_it_was_filed(store):
    replay.load_signals({"AAA": [
        {"fiscal_period": "2026Q1", "known_at": DAYS[10], "sue": 2.0,
         "sigma_quarters": 8, "sigma_periods": ["2026Q1"], "basis_changes": []},
        {"fiscal_period": "2026Q2", "known_at": DAYS[40], "sue": 3.0,
         "sigma_quarters": 8, "sigma_periods": ["2026Q2"], "basis_changes": []},
    ]})

    assert replay._signal_for("AAA", DAYS[5])["success"] is False
    assert replay._signal_for("AAA", DAYS[20])["sue"] == 2.0
    assert replay._signal_for("AAA", DAYS[45])["sue"] == 3.0


def test_the_latest_filed_quarter_wins_not_the_largest(store):
    """A lookup that scanned for the biggest surprise instead of the most
    recent one would pick its trades with hindsight."""
    replay.load_signals({"AAA": [
        {"fiscal_period": "2026Q1", "known_at": DAYS[10], "sue": 9.0,
         "sigma_quarters": 8, "sigma_periods": [], "basis_changes": []},
        {"fiscal_period": "2026Q2", "known_at": DAYS[20], "sue": 1.5,
         "sigma_quarters": 8, "sigma_periods": [], "basis_changes": []},
    ]})

    assert replay._signal_for("AAA", DAYS[30])["sue"] == 1.5


def test_a_name_with_no_signals_refuses_rather_than_returning_none(store):
    replay.load_signals({})
    out = replay._signal_for("AAA", DAYS[30])
    assert out["success"] is False
    assert out["error"]


def test_a_replay_says_which_orders_it_could_not_score(store, monkeypatch):
    """The live scorer files unfilled orders with a reason; replay skipped them
    with a bare `continue`. A sample that quietly drops the trades it could not
    price is a sample selected on something, and nothing says on what."""
    import yfinance
    monkeypatch.setattr(yfinance, "download",
                        lambda *a, **k: _frame(k["tickers"].split(), DAYS[:40]))
    replay.build_store(["AAA"], start=DAYS[0], end=DAYS[39])

    orders = [
        # Intended for a session the name never traded.
        {"ticker": "AAA", "side": "long", "sue": 2.0, "cost_bps": 5.0,
         "as_of_date": DAYS[10], "intended_session": "2026-07-04"},
        # Intended so late the horizon cannot complete.
        {"ticker": "AAA", "side": "long", "sue": 2.0, "cost_bps": 5.0,
         "as_of_date": DAYS[37], "intended_session": DAYS[38]},
    ]

    out = replay._score(orders, horizon_days=5)

    assert out["scored"] == []
    assert len(out["skipped"]) == 2
    reasons = " ".join(r["reason"] for r in out["skipped"]).lower()
    assert "did not trade" in reasons
    assert "horizon" in reasons or "sessions" in reasons


def test_a_replay_counts_one_print_once(store, monkeypatch):
    """Replayed decisions never enter the filed book, so the scanner cannot
    read the already-acted set from the store the way it does live. Without
    replay keeping its own, a single earnings event becomes one order per
    decision date for as long as the signal stays fresh -- a sample of
    overlapping trades presented as independent ones."""
    import yfinance
    monkeypatch.setattr(yfinance, "download",
                        lambda *a, **k: _frame(k["tickers"].split(), DAYS[:110]))
    monkeypatch.setattr(replay.daily_job, "_fetch_sec_tickers",
                        lambda: [{"ticker": "AAA", "cik": "1", "name": "A"}])
    replay.build_store(["AAA"], start=DAYS[0], end=DAYS[109])
    replay.record_prints([{"ticker": "AAA", "fiscal_period": "2026Q1",
                           "known_at": DAYS[70], "eps": 1.0}])
    replay.load_signals({"AAA": [
        {"fiscal_period": "2026Q1", "known_at": DAYS[70], "sue": 3.0,
         "sigma_quarters": 8, "sigma_periods": ["2026Q1"],
         "basis_changes": []}]})
    monkeypatch.setattr(replay.scanner, "_cost_for", lambda t, a, d: {
        "cost": 0.0002, "cost_floor": 0.00002, "reason": None,
        "spread": 0.0001, "resolved": True, "resolution": "measured"})

    out = replay.run([DAYS[71], DAYS[72], DAYS[75], DAYS[80]],
                     horizon_days=5, tickers=["AAA"])

    assert out["orders"] == 1, (
        f"one print produced {out['orders']} orders across four dates")


def test_the_entry_timing_caveat_is_stated(store):
    """Measured on EDGAR across 60 filings from 20 large caps: the 10-Q lands a
    median of 8 days after the first 8-K following period end, mean 12.1, range
    0 to 45. An XBRL-derived surprise cannot exist before the 10-Q does, so the
    replay's timing is right for THIS strategy -- but the published drift is
    measured from the announcement, and most of it happens in the first days.
    A null result here is not a null result about PEAD, and the difference has
    to be on the page rather than in someone's head."""
    text = " ".join(replay.CAVEATS).lower()
    assert "10-q" in text or "filing" in text
    assert "announce" in text
    assert "8" in text


def test_a_replayed_trade_carries_the_hour_it_was_announced(store, monkeypatch):
    """The live scorer splits on it; the replay is where the split matters
    most, because that is where the sample is large enough to see whether
    entering a day later costs anything."""
    import yfinance
    monkeypatch.setattr(yfinance, "download",
                        lambda *a, **k: _frame(k["tickers"].split(), DAYS[:40]))
    replay.build_store(["AAA"], start=DAYS[0], end=DAYS[39])
    pit_store.record_announcement("AAA", "2026Q1", DAYS[18], timing="bmo",
                                  recorded_at=f"{DAYS[18]}T21:00:00Z")

    out = replay._score([{"ticker": "AAA", "side": "long", "sue": 2.0,
                          "cost_bps": 5.0, "fiscal_period": "2026Q1",
                          "target_dollars": 5000.0, "as_of_date": DAYS[19],
                          "intended_session": DAYS[20]}], horizon_days=5)

    assert out["scored"][0]["timing"] == "bmo"
    assert out["by_timing"]["bmo"]["sample"] == 1


# --- the replay has to declare how many variants were tried -----------------
#
# scoring._summarise takes a comparison count and moves the significance bar
# out with it. The replay never passed one, so a 340-trade run came back
# "calibrated: True" against a bar of 2.00 -- on the fourth variant tried
# against the same names. The guard existed and the caller that most needed it
# was not using it.

def test_the_replay_passes_its_comparison_count_through(store, monkeypatch):
    import yfinance
    monkeypatch.setattr(yfinance, "download",
                        lambda *a, **k: _frame(k["tickers"].split(), DAYS[:40]))
    replay.build_store(["AAA"], start=DAYS[0], end=DAYS[39])

    orders = [{"ticker": "AAA", "side": "long", "sue": 2.0, "cost_bps": 0.0,
               "fiscal_period": "2026Q1", "target_dollars": 5000.0,
               "as_of_date": DAYS[19], "intended_session": DAYS[20]}]

    alone = replay._score(orders, horizon_days=5)
    searched = replay._score(orders, horizon_days=5, comparisons=4)

    assert alone["comparisons"] == 1
    assert searched["comparisons"] == 4
    assert searched["t_threshold"] > alone["t_threshold"]


def test_run_carries_the_count_to_the_summary(store, monkeypatch):
    import yfinance
    monkeypatch.setattr(yfinance, "download",
                        lambda *a, **k: _frame(k["tickers"].split(), DAYS[:80]))
    monkeypatch.setattr(replay.daily_job, "_fetch_sec_tickers",
                        lambda: [{"ticker": "AAA", "cik": "1", "name": "A"}])
    replay.build_store(["AAA"], start=DAYS[0], end=DAYS[79])
    monkeypatch.setattr(replay, "_signal_for", lambda t, a: {
        "ticker": t, "success": False, "error": "no filings", "sue": None})

    out = replay.run([DAYS[70]], horizon_days=5, tickers=["AAA"],
                     comparisons=4)
    assert out["comparisons"] == 4


def test_the_replay_cli_takes_a_comparison_count(store, monkeypatch):
    """`python -m research.replay` is how a replay is actually run, and its
    argparse had no way to say how many variants had been tried -- so every
    command-line replay scored at one comparison however long the search."""
    seen = {}

    def fake(dates, **kwargs):
        seen.update(kwargs)
        return {}

    monkeypatch.setattr(replay, "run", fake)
    assert replay.main(["--dates", DAYS[0], "--comparisons", "4"]) == 0
    assert seen["comparisons"] == 4


def test_a_replayed_timing_subgroup_carries_the_split(store, monkeypatch):
    """Same correction as the live scorer: two subgroups is two tests."""
    import yfinance
    monkeypatch.setattr(yfinance, "download",
                        lambda *a, **k: _frame(k["tickers"].split(), DAYS[:40]))
    replay.build_store(["AAA", "BBB"], start=DAYS[0], end=DAYS[39])
    pit_store.record_announcement("AAA", "2026Q1", DAYS[18], timing="bmo",
                                  recorded_at=f"{DAYS[18]}T21:00:00Z")
    pit_store.record_announcement("BBB", "2026Q1", DAYS[18], timing="amc",
                                  recorded_at=f"{DAYS[18]}T21:00:00Z")

    orders = [{"ticker": t, "side": "long", "sue": 2.0, "cost_bps": 5.0,
               "fiscal_period": "2026Q1", "target_dollars": 5000.0,
               "as_of_date": DAYS[19], "intended_session": DAYS[20]}
              for t in ("AAA", "BBB")]

    out = replay._score(orders, horizon_days=5)
    assert out["by_timing"]["bmo"]["comparisons"] == 2
    assert out["by_timing"]["amc"]["comparisons"] == 2


def test_the_replay_summary_can_be_told_the_count_too(store):
    """`summarise` called `_summarise` bare, so the one helper a reader is
    most likely to reach for scored at a bar of 2.00 whatever the search."""
    scored = [{"ticker": f"T{i}", "sue": 2.0, "net_bps": 50.0,
               "gross_bps": 50.0, "cost_bps": 0.0} for i in range(40)]
    assert replay.summarise(scored, comparisons=4)["comparisons"] == 4


def test_a_replayed_roll_stops_at_the_next_open(store):
    """Same bound as the live scorer, and the same reason: rolling until the
    name next prints buys days after the news, and the replay is where that
    error is multiplied by every holiday in the sample."""
    from research import spread

    def bars(ticker, days, price):
        for day in days:
            pit_store.record_bars(ticker, [
                {"trade_date": day, "open": price, "high": price * 1.01,
                 "low": price * 0.99, "close": price, "volume": 5e6}],
                recorded_at=f"{day}T21:00:00Z")

    # The exchange was shut on DAYS[20] and open on every session around it.
    bars(spread.REFERENCE_TICKER, [d for d in DAYS[:40] if d != DAYS[20]], 50.0)
    # The name's next print is five sessions after the holiday.
    bars("AAA", DAYS[:20], 100.0)
    bars("AAA", DAYS[25:40], 60.0)

    out = replay._score([{"ticker": "AAA", "side": "long", "sue": 2.0,
                          "cost_bps": 0.0, "fiscal_period": "2026Q1",
                          "target_dollars": 5000.0, "as_of_date": DAYS[19],
                          "intended_session": DAYS[20]}], horizon_days=5)

    assert out["scored"] == []
    assert "did not trade" in out["skipped"][0]["reason"]


def test_a_replayed_horizon_is_counted_in_sessions_too(store):
    """The same fix as the live scorer, through the same helper. Replay is
    where a lengthened hold does the most damage, because it is the sample the
    coefficient is measured from."""
    from research import spread

    def bars(ticker, days, prices):
        for day, price in zip(days, prices):
            pit_store.record_bars(ticker, [
                {"trade_date": day, "open": price, "high": price * 1.01,
                 "low": price * 0.99, "close": price, "volume": 5e6}],
                recorded_at=f"{day}T21:00:00Z")

    bars(spread.REFERENCE_TICKER, DAYS[:40], [50.0] * 40)
    # AAA misses one session inside the hold.
    held = [DAYS[20], DAYS[21], DAYS[23], DAYS[24], DAYS[25], DAYS[26]]
    bars("AAA", held, [100.0, 100.0, 100.0, 100.0, 110.0, 130.0])

    out = replay._score([{"ticker": "AAA", "side": "long", "sue": 2.0,
                          "cost_bps": 0.0, "fiscal_period": "2026Q1",
                          "target_dollars": 5000.0, "as_of_date": DAYS[19],
                          "intended_session": DAYS[20]}], horizon_days=5)

    assert out["scored"], out["skipped"]
    assert out["scored"][0]["exit_session"] == DAYS[25]
    assert out["scored"][0]["gross_bps"] == pytest.approx(1000.0)


# --- a decision cannot be gated on evidence that did not exist yet ----------
#
# Every store reader filters on recorded_at and the anti-lookahead sweep covers
# them. The leak was outside the store: build_signals asks sue_ts_history once,
# as of today, and every row it returns carries the basis changes of the WHOLE
# series -- so `_basis_change_in_window` rejected a 2020 signal for a split
# whose filing date is 2024. It can only cause rejections, so it does not
# inflate a result; it selects the sample with hindsight, which is the same
# defect one step removed.

def _history(rows):
    return lambda ticker, as_of=None: {
        "ticker": ticker, "success": True, "error": None, "signals": rows}


def test_a_precomputed_signal_carries_only_the_changes_it_could_know(
        store, monkeypatch):
    monkeypatch.setattr(replay.sue, "sue_ts_history", _history([
        {"fiscal_period": "2020Q1", "known_at": "2020-05-01", "sue": 2.0,
         "sigma_quarters": 8, "sigma_periods": ["2020Q1"],
         "basis_changes": [
             {"between": ["2019-11-01", "2019-12-01"], "ratio": 2.0},
             {"between": ["2023-11-01", "2024-02-01"], "ratio": 4.0}]}]))

    replay.build_signals(["AAA"])
    row = replay._signal_for("AAA", "2020-06-01")

    assert [c["ratio"] for c in row["basis_changes"]] == [2.0], (
        "a 2020 decision was handed a basis change filed in 2024")


def test_the_scanner_no_longer_rejects_that_decision_for_it(store,
                                                            monkeypatch):
    """What the leak actually did, end to end."""
    from research import scanner

    monkeypatch.setattr(replay.sue, "sue_ts_history", _history([
        {"fiscal_period": "2020Q1", "known_at": "2020-05-01", "sue": 2.0,
         "sigma_quarters": 8, "sigma_periods": ["2020Q1"],
         "basis_changes": [
             {"between": ["2023-11-01", "2024-02-01"], "ratio": 4.0}]}]))

    replay.build_signals(["AAA"])
    signal = replay._signal_for("AAA", "2020-05-04")

    assert scanner._basis_change_in_window(signal) is None


def test_a_fresh_store_is_created_by_the_replay_itself(tmp_path, monkeypatch):
    """A replay is the most likely of all of these to be the first command run
    against a new volume."""
    monkeypatch.setenv("NEMO_PIT_DB", str(tmp_path / "fresh.db"))
    replay.load_signals({})
    assert replay.main(["--dates", DAYS[10], "--tickers", "AAA"]) == 0


def test_a_replay_built_store_says_its_bars_were_backfilled(tmp_path, monkeypatch):
    """`build_store` pulls a whole history at once and stamps it day by day.

    Every row it writes is a backfill by construction -- the session it
    describes is months past and the vendor has already dropped whatever
    delisted since. `daily_bar.source` exists to say so, and taking its
    default would have every replayed row claim it was recorded on the
    evening it describes, which is the one thing that column is for.
    """
    monkeypatch.setenv("NEMO_PIT_DB", str(tmp_path / "pit.db"))
    import importlib
    from research import pit_store as store
    importlib.reload(store)
    store.init_schema()

    rows = [{"trade_date": "2026-03-02", "open": 10.0, "high": 11.0,
             "low": 9.5, "close": 10.5, "volume": 1_000_000.0}]
    monkeypatch.setattr(replay.daily_job, "_fetch_bars",
                        lambda *a, **k: {"AAA": rows})

    replay.build_store(["AAA"], start="2026-03-01", end="2026-03-03")

    with store.connect() as conn:
        got = conn.execute(
            "SELECT source FROM daily_bar WHERE ticker='AAA'").fetchone()
    assert got is not None, "build_store wrote nothing"
    assert got[0] == "backfilled", (
        f"a replayed session claims source={got[0]!r}; it was not recorded "
        f"on the evening it describes")


def test_a_replayed_signal_says_it_is_the_time_series_variant():
    """The scanner records `variant` on every order and the scorer quotes a
    coefficient only when a sample's variant is known. Replay's precomputed
    table is the time-series surprise by construction, and its lookup returned
    rows with no `variant` at all -- so a full replay produced trades and then
    `drift_bps_per_sue: None`, with the note that the rows did not say which
    quantity `sue` was."""
    replay.load_signals({"AAA": [{"sue": 2.0, "known_at": "2026-03-02",
                                  "fiscal_period": "2026Q1",
                                  "sigma_quarters": 8, "basis_changes": []}]})

    signal = replay._signal_for("AAA", "2026-03-03")

    assert signal["success"] is True
    assert signal["variant"] == "ts"
