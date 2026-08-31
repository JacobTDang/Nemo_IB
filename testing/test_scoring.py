"""Closing the loop, and the ways a scorer flatters itself.

The scanner logs an intention: a ticker, a side, a size and the session the
order is for. It deliberately does not log a price, because that session has
not happened. Scoring is where the two meet, and it is the only test any of
this has -- everything upstream is measurement, and measurement that is never
checked against an outcome is just arithmetic.

A scorer has its own failure modes, and they all point the same way:

  Scoring an order whose horizon has not finished, as though a partial move
  were a result. It is not a small error: unfinished trades are exactly the
  ones still open because they have not gone anywhere.

  Filling an order that never filled. A name that did not trade on its
  intended session is not a position bought at the next available price, it is
  a position nobody has.

  Reading a split as a return. A 10-for-1 inside the holding period is a -90%
  print in the as-traded series the store keeps on purpose.

  Forgetting the cost that was already measured and recorded at decision time.

  Reporting the drift coefficient the scanner assumed rather than the one the
  outcomes imply, which would make the calibration a mirror.
"""
import pytest

from research import pit_store, scoring


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("NEMO_PIT_DB", str(tmp_path / "pit.db"))
    pit_store.init_schema()
    return pit_store


def _sessions(n, start="2026-03-02"):
    from datetime import date, timedelta
    out, d = [], date.fromisoformat(start)
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += timedelta(days=1)
    return out


DAYS = _sessions(40)


def _bars(store, ticker, prices, recorded_at=None, days=None):
    days = days or DAYS[:len(prices)]
    rows = [{"trade_date": d, "open": p, "high": p * 1.01, "low": p * 0.99,
             "close": p, "volume": 1_000_000}
            for d, p in zip(days, prices)]
    store.record_bars(ticker, rows,
                      recorded_at=recorded_at or f"{days[-1]}T21:00:00Z")


def _calendar(store, days=None):
    """The reference instrument's own sessions, one evening at a time.

    It is the only calendar the store has. What "the exchange was open" means
    and what a horizon counted in sessions counts are both read off it, so a
    scoring test that scores anything has to say which days the market held.
    """
    from research import spread
    for day in (days or DAYS):
        store.record_bars(spread.REFERENCE_TICKER,
                          [{"trade_date": day, "open": 50.0, "high": 50.5,
                            "low": 49.5, "close": 50.0, "volume": 1_000_000}],
                          recorded_at=f"{day}T21:00:00Z")


def _order(store, ticker="AAA", side="long", sue=3.0, cost_bps=10.0,
           decided=None, session=None, dollars=5000.0,
           borrow_bps=None):  # noqa: D401
    decided = decided or DAYS[0]
    session = session or DAYS[1]
    store.record_paper_orders(
        decided,
        [{"ticker": ticker, "side": side, "sue": sue, "fiscal_period": "2026Q1",
          "variant": "ts", "strength": abs(sue),
          "expected_edge_bps": abs(sue) * 15.0, "cost_bps": cost_bps,
          "borrow_bps": borrow_bps,
          "net_edge_bps": abs(sue) * 15.0 - cost_bps,
          "target_dollars": dollars, "participation": 0.001,
          "spread": 0.0005, "spread_resolved": True, "rank": 1,
          "intended_session": session}],
        regime="calm", gross_target=100_000.0,
        recorded_at=f"{decided}T21:00:00Z")


# --- do not score what has not finished -------------------------------------

def test_an_unfinished_horizon_is_not_scored(store):
    """Not scored as zero, not scored as partial. Unfinished trades are the
    ones still open because they have not moved, so counting them early tilts
    every average toward nothing having happened."""
    _order(store)
    _calendar(store)
    _bars(store, "AAA", [100.0] * 5)

    out = scoring.score_orders(as_of=DAYS[4], horizon_days=20)

    assert out["scored"] == []
    assert out["pending"] and out["pending"][0]["ticker"] == "AAA"


def test_a_finished_horizon_is_scored(store):
    _order(store)
    _calendar(store)
    prices = [100.0] * 2 + [110.0] * 10
    _bars(store, "AAA", prices)

    out = scoring.score_orders(as_of=DAYS[11], horizon_days=5)

    assert len(out["scored"]) == 1
    assert out["pending"] == []


# --- an order that never filled ---------------------------------------------

def test_a_name_that_did_not_trade_that_session_never_filled(store):
    """No bar on the intended session means no fill. Sliding to the next
    available price is how a backtest buys the day after bad news."""
    _order(store, session=DAYS[1])
    _calendar(store, DAYS[:21])
    # AAA trades on day 0 and then not again until day 5.
    _bars(store, "AAA", [100.0], days=[DAYS[0]],
          recorded_at=f"{DAYS[0]}T21:00:00Z")
    _bars(store, "AAA", [80.0] * 8, days=DAYS[5:13],
          recorded_at=f"{DAYS[12]}T21:00:00Z")

    out = scoring.score_orders(as_of=DAYS[20], horizon_days=5)

    assert out["scored"] == []
    assert out["unfilled"], "an order with no session was quietly given a price"
    assert "did not trade" in out["unfilled"][0]["reason"].lower()


# --- the arithmetic ---------------------------------------------------------

def test_the_return_runs_from_the_intended_session(store):
    """Entry is the open of the session the order was for -- not the close of
    the day the decision was made, which is the price that was already known."""
    _order(store, session=DAYS[1], cost_bps=0.0)
    _calendar(store)
    _bars(store, "AAA", [50.0, 100.0, 100.0, 100.0, 100.0, 110.0, 110.0])

    out = scoring.score_orders(as_of=DAYS[10], horizon_days=4)
    row = out["scored"][0]

    assert row["entry_price"] == 100.0, (
        "entry was taken from a session other than the intended one")
    assert row["gross_bps"] == pytest.approx(1000.0)


def test_the_cost_recorded_at_decision_time_is_subtracted(store):
    _order(store, session=DAYS[1], cost_bps=40.0)
    _calendar(store)
    _bars(store, "AAA", [100.0, 100.0, 100.0, 100.0, 100.0, 110.0, 110.0])

    row = scoring.score_orders(as_of=DAYS[10], horizon_days=4)["scored"][0]

    assert row["gross_bps"] == pytest.approx(1000.0)
    assert row["net_bps"] == pytest.approx(960.0)


def test_a_short_earns_when_the_price_falls(store):
    _order(store, side="short", sue=-3.0, session=DAYS[1], cost_bps=0.0,
           borrow_bps=0.0)
    _calendar(store)
    _bars(store, "AAA", [100.0, 100.0, 100.0, 100.0, 100.0, 90.0, 90.0])

    row = scoring.score_orders(as_of=DAYS[10], horizon_days=4)["scored"][0]
    assert row["net_bps"] == pytest.approx(1000.0)


def test_the_borrow_charged_at_decision_time_is_subtracted_too(store):
    """A realised return that never paid its borrow is not a realised return,
    and it is the number the drift coefficient is calibrated against."""
    _order(store, side="short", sue=-3.0, session=DAYS[1], cost_bps=40.0,
           borrow_bps=24.0)
    _calendar(store)
    _bars(store, "AAA", [100.0, 100.0, 100.0, 100.0, 100.0, 90.0, 90.0])

    row = scoring.score_orders(as_of=DAYS[10], horizon_days=4)["scored"][0]

    assert row["gross_bps"] == pytest.approx(1000.0)
    assert row["borrow_bps"] == pytest.approx(24.0)
    assert row["net_bps"] == pytest.approx(1000.0 - 40.0 - 24.0)
    assert row["borrow_priced"] is True


def test_a_row_filed_before_borrow_was_priced_says_so(store):
    """Rows written before the cost model had a borrow term genuinely never
    paid one. The record is append-only, so the honest thing is to count them
    rather than to invent a charge or to drop them from the sample."""
    _order(store, side="short", sue=-3.0, session=DAYS[1], cost_bps=0.0)
    _calendar(store)
    _bars(store, "AAA", [100.0, 100.0, 100.0, 100.0, 100.0, 90.0, 90.0])

    out = scoring.score_orders(as_of=DAYS[10], horizon_days=4)

    assert out["scored"][0]["borrow_priced"] is False
    assert out["scored"][0]["net_bps"] == pytest.approx(1000.0)
    assert out["shorts_without_borrow"] == 1


def test_a_long_is_never_counted_as_an_unpriced_short(store):
    _order(store, session=DAYS[1], cost_bps=0.0)
    _calendar(store)
    _bars(store, "AAA", [100.0, 100.0, 100.0, 100.0, 100.0, 110.0, 110.0])

    out = scoring.score_orders(as_of=DAYS[10], horizon_days=4)

    assert out["shorts_without_borrow"] == 0
    assert out["scored"][0]["borrow_priced"] is True


def test_a_split_inside_the_horizon_is_not_a_return(store):
    """The store keeps as-traded prices on purpose, so a 10-for-1 is a genuine
    -90% print. Scoring the raw series would book it as the worst trade ever
    made."""
    _order(store, session=DAYS[1], cost_bps=0.0)
    _calendar(store)
    _bars(store, "AAA", [100.0, 100.0, 100.0, 10.0, 10.0, 11.0, 11.0])
    store.record_corporate_action("AAA", DAYS[3], "split", 10.0,
                                  recorded_at=f"{DAYS[3]}T21:00:00Z")

    row = scoring.score_orders(as_of=DAYS[10], horizon_days=4)["scored"][0]
    assert row["net_bps"] == pytest.approx(1000.0), (
        f"the split was scored as a return: {row['net_bps']:.0f}bp")


# --- anti-lookahead, one more time ------------------------------------------

def test_scoring_cannot_see_a_bar_recorded_after_its_date(store):
    _order(store, session=DAYS[1])
    _calendar(store, DAYS[:11])
    _bars(store, "AAA", [100.0] * 10, recorded_at=f"{DAYS[30]}T21:00:00Z")

    out = scoring.score_orders(as_of=DAYS[10], horizon_days=4)
    assert out["scored"] == []


# --- what the whole thing is for --------------------------------------------

def test_the_realised_drift_is_measured_not_echoed(store):
    """The calibration. If this reported the coefficient the scanner assumed,
    it would be a mirror rather than a measurement."""
    from research import scanner

    _order(store, ticker="AAA", sue=2.0, session=DAYS[1], cost_bps=0.0)
    _calendar(store)
    _bars(store, "AAA", [100.0, 100.0, 100.0, 100.0, 100.0, 104.0, 104.0])

    out = scoring.score_orders(as_of=DAYS[10], horizon_days=4)

    # 400bp realised on a SUE of 2.0 is 200bp per unit, whatever the scanner
    # assumed.
    assert out["drift_bps_per_sue"] == pytest.approx(200.0)
    assert out["drift_bps_per_sue"] != scanner.DRIFT_BPS_PER_SUE
    assert out["sample"] == 1


def test_a_calibration_from_too_few_trades_says_so(store):
    """One trade is not a coefficient, and quoting it as one is how an
    assumption gets replaced with a worse assumption."""
    _order(store, sue=2.0, session=DAYS[1], cost_bps=0.0)
    _calendar(store)
    _bars(store, "AAA", [100.0] * 5 + [104.0, 104.0])

    out = scoring.score_orders(as_of=DAYS[10], horizon_days=4)
    assert out["sample"] < scoring.MIN_CALIBRATION_SAMPLE
    assert out["calibrated"] is False
    assert "sample" in out["calibration_note"].lower()


def test_the_horizon_boundary_is_exact(store):
    """Off by one here either scores a shorter hold than it claims or indexes
    past the end of the series. A mutation from `<=` to `<` survived the rest
    of this file, which is how the boundary got its own test.
    """
    _order(store, session=DAYS[1], cost_bps=0.0)
    # Entry session plus exactly four more: one short of a five-day hold.
    _calendar(store, DAYS[:6])
    _bars(store, "AAA", [100.0] * 5, days=DAYS[1:6],
          recorded_at=f"{DAYS[5]}T21:00:00Z")

    out = scoring.score_orders(as_of=DAYS[30], horizon_days=5)
    assert out["scored"] == [], "a four-session hold was scored as five"
    assert out["pending"]

    # One more session, and it is exactly a five-day hold.
    _calendar(store, [DAYS[6]])
    _bars(store, "AAA", [120.0], days=[DAYS[6]],
          recorded_at=f"{DAYS[6]}T21:00:00Z")

    out = scoring.score_orders(as_of=DAYS[30], horizon_days=5)
    assert len(out["scored"]) == 1
    row = out["scored"][0]
    assert row["entry_session"] == DAYS[1]
    assert row["exit_session"] == DAYS[6], (
        f"a 5-session hold from {DAYS[1]} should exit on {DAYS[6]}, "
        f"got {row['exit_session']}")
    assert row["gross_bps"] == pytest.approx(2000.0)


def test_a_horizon_longer_than_the_record_stays_pending(store):
    """And does not index off the end."""
    _order(store, session=DAYS[1], cost_bps=0.0)
    _calendar(store)
    _bars(store, "AAA", [100.0] * 6, days=DAYS[1:7],
          recorded_at=f"{DAYS[6]}T21:00:00Z")

    out = scoring.score_orders(as_of=DAYS[30], horizon_days=60)
    assert out["scored"] == []
    assert out["pending"]


# --- a coefficient the sample does not support ------------------------------
#
# A live replay over 652 decision dates returned mean net +52.7bp against a
# median of -70.9bp, a 48% hit rate, and -153.8bp in the largest-surprise
# bucket. The mean is a few big winners; the typical trade lost money. Quoting
# 33.7bp per SUE off that would have replaced a declared assumption of 15 with
# a number 2.2x larger and made every net edge in the scanner follow it.
#
# Sample size was the only thing being checked, and 223 trades passed it.


def _scored(nets, sues=None):
    sues = sues or [2.0] * len(nets)
    return [{"ticker": f"T{i}", "sue": s, "net_bps": n, "gross_bps": n,
             "cost_bps": 0.0, "variant": "ts", "strength": abs(s)}
            for i, (n, s) in enumerate(zip(nets, sues))]


def test_a_mean_and_median_that_disagree_in_sign_is_not_calibrated():
    """One trade in twenty carrying the whole average is a tail, not an edge."""
    nets = [-70.0] * 40 + [3000.0] * 5
    out = scoring._summarise(_scored(nets))

    assert out["sample"] >= scoring.MIN_CALIBRATION_SAMPLE
    assert out["mean_net_bps"] > 0 and out["median_net_bps"] < 0
    assert out["calibrated"] is False
    assert "median" in out["calibration_note"].lower()


def test_a_consistent_sample_does_calibrate():
    out = scoring._summarise(_scored([20.0, 30.0, 25.0, 15.0, -5.0] * 8))
    assert out["calibrated"] is True
    assert out["median_net_bps"] > 0


def test_the_summary_reports_dispersion_not_just_a_mean():
    out = scoring._summarise(_scored([-70.0] * 40 + [3000.0] * 5))
    assert out["median_net_bps"] is not None
    assert out["t_stat"] is not None
    assert out["hit_rate"] < 0.5


def test_a_mean_that_cannot_clear_its_own_noise_is_not_calibrated():
    """Positive mean, positive median, and a t-statistic under two: the sign
    is not distinguishable from chance at this sample size."""
    nets = [500.0, -480.0, 20.0, 10.0, 5.0] * 8
    out = scoring._summarise(_scored(nets))
    assert out["t_stat"] < 2.0
    assert out["calibrated"] is False
    assert "t=" in out["calibration_note"] or "noise" in out["calibration_note"]


# --- an order resting on a closed exchange ----------------------------------
#
# The scanner names the next weekday as the intended session, which about ten
# times a year is a holiday. Those orders were scored as never filled -- a real
# order would have rested and filled on the next open. Live, the replay
# discarded five trades this way, all of them on Presidents Day, Memorial Day
# and Thanksgiving.
#
# The distinction that matters is the same one the recorder makes: if the
# reference instrument has no session either, the exchange was shut and the
# order rolls. If it traded and this name did not, the name is the problem and
# the order genuinely never filled.

def test_an_order_for_a_closed_exchange_fills_on_the_next_open(store):
    from research import spread
    _order(store, session=DAYS[1], cost_bps=0.0)
    # Neither the name nor the reference trades on DAYS[1]: the market is shut.
    days = [DAYS[0]] + DAYS[2:9]
    _bars(store, "AAA", [100.0, 100.0, 100.0, 100.0, 100.0, 110.0, 110.0, 110.0],
          days=days, recorded_at=f"{DAYS[8]}T21:00:00Z")
    _bars(store, spread.REFERENCE_TICKER, [50.0] * len(days), days=days,
          recorded_at=f"{DAYS[8]}T21:00:00Z")

    out = scoring.score_orders(as_of=DAYS[20], horizon_days=4)

    assert out["unfilled"] == [], out["unfilled"]
    row = out["scored"][0]
    assert row["entry_session"] == DAYS[2], (
        "the order did not roll to the next open")
    assert row["gross_bps"] == pytest.approx(1000.0)


def test_a_name_that_halted_while_the_market_traded_never_filled(store):
    """The other side of the same test. The reference traded that day, so the
    exchange was open and this name simply was not available."""
    from research import spread
    _order(store, session=DAYS[1], cost_bps=0.0)
    _bars(store, "AAA", [100.0], days=[DAYS[0]],
          recorded_at=f"{DAYS[0]}T21:00:00Z")
    _bars(store, "AAA", [80.0] * 7, days=DAYS[5:12],
          recorded_at=f"{DAYS[11]}T21:00:00Z")
    _bars(store, spread.REFERENCE_TICKER, [50.0] * 12, days=DAYS[:12],
          recorded_at=f"{DAYS[11]}T21:00:00Z")

    out = scoring.score_orders(as_of=DAYS[20], horizon_days=4)

    assert out["scored"] == []
    assert "did not trade" in out["unfilled"][0]["reason"].lower()


def test_an_order_rolls_only_as_far_as_the_next_session(store):
    """Not to whenever the name next appears. Rolling indefinitely is how a
    study buys a week after the news it was reacting to."""
    from research import spread
    _order(store, session=DAYS[1], cost_bps=0.0)
    _bars(store, "AAA", [100.0] * 8, days=DAYS[4:12],
          recorded_at=f"{DAYS[11]}T21:00:00Z")
    # The exchange was open the whole time.
    _bars(store, spread.REFERENCE_TICKER, [50.0] * 12, days=DAYS[:12],
          recorded_at=f"{DAYS[11]}T21:00:00Z")

    out = scoring.score_orders(as_of=DAYS[20], horizon_days=4)
    assert out["scored"] == []
    assert out["unfilled"]


# --- a t-statistic found by searching is not a t-statistic ------------------
#
# Three variants were replayed on the same 82 names: the time-series signal,
# the cross-sectional one entered at the earnings release, and the same one
# entered at the 10-Q. The third came back t=+2.47 on 71 trades and cleared
# every gate, which would have replaced a declared 40bp with 340.
#
# One of three passing at t>2 is roughly what chance produces. The gate cannot
# tell a pre-registered test from the best of a search unless it is told how
# many were tried, so it has to be told -- and a threshold that ignores the
# search is a threshold that ratifies it.

def _at_t(n, target_t, spread=300.0):
    """A sample whose t-statistic is exactly `target_t`, so a test can sit
    between two thresholds on purpose."""
    import math
    import statistics

    base = [spread if i % 2 else -spread for i in range(n)]
    mean = target_t * statistics.stdev(base) / math.sqrt(n)
    return [b + mean for b in base]


def test_a_result_picked_from_several_needs_a_higher_bar():
    """Bonferroni is blunt and honest: three comparisons means the bar for any
    one of them moves out, from 2.00 to 2.39. A t of 2.20 sits between."""
    nets = _at_t(40, 2.20)
    alone = scoring._summarise(_scored(nets))
    searched = scoring._summarise(_scored(nets), comparisons=3)

    assert alone["calibrated"] is True
    assert searched["calibrated"] is False
    assert "3 comparisons" in searched["calibration_note"]


def test_one_comparison_is_the_default_and_changes_nothing():
    nets = _at_t(40, 2.20)
    assert scoring._summarise(_scored(nets)) == \
        scoring._summarise(_scored(nets), comparisons=1)


def test_the_threshold_actually_used_is_reported():
    """A reader has to be able to see which bar the number cleared."""
    out = scoring._summarise(_scored([50.0] * 40), comparisons=4)
    assert out["t_threshold"] > 2.0
    assert out["comparisons"] == 4


def test_a_strong_enough_result_survives_the_correction():
    """The bar moves; it does not become unreachable."""
    out = scoring._summarise(_scored(_at_t(60, 3.10)), comparisons=3)
    assert out["calibrated"] is True
    assert out["t_threshold"] == pytest.approx(2.394, abs=0.01)


# --- before the open and after the close are not the same trade -------------
#
# A nightly scan decides on the evening of day D and enters at D+1's open.
#
# For a print after the close on D, the gap is the D-to-D+1 overnight move, and
# entering at D+1's open is entering after it -- which is what post-earnings
# drift means: the move that follows the reaction, not the reaction.
#
# For a print before the open on D, the gap was that morning. By D+1's open the
# reaction is a day old and a day of drift has already been given away. So bmo
# names are systematically entered later in the effect than amc ones, and an
# average over both hides it.

def _with_timing(store, ticker, period, announced, timing):
    store.record_announcement(ticker, period, announced, timing=timing,
                              recorded_at=f"{announced}T21:00:00Z")


def test_a_score_carries_the_hour_the_print_landed(store):
    _with_timing(store, "MISS", "2026Q1", DAYS[0], "amc")
    _calendar(store)
    _order(store, ticker="MISS", session=DAYS[1], cost_bps=0.0)
    _bars(store, "MISS", [100.0] * 3 + [110.0] * 5)

    row = scoring.score_orders(as_of=DAYS[20], horizon_days=4)["scored"][0]
    assert row["timing"] == "amc"


def test_a_trade_with_no_announcement_on_record_says_unknown(store):
    _calendar(store)
    _order(store, ticker="MISS", session=DAYS[1], cost_bps=0.0)
    _bars(store, "MISS", [100.0] * 3 + [110.0] * 5)

    row = scoring.score_orders(as_of=DAYS[20], horizon_days=4)["scored"][0]
    assert row["timing"] == "unknown"


def test_the_summary_splits_by_the_hour(store):
    """One number over both hides a difference that is structural rather than
    incidental, and the split is free -- the field is already on the row."""
    _calendar(store)
    for i, (t, timing, move) in enumerate([
            ("A", "amc", 110.0), ("B", "amc", 112.0),
            ("C", "bmo", 98.0), ("D", "bmo", 99.0)]):
        _with_timing(store, t, "2026Q1", DAYS[0], timing)
        _order(store, ticker=t, session=DAYS[1], cost_bps=0.0)
        _bars(store, t, [100.0] * 3 + [move] * 5)

    out = scoring.score_orders(as_of=DAYS[20], horizon_days=4)
    assert out["sample"] == 4
    assert out["by_timing"]["amc"]["sample"] == 2
    assert out["by_timing"]["bmo"]["sample"] == 2
    assert out["by_timing"]["amc"]["mean_net_bps"] > 0
    assert out["by_timing"]["bmo"]["mean_net_bps"] < 0


def test_the_split_is_absent_rather_than_empty_when_nothing_scored(store):
    out = scoring.score_orders(as_of=DAYS[20], horizon_days=4)
    assert out["by_timing"] == {}


def test_one_malformed_order_does_not_stop_the_run(store):
    """`fill` read side and cost tolerantly while the identifying fields were
    hard-indexed, so an order missing one raised out of the whole scoring run
    instead of being reported as the one bad row it is."""
    _calendar(store)
    _bars(store, "AAA", [100.0] * 3 + [110.0] * 5)
    store.record_paper_orders(
        DAYS[0],
        [{"ticker": "AAA", "fiscal_period": "2026Q1", "side": "long",
          "sue": 2.0, "intended_session": DAYS[1]}],   # no target_dollars
        recorded_at=f"{DAYS[0]}T21:00:00Z")

    out = scoring.score_orders(as_of=DAYS[20], horizon_days=4)
    assert out["sample"] == 1
    assert out["scored"][0]["target_dollars"] is None
    assert out["scored"][0]["gross_bps"] == pytest.approx(1000.0)


# --- the correction has to be reachable from the thing that runs -------------
#
# The Bonferroni machinery was correct and nothing in production could get to
# it. `score_orders` had no `comparisons` parameter at all, so the Saturday
# scoring job scored at one comparison however many variants had been tried,
# and it split the sample by announcement timing and judged each subgroup
# independently at t>2.00 -- two or three simultaneous tests, each allowed the
# bar for one.

def test_score_orders_takes_a_comparison_count(store):
    out = scoring.score_orders(as_of=DAYS[20], horizon_days=4, comparisons=3)
    assert out["comparisons"] == 3
    assert scoring.score_orders(as_of=DAYS[20])["comparisons"] == 1


def test_a_timing_subgroup_is_judged_against_every_subgroup_tried(store):
    """Splitting one sample into two and asking both at t>2.00 is two tests
    dressed as one. The count the subgroup is judged at has to include the
    split itself."""
    _calendar(store, DAYS[:20])
    for t, timing, move in [("A", "amc", 110.0), ("B", "amc", 112.0),
                            ("C", "bmo", 98.0), ("D", "bmo", 99.0)]:
        _with_timing(store, t, "2026Q1", DAYS[0], timing)
        _order(store, ticker=t, session=DAYS[1], cost_bps=0.0)
        _bars(store, t, [100.0] * 3 + [move] * 5)

    out = scoring.score_orders(as_of=DAYS[19], horizon_days=4)
    assert out["comparisons"] == 1
    assert out["by_timing"]["amc"]["comparisons"] == 2
    assert out["by_timing"]["bmo"]["comparisons"] == 2

    searched = scoring.score_orders(as_of=DAYS[19], horizon_days=4,
                                    comparisons=3)
    assert searched["by_timing"]["amc"]["comparisons"] == 6


def test_the_scoring_cli_takes_a_comparison_count(store, monkeypatch):
    """The Saturday job is this CLI. A count it cannot be given is a count it
    never carries."""
    seen = {}

    def fake(**kwargs):
        seen.update(kwargs)
        return {}

    monkeypatch.setattr(scoring, "score_orders", fake)
    assert scoring.main(["--comparisons", "4"]) == 0
    assert seen["comparisons"] == 4


def test_a_roll_that_would_skip_sessions_is_not_a_fill(store):
    """The roll is one session, and the code rolled to whenever the name next
    appeared. An order for Thanksgiving 2026 filled thirteen sessions later at
    a price 40% lower, and scored the gap it was supposed to be holding
    through as though the position had been open the whole time."""
    _order(store, session=DAYS[1], cost_bps=0.0)
    # The exchange was shut on DAYS[1] and open every session after it.
    _calendar(store, [DAYS[0]] + DAYS[2:30])
    # The name did not print again until DAYS[14], after a 40% gap.
    _bars(store, "AAA", [100.0], days=[DAYS[0]],
          recorded_at=f"{DAYS[0]}T21:00:00Z")
    _bars(store, "AAA", [60.0] * 11, days=DAYS[14:25],
          recorded_at=f"{DAYS[24]}T21:00:00Z")

    out = scoring.score_orders(as_of=DAYS[29], horizon_days=4)

    assert out["scored"] == [], (
        f"an order rolled {out['scored'] and out['scored'][0]['entry_session']}"
        f" past a session it could not have been filled on")
    assert out["unfilled"]


def test_a_missing_reference_series_is_a_gap_not_a_verdict(store):
    """`_exchange_shut` answered "the market was open" when the reference
    instrument had no bars at all -- so a hole in the recorder turned every
    order dated on a session the name did not print into "never filled" and
    deleted it from the sample, silently, with the sample selected on the
    hole."""
    _order(store, session=DAYS[1], cost_bps=0.0)
    _bars(store, "AAA", [100.0], days=[DAYS[0]],
          recorded_at=f"{DAYS[0]}T21:00:00Z")
    _bars(store, "AAA", [100.0] * 10, days=DAYS[5:15],
          recorded_at=f"{DAYS[14]}T21:00:00Z")

    with pytest.raises(scoring.ReferenceSeriesMissing) as raised:
        scoring.score_orders(as_of=DAYS[20], horizon_days=4)
    assert "SPY" in str(raised.value)


# --- one column, two quantities ---------------------------------------------
#
# `sue` holds a sigma when the variant is ts or af and `percentile - 0.5` when
# it is cs, which is bounded by 0.5 however large the surprise. Averaging
# gross/|sue| over a book containing both reads a 100bp cross-sectional winner
# as 220 basis points per SUE against a declared 15 -- a coefficient of neither
# signal, quoted as though it were of both.

def _mixed(nets, variants):
    return [{"ticker": f"T{i}", "sue": 2.0, "net_bps": n, "gross_bps": n,
             "cost_bps": 0.0, "variant": v}
            for i, (n, v) in enumerate(zip(nets, variants))]


def _priced(n, strengths, grosses, variant="ts", nets=None):
    """Rows as the scanner now files them: `strength` is the quantity it
    multiplied by the coefficient, kept apart from the `sue` column whose
    meaning depends on the variant."""
    nets = nets if nets is not None else grosses
    return [{"ticker": f"T{i}", "sue": strengths[i % len(strengths)],
             "strength": strengths[i % len(strengths)],
             "gross_bps": grosses[i % len(grosses)],
             "net_bps": nets[i % len(nets)],
             "cost_bps": 0.0, "variant": variant} for i in range(n)]


# --- the coefficient prices what the scanner multiplied ---------------------

def test_the_cross_sectional_coefficient_prices_a_tail_not_the_sue_column():
    """`sue` holds percentile-0.5 for this variant and the scanner prices
    twice that. Reporting gross/|sue| offered a coefficient exactly 2x the one
    that could replace DRIFT_BPS_PER_TAIL."""
    rows = [{"ticker": f"T{i}", "sue": 0.45, "strength": 0.90,
             "gross_bps": 100.0, "net_bps": 100.0, "cost_bps": 0.0,
             "variant": "cs"} for i in range(40)]

    out = scoring._summarise(rows)

    assert out["drift_bps_per_tail"] == pytest.approx(100.0 / 0.90)
    assert out["drift_bps_per_sue"] is None, (
        "a tail coefficient was offered under the name of a sigma one")
    assert out["drift_bps_prices"] == "tail"


def test_the_time_series_coefficient_still_prices_a_sigma():
    out = scoring._summarise(_priced(40, [2.0], [400.0], variant="ts"))

    assert out["drift_bps_per_sue"] == pytest.approx(200.0)
    assert out["drift_bps_per_tail"] is None
    assert out["drift_bps_prices"] == "sigma"


def test_a_row_filed_before_strength_existed_is_read_from_its_variant():
    """The column is new; rows already in the book do not have it. The variant
    says exactly what the scanner multiplied, so it is derived rather than
    guessed -- and never read off the sue column alone."""
    old = [{"ticker": f"T{i}", "sue": 0.45, "gross_bps": 100.0,
            "net_bps": 100.0, "cost_bps": 0.0, "variant": "cs"}
           for i in range(40)]

    assert scoring._summarise(old)["drift_bps_per_tail"] == pytest.approx(
        100.0 / 0.90)


def test_a_row_with_no_variant_is_left_out_of_the_coefficient():
    unknown = [{"ticker": f"T{i}", "sue": 2.0, "gross_bps": 400.0,
                "net_bps": 400.0, "cost_bps": 0.0, "variant": None}
               for i in range(40)]

    out = scoring._summarise(unknown)

    assert out["drift_bps_per_sue"] is None
    assert out["drift_bps_per_tail"] is None


# --- and it is a slope, with an error bar -----------------------------------

def test_the_coefficient_is_a_slope_through_the_origin_not_a_mean_of_ratios():
    """Averaging per-trade ratios weights a |SUE| of 1 the same as a |SUE| of
    5, which puts the most weight on the noisiest observations. Simulated at
    the declared 15 it is about 48% noisier than the slope at every sample
    size."""
    rows = _priced(40, [1.0, 5.0], [100.0, 100.0])

    out = scoring._summarise(rows)

    # mean of ratios would be (100/1 + 100/5)/2 = 60.
    assert out["drift_bps_per_sue"] == pytest.approx(600.0 / 26.0)


def test_the_coefficient_carries_a_standard_error_and_an_interval():
    out = scoring._summarise(_priced(40, [2.0], [20.0, 30.0, 25.0, 15.0, -5.0]))

    assert out["drift_bps_se"] > 0
    low, high = out["drift_bps_ci"]
    assert low < out["drift_bps_per_sue"] < high
    assert high - low == pytest.approx(2 * 1.96 * out["drift_bps_se"], rel=1e-6)


def test_a_coefficient_whose_interval_contains_zero_is_not_quoted():
    """The gate the whole thing exists for. A replay arm of 74 trades put the
    coefficient's 95% interval at roughly -50 to +57 around a true 15; nothing
    in the output said so, and `calibrated` was decided on the net return,
    which is a different question.

    gross and net are set independently here to isolate that gate from the
    t-statistic on returns, which these nets pass comfortably."""
    rows = _priced(40, [2.0], [1000.0, -1000.0], nets=[20.0])

    out = scoring._summarise(rows)

    assert out["t_stat"] is None or out["mean_net_bps"] > 0
    low, high = out["drift_bps_ci"]
    assert low < 0 < high
    assert out["calibrated"] is False
    assert "interval" in out["calibration_note"]


def test_an_interval_clear_of_zero_still_calibrates():
    out = scoring._summarise(_priced(40, [2.0], [20.0, 30.0, 25.0, 15.0, -5.0]))

    low, high = out["drift_bps_ci"]
    assert low > 0
    assert out["calibrated"] is True


def test_the_summary_says_how_far_from_a_usable_coefficient_it_is():
    """`calibrated: False` on its own does not say whether the answer is two
    more months of recording or twenty years of it. The interval's width
    relative to the estimate does, and it falls as 1/sqrt(n)."""
    out = scoring._summarise(_priced(40, [2.0], [20.0, 30.0, 25.0, 15.0, -5.0]))

    half_width = out["drift_bps_ci"][1] - out["drift_bps_per_sue"]
    assert out["interval_pct_of_estimate"] == pytest.approx(
        half_width / out["drift_bps_per_sue"])
    # Four times the sample halves the interval, so the projection scales with
    # the square of how much narrower it has to get.
    assert out["trades_for_a_50pct_interval"] == pytest.approx(
        40 * (out["interval_pct_of_estimate"] / 0.5) ** 2, rel=0.05)


def test_a_sample_that_cannot_pin_the_coefficient_says_what_would():
    """A small positive slope buried in a large dispersion: the sign is real
    in the point estimate and meaningless in the interval."""
    rows = _priced(40, [2.0], [1000.0, -900.0], nets=[20.0])

    out = scoring._summarise(rows)

    assert out["calibrated"] is False
    assert out["trades_for_a_50pct_interval"] > 1000
    assert "trades" in out["calibration_note"]


def test_an_interval_that_clears_zero_is_not_yet_precise_enough_to_price_with():
    """Two different claims, and only the first is what `calibrated` answers.

    Excluding zero establishes the sign. Replacing a declared coefficient needs
    the size, and simulated at the declared 15 over 74 trades -- the size of
    the arms actually replayed -- the slope came back +63.7 with an interval of
    +1.6 to +125.8: clear of zero, four times the truth, and no basis for
    replacing anything. `coefficient_usable` is the flag for that stronger
    claim, kept apart so neither can be read as the other."""
    rows = _priced(40, [2.0], [1000.0, -380.8], nets=[20.0])

    out = scoring._summarise(rows)

    low, _ = out["drift_bps_ci"]
    assert low > 0, "this sample is supposed to clear zero"
    assert out["interval_pct_of_estimate"] > scoring.TARGET_HALF_WIDTH
    assert out["coefficient_usable"] is False
    assert "50%" in out["coefficient_note"]


def test_a_precise_enough_coefficient_is_marked_usable():
    out = scoring._summarise(_priced(40, [2.0], [20.0, 30.0, 25.0, 15.0, -5.0]))

    assert out["interval_pct_of_estimate"] < scoring.TARGET_HALF_WIDTH
    assert out["coefficient_usable"] is True


def test_a_sample_that_passes_every_return_gate_can_still_be_too_imprecise():
    """The one that matters: nothing here is wrong with the book. The sample
    is large enough, the mean and median agree, and the t-statistic clears its
    bar -- and the coefficient is still only known to within 89%, which is not
    a number to price with. `calibrated` says yes and `coefficient_usable`
    says no, which is the whole reason they are two flags."""
    out = scoring._summarise(_scored(_at_t(40, 2.20)))

    assert out["calibrated"] is True
    assert out["interval_pct_of_estimate"] > scoring.TARGET_HALF_WIDTH
    assert out["coefficient_usable"] is False


def test_a_coefficient_nobody_can_quote_is_not_usable_either():
    out = scoring._summarise(_priced(40, [2.0], [1000.0, -900.0], nets=[20.0]))

    assert out["drift_bps_per_sue"] is not None
    assert out["calibrated"] is False
    assert out["coefficient_usable"] is False


def test_a_coefficient_of_zero_projects_nothing_rather_than_infinity():
    out = scoring._summarise(_priced(40, [2.0], [100.0, -100.0]))

    assert out["drift_bps_per_sue"] == pytest.approx(0.0)
    assert out["interval_pct_of_estimate"] is None
    assert out["trades_for_a_50pct_interval"] is None


def test_a_book_that_mixes_variants_does_not_quote_one_coefficient():
    nets = [20.0, 30.0, 25.0, 15.0, -5.0] * 8
    mixed = _mixed(nets, ["ts"] * 20 + ["cs"] * 20)

    out = scoring._summarise(mixed)

    assert out["drift_bps_per_sue"] is None, (
        "a coefficient was fitted over a sigma and a rank together")
    assert out["calibrated"] is False
    assert "variant" in out["calibration_note"]
    assert out["variants"] == ["cs", "ts"]


def test_one_variant_still_calibrates():
    nets = [20.0, 30.0, 25.0, 15.0, -5.0] * 8
    out = scoring._summarise(_mixed(nets, ["ts"] * 40))
    assert out["calibrated"] is True
    assert out["drift_bps_per_sue"] is not None


def test_the_score_splits_by_variant(store):
    """So a mixed book is still measurable -- separately, which is the only way
    either number means anything."""
    _calendar(store, DAYS[:20])
    for ticker, variant, move in [("A", "ts", 110.0), ("B", "cs", 90.0)]:
        store.record_paper_orders(
            DAYS[0],
            [{"ticker": ticker, "side": "long", "sue": 2.0, "variant": variant,
              "fiscal_period": "2026Q1", "expected_edge_bps": 30.0,
              "cost_bps": 0.0, "target_dollars": 5000.0,
              "intended_session": DAYS[1]}],
            recorded_at=f"{DAYS[0]}T21:00:00Z")
        _bars(store, ticker, [100.0] * 3 + [move] * 5)

    out = scoring.score_orders(as_of=DAYS[19], horizon_days=4)

    assert out["sample"] == 2
    assert out["drift_bps_per_sue"] is None, "the mixture was averaged over"
    assert out["by_variant"]["ts"]["sample"] == 1
    assert out["by_variant"]["cs"]["sample"] == 1
    assert out["by_variant"]["ts"]["drift_bps_per_sue"] == pytest.approx(500.0)


# --- twenty sessions means twenty sessions ----------------------------------
#
# `forward[horizon_days]` is the 21st row present in the store, not the 21st
# session. A partial night is documented as normal on a universe this size, so
# a name missing one bar was held for 21 sessions and a name missing three for
# 23 -- silently, and the pending/unfilled split inherited the same error.

def test_the_horizon_is_counted_in_sessions_not_in_recorded_bars(store):
    _order(store, session=DAYS[1], cost_bps=0.0)
    _calendar(store, DAYS[:20])
    # One session missing from the middle of the hold.
    days = [DAYS[1], DAYS[2], DAYS[4], DAYS[5], DAYS[6], DAYS[7], DAYS[8]]
    _bars(store, "AAA", [100.0, 100.0, 100.0, 100.0, 110.0, 130.0, 130.0],
          days=days, recorded_at=f"{DAYS[8]}T21:00:00Z")

    row = scoring.score_orders(as_of=DAYS[19], horizon_days=5)["scored"][0]

    assert row["exit_session"] == DAYS[6], (
        f"a five-session hold from {DAYS[1]} exited on {row['exit_session']}; "
        f"the missing bar lengthened it")
    assert row["gross_bps"] == pytest.approx(1000.0)


def test_a_horizon_measured_against_an_absent_bar_is_not_scored(store):
    """The other half: if the name did not print on the exit session there is
    no exit price, and the nearest one is a different trade."""
    _order(store, session=DAYS[1], cost_bps=0.0)
    _calendar(store, DAYS[:20])
    days = [DAYS[1], DAYS[2], DAYS[3], DAYS[7], DAYS[8], DAYS[9]]
    _bars(store, "AAA", [100.0] * 3 + [60.0] * 3, days=days,
          recorded_at=f"{DAYS[9]}T21:00:00Z")

    out = scoring.score_orders(as_of=DAYS[19], horizon_days=5)

    assert out["scored"] == []
    assert out["unfilled"]
    assert DAYS[6] in out["unfilled"][0]["reason"]


def test_the_pending_split_counts_sessions_too(store):
    """A store holding four of the five sessions in a hold has not finished it,
    and a store holding five bars spread over eight sessions has."""
    _order(store, session=DAYS[1], cost_bps=0.0)
    _calendar(store, DAYS[:6])
    # Four bars, one of them missing, over five sessions of the exchange.
    _bars(store, "AAA", [100.0] * 4, days=[DAYS[1], DAYS[2], DAYS[4], DAYS[5]],
          recorded_at=f"{DAYS[5]}T21:00:00Z")

    out = scoring.score_orders(as_of=DAYS[5], horizon_days=5)
    assert out["scored"] == []
    assert "4 of 5 sessions" in out["pending"][0]["reason"], (
        out["pending"][0]["reason"])


def test_a_fresh_store_is_created_by_the_scoring_job_itself(tmp_path,
                                                            monkeypatch):
    """`research-score` runs on a Saturday, and on a fresh volume it ran
    before anything had created the tables."""
    monkeypatch.setenv("NEMO_PIT_DB", str(tmp_path / "fresh.db"))
    assert scoring.main(["--as-of", DAYS[10]]) == 0
