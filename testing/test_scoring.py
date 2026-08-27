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


def _order(store, ticker="AAA", side="long", sue=3.0, cost_bps=10.0,
           decided=None, session=None, dollars=5000.0):
    decided = decided or DAYS[0]
    session = session or DAYS[1]
    store.record_paper_orders(
        decided,
        [{"ticker": ticker, "side": side, "sue": sue, "fiscal_period": "2026Q1",
          "expected_edge_bps": abs(sue) * 15.0, "cost_bps": cost_bps,
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
    _bars(store, "AAA", [100.0] * 5)

    out = scoring.score_orders(as_of=DAYS[4], horizon_days=20)

    assert out["scored"] == []
    assert out["pending"] and out["pending"][0]["ticker"] == "AAA"


def test_a_finished_horizon_is_scored(store):
    _order(store)
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
    _bars(store, "AAA", [50.0, 100.0, 100.0, 100.0, 100.0, 110.0, 110.0])

    out = scoring.score_orders(as_of=DAYS[10], horizon_days=4)
    row = out["scored"][0]

    assert row["entry_price"] == 100.0, (
        "entry was taken from a session other than the intended one")
    assert row["gross_bps"] == pytest.approx(1000.0)


def test_the_cost_recorded_at_decision_time_is_subtracted(store):
    _order(store, session=DAYS[1], cost_bps=40.0)
    _bars(store, "AAA", [100.0, 100.0, 100.0, 100.0, 100.0, 110.0, 110.0])

    row = scoring.score_orders(as_of=DAYS[10], horizon_days=4)["scored"][0]

    assert row["gross_bps"] == pytest.approx(1000.0)
    assert row["net_bps"] == pytest.approx(960.0)


def test_a_short_earns_when_the_price_falls(store):
    _order(store, side="short", sue=-3.0, session=DAYS[1], cost_bps=0.0)
    _bars(store, "AAA", [100.0, 100.0, 100.0, 100.0, 100.0, 90.0, 90.0])

    row = scoring.score_orders(as_of=DAYS[10], horizon_days=4)["scored"][0]
    assert row["net_bps"] == pytest.approx(1000.0)


def test_a_split_inside_the_horizon_is_not_a_return(store):
    """The store keeps as-traded prices on purpose, so a 10-for-1 is a genuine
    -90% print. Scoring the raw series would book it as the worst trade ever
    made."""
    _order(store, session=DAYS[1], cost_bps=0.0)
    _bars(store, "AAA", [100.0, 100.0, 100.0, 10.0, 10.0, 11.0, 11.0])
    store.record_corporate_action("AAA", DAYS[3], "split", 10.0,
                                  recorded_at=f"{DAYS[3]}T21:00:00Z")

    row = scoring.score_orders(as_of=DAYS[10], horizon_days=4)["scored"][0]
    assert row["net_bps"] == pytest.approx(1000.0), (
        f"the split was scored as a return: {row['net_bps']:.0f}bp")


# --- anti-lookahead, one more time ------------------------------------------

def test_scoring_cannot_see_a_bar_recorded_after_its_date(store):
    _order(store, session=DAYS[1])
    _bars(store, "AAA", [100.0] * 10, recorded_at=f"{DAYS[30]}T21:00:00Z")

    out = scoring.score_orders(as_of=DAYS[10], horizon_days=4)
    assert out["scored"] == []


# --- what the whole thing is for --------------------------------------------

def test_the_realised_drift_is_measured_not_echoed(store):
    """The calibration. If this reported the coefficient the scanner assumed,
    it would be a mirror rather than a measurement."""
    from research import scanner

    _order(store, ticker="AAA", sue=2.0, session=DAYS[1], cost_bps=0.0)
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
    _bars(store, "AAA", [100.0] * 5, days=DAYS[1:6],
          recorded_at=f"{DAYS[5]}T21:00:00Z")

    out = scoring.score_orders(as_of=DAYS[30], horizon_days=5)
    assert out["scored"] == [], "a four-session hold was scored as five"
    assert out["pending"]

    # One more session, and it is exactly a five-day hold.
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
    _bars(store, "AAA", [100.0] * 6, days=DAYS[1:7],
          recorded_at=f"{DAYS[6]}T21:00:00Z")

    out = scoring.score_orders(as_of=DAYS[30], horizon_days=60)
    assert out["scored"] == []
    assert out["pending"]
