"""The short side, which nothing has exercised end to end.

Every scenario so far has been a beat: positive surprise, long, price rises.
The sign flows through five places -- the side a candidate is given, the size
it is allowed, the cost charged, the direction a score is measured in, and the
drift coefficient derived from it -- and a sign error in any one of them is
invisible while every test is a long.

It matters more here than symmetry usually does, because a negative surprise is
the half of post-earnings drift that is harder to trade and easier to get
wrong: the drift is there, the borrow may not be, and a scorer that measures a
short in the long direction reports the exact opposite of what happened.
"""
from datetime import date, timedelta

import pytest

from research import pit_store, scanner, scoring, spread


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("NEMO_PIT_DB", str(tmp_path / "pit.db"))
    pit_store.init_schema()
    return pit_store


def _days(n, start="2025-06-02"):
    out, d = [], date.fromisoformat(start)
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += timedelta(days=1)
    return out


DAYS = _days(320)


def _series(store, ticker, price=100.0, n=300):
    rows = [{"trade_date": d, "open": price, "high": price * 1.01,
             "low": price * 0.99, "close": price, "volume": 2_000_000}
            for d in DAYS[:n]]
    store.record_bars(ticker, rows, recorded_at=f"{DAYS[n-1]}T21:00:00Z")
    return DAYS[n - 1]


@pytest.fixture
def ready(store):
    as_of = _series(store, "MISS")
    _series(store, spread.REFERENCE_TICKER)
    store.record_universe(as_of, [{"ticker": "MISS", "cik": "1",
                                   "eligible": True}],
                          recorded_at=f"{as_of}T21:00:00Z")
    store.record_consensus(as_of, "MISS", "2026Q1", eps_estimate=1.0,
                           eps_actual=0.6, recorded_at=f"{as_of}T21:00:00Z")
    return as_of


def _signal(sue, as_of):
    return lambda t, a: {
        "ticker": t, "success": True, "error": None, "sue": sue,
        "fiscal_period": "2026Q1", "known_at": a, "sigma_quarters": 8,
        "sigma_periods": ["2026Q1"], "basis_changes": [], "variant": "ts"}


def _cost(bps=10.0):
    return lambda t, a, d: {"cost": bps / 1e4, "cost_floor": 0.00002,
                            "reason": None, "spread": bps / 2e4,
                            "resolved": True, "resolution": "measured"}


# --- the decision -----------------------------------------------------------

def test_a_negative_surprise_becomes_a_short(ready, monkeypatch):
    monkeypatch.setattr(scanner, "_signal_for", _signal(-2.5, ready))
    monkeypatch.setattr(scanner, "_cost_for", _cost())

    c = scanner.scan(as_of=ready)["candidates"][0]
    assert c["side"] == "short"
    assert c["sue"] == -2.5


def test_a_short_is_sized_and_priced_like_a_long(ready, monkeypatch):
    """The edge is the magnitude of the surprise, so a -2.5 and a +2.5 buy the
    same expected move and the same position. Anything else is a directional
    view smuggled into the sizing."""
    monkeypatch.setattr(scanner, "_cost_for", _cost())

    monkeypatch.setattr(scanner, "_signal_for", _signal(-2.5, ready))
    short = scanner.scan(as_of=ready)["candidates"][0]
    monkeypatch.setattr(scanner, "_signal_for", _signal(2.5, ready))
    long = scanner.scan(as_of=ready)["candidates"][0]

    assert short["expected_edge_bps"] == pytest.approx(long["expected_edge_bps"])
    assert short["net_edge_bps"] == pytest.approx(long["net_edge_bps"])
    assert short["target_dollars"] == pytest.approx(long["target_dollars"])
    assert short["cost_bps"] == pytest.approx(long["cost_bps"])


def test_a_short_below_the_threshold_is_refused_like_a_long(ready, monkeypatch):
    monkeypatch.setattr(scanner, "_signal_for", _signal(-0.4, ready))
    monkeypatch.setattr(scanner, "_cost_for", _cost())

    out = scanner.scan(as_of=ready)
    assert out["candidates"] == []
    assert "0.40" in out["rejected"][0]["reason"]


def test_an_implausible_short_is_refused_like_an_implausible_long(ready,
                                                                  monkeypatch):
    monkeypatch.setattr(scanner, "_signal_for", _signal(-9.0, ready))
    monkeypatch.setattr(scanner, "_cost_for", _cost())

    out = scanner.scan(as_of=ready)
    assert out["candidates"] == []
    assert "9.00" in out["rejected"][0]["reason"]


def test_a_short_that_does_not_clear_its_cost_is_refused(ready, monkeypatch):
    monkeypatch.setattr(scanner, "_signal_for", _signal(-1.2, ready))
    monkeypatch.setattr(scanner, "_cost_for", _cost(bps=40.0))

    out = scanner.scan(as_of=ready)
    assert out["candidates"] == []


# --- the outcome ------------------------------------------------------------

def _filed(store, side, sue, entry_day, cost_bps=0.0):
    store.record_paper_orders(
        DAYS[DAYS.index(entry_day) - 1],
        [{"ticker": "MISS", "side": side, "sue": sue,
          "fiscal_period": "2026Q1", "expected_edge_bps": abs(sue) * 15,
          "cost_bps": cost_bps, "net_edge_bps": abs(sue) * 15 - cost_bps,
          "target_dollars": 5000.0, "intended_session": entry_day, "rank": 1}],
        recorded_at=f"{DAYS[DAYS.index(entry_day) - 1]}T21:00:00Z")


def _path(store, ticker, start_idx, prices):
    days = DAYS[start_idx:start_idx + len(prices)]
    store.record_bars(ticker, [
        {"trade_date": d, "open": p, "high": p, "low": p, "close": p,
         "volume": 1_000_000} for d, p in zip(days, prices)],
        recorded_at=f"{days[-1]}T21:00:00Z")


def test_a_short_into_a_falling_price_makes_money(store):
    _path(store, "MISS", 0, [100.0] * 3 + [80.0] * 5)
    _path(store, spread.REFERENCE_TICKER, 0, [50.0] * 8)
    _filed(store, "short", -2.0, DAYS[1])

    row = scoring.score_orders(as_of=DAYS[20], horizon_days=4)["scored"][0]
    assert row["net_bps"] == pytest.approx(2000.0)


def test_a_short_into_a_rising_price_loses(store):
    _path(store, "MISS", 0, [100.0] * 3 + [120.0] * 5)
    _path(store, spread.REFERENCE_TICKER, 0, [50.0] * 8)
    _filed(store, "short", -2.0, DAYS[1])

    row = scoring.score_orders(as_of=DAYS[20], horizon_days=4)["scored"][0]
    assert row["net_bps"] == pytest.approx(-2000.0)


def test_the_drift_coefficient_is_positive_when_shorts_work(store):
    """The coefficient is basis points per unit of |SUE|, so a short that fell
    contributes positively -- the same as a long that rose. A scorer measuring
    the short in the long direction would report the sign of the price move
    instead of the sign of the edge."""
    _path(store, "MISS", 0, [100.0] * 3 + [80.0] * 5)
    _path(store, spread.REFERENCE_TICKER, 0, [50.0] * 8)
    _filed(store, "short", -2.0, DAYS[1])

    out = scoring.score_orders(as_of=DAYS[20], horizon_days=4)
    assert out["drift_bps_per_sue"] == pytest.approx(1000.0)
    assert out["hit_rate"] == pytest.approx(1.0)


def test_a_short_and_a_long_that_both_worked_agree_on_the_coefficient(store):
    _path(store, "MISS", 0, [100.0] * 3 + [80.0] * 5)
    _path(store, "BEAT", 0, [100.0] * 3 + [120.0] * 5)
    _path(store, spread.REFERENCE_TICKER, 0, [50.0] * 8)
    _filed(store, "short", -2.0, DAYS[1])
    store.record_paper_orders(
        DAYS[0],
        [{"ticker": "BEAT", "side": "long", "sue": 2.0,
          "fiscal_period": "2026Q1", "expected_edge_bps": 30.0,
          "cost_bps": 0.0, "net_edge_bps": 30.0, "target_dollars": 5000.0,
          "intended_session": DAYS[1], "rank": 2}],
        recorded_at=f"{DAYS[0]}T21:00:00Z")

    out = scoring.score_orders(as_of=DAYS[20], horizon_days=4)
    assert out["sample"] == 2
    assert {r["side"] for r in out["scored"]} == {"long", "short"}
    # 2000bp on the short, 2000bp on the long, both over a |SUE| of 2.
    assert out["drift_bps_per_sue"] == pytest.approx(1000.0)
    assert out["hit_rate"] == pytest.approx(1.0)


def test_a_split_inside_a_short_is_not_a_windfall(store):
    """A 4-for-1 looks like a 75% fall, which to a short reads as the trade of
    the year."""
    _path(store, "MISS", 0, [100.0] * 3 + [25.0] * 5)
    _path(store, spread.REFERENCE_TICKER, 0, [50.0] * 8)
    store.record_corporate_action("MISS", DAYS[3], "split", 4.0,
                                  recorded_at=f"{DAYS[3]}T21:00:00Z")
    _filed(store, "short", -2.0, DAYS[1])

    row = scoring.score_orders(as_of=DAYS[20], horizon_days=4)["scored"][0]
    assert row["net_bps"] == pytest.approx(0.0), (
        f"the split scored as {row['net_bps']:.0f}bp of profit")
