"""Hand-computed answers, so a changed formula cannot pass by being plausible.

Everything else in this suite checks relationships -- bigger surprise ranks
higher, a split does not read as a return, a refusal carries no number. Those
survive a formula that is wrong by a factor of two in the same direction
everywhere.

These are arithmetic. Each expected value is worked out on paper in the
docstring, so a diff that changes a denominator has to change a number here and
explain itself.
"""
import math
from datetime import date, timedelta

import pytest

from research import pit_store, scanner, scoring, spread, sue_cs


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("NEMO_PIT_DB", str(tmp_path / "pit.db"))
    pit_store.init_schema()
    return pit_store


def _days(n, start="2025-01-01"):
    out, d = [], date.fromisoformat(start)
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += timedelta(days=1)
    return out


DAYS = _days(400)


def _flat(store, ticker, price=100.0, volume=1_000_000, n=300, wobble=0.0):
    """A price path. `wobble` gives the estimator something to chew on -- a
    perfectly flat series has no range at all, which EDGE correctly refuses to
    read a spread out of."""
    rows = []
    for i, d in enumerate(DAYS[:n]):
        p_ = price * (1 + math.sin(i * 0.7) * wobble)
        rows.append({"trade_date": d, "open": p_,
                     "high": p_ * (1 + wobble), "low": p_ * (1 - wobble),
                     "close": p_, "volume": volume})
    store.record_bars(ticker, rows, recorded_at=f"{DAYS[n-1]}T21:00:00Z")
    return DAYS[n - 1]


# --- adjustment -------------------------------------------------------------

def test_a_three_for_one_split_divides_by_exactly_three(store):
    """400.00 before, 133.33 after. The pre-split bar adjusts to 400/3 and its
    volume multiplies by 3: 900 -> 2700."""
    store.record_bars("X", [
        {"trade_date": "2026-03-02", "open": 400.0, "high": 400.0,
         "low": 400.0, "close": 400.0, "volume": 900},
        {"trade_date": "2026-03-03", "open": 133.33, "high": 133.33,
         "low": 133.33, "close": 133.33, "volume": 2700}],
        recorded_at="2026-03-03T21:00:00Z")
    store.record_corporate_action("X", "2026-03-03", "split", 3.0,
                                  recorded_at="2026-03-03T21:00:00Z")

    bars = store.adjusted_bars("X", "2026-03-04")
    assert bars[0]["close"] == pytest.approx(400.0 / 3.0)
    assert bars[0]["volume"] == pytest.approx(2700.0)
    assert bars[0]["adj_factor"] == pytest.approx(1.0 / 3.0)
    assert bars[1]["close"] == pytest.approx(133.33)


def test_a_dividend_adjusts_by_one_minus_its_yield(store):
    """A $2.50 dividend against a $100.00 prior close is 2.5%, so the earlier
    bar becomes 100 * (1 - 0.025) = 97.50 exactly."""
    store.record_bars("X", [
        {"trade_date": "2026-03-02", "open": 100.0, "high": 100.0,
         "low": 100.0, "close": 100.0, "volume": 1},
        {"trade_date": "2026-03-03", "open": 97.5, "high": 97.5, "low": 97.5,
         "close": 97.5, "volume": 1}], recorded_at="2026-03-03T21:00:00Z")
    store.record_corporate_action("X", "2026-03-03", "dividend", 2.50,
                                  recorded_at="2026-03-03T21:00:00Z")

    bars = store.adjusted_bars("X", "2026-03-04", total_return=True)
    assert bars[0]["close"] == pytest.approx(97.5)


def test_two_splits_multiply_rather_than_add(store):
    """2-for-1 then 5-for-1 is a tenth, not a seventh. 700 -> 70."""
    store.record_bars("X", [
        {"trade_date": "2026-01-02", "open": 700.0, "high": 700.0,
         "low": 700.0, "close": 700.0, "volume": 10},
        {"trade_date": "2026-02-02", "open": 350.0, "high": 350.0,
         "low": 350.0, "close": 350.0, "volume": 20},
        {"trade_date": "2026-03-02", "open": 70.0, "high": 70.0, "low": 70.0,
         "close": 70.0, "volume": 100}], recorded_at="2026-03-02T21:00:00Z")
    store.record_corporate_action("X", "2026-02-02", "split", 2.0,
                                  recorded_at="2026-02-02T21:00:00Z")
    store.record_corporate_action("X", "2026-03-02", "split", 5.0,
                                  recorded_at="2026-03-02T21:00:00Z")

    closes = [b["close"] for b in store.adjusted_bars("X", "2026-03-03")]
    assert closes == pytest.approx([70.0, 70.0, 70.0])


# --- cost -------------------------------------------------------------------

def test_the_tick_floor_is_one_cent_over_the_price(store):
    """A cent on a $50 stock is 0.01/50 = 2.0bp."""
    as_of = _flat(store, "X", price=50.0)
    out = spread.estimate_spread("X", as_of, window=252)
    assert out["tick_floor"] == pytest.approx(0.01 / 50.0)
    assert out["tick_floor"] * 1e4 == pytest.approx(2.0)


def test_a_floored_round_trip_is_the_tick_plus_impact(store):
    """basis='adaptive' on a name the estimator cannot resolve charges one
    tick of spread plus the impact term, and nothing else."""
    as_of = _flat(store, spread.REFERENCE_TICKER, price=100.0, wobble=0.004)
    _flat(store, "X", price=100.0, wobble=0.004)

    cost = spread.round_trip_cost("X", as_of, 1000.0, window=252,
                                  basis="adaptive")
    assert cost["resolution"] == "at_resolution_floor", cost.get("reason")
    assert cost["cost"] == pytest.approx(
        cost["spread_cost"] + cost["impact_cost"])
    # One tick against the price that was actually last printed, not against
    # the nominal the fixture started from.
    last = pit_store.bars_as_of("X", as_of)[-1]["close"]
    assert cost["spread_cost"] == pytest.approx(0.01 / last)


def test_the_half_spread_is_half_of_what_is_charged(store):
    as_of = _flat(store, spread.REFERENCE_TICKER, wobble=0.004)
    _flat(store, "X", wobble=0.004)
    cost = spread.round_trip_cost("X", as_of, 1000.0, window=252,
                                  basis="adaptive")
    assert cost["half_spread"] == pytest.approx(cost["spread_used"] / 2.0)


def test_participation_is_the_order_over_median_dollar_volume(store):
    """1,000,000 shares at $100 is $100m a day. A $1m order is 1.0% of it, and
    the cap is 1%, so it sits exactly on the limit."""
    as_of = _flat(store, "X", price=100.0, volume=1_000_000)
    out = spread.participation_rate("X", as_of, 1_000_000.0, window=252)
    assert out["median_dollar_volume"] == pytest.approx(1e8)
    assert out["participation"] == pytest.approx(0.01)
    assert out["max_position_dollars"] == pytest.approx(1e6)
    assert out["within_limit"] is True


def test_a_hair_over_the_cap_is_outside_it(store):
    as_of = _flat(store, "X", price=100.0, volume=1_000_000)
    out = spread.participation_rate("X", as_of, 1_000_001.0, window=252)
    assert out["within_limit"] is False


# --- the scanner's edge -----------------------------------------------------

def test_expected_edge_is_the_surprise_times_the_declared_drift(store,
                                                                monkeypatch):
    """|SUE| 2.5 at 15bp per unit is 37.5bp, minus a 10bp cost, is 27.5bp net.
    Every one of those three numbers is on the candidate row."""
    as_of = _flat(store, "AAA", wobble=0.004)
    _flat(store, spread.REFERENCE_TICKER, wobble=0.004)
    store.record_universe(as_of, [{"ticker": "AAA", "cik": "1",
                                   "eligible": True}],
                          recorded_at=f"{as_of}T21:00:00Z")
    store.record_consensus(as_of, "AAA", "2026Q1", eps_estimate=1.0,
                           eps_actual=1.2, recorded_at=f"{as_of}T21:00:00Z")
    monkeypatch.setattr(scanner, "_signal_for", lambda t, a: {
        "ticker": t, "success": True, "error": None, "sue": 2.5,
        "fiscal_period": "2026Q1", "known_at": a, "sigma_quarters": 8,
        "sigma_periods": ["2026Q1"], "basis_changes": [], "variant": "ts"})
    monkeypatch.setattr(scanner, "_cost_for", lambda t, a, d: {
        "cost": 0.0010, "cost_floor": 0.0001, "reason": None,
        "spread": 0.0005, "resolved": True, "resolution": "measured"})

    c = scanner.scan(as_of=as_of)["candidates"][0]
    assert c["expected_edge_bps"] == pytest.approx(2.5 * scanner.DRIFT_BPS_PER_SUE)
    assert c["expected_edge_bps"] == pytest.approx(37.5)
    assert c["cost_bps"] == pytest.approx(10.0)
    assert c["net_edge_bps"] == pytest.approx(27.5)


def test_the_per_name_size_is_the_gross_divided_by_the_name_count(store,
                                                                  monkeypatch):
    as_of = _flat(store, "AAA", wobble=0.004)
    _flat(store, spread.REFERENCE_TICKER, wobble=0.004)
    store.record_universe(as_of, [{"ticker": "AAA", "cik": "1",
                                   "eligible": True}],
                          recorded_at=f"{as_of}T21:00:00Z")
    store.record_consensus(as_of, "AAA", "2026Q1", eps_estimate=1.0,
                           eps_actual=1.2, recorded_at=f"{as_of}T21:00:00Z")
    monkeypatch.setattr(scanner, "_signal_for", lambda t, a: {
        "ticker": t, "success": True, "error": None, "sue": 3.0,
        "fiscal_period": "2026Q1", "known_at": a, "sigma_quarters": 8,
        "sigma_periods": ["2026Q1"], "basis_changes": [], "variant": "ts"})
    monkeypatch.setattr(scanner, "_cost_for", lambda t, a, d: {
        "cost": 0.0001, "cost_floor": 0.00001, "reason": None,
        "spread": 0.00005, "resolved": True, "resolution": "measured"})
    monkeypatch.setattr(scanner, "_regime_scale", lambda a: (1.0, "calm"))

    out = scanner.scan(as_of=as_of)
    expected = scanner.GROSS_TARGET / scanner.MAX_NAMES
    assert out["candidates"][0]["target_dollars"] == pytest.approx(expected)


# --- scoring ----------------------------------------------------------------

def test_a_ten_percent_move_is_a_thousand_basis_points():
    row = scoring.fill({"ticker": "A", "side": "long", "sue": 1.0,
                        "cost_bps": 0.0, "as_of_date": "x",
                        "target_dollars": 1.0},
                       {"trade_date": "d1", "open": 200.0},
                       {"trade_date": "d2", "open": 220.0})
    assert row["gross_bps"] == pytest.approx(1000.0)


def test_the_drift_coefficient_is_gross_over_the_surprise():
    """Two trades: +60bp on a SUE of 2, +30bp on a SUE of 3. Per unit that is
    30 and 10, and the mean of those is 20 -- not 90/5."""
    scored = [
        {"ticker": "A", "sue": 2.0, "gross_bps": 60.0, "net_bps": 60.0,
         "cost_bps": 0.0},
        {"ticker": "B", "sue": 3.0, "gross_bps": 30.0, "net_bps": 30.0,
         "cost_bps": 0.0},
    ]
    out = scoring._summarise(scored)
    assert out["drift_bps_per_sue"] == pytest.approx(20.0)
    assert out["mean_net_bps"] == pytest.approx(45.0)
    assert out["median_net_bps"] == pytest.approx(45.0)
    assert out["hit_rate"] == pytest.approx(1.0)


def test_the_t_statistic_is_the_mean_over_the_standard_error():
    nets = [10.0, 20.0, 30.0, 40.0]
    out = scoring._summarise(
        [{"ticker": f"T{i}", "sue": 1.0, "gross_bps": n, "net_bps": n,
          "cost_bps": 0.0} for i, n in enumerate(nets)])
    import statistics
    expected = statistics.fmean(nets) / (
        statistics.stdev(nets) / math.sqrt(len(nets)))
    assert out["t_stat"] == pytest.approx(expected)


# --- the cross-section ------------------------------------------------------

def test_the_scaled_surprise_is_the_beat_over_the_price(store):
    """A 30 cent beat on a $60 stock is 0.005, which is 50 basis points."""
    for i in range(10):
        store.record_consensus("2026-03-02", f"N{i}", "2026Q1",
                               eps_estimate=1.0, eps_actual=1.0 + i * 0.01,
                               recorded_at="2026-03-02T21:00:00Z")
        store.record_bars(f"N{i}", [{"trade_date": "2026-03-02", "open": 60.0,
                                     "high": 60.0, "low": 60.0, "close": 60.0,
                                     "volume": 1}],
                          recorded_at="2026-03-02T21:00:00Z")
    store.record_consensus("2026-03-02", "BEAT", "2026Q1", eps_estimate=1.0,
                           eps_actual=1.30, recorded_at="2026-03-02T21:00:00Z")
    store.record_bars("BEAT", [{"trade_date": "2026-03-02", "open": 60.0,
                                "high": 60.0, "low": 60.0, "close": 60.0,
                                "volume": 1}],
                      recorded_at="2026-03-02T21:00:00Z")

    out = sue_cs.surprise_rank("BEAT", as_of="2026-03-03")
    assert out["surprise"] == pytest.approx(0.30)
    assert out["scaled_surprise"] == pytest.approx(0.005)
    assert out["scaled_surprise"] * 1e4 == pytest.approx(50.0)
    assert out["percentile"] == pytest.approx(1.0)


def test_the_median_of_a_symmetric_cohort_sits_at_the_middle(store):
    for i in range(11):
        store.record_consensus("2026-03-02", f"N{i}", "2026Q1",
                               eps_estimate=1.0,
                               eps_actual=1.0 + (i - 5) * 0.01,
                               recorded_at="2026-03-02T21:00:00Z")
        store.record_bars(f"N{i}", [{"trade_date": "2026-03-02", "open": 100.0,
                                     "high": 100.0, "low": 100.0,
                                     "close": 100.0, "volume": 1}],
                          recorded_at="2026-03-02T21:00:00Z")

    middle = sue_cs.surprise_rank("N5", as_of="2026-03-03")
    assert middle["surprise"] == pytest.approx(0.0)
    assert middle["percentile"] == pytest.approx(0.5)
    assert middle["robust_z"] == pytest.approx(0.0)
