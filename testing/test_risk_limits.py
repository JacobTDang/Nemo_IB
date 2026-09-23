"""The limits the book enforces on itself, and the bar before real money.

Issue #117, with the numbers the owner set on 2026-09-23:

- No position larger than 5% of the gross target. The scan already spreads
  the gross over twenty names, which is 5% each, but only by arithmetic: a
  change to MAX_NAMES or to the sizing rule would have moved it silently.
- A drawdown switch. The scan records the book's equity every night; a loss of
  10% of the gross target from the peak trips it, and a tripped switch stops
  the scan filing orders until a person resets it. It latches, because a switch
  that re-arms itself when equity bounces is a switch that trades through the
  drawdown it exists to stop.
- A go/no-go gate: 200 forward trades with measured fills, a mean net above
  zero, and a t-statistic above the bar for the six variants tried. It cannot
  pass on modeled costs, so until fills are measured (#116) it says so.
"""
from datetime import date, timedelta

import pytest

from research import pit_store, risk, scanner, scoring, spread, status


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


DAYS = _days(300)
AS_OF = DAYS[-1]
STAMP = f"{AS_OF}T21:00:00Z"
GROSS = scanner.GROSS_TARGET


def _bars(ticker, price_on):
    pit_store.record_bars(ticker, [
        {"trade_date": d, "open": price_on(d), "high": price_on(d) * 1.01,
         "low": price_on(d) * 0.99, "close": price_on(d), "volume": 2_000_000}
        for d in DAYS], recorded_at=STAMP)


def _order(ticker, filed, session, target=5_000.0, cost_bps=10.0):
    pit_store.record_paper_orders(
        filed,
        [{"ticker": ticker, "side": "long", "sue": 3.0, "variant": "ts",
          "strength": 3.0, "fiscal_period": "2026Q2",
          "expected_edge_bps": 45.0, "cost_bps": cost_bps,
          "net_edge_bps": 35.0, "target_dollars": target,
          "intended_session": session, "rank": 1}],
        recorded_at=f"{filed}T21:00:00Z")


def _equity(dollars):
    return {"equity_dollars": dollars, "realized_dollars": dollars,
            "open_dollars": 0.0, "closed_trades": 0, "open_trades": 0,
            "unmarked_trades": 0}


# --- the position cap --------------------------------------------------------

def test_no_position_is_larger_than_five_percent_of_the_gross_target(
        store, monkeypatch):
    """Five names would give each 20% of the gross. The cap holds anyway."""
    _bars("AAA", lambda d: 100.0)
    _bars(spread.REFERENCE_TICKER, lambda d: 50.0)
    store.record_universe(AS_OF, [{"ticker": "AAA", "cik": "1",
                                   "eligible": True}], recorded_at=STAMP)
    monkeypatch.setattr(scanner, "MAX_NAMES", 5)
    monkeypatch.setattr(scanner, "_signal_for", lambda t, a: {
        "ticker": t, "success": True, "error": None, "sue": 3.0,
        "fiscal_period": "2026Q2", "known_at": a, "sigma_quarters": 8,
        "sigma_periods": [], "basis_changes": [], "variant": "ts"})
    monkeypatch.setattr(scanner, "_cost_for", lambda t, a, d: {
        "cost": 0.0005, "cost_floor": 0.00002, "reason": None,
        "spread": 0.0001, "resolved": True, "resolution": "measured"})

    out = scanner.scan(as_of=AS_OF)

    assert out["candidates"], out["rejected"]
    for candidate in out["candidates"]:
        assert candidate["target_dollars"] <= (
            risk.MAX_POSITION_FRACTION * GROSS + 1e-6), candidate


# --- what the book is worth --------------------------------------------------

def test_equity_counts_closed_trades_at_net_and_open_ones_at_the_last_close(
        store):
    """AAA was bought at 100 and sold five sessions later at 110: +10%, less
    10bp, on 5,000. BBB was bought at 100 and closed last night at 90, still
    open: -10%, less the same 10bp."""
    _bars(spread.REFERENCE_TICKER, lambda d: 50.0)
    _bars("AAA", lambda d: 110.0 if d >= DAYS[286] else 100.0)
    _bars("BBB", lambda d: 90.0 if d >= DAYS[298] else 100.0)
    _order("AAA", filed=DAYS[280], session=DAYS[281])
    _order("BBB", filed=DAYS[296], session=DAYS[297])

    eq = risk.book_equity(AS_OF, horizon_days=5)

    assert eq["closed_trades"] == 1 and eq["open_trades"] == 1
    assert eq["realized_dollars"] == pytest.approx(5_000 * (0.10 - 0.001))
    assert eq["open_dollars"] == pytest.approx(5_000 * (-0.10 - 0.001))
    assert eq["equity_dollars"] == pytest.approx(
        eq["realized_dollars"] + eq["open_dollars"])


def test_an_order_whose_session_has_not_come_carries_no_exposure(store):
    _bars(spread.REFERENCE_TICKER, lambda d: 50.0)
    _bars("CCC", lambda d: 100.0)
    _order("CCC", filed=AS_OF, session="2099-01-02")

    eq = risk.book_equity(AS_OF, horizon_days=5)

    assert eq["open_trades"] == 0 and eq["equity_dollars"] == 0.0


def test_an_open_position_that_cannot_be_marked_is_counted_not_zeroed(store):
    """Filed and past its session, but the name has no bar yet: its exposure
    is unknown, which is different from nothing."""
    _bars(spread.REFERENCE_TICKER, lambda d: 50.0)
    _order("DDD", filed=DAYS[296], session=DAYS[297])

    eq = risk.book_equity(AS_OF, horizon_days=5)

    assert eq["unmarked_trades"] == 1


# --- the drawdown switch -----------------------------------------------------

def test_the_switch_trips_at_a_ten_percent_drawdown_from_the_peak(store):
    risk.check_and_record(DAYS[-3], GROSS, equity=_equity(5_000.0))
    state = risk.check_and_record(DAYS[-2], GROSS, equity=_equity(-5_000.0))

    assert state["halted"] is True
    assert state["drawdown_dollars"] == pytest.approx(10_000.0)
    assert "10%" in state["reason"]


def test_just_below_the_limit_the_switch_stays_armed(store):
    risk.check_and_record(DAYS[-3], GROSS, equity=_equity(5_000.0))
    state = risk.check_and_record(DAYS[-2], GROSS, equity=_equity(-4_999.0))

    assert state["halted"] is False


def test_a_tripped_switch_stays_tripped_when_equity_recovers(store):
    risk.check_and_record(DAYS[-4], GROSS, equity=_equity(5_000.0))
    risk.check_and_record(DAYS[-3], GROSS, equity=_equity(-5_000.0))
    state = risk.check_and_record(DAYS[-2], GROSS, equity=_equity(9_000.0))

    assert state["halted"] is True


def test_a_reset_rebases_the_peak_so_the_switch_does_not_trip_again_at_once(
        store):
    risk.check_and_record(DAYS[-4], GROSS, equity=_equity(5_000.0))
    risk.check_and_record(DAYS[-3], GROSS, equity=_equity(-5_000.0))
    risk.reset(DAYS[-2], reason="reviewed the book; the loss was one name")
    state = risk.check_and_record(DAYS[-1], GROSS, equity=_equity(-6_000.0))

    assert state["halted"] is False
    assert state["drawdown_dollars"] == pytest.approx(1_000.0)


def test_a_reset_needs_a_reason(store):
    with pytest.raises(SystemExit):
        risk.main(["--reset", "--as-of", AS_OF])


def test_the_scan_refuses_to_file_while_the_switch_is_tripped(store):
    risk.check_and_record(DAYS[-3], GROSS, equity=_equity(5_000.0))
    risk.check_and_record(DAYS[-2], GROSS, equity=_equity(-5_000.0))

    with pytest.raises(risk.RiskHalted):
        scanner.record_scan(as_of=AS_OF)

    assert pit_store.paper_orders_as_of(AS_OF) == []
    last = status._last_run("scan")
    assert last["status"] == "failed"
    assert "risk switch" in last["error"]


# --- the go/no-go gate -------------------------------------------------------

def test_the_gate_does_not_pass_without_measured_fills():
    gate = scoring.gate_check([])

    assert gate["passed"] is False
    assert "fills" in gate["reason"]


def test_the_gate_needs_two_hundred_measured_trades():
    gate = scoring.gate_check([80.0, 20.0] * 75)

    assert gate["passed"] is False
    assert "150" in gate["reason"] and "200" in gate["reason"]


def test_the_gate_passes_with_enough_trades_and_a_clear_positive_t():
    gate = scoring.gate_check([150.0, -50.0] * 125)

    assert gate["passed"] is True
    assert gate["t_stat"] > gate["t_threshold"]


def test_a_clear_negative_result_does_not_pass():
    gate = scoring.gate_check([50.0, -150.0] * 125)

    assert gate["passed"] is False


def test_the_gate_bar_is_the_one_for_six_variants():
    assert scoring.gate_check([1.0] * 3)["t_threshold"] == pytest.approx(
        scoring.t_threshold(6))
    assert scoring.t_threshold(6) > scoring.t_threshold(1)


def test_the_weekly_score_records_the_gate(store):
    scoring.score_orders(as_of=AS_OF)

    gate = pit_store.latest_gate_check(AS_OF)
    assert gate is not None
    assert gate["passed"] == 0
    assert "fills" in gate["reason"]


# --- what the status screen says --------------------------------------------

def test_a_tripped_switch_is_on_the_status_screen_and_needs_attention(store):
    risk.check_and_record(DAYS[-3], GROSS, equity=_equity(5_000.0))
    risk.check_and_record(DAYS[-2], GROSS, equity=_equity(-5_000.0))

    report = status.collect(as_of=AS_OF)
    screen = status._render(report)

    assert any(a.startswith("risk switch") for a in report["attention"])
    assert "TRIPPED" in screen


def test_an_armed_switch_and_the_gate_show_their_numbers(store):
    risk.check_and_record(DAYS[-2], GROSS, equity=_equity(2_000.0))
    scoring.score_orders(as_of=AS_OF)

    screen = status._render(status.collect(as_of=AS_OF))

    assert "RISK" in screen and "armed" in screen
    assert "GATE" in screen and "of 200" in screen
