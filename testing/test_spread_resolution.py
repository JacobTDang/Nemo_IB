"""What EDGE can and cannot see, measured against a known answer.

The estimator is implemented correctly -- it reproduces all four published
vectors to fifteen significant figures. The question this file settles is a
different one: whether daily OHLC carries enough information to measure the
spread of a liquid stock at all.

It does not, and the evidence is SPY. Its quoted spread is one cent on a ~$600
price, which is 0.17 basis points, essentially always and for years at a time.
EDGE on the store's own SPY history returns 22bp at a 60-session window, 27bp
at 252, 41bp at 504 and 37bp at 1008 -- between 130 and 240 times the true
value. Worse, at the longer windows it reports `resolved=True`, because
`resolved` asks whether the estimate is distinguishable from zero and a biased
estimator with a tight standard error passes that test easily.

So a longer window does not fix it. It removes the warning.

What follows from that is the useful part. If a name's estimate cannot be
distinguished from the reference instrument's, then whatever its true spread
is, it is below what daily bars can resolve -- and for a name that liquid the
honest estimate is the tick, not the estimator's noise.
"""
import pytest

from research import pit_store, spread


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("NEMO_PIT_DB", str(tmp_path / "pit.db"))
    pit_store.init_schema()
    return pit_store


def _series(store, ticker, n=300, price=600.0, wobble=0.004, seed=1,
            spread_bp=0.0):
    """A price path with a controllable amount of bid-ask bounce in it."""
    import math

    from datetime import date, timedelta

    rows, d, p = [], date(2025, 1, 1), price
    half = price * spread_bp / 20_000.0
    i = 0
    while len(rows) < n:
        if d.weekday() < 5:
            # Deterministic pseudo-random walk; no Math.random equivalent
            # needed and the series is reproducible.
            step = math.sin(i * 1.7 + seed) * wobble
            p = max(1.0, p * (1 + step))
            bounce = half if i % 2 == 0 else -half
            rows.append({
                "trade_date": d.isoformat(),
                "open": p, "close": p + bounce,
                "high": p * (1 + abs(step)) + half,
                "low": p * (1 - abs(step)) - half,
                "volume": 5_000_000,
            })
            i += 1
        d += timedelta(days=1)
    store.record_bars(ticker, rows, recorded_at=f"{rows[-1]['trade_date']}T21:00:00Z")
    return rows[-1]["trade_date"]


def test_the_reference_instrument_is_the_yardstick(store):
    """It has to be in the store for any of this to work, and it is: the
    recorder writes it every night as the liveness canary."""
    assert spread.REFERENCE_TICKER


def test_a_name_indistinguishable_from_the_reference_is_priced_at_the_tick(
        store):
    """Both series carry the same (near-zero) bounce, so EDGE sees the same
    noise in each. That is not a measurement of a 40bp spread, it is the
    estimator's floor -- and the honest number for such a name is the tick."""
    as_of = _series(store, spread.REFERENCE_TICKER, spread_bp=0.0, seed=1)
    _series(store, "LIQUID", spread_bp=0.0, seed=1)

    out = spread.spread_basis("LIQUID", as_of, window=252)

    assert out["basis"] == "at_resolution_floor"
    assert out["spread"] == pytest.approx(out["tick_floor"])
    assert "reference" in out["reason"].lower() or \
        "resolve" in out["reason"].lower()


def test_a_genuinely_wide_name_keeps_its_measurement(store):
    """The other side. A name whose bounce is far above the estimator's floor
    is being measured, not guessed, and must not be flattened to a tick."""
    as_of = _series(store, spread.REFERENCE_TICKER, spread_bp=0.0, seed=1)
    _series(store, "WIDE", price=6.0, spread_bp=400.0, seed=1)

    out = spread.spread_basis("WIDE", as_of, window=252)

    assert out["basis"] == "measured", out["reason"]
    assert out["spread"] > out["tick_floor"] * 2


def test_the_reference_itself_is_priced_at_its_tick(store):
    """Not a special case -- it falls out of the same rule, and it had better,
    because SPY at 40bp is the absurdity that started this."""
    as_of = _series(store, spread.REFERENCE_TICKER, spread_bp=0.0, seed=1)

    out = spread.spread_basis(spread.REFERENCE_TICKER, as_of, window=252)
    assert out["basis"] == "at_resolution_floor"


def test_a_missing_reference_refuses_rather_than_guessing(store):
    """Without the yardstick there is no way to tell a measurement from the
    estimator's noise, and picking one silently is how 40bp becomes a fact."""
    _series(store, "LIQUID", spread_bp=0.0, seed=1)
    as_of = _series(store, "OTHER", spread_bp=0.0, seed=2)

    out = spread.spread_basis("LIQUID", as_of, window=252)
    assert out["basis"] == "unknown"
    assert out["spread"] is None
    assert spread.REFERENCE_TICKER in out["reason"]


def test_the_cost_model_can_use_the_adaptive_basis(store):
    as_of = _series(store, spread.REFERENCE_TICKER, spread_bp=0.0, seed=1)
    _series(store, "LIQUID", spread_bp=0.0, seed=1)

    adaptive = spread.round_trip_cost("LIQUID", as_of, 5_000.0, window=252,
                                      basis="adaptive")
    upper = spread.round_trip_cost("LIQUID", as_of, 5_000.0, window=252,
                                   basis="upper")

    assert adaptive["cost"] < upper["cost"], (
        "the adaptive basis charged as much as the 95% bound")
    assert adaptive["spread_basis"] == "adaptive"
    assert adaptive["resolution"] == "at_resolution_floor"
