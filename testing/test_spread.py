"""The number that decides whether the strategy is worth running at all.

Post-earnings drift is a real effect and a small one. The published estimates
put transaction costs at 63-100% of gross PEAD profits, which is to say the
median honest answer is "there is no trade here". That figure is market impact
measured at institutional size, and the entire thesis of trading the drift in a
small account is that a small order sits on a different part of the impact
curve -- but "a different part of the curve" is a claim about a number, and
this module is where that number gets produced.

So the failure that matters here is not a crash. It is a cost model that
flatters: one that returns 8 basis points where the truth is 80, backtests
beautifully, and loses money in the account. Every test below is aimed at one
of the specific ways that happens:

  A zero. The EDGE estimator can return a negative variance estimate on a
  short or degenerate sample, and the reference implementations clamp that to
  zero. A zero spread says trading is free. Here it has to be a refusal.

  A default. A name with too little history, or a window where the price never
  moved, must not quietly inherit some house-average spread. The backtest is
  allowed to skip a name; it is not allowed to price one it cannot measure.

  A silent sub-linear extrapolation. Impact is modelled square-root, which is
  calibrated for orders that are a small fraction of daily volume. Asked for
  ten times the size limit it returns a number three times too small, and
  returns it without complaint.

  Lookahead. The spread on the day of the trade must be computed from the bars
  we had on the day of the trade. `pit_store.bars_as_of` enforces that; a
  module that queries `daily_bar` directly to save a call does not.

The estimator is EDGE -- Ardia, Guidotti & Kroencke, Journal of Financial
Economics 2024 -- reimplemented here rather than imported, and the first test
in this file is the one that says the reimplementation is actually EDGE and
not forty lines of plausible algebra.
"""
import math
from datetime import date, timedelta

import numpy as np
import pytest

from research import pit_store, spread


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("NEMO_PIT_DB", str(tmp_path / "pit.db"))
    pit_store.init_schema()
    return pit_store


# --- helpers ----------------------------------------------------------------

def _sessions(n, start=date(2026, 1, 5)):
    """n consecutive weekday ISO dates. Calendar realism is irrelevant to the
    maths; only the ordering and the count matter."""
    out, day = [], start
    while len(out) < n:
        if day.weekday() < 5:
            out.append(day.isoformat())
        day += timedelta(days=1)
    return out


def _simulate_ohlc(spread_frac, days, sigma=0.02, trades_per_day=100,
                   seed=0, price0=100.0):
    """Daily OHLC from the model EDGE is derived under.

    An efficient log price does a random walk; every observed trade is that
    price pushed to one side of the market by half the spread, with the side a
    fair coin. Open is the first trade of the day, close the last, high and low
    the extremes. This is the only honest way to test a spread estimator: real
    bars do not come with the answer written on them.
    """
    rng = np.random.default_rng(seed)
    n = days * trades_per_day
    steps = rng.normal(0.0, sigma / math.sqrt(trades_per_day), n)
    efficient = math.log(price0) + np.cumsum(steps)
    side = rng.choice(np.array([-1.0, 1.0]), size=n)
    observed = np.exp(efficient + np.log1p(side * spread_frac / 2.0))
    observed = observed.reshape(days, trades_per_day)
    return {"open": observed[:, 0], "high": observed.max(axis=1),
            "low": observed.min(axis=1), "close": observed[:, -1]}


def _rows(dates, ohlc, volume):
    return [{"trade_date": d, "open": float(ohlc["open"][i]),
             "high": float(ohlc["high"][i]), "low": float(ohlc["low"][i]),
             "close": float(ohlc["close"][i]), "volume": volume}
            for i, d in enumerate(dates)]


def _record_simulated(store, ticker, spread_frac, days, volume=200_000.0,
                      seed=0, recorded_at=None, **kw):
    ohlc = _simulate_ohlc(spread_frac, days, seed=seed, **kw)
    dates = _sessions(days)
    store.record_bars(ticker, _rows(dates, ohlc, volume),
                      recorded_at=recorded_at or f"{dates[0]}T21:00:00Z")
    return dates


def _flat(store, ticker, days, price=10.0, volume=200_000.0, recorded_at=None):
    """A name whose price never moved. Degenerate for any spread estimator."""
    dates = _sessions(days)
    store.record_bars(ticker, [
        {"trade_date": d, "open": price, "high": price, "low": price,
         "close": price, "volume": volume} for d in dates],
        recorded_at=recorded_at or f"{dates[0]}T21:00:00Z")
    return dates


# --- is this actually EDGE? -------------------------------------------------
#
# The authors ship a 10,000-row simulated OHLC series with a published answer,
# plus a set of unit-test vectors in eguidotti/bidask. The full series is 660KB
# and does not belong in this repository; the first ten of its rows do, and
# they carry a published expected value of their own -- one that also happens
# to exercise the negative-variance branch, which is the branch this module
# treats differently from the reference. The full-series check is a separate
# @pytest.mark.network test below.
#
# Source: https://raw.githubusercontent.com/eguidotti/bidask/main/pseudocode/ohlc.csv

_REFERENCE_OHLC_URL = (
    "https://raw.githubusercontent.com/eguidotti/bidask/main/pseudocode/"
    "ohlc.csv")
_REFERENCE_FULL_EDGE = 0.0101849034905478  # published, all 10,000 rows

# Rows 1-10 of that file, verbatim.
_REFERENCE_HEAD = {
    "open": [1.00240075994025, 0.929743096395543, 0.943911414384193,
             0.977605378856722, 0.961535370407048, 0.982409003083975,
             1.03435413274433, 1.0201424300403, 1.03650094499406,
             1.05368509087896],
    "high": [1.00935878365403, 0.929774737723069, 0.968140794511406,
             0.977605378856722, 0.99982474278851, 0.992213437101346,
             1.03435413274433, 1.04025465277235, 1.03650094499406,
             1.05368509087896],
    "low": [0.976061138469904, 0.928919694164385, 0.943911414384193,
            0.977605378856722, 0.961535370407048, 0.982409003083975,
            1.03435413274433, 1.0201424300403, 1.03650094499406,
            1.02605242183949],
    "close": [0.976061138469904, 0.929774737723069, 0.968140794511406,
              0.977605378856722, 0.986215329667607, 0.992213437101346,
              1.03435413274433, 1.03219266262805, 1.03650094499406,
              1.02798086837104],
}
# Published in eguidotti/bidask python/tests/test_edge.py for these ten rows.
_REFERENCE_HEAD_SIGNED = -0.016889917516422


def test_edge_reproduces_the_published_reference_value():
    """The test that makes the other twenty mean something.

    EDGE is forty lines of index algebra with no intuition to check it
    against: a transposed subscript, a dropped de-meaning term, or the two
    inverse-variance weights applied to the wrong moments each produce a number
    in the right range that is simply wrong, and no downstream test would
    notice. The authors ship worked examples precisely so that an independent
    implementation can prove itself against them. If this fails, nothing else
    in this file is evidence of anything.
    """
    got = spread.edge(signed=True, **_REFERENCE_HEAD)
    assert got == pytest.approx(_REFERENCE_HEAD_SIGNED, rel=1e-12), (
        f"EDGE returned {got!r} where the published example gives "
        f"{_REFERENCE_HEAD_SIGNED!r}; this is not the estimator in the paper")


def test_edge_reproduces_the_published_degenerate_case():
    """The authors' own NaN vector. Three bars that between them contain no
    evidence of a spread; the estimator must say so rather than produce a
    number from an empty mean."""
    assert spread.edge([18.21, 17.61, 17.61], [18.21, 17.61, 17.61],
                       [17.61, 17.61, 17.61], [17.61, 17.61, 17.61]) is None


@pytest.mark.network
def test_edge_reproduces_the_full_published_series():
    """The headline number from the paper's own worked example, over all
    10,000 rows. Network-gated because the series is 660KB and belongs to the
    authors rather than in this repository; the ten-row vector above is the
    offline guarantee."""
    import csv
    import io
    import urllib.request

    with urllib.request.urlopen(_REFERENCE_OHLC_URL, timeout=60) as response:
        rows = list(csv.DictReader(io.StringIO(response.read().decode())))

    got = spread.edge(*[[float(r[k]) for r in rows]
                        for k in ("Open", "High", "Low", "Close")])
    assert got == pytest.approx(_REFERENCE_FULL_EDGE, rel=1e-12)


def test_edge_recovers_a_spread_it_was_never_shown():
    """Reproducing one published constant proves the arithmetic was copied
    correctly. It does not prove the estimator estimates anything. Simulating
    a market with a known spread and getting that spread back does."""
    for true_spread in (0.005, 0.02):
        ohlc = _simulate_ohlc(true_spread, days=8000, seed=7)
        got = spread.edge(ohlc["open"], ohlc["high"], ohlc["low"],
                          ohlc["close"])
        assert got == pytest.approx(true_spread, rel=0.10), (
            f"a market simulated with a {true_spread:.3%} spread estimated at "
            f"{got:.4%}")


def test_edge_orders_two_markets_by_their_true_spread():
    """The property the whole strategy leans on. An absolute estimate that is
    30% off is survivable; one that ranks a wide market as tighter than a
    narrow one routes every trade to the wrong names."""
    wide = _simulate_ohlc(0.03, days=4000, seed=11)
    narrow = _simulate_ohlc(0.003, days=4000, seed=11)
    assert spread.edge(**narrow) < spread.edge(**wide)


def test_the_reported_standard_error_describes_the_real_sampling_error():
    """Everything pessimistic in this module rests on this one number.

    The cost a backtest subtracts is the upper confidence bound, and a bound
    is only conservative if the standard error under it is honest. An
    understated error produces a bound that looks careful and is not, which is
    the same failure as a flattering point estimate wearing a disguise.

    So: simulate many markets with a known spread, and check that the error
    the estimator reports for itself matches the spread of the estimates it
    actually produces, and that intervals built from it cover the truth at
    their stated rate.
    """
    true_spread, days, trials = 0.01, 60, 150
    estimates, errors, covered = [], [], 0
    for seed in range(trials):
        m = spread.edge_moments(**_simulate_ohlc(true_spread, days, seed=seed))
        estimates.append(m["s2"])
        errors.append(m["standard_error"])
        if abs(m["s2"] - true_spread ** 2) <= 1.96 * m["standard_error"]:
            covered += 1

    observed = float(np.std(estimates, ddof=1))
    reported = float(np.mean(errors))
    assert reported == pytest.approx(observed, rel=0.15), (
        f"the estimator reports a standard error of {reported:.3e} while its "
        f"estimates actually scatter by {observed:.3e}")
    assert 0.88 <= covered / trials <= 1.0, (
        f"95% intervals covered the true spread {covered / trials:.0%} of the "
        f"time")
    assert float(np.mean(estimates)) == pytest.approx(true_spread ** 2,
                                                      rel=0.10), (
        "the estimator is biased, so no confidence bound built on it is valid")


def test_edge_refuses_a_price_series_that_never_moves():
    """Every bar identical is not a zero spread, it is no information. The
    reference implementation returns NaN here and this one must not turn that
    into a number."""
    flat = np.full(200, 10.0)
    assert spread.edge(flat, flat, flat, flat) is None


# --- refusals rather than plausible numbers ---------------------------------

def test_too_few_sessions_is_a_refusal_with_a_reason(store):
    """EDGE is consistent, not exact: on a handful of bars its sampling error
    is the same order as the quantity being measured. A number produced from
    twelve sessions is noise wearing a decimal point."""
    _record_simulated(store, "THIN", 0.01, days=spread.MIN_SESSIONS - 1)

    got = spread.estimate_spread("THIN", as_of="2026-12-31")
    assert got["spread"] is None
    assert got["reason"] and "session" in got["reason"].lower()
    assert str(spread.MIN_SESSIONS) in got["reason"]


def test_a_name_we_have_never_seen_is_a_refusal_not_a_zero(store):
    got = spread.estimate_spread("NOSUCH", as_of="2026-12-31")
    assert got["spread"] is None
    assert got["reason"]


def test_a_flat_window_refuses_rather_than_reporting_a_free_market(store):
    """The most dangerous single output this module could produce. A name that
    printed the same price for sixty sessions is a name that barely traded;
    reporting its spread as 0.0 tells the backtest to trade it without limit."""
    _flat(store, "FROZEN", days=90)

    got = spread.estimate_spread("FROZEN", as_of="2026-12-31")
    assert got["spread"] is None, "a frozen price was priced as costless"
    assert got["reason"]


def test_a_negative_variance_estimate_is_refused_not_clamped_to_zero(store):
    """EDGE estimates a *squared* spread, and in a short sample that estimate
    comes out negative often enough to matter.

    The paper reports sqrt(max(0, S^2)) for comparability with the earlier
    literature and the reference code returns sqrt(|S^2|). Both are reasonable
    for a research library and neither is safe here. Zero is the single most
    flattering number this module could emit -- it says trading is free.
    sqrt(|S^2|) is a point estimate whose sign the estimator itself got wrong,
    and the authors warn it biases anything built by averaging. So: refuse,
    and hand the caller the tick floor to make its own decision with.
    """
    for seed in range(60):
        ohlc = _simulate_ohlc(0.0, days=40, seed=seed)
        if spread.edge(**ohlc, signed=True) < 0:
            break
    else:  # pragma: no cover - the loop finds a case in the first few seeds
        pytest.fail("no seed produced a negative variance estimate to test")

    # The estimator itself stays faithful to the published reference.
    assert spread.edge(**ohlc) > 0

    dates = _sessions(40)
    store.record_bars("NOSPREAD", _rows(dates, ohlc, 200_000.0),
                      recorded_at=f"{dates[0]}T21:00:00Z")

    got = spread.estimate_spread("NOSPREAD", as_of="2026-12-31", window=40)
    assert got["spread"] is None, (
        "a negative variance estimate became a tradeable spread")
    assert got["signed_estimate"] < 0
    assert got["tick_floor"] > 0, "refused without offering anything checkable"
    assert got["reason"]


def test_a_non_positive_price_never_reaches_a_logarithm(store):
    """EDGE works in log prices. A zero or negative print -- a vendor
    placeholder, a bad tick -- produces -inf or NaN, and NaN arithmetic
    propagates silently into a mean. Those sessions have to be excluded and
    counted, and if too few survive the answer is a refusal."""
    dates = _sessions(90)
    ohlc = _simulate_ohlc(0.01, days=90, seed=3)
    rows = _rows(dates, ohlc, volume=200_000.0)
    for i in (10, 11, 12):
        rows[i]["low"] = 0.0
    store.record_bars("BADTICK", rows, recorded_at=f"{dates[0]}T21:00:00Z")

    got = spread.estimate_spread("BADTICK", as_of="2026-12-31", window=90)
    assert got["spread"] is not None
    assert got["sessions_excluded"] == 3
    assert math.isfinite(got["spread"])


def test_a_window_mostly_made_of_bad_prices_is_refused(store):
    dates = _sessions(90)
    ohlc = _simulate_ohlc(0.01, days=90, seed=3)
    rows = _rows(dates, ohlc, volume=200_000.0)
    for row in rows[:80]:
        row["close"] = None
    store.record_bars("MOSTLYBAD", rows, recorded_at=f"{dates[0]}T21:00:00Z")

    got = spread.estimate_spread("MOSTLYBAD", as_of="2026-12-31", window=90)
    assert got["spread"] is None
    assert got["reason"] and "usable" in got["reason"].lower()


def test_a_window_with_no_volume_is_refused(store):
    """Prices without trades are quotes, or worse, a vendor carrying the last
    print forward. A spread estimated off a price path nobody transacted on
    describes a market that did not exist."""
    dates = _sessions(90)
    ohlc = _simulate_ohlc(0.01, days=90, seed=5)
    store.record_bars("NOTRADE", _rows(dates, ohlc, volume=0.0),
                      recorded_at=f"{dates[0]}T21:00:00Z")

    got = spread.estimate_spread("NOTRADE", as_of="2026-12-31", window=90)
    assert got["spread"] is None
    assert got["reason"] and "volume" in got["reason"].lower()


# --- the tick floor ---------------------------------------------------------

def test_the_estimate_never_goes_below_one_tick(store):
    """A US equity cannot be quoted tighter than a penny, so on a $6 stock no
    effective spread below ~1.7bp is physically reachable. EDGE is a
    statistical estimate and will sometimes return less; taking it at face
    value hands the backtest a fill no venue would give. The floor is where
    pessimism costs nothing, because it is also the truth."""
    _record_simulated(store, "PENNYISH", 0.0005, days=200, price0=6.0,
                      sigma=0.01, seed=0)

    got = spread.estimate_spread("PENNYISH", as_of="2026-12-31", window=200)
    assert 0 < got["signed_estimate"] < got["tick_floor"], (
        "the fixture no longer produces an estimate below the floor")
    assert got["spread"] == pytest.approx(spread.TICK_SIZE / got["last_price"])
    assert got["floored_at_tick"] is True


# --- point in time ----------------------------------------------------------

def test_the_estimate_sees_only_bars_known_on_the_as_of_date(store):
    """The lookahead this whole record exists to prevent, restated for cost.
    A spread computed on the day of the trade must use the bars we had on the
    day of the trade -- not the ones back-filled a month later, which is
    exactly when a name's post-event bars would arrive."""
    _record_simulated(store, "LATE", 0.01, days=200,
                      recorded_at="2026-11-02T21:00:00Z")

    early = spread.estimate_spread("LATE", as_of="2026-06-01", window=200)
    assert early["spread"] is None, (
        "bars recorded in November were visible to a June estimate")
    assert early["reason"] and "history" in early["reason"]

    late = spread.estimate_spread("LATE", as_of="2026-11-03", window=200)
    assert late["spread"] is not None


def test_a_raw_split_does_not_become_a_giant_spread(store):
    """Where two of this project's own design decisions collide.

    `pit_store` keeps prices raw on purpose, because adjusted closes are
    rewritten under you every time a new action lands. EDGE compares adjacent
    sessions. So an unadjusted 10-for-1 split is a -90% print between two
    neighbouring bars, and the estimator reads it as trading cost: left alone
    it turns a 1% spread into something like 20%, which would make a perfectly
    tradeable name look untouchable.

    Only the transitions touching the ex-date are affected -- every term in
    EDGE is a difference between adjacent bars, so a level shift cannot
    contaminate the rest of the window. Dropping that one session is enough,
    and it is counted rather than hidden.
    """
    dates = _sessions(200)
    ohlc = _simulate_ohlc(0.01, days=200, seed=4, price0=10.0)
    clean = _rows(dates, ohlc, volume=200_000.0)

    raw = [dict(r) for r in clean]
    for row in raw[:100]:  # pre-split prices, ten times higher
        for field in ("open", "high", "low", "close"):
            row[field] *= 10.0
        row["volume"] /= 10.0

    store.record_bars("CLEAN", clean, recorded_at=f"{dates[0]}T21:00:00Z")
    store.record_bars("SPLIT", raw, recorded_at=f"{dates[0]}T21:00:00Z")
    store.record_corporate_action("SPLIT", dates[100], "split", 10.0,
                                  recorded_at=f"{dates[100]}T12:00:00Z")

    reference = spread.estimate_spread("CLEAN", as_of="2026-12-31", window=200)
    got = spread.estimate_spread("SPLIT", as_of="2026-12-31", window=200)

    assert got["splits_excluded"] == 1
    assert got["spread"] == pytest.approx(reference["spread"], rel=0.05), (
        f"a 10-for-1 split moved the estimate from {reference['spread']:.4f} "
        f"to {got['spread']:.4f}")


def test_a_split_recorded_later_does_not_alter_an_earlier_estimate(store):
    """The as-of discipline again. A split we had not yet learned about cannot
    be corrected for, and pretending otherwise would make a past estimate
    depend on a future filing."""
    dates = _sessions(200)
    ohlc = _simulate_ohlc(0.01, days=200, seed=4, price0=10.0)
    rows = _rows(dates, ohlc, volume=200_000.0)
    store.record_bars("SPLIT", rows, recorded_at=f"{dates[0]}T21:00:00Z")
    store.record_corporate_action("SPLIT", dates[100], "split", 10.0,
                                  recorded_at="2026-12-01T12:00:00Z")

    before = spread.estimate_spread("SPLIT", as_of=dates[150], window=200)
    after = spread.estimate_spread("SPLIT", as_of="2026-12-02", window=200)
    assert before["splits_excluded"] == 0
    assert after["splits_excluded"] == 1


def test_a_self_contradictory_bar_is_dropped(store):
    """A close outside its own high-low range is a vendor error, and the
    estimator has no way to know that: it reads the impossible range as
    trading cost. The paper's own sample construction drops these."""
    dates = _sessions(200)
    ohlc = _simulate_ohlc(0.01, days=200, seed=6)
    rows = _rows(dates, ohlc, volume=200_000.0)
    rows[50]["close"] = rows[50]["high"] * 3
    rows[51]["low"] = rows[51]["high"] * 2

    store.record_bars("BADBAR", rows, recorded_at=f"{dates[0]}T21:00:00Z")
    got = spread.estimate_spread("BADBAR", as_of="2026-12-31", window=200)
    assert got["sessions_excluded"] == 2
    assert got["spread"] == pytest.approx(0.01, rel=0.25)


def test_the_window_is_the_sessions_before_as_of_not_the_whole_history(store):
    _record_simulated(store, "LONG", 0.01, days=400)

    got = spread.estimate_spread("LONG", as_of="2026-12-31", window=60)
    assert got["sessions_used"] <= 60


# --- position size against liquidity ----------------------------------------

def test_participation_is_the_position_against_median_dollar_volume(store):
    _record_simulated(store, "MID", 0.01, days=90, volume=10_000.0,
                      price0=50.0, sigma=0.01)

    got = spread.participation_rate("MID", as_of="2026-12-31",
                                    position_dollars=25_000.0, window=90)
    # 10,000 shares a day around $50 is roughly $500k of median dollar volume.
    assert got["median_dollar_volume"] == pytest.approx(500_000.0, rel=0.15)
    assert got["participation"] == pytest.approx(
        25_000.0 / got["median_dollar_volume"])


def test_the_liquidity_cap_names_the_largest_allowed_position(store):
    """A rule nobody can compute is a rule nobody follows. The caller should
    be able to ask how big it is allowed to be, not only be told afterwards
    that it was too big."""
    _record_simulated(store, "MID", 0.01, days=90, volume=10_000.0,
                      price0=50.0, sigma=0.01)

    got = spread.participation_rate("MID", as_of="2026-12-31",
                                    position_dollars=1.0, window=90)
    assert got["max_position_dollars"] == pytest.approx(
        got["median_dollar_volume"] * spread.MAX_PARTICIPATION)
    assert got["within_limit"] is True


def test_a_position_past_the_cap_is_flagged(store):
    _record_simulated(store, "MID", 0.01, days=90, volume=10_000.0,
                      price0=50.0, sigma=0.01)

    got = spread.participation_rate("MID", as_of="2026-12-31",
                                    position_dollars=250_000.0, window=90)
    assert got["within_limit"] is False
    assert got["warning"] and "1" in got["warning"]


def test_no_volume_means_no_participation_number(store):
    """Dividing by a median dollar volume of zero is either a crash or an
    infinity, and an infinity that reaches a backtest becomes a NaN return."""
    dates = _sessions(90)
    ohlc = _simulate_ohlc(0.01, days=90, seed=5)
    store.record_bars("NOTRADE", _rows(dates, ohlc, volume=0.0),
                      recorded_at=f"{dates[0]}T21:00:00Z")

    got = spread.participation_rate("NOTRADE", as_of="2026-12-31",
                                    position_dollars=1000.0, window=90)
    assert got["participation"] is None
    assert got["reason"]


def test_a_nonsense_position_is_an_error_not_a_reason(store):
    """A bad argument is a bug in the caller, and returning it a polite
    reason-carrying dict lets the bug travel. Bad *data* gets a reason; a bad
    *call* gets an exception."""
    _record_simulated(store, "MID", 0.01, days=90)
    for bad in (0.0, -1000.0):
        with pytest.raises(ValueError):
            spread.round_trip_cost("MID", as_of="2026-12-31",
                                   position_dollars=bad)


# --- the round trip ---------------------------------------------------------

def test_a_round_trip_pays_the_half_spread_twice(store):
    """In and out. Charging one half-spread for the pair is the single easiest
    way to halve a cost model, and it is invisible in the output."""
    _record_simulated(store, "MID", 0.02, days=200, volume=100_000.0,
                      price0=50.0)

    est = spread.estimate_spread("MID", as_of="2026-12-31", window=200)
    cost = spread.round_trip_cost("MID", as_of="2026-12-31",
                                  position_dollars=100.0, window=200,
                                  basis="point")
    assert cost["spread_cost"] == pytest.approx(est["spread"])
    assert cost["spread_cost"] == pytest.approx(2 * cost["half_spread"])


def test_the_cost_charges_the_upper_bound_by_default(store):
    """Measured against live bars: over 60 sessions EDGE's sampling error is
    larger than the entire spread of a large cap, so a point estimate is a
    coin flip that happens to have landed somewhere. Charging the bound is the
    difference between a cost model that is pessimistic and one that is
    merely noisy in a direction nobody checked."""
    _record_simulated(store, "MID", 0.02, days=200, volume=100_000.0,
                      price0=50.0)

    est = spread.estimate_spread("MID", as_of="2026-12-31", window=200)
    default = spread.round_trip_cost("MID", as_of="2026-12-31",
                                     position_dollars=100.0, window=200)
    assert default["spread_basis"] == "upper"
    assert default["spread_used"] == pytest.approx(est["spread_upper"])
    assert est["spread_upper"] > est["spread"], (
        "the upper bound is not above the point estimate")


def test_an_unresolved_name_still_gets_a_cost_and_a_warning(store):
    """The case that made this design necessary. A name whose spread the
    window cannot resolve is not a name that trades for free, and it is not a
    name the backtest should silently drop either -- the bound is a real
    number and it is the honest one. What must not happen is the bound being
    handed over without saying it is a bound."""
    for seed in range(60):
        ohlc = _simulate_ohlc(0.0, days=40, seed=seed)
        if spread.edge(**ohlc, signed=True) < 0:
            break
    else:  # pragma: no cover
        pytest.fail("no seed produced an unresolved sample")

    dates = _sessions(40)
    store.record_bars("FOGGY", _rows(dates, ohlc, 200_000.0),
                      recorded_at=f"{dates[0]}T21:00:00Z")

    est = spread.estimate_spread("FOGGY", as_of="2026-12-31", window=40)
    assert est["spread"] is None and est["resolved"] is False
    assert est["spread_upper"] > 0

    on_bound = spread.round_trip_cost("FOGGY", as_of="2026-12-31",
                                      position_dollars=100.0, window=40)
    assert on_bound["cost"] is not None
    assert on_bound["spread_used"] == pytest.approx(est["spread_upper"])

    on_point = spread.round_trip_cost("FOGGY", as_of="2026-12-31",
                                      position_dollars=100.0, window=40,
                                      basis="point")
    assert on_point["cost"] is None
    assert on_point["reason"]


def test_a_resolved_estimate_says_so_and_an_unresolved_one_warns(store):
    """`resolved` is the flag that stops a 60-day number being mistaken for a
    measurement. Without it the caller cannot tell 52bp of spread from 52bp of
    standard error, and the two look identical in a dataframe."""
    _record_simulated(store, "SHARP", 0.03, days=400, volume=100_000.0,
                      price0=50.0, seed=2)
    sharp = spread.estimate_spread("SHARP", as_of="2027-12-31", window=400)
    assert sharp["resolved"] is True
    assert sharp["warning"] is None
    assert sharp["standard_error"] > 0

    _record_simulated(store, "FUZZY", 0.001, days=25, volume=100_000.0,
                      price0=50.0, sigma=0.04, seed=3)
    fuzzy = spread.estimate_spread("FUZZY", as_of="2027-12-31", window=25)
    assert fuzzy["resolved"] is False
    assert fuzzy["warning"] and str(spread.RESOLVING_WINDOW) in fuzzy["warning"]


def test_the_upper_bound_tightens_as_the_window_grows(store):
    """The property that makes the bound usable rather than merely safe: it is
    wide when we know little and converges on the estimate when we know more,
    so a cost charged on it stops being dominated by measurement as the record
    accumulates."""
    _record_simulated(store, "GROW", 0.02, days=1200, volume=100_000.0,
                      price0=50.0, seed=5)

    short = spread.estimate_spread("GROW", as_of="2032-12-31", window=40)
    long = spread.estimate_spread("GROW", as_of="2032-12-31", window=1200)
    assert short["spread_upper"] > long["spread_upper"]
    assert long["spread_upper"] == pytest.approx(0.02, rel=0.25)


def test_two_caveats_do_not_cancel_each_other_out(store):
    """An oversized position in a name whose spread is unresolved is two
    separate problems. Reporting only the size one -- because it was written
    last -- drops precisely the caveat a reader is least likely to reconstruct,
    and leaves a cost that looks merely large rather than large and unmeasured.
    """
    dates = _sessions(30)
    for seed in range(60):
        # A fresh ticker each time: the store is append-only by design, so
        # re-recording over a rejected sample would silently keep the first.
        ticker = f"BOTH{seed}"
        ohlc = _simulate_ohlc(0.004, days=30, sigma=0.05, seed=seed)
        store.record_bars(ticker, _rows(dates, ohlc, 1_000.0),
                          recorded_at=f"{dates[0]}T21:00:00Z")
        est = spread.estimate_spread(ticker, as_of="2026-12-31", window=30)
        if est["spread"] is not None and not est["resolved"]:
            break
    else:  # pragma: no cover
        pytest.fail("no seed produced an unresolved positive estimate")

    cap = spread.participation_rate(ticker, as_of="2026-12-31",
                                    position_dollars=1.0,
                                    window=30)["max_position_dollars"]
    cost = spread.round_trip_cost(ticker, as_of="2026-12-31",
                                  position_dollars=cap * 20, window=30)

    assert cost["exceeds_liquidity_limit"] is True
    assert "resolve" in cost["warning"], "the unresolved-spread caveat was lost"
    assert "limit" in cost["warning"], "the oversize caveat was lost"


def test_a_bad_basis_is_an_error(store):
    _record_simulated(store, "MID", 0.02, days=200)
    with pytest.raises(ValueError):
        spread.round_trip_cost("MID", as_of="2026-12-31",
                               position_dollars=100.0, window=200,
                               basis="optimistic")


def test_the_round_trip_costs_more_than_the_spread_alone(store):
    """Impact is not optional. A model that charges only the spread says an
    order of any size fills at the quote."""
    _record_simulated(store, "MID", 0.02, days=200, volume=100_000.0,
                      price0=50.0)

    cost = spread.round_trip_cost("MID", as_of="2026-12-31",
                                  position_dollars=50_000.0, window=200)
    assert cost["impact_cost"] > 0
    assert cost["cost"] > cost["spread_cost"]
    assert cost["cost"] == pytest.approx(
        cost["spread_cost"] + cost["impact_cost"])


def test_cost_rises_with_size(store):
    _record_simulated(store, "MID", 0.02, days=200, volume=100_000.0,
                      price0=50.0)

    small = spread.round_trip_cost("MID", as_of="2026-12-31",
                                   position_dollars=1_000.0, window=200)
    large = spread.round_trip_cost("MID", as_of="2026-12-31",
                                   position_dollars=100_000.0, window=200)
    assert large["cost"] > small["cost"]


def test_past_the_cap_impact_grows_faster_than_the_square_root(store):
    """The square-root law is calibrated on orders that are a small slice of
    the day. Extrapolated to a third of the day's volume it says ten times the
    size costs three times the impact, which is the sort of number that makes
    an illiquid strategy look scalable. Past the cap the model has no evidence,
    so it charges linearly -- the pessimistic reading -- and says so.
    """
    _record_simulated(store, "MID", 0.02, days=200, volume=100_000.0,
                      price0=50.0)
    at_cap = spread.participation_rate(
        "MID", as_of="2026-12-31", position_dollars=1.0,
        window=200)["max_position_dollars"]

    on_limit = spread.round_trip_cost("MID", as_of="2026-12-31",
                                      position_dollars=at_cap, window=200)
    over = spread.round_trip_cost("MID", as_of="2026-12-31",
                                  position_dollars=at_cap * 10, window=200)

    assert on_limit["exceeds_liquidity_limit"] is False
    assert over["exceeds_liquidity_limit"] is True
    assert over["warning"]
    ratio = over["impact_cost"] / on_limit["impact_cost"]
    assert ratio == pytest.approx(10.0, rel=0.01), (
        f"ten times the size cost {ratio:.2f}x the impact; the square root "
        f"law was extrapolated past where it was measured")


def test_the_cost_is_continuous_at_the_cap(store):
    """A step change at the boundary would let a caller shave a cent off the
    order and save a visible amount of modelled cost, which is an artefact of
    the model rather than of the market."""
    _record_simulated(store, "MID", 0.02, days=200, volume=100_000.0,
                      price0=50.0)
    at_cap = spread.participation_rate(
        "MID", as_of="2026-12-31", position_dollars=1.0,
        window=200)["max_position_dollars"]

    just_under = spread.round_trip_cost("MID", as_of="2026-12-31",
                                        position_dollars=at_cap * 0.999,
                                        window=200)["cost"]
    just_over = spread.round_trip_cost("MID", as_of="2026-12-31",
                                       position_dollars=at_cap * 1.001,
                                       window=200)["cost"]
    assert just_over == pytest.approx(just_under, rel=0.01)
    assert just_over > just_under


def test_a_refused_spread_refuses_the_cost(store):
    """No house average, no last-known value, no zero. If the spread could not
    be measured the trade cannot be priced, and the backtest has to drop the
    name rather than assume one."""
    _flat(store, "FROZEN", days=200)

    cost = spread.round_trip_cost("FROZEN", as_of="2026-12-31",
                                  position_dollars=1_000.0, window=200)
    assert cost["cost"] is None
    assert cost["reason"]


def test_a_wider_spread_costs_more_at_the_same_size(store):
    """The ordering the strategy is built on, end to end."""
    _record_simulated(store, "TIGHT", 0.002, days=300, volume=100_000.0,
                      price0=50.0, seed=1)
    _record_simulated(store, "WIDE", 0.03, days=300, volume=100_000.0,
                      price0=50.0, seed=1)

    tight = spread.round_trip_cost("TIGHT", as_of="2026-12-31",
                                   position_dollars=10_000.0, window=300)
    wide = spread.round_trip_cost("WIDE", as_of="2026-12-31",
                                  position_dollars=10_000.0, window=300)
    assert wide["cost"] > tight["cost"]


def test_the_cost_report_carries_its_inputs(store):
    """A cost the backtest cannot explain afterwards is a cost nobody will
    trust when the strategy underperforms. Every number that went in comes
    back out."""
    _record_simulated(store, "MID", 0.02, days=200, volume=100_000.0,
                      price0=50.0)

    cost = spread.round_trip_cost("MID", as_of="2026-12-31",
                                  position_dollars=10_000.0, window=200)
    for key in ("spread", "half_spread", "spread_cost", "impact_cost", "cost",
                "participation", "median_dollar_volume", "daily_volatility",
                "sessions_used", "as_of", "ticker"):
        assert key in cost, f"the cost report does not say {key}"
    assert cost["as_of"] == "2026-12-31"
    assert cost["ticker"] == "MID"
