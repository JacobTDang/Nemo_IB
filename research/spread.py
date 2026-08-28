"""What it costs to get in and back out, estimated from bars we actually had.

This module exists because of one number in the literature: transaction costs
consume 63-100% of gross post-earnings-drift profits. That figure is market
impact measured at institutional size, and the whole premise of trading the
drift in a small account is that a small order sits somewhere else on the
impact curve. "Somewhere else on the curve" is a claim about a number, so the
cost model is not a detail of the backtest -- it is the thing the backtest is
for. A model that flatters produces a strategy that looks profitable on paper
and loses money in the account, which is strictly worse than no model at all.

**The spread estimator is EDGE.** Ardia, Guidotti & Kroencke, "Efficient
estimation of bid-ask spreads from open, high, low, and close prices", Journal
of Financial Economics 161 (2024) 103916. It recovers the effective spread from
daily OHLC alone, beats Corwin-Schultz and Roll, and beats them by the widest
margin in exactly the illiquid names this strategy lives in. It is
reimplemented here rather than imported: the estimator is closed-form and about
forty lines against numpy, and the reimplementation is checked against the
authors' own published worked example in testing/test_spread.py. If that test
is not passing, nothing in this module is trustworthy.

**The point estimate is not the number to trade on.** Measured here against
live data (see testing/test_spread.py and the note below), EDGE's sampling
error over a 60-session window is around 40-60bp for a typical US equity. That
is larger than the entire true spread of a large cap, so a 60-day point
estimate of "52bp" for JPM is not a spread, it is a noise floor with a decimal
point. The estimator is unbiased and its standard error is computable from
quantities it already produces, so this module reports both, flags whether the
estimate is distinguishable from zero at all, and makes `round_trip_cost`
charge the **upper confidence bound** by default. The bound is stable across
windows, never negative, never zero, and tightens as the sample grows -- which
is the correct behaviour for a cost that is uncertain.

Measured over 14 live names on 2026-08-27, window=252, ranking against median
dollar volume: the upper bound has a Spearman correlation with illiquidity of
+0.94 and separates cleanly by tier (mega caps and SPY 22-54bp, mid and small
86-116bp, micro and nano 160-458bp). The point estimate manages +0.50 over the
same names, and returns a negative squared spread for AAPL, MSFT, SHOO, CRI and
UUU. Same estimator, same data; the difference is entirely whether the
uncertainty is carried or discarded.

Four other decisions are deliberately less convenient than the reference
implementation, because a research library and a trading cost model want
different defaults:

**A non-positive variance estimate is a refusal, not a zero.** EDGE estimates
the *squared* spread, and in small samples that estimate can come out negative.
The paper reports sqrt(max(0, S^2)) for comparability with prior literature;
the reference code returns sqrt(|S^2|). Zero is the most flattering number this
module could ever emit -- it says trading is free -- and |S^2| is a point
estimate whose sign the estimator itself got wrong. Neither is a number a
backtest should subtract, so `spread` is None and `reason` says why. The upper
bound survives that case and is still reported.

**Nothing below one tick.** A US equity cannot be quoted tighter than a penny,
so on a $6 stock no effective spread under about 17bp is reachable. EDGE is a
statistical estimate and will sometimes return less. Flooring at the tick is
where pessimism costs nothing, because it is also the truth.

**Past the size limit, impact stops being square-root.** The square-root law is
calibrated on orders that are a small slice of the day's volume. Extrapolated
to ten times the limit it says ten times the size costs three times the
impact, which is how an illiquid strategy comes to look scalable. Past the cap
this model has no evidence, so it charges linearly and says so in the result.

Everything reads through `pit_store.bars_as_of`, never the table directly.
That is the only thing standing between this and a cost model that knows what
the spread turned out to be.
"""
from __future__ import annotations

import math
import statistics
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from research import pit_store

# --- estimator settings -----------------------------------------------------

DEFAULT_WINDOW = 60

# The authors' guidance (repo FAQ, and issue #2 on eguidotti/bidask): there is
# no universal minimum, but roughly one month of daily bars is a sensible
# default and the estimation error grows quickly below it. The hard floor in
# the estimator itself is three bars, which would produce a number nobody could
# defend. The universe screen in daily_job already demands 60 sessions before a
# name is eligible at all; this is the floor for the estimate, not for the name.
MIN_SESSIONS = 21

# What a 60-session window does not buy you. Measured 2026-08-27 against live
# Yahoo bars: at window=60 the estimator's own standard error is 25-50bp for
# large caps and 70-235bp for micro caps, so essentially nothing resolves --
# AAPL, MSFT, CRI, SIF and UUU all came back with a negative squared-spread
# estimate, and JPM's "52bp" carried a t-statistic of 1.0. At 252 sessions the
# genuinely illiquid names separate cleanly (CULP 172bp, GNSS 152bp, BSET
# 130bp, PW 350bp, all resolving) while the liquid ones still do not, which is
# the paper's own claim about where EDGE earns its keep, stated as a number.
#
# The default stays at 60 because that is the interface the strategy spec asks
# for and shortening the window is sometimes the right call. `resolved` in the
# result says whether it worked, and this constant is what the warning points
# the caller at.
RESOLVING_WINDOW = 252

# The instrument the estimator is measured against, and the reason it can be.
# SPY's quoted spread is one cent on a ~$600 price -- 0.17 basis points --
# essentially always and for years at a time. EDGE on the store's own SPY
# history returns 22bp at a 60-session window, 27bp at 252, 41bp at 504 and
# 37bp at 1008: between 130 and 240 times the truth, and `resolved=True` at the
# longer two, because `resolved` asks whether the estimate differs from zero
# and a biased estimator with a tight standard error passes that easily.
#
# So a longer window does not sharpen this. It removes the warning.
#
# What the reference buys is a floor for the estimator itself. If a name's
# estimate cannot be told apart from the reference's, then whatever its true
# spread is, it is under what daily bars resolve -- and for a name that liquid
# the honest number is the tick, not the estimator's noise. It is already in
# the store: the recorder writes it every night as the liveness canary.
REFERENCE_TICKER = "SPY"

# How much larger than the reference an estimate must be before it is treated
# as a measurement rather than as the same noise. Two, because the reference
# itself moves by a factor of about 1.9 across windows (22bp to 41bp) and
# anything inside that range is indistinguishable from it.
RESOLUTION_MULTIPLE = 2.0

# Two-sided 95%. Used one-sided here: the upper end of the interval is the
# pessimistic reading and the only one this module volunteers.
CONFIDENCE_Z = 1.96

# The minimum quoted increment for a US equity above $1.00. Sub-penny prints do
# happen -- midpoint fills, price improvement, hidden liquidity -- but a retail
# order is not the flow that gets them, so assuming they do not is the
# pessimistic reading and the one this model takes.
TICK_SIZE = 0.01

# --- impact settings --------------------------------------------------------

# Impact is modelled as C * sigma * sqrt(Q/ADV), the square-root law: Torre
# (1997) for BARRA, Almgren, Thum, Hauptmann & Li (2005), and a large
# literature since. Published estimates of C span roughly 0.3 to 1.0 depending
# on venue, period and whether permanent impact is included. This model takes
# the top of that range, because the cost of being too pessimistic is a trade
# not taken and the cost of being too optimistic is a strategy that backtests
# well and loses money.
IMPACT_COEFFICIENT = 1.0

# The spec's sizing rule: a position may not exceed 1% of median daily dollar
# volume. Below this the square-root law is inside the range it was measured
# on; above it, the model is extrapolating.
MAX_PARTICIPATION = 0.01

# Floating point makes "exactly at the cap" unrepresentable -- max_position
# divided back by median dollar volume is 0.010000000000000002 as often as it
# is 0.01. A caller sizing to the documented limit must not be told it broke
# the limit.
_CAP_TOLERANCE = 1e-9


# --------------------------------------------------------------- the estimator

def edge_moments(open: Sequence[float], high: Sequence[float],
                 low: Sequence[float],
                 close: Sequence[float]) -> Optional[Dict[str, float]]:
    """EDGE's squared-spread estimate together with its sampling error.

    The published estimator throws away everything except the final number.
    That is fine for a paper reporting cross-sectional averages and wrong for a
    cost model, where the question is not "what is the spread" but "how much of
    this figure is measurement". The two moment conditions x1 and x2 are
    per-observation series; the estimator is their inverse-variance weighted
    mean; so the standard error of that mean is available for the price of one
    more `np.std`.

    Verified against 2,400 simulated markets: the standard error returned here
    matches the observed sampling standard deviation to within 5%, and 95%
    intervals built from it cover the true squared spread 94% of the time.

    Returns None on exactly the inputs the reference implementation returns NaN
    for, and otherwise a dict of `s2` (the squared-spread estimate, which may be
    negative), `standard_error` (of `s2`), `observations`, and the intermediate
    probabilities and moments for anyone auditing a surprising number.
    """
    return _moments(open, high, low, close)


def edge(open: Sequence[float], high: Sequence[float], low: Sequence[float],
         close: Sequence[float], signed: bool = False) -> Optional[float]:
    """The EDGE effective-spread estimator, as published.

    A faithful reimplementation of Ardia, Guidotti & Kroencke (JFE 2024),
    matching the authors' reference implementation term for term. Returns the
    spread as a fraction of price -- 0.01 is a 1% spread -- and it is the
    *full* effective spread, not the half-spread: on a $100 stock a returned
    0.025 means a $98.75 bid against a $101.25 ask.

    `signed=False` returns sqrt(|S^2|), the reference default. `signed=True`
    returns sign(S^2) * sqrt(|S^2|), which is the only way a caller can tell a
    sample that identified a small spread from one that did not identify a
    spread at all. `estimate_spread` uses the signed form for exactly that
    reason.

    Returns None -- not NaN, and never a number -- where the estimator is
    undefined: fewer than three bars, fewer than two bars on which the price
    moved at all, or an open that never differs from the day's extremes.

    Prices must be strictly positive and on a consistent scale across adjacent
    bars. A single zero poisons the whole series (log(0) is -inf, which is not
    NaN and so survives every nanmean), and an unadjusted split injects a
    return of several hundred percent into one transition. `_window` cleans
    both before anything reaches here; a caller using this function directly
    is responsible for the same.
    """
    moments = _moments(open, high, low, close)
    if moments is None:
        return None
    s2 = moments["s2"]
    s = math.sqrt(abs(s2))
    return -s if (signed and s2 < 0) else s


def _moments(open: Sequence[float], high: Sequence[float],
             low: Sequence[float],
             close: Sequence[float]) -> Optional[Dict[str, float]]:
    o = np.asarray(open, dtype=float)
    h = np.asarray(high, dtype=float)
    lo = np.asarray(low, dtype=float)
    c = np.asarray(close, dtype=float)

    nobs = len(o)
    if len(h) != nobs or len(lo) != nobs or len(c) != nobs:
        raise ValueError(
            f"open/high/low/close must be the same length; got "
            f"{nobs}/{len(h)}/{len(lo)}/{len(c)}")
    if nobs < 3:
        return None

    with np.errstate(divide="ignore", invalid="ignore"):
        o, h, lo, c = np.log(o), np.log(h), np.log(lo), np.log(c)
    m = (h + lo) / 2.0

    h1, l1, c1, m1 = h[:-1], lo[:-1], c[:-1], m[:-1]
    o, h, lo, c, m = o[1:], h[1:], lo[1:], c[1:], m[1:]

    r1 = m - o
    r2 = o - m1
    r3 = m - c1
    r4 = c1 - m1
    r5 = o - c1

    # tau marks the bars on which the price moved. A bar whose high, low and
    # previous close are all equal carries no information about a spread, and
    # counting it would drag every probability below toward zero.
    tau = np.where(np.isnan(h) | np.isnan(lo) | np.isnan(c1), np.nan,
                   (h != lo) | (lo != c1))
    po1 = tau * np.where(np.isnan(o) | np.isnan(h), np.nan, o != h)
    po2 = tau * np.where(np.isnan(o) | np.isnan(lo), np.nan, o != lo)
    pc1 = tau * np.where(np.isnan(c1) | np.isnan(h1), np.nan, c1 != h1)
    pc2 = tau * np.where(np.isnan(c1) | np.isnan(l1), np.nan, c1 != l1)

    with np.errstate(invalid="ignore"):
        if np.all(np.isnan(tau)):
            return None
        pt = float(np.nanmean(tau))
        po = float(np.nanmean(po1) + np.nanmean(po2))
        pc = float(np.nanmean(pc1) + np.nanmean(pc2))

        if np.nansum(tau) < 2 or po == 0 or pc == 0 or not np.isfinite(pt):
            return None

        # The de-meaning of eq. (9): each return is centred on its own
        # conditional mean given that the price moved, which is what makes the
        # estimator unbiased under drift.
        d1 = r1 - np.nanmean(r1) / pt * tau
        d3 = r3 - np.nanmean(r3) / pt * tau
        d5 = r5 - np.nanmean(r5) / pt * tau

        x1 = -4.0 / po * d1 * r2 + -4.0 / pc * d3 * r4
        x2 = -4.0 / po * d1 * r5 + -4.0 / pc * d5 * r4

        e1 = float(np.nanmean(x1))
        e2 = float(np.nanmean(x2))
        v1 = float(np.nanmean(x1 ** 2) - e1 ** 2)
        v2 = float(np.nanmean(x2 ** 2) - e2 ** 2)

    # Eq. (25): the two moment conditions are combined by inverse variance, so
    # the noisier one gets less say. The cross-assignment (v2 weights e1) is
    # correct and is the part a reimplementation most easily gets backwards.
    vt = v1 + v2
    if vt > 0:
        w1, w2 = v2 / vt, v1 / vt
    else:
        w1 = w2 = 0.5
    s2 = w1 * e1 + w2 * e2
    if not math.isfinite(s2):
        return None

    # The estimate is a sample mean of w1*x1 + w2*x2, so its standard error is
    # that series' own. Treating the weights as fixed understates it slightly;
    # against simulation the understatement is inside 5%.
    with np.errstate(invalid="ignore"):
        combined = w1 * x1 + w2 * x2
    combined = combined[np.isfinite(combined)]
    n = int(combined.size)
    standard_error = (float(np.std(combined, ddof=1) / math.sqrt(n))
                      if n > 1 else float("nan"))

    return {"s2": float(s2), "standard_error": standard_error,
            "observations": n, "pt": pt, "po": po, "pc": pc,
            "e1": e1, "e2": e2, "v1": v1, "v2": v2}


# ------------------------------------------------------------------- the window

def _window(ticker: str, as_of: str,
            window: int) -> Dict[str, Any]:
    """The cleaned bars for one name as they stood on `as_of`, or a reason.

    Everything the three public functions need comes from here, so every rule
    about what counts as a usable session is stated once. The cleaning is not
    cosmetic -- each rule is a way the estimate goes silently wrong:

    A non-positive price makes log(price) -inf, which is not NaN, so it
    survives np.nanmean and turns the whole estimate into NaN or worse into a
    finite number built out of infinities.

    A bar whose open or close sits outside its own high-low range, or whose low
    exceeds its high, is a vendor error. The paper's own sample construction
    drops these; a spread estimator handed one reads the impossible range as
    trading cost.

    A split is the sharp one. `pit_store` keeps raw prices deliberately, so a
    10-for-1 split is a genuine -90% print between two adjacent sessions. EDGE
    only ever compares adjacent bars, so one split breaks exactly the
    transitions that touch it -- and inflates the estimate by an order of
    magnitude if it is left in. Only splits recorded on or before `as_of` are
    known, which is why they are read through the point-in-time accessor.
    """
    if window < MIN_SESSIONS:
        raise ValueError(
            f"window={window} is below MIN_SESSIONS={MIN_SESSIONS}; a shorter "
            f"window cannot produce an estimate this module would return")

    bars = pit_store.bars_as_of(ticker, as_of)[-window:]
    out: Dict[str, Any] = {
        "sessions_known": len(bars), "sessions_used": 0,
        "sessions_excluded": 0, "splits_excluded": 0, "reason": None,
        "last_price": None, "median_dollar_volume": None,
    }
    if len(bars) < MIN_SESSIONS:
        out["reason"] = (
            f"insufficient history: {len(bars)} sessions known on {as_of}, "
            f"{MIN_SESSIONS} required")
        return out

    split_dates = {a["ex_date"] for a
                   in pit_store.corporate_actions_as_of(ticker, as_of)
                   if a["action_type"] == "split"}

    n = len(bars)
    o = np.full(n, np.nan)
    h = np.full(n, np.nan)
    lo = np.full(n, np.nan)
    c = np.full(n, np.nan)
    dollar: List[float] = []
    splits_hit = 0

    for i, bar in enumerate(bars):
        values = [bar.get(f) for f in ("open", "high", "low", "close")]
        usable = all(v is not None and math.isfinite(float(v)) and float(v) > 0
                     for v in values)
        if usable:
            bo, bh, bl, bc = (float(v) for v in values)
            # The paper's screen: a bar that contradicts itself is not data.
            if bl > bh or not (bl <= bo <= bh) or not (bl <= bc <= bh):
                usable = False

        if usable and bar["trade_date"] in split_dates:
            usable = False
            splits_hit += 1

        if usable:
            o[i], h[i], lo[i], c[i] = bo, bh, bl, bc

        close, volume = bar.get("close"), bar.get("volume")
        if (close is not None and volume is not None
                and math.isfinite(float(close)) and float(close) > 0
                and math.isfinite(float(volume))):
            dollar.append(float(close) * float(volume))

    used = int(np.count_nonzero(np.isfinite(c)))
    out.update({"sessions_used": used, "splits_excluded": splits_hit,
                "sessions_excluded": n - used})

    if used < MIN_SESSIONS:
        out["reason"] = (
            f"only {used} of {n} sessions are usable "
            f"({splits_hit} split, {n - used - splits_hit} with a missing, "
            f"non-positive or self-contradictory price); "
            f"{MIN_SESSIONS} required")
        return out

    if not dollar or statistics.median(dollar) <= 0:
        out["reason"] = (
            f"no session in the {n}-session window recorded any dollar "
            f"volume; a price path nobody traded on is not evidence of a "
            f"spread")
        return out

    finite = c[np.isfinite(c)]
    out.update({
        "open": o, "high": h, "low": lo, "close": c,
        "last_price": float(finite[-1]),
        "median_dollar_volume": float(statistics.median(dollar)),
        "sessions_counted": n,
    })
    return out


def _daily_volatility(close: np.ndarray) -> Optional[float]:
    """Close-to-close log-return volatility, the sigma in the impact term.

    Computed on the cleaned closes, so a return spanning an excluded session
    is NaN and drops out rather than contributing a fabricated jump.
    """
    with np.errstate(invalid="ignore"):
        returns = np.diff(np.log(close))
    returns = returns[np.isfinite(returns)]
    if len(returns) < 2:
        return None
    sigma = float(np.std(returns, ddof=1))
    return sigma if sigma > 0 else None


def _refusal(ticker: str, as_of: str, window: int, base: Dict[str, Any],
             keys: Sequence[str]) -> Dict[str, Any]:
    out = {"ticker": ticker, "as_of": as_of, "window": window,
           "sessions_used": base.get("sessions_used", 0),
           "sessions_excluded": base.get("sessions_excluded", 0),
           "splits_excluded": base.get("splits_excluded", 0),
           "median_dollar_volume": base.get("median_dollar_volume"),
           "reason": base["reason"], "warning": None}
    for key in keys:
        out.setdefault(key, None)
    return out


_SPREAD_KEYS = ("spread", "spread_upper", "standard_error", "resolved",
                "signed_estimate", "last_price", "tick_floor",
                "floored_at_tick")
_PARTICIPATION_KEYS = ("participation", "max_position_dollars",
                       "within_limit", "last_price")


# ----------------------------------------------------------------- public API

def estimate_spread(ticker: str, as_of: str,
                    window: int = DEFAULT_WINDOW) -> Dict[str, Any]:
    """The EDGE effective spread for `ticker`, as a fraction of price.

    Reads only what was known on `as_of`. The spread is the *full* effective
    spread, so 0.004 means paying 20bp against the midpoint on each side of a
    round trip.

    Three numbers come back, and the difference between them is the point:

      `spread` -- EDGE's point estimate, floored at one tick. None when the
      window does not identify a spread at all.

      `spread_upper` -- the 95% upper confidence bound, floored at one tick.
      This survives a negative point estimate, is stable across window lengths
      where the point estimate is not, and is what `round_trip_cost` charges.

      `resolved` -- whether the point estimate is distinguishable from zero.
      At `window=60` this is False for almost every US equity, including names
      whose spread is genuinely wide; see `RESOLVING_WINDOW`. A False here does
      not mean the name is untradeable, it means 60 daily bars were not enough
      to measure it and the cost model is running on the bound instead.

    `standard_error` here is on the *spread* scale, converted from the squared
    scale by the delta method, so that it is comparable with `spread` -- unlike
    `edge_moments`, whose `standard_error` is on the squared scale the
    estimator natively works in. It is None when the point estimate is zero,
    where the conversion divides by zero and means nothing anyway.

    The refusals are the other point of this function: a backtest that skips a
    name loses an opportunity, and a backtest handed an invented spread loses
    money.
    """
    return _estimate(_window(ticker, as_of, window), ticker, as_of, window)


def _estimate(base: Dict[str, Any], ticker: str, as_of: str,
              window: int) -> Dict[str, Any]:
    """The estimate given an already-loaded window.

    Split out so `round_trip_cost` does not re-read the same sixty bars from
    sqlite to compute a volatility it could have had the first time. A backtest
    calls this once per name per day, so halving the query count is not
    micro-optimisation.
    """
    if base["reason"]:
        return _refusal(ticker, as_of, window, base, _SPREAD_KEYS)

    moments = _moments(base["open"], base["high"], base["low"], base["close"])
    last_price = base["last_price"]
    tick_floor = TICK_SIZE / last_price

    result = {
        "ticker": ticker, "as_of": as_of, "window": window,
        "sessions_used": base["sessions_used"],
        "sessions_excluded": base["sessions_excluded"],
        "splits_excluded": base["splits_excluded"],
        "median_dollar_volume": base["median_dollar_volume"],
        "last_price": last_price, "tick_floor": tick_floor,
        "signed_estimate": None, "spread": None, "spread_upper": None,
        "standard_error": None, "resolved": False, "floored_at_tick": None,
        "reason": None, "warning": None,
    }

    if moments is None:
        result["reason"] = (
            f"EDGE is undefined on this window: fewer than two of "
            f"{base['sessions_used']} usable sessions saw the price move, or "
            f"the open never differed from the day's range. This is a name "
            f"that barely trades, not a name that trades for free")
        return result

    s2, se = moments["s2"], moments["standard_error"]
    signed = math.copysign(math.sqrt(abs(s2)), s2)
    result["signed_estimate"] = signed

    if not math.isfinite(se):
        result["reason"] = (
            f"EDGE produced an estimate from {moments['observations']} "
            f"observations but no sampling error to judge it by, so there is "
            f"no way to tell {signed:.5f} from noise")
        return result

    upper = math.sqrt(max(0.0, s2 + CONFIDENCE_Z * se))
    result["spread_upper"] = max(upper, tick_floor)
    result["standard_error"] = se / (2 * abs(signed)) if signed else None
    result["resolved"] = bool(s2 > CONFIDENCE_Z * se)

    if s2 <= 0:
        result["reason"] = (
            f"EDGE returned a non-positive squared-spread estimate "
            f"({s2:.3e} against a standard error of {se:.3e}); the window "
            f"does not identify a spread. Reporting zero would say trading is "
            f"free and reporting sqrt(|S^2|)={abs(signed):.5f} would report a "
            f"number whose sign the estimator got wrong, so there is no point "
            f"estimate. spread_upper={result['spread_upper']:.5f} still holds")
        return result

    result["spread"] = max(signed, tick_floor)
    result["floored_at_tick"] = bool(signed < tick_floor)
    if not result["resolved"]:
        result["warning"] = (
            f"the point estimate {signed:.5f} is inside its own sampling "
            f"error ({se:.3e} on the squared spread); {window} sessions do "
            f"not resolve this name. Costs are charged on "
            f"spread_upper={result['spread_upper']:.5f}; "
            f"window={RESOLVING_WINDOW} is where real names start to resolve")
    return result


def participation_rate(ticker: str, as_of: str, position_dollars: float,
                       window: int = DEFAULT_WINDOW) -> Dict[str, Any]:
    """How much of a normal day's trading this position is.

    Impact scales with the order against the flow it has to hide in, so size
    is meaningless in dollars and meaningful only as a share of volume. The
    spec's rule is that a position may not exceed `MAX_PARTICIPATION` of median
    daily dollar volume; `max_position_dollars` is that rule stated as a number
    the caller can size to, rather than a verdict delivered after the fact.
    """
    if not (position_dollars > 0):
        raise ValueError(
            f"position_dollars must be positive; got {position_dollars!r}. A "
            f"zero or negative position is a caller bug, and answering it "
            f"with a cost of zero would let the bug travel")

    base = _window(ticker, as_of, window)
    result = {"ticker": ticker, "as_of": as_of, "window": window,
              "position_dollars": position_dollars}
    if base["reason"]:
        result.update(_refusal(ticker, as_of, window, base,
                               _PARTICIPATION_KEYS))
        result["within_limit"] = False
        return result

    mdv = base["median_dollar_volume"]
    participation = position_dollars / mdv
    within = participation <= MAX_PARTICIPATION * (1 + _CAP_TOLERANCE)
    result.update({
        "median_dollar_volume": mdv,
        "participation": participation,
        "max_position_dollars": mdv * MAX_PARTICIPATION,
        "within_limit": bool(within),
        "last_price": base["last_price"],
        "sessions_used": base["sessions_used"],
        "sessions_excluded": base["sessions_excluded"],
        "splits_excluded": base["splits_excluded"],
        "reason": None,
        "warning": None if within else (
            f"position is {participation:.2%} of median daily dollar volume, "
            f"over the {MAX_PARTICIPATION:.0%} limit; the largest position "
            f"this name supports is ${mdv * MAX_PARTICIPATION:,.0f}"),
    })
    return result


def _impact_shape(participation: float) -> float:
    """sqrt(p) inside the size limit, linear beyond it.

    The square-root law is an empirical fit over orders that are a small share
    of the day. Run past the cap it keeps quoting a sub-linear cost -- ten
    times the size for 3.2 times the impact -- which is precisely the number
    that makes an illiquid strategy look like it scales. Past the cap this
    model has no evidence, so it takes the pessimistic reading and charges
    proportionally. The two branches meet at the cap, so a caller cannot save
    modelled cost by shaving a cent off the order.
    """
    cap = MAX_PARTICIPATION
    if participation <= cap:
        return math.sqrt(participation)
    return math.sqrt(cap) * (participation / cap)


def _join_warnings(*warnings: Optional[str]) -> Optional[str]:
    present = [w for w in warnings if w]
    return "; also ".join(present) if present else None


def spread_basis(ticker: str, as_of: str,
                 window: int = DEFAULT_WINDOW) -> Dict[str, Any]:
    """The spread to charge, and whether it was measured or floored.

    Three outcomes, and the difference between them is the whole point:

      `measured` -- the estimate stands clear of what the reference instrument
      produces on the same window, so it is carrying information about this
      name rather than about the estimator.

      `at_resolution_floor` -- it does not. Whatever this name's spread is, it
      is below what daily bars can resolve, which for a name this liquid means
      the tick. Charging EDGE's number here would charge the estimator's bias:
      40bp for SPY, whose market is a cent wide.

      `unknown` -- the reference is not in the store for this window, so there
      is no yardstick. Picking one of the other two silently is how 40bp
      becomes a fact.
    """
    own = estimate_spread(ticker, as_of, window)
    tick_floor = own.get("tick_floor")

    reference = estimate_spread(REFERENCE_TICKER, as_of, window)
    ref_level = reference.get("spread") or reference.get("spread_upper")
    if ref_level is None:
        return {"ticker": ticker, "as_of": as_of, "window": window,
                "basis": "unknown", "spread": None, "tick_floor": tick_floor,
                "reference_spread": None,
                "reason": (f"{REFERENCE_TICKER} has no estimate on this window, "
                           f"so there is nothing to measure the estimator "
                           f"against: {reference.get('reason') or 'no data'}")}

    own_level = own.get("spread")
    if own_level is not None and own_level > ref_level * RESOLUTION_MULTIPLE:
        return {"ticker": ticker, "as_of": as_of, "window": window,
                "basis": "measured", "spread": own_level,
                "tick_floor": tick_floor, "reference_spread": ref_level,
                "reason": (f"{own_level * 1e4:.1f}bp stands clear of the "
                           f"{ref_level * 1e4:.1f}bp the reference produces on "
                           f"the same window")}

    if tick_floor is None:
        return {"ticker": ticker, "as_of": as_of, "window": window,
                "basis": "unknown", "spread": None, "tick_floor": None,
                "reference_spread": ref_level,
                "reason": "no last price, so there is no tick to floor at"}

    shown = f"{own_level * 1e4:.1f}bp" if own_level is not None else "no estimate"
    return {"ticker": ticker, "as_of": as_of, "window": window,
            "basis": "at_resolution_floor", "spread": tick_floor,
            "tick_floor": tick_floor, "reference_spread": ref_level,
            "reason": (f"{shown} is not clear of the {ref_level * 1e4:.1f}bp "
                       f"the reference produces on the same window, so this is "
                       f"the estimator's floor rather than this name's spread; "
                       f"charging one tick ({tick_floor * 1e4:.2f}bp) instead")}


def round_trip_cost(ticker: str, as_of: str, position_dollars: float,
                    window: int = DEFAULT_WINDOW,
                    basis: str = "upper") -> Dict[str, Any]:
    """What a round trip costs, as a fraction of the position. Subtract this.

    Two components, both round-trip:

      spread_cost -- half the effective spread on the way in and half on the
      way out, which is one full effective spread for the pair. Charging one
      half-spread for the round trip is the easiest way to halve a cost model
      and it leaves no trace in the output.

      impact_cost -- twice C * sigma * sqrt(Q/ADV), the square-root law, for
      the two crossings. Past `MAX_PARTICIPATION` the shape goes linear; see
      `_impact_shape`.

    `basis` decides which spread is charged. The default `"upper"` uses the
    95% upper confidence bound, because over a 60-session window the point
    estimate's sampling error is larger than the spread of most US equities and
    subtracting a point estimate would be subtracting noise that happens to
    have landed low. `"point"` uses EDGE's point estimate and refuses whenever
    that estimate does not exist. Running the backtest both ways is the only
    way to see how much of a result is measurement error, which is why the
    second option exists at all.

    `cost` is None with a `reason` whenever the spread could not be measured.
    There is no house average and no last-known value to fall back to, because
    a fallback here is a number nobody can check that gets subtracted from
    every trade in the study.
    """
    if not (position_dollars > 0):
        raise ValueError(
            f"position_dollars must be positive; got {position_dollars!r}. A "
            f"zero or negative position is a caller bug, and answering it "
            f"with a cost of zero would let the bug travel")
    if basis not in ("upper", "point", "adaptive"):
        raise ValueError(
            f"basis must be 'upper', 'point' or 'adaptive'; got {basis!r}")

    base = _window(ticker, as_of, window)
    est = _estimate(base, ticker, as_of, window)
    result = {**est, "position_dollars": position_dollars,
              "spread_basis": basis, "resolution": None, "spread_used": None,
              "half_spread": None, "spread_cost": None, "impact_cost": None,
              "cost": None, "participation": None,
              "max_position_dollars": None, "daily_volatility": None,
              "exceeds_liquidity_limit": None}

    if basis == "adaptive":
        # Measured where the estimator has something to say, floored at the
        # tick where it does not. See `spread_basis`.
        decided = spread_basis(ticker, as_of, window)
        result["resolution"] = decided["basis"]
        charged = decided["spread"]
        if charged is None:
            result["reason"] = decided["reason"]
            return result
    else:
        charged = est["spread_upper"] if basis == "upper" else est["spread"]
    if charged is None:
        if basis == "point" and est["spread_upper"] is not None:
            result["reason"] = (
                f"{est['reason']} basis='point' was asked for, so there is no "
                f"cost; basis='upper' would charge "
                f"{est['spread_upper']:.5f}")
        return result

    sigma = _daily_volatility(base["close"])
    if sigma is None:
        result["reason"] = (
            f"the spread came out at {charged:.5f} but the window has no "
            f"usable close-to-close volatility, so impact cannot be priced; a "
            f"spread-only cost would say an order of any size fills at the "
            f"quote")
        return result

    mdv = base["median_dollar_volume"]
    participation = position_dollars / mdv
    within = participation <= MAX_PARTICIPATION * (1 + _CAP_TOLERANCE)

    spread_cost = charged
    impact_cost = 2.0 * IMPACT_COEFFICIENT * sigma * _impact_shape(
        participation)

    result.update({
        "spread_used": charged,
        "half_spread": charged / 2.0,
        "spread_cost": spread_cost,
        "impact_cost": impact_cost,
        "cost": spread_cost + impact_cost,
        "participation": participation,
        "median_dollar_volume": mdv,
        "max_position_dollars": mdv * MAX_PARTICIPATION,
        "daily_volatility": sigma,
        "exceeds_liquidity_limit": not within,
        # Both warnings can be true at once, and the size one used to overwrite
        # the "this spread is unresolved" one -- which is exactly the caveat a
        # caller most needs to keep.
        "warning": _join_warnings(est["warning"], None if within else (
            f"position is {participation:.2%} of median daily dollar volume, "
            f"over the {MAX_PARTICIPATION:.0%} limit; impact is charged "
            f"linearly rather than square-root past that point because the "
            f"square-root law was never measured there. The largest position "
            f"this name supports is ${mdv * MAX_PARTICIPATION:,.0f}")),
    })
    return result
