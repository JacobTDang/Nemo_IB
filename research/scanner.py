"""Where a signal becomes a position, and what has to be true first.

Everything upstream measures. `sue` says how far a quarter came in from what
the series implied; `spread` says what a round trip costs; `pit_store` says what
was known on a date. None of them decide anything. This does, and the step from
measurement to decision is where a study stops being arithmetic and starts
being a claim about money.

Four things shape it.

**Cost is not a haircut applied at the end.** Post-earnings drift is a few tens
of basis points over a few weeks. A 60bp round trip is not a rounding error
against that, it is the whole thing. Worse, the two are correlated in the
direction that flatters a backtest: illiquid names have wider spreads *and*
noisier earnings, so they produce the largest surprises, and a scanner ranked
on the signal alone fills its book with exactly the names whose costs eat it.
So the rank here is net of cost, and a name that does not clear its own cost is
not a marginal trade -- it is not a trade.

**A fill has to have been available.** A company reporting after Monday's close
cannot be bought at Monday's close, and Monday's close is what sits in the
store when the scan runs. Nothing here records a fill price: it records the
session the order is for, and the price is whatever that session turns out to
print. Logging an intention and scoring it later is slower than assuming a
fill, and it is the only version that measures anything.

**Size is a share of volume, not a number of dollars.** $100k into a name that
trades $200k a day is not a position. `spread.participation_rate` states the
cap and every target is clipped to it.

**A missing input is a refusal.** There is no house-average spread and no
last-known value. A name whose cost cannot be measured is not a name with an
average cost, and a fallback here is a number nobody can check being subtracted
from every trade in the study.

The one number that is not measured is the drift itself -- how much a unit of
SUE is worth over the holding period. It is declared as an assumption, reported
as one, and marked uncalibrated, because the honest source for it is this
store's own accumulated history rather than a coefficient from a paper about a
different decade. That is what the recorder is for.
"""
from __future__ import annotations

import statistics
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

from research import pit_store, spread, sue

# --- what the book is, before the regime touches it -------------------------

GROSS_TARGET = 100_000.0
MAX_NAMES = 20

# --- what makes a signal worth acting on ------------------------------------

MIN_ABS_SUE = 1.0

# And an upper one. A ten-sigma move in GAAP EPS is not an earnings surprise,
# it is a non-operating item: Alphabet runs equity revaluations through EPS,
# and a live scan ranked it and Amazon first on SUEs of 10.98 and 12.68 from
# correctly parsed filings. Drift is the market underreacting to OPERATING
# news, and every published coefficient is fit over roughly |SUE| <= 4, so
# pricing 12.68 of it extrapolates a linear model six standard deviations past
# anything it was estimated on -- and puts the least trustworthy signals at the
# top of the book while doing it.
MAX_ABS_SUE = 6.0
MIN_SIGMA_QUARTERS = sue.MIN_SIGMA_QUARTERS

# Which surprise to rank on. "ts" is the time-series variant, computed from
# filings, and it works today. "af" is the analyst one, which the replication
# literature finds carries the stronger surviving effect and which needs about
# nine months more recorded consensus before it answers for anything.
# "af_or_ts" prefers it and falls back -- convenient, and it builds a book half
# from one signal and half from the other, which is not a study of either.
SIGNAL_VARIANT = "ts"

# The drift is a few weeks long and then it is gone. A surprise from last
# quarter is not a reason to buy today, and nothing in the signal itself says
# so -- `sue_ts` will happily return a six-month-old quarter as "the most
# recent one filed".
MAX_SIGNAL_AGE_DAYS = 45

# THE assumption. Not measured here, and deliberately not buried: how many
# basis points of drift one unit of standardised surprise is worth over the
# holding period. Every net edge in this module is this number times a SUE,
# so a study built on it is a study of this number. It is reported with
# `calibrated: False` on every scan for that reason, and the point of
# recording scans is to eventually replace it with a figure measured on this
# book's own history.
DRIFT_BPS_PER_SUE = 15.0
DRIFT_CALIBRATED = False

# The same assumption for the cross-sectional variant, and deliberately a
# separate number. That signal is a rank, not a sigma -- `sue_cs` reports
# `rank_on: percentile` precisely because neither z it can compute means what a
# standard deviation means in a distribution with those tails. Pricing a rank
# with the surprise coefficient would make the two variants look comparable
# when they measure different things. Basis points at a full tail, scaling
# linearly with distance from the median.
DRIFT_BPS_PER_TAIL = 40.0

# How far into a tail a name has to be. At the median it reported in line with
# its peers, which is not news however large the beat was in isolation.
MIN_TAIL_DISTANCE = 0.8

# --- regime -----------------------------------------------------------------

# The index, used only for its own realised volatility. It is in the store
# because every bar fetch carries it as a liveness canary.
REGIME_TICKER = "SPY"
REGIME_WINDOW = 20
REGIME_BASELINE_WINDOW = 252
REGIME_MIN_SCALE = 0.25


def _today() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).date().isoformat()


def _next_session(as_of: str) -> str:
    """The next weekday. Holidays are not modelled, and it does not matter:
    this names the session an order is *for*, and an order resting on a closed
    exchange fills on the next open rather than becoming a different order."""
    day = date.fromisoformat(as_of) + timedelta(days=1)
    while day.weekday() >= 5:
        day += timedelta(days=1)
    return day.isoformat()


# --- seams, so the decision logic can be tested without a network -----------

# The cohort for one date, held only for the length of a scan. It is the same
# list for every name on that date by construction, and rebuilding it per name
# is quadratic: on the real store a cohort of 305 takes 386ms, so a date with
# 305 reporters spent two minutes rebuilding one list and a replay over 652
# dates would have run for about twenty-one hours.
_COHORT: Dict[str, Any] = {}


def _peers_for(as_of: str):
    from research import sue_cs

    if _COHORT.get("as_of") != as_of:
        _COHORT["as_of"] = as_of
        _COHORT["peers"] = sue_cs.cohort(as_of)
    return _COHORT["peers"]


def _signal_for(ticker: str, as_of: str) -> Dict[str, Any]:
    """The surprise, from whichever variant `SIGNAL_VARIANT` names.

    Each answer carries the variant that produced it, so a book cannot end up
    mixing the two without saying so.
    """
    if SIGNAL_VARIANT == "ts":
        return {**sue.sue_ts(ticker, as_of=as_of), "variant": "ts"}
    if SIGNAL_VARIANT == "af":
        return {**sue.sue_af(ticker, as_of=as_of), "variant": "af"}
    if SIGNAL_VARIANT == "af_or_ts":
        analyst = sue.sue_af(ticker, as_of=as_of)
        if analyst.get("success"):
            return {**analyst, "variant": "af"}
        return {**sue.sue_ts(ticker, as_of=as_of), "variant": "ts"}
    if SIGNAL_VARIANT == "cs":
        from research import sue_cs
        return {**sue_cs.surprise_rank(ticker, as_of=as_of,
                                       peers=_peers_for(as_of)),
                "variant": "cs"}
    raise ValueError(
        f"SIGNAL_VARIANT must be 'ts', 'af', 'af_or_ts' or 'cs'; got "
        f"{SIGNAL_VARIANT!r}")


def _cost_for(ticker: str, as_of: str, position_dollars: float) -> Dict[str, Any]:
    """Round-trip cost, with the floor beside the bound.

    EDGE cannot resolve a mega-cap's spread from daily bars, and a longer
    window does not help -- it removes the warning. Measured against SPY, whose
    market is a cent wide at 0.17bp, the estimator returns 22bp at 60 sessions,
    27bp at 252 and 41bp at 504, reporting `resolved=True` at the last of
    those. Charging its upper bound put every name in the universe inside the
    same 20-50bp band and erased the liquidity gradient entirely.

    `basis="adaptive"` charges EDGE where its estimate stands clear of what the
    reference produces on the same window, and one tick where it does not --
    because a name that liquid has a tick-wide market, and the estimator's
    output there is its own bias. The gradient comes back: 0.2bp for SPY, 0.5
    for AAPL and MSFT, 1.6 for KO, 3.0 for BJ, 82 for MOV, 120 for PLAB.

    The floor is still carried beside it, so a name whose cost is a bound
    rather than a measurement can still be told apart from one the strategy
    turned down.
    """
    cost = spread.round_trip_cost(ticker, as_of, position_dollars,
                                  window=spread.RESOLVING_WINDOW,
                                  basis="adaptive")
    if cost.get("cost") is None:
        return cost
    tick_floor = cost.get("tick_floor")
    impact = cost.get("impact_cost") or 0.0
    cost["cost_floor"] = (tick_floor + impact) if tick_floor is not None \
        else cost["cost"]
    return cost


def _regime_scale(as_of: str) -> tuple:
    """How much of the target book to run, from the index's own realised vol.

    Not a forecast. Realised volatility is high after a market has already
    moved, which is exactly when a fixed-dollar book is carrying more risk than
    it was sized for. Scaling by the ratio of recent vol to its own longer
    baseline holds risk roughly constant instead of holding dollars constant.

    Reads only sessions on or before `as_of`, like everything else here.
    """
    bars = pit_store.adjusted_bars(REGIME_TICKER, as_of)
    if len(bars) < REGIME_WINDOW + 2:
        # Not a guess and not a refusal to trade: with no history the book runs
        # at its stated target, which is the same thing "no regime signal"
        # means.
        return 1.0, "unknown"

    closes = [b["close"] for b in bars if b["close"]]
    returns = [closes[i] / closes[i - 1] - 1.0 for i in range(1, len(closes))]
    recent = returns[-REGIME_WINDOW:]
    baseline = returns[-REGIME_BASELINE_WINDOW:]
    if len(recent) < 2 or len(baseline) < REGIME_WINDOW:
        return 1.0, "unknown"

    recent_vol = statistics.pstdev(recent)
    baseline_vol = statistics.pstdev(baseline)
    if not baseline_vol or not recent_vol:
        return 1.0, "unknown"

    scale = min(1.0, max(REGIME_MIN_SCALE, baseline_vol / recent_vol))
    if scale >= 0.95:
        label = "calm"
    elif scale >= 0.6:
        label = "elevated"
    else:
        label = "stressed"
    return scale, label


# --- the gates, each one saying why -----------------------------------------

def _basis_change_in_window(signal: Dict[str, Any]) -> Optional[str]:
    """Only a change among the quarters sigma is built from matters.

    `sue` reports every basis change it found across the whole series and has
    already rebased the figures onto one basis before computing anything, so
    the mere presence of one is not a defect -- AAPL's are from 2014 and 2020.
    Rejecting on presence threw out AAPL, AMZN, GOOGL and NVDA in a live scan,
    four of the most liquid names on the tape, for splits a decade outside the
    window that were corrected for.

    What would still bite is a change among the quarters sigma spans, where a
    rebase that is even slightly wrong shows up as dispersion.
    """
    periods = signal.get("sigma_periods") or []
    if not periods:
        return None
    ends = [p for p in periods if p]
    if not ends:
        return None
    # sigma_periods are fiscal labels ("2025Q3"); the earliest one bounds the
    # window. A change is in-window if it was noticed at or after that quarter
    # began, which the later of its two filing dates is a safe proxy for.
    earliest = min(ends)
    try:
        year = int(earliest[:4])
    except (TypeError, ValueError):
        return None
    boundary = f"{year - 1}-01-01"

    for change in signal.get("basis_changes") or []:
        between = change.get("between") or []
        when = max(between) if between else change.get("period")
        if when and str(when) >= boundary:
            return str(when)
    return None


def _tail_distance(percentile: Optional[float]) -> Optional[float]:
    """How far into a tail a rank sits, on 0 at the median to 1 at either end.

    Direction is carried separately by the sign of `percentile - 0.5`, so this
    is a magnitude and reads like |SUE| does.
    """
    if percentile is None:
        return None
    return abs(percentile - 0.5) * 2.0


def _cross_sectional_problem(signal: Dict[str, Any]) -> Optional[str]:
    distance = _tail_distance(signal.get("percentile"))
    if distance is None:
        return "no percentile computed"
    if distance < MIN_TAIL_DISTANCE:
        return (f"percentile {signal['percentile']:.2f} is "
                f"{distance:.2f} into a tail, under the "
                f"{MIN_TAIL_DISTANCE} required; a name near the middle of its "
                f"cohort reported in line with its peers, which is not news "
                f"however large the beat looks on its own")
    return None


def _signal_problem(signal: Dict[str, Any], as_of: str) -> Optional[str]:
    if not signal or not signal.get("success"):
        return (signal or {}).get("error") or "no signal"

    known_at = signal.get("known_at")
    if not known_at:
        return "signal carries no date, so it cannot be placed in time"
    if known_at > as_of:
        return f"not yet known on {as_of}; filed {known_at}"

    age = (date.fromisoformat(as_of) - date.fromisoformat(known_at[:10])).days
    if age > MAX_SIGNAL_AGE_DAYS:
        return (f"stale: filed {known_at}, {age} days before {as_of}, past the "
                f"{MAX_SIGNAL_AGE_DAYS}-day window the drift lives in")

    if signal.get("variant") == "cs":
        # A rank has no sigma window and no basis to change; its own gate is
        # the tail distance.
        return _cross_sectional_problem(signal)

    quarters = signal.get("sigma_quarters") or 0
    if quarters < MIN_SIGMA_QUARTERS:
        return (f"sigma from {quarters} quarters, under the {MIN_SIGMA_QUARTERS} "
                f"needed for it to mean anything")

    inside = _basis_change_in_window(signal)
    if inside:
        return (f"basis changed inside the window ({inside}), so sigma "
                f"measures the redenomination rather than how much this "
                f"company surprises")

    value = signal.get("sue")
    if value is None:
        return "no surprise computed"
    if abs(value) < MIN_ABS_SUE:
        return f"|SUE| {abs(value):.2f} under the {MIN_ABS_SUE} threshold"
    if abs(value) > MAX_ABS_SUE:
        return (f"|SUE| {abs(value):.2f} is outside the range drift is "
                f"estimated over (<={MAX_ABS_SUE}); a move this large in GAAP "
                f"EPS is usually a non-operating item rather than operating "
                f"news the market can underreact to")
    return None


def scan(as_of: Optional[str] = None,
         already_acted: Optional[set] = None,
         signal_for=None) -> Dict[str, Any]:
    """Today's candidates, ranked net of cost, with every rejection kept.

    Rejections are half the output. A scanner that quietly stops finding
    anything looks exactly like a market with nothing in it, and the difference
    only shows up in the reasons.

    One print is one trade. `already_acted` defaults to every (ticker, period)
    the filed book already holds; replay passes its own, because a replayed
    decision never enters that book.

    `signal_for` defaults to a live EDGAR lookup. Replay supplies a table it
    built in one pass per name -- as a parameter rather than by rebinding the
    module's seam, which works only until something rebinds it back.
    """
    as_of = as_of or _today()
    # A new scan must not inherit a cohort assembled for another date.
    _COHORT.clear()
    signal_for = signal_for or _signal_for
    if already_acted is None:
        already_acted = pit_store.filed_periods(as_of)
    scale, regime = _regime_scale(as_of)
    gross = GROSS_TARGET * scale
    per_name = gross / MAX_NAMES

    universe = [m["ticker"] for m in pit_store.universe_as_of(as_of)
                if m["eligible"]]

    # Narrow before spending a request, not after. Asking EDGAR for a signal on
    # every eligible name is 2,435 companyconcept calls and about nineteen
    # minutes on the real universe, and nearly all of it buys the news that a
    # company last reported in May -- which the staleness gate below then
    # throws away. The store already knows who printed, because the consensus
    # recorder captures the vendor's actual within days of each one.
    since = (date.fromisoformat(as_of)
             - timedelta(days=MAX_SIGNAL_AGE_DAYS)).isoformat()
    if pit_store.has_consensus_history(as_of):
        # The recorder has been running, so an empty window is information:
        # nothing reported, and there is nothing to scan. Falling back to the
        # whole universe here would spend 2,435 requests to rediscover it.
        printed = {r["ticker"] for r in pit_store.reporters_since(since, as_of)}
        considered = [t for t in universe if t in printed]
        narrowed_by = "recorded prints"
        narrowing_note = (f"{len(considered)} of {len(universe)} eligible "
                          f"names have a print recorded since {since}")
    else:
        # Nothing recorded at all -- a young store, not a quiet tape. Scanning
        # nothing here would look identical to a market with no earnings in it,
        # every night, until the recorder caught up.
        considered = universe
        narrowed_by = None
        narrowing_note = (
            f"no prints recorded by {as_of}, so every eligible name was asked "
            f"about; slow, and it eases as the recorder accumulates")

    candidates: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    undetermined: List[Dict[str, Any]] = []
    priced = 0
    measured_count = 0
    floored_count = 0

    for ticker in considered:
        try:
            signal = signal_for(ticker, as_of)
        except Exception as exc:  # noqa: BLE001 - recorded, not masked
            # EDGAR times out on one name and the other four hundred are fine.
            # Letting it propagate loses a whole night's decisions to a single
            # upstream hiccup, and the run log would show a crash rather than
            # the one ticker that caused it.
            rejected.append({"ticker": ticker, "sue": None,
                             "reason": f"signal unavailable: "
                                       f"{type(exc).__name__}: {exc}"})
            continue
        problem = _signal_problem(signal, as_of)
        if problem:
            rejected.append({"ticker": ticker, "reason": problem,
                             "sue": (signal or {}).get("sue")})
            continue

        period = signal.get("fiscal_period")
        if period and (ticker, period) in already_acted:
            rejected.append({
                "ticker": ticker, "sue": signal.get("sue"),
                "reason": (f"already acted on {ticker} {period}; the signal "
                           f"stays fresh for {MAX_SIGNAL_AGE_DAYS} days and "
                           f"one print is one trade")})
            continue

        if signal.get("variant") == "cs":
            value = signal["percentile"] - 0.5
            strength = _tail_distance(signal["percentile"])
        else:
            value = signal["sue"]
            strength = abs(value)
        side = "long" if value > 0 else "short"

        # Size first, because the cost depends on it: impact is a function of
        # the order against the flow, so a cost quoted without a size is a
        # cost for some other trade.
        try:
            fit = spread.participation_rate(ticker, as_of, per_name,
                                            window=spread.RESOLVING_WINDOW)
        except Exception as exc:  # noqa: BLE001 - recorded, not masked
            rejected.append({"ticker": ticker, "sue": value,
                             "reason": f"sizing unavailable: "
                                       f"{type(exc).__name__}: {exc}"})
            continue
        if fit.get("reason"):
            rejected.append({"ticker": ticker, "reason": fit["reason"],
                             "sue": value})
            continue
        target = min(per_name, fit["max_position_dollars"])
        if target <= 0:
            rejected.append({"ticker": ticker, "sue": value,
                             "reason": "no tradeable size at this liquidity"})
            continue

        try:
            cost = _cost_for(ticker, as_of, target)
        except Exception as exc:  # noqa: BLE001 - recorded, not masked
            rejected.append({"ticker": ticker, "sue": value,
                             "reason": f"cost unavailable: "
                                       f"{type(exc).__name__}: {exc}"})
            continue
        if cost.get("cost") is None:
            rejected.append({"ticker": ticker, "sue": value,
                             "reason": cost.get("reason")
                             or "cost could not be measured"})
            continue

        expected_bps = strength * (DRIFT_BPS_PER_TAIL
                                   if signal.get("variant") == "cs"
                                   else DRIFT_BPS_PER_SUE)
        cost_bps = cost["cost"] * 10_000
        floor_bps = (cost.get("cost_floor") or cost["cost"]) * 10_000
        net_bps = expected_bps - cost_bps
        # Not EDGE's `resolved` flag, which asks only whether the estimate
        # differs from zero -- SPY passes that at 41bp against a market a cent
        # wide. What a reader needs is whether the charge is this name's
        # measured spread or the tick it was floored to.
        resolution = cost.get("resolution")
        resolved = resolution == "measured"
        priced += 1
        if resolved:
            measured_count += 1
        elif resolution == "at_resolution_floor":
            floored_count += 1

        if net_bps <= 0:
            row = {"ticker": ticker, "sue": value,
                   "cost_bps_high": cost_bps, "cost_bps_low": floor_bps,
                   "expected_edge_bps": expected_bps}
            if not resolved and expected_bps > floor_bps:
                # The strategy did not turn this down; the estimator could not
                # tell. Filing it as a rejection would hide a name that trades
                # well inside its own bound, and would make a scanner starved
                # by measurement error look like one facing a quiet tape.
                row["reason"] = (
                    f"unresolved spread: expected {expected_bps:.1f}bp sits "
                    f"inside a cost band of {floor_bps:.1f}-{cost_bps:.1f}bp, "
                    f"so the estimator cannot say whether this clears")
                undetermined.append(row)
            else:
                row["reason"] = (f"expected {expected_bps:.1f}bp does not clear "
                                 f"{cost_bps:.1f}bp of round-trip cost")
                rejected.append(row)
            continue

        candidates.append({
            "ticker": ticker, "side": side, "sue": value,
            "fiscal_period": signal.get("fiscal_period"),
            "known_at": signal.get("known_at"),
            "variant": signal.get("variant"),
            "expected_edge_bps": expected_bps, "cost_bps": cost_bps,
            "cost_bps_low": floor_bps,
            "net_edge_bps": net_bps, "target_dollars": target,
            "participation": fit.get("participation"),
            "spread": cost.get("spread"),
            "spread_resolved": bool(cost.get("resolved")),
            # The session this order is for. No price: that session has not
            # happened, and the whole point of logging an intention is that
            # nobody gets to choose the fill afterwards.
            "intended_session": _next_session(as_of),
        })

    candidates.sort(key=lambda c: c["net_edge_bps"], reverse=True)
    for rank, candidate in enumerate(candidates[:MAX_NAMES], start=1):
        candidate["rank"] = rank
    dropped = candidates[MAX_NAMES:]
    for candidate in dropped:
        rejected.append({"ticker": candidate["ticker"], "sue": candidate["sue"],
                         "reason": (f"ranked below the top {MAX_NAMES} on net "
                                    f"edge")})

    return {
        "as_of": as_of,
        "regime": regime,
        "regime_scale": scale,
        "gross_target": gross,
        "screened": len(universe),
        "considered": len(considered),
        "narrowed_by": narrowed_by,
        "narrowing_note": narrowing_note,
        "candidates": candidates[:MAX_NAMES],
        "rejected": rejected,
        # Names the estimator could not judge, kept apart from names the
        # strategy turned down.
        "undetermined": undetermined,
        "costs_total": priced,
        "costs_measured": measured_count,
        "costs_floored": floored_count,
        "assumptions": {
            "variant": SIGNAL_VARIANT,
            "drift_bps_per_sue": DRIFT_BPS_PER_SUE,
            "drift_bps_per_tail": DRIFT_BPS_PER_TAIL,
            "calibrated": DRIFT_CALIBRATED,
            "note": ("drift is assumed, not measured here, and every net edge "
                     "is proportional to it. The two coefficients price "
                     "different quantities -- a sigma and a rank -- and are "
                     "deliberately not the same number"),
        },
    }


def record_scan(as_of: Optional[str] = None) -> Dict[str, Any]:
    """Run a scan and file it, candidates and rejections alike.

    Filed rather than returned so it can be scored against what the market
    actually did, which is the only test of any of this. Idempotent within a
    day for the same reason the bar recorder is: a second run must not double
    the book.

    A re-run whose answer has changed -- a threshold moved at lunchtime, say --
    keeps the filed decision and reports `superseded`. Silently returning
    candidates the record does not contain is how someone ends up reading one
    book and holding another.
    """
    as_of = as_of or _today()
    result = scan(as_of)

    # What this day already decided, if anything. Filing is append-only, so a
    # re-run cannot change it -- and should not, because the filed decision is
    # the one that would have been acted on. What it must not do is return a
    # different answer without mentioning that the record holds another one.
    already = {o["ticker"] for o in
               pit_store.paper_orders_as_of(as_of, accepted_only=True)
               if o["as_of_date"] == as_of}
    proposed = {c["ticker"] for c in result["candidates"]}
    if already and already != proposed:
        result["superseded"] = {
            "filed": sorted(already), "proposed": sorted(proposed),
            "note": (f"{as_of} was already decided and that decision stands; "
                     f"these candidates were not filed")}

    pit_store.start_run("scan", as_of_date=as_of)
    try:
        written = pit_store.record_paper_orders(
            as_of, result["candidates"], result["rejected"],
            regime=result["regime"], gross_target=result["gross_target"],
            # End of the day it decided, not the moment the process ran. A
            # scan for a past date stamped `now` is invisible to its own date
            # and visible to every date after it -- the same lookahead the
            # bar recorder stamps around.
            recorded_at=f"{as_of}T21:00:00Z")
    except Exception as exc:  # noqa: BLE001 - reported, never masked
        pit_store.finish_run(rows_written=0, status="failed",
                             error=f"{type(exc).__name__}: {exc}")
        raise
    # A day with no orders and a day the scan never ran are both an empty
    # paper_order table. The run log is the only place that difference can
    # live, so the narrowing note goes there whether or not anything was found.
    note = result["narrowing_note"]
    if not result["candidates"]:
        note = (f"no candidates: {note}; "
                f"{len(result['undetermined'])} undetermined, "
                f"{len(result['rejected'])} rejected")
    pit_store.finish_run(rows_written=written, status="ok", error=note)
    return {**result, "written": written}


# ------------------------------------------------------------- entry point

def main(argv: Optional[List[str]] = None) -> int:
    """`python -m research.scanner`.

    Its own entry point rather than a stage of the recorder. Recording is not
    repeatable -- a night missed is a night gone -- while a scan is a decision
    over a record that already exists and can be re-run against the same day
    whenever the parameters change. Sharing one command would mean re-recording
    in order to re-decide.

    A scan that finds nothing exits 0. An empty tape is not a failure, and
    paging someone for one teaches them to ignore the pager.
    """
    import argparse
    import json

    parser = argparse.ArgumentParser(
        prog="scanner",
        description="Rank today's candidates net of cost and file them.")
    parser.add_argument("--as-of", dest="as_of", default=None,
                        help="date to decide as (default: today)")
    args = parser.parse_args(argv)

    try:
        result = record_scan(as_of=args.as_of)
    except Exception as exc:  # noqa: BLE001 - reported, then non-zero
        print(json.dumps({"as_of": args.as_of, "status": "failed",
                          "error": f"{type(exc).__name__}: {exc}"}, indent=2))
        return 1

    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via main()
    raise SystemExit(main())
