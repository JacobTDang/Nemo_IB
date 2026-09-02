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
from every trade in the study. The same applies to the cost of being short: a
name nobody can price a borrow on is refused rather than charged nothing, and
charging nothing would put it top of a book ranked net of cost. See
`research.borrow`.

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

from research import borrow, pit_store, spread, sue

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

# And how large the surprise has to be in its own right, as a fraction of the
# share price. The tail distance is a rank, so some name is always in the top
# decile of any cohort -- including a season where the largest beat was a
# hundredth of a cent, which the rank cannot tell from a season of real news.
# Five basis points of price is roughly a few percent of a quarter's earnings
# for a typical name: small, and not a rounding error.
MIN_SCALED_SURPRISE = 0.0005

# Which jobs count as "the recorder", and how long one can be silent before an
# empty window stops being information about the tape. Five days spans a long
# weekend -- the recorder does not run on a Sunday and a Tuesday scan after
# Thanksgiving is not an emergency -- and does not span a month.
RECORDER_JOBS = ("daily_bars", "consensus")
MAX_RECORDER_GAP_DAYS = 5


class RecorderNotRunning(RuntimeError):
    """An empty window that nothing was watching. Not a quiet tape."""


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
    if SIGNAL_VARIANT == "ts_release":
        # The same surprise, dated from the 8-K rather than the 10-Q. See
        # `research.release_eps` for why the eight days matter.
        from research import release_eps
        return {**release_eps.sue_ts_release(ticker, as_of=as_of),
                "variant": "ts_release"}
    if SIGNAL_VARIANT == "cs":
        from research import sue_cs
        return {**sue_cs.surprise_rank(ticker, as_of=as_of,
                                       peers=_peers_for(as_of)),
                "variant": "cs"}
    raise ValueError(
        f"SIGNAL_VARIANT must be 'ts', 'af', 'af_or_ts', 'ts_release' or "
        f"'cs'; got "
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


def _borrow_for(ticker: str, as_of: str, side: str,
                declared_rate: Optional[float]) -> Dict[str, Any]:
    """What the position costs to finance over the holding period.

    Zero for a long. For a short it is a rate somebody quoted -- recorded in
    the store on a date, or declared by the caller -- over the calendar days a
    twenty-session hold actually covers, and a refusal when there is neither.

    A seam rather than a call so the decision logic can be exercised without a
    borrow feed, the same way `_cost_for` is.
    """
    return borrow.carry_cost(ticker, as_of, side, declared_rate=declared_rate)


def _recorder_gap(as_of: str) -> Optional[str]:
    """Why an empty window cannot be read as information, or None.

    `has_consensus_history` says only that the recorder wrote something once,
    ever. A recorder that died in January still satisfies it, so a scan in June
    narrows to the names that printed in the last six weeks, finds none, files
    an empty book and exits 0 -- a run-log line byte-identical to a genuinely
    quiet tape, every night, forever. The run log is where the difference
    lives, and nothing here had ever read it.
    """
    stale = []
    for job in RECORDER_JOBS:
        last = pit_store.last_successful_run(job, as_of)
        if last is None:
            stale.append(f"{job} has never finished successfully by {as_of}")
            continue
        gap = (date.fromisoformat(as_of) - date.fromisoformat(last)).days
        if gap > MAX_RECORDER_GAP_DAYS:
            stale.append(f"{job} last finished for {last}, {gap} days before "
                         f"{as_of}")
    return "; ".join(stale) if stale else None


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
    scaled = signal.get("scaled_surprise")
    if scaled is None:
        return "no scaled surprise, so the rank cannot be given a size"
    if abs(scaled) < MIN_SCALED_SURPRISE:
        return (f"surprise of {abs(scaled) * 1e4:.2f}bp of price is under the "
                f"{MIN_SCALED_SURPRISE * 1e4:.1f}bp floor; a rank always has a "
                f"top decile, so without this the largest of a cohort of "
                f"rounding errors is priced at a full tail")
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
         signal_for=None,
         borrow_rate: Optional[float] = None) -> Dict[str, Any]:
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

    `borrow_rate` is a flat annualised rate to charge every short the store has
    no quote for. There is deliberately no default: a short nobody can price is
    refused, because the alternative -- charging zero -- ranks the least
    borrowable name in the universe first. Declaring one is an assumption made
    out loud, and it is recorded on every row it priced.
    """
    as_of = as_of or _today()
    # A new scan must not inherit a cohort assembled for another date.
    _COHORT.clear()
    signal_for = signal_for or _signal_for
    if already_acted is None:
        already_acted = pit_store.filed_issuer_periods(as_of)
    scale, regime = _regime_scale(as_of)
    gross = GROSS_TARGET * scale
    per_name = gross / MAX_NAMES

    members = [m for m in pit_store.universe_as_of(as_of) if m["eligible"]]
    universe = [m["ticker"] for m in members]
    # Who each line belongs to. One issuer lists more than one -- eight
    # tickers share Morgan Stanley's CIK, fourteen ProShares funds share one
    # sponsor's -- and a print belongs to the issuer, not to the line.
    issuer_of = {m["ticker"]: m["cik"] for m in members if m["cik"]}
    refusal = None

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
        if not considered:
            # Nothing to scan is a fact about the tape only if something was
            # watching it. The gate is on the empty answer rather than on every
            # scan: a window with names in it is its own evidence that the
            # recorder works.
            gap = _recorder_gap(as_of)
            if gap:
                refusal = (
                    f"no eligible name has a print recorded since {since}, and "
                    f"the recorder is not known to have been running: {gap}. "
                    f"An empty window is information only when something was "
                    f"watching, and this cannot tell a quiet tape from a "
                    f"recorder that stopped")
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
    borrow_unpriced = 0

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
        # Two identities, both meaning "already acted", and the set may carry
        # either. A ticker key says this exact line traded; an issuer key says
        # this issuer's print did, whichever line carried it. Checking both is
        # a union of two rules rather than a fallback -- replay keys its own
        # set on the ticker, and the filed book is keyed on the issuer.
        #
        # `issuer_of.get(ticker)` and not `.get(ticker, ticker)`: an
        # unrecorded issuer must stay None rather than borrow the ticker,
        # or two names nobody screened would collapse into each other.
        issuer = issuer_of.get(ticker)
        if period and ((ticker, period) in already_acted
                       or (issuer is not None
                           and (issuer, period) in already_acted)):
            rejected.append({
                "ticker": ticker, "sue": signal.get("sue"),
                "reason": (f"already acted on this issuer's {period}; the "
                           f"signal stays fresh for {MAX_SIGNAL_AGE_DAYS} days "
                           f"and one print is one trade")})
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

        # The round trip is a crossing. A short is also a position held open,
        # and the stock loan bills for every calendar day of it -- 23bp at a 3%
        # borrow over twenty sessions, against a drift of a few tens of basis
        # points. Priced separately from `cost` because it is a different
        # quantity: one is charged twice on the way through, the other accrues.
        try:
            carry = _borrow_for(ticker, as_of, side, borrow_rate)
        except Exception as exc:  # noqa: BLE001 - recorded, not masked
            rejected.append({"ticker": ticker, "sue": value,
                             "reason": f"borrow unavailable: "
                                       f"{type(exc).__name__}: {exc}"})
            continue
        if carry.get("cost") is None:
            borrow_unpriced += 1
            rejected.append({"ticker": ticker, "sue": value,
                             "reason": carry.get("reason")
                             or "borrow could not be priced"})
            continue

        coefficient = (DRIFT_BPS_PER_TAIL if signal.get("variant") == "cs"
                       else DRIFT_BPS_PER_SUE)
        expected_bps = strength * coefficient
        cost_bps = cost["cost"] * 10_000
        borrow_bps = carry["cost"] * 10_000
        floor_bps = (cost.get("cost_floor") or cost["cost"]) * 10_000
        net_bps = expected_bps - cost_bps - borrow_bps
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
                   "borrow_bps": borrow_bps,
                   "expected_edge_bps": expected_bps}
            # Borrow is charged whether or not the spread resolved, so the
            # bound the edge has to clear carries it too. Without that a short
            # killed outright by its carry would be filed as a name the
            # estimator could not judge.
            if not resolved and expected_bps > floor_bps + borrow_bps:
                # The strategy did not turn this down; the estimator could not
                # tell. Filing it as a rejection would hide a name that trades
                # well inside its own bound, and would make a scanner starved
                # by measurement error look like one facing a quiet tape.
                row["reason"] = (
                    f"unresolved spread: expected {expected_bps:.1f}bp sits "
                    f"inside a cost band of {floor_bps + borrow_bps:.1f}-"
                    f"{cost_bps + borrow_bps:.1f}bp, so the estimator cannot "
                    f"say whether this clears")
                undetermined.append(row)
            else:
                carried = (f" and {borrow_bps:.1f}bp of borrow"
                           if borrow_bps else "")
                row["reason"] = (f"expected {expected_bps:.1f}bp does not clear "
                                 f"{cost_bps:.1f}bp of round-trip cost"
                                 f"{carried}")
                rejected.append(row)
            continue

        candidates.append({
            "ticker": ticker, "side": side, "sue": value,
            "fiscal_period": signal.get("fiscal_period"),
            "known_at": signal.get("known_at"),
            # What quantity `sue` is, what it was priced with, and how much of
            # it was reconstructed rather than watched. All three are module
            # constants or upstream fields at decision time and none of them
            # can be recovered from the row afterwards.
            "variant": signal.get("variant"),
            "issuer_cik": issuer,
            # The quantity the coefficient was multiplied by. `sue` is a sigma
            # for ts and af and percentile-0.5 for cs, which is half what cs
            # is priced on -- so a scorer deriving the multiplier from that
            # column offered a coefficient twice the one it would replace.
            "strength": strength,
            "drift_coefficient": coefficient,
            "drift_calibrated": DRIFT_CALIBRATED,
            "seeded_quarters": signal.get("seeded_quarters"),
            "recorded_quarters": signal.get("recorded_quarters"),
            "expected_edge_bps": expected_bps, "cost_bps": cost_bps,
            "cost_bps_low": floor_bps,
            # Carry, kept apart from the round trip. `cost_bps` has meant
            # spread-plus-impact since the first row was filed and folding
            # borrow into it would change what every historical row says.
            "borrow_bps": borrow_bps,
            "borrow_rate": carry.get("annual_rate"),
            "borrow_source": carry.get("rate_source"),
            "net_edge_bps": net_bps, "target_dollars": target,
            "participation": fit.get("participation"),
            "spread": cost.get("spread"),
            # `resolved` is EDGE's flag for "not zero", which SPY passes at
            # 41bp against a market a cent wide. What the row has to say is
            # whether the charge was this name's measured spread or the tick it
            # was floored to, which is what `resolution` answers.
            "spread_resolved": resolved,
            # The session this order is for. No price: that session has not
            # happened, and the whole point of logging an intention is that
            # nobody gets to choose the fill afterwards.
            "intended_session": _next_session(as_of),
        })

    candidates.sort(key=lambda c: c["net_edge_bps"], reverse=True)

    # One print is one trade, and a print belongs to an issuer rather than to
    # a line it lists under. Collapsed AFTER the ranking, deliberately: taking
    # whichever sibling the loop reached first would hand the book the worse
    # execution, and these differ only in what they cost to cross.
    #
    # A live scan without this put six Morgan Stanley preferred classes into a
    # twenty-name book -- same CIK, same quarter, same SUE of 3.95 -- which is
    # 30% of gross on one bank's earnings in a book sized for twenty
    # independent bets.
    best: Dict[Any, Dict[str, Any]] = {}
    kept: List[Dict[str, Any]] = []
    for candidate in candidates:
        # No recorded issuer means no claim that two names are the same one.
        key = (candidate.get("issuer_cik"), candidate.get("fiscal_period"))
        if key[0] is None or key[1] is None:
            kept.append(candidate)
            continue
        winner = best.get(key)
        if winner is None:
            best[key] = candidate
            kept.append(candidate)
            continue
        rejected.append({
            "ticker": candidate["ticker"], "sue": candidate["sue"],
            "reason": (f"{winner['ticker']} is the same issuer "
                       f"({key[0]}) on the same {key[1]} print and ranks "
                       f"higher at {winner['net_edge_bps']:.1f}bp net against "
                       f"{candidate['net_edge_bps']:.1f}bp; one print is one "
                       f"trade")})
    candidates = kept

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
        # Set when the scan cannot say what it found. Nothing downstream may
        # file this result; see `record_scan`.
        "refusal": refusal,
        "candidates": candidates[:MAX_NAMES],
        "rejected": rejected,
        # Names the estimator could not judge, kept apart from names the
        # strategy turned down.
        "undetermined": undetermined,
        "costs_total": priced,
        "costs_measured": measured_count,
        "costs_floored": floored_count,
        # Shorts turned down for want of a borrow rate rather than by the
        # strategy. Without this a book that has quietly gone long-only reads
        # like a tape with no bad prints in it.
        "borrow_unpriced": borrow_unpriced,
        "assumptions": {
            "variant": SIGNAL_VARIANT,
            "drift_bps_per_sue": DRIFT_BPS_PER_SUE,
            "drift_bps_per_tail": DRIFT_BPS_PER_TAIL,
            "calibrated": DRIFT_CALIBRATED,
            "note": ("drift is assumed, not measured here, and every net edge "
                     "is proportional to it. The two coefficients price "
                     "different quantities -- a sigma and a rank -- and are "
                     "deliberately not the same number"),
            "borrow": {
                "declared_rate": borrow_rate,
                "day_count": borrow.DAY_COUNT,
                "calibrated": False,
                "note": ("a declared rate is a blanket assumption about every "
                         "name; a rate recorded in the store for one name "
                         "beats it. With neither, the short is refused rather "
                         "than charged nothing"),
            },
        },
    }


def record_scan(as_of: Optional[str] = None,
                borrow_rate: Optional[float] = None) -> Dict[str, Any]:
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
    result = scan(as_of, borrow_rate=borrow_rate)

    # What this day already decided, if anything. Filing is append-only, so a
    # re-run cannot change it -- and should not, because the filed decision is
    # the one that would have been acted on. What it must not do is return a
    # different answer without mentioning that the record holds another one.
    #
    # Every row, not just the accepted ones. paper_order is keyed on
    # (as_of_date, ticker), so a rejection already occupies the key a later
    # acceptance would need -- and asking only about acceptances meant a first
    # run that accepted nothing looked like a date nobody had decided.
    filed_rows = [o for o in pit_store.paper_orders_as_of(as_of)
                  if o["as_of_date"] == as_of]
    decided = {o["ticker"] for o in filed_rows}
    already = {o["ticker"] for o in filed_rows if o["accepted"]}
    proposed = {c["ticker"] for c in result["candidates"]}
    if decided and already != proposed:
        result["superseded"] = {
            "filed": sorted(already), "proposed": sorted(proposed),
            "note": (f"{as_of} was already decided and that decision stands; "
                     f"these candidates were not filed")}

    pit_store.start_run("scan", as_of_date=as_of)
    if result["refusal"]:
        # Filing an empty book here would record a decision nobody is in a
        # position to make, and status=ok would say the night went fine.
        pit_store.finish_run(rows_written=0, status="failed",
                             error=result["refusal"])
        raise RecorderNotRunning(result["refusal"])
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

    # What the record holds now, which is not what was proposed whenever an
    # earlier row already owned the key. INSERT OR IGNORE drops the second
    # write silently, so the shortfall has to be looked for rather than
    # assumed absent.
    held = {o["ticker"] for o in
            pit_store.paper_orders_as_of(as_of, accepted_only=True)
            if o["as_of_date"] == as_of}
    dropped = sorted(proposed - held)
    if dropped:
        result["superseded"] = {
            "filed": sorted(held), "proposed": sorted(proposed),
            "dropped": dropped,
            "note": (f"{as_of} already holds a row for "
                     f"{', '.join(dropped)}, so nothing was filed for "
                     f"{'them' if len(dropped) > 1 else 'it'} and the "
                     f"decision on record is the earlier one")}
        note = f"{note}; superseded: {result['superseded']['note']}"
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

    # Nothing else does, and the ordering that hides it is not enforced
    # anywhere: the recorder normally runs first and creates the store, so the
    # first command against a fresh volume dies on "no such table" instead.
    # Cheap and idempotent, so it runs every time rather than once.
    pit_store.init_schema()

    parser = argparse.ArgumentParser(
        prog="scanner",
        description="Rank today's candidates net of cost and file them.")
    parser.add_argument("--as-of", dest="as_of", default=None,
                        help="date to decide as (default: today)")
    parser.add_argument("--borrow-rate", dest="borrow_rate", type=float,
                        default=None,
                        help="annualised borrow to charge every short the "
                             "store holds no quote for, e.g. 0.03 for 3%%. "
                             "Without it a short nobody can price is refused "
                             "rather than charged nothing; the count is "
                             "reported as borrow_unpriced")
    args = parser.parse_args(argv)

    try:
        result = record_scan(as_of=args.as_of, borrow_rate=args.borrow_rate)
    except Exception as exc:  # noqa: BLE001 - reported, then non-zero
        print(json.dumps({"as_of": args.as_of, "status": "failed",
                          "error": f"{type(exc).__name__}: {exc}"}, indent=2))
        return 1

    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via main()
    raise SystemExit(main())
