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

def _signal_for(ticker: str, as_of: str) -> Dict[str, Any]:
    return sue.sue_ts(ticker, as_of=as_of)


def _cost_for(ticker: str, as_of: str, position_dollars: float) -> Dict[str, Any]:
    """Round-trip cost, with the floor beside the bound.

    EDGE cannot resolve a mega-cap's spread from daily bars. Live at 252
    sessions MSFT, AAPL and JPM all came back unresolved, so the model charged
    the 95% upper bound -- 19.8bp, 39.6bp and 54.0bp against true spreads near
    a basis point. Charging the bound is the right conservative default and it
    is also, for these names, mostly sampling error.

    The floor is knowable: one tick against the price, which no spread can be
    tighter than. Carrying both turns an unmeasurable number into a stated
    range, so a name the estimator could not judge stops looking like a name
    the strategy turned down.
    """
    cost = spread.round_trip_cost(ticker, as_of, position_dollars,
                                  window=spread.RESOLVING_WINDOW)
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


def scan(as_of: Optional[str] = None) -> Dict[str, Any]:
    """Today's candidates, ranked net of cost, with every rejection kept.

    Rejections are half the output. A scanner that quietly stops finding
    anything looks exactly like a market with nothing in it, and the difference
    only shows up in the reasons.
    """
    as_of = as_of or _today()
    scale, regime = _regime_scale(as_of)
    gross = GROSS_TARGET * scale
    per_name = gross / MAX_NAMES

    universe = [m["ticker"] for m in pit_store.universe_as_of(as_of)
                if m["eligible"]]

    candidates: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    undetermined: List[Dict[str, Any]] = []
    measured = 0
    resolved_count = 0

    for ticker in universe:
        signal = _signal_for(ticker, as_of)
        problem = _signal_problem(signal, as_of)
        if problem:
            rejected.append({"ticker": ticker, "reason": problem,
                             "sue": (signal or {}).get("sue")})
            continue

        value = signal["sue"]
        side = "long" if value > 0 else "short"

        # Size first, because the cost depends on it: impact is a function of
        # the order against the flow, so a cost quoted without a size is a
        # cost for some other trade.
        fit = spread.participation_rate(ticker, as_of, per_name,
                                        window=spread.RESOLVING_WINDOW)
        if fit.get("reason"):
            rejected.append({"ticker": ticker, "reason": fit["reason"],
                             "sue": value})
            continue
        target = min(per_name, fit["max_position_dollars"])
        if target <= 0:
            rejected.append({"ticker": ticker, "sue": value,
                             "reason": "no tradeable size at this liquidity"})
            continue

        cost = _cost_for(ticker, as_of, target)
        if cost.get("cost") is None:
            rejected.append({"ticker": ticker, "sue": value,
                             "reason": cost.get("reason")
                             or "cost could not be measured"})
            continue

        expected_bps = abs(value) * DRIFT_BPS_PER_SUE
        cost_bps = cost["cost"] * 10_000
        floor_bps = (cost.get("cost_floor") or cost["cost"]) * 10_000
        net_bps = expected_bps - cost_bps
        resolved = bool(cost.get("resolved"))
        measured += 1
        if resolved:
            resolved_count += 1

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
        "candidates": candidates[:MAX_NAMES],
        "rejected": rejected,
        # Names the estimator could not judge, kept apart from names the
        # strategy turned down.
        "undetermined": undetermined,
        "costs_total": measured,
        "costs_resolved": resolved_count,
        "assumptions": {
            "drift_bps_per_sue": DRIFT_BPS_PER_SUE,
            "calibrated": DRIFT_CALIBRATED,
            "note": ("drift per unit of SUE is assumed, not measured here; "
                     "every net edge is proportional to it"),
        },
    }


def record_scan(as_of: Optional[str] = None) -> Dict[str, Any]:
    """Run a scan and file it, candidates and rejections alike.

    Filed rather than returned so it can be scored against what the market
    actually did, which is the only test of any of this. Idempotent within a
    day for the same reason the bar recorder is: a second run must not double
    the book.
    """
    as_of = as_of or _today()
    result = scan(as_of)
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
    pit_store.finish_run(rows_written=written, status="ok")
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
