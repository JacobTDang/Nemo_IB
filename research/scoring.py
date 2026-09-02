"""What the filed orders actually did, and what that implies about the drift.

The scanner logs an intention -- a ticker, a side, a size, and the session the
order is for -- and deliberately no price, because that session has not
happened when the row is written. This is where the two meet. Everything
upstream measures something; nothing upstream is ever checked against an
outcome, and measurement that is never checked is arithmetic.

It is also the only route to the one number the scanner admits it assumed.
`DRIFT_BPS_PER_SUE` is declared, reported uncalibrated on every scan, and every
net edge is proportional to it. The honest replacement is the coefficient this
book's own outcomes imply -- measured from realised returns, never echoed back
from the assumption.

That coefficient is a slope, and it is reported with an interval. A point
estimate on its own cannot be told from a number the sample happened to
produce: at the size of the replay arms actually run, 74 trades, a true 15
comes back with a 95% interval spanning roughly -50 to +57. So the estimate
carries its standard error, refuses to be quoted when the interval contains
zero, and says how many trades at the observed dispersion would pin it.

`calibrated` and `coefficient_usable` answer different questions and are kept
apart for that reason. The first says the book's returns clear their gates and
the coefficient's sign is not chance. The second says the coefficient is known
closely enough to price with -- within 50%, which is loose and still a much
harder bar. A slope of +64 give or take 97% has a sign and no size, and only
the second flag should ever be read by anything replacing a declared number.

It is also reported per unit of what the scanner multiplied, which is not
always the `sue` column: `ts` and `af` price a sigma, `cs` prices a rank's
distance into a tail, and the column holds half of the latter. One key each --
`drift_bps_per_sue` and `drift_bps_per_tail` -- so a tail coefficient can never
be read as a sigma one.

The failure modes of a scorer all point the same way, toward flattering the
strategy:

  Scoring an unfinished horizon. Not a small error: the trades still open are
  disproportionately the ones that have not gone anywhere, so counting them
  early drags every average toward nothing having happened.

  Filling an order that never filled. A name that did not trade on its intended
  session is not a position bought at the next available price; it is a
  position nobody has. Sliding to the next print is how a study buys the day
  after bad news.

  Reading a split as a return. The store keeps as-traded prices on purpose, so
  a 10-for-1 is a real -90% print sitting in the middle of a holding period.
  Everything here reads `adjusted_bars`, which rebuilds from actions the
  scorer could have known.

  Dropping the cost that was already measured and written down at decision
  time. It is on the order row; there is no excuse for scoring gross.
"""
from __future__ import annotations

import statistics
from typing import Any, Dict, List, Optional

from research import pit_store, spread

# How many scored trades before the realised coefficient is worth quoting. One
# trade is not a coefficient, and replacing a declared assumption with a
# one-sample estimate is trading a stated guess for an unstated one.
MIN_CALIBRATION_SAMPLE = 30

DEFAULT_HORIZON_DAYS = 20


def _today() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).date().isoformat()


def score_orders(as_of: Optional[str] = None,
                 horizon_days: int = DEFAULT_HORIZON_DAYS,
                 comparisons: int = 1) -> Dict[str, Any]:
    """Score every filed order whose holding period has finished by `as_of`.

    Reads only what a reader standing on `as_of` could have seen, like
    everything else here -- so a scorer run today cannot use a bar that was
    back-filled tomorrow, and re-running it for a past date gives that date's
    answer rather than today's.

    `comparisons` is how many variants have been tried against this book, and
    it is the caller's to state because nothing here can know it. It had no way
    in at all until now: the correction lived in `_summarise`, the scheduled
    job called this, and this scored everything at one comparison.
    """
    as_of = as_of or _today()
    # On the record before any work, so a crash leaves a started row with no
    # finish -- which `missing_days` correctly refuses to count as coverage.
    # This job is the one that checks the others, so a week it silently did
    # not run is the most expensive gap in the log.
    run_id = pit_store.start_run("score", as_of_date=as_of)
    orders = [o for o in pit_store.paper_orders_as_of(as_of, accepted_only=True)]

    scored: List[Dict[str, Any]] = []
    pending: List[Dict[str, Any]] = []
    unfilled: List[Dict[str, Any]] = []

    for order in orders:
        entry_session = order["intended_session"]
        if not entry_session:
            unfilled.append({**_stub(order),
                             "reason": "no intended session on the order"})
            continue

        bars = pit_store.adjusted_bars(order["ticker"], as_of)
        forward = [b for b in bars if b["trade_date"] >= entry_session]

        entry = next((b for b in forward if b["trade_date"] == entry_session),
                     None)
        if entry is None and _exchange_shut(entry_session, as_of):
            # The scanner names the next weekday, which about ten times a year
            # is a holiday. A real order rests and fills on the next open, so
            # this rolls one session -- and one only, named off the exchange
            # calendar rather than off this name's own prints. Rolled to
            # whenever the name next appeared, an order for Thanksgiving filled
            # thirteen sessions later at a price 40% lower and scored the gap
            # it was supposed to be holding through.
            next_open = _next_open_session(entry_session, as_of)
            entry_session = next_open or entry_session
            entry = next((b for b in forward
                          if b["trade_date"] == entry_session), None)
        if entry is None:
            # Two different nothings. If the horizon has not elapsed yet the
            # session may simply not have been recorded; if it has, the name
            # did not trade while the exchange was open, and the order never
            # filled. Counted on the exchange calendar, because counting this
            # name's own bars answers the question with the very gap being
            # asked about.
            later = [d for d in _reference_sessions(as_of) if d > entry_session]
            if len(later) < horizon_days:
                pending.append({**_stub(order),
                                "reason": f"{entry_session} not yet recorded"})
            else:
                unfilled.append({
                    **_stub(order),
                    "reason": (f"{order['ticker']} did not trade on "
                               f"{entry_session}, so the order never filled")})
            continue

        exit_bar, status, why = _horizon_exit(forward, entry_session,
                                              horizon_days, as_of)
        if exit_bar is None:
            bucket = pending if status == "pending" else unfilled
            bucket.append({**_stub(order), "reason": why})
            continue

        row = fill(order, entry, exit_bar)
        if row is None:
            unfilled.append({**_stub(order),
                             "reason": "a price on the path is missing"})
            continue
        scored.append({**row,
                       "expected_edge_bps": order["expected_edge_bps"],
                       "timing": _timing_of(order, as_of)})

    # Scored, not written: this job writes no rows. `rows_written` is what a
    # reader compares against zero, and for a scorer the count that means "it
    # did something" is how many trades it could close.
    # `ok` with zero rows, not `closed`. In this store `closed` means the
    # exchange was shut, and `missing_days` counts it as coverage for that
    # reason -- borrowing the word for "nothing had finished yet" would make
    # a scorer's quiet week indistinguishable from a market holiday.
    pit_store.finish_run(
        rows_written=len(scored), status="ok",
        error=None if scored else "no finished horizons to score",
        run_id=run_id)

    return {"as_of": as_of, "horizon_days": horizon_days,
            "scored": scored, "pending": pending, "unfilled": unfilled,
            "by_timing": split_by_timing(scored, comparisons),
            "by_variant": split_by_variant(scored, comparisons),
            **_summarise(scored, comparisons=comparisons)}


def split_by_variant(scored: List[Dict[str, Any]],
                     comparisons: int = 1) -> Dict[str, Any]:
    """The same numbers per signal variant, which is the only way to read them.

    `sue` holds a sigma for the time-series and analyst variants and a rank for
    the cross-sectional one, so a book spanning a change to `SIGNAL_VARIANT`
    holds two incomparable quantities in one column. Split, each half is
    measurable; averaged, neither is.
    """
    names = sorted({r.get("variant") or "unknown" for r in scored})
    return {name: _summarise(
        [r for r in scored if (r.get("variant") or "unknown") == name],
        comparisons=comparisons * len(names)) for name in names}


def split_by_timing(scored: List[Dict[str, Any]],
                    comparisons: int = 1) -> Dict[str, Any]:
    """The same numbers per announcement hour, each judged against the split.

    Cutting one sample into two or three and asking each half to clear t>2.00
    is two or three tests wearing one test's clothes: with enough hours to cut
    on, one of them clears. So each subgroup is told how many subgroups were
    tried, multiplied by whatever the caller was already counting.
    """
    hours = sorted({r.get("timing") or "unknown" for r in scored})
    return {hour: _summarise(
        [r for r in scored if (r.get("timing") or "unknown") == hour],
        comparisons=comparisons * len(hours)) for hour in hours}


def _timing_of(order: Dict[str, Any], as_of: str) -> str:
    """Whether the print landed before the open or after the close.

    A nightly scan decides on the evening of D and enters at D+1's open. For a
    print after the close on D that is entering after the gap, which is what
    drift means. For one before the open on D the gap was that morning, so by
    D+1's open a day of the effect has already gone. The two are not the same
    trade and an average over both hides it.
    """
    period = order.get("fiscal_period")
    if not period:
        return "unknown"
    for row in pit_store.announcements_as_of(order["ticker"], as_of):
        if row["fiscal_period"] == period:
            return row.get("timing") or "unknown"
    return "unknown"


class ReferenceSeriesMissing(LookupError):
    """No reference bars, so nothing here knows what a session is."""


def _reference_sessions(as_of: str) -> List[str]:
    """The exchange's own calendar, as the recorder has it on `as_of`.

    The reference instrument trades every session there is, so its bars are the
    only calendar the store holds. Absent, this used to answer "the market was
    open" -- which turned a hole in the recorder into a verdict about every
    order dated on a session the name did not print, and deleted them from the
    sample rather than reporting the hole.
    """
    reference = pit_store.bars_as_of(spread.REFERENCE_TICKER, as_of)
    if not reference:
        raise ReferenceSeriesMissing(
            f"{spread.REFERENCE_TICKER} has no bars on or before {as_of}, so "
            f"there is no session calendar to score against. That is a gap in "
            f"the recorder, not a fact about any order in the book")
    return [b["trade_date"] for b in reference]


def _exchange_shut(session: str, as_of: str) -> bool:
    """Whether the market was closed that day, as the recorder decides it.

    If the reference has no bar either, the exchange was shut and an order
    resting on that date fills at the next open. If it traded and this name did
    not, the name is the problem.
    """
    return session not in _reference_sessions(as_of)


def _next_open_session(session: str, as_of: str) -> Optional[str]:
    """The first session the exchange held after `session`, or None."""
    later = [d for d in _reference_sessions(as_of) if d > session]
    return later[0] if later else None


def _horizon_exit(forward: List[Dict[str, Any]], entry_session: str,
                  horizon_days: int, as_of: str):
    """The bar `horizon_days` sessions after entry, or why there is not one.

    Sessions of the exchange, not rows in the store. `forward[horizon_days]`
    is the 21st bar the store happens to hold, and a partial night is
    documented as normal on a universe this size -- so a name missing one bar
    was held for 21 sessions and one missing three for 23, silently, and the
    pending/unfilled split inherited the same error. A twenty-day horizon that
    is not the same length across the sample is not a horizon.

    Returns (bar, status, reason) with status "ok", "pending" -- the session
    has not happened yet -- or "unfilled", the name did not print on it and
    the nearest print is a different trade.
    """
    sessions = _reference_sessions(as_of)
    if entry_session not in sessions:
        return None, "unfilled", (
            f"{entry_session} is not a session the exchange held, so a "
            f"{horizon_days}-session horizon cannot be counted from it")

    start = sessions.index(entry_session)
    if start + horizon_days >= len(sessions):
        return None, "pending", (
            f"{len(sessions) - 1 - start} of {horizon_days} sessions elapsed "
            f"since {entry_session}")

    exit_session = sessions[start + horizon_days]
    bar = next((b for b in forward if b["trade_date"] == exit_session), None)
    if bar is None:
        return None, "unfilled", (
            f"no bar for {exit_session}, the session {horizon_days} after "
            f"{entry_session}; the nearest print is a different holding period")
    return bar, "ok", None


def fill(order: Dict[str, Any], entry: Dict[str, Any],
         exit_bar: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """One trade's arithmetic, shared by the live path and by replay.

    They differ only in where the orders come from -- the filed table or a
    replayed run -- and this used to be copied between them, which is how a fix
    to one silently stops applying to the other.

    Entry and exit are opens. The close of the day a decision was made is a
    price that was already known when the decision was made, and using it would
    hand the study the move it was deciding about.
    """
    entry_price = entry.get("open")
    exit_price = exit_bar.get("open")
    if not entry_price or not exit_price:
        return None

    move = (exit_price / entry_price) - 1.0
    if order.get("side") == "short":
        move = -move
    gross_bps = move * 10_000
    cost_bps = order.get("cost_bps") or 0.0
    # The carry, which the round trip does not contain. A short that never paid
    # its borrow shows a realised return nobody could have earned, and that
    # return is what the calibration gate reads.
    #
    # Absent is not zero, and it is not a reason to drop the trade either: a
    # row filed before the cost model had a borrow term genuinely was decided
    # without one. Charging it a rate invented today would be worse -- the
    # record is append-only precisely so a later opinion cannot be applied to
    # an earlier decision -- so it is scored as it was and counted as unpriced.
    borrow = order.get("borrow_bps")
    priced = borrow is not None or order.get("side") != "short"
    borrow_bps = borrow or 0.0
    return {
        **_stub(order),
        "entry_session": entry["trade_date"], "entry_price": entry_price,
        "exit_session": exit_bar["trade_date"], "exit_price": exit_price,
        "gross_bps": gross_bps, "cost_bps": cost_bps,
        "borrow_bps": borrow_bps, "borrow_priced": priced,
        "net_bps": gross_bps - cost_bps - borrow_bps,
    }


def _stub(order: Dict[str, Any]) -> Dict[str, Any]:
    """The identifying fields, read tolerantly.

    `fill` reads side and cost with .get and this hard-indexed three more, so
    one order missing a field took down the whole scoring run rather than that
    order. None of these enter the arithmetic -- they name the trade -- so an
    absent one is worth reporting, not worth stopping for.
    """
    return {"ticker": order.get("ticker"),
            "as_of_date": order.get("as_of_date"),
            "side": order.get("side"), "sue": order.get("sue"),
            # Which quantity that `sue` is. Without it the summary cannot tell
            # a sigma from a rank, and averages over both.
            "variant": order.get("variant"),
            # What the coefficient was multiplied by at decision time. Read
            # rather than re-derived; see `_priced_strength`.
            "strength": order.get("strength"),
            "fiscal_period": order.get("fiscal_period"),
            "target_dollars": order.get("target_dollars")}


# What the coefficient is per unit of, by variant. `ts` and `af` price a
# standardised surprise -- a sigma. `cs` prices a rank's distance into a tail,
# which is a different quantity with a different declared coefficient, and one
# name over both is a coefficient of neither.
UNIT_OF = {"ts": "sigma", "af": "sigma", "ts_release": "sigma", "cs": "tail"}

# Two-sided 95%.
CI_Z = 1.96

# How narrow the coefficient's interval has to be before it is worth calling a
# measurement, as a fraction of the estimate. Half is generous: a coefficient
# known to plus or minus 50% still moves every expected edge in the book by
# that much. Anything looser is not a replacement for a declared number, it is
# a differently-sourced guess with a confidence interval attached.
TARGET_HALF_WIDTH = 0.5


def _priced_strength(row: Dict[str, Any]) -> Optional[float]:
    """The quantity the scanner multiplied by the coefficient, for this row.

    Not `|sue|`. That column holds a sigma for `ts` and `af` and
    `percentile - 0.5` for `cs`, while the scanner prices `cs` with the tail
    distance -- twice that. Dividing the realised move by the column offered a
    cross-sectional coefficient exactly 2x the one that could replace
    `DRIFT_BPS_PER_TAIL`, under a name that read like a sigma.

    The scanner records `strength` now. The derivation below is for rows filed
    before that column existed: it is exact given the variant, not a guess, and
    a row whose variant is unknown is left out rather than assumed to be one.
    """
    strength = row.get("strength")
    if strength is not None:
        return abs(strength) or None

    sue = row.get("sue")
    if sue in (None, 0):
        return None
    variant = row.get("variant")
    if variant == "cs":
        return abs(sue) * 2.0
    if variant in ("ts", "af"):
        return abs(sue)
    return None


def _slope_through_origin(points: List[tuple]) -> tuple:
    """Least squares slope with no intercept, and its standard error.

    A coefficient in basis points per unit of surprise IS a slope, and the mean
    of per-trade ratios is a worse estimator of it: it weights a |SUE| of 1 the
    same as a |SUE| of 5, so a fixed noise in basis points counts five times as
    heavily where the signal is weakest. Simulated at the declared 15 over the
    scanner's own admission band, the ratio average is about 48% noisier at
    every sample size.

    The standard error is the point of it. Without one there is no way to tell
    a measured coefficient from a number the sample happened to produce, and
    the gate below was reading a t-statistic on returns, which answers whether
    the book made money rather than whether this coefficient is known.
    """
    sxx = sum(x * x for x, _ in points)
    if not sxx:
        return None, None
    beta = sum(x * y for x, y in points) / sxx
    if len(points) < 2:
        return beta, None
    residual = sum((y - beta * x) ** 2 for x, y in points)
    return beta, (residual / (len(points) - 1) / sxx) ** 0.5


def _coefficient_note(beta, unit, width, projection) -> str:
    """What is known about the coefficient, in one line, always said."""
    if beta is None or unit is None:
        return "no coefficient could be fitted to this sample"
    if width is None:
        return (f"{beta:+.1f} bps per {unit} from a sample too small to carry "
                f"a standard error")
    if width <= TARGET_HALF_WIDTH:
        return (f"{beta:+.1f} bps per {unit}, give or take {width:.0%} -- "
                f"inside the {TARGET_HALF_WIDTH:.0%} this treats as knowing it")
    return (f"{beta:+.1f} bps per {unit}, give or take {width:.0%}. That is a "
            f"sign, not a size: about {projection:,} trades at this dispersion "
            f"would pin it to within {TARGET_HALF_WIDTH:.0%}")


def _summarise(scored: List[Dict[str, Any]],
               comparisons: int = 1) -> Dict[str, Any]:
    """The realised numbers, and every reason a coefficient may not be quoted.

    The coefficient is gross of cost on purpose. Cost is a property of how a
    position was traded; drift is a property of the surprise, and mixing them
    gives a coefficient that moves with the size of the book.

    Sample size used to be the only gate, and a live replay walked straight
    through it: 223 trades, mean net +52.7bp, median -70.9bp, hit rate 48%. The
    mean was a handful of large winners and the typical trade lost money.
    Adopting 33.7bp per SUE off that would have replaced a stated assumption of
    15 with something 2.2x larger and made every net edge in the scanner
    inherit it. So the mean now has to survive the sample it came from:

      enough trades, because a handful is not a coefficient;

      a median on the same side as the mean, because a mean carried by its tail
      describes a lottery rather than an edge;

      a t-statistic clear of two, because a sign that cannot be told from
      chance is not a measurement;

      and an interval on the coefficient itself that does not contain zero. The
      three above are all about the book's returns, which is a different
      question from whether this coefficient is known: a sample can make money
      reliably and still not say what the coefficient is to within a factor of
      four. That gap is where a declared 15 would have been replaced by noise.

    `comparisons` moves that last bar out when the result was chosen from
    several. Three variants were replayed on the same names -- the time-series
    signal, the cross-sectional one entered at the earnings release, and the
    same one entered at the 10-Q -- and the third came back t=+2.47 on 71
    trades, clearing every gate and offering to replace a declared 40bp with
    340. One of three passing at t>2 is about what chance produces. A gate that
    cannot see the search ratifies it, so it is told the count and applies a
    Bonferroni correction: blunt, conservative, and honest about which bar the
    number actually cleared.
    """
    if not scored:
        return {"sample": 0, "hit_rate": None, "mean_net_bps": None,
                "median_net_bps": None, "mean_gross_bps": None,
                "t_stat": None, "t_threshold": None,
                "comparisons": comparisons, "variants": [],
                "shorts_without_borrow": 0,
                "drift_bps_per_sue": None, "drift_bps_per_tail": None,
                "drift_bps_prices": None, "drift_bps_se": None,
                "drift_bps_ci": None, "interval_pct_of_estimate": None,
                "trades_for_a_50pct_interval": None,
                "coefficient_usable": False,
                "coefficient_note": "no finished trades to measure",
                "calibrated": False,
                "calibration_note": "no finished trades to measure"}

    nets = [r["net_bps"] for r in scored]
    grosses = [r["gross_bps"] for r in scored]
    # Against what the scanner priced with, not against the `sue` column. See
    # `_priced_strength`.
    points = [(x, r["gross_bps"]) for r in scored
              for x in (_priced_strength(r),) if x]

    # A sigma and a rank are not the same quantity. The cross-sectional variant
    # stores `percentile - 0.5`, so its |sue| is bounded by 0.5 however large
    # the surprise, and gross/|sue| over a mixed book read a 100bp winner as
    # 220 basis points per SUE against a declared 15 -- a coefficient of
    # neither signal. The split is reported per variant instead; this refuses.
    variants = sorted({r.get("variant") or "unknown" for r in scored})
    sample = len(scored)
    unit = UNIT_OF.get(variants[0]) if len(variants) == 1 else None
    beta, beta_se = _slope_through_origin(points) if points else (None, None)
    # A single trade has a slope and no interval. The sample gate already
    # refuses it; withholding the number as well would say "not measurable"
    # where the truth is "measured once", which is a different fact.
    interval = ((beta - CI_Z * beta_se, beta + CI_Z * beta_se)
                if beta is not None and beta_se is not None else None)
    quotable = beta if unit is not None else None
    # How wide the interval is against the estimate, and how many trades at
    # this sample's own dispersion would bring it to TARGET_HALF_WIDTH. The
    # standard error falls as 1/sqrt(n), so the projection is quadratic in how
    # much narrower it has to get. Without this, `calibrated: False` does not
    # distinguish two more months of recording from twenty more years.
    if beta and beta_se is not None:
        width = CI_Z * beta_se / abs(beta)
        # What a 50% interval takes, not what is still missing. Below the
        # current sample it says the interval is already there, which is worth
        # reading as plainly as the other direction.
        projection = round(sample * (width / TARGET_HALF_WIDTH) ** 2)
    else:
        # A coefficient of zero has no interval that is 50% of it, and a single
        # trade has no standard error. Neither is a large number.
        width = projection = None

    unpriced = sum(1 for r in scored if r.get("borrow_priced") is False)
    mean_net = statistics.fmean(nets)
    median_net = statistics.median(nets)
    if sample > 1:
        sd = statistics.stdev(nets)
        t_stat = (mean_net / (sd / (sample ** 0.5))) if sd else None
    else:
        t_stat = None

    failures = []
    if len(variants) > 1:
        failures.append(
            f"this window mixes {len(variants)} signal variants "
            f"({', '.join(variants)}); a sigma and a rank are different "
            f"quantities and one coefficient over both is a coefficient of "
            f"neither. See by_variant")
    if sample < MIN_CALIBRATION_SAMPLE:
        failures.append(
            f"sample of {sample} is under the {MIN_CALIBRATION_SAMPLE} this "
            f"will quote a coefficient from")
    if unit is None and len(variants) == 1:
        failures.append(
            f"these rows do not say which signal variant they came from, so "
            f"it is not known whether sue holds a sigma or a rank -- and the "
            f"two take different coefficients. No coefficient is quoted")
    if quotable is not None and interval and interval[0] <= 0 <= interval[1]:
        failures.append(
            f"the coefficient's 95% interval is {interval[0]:+.0f} to "
            f"{interval[1]:+.0f} bps per {unit} and contains zero, so this "
            f"sample does not measure a coefficient however the returns came "
            f"out" + (
                f"; about {projection:,} trades at this dispersion would pin "
                f"it to within {TARGET_HALF_WIDTH:.0%}"
                if projection else ""))
    if unpriced:
        failures.append(
            f"{unpriced} of {sample} trades are shorts that never paid a "
            f"borrow, so their net return is overstated by whatever it would "
            f"have been -- 23bp at a 3% rate over twenty sessions, against a "
            f"mean of {mean_net:+.1f}bp")
    if (mean_net > 0) != (median_net > 0):
        failures.append(
            f"the mean ({mean_net:+.1f}bp) and the median ({median_net:+.1f}bp) "
            f"fall on opposite sides of zero, so the average is carried by its "
            f"tail rather than by the typical trade")
    # Bonferroni on the two-sided normal quantile: one comparison keeps the
    # familiar 2.0, and each additional one moves the bar out.
    if comparisons > 1:
        from statistics import NormalDist
        threshold = NormalDist().inv_cdf(1 - 0.05 / (2 * comparisons))
    else:
        threshold = 2.0

    if t_stat is None or abs(t_stat) < threshold:
        shown = "undefined" if t_stat is None else f"{t_stat:+.2f}"
        extra = (f" at the bar for {comparisons} comparisons"
                 if comparisons > 1 else "")
        failures.append(
            f"t={shown} does not clear {threshold:.2f}{extra}, so the sign of "
            f"this mean is not distinguishable from chance at this sample size")

    return {
        "sample": sample,
        "hit_rate": sum(1 for n in nets if n > 0) / sample,
        # Shorts in this sample whose net return never paid a borrow, because
        # they were decided before the cost model had one. Their net is
        # overstated by whatever the borrow was, and the gate below reads nets.
        "shorts_without_borrow": unpriced,
        "mean_net_bps": mean_net,
        "median_net_bps": median_net,
        "mean_gross_bps": statistics.fmean(grosses),
        "t_stat": t_stat,
        "t_threshold": threshold,
        "comparisons": comparisons,
        "variants": variants,
        # One key per unit, never one key for both. A tail coefficient read
        # as a sigma coefficient is the 2x error this split exists to stop.
        "drift_bps_per_sue": quotable if unit == "sigma" else None,
        "drift_bps_per_tail": quotable if unit == "tail" else None,
        "drift_bps_prices": unit if quotable is not None else None,
        "drift_bps_se": beta_se,
        "drift_bps_ci": interval,
        "interval_pct_of_estimate": width,
        "trades_for_a_50pct_interval": projection,
        # Two claims, kept apart. `calibrated` says the book's returns clear
        # their gates and the coefficient's sign is not chance. This says the
        # coefficient is known closely enough to price with -- which is the
        # only question that matters to anything replacing a declared number,
        # and a strictly harder one. A slope of +64 give or take 97% has a
        # sign and no size.
        "coefficient_usable": bool(
            not failures and width is not None
            and width <= TARGET_HALF_WIDTH),
        "coefficient_note": _coefficient_note(beta, unit, width, projection),
        "calibrated": not failures,
        "calibration_note": (f"{sample} finished trades, median and mean agree, "
                             f"t={t_stat:+.2f} against a bar of "
                             f"{threshold:.2f}" if not failures
                             else "; ".join(failures)),
    }


# ------------------------------------------------------------- entry point

def main(argv: Optional[List[str]] = None) -> int:
    """`python -m research.scoring`. Reports; changes nothing."""
    import argparse
    import json

    # Nothing else does, and the ordering that hides it is not enforced
    # anywhere: the recorder normally runs first and creates the store, so the
    # first command against a fresh volume dies on "no such table" instead.
    # Cheap and idempotent, so it runs every time rather than once.
    pit_store.init_schema()

    parser = argparse.ArgumentParser(
        prog="scoring",
        description="Score filed paper orders whose horizon has finished.")
    parser.add_argument("--as-of", dest="as_of", default=None)
    parser.add_argument("--horizon-days", dest="horizon_days", type=int,
                        default=DEFAULT_HORIZON_DAYS)
    parser.add_argument("--comparisons", type=int, default=1,
                        help="how many variants have been tried against this "
                             "book; the significance bar moves out with it")
    args = parser.parse_args(argv)

    result = score_orders(as_of=args.as_of, horizon_days=args.horizon_days,
                          comparisons=args.comparisons)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via main()
    raise SystemExit(main())
