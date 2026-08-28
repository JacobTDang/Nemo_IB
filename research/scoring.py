"""What the filed orders actually did, and what that implies about the drift.

The scanner logs an intention -- a ticker, a side, a size, and the session the
order is for -- and deliberately no price, because that session has not
happened when the row is written. This is where the two meet. Everything
upstream measures something; nothing upstream is ever checked against an
outcome, and measurement that is never checked is arithmetic.

It is also the only route to the one number the scanner admits it assumed.
`DRIFT_BPS_PER_SUE` is declared, reported uncalibrated on every scan, and every
net edge is proportional to it. The honest replacement is the coefficient this
book's own outcomes imply, which is what `drift_bps_per_sue` here reports --
measured from realised returns, never echoed back from the assumption.

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
                 horizon_days: int = DEFAULT_HORIZON_DAYS) -> Dict[str, Any]:
    """Score every filed order whose holding period has finished by `as_of`.

    Reads only what a reader standing on `as_of` could have seen, like
    everything else here -- so a scorer run today cannot use a bar that was
    back-filled tomorrow, and re-running it for a past date gives that date's
    answer rather than today's.
    """
    as_of = as_of or _today()
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
        if entry is None and forward and _exchange_shut(entry_session, as_of):
            # The scanner names the next weekday, which about ten times a year
            # is a holiday. A real order rests and fills on the next open, so
            # this rolls one session -- and one only. Rolling until the name
            # next appears is how a study buys a week after the news.
            entry = forward[0]
            entry_session = entry["trade_date"]
        if entry is None:
            # Two different nothings. If the horizon has not elapsed yet the
            # session may simply not have been recorded; if it has, the name
            # did not trade while the exchange was open, and the order never
            # filled.
            later = [b for b in bars if b["trade_date"] > entry_session]
            if len(later) < horizon_days:
                pending.append({**_stub(order),
                                "reason": f"{entry_session} not yet recorded"})
            else:
                unfilled.append({
                    **_stub(order),
                    "reason": (f"{order['ticker']} did not trade on "
                               f"{entry_session}, so the order never filled")})
            continue

        if len(forward) <= horizon_days:
            pending.append({
                **_stub(order),
                "reason": (f"{len(forward) - 1} of {horizon_days} sessions "
                           f"elapsed since {entry_session}")})
            continue

        row = fill(order, entry, forward[horizon_days])
        if row is None:
            unfilled.append({**_stub(order),
                             "reason": "a price on the path is missing"})
            continue
        scored.append({**row,
                       "expected_edge_bps": order["expected_edge_bps"],
                       "timing": _timing_of(order, as_of)})

    by_timing = {}
    for hour in sorted({r.get("timing") or "unknown" for r in scored}):
        subset = [r for r in scored if (r.get("timing") or "unknown") == hour]
        by_timing[hour] = _summarise(subset)

    return {"as_of": as_of, "horizon_days": horizon_days,
            "scored": scored, "pending": pending, "unfilled": unfilled,
            "by_timing": by_timing, **_summarise(scored)}


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


def _exchange_shut(session: str, as_of: str) -> bool:
    """Whether the market was closed that day, as the recorder decides it.

    The reference instrument trades every session there is. If it has no bar
    either, the exchange was shut and an order resting on that date fills at
    the next open. If it traded and this name did not, the name is the problem.
    """
    reference = pit_store.bars_as_of(spread.REFERENCE_TICKER, as_of)
    if not reference:
        return False
    return not any(b["trade_date"] == session for b in reference)


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
    return {
        **_stub(order),
        "entry_session": entry["trade_date"], "entry_price": entry_price,
        "exit_session": exit_bar["trade_date"], "exit_price": exit_price,
        "gross_bps": gross_bps, "cost_bps": cost_bps,
        "net_bps": gross_bps - cost_bps,
    }


def _stub(order: Dict[str, Any]) -> Dict[str, Any]:
    return {"ticker": order["ticker"], "as_of_date": order["as_of_date"],
            "side": order["side"], "sue": order["sue"],
            "target_dollars": order["target_dollars"]}


def _summarise(scored: List[Dict[str, Any]],
               comparisons: int = 1) -> Dict[str, Any]:
    """The realised numbers, and three reasons a coefficient may not be quoted.

    `drift_bps_per_sue` is gross of cost on purpose. Cost is a property of how
    a position was traded; drift is a property of the surprise, and mixing them
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

      and a t-statistic clear of two, because a sign that cannot be told from
      chance is not a measurement.

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
                "comparisons": comparisons, "drift_bps_per_sue": None,
                "calibrated": False,
                "calibration_note": "no finished trades to measure"}

    nets = [r["net_bps"] for r in scored]
    grosses = [r["gross_bps"] for r in scored]
    usable = [(r["gross_bps"], abs(r["sue"])) for r in scored
              if r["sue"] not in (None, 0)]
    per_sue = [g / s for g, s in usable] if usable else []

    sample = len(scored)
    mean_net = statistics.fmean(nets)
    median_net = statistics.median(nets)
    if sample > 1:
        sd = statistics.stdev(nets)
        t_stat = (mean_net / (sd / (sample ** 0.5))) if sd else None
    else:
        t_stat = None

    failures = []
    if sample < MIN_CALIBRATION_SAMPLE:
        failures.append(
            f"sample of {sample} is under the {MIN_CALIBRATION_SAMPLE} this "
            f"will quote a coefficient from")
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
        "mean_net_bps": mean_net,
        "median_net_bps": median_net,
        "mean_gross_bps": statistics.fmean(grosses),
        "t_stat": t_stat,
        "t_threshold": threshold,
        "comparisons": comparisons,
        "drift_bps_per_sue": statistics.fmean(per_sue) if per_sue else None,
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

    parser = argparse.ArgumentParser(
        prog="scoring",
        description="Score filed paper orders whose horizon has finished.")
    parser.add_argument("--as-of", dest="as_of", default=None)
    parser.add_argument("--horizon-days", dest="horizon_days", type=int,
                        default=DEFAULT_HORIZON_DAYS)
    args = parser.parse_args(argv)

    result = score_orders(as_of=args.as_of, horizon_days=args.horizon_days)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via main()
    raise SystemExit(main())
