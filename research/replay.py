"""Measuring the drift coefficient before there is forward history to measure.

`scanner.DRIFT_BPS_PER_SUE` is declared, reported uncalibrated on every scan,
and every net edge is proportional to it. Until it is measured the whole stack
is a study of an assumption. The recorder answers this eventually and cannot
answer it now, so this answers it from history.

The honest way to do that is not an escape hatch in the store. `bars_as_of`
filters on `recorded_at` for a reason, and a flag that turns that off is the
kind of thing that gets left on -- after which every result in the project is
quietly contaminated and nothing says so. Replay builds its own store instead,
in which each bar is stamped at the session it describes. That is a claim, and
a narrow one: a day's prices were known that evening. It is true of prices, it
is the only thing being assumed, and everything downstream -- the screen, the
spread estimator, the scanner -- then works unmodified and correctly against
any past date.

Two biases follow and neither is fixable here, so both ride on every result:

  **Survivorship.** The vendor sells history only for names that still exist,
  so a replay universe is the set of companies that made it. Delisted names
  are absent, and they are absent for reasons correlated with returns. This
  inflates results and the direction is not in doubt.

  **No consensus.** What analysts expected on a past Tuesday is unrecoverable
  by any means, so the analyst variant of the surprise cannot be replayed at
  all. Only the time-series variant is available, computed from filings whose
  dates are on the public record.

  **Entry timing.** The surprise comes from XBRL, and XBRL does not exist until
  the 10-Q is filed -- measured on EDGAR, a median of 8 days after the first
  8-K following period end, mean 12.1, range 0 to 45. That is not a defect in
  the replay: a strategy reading XBRL genuinely cannot act sooner. It does mean
  the entry is days behind the announcement the published drift is dated from,
  and most of that drift happens early. So a null result here is a null result
  about an XBRL-timed strategy, not about post-earnings drift, and the way to
  close the gap is to read the number out of the 8-K where the market gets it.

Replayed decisions are filed in their own table. A decision that was made and a
decision that was imagined must never share one, or a scoring run reports the
imagined ones as results.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Dict, Iterable, List, Optional, Sequence

from research import daily_job, pit_store, scanner, spread, sue

_MISSING = object()

CAVEATS = (
    "survivorship: the vendor sells history only for names that still exist, "
    "so delisted companies are absent and their absence is correlated with "
    "returns; this inflates any result here and the direction is not in doubt",
    "no consensus: what analysts expected on a past date is unrecoverable, so "
    "only the time-series surprise is replayable and the analyst variant is "
    "not measured at all",
    "prices are stamped at their own session, which assumes a day's prices "
    "were known that evening; true of prices, and the only thing assumed",
    "scaled to 595 names the cross-sectional variant returns 340 trades, hit "
    "55%, mean +195bp and median +184bp at t=+2.04 -- which clears a bar of "
    "2.00 and not the 2.50 that four variants tried against these same names "
    "requires. Reported uncalibrated for that reason rather than for want of "
    "a sample",
    "the timing hypothesis was tested and not supported: the same "
    "cross-sectional signal entered at the earnings release returned t=+1.12 "
    "over 74 trades, and entered at the 10-Q a median of eight days later "
    "returned t=+2.47 over 71. Entering later did better, which is the "
    "opposite of what the drift literature predicts and is most likely a small "
    "sample rather than a finding -- the two arms do not trade the same events",
    "entry timing: the surprise is computed from XBRL, which does not exist "
    "until the 10-Q is filed -- a median of 8 days after the earnings 8-K, "
    "mean 12.1, range 0 to 45, measured across 60 filings from 20 large caps. "
    "The published drift is dated from the announcement and most of it happens "
    "in the first days, so a null result here is a null result about an "
    "XBRL-timed strategy and not about post-earnings drift",
)


def _stamp(trade_date: str) -> str:
    """Known on the evening of its own session. The one assumption."""
    return f"{trade_date}T21:00:00Z"


def build_store(tickers: Sequence[str], start: str, end: str) -> Dict[str, Any]:
    """Fetch history once and record each bar as known the day it happened.

    The reference instrument comes along whether or not it was asked for: the
    regime and the cost model both read it, and a replay without it silently
    runs at full size against an unmeasurable spread.
    """
    names = list(dict.fromkeys(t for t in tickers if t))
    if spread.REFERENCE_TICKER not in names:
        names.append(spread.REFERENCE_TICKER)

    fetched = daily_job._fetch_bars(
        names, start=start,
        end=(date.fromisoformat(end) + timedelta(days=1)).isoformat())

    written = 0
    for ticker, rows in fetched.items():
        as_traded = daily_job._to_as_traded(rows)
        for row in as_traded:
            when = _stamp(row["trade_date"])
            if row.get("split"):
                pit_store.record_corporate_action(
                    ticker, row["trade_date"], "split", float(row["split"]),
                    recorded_at=when)
            if row.get("dividend"):
                pit_store.record_corporate_action(
                    ticker, row["trade_date"], "dividend",
                    float(row["dividend"]), recorded_at=when)
            # Every row here is a backfill by construction: the session is
            # months past and the vendor has already dropped whatever delisted
            # since. Taking the default would have a replayed store claim its
            # bars were recorded on the evening they describe, which is the
            # one thing `source` exists to deny.
            written += pit_store.record_bars(ticker, [row], recorded_at=when,
                                             source="backfilled")

    return {"tickers": len(fetched), "rows": written,
            "start": start, "end": end, "caveats": list(CAVEATS)}


def record_prints(prints: Iterable[Dict[str, Any]]) -> int:
    """A filing's own date, standing in for the vendor calendar.

    The consensus recorder cannot be replayed. A filing date is on the record
    though, and it is when the reported figure became public -- which is what
    the scanner narrows on. No estimate is written beside it, because there was
    none and inventing one would manufacture the analyst surprise this
    deliberately does not compute.
    """
    written = 0
    for row in prints:
        known = row["known_at"]
        pit_store.record_consensus(
            known, row["ticker"], row["fiscal_period"],
            eps_estimate=None, eps_actual=row.get("eps"),
            recorded_at=_stamp(known))
        written += 1
    return written


def prints_from_filings(tickers: Sequence[str],
                        as_of: Optional[str] = None) -> List[Dict[str, Any]]:
    """Every quarter each name has filed, with the date it was filed.

    One EDGAR pass per ticker for the whole replay rather than one per ticker
    per decision date. The filing dates are point-in-time by construction, so
    fetching them all at once costs nothing in correctness.
    """
    out: List[Dict[str, Any]] = []
    for ticker in tickers:
        series = sue.eps_series(ticker, as_of=as_of)
        if not series.get("success"):
            continue
        for quarter in series["quarters"]:
            known = quarter.get("known_at")
            if not known:
                continue
            out.append({"ticker": ticker,
                        "fiscal_period": quarter["fiscal_period"],
                        "known_at": known[:10], "eps": quarter.get("eps")})
    return out


# Precomputed signals, keyed by ticker. A replay over 180 decision dates
# calling sue_ts per name per date is thousands of EDGAR requests for a series
# that does not change; sue_ts_history returns every quarter in one pass and
# agrees with a point-in-time sue_ts on each quarter's own filing date to
# fifteen digits.
_SIGNALS: Dict[str, List[Dict[str, Any]]] = {}


def load_signals(signals: Dict[str, List[Dict[str, Any]]]) -> int:
    """Install the precomputed table. Replaces whatever was there."""
    _SIGNALS.clear()
    for ticker, rows in signals.items():
        _SIGNALS[ticker] = sorted(
            (r for r in rows if r.get("known_at") and r.get("sue") is not None),
            key=lambda r: r["known_at"])
    return sum(len(v) for v in _SIGNALS.values())


def _changes_known_by(changes: Optional[List[Dict[str, Any]]],
                      known_at: str) -> List[Dict[str, Any]]:
    """The basis changes on the record when this quarter was filed.

    A change is evidence of two filings disagreeing about one period, so it
    does not exist until the later of them does. The history is fetched once
    as of today and every row it returns carries every change the series ever
    showed, which is the one place a replay reaches past its own date: the
    scanner then rejected a 2020 signal for a split filed in 2024. Only
    rejections come of it, so it does not inflate a result -- it picks the
    sample with hindsight, which is the same defect one step removed.
    """
    kept = []
    for change in changes or []:
        between = change.get("between") or []
        when = max(between) if between else change.get("period")
        if when is None or str(when)[:10] <= known_at[:10]:
            kept.append(change)
    return kept


def build_signals(tickers: Sequence[str],
                  as_of: Optional[str] = None) -> Dict[str, Any]:
    """One EDGAR pass per name for every quarter it ever filed."""
    table: Dict[str, List[Dict[str, Any]]] = {}
    for ticker in tickers:
        history = sue.sue_ts_history(ticker, as_of=as_of)
        rows = history.get("signals") or history.get("quarters") or []
        table[ticker] = [
            {**r, "basis_changes": _changes_known_by(r.get("basis_changes"),
                                                     r["known_at"])}
            for r in rows if r.get("sue") is not None and r.get("known_at")]
    loaded = load_signals(table)
    return {"tickers": len(table), "signals": loaded}


def _signal_for(ticker: str, as_of: str) -> Dict[str, Any]:
    """The most recently filed signal for this name on this date.

    The most recent, never the largest. A lookup that scanned for the biggest
    surprise available would be picking its trades with hindsight.
    """
    rows = [r for r in _SIGNALS.get(ticker, []) if r["known_at"][:10] <= as_of]
    if not rows:
        return {"ticker": ticker, "success": False, "sue": None,
                "error": f"no filed signal for {ticker} on or before {as_of}"}
    return {**rows[-1], "ticker": ticker, "success": True, "error": None}


def _refresh_universe_for(tickers: Sequence[str], as_of: str) -> int:
    """Screen just these names, on what they had done by `as_of`.

    run() used daily_job.refresh_universe, which fetches the SEC registrant
    list and screens all ten thousand of them -- on every decision date, with a
    network call each time. A replay's universe is the set it built a store
    for.
    """
    entries = []
    for index, ticker in enumerate(tickers):
        entries.append({"ticker": ticker, "cik": str(index),
                        **daily_job._screen(ticker, as_of)})
    return pit_store.record_universe(as_of, entries,
                                     recorded_at=_stamp(as_of))


def run(dates: Sequence[str], horizon_days: int = 20,
        tickers: Optional[Sequence[str]] = None,
        signal_for: Any = _MISSING,
        comparisons: int = 1,
        borrow_rate: Optional[float] = None) -> Dict[str, Any]:
    """Scan on each date, then score what those decisions did.

    Uses the scanner unmodified. That is the point of building a store rather
    than a special mode: if the replay needed its own ranking logic it would be
    measuring something other than the thing that will run tomorrow.
    """
    # The precomputed time-series table by default; None lets the scanner use
    # whichever variant it is set to, which is how the cross-sectional one --
    # dated by the release rather than the filing -- gets replayed at all.
    lookup = _signal_for if signal_for is _MISSING else signal_for

    orders: List[Dict[str, Any]] = []
    # One print is one trade. Replayed decisions never enter the filed book, so
    # the scanner cannot read this from the store the way it does live -- and
    # without it a single earnings event becomes one order per decision date
    # for as long as the signal stays fresh, which is a sample of overlapping
    # trades masquerading as independent ones.
    acted: set = set()
    for as_of in dates:
        if tickers:
            _refresh_universe_for(tickers, as_of)
        else:
            daily_job.refresh_universe(as_of=as_of)
        result = scanner.scan(as_of=as_of, already_acted=acted,
                              signal_for=lookup, borrow_rate=borrow_rate)
        for candidate in result["candidates"]:
            orders.append({**candidate, "as_of_date": as_of})
            period = candidate.get("fiscal_period")
            if period:
                acted.add((candidate["ticker"], period))

    # How many variants have been tried against these names. A t-statistic
    # from the best of several is not a t-statistic from one, and the caller
    # is the only thing that knows the count.
    scored = _score(orders, horizon_days, comparisons=comparisons)
    return {"dates": len(dates), "orders": len(orders), **scored,
            "caveats": list(CAVEATS)}


def _score(orders: List[Dict[str, Any]], horizon_days: int,
           comparisons: int = 1) -> Dict[str, Any]:
    """`scoring.fill` applied to replay orders rather than filed ones.

    One implementation, deliberately: the two paths differ only in where the
    orders come from, and copying the arithmetic between them is how a fix to
    one stops applying to the other. What is not shared is the source --
    paper_order holds decisions that were made, and these are decisions that
    were imagined, so a scoring run must never see both.
    """
    from research import scoring

    scored: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    def skip(order, reason):
        skipped.append({"ticker": order.get("ticker"),
                        "as_of_date": order.get("as_of_date"),
                        "reason": reason})

    for order in orders:
        entry_session = order.get("intended_session")
        if not entry_session:
            skip(order, "no intended session on the order")
            continue
        # Standing far enough past the horizon to see the whole hold; the
        # replay store's own stamps still stop it seeing anything else.
        as_of = (date.fromisoformat(entry_session)
                 + timedelta(days=horizon_days * 3)).isoformat()
        bars = pit_store.adjusted_bars(order["ticker"], as_of)
        forward = [b for b in bars if b["trade_date"] >= entry_session]
        if forward and forward[0]["trade_date"] != entry_session \
                and scoring._exchange_shut(entry_session, as_of):
            # A holiday. The order rests and fills at the next open, one
            # session only -- five of these were discarded in a live replay,
            # all on Presidents Day, Memorial Day and Thanksgiving. The next
            # open comes off the exchange calendar, never off this name's own
            # prints: rolling to whenever it next appears is how a study buys
            # a week after the news it was reacting to.
            entry_session = (scoring._next_open_session(entry_session, as_of)
                             or entry_session)
        if not forward or forward[0]["trade_date"] != entry_session:
            skip(order, f"{order['ticker']} did not trade on {entry_session}, "
                        f"so the order never filled")
            continue
        # Sessions of the exchange, not rows in the store; see
        # `scoring._horizon_exit`. One implementation for both paths, because
        # the two differ only in where the orders come from.
        exit_bar, _, why = scoring._horizon_exit(forward, entry_session,
                                                 horizon_days, as_of)
        if exit_bar is None:
            skip(order, why)
            continue

        row = scoring.fill(order, forward[0], exit_bar)
        if row is None:
            skip(order, "a price on the path is missing")
            continue
        scored.append({**row, "as_of_date": order["as_of_date"],
                       "timing": scoring._timing_of(order, as_of)})

    return {"scored": scored, "skipped": skipped,
            "by_timing": scoring.split_by_timing(scored, comparisons),
            "by_variant": scoring.split_by_variant(scored, comparisons),
            **scoring._summarise(scored, comparisons=comparisons)}


def summarise(scored: List[Dict[str, Any]],
              caveats_only: bool = False,
              comparisons: int = 1) -> Dict[str, Any]:
    """The numbers, never without the caveats attached to them.

    `comparisons` for the same reason `run` takes one: this scored at a bar of
    2.00 whatever had been tried, and it is the shortest route to a number.
    """
    from research import scoring

    if caveats_only:
        return {"caveats": list(CAVEATS)}
    return {**scoring._summarise(scored, comparisons=comparisons),
            "caveats": list(CAVEATS)}


# ------------------------------------------------------------- entry point

def main(argv: Optional[List[str]] = None) -> int:
    """`python -m research.replay`. Builds nothing by itself; see --help."""
    import argparse
    import json

    # Nothing else does, and the ordering that hides it is not enforced
    # anywhere: the recorder normally runs first and creates the store, so the
    # first command against a fresh volume dies on "no such table" instead.
    # Cheap and idempotent, so it runs every time rather than once.
    pit_store.init_schema()

    parser = argparse.ArgumentParser(
        prog="replay",
        description="Replay the scanner over history to measure the drift "
                    "coefficient. Reports survivorship-biased results.")
    parser.add_argument("--dates", nargs="+", required=True,
                        help="decision dates to scan on")
    parser.add_argument("--horizon-days", type=int, default=20)
    parser.add_argument("--tickers", nargs="*", default=None,
                        help="limit the universe to these names")
    parser.add_argument("--comparisons", type=int, default=1,
                        help="how many variants have been tried against these "
                             "names; the significance bar moves out with it")
    parser.add_argument("--borrow-rate", dest="borrow_rate", type=float,
                        default=None,
                        help="annualised borrow to charge every short, e.g. "
                             "0.03 for 3%%. Without it the replay is long-only "
                             "and says so, because a short nobody can price is "
                             "refused rather than charged nothing")
    args = parser.parse_args(argv)

    result = run(args.dates, horizon_days=args.horizon_days,
                 tickers=args.tickers, comparisons=args.comparisons,
                 borrow_rate=args.borrow_rate)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via main()
    raise SystemExit(main())
