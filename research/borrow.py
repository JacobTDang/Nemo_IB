"""What it costs to be short, which the cost model priced at zero.

`spread.round_trip_cost` prices a crossing: half an effective spread each way
plus impact for the two of them. Being short is not a crossing. It is a
position held open, and the stock loan bills for every calendar day of it --
including the weekends, which is forty percent of a twenty-session hold.

The size of the omission, against the 40bp the cross-sectional variant claims
at a full tail over twenty sessions:

    0.3% general collateral      2.4bp     6% of the edge
    3%   easy to borrow         23.3bp    58%
    10%  tight                  77.8bp   194%
    30%  hard to borrow        233.3bp   583%

And it is not hypothetical for this universe. The screen admits $5 stocks
trading $500k a day, and a third of the eligible names trade under $5M -- which
is deliberate, because small caps are where post-earnings drift lives. So the
term nobody was charging is largest exactly where the edge is claimed to be
largest.

**There is no default rate.** A borrow rate is not free data: it is not in
yfinance, not in Finnhub's free tier, and the only honest sources are a broker
file or a paid feed. So this module prices a short from a rate somebody
actually quoted -- recorded in the store on a date, like everything else here
-- or from one the caller declares out loud, and refuses when it has neither.
A refusal keeps an unpriceable name out of the book. A zero would put it at the
top of it, because a book ranked on edge net of cost hands first place to
whichever name was charged least.

The long leg's financing is not modelled and is not zero either; it is small
next to borrow at these holding periods, and pretending otherwise here would be
the same mistake in the other direction. It is named in the output rather than
implied by silence.
"""
from __future__ import annotations

import statistics
from datetime import date
from typing import Any, Dict, Optional

from research import pit_store, scoring, spread

# Stock loan is quoted annualised and billed on a 360-day year, on calendar
# days rather than sessions -- a position held over a long weekend is a
# position held over a long weekend.
DAY_COUNT = 360.0

# How far back to look when measuring what a horizon costs in days. A single
# span is 26 days or 30 depending on which holidays it happens to straddle;
# a year of them has a stable middle.
CALENDAR_LOOKBACK = 252

SIDES = ("long", "short")


def horizon_calendar_days(as_of: str, horizon_days: int) -> Dict[str, Any]:
    """How many days a `horizon_days`-session hold covers, from the calendar.

    Measured, not assumed. Seven-fifths of the sessions is the obvious guess
    and it undercharges by every market holiday in the window -- always in the
    direction that flatters the short, which is the direction this module
    exists to stop flattering.

    The reference instrument's bars are the only session calendar the store
    holds; `scoring` counts the holding period off the same one, so the days
    charged here and the sessions scored there describe one hold rather than
    two. With no calendar there is no answer, and no answer is a refusal.
    """
    sessions = [b["trade_date"]
                for b in pit_store.bars_as_of(spread.REFERENCE_TICKER, as_of)]
    if len(sessions) <= horizon_days:
        return {"calendar_days": None, "sessions_available": len(sessions),
                "reason": (
                    f"{spread.REFERENCE_TICKER} has {len(sessions)} sessions "
                    f"on or before {as_of}, which is not enough to measure "
                    f"how many days a {horizon_days}-session hold covers. "
                    f"Without the calendar there is no borrow charge, and a "
                    f"short charged nothing outranks every name that paid")}

    window = sessions[-(CALENDAR_LOOKBACK + horizon_days):]
    spans = [
        (_d(window[i + horizon_days]) - _d(window[i])).days
        for i in range(len(window) - horizon_days)
    ]
    return {"calendar_days": int(statistics.median(spans)),
            "sessions_available": len(sessions), "reason": None}


def annual_rate(ticker: str, as_of: str,
                declared_rate: Optional[float] = None) -> Dict[str, Any]:
    """The rate to charge, and who said so.

    A rate recorded for this name on this date wins over a declared one: the
    declaration is a blanket assumption about every name, and a quote is about
    this one. Both are reported by source, because a study that mixes them and
    says only "borrow was charged" cannot be read afterwards.
    """
    recorded = pit_store.borrow_rate_as_of(ticker, as_of)
    if recorded is not None:
        return {"annual_rate": float(recorded["annual_rate"]),
                "rate_source": "recorded",
                "rate_quoted_by": recorded.get("source"),
                "rate_as_of": recorded.get("as_of_date"), "reason": None}

    if declared_rate is not None:
        if declared_rate < 0:
            raise ValueError(
                f"declared borrow rate must not be negative; got "
                f"{declared_rate!r}")
        return {"annual_rate": float(declared_rate), "rate_source": "declared",
                "rate_quoted_by": None, "rate_as_of": None, "reason": None}

    return {"annual_rate": None, "rate_source": None, "rate_quoted_by": None,
            "rate_as_of": None,
            "reason": (
                f"no borrow rate is on record for {ticker} on or before "
                f"{as_of} and none was declared, so the cost of holding it "
                f"short is unknown. It is not zero: at 3% a twenty-session "
                f"hold costs 23bp, which is most of the drift being chased")}


def carry_cost(ticker: str, as_of: str, side: str,
               horizon_days: int = scoring.DEFAULT_HORIZON_DAYS,
               declared_rate: Optional[float] = None) -> Dict[str, Any]:
    """What holding `ticker` for the horizon costs to finance, as a fraction.

    Subtract it from the edge alongside the round trip. `cost` is 0.0 for a
    long -- the long leg's financing is not modelled, and `rate_source` says
    `not_short` rather than leaving a reader to guess whether zero meant free
    or meant unpriced.

    `cost` is None with a `reason` whenever a short cannot be priced, and every
    caller must treat that as a refusal to trade the name rather than as a cost
    of nothing.
    """
    if side not in SIDES:
        raise ValueError(
            f"side must be one of {SIDES}; got {side!r}. Guessing it would "
            f"decide whether a position pays borrow, which is the question")
    if horizon_days <= 0:
        raise ValueError(
            f"horizon_days must be positive; got {horizon_days!r}")

    out: Dict[str, Any] = {
        "ticker": ticker, "as_of": as_of, "side": side,
        "horizon_days": horizon_days, "calendar_days": None,
        "annual_rate": None, "rate_source": None, "rate_quoted_by": None,
        "day_count": DAY_COUNT, "cost": None, "reason": None,
    }

    if side == "long":
        return {**out, "cost": 0.0, "rate_source": "not_short",
                "note": ("the long leg's financing is not modelled; it is "
                         "small next to borrow over this horizon, not zero")}

    rate = annual_rate(ticker, as_of, declared_rate)
    out.update({k: rate[k] for k in
                ("annual_rate", "rate_source", "rate_quoted_by")})
    if rate["annual_rate"] is None:
        return {**out, "reason": rate["reason"]}

    span = horizon_calendar_days(as_of, horizon_days)
    out["calendar_days"] = span["calendar_days"]
    if span["calendar_days"] is None:
        return {**out, "reason": span["reason"]}

    out["cost"] = rate["annual_rate"] * span["calendar_days"] / DAY_COUNT
    return out


def _d(day: str):
    return date.fromisoformat(day)


# ------------------------------------------------------------- entry point

def _rows_from_csv(path: str, units: str) -> list:
    """Every row, or an exception. Never some of them.

    A file half-recorded is the worst outcome available: the names that landed
    get priced, the ones that did not get refused for want of a rate, and
    nothing in the book says which was which.
    """
    import csv

    scale = 1.0 if units == "fraction" else 0.01
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        if "ticker" not in columns or "annual_rate" not in columns:
            raise ValueError(
                f"{path} needs 'ticker' and 'annual_rate' columns; found "
                f"{columns}")
        rows = []
        for line, record in enumerate(reader, start=2):
            raw = (record.get("annual_rate") or "").strip()
            try:
                rate = float(raw) * scale
            except ValueError:
                raise ValueError(
                    f"{path} line {line}: annual_rate is {raw!r}, which is not "
                    f"a number") from None
            rows.append({"ticker": (record.get("ticker") or "").strip().upper(),
                         "annual_rate": rate,
                         "source": (record.get("source") or "").strip() or None})
    if not rows:
        raise ValueError(f"{path} has a header and no rates")
    return rows


def main(argv: Optional[list] = None) -> int:
    """`python -m research.borrow`. Loads a day's borrow rates into the store.

    A broker's daily short list, a vendor extract, a hand-typed file for the
    dozen names a book actually holds -- whatever the source, it lands here
    dated to the day it describes, and the scanner reads it point-in-time like
    everything else.
    """
    import argparse

    pit_store.init_schema()

    parser = argparse.ArgumentParser(
        prog="borrow",
        description="Record what it costs to be short, per name, for a date.")
    parser.add_argument("--as-of", dest="as_of", required=True,
                        help="the date these rates were quoted for")
    parser.add_argument("--from-csv", dest="from_csv", required=True,
                        help="a CSV with 'ticker' and 'annual_rate' columns, "
                             "and optionally 'source'")
    # No default, deliberately. 3 and 0.03 are both plausible readings of the
    # same file and they differ by a hundred, which is the difference between
    # a general-collateral name and one nobody will lend.
    parser.add_argument("--units", required=True,
                        choices=("fraction", "percent"),
                        help="whether annual_rate is 0.03 or 3.0 for 3%%")
    parser.add_argument("--source", default="recorded",
                        help="who quoted these, for rows that do not name it")
    # Off by default, and it has to stay off by default: a rate typed in today
    # is not something last month's decision could have used, and backdating
    # every load would reintroduce the lookahead `recorded_at` exists to stop.
    parser.add_argument("--backfill", action="store_true",
                        help="stamp the rows at the end of --as-of rather than "
                             "now, so they are visible to the date they "
                             "describe. For a broker file collected after the "
                             "fact; the rows are marked backfilled so a study "
                             "can tell them from ones captured on the day")
    args = parser.parse_args(argv)

    today = date.today().isoformat()
    if args.backfill and args.as_of > today:
        # Stamping a row at the end of a day that has not happened makes it
        # visible to every date after it and to none before. That is not a
        # backfill, it is a forecast wearing one's clothes.
        print(f"--backfill needs a date that has happened; {args.as_of} is "
              f"after {today}")
        return 1

    try:
        rows = _rows_from_csv(args.from_csv, args.units)
        written = pit_store.record_borrow_rates(
            args.as_of, rows, source=args.source,
            recorded_at=f"{args.as_of}T21:00:00Z" if args.backfill else None,
            backfilled=args.backfill)
    except (OSError, ValueError) as exc:
        print(f"{type(exc).__name__}: {exc}")
        return 1

    rates = sorted(r["annual_rate"] for r in rows)
    # Printed so a hundred-fold units error is visible in the output rather
    # than three weeks later in a book of shorts nobody could afford.
    print(f"{written} of {len(rows)} rates recorded for {args.as_of}; "
          f"min {rates[0]:g}, median {statistics.median(rates):g}, "
          f"max {rates[-1]:g}")
    if not args.backfill and args.as_of < today:
        # The confusing pairing this exists to stop: "recorded" printed over a
        # write no reader will ever return, followed three minutes later by
        # every short refusing for want of a rate.
        print(f"warning: these are stamped now, so a scan dated {args.as_of} "
              f"will not see them -- `date(recorded_at) <= as_of` hides a rate "
              f"written after the decision. Pass --backfill to stamp them at "
              f"{args.as_of} instead")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via main()
    raise SystemExit(main())
