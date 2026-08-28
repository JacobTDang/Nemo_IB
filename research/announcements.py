"""When the market learned the number, from the filing that told it.

The store dates a quarter by its 10-Q, because that is where the XBRL lives.
The market learns the figure earlier, in the Item 2.02 8-K -- "Results of
Operations and Financial Condition", which is the earnings release itself.

Across 60 filings from 20 large caps the gap is a median of 2 days and a mean
of 6.1, which sounds ignorable until the tail: JPM 31 days, TGT 28, LOW 26, WMT
22, HD 22. Banks and retailers, systematically. Post-earnings drift is largest
in the first days, so a study entered off the 10-Q misses most of it precisely
where the lag is worst -- and reports a null result that is really a result
about its own timing.

Two things come out of the 8-K and neither is a guess:

  **The date.** The Item 2.02 filing after the quarter closed and no later than
  the 10-Q that reported it. A quarter with no such filing is left out rather
  than dated by its 10-Q, because a fallback here reintroduces the error
  silently and this module exists to remove it.

  **The hour.** `acceptance_datetime` says when SEC took it, read in New York
  rather than UTC because the boundary is 16:00 local and the offset moves with
  daylight saving. Before the open, during the session, or after the close --
  which decides which bar is the reaction. AMAT's 13 August print landed after
  the close, so its gap is the 14th, and dating it to the announcement gives
  -2.48% instead of -6.57%. A filing with no acceptance time is `unknown`,
  never a default, because a default answers that question with a coin flip.
"""
from __future__ import annotations

from datetime import date, time, timedelta
from typing import Any, Dict, List, Optional, Sequence
from zoneinfo import ZoneInfo

from research import pit_store

EXCHANGE_TZ = ZoneInfo("America/New_York")
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)

# US equities close at 13:00 on about three sessions a year, and this used to
# be a fixed 16:00 -- so a release at 14:30 on one of them read as `dmh`, the
# market open, when it had been shut for ninety minutes. `timing` decides
# which session an order may enter and `scoring` splits its results on it, so
# the release landed in the wrong bucket on both sides.
HALF_DAY_CLOSE = time(13, 0)


def _half_day_candidates(year: int) -> "set[date]":
    """The three dates NYSE closes early on, before the weekday test.

    Computed rather than tabulated so this does not expire: a table would go
    stale silently, which is the failure being fixed rather than a new one.
    """
    november = date(year, 11, 1)
    # 4th Thursday: the first Thursday, plus three weeks. Thanksgiving itself
    # is a holiday; the early close is the Friday after it.
    first_thursday = november + timedelta(days=(3 - november.weekday()) % 7)
    return {
        first_thursday + timedelta(weeks=3, days=1),
        date(year, 12, 24),
        date(year, 7, 3),
    }


def _closes_early(day: date) -> bool:
    """Whether the exchange shut at 13:00 on this date.

    The weekday test is the whole rule. In years where one of these falls
    beside an observed holiday the exchange is shut all day rather than half
    of it -- 24 December on a Friday when Christmas is the Saturday, 3 July
    when Independence Day is the Saturday -- and this returns True for those
    too. That is deliberate and cannot make an answer worse: on a day the
    market never opened, `dmh` is definitely wrong and `amc` is the closer of
    the two. Being exactly right about a fully-closed day needs the session
    calendar, which lives in the store rather than here.
    """
    return day.weekday() < 5 and day in _half_day_candidates(day.year)

# "Results of Operations and Financial Condition". The only 8-K item that is an
# earnings release; 5.02 is a director leaving and 8.01 is anything else.
EARNINGS_ITEM = "2.02"

# How far back to look. Four years of quarters is well inside it, and the cap
# keeps one company from pulling its whole filing history.
_MAX_FILINGS = 80


def _today() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).date().isoformat()


def _is_earnings(items: Any) -> bool:
    """Whether this filing carries item 2.02, matched whole.

    A substring test takes "12.02" and "2.021" as well. Neither exists today,
    which is the moment to pin it rather than after one does. edgartools joins
    the codes with commas; a list is accepted too, because a shape change
    upstream would otherwise match nothing and read as a company that files no
    earnings releases at all.
    """
    if items is None:
        return False
    codes = items if isinstance(items, (list, tuple, set)) else str(items).split(",")
    return any(str(c).strip() == EARNINGS_ITEM for c in codes)


def _fetch_8k(ticker: str, limit: int = _MAX_FILINGS) -> List[Any]:
    import os

    from edgar import Company, set_identity

    email = os.environ.get("SEC_EMAIL", "").strip()
    if not email:
        raise ValueError(
            "SEC_EMAIL is not set. SEC fair access requires a real contact "
            "address in the User-Agent header.")
    set_identity(email)
    return list(Company(ticker).get_filings(form="8-K").head(limit))


def _quarters(ticker: str, as_of: Optional[str] = None) -> Dict[str, Dict]:
    from research import sue

    series = sue.eps_series(ticker, as_of=as_of)
    if not series.get("success"):
        return {}
    return {q["fiscal_period"]: {"period_end": q.get("period_end"),
                                 "known_at": (q.get("known_at") or "")[:10]}
            for q in series["quarters"]}


def classify(accepted_at) -> str:
    """Before the open, in the session, or after the close -- in New York.

    Read locally rather than in UTC because the boundary is 16:00 exchange time
    and the offset from UTC moves with daylight saving: 21:05 UTC is 16:05 in
    January and 17:05 in July, and only one of those is after the close.

    A timestamp with no zone is refused rather than assumed. `astimezone` on a
    naive datetime reads it in the machine's own zone, so the same filing would
    classify one way on a laptop in New York and another on a server in UTC --
    a result that depends on where it ran is worse than no result.
    """
    if accepted_at is None:
        return "unknown"
    if accepted_at.tzinfo is None or accepted_at.utcoffset() is None:
        return "unknown"
    local = accepted_at.astimezone(EXCHANGE_TZ)
    close = HALF_DAY_CLOSE if _closes_early(local.date()) else MARKET_CLOSE
    if local.time() < MARKET_OPEN:
        return "bmo"
    if local.time() >= close:
        return "amc"
    return "dmh"


def earnings_releases(ticker: str,
                      as_of: Optional[str] = None) -> List[Dict[str, Any]]:
    """Every Item 2.02 filing this company has made, newest first."""
    out = []
    for filing in _fetch_8k(ticker):
        if not _is_earnings(getattr(filing, "items", None)):
            continue
        day = str(filing.filing_date)
        if as_of and day > as_of:
            continue
        out.append({
            "ticker": ticker.upper(),
            "announced_date": day,
            "timing": classify(getattr(filing, "acceptance_datetime", None)),
            "accession": getattr(filing, "accession_no", None),
        })
    return sorted(out, key=lambda r: r["announced_date"], reverse=True)


def for_quarters(ticker: str, as_of: Optional[str] = None,
                 quarters: Optional[Dict[str, Dict]] = None
                 ) -> Dict[str, Dict[str, Any]]:
    """The release that reported each quarter, keyed on fiscal period.

    A release belongs to a quarter if it fell after that quarter closed and no
    later than the filing that reported it. Outside that window it is a
    different quarter's release, and no release at all means the quarter is
    absent rather than dated by its 10-Q.

    `quarters` lets a caller that already has the series pass it in. Seeding
    fetches it for the filing dates and would otherwise fetch it again here,
    which over 595 names is 1,190 requests for the same data.
    """
    releases = earnings_releases(ticker, as_of=as_of)
    if not releases:
        return {}

    known = quarters if quarters is not None else _quarters(ticker, as_of=as_of)
    out: Dict[str, Dict[str, Any]] = {}
    for period, dates in (known or {}).items():
        period_end, known_at = dates.get("period_end"), dates.get("known_at")
        if not period_end or not known_at:
            continue
        inside = [r for r in releases
                  if period_end < r["announced_date"] <= known_at]
        if not inside:
            continue
        # The earliest one inside the window: a company that files a correction
        # later has still announced at the first.
        out[period] = {**min(inside, key=lambda r: r["announced_date"]),
                       "fiscal_period": period}
    return out


def backfill(tickers: Sequence[str],
             as_of: Optional[str] = None) -> Dict[str, Any]:
    """Record every release found, dated and timed from the filing itself.

    Stamped at its own date, so a reader standing before the announcement does
    not see it -- the record is of when the market learned, and back-filling it
    with today's stamp would make it invisible to exactly the questions it
    exists to answer.
    """
    as_of = as_of or _today()
    written = 0
    covered = 0
    failed: List[str] = []

    for ticker in tickers:
        try:
            found = for_quarters(ticker, as_of=as_of)
        except Exception as exc:  # noqa: BLE001 - counted and reported
            failed.append(f"{ticker}: {type(exc).__name__}: {exc}")
            continue
        if found:
            covered += 1
        for period, row in found.items():
            written += pit_store.record_announcement(
                ticker.upper(), period, row["announced_date"],
                timing=row["timing"], source=pit_store.PRIMARY_SOURCE,
                recorded_at=f"{row['announced_date']}T21:00:00Z")

    return {"tickers": len(tickers), "covered": covered, "written": written,
            "failed": failed}


# ------------------------------------------------------------- entry point

def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        prog="announcements",
        description="Record earnings announcement dates and times from Item "
                    "2.02 filings.")
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument("--as-of", dest="as_of", default=None)
    args = parser.parse_args(argv)

    names = args.tickers
    if not names:
        from research import daily_job
        names = daily_job.eligible_tickers(args.as_of)

    result = backfill(names, as_of=args.as_of)
    print(json.dumps(result, indent=2, default=str))
    return 1 if result["failed"] and not result["written"] else 0


if __name__ == "__main__":  # pragma: no cover - exercised via main()
    raise SystemExit(main())
