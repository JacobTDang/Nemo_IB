"""The consensus history that cannot be recorded backwards.

The analyst surprise needs eight quarters of what the street expected before
each print. Finnhub serves four and no more -- verified at limit=12 and
limit=30 alike -- so the recorder fills the rest one day at a time and `sue_af`
refuses for two years while it does.

The four it does serve carry both legs: a vendor estimate and a vendor actual,
quoted on one basis. Whether they are usable turns on one question. If that
estimate is revised after the print it has converged toward the actual, which
shrinks every surprise and every sigma -- inflating the signal, the direction
that flatters a strategy and the reason to be suspicious of it.

It is frozen, and this project could check rather than assume. Ten estimates
were recorded off the forward calendar on 2026-08-26, before those companies
reported. After they reported, the surprises endpoint returns them identical to
four decimal places: NVDA 2.1283, CRWD 0.2984, OKTA 0.9841, DY 4.8221, MOV
0.3560, PLAB 0.4114, BJ 1.1951, DG 2.0559, HPQ 0.6639, CRM 3.3057. Ten of ten,
against our own point-in-time recording rather than against the vendor's word.

So seeding is sound and still has to be visible. A seeded row is a
reconstruction stamped at a date nobody was watching, and it carries
`source='seeded'` so that anything reading it can tell, exclude it, or report
how much of an answer rests on it. That is the same bargain `replay` makes with
prices: state the assumption in the data rather than in a comment.

The dates come from the filings, not from the vendor. Finnhub's `year` and
`quarter` are the filer's own, but its `period` is a calendar bucket: it
reports NVDA's fiscal 2027 Q2 as 2027-06-30, a quarter that ended 2026-07-26
and was announced a month after that. Stamping at the bucket puts the estimate
a year in the future where nothing can read it, which is what happened the
first time this ran -- NVDA, COST, WMT, CRM and ORCL all came back "no
consensus was recorded before that print" from a store that had just seeded
them. So each quarter is matched to the XBRL series on fiscal identity and
stamped at the filer's own period end and filing date.

One imprecision remains and is worth naming: the estimate is stamped at period
end, definitively before the print, while the figure itself is the final
pre-print consensus rather than whatever stood on that particular day. For a
surprise that is the right number; only its date is approximate.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from research import pit_store

SOURCE = "seeded"


def _finnhub_server():
    from tools.news_agregator.finnhub_server import FinnhubServer

    return FinnhubServer()


def _fetch_surprises(ticker: str) -> List[Dict[str, Any]]:
    """One name's four quarters, with the session closed before the loop ends.

    `asyncio.run` opens a loop, runs the call and closes the loop; a session
    created inside it survives, holding a connector bound to a loop that no
    longer exists. Nothing notices for a while. Then the collector reaches one,
    its destructor schedules cleanup on that dead loop, and the run dies with
    "Event loop is closed" -- which is what happened partway through seeding
    six hundred names, after twelve had worked fine in a test.

    Only the loop that made the session can close it, so it is closed here.
    """
    import asyncio
    import json

    async def pull():
        server = _finnhub_server()
        try:
            return await server.get_earnings_surprises(ticker)
        finally:
            client = getattr(server, "client", None)
            close = getattr(client, "close", None)
            if close is not None:
                await close()

    raw = asyncio.run(pull())
    payload = json.loads(raw[0].text)
    if payload.get("data", {}).get("error"):
        raise RuntimeError(f"{ticker}: {payload['data']['error']}")
    return (payload.get("data") or {}).get("quarters") or []


def _stamp(day: str) -> str:
    return f"{day}T21:00:00Z"


def _announcements(ticker: str,
                   as_of: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """When the market learned each quarter, from the Item 2.02 filing."""
    from research import announcements

    return announcements.for_quarters(ticker, as_of=as_of)


def _filing_dates(ticker: str,
                  as_of: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """The filer's own period end and filing date, per fiscal period."""
    from research import sue

    series = sue.eps_series(ticker, as_of=as_of)
    if not series.get("success"):
        return {}
    return {q["fiscal_period"]: {"period_end": q.get("period_end"),
                                 "known_at": (q.get("known_at") or "")[:10]}
            for q in series["quarters"]}


def seed(tickers: Sequence[str],
         as_of: Optional[str] = None) -> Dict[str, Any]:
    """Reconstruct up to four quarters of both legs for each name.

    The estimate is stamped at the filer's own period end, which is before the
    print by construction. The actual is stamped at the filing date, because it
    had not happened at period end and a reconstruction may not reconstruct the
    future.

    Never overwrites: a quarter already in the record was watched, and a real
    observation outranks a reconstruction of it.
    """
    written = 0
    incomplete = 0
    undated = 0
    announcement_dated = 0
    filing_dated = 0
    # Which quarters had no matching filing, by name. A count alone hides the
    # difference between one derived quarter that did not resolve and a filer
    # whose vendor labels are a year off its own -- the second can never be
    # seeded at all, and an operator has no way to find that out from a total.
    unmatched: Dict[str, List[str]] = {}
    duplicates: Dict[str, List[str]] = {}
    failed: List[str] = []
    seen = 0

    for ticker in tickers:
        try:
            rows = _fetch_surprises(ticker)
            dates = _filing_dates(ticker, as_of=as_of)
            announced = _announcements(ticker, as_of=as_of)
        except Exception as exc:  # noqa: BLE001 - counted and reported
            failed.append(f"{ticker}: {type(exc).__name__}: {exc}")
            continue

        # Finnhub returns TGT's 2027Q2 twice, with different calendar buckets
        # and the same actual. Which one is the quarter is not answerable from
        # the payload, and taking whichever arrives first attaches an estimate
        # to a quarter on a coin flip, so neither is seeded.
        counts: Dict[str, int] = {}
        for row in rows:
            if row.get("year") and row.get("quarter"):
                label = f"{row['year']}Q{row['quarter']}"
                counts[label] = counts.get(label, 0) + 1
        repeated = sorted(k for k, n in counts.items() if n > 1)
        if repeated:
            duplicates[ticker] = repeated

        for row in rows:
            seen += 1
            year, quarter = row.get("year"), row.get("quarter")
            estimate, actual = row.get("estimate_eps"), row.get("actual_eps")
            if not (year and quarter):
                incomplete += 1
                continue
            if estimate is None or actual is None:
                # Half a pair cannot make a surprise, and writing one leg would
                # look like coverage this does not have.
                incomplete += 1
                continue

            fiscal = f"{year}Q{quarter}"
            if fiscal in duplicates.get(ticker, ()):
                continue
            filed = dates.get(fiscal) or {}
            period_end, known_at = filed.get("period_end"), filed.get("known_at")
            if not period_end or not known_at:
                # No filer dates, nothing to stamp it at. Guessing is what put
                # an estimate a year into the future the first time round.
                undated += 1
                unmatched.setdefault(ticker, []).append(fiscal)
                continue

            # Skip on a recorded ACTUAL, not on the mere presence of a row.
            # The daily job writes an estimate for everything on the calendar,
            # so a quarter it was down for has an estimate and no actual --
            # and skipping that leaves it half-filled forever, one quarter
            # fewer for a window that needs six. Writes are INSERT OR IGNORE
            # keyed on the date, so an observed estimate is never restated by
            # the reconstruction of its actual.
            if pit_store.actual_as_of(ticker, fiscal, "9999-12-31") is not None:
                continue

            written += pit_store.record_consensus(
                period_end, ticker, fiscal, eps_estimate=estimate,
                recorded_at=_stamp(period_end), source=SOURCE) or 0

            # The actual is known when the market learned it, which is the
            # Item 2.02 release rather than the 10-Q -- a median of eight days
            # earlier on the names measured, twenty-three for JPM. Everything
            # downstream takes its timing from this row, and eight days into a
            # drift that is largest in its first days is most of the drift.
            # With no release on record the filing date stands, because late
            # is the safe direction.
            release = announced.get(fiscal) or {}
            learned = release.get("announced_date") or known_at
            if release.get("announced_date"):
                announcement_dated += 1
            else:
                filing_dated += 1
            written += pit_store.record_consensus(
                learned, ticker, fiscal, eps_estimate=estimate,
                eps_actual=actual, recorded_at=_stamp(learned),
                source=SOURCE) or 0

    return {"tickers": len(tickers), "quarters_seen": seen,
            "written": written, "incomplete": incomplete, "undated": undated,
            "unmatched": unmatched, "duplicates": duplicates,
            "announcement_dated": announcement_dated,
            "filing_dated": filing_dated, "failed": failed}


# ------------------------------------------------------------- entry point

def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        prog="seed_consensus",
        description="Reconstruct up to four quarters of consensus history per "
                    "name from the vendor's own surprise record.")
    parser.add_argument("--tickers", nargs="*", default=None,
                        help="names to seed (default: today's eligible universe)")
    args = parser.parse_args(argv)

    names = args.tickers
    if not names:
        from research import daily_job
        names = daily_job.eligible_tickers()

    result = seed(names)
    print(json.dumps(result, indent=2, default=str))
    return 1 if result["failed"] and not result["written"] else 0


if __name__ == "__main__":  # pragma: no cover - exercised via main()
    raise SystemExit(main())
