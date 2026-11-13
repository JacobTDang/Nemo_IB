"""Who a company says it buys from, sells to and competes with.

`get_supply_chain` reads the Business section of a 10-K and matches it against
a curated list of about 150 large-cap names, putting each match on a side --
supplier, customer, competitor, partner, or `unknown` when no cue phrase sat
near the mention. That is the graph a read-through needs: a foundry's guidance
reads through to the names downstream of it, and the direction is the whole
value of the fact. This module stores it, dates it, and paces the reads.

Three things it is careful about.

**It is an extraction, not a quote.** The name list is finite and the
relationship is inferred from prose with a cue phrase, so a filing that names
nobody on the list and a filing that names nobody are indistinguishable in
`related_companies`. Coverage is therefore counted where it belongs: a name's
read is recorded on its own (`supply_chain_capture`), and an empty read is
`ok` with zero links rather than a missing row. A reader can then tell "read,
found nobody" from "never read".

**The date is the filing's, not the run's.** A 10-K is public from the moment
it is filed, so a link is stamped at the end of its filing date and
`backfilled=1` when this project read it later than that. That is the same
line `announcements.backfill` draws: a fact that was public then may be loaded
now, and a study has to be able to tell the two kinds of evidence apart --
including the one this flag cannot fix, which is that a run years later matches
against today's name list and so reconstructs what *could* be found, not what
was found.

**A 10-K comes once a year.** Reading every eligible name every night would be
1,500 EDGAR documents a night for a fact that changes annually. The refresh is
keyed on the filing date: nothing newer exists until about a year after the
last one, so a name is not due before then, and past that date it is asked
again monthly until the next 10-K lands -- a company can be late, can change
its fiscal year end, or can stop filing. Failures retry monthly, never nightly,
or one broken name would own every night's slice.

The slice is bounded (`MAX_PER_RUN`) because the nightly pass has other stages
to run: at about two seconds a name, thirty names is a minute, and the whole
universe comes up over roughly two months on a cold store -- after which the
yearly cycle asks for a handful of names a night.

Foreign listings come through as the extractor resolves them (`2317.TW`,
`000660.KS`). They are kept as-is: a ticker this store cannot price is still a
true statement about the filing, and dropping it would quietly shrink the
graph to the names it happens to trade.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence

from research import pit_store

SOURCE = "edgar"

# A 10-K covers a fiscal year, so the next one is due about 365 days after the
# last. Before that date there is nothing newer to read.
REFRESH_DAYS = 365

# Past it, ask again -- but not every night. A company can file late, move its
# fiscal year end, or stop filing altogether, and a nightly retry of the same
# document would spend the whole slice on names that have not moved.
REFRESH_RETRY_DAYS = 30

# A read that failed is not a fact about the company. It is retried, monthly,
# so the rest of the universe still comes up.
FAILURE_RETRY_DAYS = 30

# Names read per pass. The extractor takes one to three seconds and fetches a
# whole 10-K; thirty is about a minute inside a nightly pass that already runs
# for four.
MAX_PER_RUN = 30

# The sides, in the order a card reads them. `unknown` is last because it is
# the extractor's honest "mentioned, no cue phrase nearby", not a relationship.
SIDES = ("supplier", "customer", "competitor", "partner", "unknown")


class ReadError(RuntimeError):
    """A read that came back but could not be turned into a dated fact.

    Phrased for the run log at the moment it is raised, because the message is
    the whole answer an operator gets: "the extractor refused", "no filing
    date", "dated after the date asked for" are three different repairs.
    """


def _today() -> str:
    """UTC, like every other date this package derives from the clock."""
    return datetime.now(timezone.utc).date().isoformat()


def _age(stamp: str, as_of: str) -> int:
    """Days from a stored timestamp to the date being read, never negative."""
    try:
        then = date.fromisoformat(str(stamp or "")[:10])
    except ValueError:
        # A capture with an unreadable stamp is treated as ancient rather than
        # as never due -- a name that stops being read is the failure mode
        # worth avoiding, and this one is recoverable by reading it again.
        return REFRESH_DAYS
    return (date.fromisoformat(as_of[:10]) - then).days


def _extract(ticker: str) -> Dict[str, Any]:
    """One name's Business section, through the extractor.

    Imported on use. `sec_utils` pulls in edgartools, pandas and the filing
    cache, and the nightly job imports this module on every run whether or not
    anything is due.
    """
    from tools.web_search_server.sec_utils import get_supply_chain

    return get_supply_chain(ticker)


def _universe(as_of: str) -> List[str]:
    """Today's eligible names, when the caller did not already have them."""
    from research import daily_job

    return daily_job.eligible_tickers(as_of)


def _links(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The edges in a payload, in the shape the store writes.

    `related_companies` is the one place the direction is attached per name;
    the top-level `suppliers`/`customers`/... lists are the same data as
    ticker strings. Reading both would double every edge, so only the first
    is read and the second is left for a caller that wants a quick list.
    """
    out: List[Dict[str, Any]] = []
    for row in payload.get("related_companies") or []:
        related = str(row.get("ticker") or "").strip()
        if not related:
            # A curated name whose ticker the extractor could not resolve is
            # not an edge; it is a name it matched and could not place. The
            # mention count is dropped with it rather than attached to a
            # neighbouring row.
            continue
        out.append({"related_ticker": related,
                    "relationship": row.get("relationship") or "unknown",
                    "mention_count": row.get("mention_count") or 0,
                    "basis": row.get("relationship_basis")})
    return out


def _filing_date(payload: Dict[str, Any]) -> Optional[str]:
    """The 10-K's own date, as a date string, or None if it did not say.

    `get_supply_chain` returns a `datetime.date`, so it is stringified here
    rather than trusted to compare as text elsewhere.
    """
    raw = payload.get("filing_date")
    if raw is None:
        return None
    text = str(raw)[:10]
    try:
        date.fromisoformat(text)
    except ValueError:
        return None
    return text


def _due(capture: Optional[Dict[str, Any]], as_of: str) -> bool:
    """Whether this name's 10-K is worth reading again on `as_of`.

    Never read: yes. Last read failed: not until the retry window, or one
    dead name takes the slice every night. Last read succeeded: not until a
    year has passed since the filing it read, then monthly until a newer
    filing appears.
    """
    if capture is None:
        return True
    if capture.get("outcome") == "failed":
        return _age(capture.get("captured_at"), as_of) >= FAILURE_RETRY_DAYS
    filed = capture.get("filing_date")
    if not filed:
        # `ok` always carries the filing it read. A row without one is from a
        # version that did not, and reading it again is the cheap repair.
        return True
    try:
        expected = date.fromisoformat(str(filed)[:10]) + timedelta(
            days=REFRESH_DAYS)
    except ValueError:
        return True
    if date.fromisoformat(as_of[:10]) < expected:
        return False
    return _age(capture.get("captured_at"), as_of) >= REFRESH_RETRY_DAYS


def due(names: Iterable[str], as_of: str,
        limit: int = MAX_PER_RUN) -> List[str]:
    """Which of `names` to read tonight, and in what order.

    Never-read names come first, then whoever was asked longest ago. Both are
    deterministic, which matters more here than it looks: the eligible
    universe is ordered by ticker, so a slice that took the first N due names
    without sorting would spend the first month on the A's.
    """
    captures = pit_store.supply_chain_captures()
    pending = []
    for name in names:
        ticker = (name or "").strip().upper()
        if not ticker:
            continue
        capture = captures.get(ticker)
        if _due(capture, as_of):
            pending.append(((capture is not None),
                            str((capture or {}).get("captured_at") or ""),
                            ticker))
    pending.sort()
    return [ticker for _, _, ticker in pending[:max(0, int(limit))]]


def record(names: Optional[Sequence[str]] = None,
           as_of: Optional[str] = None,
           limit: int = MAX_PER_RUN) -> Dict[str, Any]:
    """Read the names that are due, after filing them into the store.

    Every name's outcome is written, success or failure: the failure is what
    keeps it from being asked again tonight, and the success is what says the
    empty answer was read rather than missing.

    A payload whose newest 10-K is dated after `as_of` is a failure for this
    date, not a row. The extractor always returns the latest filing, so a
    replay of last quarter would otherwise write September's suppliers into
    March's record -- dated September, so a reader would not see it, but
    captured under a run that claimed to be reading March.
    """
    as_of = as_of or _today()
    names = list(dict.fromkeys(
        (n or "").strip().upper()
        for n in (names if names is not None else _universe(as_of))))
    names = [n for n in names if n]
    asked = due(names, as_of, limit=limit)

    pit_store.start_run("supply_chain", as_of_date=as_of)
    if not asked:
        # The ordinary night once the universe is caught up: at most a
        # handful of names are due, and often none. The run row still lands,
        # because a stage that writes nothing and says nothing is
        # indistinguishable from one that is not scheduled.
        pit_store.finish_run(rows_written=0, status="ok")
        return {"status": "ok", "as_of": as_of, "universe": len(names),
                "asked": 0, "captured": 0, "written": 0, "zero_link": 0,
                "failed": []}

    written = captured = zero_link = 0
    failures: List[str] = []
    # The capture is stamped with the date this pass reads as, not with the
    # wall clock. A pass is a date's work, and a rerun of that date has to ask
    # for exactly what the first run asked for -- the same property
    # `_nightly_split` relies on. It is scheduling state either way, and it is
    # never read as a fact about a company.
    stamp = f"{as_of}T21:00:00Z"
    for ticker in asked:
        # One name's whole read, so a payload this module cannot turn into a
        # dated fact costs that name and not the night. A malformed link is a
        # layout change, and it is recorded as the failure it is rather than
        # raised through a run that would then never close its run row.
        try:
            payload = _extract(ticker)
            if not payload.get("success"):
                raise ReadError(str(payload.get("error")
                                    or "the extractor refused"))
            filed = _filing_date(payload)
            if filed is None:
                raise ReadError("the extractor returned no filing date, so "
                                "the links cannot be dated")
            if filed > as_of:
                raise ReadError(f"the newest 10-K is dated {filed}, after "
                                f"the {as_of} this run reads as")
            links = _links(payload)
            # Read on the filing's own date means not backfilled; anything
            # later is a reconstruction, and says so on the row.
            written += pit_store.record_supply_chain(
                ticker, filed, links, recorded_at=f"{filed}T21:00:00Z",
                source=SOURCE, backfilled=filed < as_of)
        except Exception as exc:  # noqa: BLE001 - counted and reported
            reason = (str(exc) if isinstance(exc, ReadError)
                      else f"{type(exc).__name__}: {exc}")
            failures.append(f"{ticker}: {reason}")
            pit_store.record_supply_chain_capture(
                ticker, None, 0, "failed", reason=reason, captured_at=stamp)
            continue

        pit_store.record_supply_chain_capture(ticker, filed, len(links), "ok",
                                              captured_at=stamp)
        captured += 1
        if not links:
            zero_link += 1

    # `failed` only when nothing was read at all. A night of thirty names with
    # two EDGAR timeouts is a partial night, and saying so is more useful than
    # either a clean success or a hard failure.
    status = ("failed" if failures and not captured
              else "partial" if failures else "ok")
    pit_store.finish_run(
        rows_written=written, status=status,
        error=("; ".join(failures[:5]) if failures else None))
    return {"status": status, "as_of": as_of, "universe": len(names),
            "asked": len(asked), "captured": captured, "written": written,
            "zero_link": zero_link, "failed": failures}


# -------------------------------------------------------------- entry point

def main(argv: Optional[List[str]] = None) -> int:
    """`python -m research.supply_chain`, for one name or the universe.

    The nightly pass runs this as a stage with the names it has already
    screened. This entry point is for the rest: reading one name by hand, or
    catching up a store whose eligible universe has moved.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="supply_chain",
        description=("Read 10-K Business sections and record who each filer "
                     "names as a supplier, customer, competitor or partner. "
                     "One filing's worth of edges per name, stamped at the "
                     "filing date."))
    parser.add_argument("--tickers", nargs="*", default=None,
                        help="names to read (default: today's eligible universe)")
    parser.add_argument("--as-of", dest="as_of", default=None,
                        help="the date to read as (default: today UTC)")
    parser.add_argument("--limit", type=int, default=MAX_PER_RUN,
                        help=f"how many names to read (default {MAX_PER_RUN})")
    args = parser.parse_args(argv)

    # A fresh volume answers "no such table" otherwise, and this job can be
    # the first command run against one.
    pit_store.init_schema()
    result = record(args.tickers, as_of=args.as_of, limit=args.limit)
    print(json.dumps(result, indent=1, sort_keys=True))
    return 0 if result["status"] in ("ok", "partial") else 1


if __name__ == "__main__":  # pragma: no cover - exercised via main()
    raise SystemExit(main())
