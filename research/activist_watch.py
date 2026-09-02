"""A watcher for Schedule 13D filings, where the edge is latency not analysis.

A 13D is filed when someone takes a >5% stake with the declared intent to
influence management. Brav, Jiang, Partnoy & Thomas measure 6.3-8% abnormal
returns around the filing. Treat that as an upper bound -- the signal has
compressed as activism grew crowded -- but note that an event does not decay
under crowding the way a factor does, because the disclosure happens once and
either you saw it or you did not.

Two things make it worth watching now. The 2024 rule change cut the initial
deadline from 10 calendar days to 5 business days, and amendments to 2, which
leaves less room to accumulate quietly before disclosing; proportionally more
of the move therefore lands at publication, where a detector can act on it. And
the thing being competed over is minutes, not insight: reading the filing when
it posts rather than the write-up next morning.

So the number this module exists to make small is the gap between SEC accepting
a filing and us seeing it, and every pass records that gap per filing rather
than reporting a count. A watcher whose latency is not measured is a watcher
nobody can tell has stopped working.

Which is the other half. **A quiet day and a broken watcher produce the same
empty list.** "No new 13Ds today" is information; "EDGAR did not answer" is an
outage that will be noticed weeks later as an unexplained hole. Every pass
writes a run-log entry recording which of the two it was, a pass that reached
nobody is never logged as coverage, and a pass that reached some of the
watchlist says how much of it.

The trap this had to be built around: a company's EDGAR folder holds BOTH sides
of a Schedule 13 relationship -- filings where it is the subject and filings it
made about other issuers. On INTC, 71 of the first 100 rows were Intel filing
on MariaDB, Mobileye, Joby and Vuzix. The rule that separates them is not
reinvented here; `sec_utils.classify_schedule13` owns it, this module applies
it, and the regression test uses INTC-shaped data.

Scale, honestly: one SEC request per ticker per pass, paced. That fits a
watchlist of tens polled every few minutes. It does not fit sweeping thousands
of names every minute -- for that the mechanism is EDGAR's market-wide
current-filings feed, one request for the whole market, which this module does
not implement. `max_seconds` exists so a universe-wide pass degrades into an
honestly-labelled partial rather than a fan-out into a rate limit -- and a
cursor means the next pass resumes where it stopped, so a slow day costs
coverage rate rather than leaving the tail of the list permanently unwatched.
The budget rather than `max_tickers` is the bound that holds: the same pass
cost 0.90s, 3.0s and 6.0s a ticker in three measurements on one afternoon.
"""
from __future__ import annotations

import statistics
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from research import daily_job, pit_store
from tools import filing_cache

JOB = "activist_watch"

# Initial and amendment. Kept apart everywhere downstream: a 13D says a stake
# was built, a 13D/A says something about a stake already disclosed -- an
# increase, a partial sale, a settlement, sometimes an exit. The expected
# reaction differs in sign, so collapsing them is a coin flip wearing a signal's
# clothes.
WATCHED_FORMS = ("SC 13D", "SC 13D/A")

# Between SEC calls. SEC's published ceiling is 10 requests/second and it
# enforces it hard; this project has been throttled repeatedly. Sequential and
# paced, never fanned out.
THROTTLE_SECONDS = 0.15


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z")


def _be_gentle() -> None:
    """Pace every SEC call, including the header reads inside a pass.

    A first look at a ticker turns its whole 13D history into new rows, each
    needing a header fetch. Pacing here rather than only around the folder
    request is what keeps that one-off burst under the rate limit instead of
    earning a 429 in the middle of the pass that finally sees something.
    """
    time.sleep(THROTTLE_SECONDS)


def _resolved_cik(ticker: str, cik: Any) -> str:
    """The registrant's CIK, or a refusal. Never a sentinel.

    `Company("PARA")` does not raise for a ticker EDGAR no longer maps. It
    returns a company whose CIK is -999999999 and whose filing list is empty --
    observed live. Empty is the same shape a genuinely quiet company returns,
    so without this the watchlist could rot one delisted name at a time, each
    reporting "no 13Ds" forever and counting as coverage while it did. That is
    the confusion this module exists to prevent, arriving through the vendor
    instead of through the network.
    """
    try:
        value = int(cik)
    except (TypeError, ValueError):
        value = -1
    if value <= 0:
        raise LookupError(
            f"EDGAR does not map {ticker} to a registrant (CIK {cik!r}). "
            f"Refused rather than treated as a company with no filings, which "
            f"is what an empty result from an unresolvable ticker looks like.")
    return str(value)


# --------------------------------------------------------------- seam

def _fetch_company_filings(ticker: str) -> Tuple[str, List[Any]]:
    """A ticker's Schedule 13D folder, from the submissions index.

    One network request: constructing the company fetches its submissions
    index and the form filters run in memory. Nothing here reads a document,
    which matters because the index already carries the `005-` file number that
    tells the two sides of the relationship apart, so most of a folder is
    classified for free.

    edgartools returns the same filing under both `SC 13D` and `SC 13D/A` when
    it is an amendment, so accessions are deduped here rather than left to
    produce two events for one filing.

    The single seam every network call in this module sits behind, so the
    failure modes above can be tested without a network -- which is the only
    way they get covered at all.
    """
    from edgar import Company

    from tools.web_search_server.sec_series import _require_identity

    # Refuses to invent a contact address, and sets edgartools' identity as a
    # side effect. SEC fair access requires a real one.
    _require_identity()

    company = Company(ticker)
    cik = _resolved_cik(ticker, company.cik)

    seen: set = set()
    filings: List[Any] = []
    for form in WATCHED_FORMS:
        for filing in company.get_filings(form=form):
            accession = getattr(filing, "accession_number", None)
            if not accession or accession in seen:
                continue
            seen.add(accession)
            filings.append(filing)
    _be_gentle()
    return cik, filings


# --------------------------------------------------------- run bookkeeping

def last_run() -> Optional[Dict[str, Any]]:
    return daily_job.last_run(JOB)


def _watchlist(as_of: str) -> List[str]:
    """Who to watch on `as_of`, read as the universe stood that day.

    Not a list in a config file. A name eligible in March and delisted in June
    was watched in March, and reading today's universe when replaying a March
    pass silently rewrites who was being observed -- the same survivorship the
    store exists to prevent, one level up.
    """
    return daily_job.eligible_tickers(as_of)


# ------------------------------------------------------------------ pass

def watch_pass(tickers: Optional[Sequence[str]] = None,
               as_of: Optional[str] = None,
               detected_at: Optional[str] = None,
               max_tickers: Optional[int] = None,
               max_seconds: Optional[float] = None) -> Dict[str, Any]:
    """One sweep. Records new subject-side 13D filings and how late we were.

    `tickers` defaults to the eligible universe as of the date. The pass is
    idempotent: a filing already recorded is not re-reported, and its original
    detection time is never overwritten -- a watcher on a timer re-reads the
    same folder every few minutes, and the natural bug is for the tenth pass to
    stamp its own time onto the first pass's discovery and report a latency of
    zero forever.

    The returned status is the point. `ok` means every requested name was
    reached, whether or not anything was found; `partial` means some were not,
    with the count; `failed` means none were, and a failed pass is never
    counted as coverage by `pit_store.missing_days`.

    `detected_at` defaults to now and is never derived from `as_of`, which is
    the opposite of what `daily_job._stamp` does and is deliberate. A replayed
    bar for a past session genuinely belongs to that day's knowledge; a
    detection does not, because when we saw something is a measurement and
    backdating it would fabricate the one number this watcher is judged on. A
    consequence worth knowing: a pass replayed for a past `as_of` records
    detections stamped today, and an as-of read of that past date correctly
    will not show them.
    """
    from tools.web_search_server import sec_utils

    as_of = as_of or _today()
    stamp = detected_at or _now()
    run_id = pit_store.start_run(JOB, as_of_date=as_of)

    if tickers is None:
        requested = _watchlist(as_of)
        empty_reason = (
            f"no eligible universe recorded for {as_of}, so this pass watched "
            f"nobody. Reporting it as a quiet day would claim the market was "
            f"observed when it was not -- run research.daily_job.refresh_"
            f"universe first.")
    else:
        requested = [t for t in tickers if t]
        empty_reason = ("the watchlist was empty, so this pass watched nobody "
                        "and its silence means nothing")

    if not requested:
        pit_store.finish_run(rows_written=0, status="failed",
                             error=empty_reason, run_id=run_id)
        return {"as_of": as_of, "status": "failed", "error": empty_reason,
                "requested": 0, "covered": 0, "new_events": 0, "events": [],
                "filed_by_this_company": 0, "subject_filter_disagreements": 0}

    total_requested = len(requested)
    notes: List[str] = []

    # Where the last pass stopped. A bounded pass that always starts at the
    # head is the blind spot with extra steps: whoever sorts last is never
    # watched on any pass, forever, while the status says `partial` and names
    # nobody.
    start = pit_store.cursor_for(JOB) % total_requested if total_requested \
        else 0
    if start:
        requested = requested[start:] + requested[:start]

    # A ticker cost 0.90s, then 3.0s, then 6.0s across three measurements on
    # one machine in one afternoon -- the variable is EDGAR's throttle state,
    # not the ticker count, so no fixed ticker cap can be sized against it. A
    # deadline holds whatever the vendor is doing, and the cursor turns "ran
    # out of time" into "got this far" rather than into a hole.
    deadline = (time.monotonic() + max_seconds) if max_seconds else None
    if max_tickers is not None and len(requested) > max_tickers:
        requested = requested[:max_tickers]

    covered = 0
    failures: List[str] = []
    events: List[Dict[str, Any]] = []
    filed_by_this_company = 0
    disagreements: List[str] = []

    watched = 0
    stopped_by = None
    for ticker in requested:
        if deadline is not None and time.monotonic() >= deadline:
            stopped_by = "budget"
            break
        watched += 1
        # The submissions index and every header fetched below are cached under
        # /root/.edgar and nothing removes one. The eviction that keeps the
        # servers alive is an asyncio task in the HTTP app's lifespan, and this
        # process never starts it -- so a sweep over the eligible universe
        # fills the 512MB tmpfs and dies mid-pass with `[Errno 28]`. Between
        # names is the only place a job with no event loop can prune; the gate
        # keeps it to one walk per interval. Asked before the fetch, so a name
        # that raises has still had what it already pulled counted.
        filing_cache.prune_if_due()

        # One try around the whole name, not just its fetch. The names in a
        # watchlist are independent, and a filing EDGAR describes in a way we
        # cannot parse should cost that name alone -- letting it escape would
        # silence every name after it and leave the run unfinished, which is a
        # silent outage wearing the shape of a short watchlist.
        try:
            cik, filings = _fetch_company_filings(ticker)

            known = pit_store.known_activist_accessions(ticker)
            fresh: List[Dict[str, Any]] = []
            local_filer_side = 0
            local_disagreements: List[str] = []

            for filing in filings:
                accession = getattr(filing, "accession_number", None)
                if not accession or accession in known:
                    continue

                # The free half first. A filing with no `005-` file number is
                # one this company made about someone else, and spending a
                # document fetch to confirm what the index already said is how
                # a watcher over a real watchlist earns a rate limit.
                if not sec_utils.schedule13_file_number_is_subject_side(filing):
                    local_filer_side += 1
                    continue

                _be_gentle()  # the header below is a document fetch
                verdict = sec_utils.classify_schedule13(filing, cik)
                if verdict["header_disagrees"]:
                    # The header is the authority and it says this company is
                    # not the subject. Noted rather than silently dropped: the
                    # file number agreed with headers on 28 of 28 INTC filings
                    # when it was chosen, and a non-zero count here means it
                    # has drifted.
                    local_disagreements.append(accession)
                    local_filer_side += 1
                    continue

                form = str(getattr(filing, "form", "") or "")
                fresh.append({
                    "accession": accession,
                    "subject_ticker": ticker,
                    "subject_cik": verdict["subject_cik"],
                    "subject_name": verdict["subject_name"],
                    "filer_name": verdict["filer_name"],
                    "filer_cik": verdict["filer_cik"],
                    "form": form,
                    "is_amendment": form.endswith("/A"),
                    "filing_date": str(getattr(filing, "filing_date", "")),
                    # Normalised here rather than on the way into the store, so
                    # the event handed back and the row written carry the same
                    # timestamp in the same shape. A caller joining one against
                    # the other must not have to guess which is a string.
                    "accepted_at": pit_store.iso_utc(
                        getattr(filing, "acceptance_datetime", None)),
                    "detected_at": stamp,
                    "subject_verified": verdict["subject_verified"],
                    "is_subject": verdict["is_subject"],
                    "url": getattr(filing, "filing_url", None),
                })

            if fresh:
                pit_store.record_activist_filings(fresh, detected_at=stamp)
        except Exception as exc:  # noqa: BLE001 - counted and reported, never masked
            failures.append(f"{ticker}: {type(exc).__name__}: {exc}")
            continue

        covered += 1
        events.extend(fresh)
        filed_by_this_company += local_filer_side
        disagreements.extend(local_disagreements)

    # Advance by what was actually watched, not by a nominal window: a pass
    # cut short by the budget must not skip the names it never reached.
    if total_requested:
        pit_store.set_cursor(JOB, (start + watched) % total_requested)

    # Against the whole watchlist, not against the slice this pass attempted.
    # A bounded pass genuinely has not covered the watchlist, and reporting it
    # as `ok` would let `missing_days` count a twentieth of a sweep as a
    # covered day. The status is the truth about coverage.
    #
    # That `partial` is the steady state for this job rather than a fault is a
    # question for whoever reads it, and `research.status` knows it.
    status = daily_job.coverage_status(covered, total_requested)
    if failures:
        notes.append(f"{len(failures)} of {total_requested} lookups failed: "
                     + "; ".join(failures[:5]))
    if watched < total_requested:
        resume = (start + watched) % total_requested
        reason = ("its time budget" if stopped_by == "budget"
                  else f"the {max_tickers}-name cap")
        notes.append(
            f"watched {watched} of {total_requested} names, from offset "
            f"{start}; stopped by {reason} and the next pass resumes at "
            f"{resume}. The names not reached were not watched, so their "
            f"silence carries no information")
    error = " | ".join(notes) if notes else None
    pit_store.finish_run(rows_written=len(events), status=status, error=error,
                         run_id=run_id)

    return {
        "as_of": as_of,
        "status": status,
        "error": error,
        "requested": total_requested,
        "covered": covered,
        "new_events": len(events),
        "events": events,
        # Real information, just the answer to a different question: these are
        # stakes the watched company took in other issuers.
        "filed_by_this_company": filed_by_this_company,
        "subject_filter_disagreements": len(disagreements),
    }


# ---------------------------------------------------------------- latency

def latency_report(as_of: Optional[str] = None,
                   ticker: Optional[str] = None) -> Dict[str, Any]:
    """How late we have been, over the detections that were detections.

    Back-filled rows are counted and excluded from the statistic rather than
    dropped from the table. The first look at any ticker turns its entire 13D
    history into new rows; averaging a 2019 filing first seen in 2026 into the
    latency would bury a genuine two-minute catch under five years of nothing,
    and the resulting median would say the watcher is useless when it is
    working, or -- worse, once the backlog ages out -- say nothing at all.

    A filing with no acceptance time is counted separately too. An unknown
    latency is not a fast one.
    """
    as_of = as_of or _today()
    rows = pit_store.activist_filings_as_of(as_of, ticker=ticker)

    live = [r["latency_seconds"] for r in rows if r["is_backfill"] is False]
    backfilled = sum(1 for r in rows if r["is_backfill"] is True)
    unknown = sum(1 for r in rows if r["is_backfill"] is None)

    return {
        "as_of": as_of,
        "events": len(rows),
        "live_detections": len(live),
        "backfilled": backfilled,
        "unknown_latency": unknown,
        "median_latency_seconds": statistics.median(live) if live else None,
        "worst_latency_seconds": max(live) if live else None,
    }


# ------------------------------------------------------------- entry point

def main(argv: Optional[List[str]] = None) -> int:
    """`python -m research.activist_watch`, on a short timer.

    Not a stage of the nightly job. A 13D moves the price when it lands, so the
    edge is the gap between the filing appearing on EDGAR and anyone reading
    it -- and a once-a-night pass measures that gap in hours however fast the
    code is. Running it every few minutes during the day is what the latency
    figures are for.

    A pass that finds nothing exits 0. Most of them find nothing; paging on
    that is how the one that matters gets ignored.

    Every run also reports the latency accumulated so far, because catching a
    filing quickly is the whole reason for the timer and a job that never says
    how quickly cannot be judged.
    """
    import argparse
    import json

    # Nothing else does, and the ordering that hides it is not enforced
    # anywhere: the recorder normally runs first and creates the store, so the
    # first command against a fresh volume dies on "no such table" instead.
    # This job is the least ordered of them all -- it runs on its own short
    # timer rather than in the nightly chain. Cheap and idempotent, so it runs
    # every pass rather than once.
    pit_store.init_schema()

    parser = argparse.ArgumentParser(
        prog="activist_watch",
        description="Sweep EDGAR for subject-side 13D filings.")
    parser.add_argument("--as-of", dest="as_of", default=None,
                        help="date whose eligible universe to sweep "
                             "(default: today)")
    parser.add_argument("--max-seconds", dest="max_seconds", type=float,
                        default=None,
                        help="stop requesting new names after this many "
                             "seconds and resume there next pass. The bound "
                             "that holds: a ticker cost 0.90s, 3.0s and 6.0s "
                             "in three measurements on one afternoon, so the "
                             "variable is EDGAR's throttle state rather than "
                             "the ticker count")
    parser.add_argument("--max-tickers", dest="max_tickers", type=int,
                        default=None,
                        help="cap the sweep, for a first pass on a cold store")
    args = parser.parse_args(argv)

    result = watch_pass(as_of=args.as_of, max_tickers=args.max_tickers,
                        max_seconds=args.max_seconds)

    # What it recorded is half the answer. The other half is how late it was,
    # which is the only figure that says whether a twenty-minute timer is
    # buying anything over a nightly one -- and it was computed here and never
    # reported. It is a property of the record rather than of this pass, so a
    # pass that could not reach EDGAR has not lost it.
    result["latency"] = latency_report(as_of=args.as_of)

    print(json.dumps(result, indent=2, default=str))
    return 1 if result.get("status") == "failed" else 0


if __name__ == "__main__":  # pragma: no cover - exercised via main()
    raise SystemExit(main())
