"""How crowded the short side is, from the only free feed that publishes it.

`research.borrow` prices a short position and refuses when it has no quote.
This module does not lift that refusal and cannot: a borrow fee is what a
lender charges to hand over shares, and nobody publishes it for free. What
FINRA publishes, twice a month and at no cost, is how many shares *are* sold
short, the average daily volume behind them, and the days of trading it would
take to cover them. That is a crowding measure, not a price, and the two are
kept apart everywhere in the store so that one can never be read as the other:

  borrow_rate      what a lender quoted, annualised. Prices a short.
  short_interest   how many shares were short on a settlement date. Does not.

The distinction is not pedantry, it is the direction of the error. A name that
is cheap to borrow and heavily shorted is a crowded trade with a low fee; a
name that is barely shorted and expensive to borrow is neither. Substituting
the count for the fee would price the most crowded name at zero, which is the
exact failure `borrow.py` was written to prevent, arriving from a new place.

What the count is good for, and why it is worth a nightly job:

  * A days-to-cover figure bounds how much of a name's float a short exit has
    to buy back. Five days of average volume is a different proposition from
    half a day, at any borrow fee.
  * The change against the previous settlement is a flow. A short position
    that doubled in a fortnight says the trade got popular, which is what a
    drift strategy wants to know before it joins the crowd.
  * A name with no reported short interest at all is evidence about
    borrowability even though it is not a rate.

Coverage, measured rather than assumed: FINRA's consolidated file covers
exchange-listed and over-the-counter equity securities and runs to about
20,000 symbols per settlement date. Settlement dates are the 15th and the last
business day of each month. Reported positions are what member firms filed;
they exclude certain market-making and arbitrage positions, so the count is
lower than total short exposure, and it is a stock rather than a flow.

The point-in-time problem, which is the whole reason this is a job instead of a
reader. A settlement dated 30 October becomes readable somewhere around 10
November -- FINRA aggregates what its members reported and publishes after.
So a row carries two dates: `settlement_date`, the fact's date, and
`recorded_at`, when we could first have known it. `pit_store` filters on both,
which means:

  * the nightly job writes what it can see tonight, stamped tonight, and a
    scan dated before tonight does not see it;
  * `--backfill` exists for loading history, and it refuses to invent the
    date it became public. It requires `--published`, the day the settlement
    actually appeared, and records `backfilled=1` so a study can tell the two
    kinds of evidence apart. `borrow.py` makes the same argument about broker
    files, and the warning it prints applies here word for word: a row stamped
    at a date it was not available to is lookahead wearing a backfill's
    clothes.

Transport. FINRA's data API answers anonymously and takes a POST with a
`compareFilters` body; the GET form silently ignores the filter and returns
2020, which is a wrong answer that looks like a right one, so the POST is not
optional. A settlement longer than one page (5,000 rows) is paged and
reassembled in full or not written at all -- a half-recorded settlement prices
a third of the universe and refuses the rest with nothing saying which.

Register nothing and configure nothing: no key, no account, no environment
variable beyond `NEMO_PIT_DB`, which every store reader already reads.
"""
from __future__ import annotations

import json
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from research import pit_store

FINRA_URL = ("https://api.finra.org/data/group/otcMarket/name/"
             "consolidatedShortInterest")

# The API caps a page at 5,000 rows whatever is asked for. A settlement runs to
# about 20,000 symbols, so four pages is the ordinary shape and MAX_PAGES is
# only a stop for a paging bug that would otherwise loop forever.
PAGE_SIZE = 5000
MAX_PAGES = 40

TIMEOUT_SECONDS = 90.0
RETRIES = 2
RETRY_BACKOFF_SECONDS = 3.0
# 429 is "ask again"; 5xx is "the service is unwell". Both are transport
# failures rather than answers. A 400 is our fault and this code does not
# retry it into a hanging job.
RETRY_STATUSES = (429, 500, 502, 503, 504)

# On a cold store, how far back the first run reaches. Four settlements is two
# months and about 80,000 rows: enough for a change figure and a days-to-cover
# trend, not so much that the first night is a backfill nobody asked for.
COLD_START_DAYS = 60

# How many settlements one run will ingest. A store left alone for a year is
# sixty settlements; fetching all of them in one nightly pass would hold the
# job open for an hour. The missing ones are taken oldest first, so the file
# grows contiguously and catches up over consecutive nights; a store neglected
# for months reads current again before it reads complete.
MAX_SETTLEMENTS_PER_RUN = 6


class ShortInterestError(RuntimeError):
    """A settlement that could not be fetched, with a machine-readable cause.

    Shaped like `quiver_server.client.QuiverError` so the same vocabulary
    applies: `provider_unavailable` when nothing answered, `rate_limited` when
    the answer was 429, `parse_failed` when the payload was not what FINRA
    documents, `bad_request` when the API rejected the query itself.
    """

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason
        self.message = message


# --------------------------------------------------------------------- fetch

def _client():
    import httpx

    return httpx.Client(
        timeout=TIMEOUT_SECONDS,
        headers={"Accept": "application/json",
                 "Content-Type": "application/json",
                 # FINRA's API is a public dataset; naming the tool is the
                 # same courtesy SEC fair access asks for, and it makes an
                 # operator's traffic identifiable when they rate limit.
                 "User-Agent": "nemo-research/1.0 (short-interest ingest)"})


def _post(settlement: str, *, limit: int = PAGE_SIZE, offset: int = 0,
          compare: str = "EQUAL", client=None) -> List[Dict[str, Any]]:
    """One page, filtered on the server. Raises rather than returning part.

    The filter is `compareFilters` in a POST body. The same filter as a query
    string is accepted and ignored -- measured: `domainFilters` with a 2026
    date returned the 2020-04-15 rows -- so a GET would look like it worked
    and quietly fill the store with a six-year-old settlement.
    """
    body = {
        "limit": limit,
        "offset": offset,
        "compareFilters": [{"fieldName": "settlementDate",
                            "fieldValue": settlement,
                            "compareType": compare}],
    }
    own = client is None
    conn = _client() if own else client
    import httpx

    try:
        for attempt in range(RETRIES + 1):
            try:
                resp = conn.post(FINRA_URL, content=json.dumps(body))
            # Only the transport's own errors are worth another try. A bug in
            # this module, or anything else standing in for the network, will
            # raise the same way however many times it is asked, and retrying
            # it buys three sleeps and no answer.
            except httpx.HTTPError as exc:
                if attempt < RETRIES:
                    time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))
                    continue
                raise ShortInterestError(
                    "provider_unavailable",
                    f"FINRA did not answer for settlement {settlement}: "
                    f"{type(exc).__name__}: {exc}") from None

            if resp.status_code in RETRY_STATUSES:
                if attempt < RETRIES:
                    delay = RETRY_BACKOFF_SECONDS * (attempt + 1)
                    retry_after = resp.headers.get("retry-after")
                    if retry_after:
                        try:
                            delay = min(max(float(retry_after), 0.0), 30.0)
                        except ValueError:
                            pass  # an HTTP date, not a delay in seconds
                    time.sleep(delay)
                    continue
                reason = ("rate_limited" if resp.status_code == 429
                          else "provider_unavailable")
                raise ShortInterestError(
                    reason,
                    f"FINRA returned HTTP {resp.status_code} for settlement "
                    f"{settlement} after {RETRIES} retries")

            if resp.status_code == 204:
                # No content is a documented answer, not an error: the
                # settlement exists in the calendar and has not been published
                # yet. An empty list says so without inventing a failure.
                return []

            if resp.status_code != 200:
                detail = (resp.text or "")[:200]
                raise ShortInterestError(
                    "bad_request",
                    f"FINRA returned HTTP {resp.status_code} for settlement "
                    f"{settlement}: {detail}")

            raw = resp.text or ""
            if not raw.strip():
                return []
            try:
                payload = json.loads(raw)
            except ValueError as exc:
                raise ShortInterestError(
                    "parse_failed",
                    f"FINRA's answer for settlement {settlement} was not "
                    f"JSON: {exc}") from None
            if not isinstance(payload, list):
                raise ShortInterestError(
                    "parse_failed",
                    f"FINRA's answer for settlement {settlement} was "
                    f"{type(payload).__name__}, not a list of rows")
            return payload
    finally:
        if own:
            conn.close()


# The fields this store keeps, and the name each is filed under. The source
# names are FINRA's, in full, because the next person to check the mapping
# against their documentation needs to find it without guessing.
FINRA_FIELDS = {
    "symbolCode": "ticker",
    "currentShortPositionQuantity": "shares_short",
    "previousShortPositionQuantity": "prev_shares_short",
    "averageDailyVolumeQuantity": "avg_daily_volume",
    "daysToCoverQuantity": "days_to_cover",
    "changePercent": "change_pct",
    "issueName": "issue_name",
    "marketClassCode": "market",
    "stockSplitFlag": "split_flag",
}


def _row(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """One FINRA row in the store's names, or None when it has no symbol.

    A row with no `symbolCode` cannot be joined to anything: non-equity entries
    come back on some settlements (rights, units, test symbols) and they are
    dropped here rather than written under a blank ticker where they would sit
    forever. The run reports the rows it kept, so a settlement that lost a
    slice to this reads as the smaller settlement it is.
    """
    ticker = (record.get("symbolCode") or "").strip().upper()
    if not ticker:
        return None
    row = {name: record.get(source) for source, name in FINRA_FIELDS.items()}
    row["ticker"] = ticker
    return row


def fetch_settlement(settlement: str,
                     client=None) -> List[Dict[str, Any]]:
    """Every row for one settlement date. All of it, or an exception.

    Paged at 5,000 and reassembled before anything is written. A page that
    fails after two others succeeded raises rather than returning what it has:
    eleven thousand rows of a nineteen-thousand-row settlement is a dataset
    that looks whole and answers for 58% of the universe.
    """
    rows: List[Dict[str, Any]] = []
    offset = 0
    for _ in range(MAX_PAGES):
        page = _post(settlement, limit=PAGE_SIZE, offset=offset,
                     client=client)
        if not page:
            break
        for record in page:
            if not isinstance(record, dict):
                raise ShortInterestError(
                    "parse_failed",
                    f"FINRA's settlement {settlement} held a "
                    f"{type(record).__name__} where a row belongs")
            mapped = _row(record)
            if mapped is not None:
                rows.append(mapped)
        if len(page) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
    else:
        raise ShortInterestError(
            "parse_failed",
            f"settlement {settlement} did not end after {MAX_PAGES} pages; "
            f"paging is not making progress and a partial settlement is worse "
            f"than none")
    return rows


def next_settlement(after: str, client=None) -> Optional[str]:
    """The earliest settlement date after `after`, or None if there is none.

    One row is enough to learn the date, and the API orders by settlement
    date, so this costs a single call. This is how the job finds what to
    fetch without guessing at the calendar: a settlement is the 15th or the
    last business day, but a holiday moves it, and a guessed date that misses
    is indistinguishable from a settlement nobody filed.
    """
    page = _post(after, limit=1, compare="GREATER", client=client)
    if not page or not isinstance(page[0], dict):
        return None
    day = (page[0].get("settlementDate") or "").strip()
    return day or None


def settlements_after(after: str, *, cap: int = MAX_SETTLEMENTS_PER_RUN,
                      until: Optional[str] = None,
                      client=None) -> List[str]:
    """Settlement dates after `after`, oldest first, at most `cap` of them.

    `until` stops the walk at a date. A run replayed for last month asks for
    last month's settlements, and walking past them would spend requests to
    find rows the run is going to discard.
    """
    found: List[str] = []
    cursor = after
    for _ in range(max(cap, 1)):
        day = next_settlement(cursor, client=client)
        if day is None or day <= cursor:
            break
        if until is not None and day > until:
            break
        found.append(day)
        cursor = day
    return found


def newest_settlement(as_of: Optional[str] = None,
                      client=None) -> Optional[str]:
    """The most recent published settlement on or before `as_of`.

    Walks forward from a floor rather than backwards through the calendar, so
    a settlement Friday the 14th (the 15th was a holiday) is found by the same
    code that finds the ordinary ones.
    """
    as_of = as_of or _today()
    floor = (date.fromisoformat(as_of)
             - timedelta(days=COLD_START_DAYS)).isoformat()
    days = settlements_after(floor, cap=MAX_SETTLEMENTS_PER_RUN,
                             client=client)
    return days[-1] if days else None


# ------------------------------------------------------------------- record

def record(as_of: Optional[str] = None, *, settlement: Optional[str] = None,
           since: Optional[str] = None, backfill: Optional[str] = None,
           source: str = pit_store.SOURCE_FINRA,
           client=None) -> Dict[str, Any]:
    """Bring the store up to date. Returns what was written, per settlement.

    `backfill` is the date the settlement became public -- the day a reader
    could have had it. Supplying it is how history is loaded, and it is the
    only way to load it: this function will not infer a publication date from
    a settlement date, because the lag is FINRA's to decide and an inferred one
    that runs early is lookahead.

    An empty run is the normal case. Settlements arrive twice a month, so most
    nights find nothing new and say so; only a failure is reported as one.

    `client` exists for the tests and for a caller that already holds a
    connection; everything below goes through it, so a test never reaches the
    network to discover what a *second* settlement date is.
    """
    as_of = as_of or _today()
    pit_store.init_schema()
    pit_store.start_run("short_interest", as_of_date=as_of)

    try:
        if settlement:
            targets = [settlement]
        else:
            known = pit_store.short_interest_settlements()
            floor = known[0]["settlement_date"] if known else (
                since or (date.fromisoformat(as_of)
                          - timedelta(days=COLD_START_DAYS)).isoformat())
            if since and known and since > floor:
                floor = since
            # A run for a past date records at most that date's settlement. The
            # row would be stamped today either way, and a settlement that
            # closed after the day being replayed has no business being found
            # by it.
            targets = settlements_after(floor, until=as_of, client=client)
    except ShortInterestError as exc:
        pit_store.finish_run(rows_written=0, status="failed",
                             error=exc.message)
        return {"status": "failed", "reason": exc.reason,
                "error": exc.message, "written": 0, "settlements": []}

    if backfill is not None and not settlement:
        detail = ("--backfill loads one settlement at a published date and "
                  "needs the settlement named; a bulk run cannot be stamped "
                  "with one date, because each settlement became public on a "
                  "different day")
        pit_store.finish_run(rows_written=0, status="failed", error=detail)
        return {"status": "failed", "reason": "bad_request", "error": detail,
                "written": 0, "settlements": []}

    # End of the day it describes when backfilling, now otherwise. Reads
    # compare at date granularity, so the time is a formality in both cases.
    recorded_at = (f"{backfill}T21:00:00Z" if backfill else None)
    written = 0
    done: List[Dict[str, Any]] = []
    for day in targets:
        try:
            rows = fetch_settlement(day, client=client)
        except ShortInterestError as exc:
            pit_store.finish_run(rows_written=written, status="failed",
                                 error=exc.message)
            return {"status": "failed", "reason": exc.reason,
                    "error": exc.message, "written": written,
                    "settlements": done}
        counts = pit_store.record_short_interest(
            day, rows, source=source, recorded_at=recorded_at,
            backfilled=backfill is not None)
        written += counts["written"]
        done.append({"settlement_date": day, "rows": len(rows), **counts})

    pit_store.finish_run(rows_written=written, status="ok")
    return {"status": "ok", "written": written, "settlements": done,
            "note": (None if done else
                     "no settlement has been published since the last one on "
                     "file; FINRA publishes twice a month, so this is the "
                     "ordinary result")}


def _today() -> str:
    """Today in UTC, for the reason `borrow.py` gives: the store stamps UTC."""
    return datetime.now(timezone.utc).date().isoformat()


# -------------------------------------------------------------- entry point

def main(argv: Optional[List[str]] = None) -> int:
    """`python -m research.short_interest`.

    Nightly by default: fetch whatever settlements have appeared since the
    newest one on file. The two other shapes are for loading history and for
    recovering from a missed night, and both are explicit.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="short_interest",
        description=("Record FINRA's consolidated short interest, per "
                     "settlement date. This is a crowding count, not a borrow "
                     "rate: it does not price a short."))
    parser.add_argument("--as-of", dest="as_of",
                        help="the date this run is for (default: today UTC)")
    parser.add_argument("--settlement", dest="settlement",
                        help="one settlement date to ingest (YYYY-MM-DD)")
    parser.add_argument("--since", dest="since",
                        help=("start after this settlement date instead of "
                              "after the newest one on file"))
    parser.add_argument("--backfill", dest="backfill",
                        help=("the date this settlement became public, for "
                              "loading history. Requires --settlement. Rows "
                              "are marked backfilled so a study can tell them "
                              "from ones captured on the day"))
    args = parser.parse_args(argv)

    if args.backfill and not args.settlement:
        print("--backfill needs --settlement: one published date cannot "
              "describe a bulk load of several settlements")
        return 2

    # The container entry point never initialises this database, and a fresh
    # volume answered "no such table: run_log" until every job did this itself.
    # Idempotent, so it costs nothing on a store that exists.
    pit_store.init_schema()

    result = record(args.as_of, settlement=args.settlement, since=args.since,
                    backfill=args.backfill)

    if result["status"] != "ok":
        print(f"{result.get('reason') or 'failed'}: {result.get('error')}")
        return 1

    if not result["settlements"]:
        print(result["note"])
        return 0

    for done in result["settlements"]:
        print(f"{done['settlement_date']}: {done['rows']} rows, "
              f"{done['written']} written, {done['revised']} revised, "
              f"{done['unchanged']} unchanged")
    print(f"{result['written']} short-interest rows recorded")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via main()
    raise SystemExit(main())
