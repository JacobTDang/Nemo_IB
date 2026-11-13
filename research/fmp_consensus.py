"""The consensus history that a second vendor can reach back far enough for.

`sue_af` needs eight quarters of what the street expected before each print.
Three things stand in its way, and this module is about the first one only.

  1. The free vendor serves four quarters and no more -- Finnhub returns four
     at limit=12 and at limit=30 alike, verified -- so the record accrues one
     day at a time and the analyst leg is refused for two years. This is a
     vendor limit, and another vendor lifts it.

  2. A street estimate is not a GAAP number. The pair has to come from one
     table or the surprise measures the difference in definitions, not the
     difference from expectation.

  3. An estimate revised after the print would be the answer to the question
     being asked, so any reconstruction has to say so in the data.

What this adds is a vendor whose surprise history runs to about forty quarters
per name, each row carrying both the estimate and the actual on that vendor's
own basis. That is enough for the eight-quarter standardisation on the day the
first run happens rather than two years later, and it is a *reconstruction*: no
key was being watched on those days, so every row it writes is stamped at the
period it describes and carries `source='fmp'`. `sue_af` reports how many of
its quarters came that way, and a reader that wants observations only can
filter them out. The grouping `seeded` already draws is the same one, and it is
counted the same way rather than hidden in a column nobody reads.

It runs inside `research-seed`, on the same two EDGAR reads that pass already
makes, and before the free vendor's four quarters -- so a quarter reconstructed
here is skipped there rather than derived twice. `main()` is the entry point
for running it alone, which is how a new key gets tested on one name instead of
six hundred.

Two things this refuses to do.

**Guess the fiscal quarter.** The vendor dates a surprise by the day it was
released. Everything else in this package keys a quarter by the filer's own
fiscal identity, and the two do not always agree -- a calendar-bucket date put
NVDA's fiscal 2027 Q2 a year into the future the first time seeding ran. So a
row is matched to a quarter through the release EDGAR already recorded for it,
within a week, and a row that matches nothing or matches two quarters is
counted and dropped rather than attached to a coin flip.

**Guess the payload.** The vendor has shipped two spellings of the same two
fields over its versions, and a third shape would be a layout change. Both
known spellings parse; anything else is a refusal that names the keys that
arrived, so the first run with a real key says what changed instead of writing
zeros.

There is no key in the repository and none is needed to run the offline suite:
without `FMP_API_KEY` the module reports `no_api_key`, writes nothing, and
leaves the Finnhub path to do what it already does. Nothing else degrades,
because nothing else reads it until it has written something.
"""
from __future__ import annotations

import json
import os
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from research import pit_store

# The stable API. The v3 path still answers for older accounts, but a new key
# is issued for this one, and a base URL that can be pointed at a test double is
# worth more than the difference between them.
FMP_BASE = "https://financialmodelingprep.com/stable"

KEY_ENV = "FMP_API_KEY"

# This vendor's rows become the record, so the row says which vendor. Not
# `seeded`: a different vendor's estimate is a different number, and a window
# that mixes them without saying so measures the vendors' disagreement as
# dispersion.
SOURCE = "fmp"

TIMEOUT_SECONDS = 30.0
RETRIES = 1
RETRY_BACKOFF_SECONDS = 2.0
RETRY_STATUSES = (429, 500, 502, 503, 504)

# How far a vendor's release date may sit from the release EDGAR recorded and
# still be the same print. The two sources read the same 8-K, so they agree to
# the day in the ordinary case; a market-holiday rollover or a vendor dating a
# release to its own timezone moves it by one. Quarters are about ninety days
# apart, so a week cannot reach the neighbouring print.
MATCH_DAYS = 7

# The two spellings the vendor has used for the same pair. Ordered, because a
# payload carrying both should be read the same way every run.
_ACTUAL_FIELDS = ("actualEarningResult", "actualEPS")
_ESTIMATE_FIELDS = ("estimatedEarningResult", "estimatedEPS")


class FmpError(RuntimeError):
    """A fetch that failed, with a machine-readable cause.

    The same vocabulary as the rest of the package: `no_api_key` when the
    module was asked to fetch without one, `provider_unavailable` when nothing
    answered, `rate_limited` on a 429, `parse_failed` when the payload is not
    the shape this reads, `bad_request` when the vendor rejected the query.
    """

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason
        self.message = message


# ------------------------------------------------------------------ availability

def api_key() -> Optional[str]:
    """The key, or None. Never a default and never a placeholder.

    The vendor rejects `demo` outright -- measured on both the v3 and the stable
    path -- so a placeholder would turn "not configured" into "failed" and make
    the monthly job's report say the wrong thing.
    """
    key = (os.environ.get(KEY_ENV) or "").strip()
    return key or None


def available() -> bool:
    return api_key() is not None


# ----------------------------------------------------------------------- fetch

def _client():
    import httpx

    return httpx.Client(
        timeout=TIMEOUT_SECONDS,
        headers={"Accept": "application/json",
                 "User-Agent": "nemo-research/1.0 (consensus history)"})


def _post_get(client, url: str, params: Dict[str, Any]):
    """One GET with a retry on the transport and on a busy edge.

    A 404 is an answer (the vendor has no history for that symbol) and comes
    back as the response; a 429 or a 5xx is worth asking again once.

    Only the transport's own errors are retried, and only transport answers
    with a status. A bug in this module, or anything else standing in for the
    network, will raise the same way however many times it is asked, and
    retrying it buys a sleep and no answer.
    """
    import time

    import httpx

    for attempt in range(RETRIES + 1):
        try:
            resp = client.get(url, params=params)
        except httpx.HTTPError as exc:
            if attempt < RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))
                continue
            raise FmpError(
                "provider_unavailable",
                f"FMP did not answer: {type(exc).__name__}: {exc}") from None
        if resp.status_code in RETRY_STATUSES and attempt < RETRIES:
            time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))
            continue
        return resp
    raise FmpError("provider_unavailable", "FMP did not answer")


def fetch_surprises(ticker: str, client=None) -> List[Dict[str, Any]]:
    """One name's surprise history, oldest first, in this module's names.

    Returns rows of `{"announced_date", "estimate", "actual"}`. A row with no
    date, or with only one leg, is dropped here: half a pair cannot make a
    surprise, and the count of what was kept is what the caller reports.
    """
    key = api_key()
    if key is None:
        raise FmpError("no_api_key",
                       f"{KEY_ENV} is not set, so there is no vendor to ask. "
                       f"This is a paid endpoint and there is no default")
    own = client is None
    conn = _client() if own else client
    try:
        resp = _post_get(conn, f"{FMP_BASE}/earnings-surprises",
                         {"symbol": ticker.upper(), "apikey": key})
        if resp.status_code == 429:
            raise FmpError("rate_limited",
                           f"FMP rate-limited the request for {ticker}")
        if resp.status_code in (401, 403):
            raise FmpError("bad_request",
                           f"FMP refused the key for {ticker} "
                           f"(HTTP {resp.status_code}). The free tier does not "
                           f"cover this endpoint")
        if resp.status_code != 200:
            raise FmpError("provider_unavailable",
                           f"FMP returned HTTP {resp.status_code} for {ticker}")
        raw = resp.text or ""
        try:
            payload = json.loads(raw) if raw.strip() else []
        except ValueError as exc:
            raise FmpError(
                "parse_failed",
                f"FMP's answer for {ticker} was not JSON: {exc}") from None
    finally:
        if own:
            conn.close()

    # A quota error arrives as a 200 with an object rather than a list, and
    # reading it as an empty history would say "this company has never
    # reported" about every company at once.
    if isinstance(payload, dict):
        detail = payload.get("Error Message") or payload.get("message") \
            or payload
        raise FmpError("bad_request",
                       f"FMP answered {ticker} with an object, not a list: "
                       f"{str(detail)[:160]}")
    if not isinstance(payload, list):
        raise FmpError("parse_failed",
                       f"FMP's answer for {ticker} was "
                       f"{type(payload).__name__}, not a list of rows")

    rows: List[Dict[str, Any]] = []
    for record in payload:
        if not isinstance(record, dict):
            raise FmpError("parse_failed",
                           f"FMP's answer for {ticker} held a "
                           f"{type(record).__name__} where a row belongs")
        announced = str(record.get("date") or "").strip()[:10]
        # A date is parsed here rather than trusted: everything below compares
        # it as a string and matches it against a release, and a value that is
        # not a date would raise from the middle of a six-hundred-name run
        # instead of being counted as the unreadable row it is.
        try:
            date.fromisoformat(announced)
        except ValueError:
            continue
        estimate, actual = None, None
        for name in _ESTIMATE_FIELDS:
            if record.get(name) is not None:
                estimate = _number(record.get(name))
                break
        for name in _ACTUAL_FIELDS:
            if record.get(name) is not None:
                actual = _number(record.get(name))
                break
        if not announced or estimate is None or actual is None:
            continue
        rows.append({"announced_date": announced, "estimate": estimate,
                     "actual": actual})
    if payload and not rows:
        # Every row had a shape this does not know. The keys are the whole
        # message an operator needs on the first run with a real key.
        raise FmpError(
            "parse_failed",
            f"FMP's answer for {ticker} held {len(payload)} rows and none "
            f"carried a date with both figures; the keys were "
            f"{sorted(payload[0].keys()) if isinstance(payload[0], dict) else payload[0]!r}")
    return sorted(rows, key=lambda r: r["announced_date"])


def _number(value: Any) -> Optional[float]:
    """A finite float, or None. The vendor sends null for a missing leg."""
    import math

    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


# ---------------------------------------------------------------------- record

def _match(announced: str,
           releases: Dict[str, Dict[str, Any]]) -> tuple:
    """Which fiscal period a vendor row belongs to, or why it is not one.

    Returns `(period, None)` or `(None, reason)`. Matching on the release EDGAR
    recorded is the only join that keeps the filer's own fiscal identity: the
    vendor's date is a release date, and its fiscal labels are a calendar
    bucket in at least one version of the API.
    """
    day = date.fromisoformat(announced)
    near = []
    for period, release in releases.items():
        theirs = (release.get("announced_date") or "")[:10]
        if not theirs:
            continue
        gap = abs((date.fromisoformat(theirs) - day).days)
        if gap <= MATCH_DAYS:
            near.append((gap, period))
    if not near:
        return None, "no release on record within a week of that date"
    if len(near) > 1:
        return None, (f"{len(near)} releases fall within a week of that date "
                      f"({sorted(p for _, p in near)})")
    return near[0][1], None


def record(ticker: str, as_of: Optional[str] = None, *,
           client=None, series: Optional[Dict[str, Any]] = None,
           releases: Optional[Dict[str, Dict[str, Any]]] = None
           ) -> Dict[str, Any]:
    """Write one name's reconstructed history. Reports what it could not place.

    The estimate is stamped at the filer's period end, which is before the print
    by construction; the figure is the final pre-print consensus rather than
    whatever stood on that day, which is the right number for a surprise and an
    approximate date for it. The actual is stamped at the release, because it
    had not happened before then and a reconstruction may not reconstruct the
    future.

    `as_of` bounds everything: a row whose release is after it is not written,
    so replaying last quarter does not learn this quarter's print.
    """
    as_of = as_of or _today()
    if series is None:
        from research import sue

        series = sue.eps_series(ticker, as_of=as_of)
    if not series.get("success"):
        return {"ticker": ticker, "status": "failed",
                "reason": series.get("error") or "no EPS series",
                "written": 0, "rows": 0, "unmatched": 0, "ambiguous": 0,
                "future": 0, "already": 0}

    periods = {q["fiscal_period"]: q for q in series["quarters"]}
    if releases is None:
        from research import announcements

        by_period = {p: {"period_end": q.get("period_end"),
                         "known_at": q.get("known_at")}
                     for p, q in periods.items()}
        releases = announcements.for_quarters(
            ticker, as_of=as_of, quarters=by_period,
            cik=series.get("cik"))

    try:
        rows = fetch_surprises(ticker, client=client)
    except FmpError as exc:
        return {"ticker": ticker, "status": "failed", "reason": exc.reason,
                "error": exc.message, "written": 0, "rows": 0,
                "unmatched": 0, "ambiguous": 0, "future": 0, "already": 0}

    written = unmatched = ambiguous = future = already = half = 0
    unmatched_dates: List[str] = []
    for row in rows:
        announced = row["announced_date"]
        if announced > as_of:
            # A surprise that has not been released yet. Its actual is the
            # future, and writing it would be the purest lookahead there is.
            future += 1
            continue
        period, reason = _match(announced, releases)
        if period is None:
            if reason and reason.startswith("no release"):
                unmatched += 1
                unmatched_dates.append(announced)
            else:
                ambiguous += 1
            continue
        quarter = periods.get(period) or {}
        period_end = (quarter.get("period_end") or "")[:10]
        if not period_end:
            half += 1
            continue
        # A quarter already on record was watched. A reconstruction does not
        # restate an observation, and INSERT OR IGNORE would leave the two
        # rows disagreeing about which one a reader should prefer.
        if pit_store.actual_as_of(ticker, period, "9999-12-31") is not None:
            already += 1
            continue
        written += pit_store.record_consensus(
            period_end, ticker, period, eps_estimate=row["estimate"],
            recorded_at=_stamp(period_end), source=SOURCE) or 0
        written += pit_store.record_consensus(
            announced, ticker, period, eps_estimate=row["estimate"],
            eps_actual=row["actual"], recorded_at=_stamp(announced),
            source=SOURCE) or 0

    return {"ticker": ticker, "status": "ok", "reason": None, "error": None,
            "written": written, "rows": len(rows), "unmatched": unmatched,
            "unmatched_dates": unmatched_dates, "ambiguous": ambiguous,
            "future": future, "already": already, "undated": half}


def _stamp(day: str) -> str:
    """End of the day it describes, the way every other reconstruction here is
    stamped. Reads compare at date granularity, so the time is a formality."""
    return f"{day}T21:00:00Z"


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


# ----------------------------------------------------------------------- seed

def seed(tickers: Optional[Sequence[str]] = None,
         as_of: Optional[str] = None) -> Dict[str, Any]:
    """Reconstruct what this vendor covers, for each name, and say so.

    `tickers` defaults to the eligible universe, the same list the Finnhub
    seeding reads, so the two fill the same record and a quarter one has already
    reconstructed is not rebuilt by the other.
    """
    if not available():
        return {"status": "skipped", "reason": "no_api_key",
                "note": (f"{KEY_ENV} is not set, so no FMP history was asked "
                         f"for. The free vendor's four quarters are unaffected"),
                "tickers": 0, "written": 0, "names": []}
    if tickers is None:
        from research import daily_job

        names = [t for t, _ in daily_job.eligible_names(as_of)]
    else:
        names = list(tickers)

    summary = {"as_of": as_of or _today(), "tickers": len(names),
               "written": 0, "names": [], "failed": []}
    for ticker in names:
        result = record(ticker, as_of=as_of)
        summary["written"] += result["written"]
        if result["status"] != "ok":
            summary["failed"].append(
                f"{ticker}: {result.get('error') or result.get('reason')}")
        elif result["written"] or result["unmatched"] or result["ambiguous"]:
            summary["names"].append(result)
    summary["status"] = ("failed" if summary["failed"] and not summary["written"]
                         else "ok")
    return summary


# -------------------------------------------------------------- entry point

def main(argv: Optional[List[str]] = None) -> int:
    """`python -m research.fmp_consensus`, for one name or the universe.

    The monthly path is `research-seed`, which calls this for every name it is
    already reading EDGAR for. This entry point exists because a new key is
    best tested on one name, and because a name outside the eligible universe
    can be reconstructed without asking the free vendor for its four quarters
    as well.

    Safe to re-run: the vendor's history changes only when a company reports,
    and a quarter already on record is never restated.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="fmp_consensus",
        description=("Reconstruct consensus history from FMP's surprise "
                     "record. A reconstruction, not an observation: rows are "
                     "marked source='fmp'."))
    parser.add_argument("--tickers", nargs="*", default=None,
                        help="names to reconstruct (default: today's eligible universe)")
    parser.add_argument("--as-of", dest="as_of", default=None,
                        help="the date to read as (default: today UTC)")
    args = parser.parse_args(argv)

    # A fresh volume answers "no such table" otherwise, and this job is a
    # likely first command against one.
    pit_store.init_schema()
    run_id = pit_store.start_run("fmp_consensus",
                                 as_of_date=args.as_of or _today())
    result = seed(args.tickers or None, as_of=args.as_of)

    if result["status"] == "skipped":
        # Not a failure: an unconfigured paid vendor is the expected state on
        # a deployment that has not bought one.
        pit_store.finish_run(rows_written=0, status="ok",
                             error=result["reason"], run_id=run_id)
        print(json.dumps(result, indent=2, default=str))
        return 0

    pit_store.finish_run(
        rows_written=result["written"],
        status="failed" if result["status"] == "failed" else "ok",
        error=("; ".join(result["failed"][:5])
               if result.get("failed") else None),
        run_id=run_id)
    print(json.dumps(result, indent=2, default=str))
    return 1 if result["status"] == "failed" else 0


if __name__ == "__main__":  # pragma: no cover - exercised via main()
    raise SystemExit(main())
