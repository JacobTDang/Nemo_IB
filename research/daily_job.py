"""The job that starts the clock.

`pit_store` is only worth having if something feeds it every day. This is that
something. Its interesting failure modes are not crashes -- a crash is loud and
gets fixed -- but the quiet ones: a run that fetched nothing and logged
success, a ticker the vendor ignored recorded as a session with no volume, a
back-fill stamped as though it had been known at the time.

Each sub-job writes its own run-log entry, so a day where bars landed and
consensus did not is visible as exactly that rather than as a good day.

Every network call sits behind a module-level `_fetch_*` function. That is not
indirection for its own sake -- it is what lets the whole job be tested without
a network, which is the only way the failure modes above get covered at all.
"""
from __future__ import annotations

import os
import statistics
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional

from research import _DOTENV_PATH, pit_store  # noqa: F401 - .env on import

# --- screen ---------------------------------------------------------------
# The floor is where spread starts eating the drift; the price floor is
# because a $2 stock's tick is a large fraction of a percent. Both are
# deliberately conservative -- a name wrongly excluded costs an opportunity,
# a name wrongly included costs money.
MIN_MEDIAN_DOLLAR_VOLUME = 500_000.0
MIN_PRICE = 5.0
MIN_HISTORY_SESSIONS = 60
SCREEN_LOOKBACK_SESSIONS = 60

_SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _today() -> str:
    return _utc_now().date().isoformat()


def _session_is_pending(as_of: str) -> bool:
    """Whether this run has dated itself ahead of the session it is recording.

    `_today()` is UTC and the schedule is host-local, so a 22:30 run in New
    York is stamped with tomorrow's date. A day that has not happened hands
    back exactly what a holiday hands back -- the canary's last bar is the
    previous session either way -- so no amount of vendor data separates the
    two and the clock has to. 21:00Z is the end of a session everywhere else
    in this module and is the end of one here.

    Only a run that took its date from the clock can be wrong about it, which
    is why a replay's explicit `as_of` is not second-guessed.
    """
    if as_of != _today():
        return False
    return _utc_now() < datetime.fromisoformat(f"{as_of}T21:00:00+00:00")


def _stamp(as_of: str) -> str:
    """When a run for `as_of` learned what it learned: the end of that day.

    Not `datetime.now()`. The two agree in production, where as_of is today,
    and differ exactly when a run is replayed for a past date -- where using
    now() would stamp old sessions as freshly learned and quietly grant every
    later query the lookahead the store exists to prevent. Reads compare at
    date granularity, so the time component is a formality.
    """
    return f"{as_of}T21:00:00Z"


def _source(as_of: str) -> str:
    """Whether this run stood on the session it is recording.

    `_stamp` backdates a replay to the day it describes, which is right for
    prices and says nothing about how the row was obtained -- and the two are
    not the same evidence. yfinance drops delisted tickers, so a night filled
    in three weeks later is missing exactly the names this store exists to
    preserve, and no reader could tell it from a night watched live.

    The store already draws this line everywhere else: `consensus_snapshot`
    separates 'recorded' from 'seeded', `activist_filing` derives
    `is_backfill`. This is the same distinction, in the same words.
    """
    return "recorded" if as_of == _today() else "backfilled"


# --------------------------------------------------------------- seams

def _record_probe(fetched: Dict[str, List[Dict[str, Any]]], stamp: str,
                  keep, requested: List[str],
                  source: str = "recorded") -> None:
    """Keep the canary's own sessions.

    It is fetched every night anyway, as the liveness probe, and it is also
    what the regime is measured from -- so throwing it away left that
    measurement depending on the index happening to fall in the rotation, and
    failing open when it did not: full book size, reported as "unknown", which
    reads like a calm market rather than a missing one.

    Recorded for its prices only. Universe membership comes from the SEC
    registrant list and is untouched by this, so it is never proposed as a
    trade on account of being stored.
    """
    if FETCH_CANARY in requested:
        return
    rows = fetched.get(FETCH_CANARY) or []
    if rows:
        _record_ticker(FETCH_CANARY, rows, stamp, keep=keep, source=source)


def _fetch_sec_tickers() -> List[Dict[str, Any]]:
    """Every ticker SEC maps to a registrant.

    SEC's own file rather than a vendor's list: a vendor's universe is already
    survivorship-filtered, which is the bias this whole store exists to avoid.
    """
    import json
    import urllib.request

    email = os.environ.get("SEC_EMAIL", "").strip()
    if not email:
        raise ValueError(
            "SEC_EMAIL is not set. SEC fair access requires a real contact "
            "address in the User-Agent header.")
    request = urllib.request.Request(
        _SEC_TICKERS_URL, headers={"User-Agent": f"Nemo_IB research {email}"})
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode())

    out = []
    for row in payload.values():
        ticker = str(row.get("ticker", "")).strip().upper()
        if ticker:
            out.append({"ticker": ticker,
                        "cik": str(row.get("cik_str", "")),
                        "name": row.get("title")})
    return out


# yfinance starts a thread per ticker, so the request size is bounded by the
# process thread limit rather than by anything about the data. The full SEC
# list raises "can't start new thread" outright; 200 is comfortably under it
# and still amortises the round trip.
FETCH_BATCH_SIZE = 200

# The first request against a cold yfinance session answers 401 "Invalid
# Crumb" and the next one works. Measured: the top 200 names returned 0, then
# 200 unretried. Against the full universe that lost every mega-cap, because
# the batches that fail are the ones that go out first.
FETCH_RETRIES = 3
FETCH_RETRY_BACKOFF = 5.0

# A rate-limited batch does not raise. yfinance catches per-ticker errors
# itself and returns an empty frame, so 200 unanswered names look exactly like
# 200 companies that did not trade -- and a batch of delisted shells, which the
# SEC list has thousands of, really is empty. Nothing in the response tells the
# two apart, so each batch carries a name that always trades. If it comes back,
# the batch reached the vendor and the absences are real.
FETCH_CANARY = "SPY"

# How far back the daily window reaches. A single-session request cannot tell a
# market holiday from a vendor that did not answer -- both return nothing for
# every ticker, canary included. With a week of context the canary settles it:
# bars on the neighbouring days and none on this one means the exchange was
# shut, no bars at all means the request failed. The extra sessions are never
# recorded; they exist to be compared against.
CLOSED_PROBE_DAYS = 7

# A short pause between batches. 52 of them arriving as fast as the process can
# issue them is what provokes the rate limiting in the first place, and half a
# second each costs a nightly job under half a minute.
FETCH_BATCH_PAUSE = 0.5


def _fetch_bars(tickers: Iterable[str], start: Optional[str] = None,
                end: Optional[str] = None,
                failures: Optional[List[str]] = None
                ) -> Dict[str, List[Dict[str, Any]]]:
    """Raw OHLCV per ticker, in batches, with the actions that shaped them.

    `auto_adjust=False` is load-bearing but not sufficient -- see
    `_to_as_traded` for what the vendor means by unadjusted. `actions=True`
    costs nothing and carries the splits that conversion needs.

    A batch that fails costs its own names and no others: on a universe this
    size an occasional 502 is routine, and each of those names loses one
    session out of the sixty the screen reads. What it must not do is go
    unmentioned, so anything lost is appended to `failures` and ends up in the
    run log. Every batch failing is a different thing and raises, because a
    market where no ticker returned data is not a market where nobody traded.
    """
    import yfinance as yf

    tickers = [t for t in tickers if t]
    if not tickers:
        return {}

    out: Dict[str, List[Dict[str, Any]]] = {}
    canary: List[Dict[str, Any]] = []
    lost: List[str] = []
    batches = [tickers[i:i + FETCH_BATCH_SIZE]
               for i in range(0, len(tickers), FETCH_BATCH_SIZE)]

    for index, batch in enumerate(batches):
        asked = batch if FETCH_CANARY in batch else [*batch, FETCH_CANARY]
        last: Optional[BaseException] = None
        for attempt in range(FETCH_RETRIES):
            try:
                frame = yf.download(tickers=" ".join(asked), start=start,
                                    end=end, auto_adjust=False, actions=True,
                                    group_by="ticker", progress=False,
                                    threads=True)
                rows = _rows_from_frame(frame, asked)
                if FETCH_CANARY not in rows:
                    raise RuntimeError(
                        f"batch returned no data for {FETCH_CANARY}, so it did "
                        f"not reach the vendor")
            except Exception as exc:  # noqa: BLE001 - retried, then raised
                last = exc
                if attempt + 1 < FETCH_RETRIES and FETCH_RETRY_BACKOFF:
                    time.sleep(FETCH_RETRY_BACKOFF * (attempt + 1))
                continue
            # Kept from the first batch that answers, not only from the batch
            # that happens to own it. The canary is appended to the end of the
            # ask, so the batch that owns it is a batch of one name -- the most
            # fragile request in the run, and losing it took the market-closed
            # check and the regime's own price series down with it.
            if not canary:
                canary = rows.get(FETCH_CANARY) or []
            if FETCH_CANARY not in batch:
                rows.pop(FETCH_CANARY, None)
            out.update(rows)
            break
        else:
            lost.append(
                f"batch {index + 1} of {len(batches)} failed after "
                f"{FETCH_RETRIES} attempts ({len(batch)} tickers, "
                f"{batch[0]}..{batch[-1]}): {type(last).__name__}: {last}")
        if FETCH_BATCH_PAUSE and index + 1 < len(batches):
            time.sleep(FETCH_BATCH_PAUSE)

    # A caller who asked for it by name gets it from whichever batch answered.
    if FETCH_CANARY in tickers and FETCH_CANARY not in out and canary:
        out[FETCH_CANARY] = canary
    if lost and not out:
        raise RuntimeError(f"every batch failed; first: {lost[0]}")
    if failures is not None:
        failures.extend(lost)
    return out


def _quantity(value: Any) -> float:
    """A count of something, or zero -- never NaN.

    The vendor leaves NaN in the action columns on ordinary sessions, and NaN
    is truthy: `NaN or 0.0` is NaN, which sails through the
    `if row.get("split")` guard in `_record_ticker` and then fails
    `record_corporate_action`'s
    `value > 0` check, taking the night down mid-list. `_to_as_traded` beside
    it already ignores the same value, so the two paths disagreed about one
    number and only one of them said so.
    """
    number = float(value or 0.0)
    return 0.0 if number != number else number


def _rows_from_frame(frame: Any, tickers: List[str]
                     ) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for ticker in tickers:
        # Not `if len(tickers) > 1`. yfinance returns a two-level column
        # index either way, so counting tickers reads a grouped frame as flat
        # and finds no "Close" at all.
        try:
            sub = frame[ticker] if getattr(frame.columns, "nlevels", 1) > 1 \
                else frame
        except (KeyError, TypeError):
            continue
        rows = []
        for stamp, row in sub.dropna(how="all").iterrows():
            if row.get("Close") is None or row.isna().get("Close", False):
                continue
            rows.append({
                "trade_date": str(stamp.date()),
                "open": float(row["Open"]), "high": float(row["High"]),
                "low": float(row["Low"]), "close": float(row["Close"]),
                "volume": _quantity(row.get("Volume")),
                "split": _quantity(row.get("Stock Splits")),
                "dividend": _quantity(row.get("Dividends")),
            })
        if rows:
            out[ticker] = rows
    return out


def _to_as_traded(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Put vendor prices back into the space the stock actually traded in.

    `auto_adjust=False` is a misleading name: the prices are split-adjusted to
    the moment of the request, and only dividends are left alone. NVDA's
    2024-06-07 close comes back 120.89 against a real print of 1208.88.

    Left alone, the store would hold two different price spaces with no marker
    between them -- bars recorded forward are genuinely as-traded, because a
    one-day window cannot contain a later split, while bootstrapped history is
    in whatever space the vendor was in on bootstrap day. A read-time
    adjustment would then be right on one side of that seam and out by the
    split ratio on the other.

    A bar is scaled by every split strictly after it: the ex-date session
    already prints the new price, so it belongs with what follows. The
    operation is exactly invertible, which is the point -- `adjusted_bars`
    re-applies the same factor and returns the vendor's own number, so an
    incomplete action list can never invent a price that did not exist.
    """
    factor = 1.0
    out: List[Dict[str, Any]] = []
    for row in reversed(rows):
        if factor != 1.0:
            row = dict(row)
            for field in ("open", "high", "low", "close"):
                if row.get(field) is not None:
                    row[field] = row[field] * factor
            if row.get("volume"):
                row["volume"] = row["volume"] / factor
        out.append(row)
        split = row.get("split") or 0.0
        if split > 0:
            factor *= split
    out.reverse()
    return out


def _record_ticker(ticker: str, rows: List[Dict[str, Any]], stamp: str,
                   keep=None, source: str = "recorded") -> int:
    """Actions first, then the bars they explain, in one transaction.

    Order matters on a cold store: a reader that finds bars without the split
    that shaped them has no way to tell an as-traded series from an adjusted
    one, and this is the only moment both are in hand.

    One connection for both because that order is only worth anything if it
    cannot be interrupted. Committed separately, a crash in between leaves
    exactly the state the paragraph above says must never exist -- and on a
    volume six jobs write to, the interruption is a lock, not a power cut.
    """
    with pit_store.connect() as conn:
        for row in rows:
            # Never earlier than the action's own ex-date. On a backfill the
            # window runs past `as_of` to catch intervening splits, and
            # stamping those at `as_of` would claim a 2025 split was known in
            # 2024. For a bootstrap every ex-date is at or before the stamp, so
            # this changes nothing.
            when = max(stamp, f"{row['trade_date']}T21:00:00Z")
            if row.get("split"):
                pit_store.record_corporate_action(
                    ticker, row["trade_date"], "split", float(row["split"]),
                    recorded_at=when, conn=conn)
            if row.get("dividend"):
                pit_store.record_corporate_action(
                    ticker, row["trade_date"], "dividend",
                    float(row["dividend"]), recorded_at=when, conn=conn)
        as_traded = _to_as_traded(rows)
        if keep is not None:
            # A past-dated run fetches beyond `as_of` only so the conversion
            # above can see the splits in between. Those later sessions belong
            # to their own days, with their own stamps, and writing them under
            # this one would both misdate them and hand the reader a session
            # that had not happened yet.
            as_traded = [r for r in as_traded if keep(r["trade_date"])]
        return pit_store.record_bars(ticker, as_traded, recorded_at=stamp,
                                     conn=conn, source=source)


def _fetch_calendar(start: str, end: str) -> List[Dict[str, Any]]:
    """Who reports in a window, with the estimate as it stands today and the
    actual once it lands.

    Straight at the endpoint, deliberately, rather than through the MCP tool.
    That tool condenses for an LLM's context budget: it caps the event list at
    15 and drops the fields a recorder needs. On a single ordinary week that
    cap hid 273 of 288 reporters. A recorder has no context budget to protect
    and every name it drops is a hole no later run can fill, so the two
    callers want genuinely different things from the same endpoint.

    One call for the whole window rather than one per name: at several thousand
    tickers a per-name calendar would exhaust the rate limit long before it
    finished, and the point of this job is that it completes every day.
    """
    import asyncio

    from tools.news_agregator.finnhub_utils import FinnhubClient

    async def pull():
        client = FinnhubClient()
        try:
            return await client.get("/calendar/earnings",
                                    {"from": start, "to": end})
        finally:
            close = getattr(client, "close", None)
            if close is not None:
                await close()

    payload = asyncio.run(pull())

    # Fail loud. An error dict here used to become an empty calendar, and an
    # empty calendar is indistinguishable from a quiet week -- which is how
    # this seam ran green for its whole life while recording nothing.
    if not isinstance(payload, dict):
        raise RuntimeError(f"earnings calendar: unexpected payload "
                           f"{type(payload).__name__}")
    if payload.get("error"):
        raise RuntimeError(f"earnings calendar: {payload['error']}")
    if "earningsCalendar" not in payload:
        raise RuntimeError(
            f"earnings calendar: no 'earningsCalendar' key; got "
            f"{sorted(payload)[:8]}")

    out = []
    for row in payload["earningsCalendar"] or []:
        ticker = str(row.get("symbol", "")).strip().upper()
        year, quarter = row.get("year"), row.get("quarter")
        # No fiscal identity, no row. Keying a snapshot on None would have
        # every such company collide on one key and overwrite each other.
        if not ticker or not year or not quarter:
            continue
        out.append({
            "ticker": ticker,
            "fiscal_period": f"{year}Q{quarter}",
            "eps_estimate": row.get("epsEstimate"),
            # Only present once the company has reported, which is why the
            # window looks backwards as well as forwards.
            "eps_actual": row.get("epsActual"),
            "analyst_count": row.get("numberOfAnalysts"),
            "timing": row.get("hour") or "unknown",
            "date": row.get("date"),
        })
    return out


# --------------------------------------------------------- run bookkeeping

def last_run(job: str) -> Optional[Dict[str, Any]]:
    with pit_store.connect() as conn:
        row = conn.execute(
            "SELECT * FROM run_log WHERE job = ? ORDER BY run_id DESC LIMIT 1",
            (job,)).fetchone()
    return dict(row) if row else None


def coverage_status(covered: int, requested: int) -> str:
    """Nothing is a failure, everything is ok, and the middle says so.

    Reported separately from `rows_written` on purpose: a run that reached one
    ticker out of five thousand wrote rows and is not a working day, and only
    the ratio shows that.

    What `requested` means is the caller's problem and it matters. A watcher
    that looks up one folder per name is asked for names it can reach, so its
    own count is the right denominator. The bar recorder is not: its ask
    carries a rotating slice of the SEC registrant list, thousands of delisted
    shells that will never answer again, and measured against that no night is
    ever complete. See `_bars_status`.
    """
    if requested == 0:
        # Asked for nothing and got nothing. That is a complete run over an
        # empty list, not an upstream that refused.
        return "ok"
    if covered == 0:
        return "failed"
    return "ok" if covered >= requested else "partial"


def _bars_status(covered: int, answered: int, requested: int,
                 probe_answered: bool, failures: List[str]) -> str:
    """A bar run's verdict, measured against what could be reached.

    `covered >= requested` was the old rule and `requested` is the nightly ask,
    which deliberately contains names expected to be empty. So every weekday
    logged `partial` with "1847 of 3000 tickers returned data" in the error
    column, every weekday landed in `missing_days`, and the store's own gap
    detector could no longer point at the one night that really was missed.

    The three facts that do say whether a night worked:

      The probe answered. Each batch carries a name that always trades, so a
      run with no canary anywhere in the window did not reach the vendor and
      nothing about its absences means anything.

      No batch was lost. Every name in a lost batch is simply absent from the
      response, so a ratio computed over what came back cannot see it -- two
      hundred names lost to a 502 would read as two hundred companies that do
      not trade.

      Every name the vendor answered for in the window has its session. A
      shell that answers nowhere is not a hole in the record; a name that
      printed on Monday and is missing on Tuesday is.
    """
    if requested == 0:
        # Asked for nothing, so there was nothing to reach. A run over an empty
        # list is complete, and the probe has no verdict to give about it.
        return "ok"
    if not probe_answered:
        return "failed"
    status = coverage_status(covered, answered)
    return "partial" if failures and status == "ok" else status


# ------------------------------------------------------------------ bars

def record_daily_bars(tickers: List[str],
                      as_of: Optional[str] = None) -> Dict[str, Any]:
    """One session for the universe. Idempotent within a day."""
    as_of = as_of or _today()
    pit_store.start_run("daily_bars", as_of_date=as_of)

    # Through today, not through tomorrow, whenever `as_of` is in the past.
    # The vendor has already divided that session's price by every split since,
    # and a one-day window carries no split column to undo it with -- backfill
    # 2024-06-07 after NVDA's 10:1 and a 1208.88 print lands in the store as
    # 120.89, in a table whose whole premise is that its numbers are what the
    # stock actually printed. Only the requested session is recorded; the rest
    # of the window is there to be divided by.
    today = _today()
    end = (date.fromisoformat(max(as_of, today))
           + timedelta(days=1)).isoformat()
    start = (date.fromisoformat(as_of)
             - timedelta(days=CLOSED_PROBE_DAYS)).isoformat()

    # Asked for alongside the universe so its own sessions come back. Excluded
    # from what gets written unless the caller wanted it in its own right.
    probe = FETCH_CANARY not in tickers
    asked = [*tickers, FETCH_CANARY] if probe else list(tickers)

    failures: List[str] = []
    try:
        fetched = _fetch_bars(asked, start=start, end=end,
                              failures=failures)
    except Exception as exc:  # noqa: BLE001 - reported, never masked
        pit_store.finish_run(rows_written=0, status="failed",
                             error=f"{type(exc).__name__}: {exc}")
        return {"status": "failed", "error": str(exc), "written": 0}

    # The exchange, before the tickers. If the index traded during the window
    # but not on this date, the market was shut -- which is a complete run over
    # nothing, not a failure, and must not sit in missing_days for years.
    index_rows = fetched.get(FETCH_CANARY) or []
    if index_rows and not any(r["trade_date"] == as_of for r in index_rows):
        last_session = max(r["trade_date"] for r in index_rows)
        if _session_is_pending(as_of):
            # Not a holiday: a session that has not happened yet. Read as a
            # closed exchange it records nothing, exits 0 and counts the night
            # as covered -- and on a US-timezone host that is every night, for
            # as long as nobody reads the run log.
            detail = (f"this run is dated {as_of} but that session has not "
                      f"closed yet; {FETCH_CANARY}'s last is {last_session}. "
                      f"The date came from a UTC clock and the schedule is "
                      f"host-local")
            pit_store.finish_run(rows_written=0, status="failed", error=detail)
            return {"status": "failed", "error": detail, "written": 0,
                    "covered": 0, "requested": len(tickers),
                    "failures": failures}
        pit_store.finish_run(
            rows_written=0, status="closed",
            error=f"{FETCH_CANARY} has no session on {as_of}; the exchange "
                  f"was closed")
        return {"status": "closed", "written": 0, "covered": 0,
                "requested": len(tickers), "failures": failures}

    written = 0
    covered = 0
    # Names the vendor answered for somewhere in the window: it knows them and
    # they still trade. This is the denominator -- not the ask, which carries
    # the rotation slice of the registrant list and is thousands of names
    # expected to be empty.
    answered = 0
    source = _source(as_of)
    for ticker in tickers:
        rows = fetched.get(ticker) or []
        if not rows:
            # No bar at all. A vendor that did not answer for a ticker is not
            # a session in which nobody traded, and writing a zero-volume row
            # here would make the two indistinguishable forever.
            continue
        answered += 1
        if not any(r["trade_date"] == as_of for r in rows):
            # Present in the window but absent on the day. The window is wider
            # than the session being recorded, so this is the same absence as
            # above rather than coverage.
            continue
        # One ticker's write is one ticker's problem. A locked database or a
        # value the recorder refuses used to propagate out of here, which
        # abandoned every name after it in the list, left the run log open,
        # and cost the universe screen and the consensus snapshot that had not
        # run yet -- and consensus is the one series no later run can refetch.
        try:
            written += _record_ticker(ticker, rows, _stamp(as_of),
                                      keep=lambda d, _o=as_of: d == _o,
                                      source=source)
        except Exception as exc:  # noqa: BLE001 - reported, never masked
            failures.append(f"{ticker}: {type(exc).__name__}: {exc}")
            continue
        # After the write, not before it: a session nothing recorded is not
        # coverage, and counting it here is what would let a night of failed
        # writes report itself complete.
        covered += 1

    try:
        _record_probe(fetched, _stamp(as_of),
                      lambda d, _o=as_of: d == _o, list(tickers), source)
    except Exception as exc:  # noqa: BLE001 - reported, never masked
        failures.append(f"{FETCH_CANARY}: {type(exc).__name__}: {exc}")

    status = _bars_status(covered, answered, len(tickers), bool(index_rows),
                          failures)
    detail = (f"{covered} of {answered} reachable tickers recorded, "
              f"{len(tickers)} asked")
    if tickers and not index_rows:
        detail = (f"{FETCH_CANARY} returned no session anywhere in the window, "
                  f"so the vendor did not answer and no absence here means "
                  f"anything; {detail}")
    if failures:
        detail = f"{detail}; {len(failures)} lost: " \
                 + " | ".join(failures[:3])
    pit_store.finish_run(rows_written=written, status=status,
                         error=None if status == "ok" and not failures
                         else detail)
    return {"status": status, "written": written, "covered": covered,
            "answered": answered, "requested": len(tickers),
            "source": source, "failures": failures}


def bootstrap_history(tickers: List[str], lookback_days: int = 730,
                      as_of: Optional[str] = None,
                      job: str = "bootstrap_history") -> Dict[str, Any]:
    """Pull history once so the screen has something to stand on.

    Stamped with today's date, not the sessions' own. That is the whole point:
    this history was learned today, and a simulation standing in 2024 must not
    see it. Useful for computing today's screen; correctly invisible to any
    past-dated question.

    `job` is the run-log label. Cold-store bootstrapping is not the only caller
    any more -- the nightly newcomer pass runs the same fetch over the rotation
    slice -- and a nightly row filed under `bootstrap_history` would say a
    one-off migration happens every night.
    """
    as_of = as_of or _today()
    start = (date.fromisoformat(as_of) - timedelta(days=lookback_days)).isoformat()
    pit_store.start_run(job, as_of_date=as_of)

    # Through today when `as_of` is in the past, for the same reason the daily
    # path does it: the vendor has divided this whole history by every split
    # since, and only the sessions in the window carry the split column that
    # undoes it. Everything past `as_of` is dropped before writing.
    #
    # The extra day is because the vendor's `end` is exclusive. Without it a
    # run stops one session short of the day it is for, which was invisible
    # while a nightly pass always followed for the same names -- and is a
    # permanent hole now that a newcomer's first night IS this call.
    end = (date.fromisoformat(max(as_of, _today()))
           + timedelta(days=1)).isoformat()

    # Asked for by name, so it survives the canary strip. The nightly path
    # already does this; without it here the index's HISTORY is never written,
    # and the regime needs 252 sessions of it -- which is why every scan
    # reported "unknown" at full book size after a complete bootstrap.
    asked = list(tickers) if FETCH_CANARY in tickers \
        else [*tickers, FETCH_CANARY]

    failures: List[str] = []
    try:
        fetched = _fetch_bars(asked, start=start, end=end, failures=failures)
    except Exception as exc:  # noqa: BLE001
        pit_store.finish_run(rows_written=0, status="failed",
                             error=f"{type(exc).__name__}: {exc}")
        return {"status": "failed", "error": str(exc), "written": 0}

    written = 0
    covered = 0
    # As in the nightly pass: the names the vendor answered for, not the names
    # asked for. This runs over the rotation slice every night now, and that
    # slice is mostly registrants that have not traded in years.
    answered = 0
    source = _source(as_of)
    for ticker in tickers:
        rows = fetched.get(ticker) or []
        if not rows:
            continue
        answered += 1
        # Guarded for the same reason the nightly loop is, and more so: a
        # two-year window is where the splits actually are, so this is the loop
        # a refused action value stops first.
        try:
            written += _record_ticker(ticker, rows, _stamp(as_of),
                                      keep=lambda d, _o=as_of: d <= _o,
                                      source=source)
        except Exception as exc:  # noqa: BLE001 - reported, never masked
            failures.append(f"{ticker}: {type(exc).__name__}: {exc}")
            continue
        covered += 1

    try:
        _record_probe(fetched, _stamp(as_of),
                      lambda d, _o=as_of: d <= _o, list(tickers), source)
    except Exception as exc:  # noqa: BLE001 - reported, never masked
        failures.append(f"{FETCH_CANARY}: {type(exc).__name__}: {exc}")

    status = _bars_status(covered, answered, len(tickers),
                          bool(fetched.get(FETCH_CANARY)), failures)
    detail = (f"{covered} of {answered} reachable tickers returned history, "
              f"{len(tickers)} asked")
    if failures:
        detail = f"{detail}; {len(failures)} lost: " + " | ".join(failures[:3])
    pit_store.finish_run(rows_written=written, status=status,
                         error=None if status == "ok" and not failures
                         else detail)
    return {"status": status, "written": written, "covered": covered,
            "answered": answered, "requested": len(tickers), "source": source,
            "failures": failures}


# -------------------------------------------------------------- universe

def _screen(ticker: str, as_of: str) -> Dict[str, Any]:
    """Eligible, or the reason not -- never a bare exclusion.

    "No history" and "too thin" are different facts about a name, and a screen
    that returns the same answer for both cannot be audited later.
    """
    bars = pit_store.bars_as_of(ticker, as_of)
    if len(bars) < MIN_HISTORY_SESSIONS:
        return {"eligible": False,
                "exclusion_reason": (
                    f"insufficient history: {len(bars)} sessions recorded, "
                    f"{MIN_HISTORY_SESSIONS} required")}

    window = bars[-SCREEN_LOOKBACK_SESSIONS:]
    closes = [b["close"] for b in window if b["close"] is not None]
    dollar = [b["close"] * b["volume"] for b in window
              if b["close"] is not None and b["volume"] is not None]
    if not closes or not dollar:
        return {"eligible": False,
                "exclusion_reason": "recorded sessions carry no price or volume"}

    median_dollar = statistics.median(dollar)
    last_price = closes[-1]

    if last_price < MIN_PRICE:
        return {"eligible": False,
                "exclusion_reason": (
                    f"price {last_price:.2f} below the {MIN_PRICE:.2f} floor; "
                    f"tick size is a large fraction of a percent here")}
    if median_dollar < MIN_MEDIAN_DOLLAR_VOLUME:
        return {"eligible": False,
                "exclusion_reason": (
                    f"median dollar volume {median_dollar:,.0f} below the "
                    f"{MIN_MEDIAN_DOLLAR_VOLUME:,.0f} floor")}
    return {"eligible": True, "exclusion_reason": None,
            "median_dollar_volume": median_dollar}


def refresh_universe(as_of: Optional[str] = None) -> Dict[str, Any]:
    """Today's membership, eligible and rejected alike.

    Rejections are recorded rather than dropped. A name excluded for liquidity
    this quarter may qualify next, and a record that only ever kept the
    survivors cannot distinguish a screen that worked from one never applied.
    """
    as_of = as_of or _today()
    pit_store.start_run("universe", as_of_date=as_of)

    try:
        registrants = _fetch_sec_tickers()
    except Exception as exc:  # noqa: BLE001
        pit_store.finish_run(rows_written=0, status="failed",
                             error=f"{type(exc).__name__}: {exc}")
        return {"status": "failed", "error": str(exc)}

    entries = []
    eligible = 0
    for reg in registrants:
        verdict = _screen(reg["ticker"], as_of)
        eligible += 1 if verdict["eligible"] else 0
        entries.append({**reg, **verdict})

    # End of the day it screened, like every other recorder here. Stamped
    # `now`, a screen run for a past date is visible to that date -- and the
    # universe is what the scanner draws candidates from.
    written = pit_store.record_universe(as_of, entries,
                                        recorded_at=_stamp(as_of))
    status = "ok" if registrants else "failed"
    pit_store.finish_run(rows_written=written, status=status,
                         error=None if status == "ok" else "no registrants")
    return {"status": status, "written": written, "eligible": eligible,
            "screened": len(entries)}


def eligible_tickers(as_of: Optional[str] = None) -> List[str]:
    as_of = as_of or _today()
    return [m["ticker"] for m in pit_store.universe_as_of(as_of)
            if m["eligible"]]


# ------------------------------------------------------------- consensus

def record_consensus_snapshots(as_of: Optional[str] = None,
                               horizon_days: int = 10,
                               lookback_days: int = 5) -> Dict[str, Any]:
    """Today's view of what the street expects, for names about to report.

    The one series that cannot be fetched retroactively. Finnhub returns four
    quarters at limit=12 and at limit=30 -- verified -- so this history only
    ever accrues forward, which is why a day missed here is a day lost.

    Scoped to the horizon rather than the whole universe because the estimate
    only matters near the print, and a full sweep would exhaust the rate limit
    before finishing.
    """
    as_of = as_of or _today()
    # Backwards as well as forwards. The estimate is visible before the print
    # and the actual only after it, so a forward-only window would snapshot
    # what the street expected every single day and never once what happened.
    start = (date.fromisoformat(as_of) - timedelta(days=lookback_days)).isoformat()
    end = (date.fromisoformat(as_of) + timedelta(days=horizon_days)).isoformat()
    pit_store.start_run("consensus", as_of_date=as_of)

    try:
        rows = _fetch_calendar(start, end)
    except Exception as exc:  # noqa: BLE001
        pit_store.finish_run(rows_written=0, status="failed",
                             error=f"{type(exc).__name__}: {exc}")
        return {"status": "failed", "error": str(exc), "written": 0}

    written = 0
    for row in rows:
        if not row.get("fiscal_period"):
            # Without a fiscal identity there is nothing to join this to
            # later; the vendor's calendar bucket is not one.
            continue
        # The recorder's own count, not one per row seen. `rows_written` is
        # the only evidence a night was captured, and consensus is the series
        # where that matters most -- a replay that changed nothing must not
        # read like a capture.
        written += pit_store.record_consensus(
            as_of, row["ticker"], row["fiscal_period"],
            eps_estimate=row.get("eps_estimate"),
            eps_actual=row.get("eps_actual"),
            analyst_count=row.get("analyst_count"),
            # End of the day it describes, like every other recorder here.
            # Stamped `now`, a consensus run for a past date is invisible to
            # that date and visible to every date after it.
            recorded_at=_stamp(as_of))

        # The announcement itself, once it has happened. `timing` is what makes
        # the table worth having: bmo or amc decides which session is the
        # reaction, and AMAT's 13 August print after the close belongs to the
        # 14th -- dating it to the announcement gives -2.48% instead of -6.57%.
        # Written only with an actual beside it, because a company on next
        # week's calendar has not announced, and a reaction date recorded
        # before the reaction is a date that means nothing.
        if row.get("eps_actual") is not None and row.get("date"):
            pit_store.record_announcement(
                row["ticker"], row["fiscal_period"], row["date"],
                timing=row.get("timing") or "unknown",
                recorded_at=_stamp(as_of))

    status = "ok" if rows else "failed"
    pit_store.finish_run(rows_written=written, status=status,
                         error=None if rows else "calendar returned no rows")
    return {"status": status, "written": written, "seen": len(rows)}

# Yahoo answers a 10,388-name night with YFRateLimitError, and a universe that
# only ever fetches its existing members can never admit a new listing: no
# history means not eligible, not eligible means never fetched, never fetched
# means no history. So the nightly ask is every eligible name plus a rotating
# slice of the rest, which bounds the request and still walks the whole
# registrant list inside a cycle.
MAX_NIGHTLY_TICKERS = 3000

# And a floor under the rotation. `MAX_NIGHTLY_TICKERS - len(eligible)` reaches
# zero the moment the eligible set fills the cap, and the ask then contains
# nothing but existing members -- so a new listing is never fetched, never
# gains history and never becomes eligible. That is the same starvation the
# rotation exists to prevent, arriving a few weeks in rather than on day one,
# and the real eligible universe is a few thousand names so it arrives in
# ordinary operation. The cap is a target; this is the guarantee.
MIN_NEWCOMER_SLOTS = 500

# How far back a newcomer's first fetch reaches. The screen counts recorded
# sessions, so a name fetched one session per rotation needs
# MIN_HISTORY_SESSIONS whole cycles before it can be screened at all -- at two
# thousand eligible names the cycle is nine nights, which is 540 weekdays, and
# at two and a half thousand it is nearly four years. The universe was frozen
# and the backtest inherited the survivorship bias through the ask policy
# rather than through the vendor.
#
# Sixty sessions is about eighty-four calendar days; the rest is holidays and
# the margin that keeps a thin month from landing one session short.
NEWCOMER_HISTORY_DAYS = 120


def _nightly_split(as_of: str,
                   registrants: List[str]) -> tuple[List[str], List[str]]:
    """Tonight's ask, in its two halves: the members, then the newcomers.

    Eligible names are never rotated out. They are what a signal actually
    reads, and a hole in one of those series is a hole where it costs most.
    The slice is keyed on the date so it is deterministic -- a rerun of a given
    night asks for exactly what that night asked for, which is the same
    property the rest of the store is built on.

    They are returned apart because they want different requests. A member
    needs one session appended to a series it already has; a newcomer needs
    enough history to be screened, and asking for it one session at a time is
    what froze the universe.
    """
    eligible = eligible_tickers(as_of)
    keep = list(dict.fromkeys(eligible))
    known = set(keep)
    rest = [t for t in registrants if t not in known]

    room = max(MIN_NEWCOMER_SLOTS, MAX_NIGHTLY_TICKERS - len(keep))
    if not rest:
        return keep, []

    # Ordinal of the date, so consecutive nights take consecutive slices and
    # the cycle length is len(rest) / room nights.
    offset = (date.fromisoformat(as_of).toordinal() * room) % len(rest)
    slice_ = rest[offset:offset + room]
    if len(slice_) < room:
        slice_ += rest[:room - len(slice_)]
    return keep, slice_


def nightly_tickers(as_of: str, registrants: List[str]) -> List[str]:
    """Everyone eligible, plus this night's slice of everyone else."""
    keep, newcomers = _nightly_split(as_of, registrants)
    return keep + newcomers


def run_all(as_of: Optional[str] = None,
            bootstrap: bool = False) -> Dict[str, Any]:
    """A day's pass. Each stage logs separately so a partial day reads as one.

    Order matters: the universe screen reads bars, so bars are recorded first.
    On a cold store nothing is eligible until `bootstrap=True` has run once.
    """
    as_of = as_of or _today()
    results: Dict[str, Any] = {"as_of": as_of}

    try:
        registrants = [r["ticker"] for r in _fetch_sec_tickers()]
    except Exception as exc:  # noqa: BLE001
        return {**results, "error": f"universe unavailable: {exc}"}

    keep, newcomers = _nightly_split(as_of, registrants)
    results["asked"] = len(keep) + len(newcomers)

    stages: List[Any] = []
    if bootstrap:
        stages.append(("bootstrap",
                       lambda: bootstrap_history(keep + newcomers,
                                                 as_of=as_of)))
    stages.append(("daily_bars", lambda: record_daily_bars(keep, as_of=as_of)))
    # The rotation slice, with enough history to be screened tonight rather
    # than in two years. Skipped under `bootstrap`, which has already asked for
    # the same names over a far longer window -- asking twice is a doubled
    # request against the rate limit that `MAX_NIGHTLY_TICKERS` exists for.
    if newcomers and not bootstrap:
        stages.append(("newcomers",
                       lambda: bootstrap_history(
                           newcomers, lookback_days=NEWCOMER_HISTORY_DAYS,
                           as_of=as_of, job="newcomers")))
    stages += [("universe", lambda: refresh_universe(as_of=as_of)),
               ("consensus", lambda: record_consensus_snapshots(as_of=as_of))]

    # In order, but not on each other's success. Called as bare statements, an
    # exception anywhere in the first stage ended the process, and the stage
    # that never ran was consensus -- the one series that cannot be fetched
    # again later. A stage that raises is reported as a failed stage, which is
    # what `main` turns into a non-zero exit; its run-log row stays unfinished,
    # which `missing_days` already reads as the night it was.
    for name, stage in stages:
        try:
            results[name] = stage()
        except Exception as exc:  # noqa: BLE001 - reported, never masked
            results[name] = {"status": "failed",
                             "error": f"{type(exc).__name__}: {exc}"}

    # Reported after the stages, so tonight counts as tonight. A stage that
    # failed leaves this night in the list, which is the point.
    results["gaps"] = coverage_gaps(as_of=as_of)
    return results


# ------------------------------------------------------- gaps, and filling them
#
# `missing_days` has been in the store since the first commit and nothing ever
# called it: it appeared in its own definition and in two comments. A gap
# detector nothing reads is not a detector, and the nights it would have named
# were the ones an operator most needed told about.

# How far back the nightly report looks. Long enough to cover a long weekend
# and the outage that started over it, short enough that a store down for a
# quarter says so once rather than growing a longer list every night.
GAP_REPORT_DAYS = 30


def coverage_gaps(as_of: Optional[str] = None, days: int = GAP_REPORT_DAYS,
                  job: str = "daily_bars") -> List[str]:
    """Weekdays in the trailing window that no finished run covers.

    Clamped to the first date the job ever ran for. Before that the recorder
    did not exist, so a fresh volume would report its first month as
    twenty-two missed nights -- the same saturation that made the status column
    worth nothing, arriving through the report instead.
    """
    as_of = as_of or _today()
    first = pit_store.first_run(job)
    if first is None:
        return []
    start = (date.fromisoformat(as_of) - timedelta(days=days)).isoformat()
    return pit_store.missing_days(job, max(start, first), as_of)


def backfill(since: Optional[str] = None, until: Optional[str] = None,
             job: str = "daily_bars") -> Dict[str, Any]:
    """Replay every weekday in the window that no finished run covers.

    There was no backfill mechanism at all: after a multi-day outage an
    operator hand-ran `--as-of` for each date and nothing enumerated which
    dates those were. This is that enumeration, and it is the same list the
    nightly run now reports.

    Never today. Today is the nightly job's own date, and replaying it here
    would either duplicate the run about to happen or record a session that has
    not closed yet.

    What it writes is genuinely worse evidence than the night it stands in for
    -- the vendor drops delisted tickers, so filling 1 August on 26 August is
    missing exactly the names the store exists to preserve -- which is why
    those rows are written as `backfilled` and say so.
    """
    # Clamped rather than defaulted, so an `until` of today or later cannot
    # smuggle tonight's own date into a list of nights to replay.
    yesterday = (date.fromisoformat(_today()) - timedelta(days=1)).isoformat()
    until = min(until or yesterday, yesterday)
    if since is None:
        first = pit_store.first_run(job)
        if first is None:
            # No run log is not a record with holes in it; it is no record.
            # Filling a month of history unasked is a different operation, and
            # `since` is how someone asks for it.
            return {"gaps": [], "days": {},
                    "note": f"{job} has never run, so it has no gaps -- only "
                            f"an empty record. Pass --since to fill a window "
                            f"anyway"}
        window = (date.fromisoformat(until)
                  - timedelta(days=GAP_REPORT_DAYS)).isoformat()
        since = max(first, window)

    gaps = pit_store.missing_days(job, since, until)
    days: Dict[str, Any] = {}
    for day in gaps:
        days[day] = run_all(as_of=day)
    return {"gaps": gaps, "days": days, "since": since, "until": until}


# ------------------------------------------------------------- entry point

def _any_stage_failed(result: Dict[str, Any]) -> bool:
    """Whether a `run_all` result contains a stage that failed."""
    if result.get("error"):
        return True
    return any(v["status"] == "failed" for v in result.values()
               if isinstance(v, dict) and "status" in v)


def main(argv: Optional[List[str]] = None) -> int:
    """`python -m research.daily_job`, which is what the schedule invokes.

    Returns an exit code rather than printing and hoping. A partial night is
    normal on a universe this size and exits 0; a stage that failed exits 1,
    because the only way a scheduler ever learns about it is the exit code, and
    a hole nobody was told about is the one that later reads as data.
    """
    import argparse
    import json

    parser = argparse.ArgumentParser(
        prog="daily_job",
        description="Record one day into the point-in-time store.")
    parser.add_argument("--as-of", dest="as_of", default=None,
                        help="date to record as (default: today); with "
                             "--backfill, the last date the gap list covers")
    parser.add_argument("--bootstrap", action="store_true",
                        help="pull history first; needed once on a cold store")
    parser.add_argument("--backfill", action="store_true",
                        help="replay every weekday with no finished run "
                             "instead of recording tonight")
    parser.add_argument("--since", default=None,
                        help="earliest date --backfill will consider "
                             "(default: the trailing window, never earlier "
                             "than the first night the recorder ran)")
    args = parser.parse_args(argv)

    # Nothing else does. The container entry point every batch service
    # overrides initialises a different database, so the first run on a fresh
    # volume died on "no such table: run_log" before it fetched anything --
    # while the suite stayed green, because every test builds the schema in a
    # fixture. Cheap and idempotent, so it runs every night rather than once.
    pit_store.init_schema()

    if args.backfill:
        filled = backfill(since=args.since, until=args.as_of)
        print(json.dumps(filled, indent=2, default=str))
        # One bad night in the list is a night still missing, and the operator
        # who ran this is the one person who will act on knowing that.
        return 1 if any(_any_stage_failed(day)
                        for day in filled["days"].values()) else 0

    kwargs: Dict[str, Any] = {"as_of": args.as_of}
    if args.bootstrap:
        kwargs["bootstrap"] = True
    result = run_all(**kwargs)
    print(json.dumps(result, indent=2, default=str))

    return 1 if _any_stage_failed(result) else 0


if __name__ == "__main__":  # pragma: no cover - exercised via main()
    raise SystemExit(main())
