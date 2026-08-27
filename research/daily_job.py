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
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from dotenv import load_dotenv

from research import pit_store

# Cron has no shell profile. Every other entry point here inherits credentials
# from the shell a human ran it in; a nightly job inherits nothing, so it reads
# the same repo-root .env the container mounts.
_DOTENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_DOTENV_PATH)

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


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _stamp(as_of: str) -> str:
    """When a run for `as_of` learned what it learned: the end of that day.

    Not `datetime.now()`. The two agree in production, where as_of is today,
    and differ exactly when a run is replayed for a past date -- where using
    now() would stamp old sessions as freshly learned and quietly grant every
    later query the lookahead the store exists to prevent. Reads compare at
    date granularity, so the time component is a formality.
    """
    return f"{as_of}T21:00:00Z"


# --------------------------------------------------------------- seams

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

    if lost and not out:
        raise RuntimeError(f"every batch failed; first: {lost[0]}")
    if failures is not None:
        failures.extend(lost)
    return out


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
                "volume": float(row.get("Volume") or 0.0),
                "split": float(row.get("Stock Splits") or 0.0),
                "dividend": float(row.get("Dividends") or 0.0),
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


def _record_ticker(ticker: str, rows: List[Dict[str, Any]], stamp: str) -> int:
    """Actions first, then the bars they explain.

    Order matters on a cold store: a reader that finds bars without the split
    that shaped them has no way to tell an as-traded series from an adjusted
    one, and this is the only moment both are in hand.
    """
    for row in rows:
        if row.get("split"):
            pit_store.record_corporate_action(
                ticker, row["trade_date"], "split", float(row["split"]),
                recorded_at=stamp)
        if row.get("dividend"):
            pit_store.record_corporate_action(
                ticker, row["trade_date"], "dividend", float(row["dividend"]),
                recorded_at=stamp)
    return pit_store.record_bars(ticker, _to_as_traded(rows), recorded_at=stamp)


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
    """
    if covered == 0:
        return "failed"
    return "ok" if covered >= requested else "partial"


# ------------------------------------------------------------------ bars

def record_daily_bars(tickers: List[str],
                      as_of: Optional[str] = None) -> Dict[str, Any]:
    """One session for the universe. Idempotent within a day."""
    as_of = as_of or _today()
    pit_store.start_run("daily_bars", as_of_date=as_of)

    failures: List[str] = []
    try:
        fetched = _fetch_bars(tickers, start=as_of,
                              end=(date.fromisoformat(as_of)
                                   + timedelta(days=1)).isoformat(),
                              failures=failures)
    except Exception as exc:  # noqa: BLE001 - reported, never masked
        pit_store.finish_run(rows_written=0, status="failed",
                             error=f"{type(exc).__name__}: {exc}")
        return {"status": "failed", "error": str(exc), "written": 0}

    written = 0
    covered = 0
    for ticker in tickers:
        rows = fetched.get(ticker) or []
        if not rows:
            # No bar at all. A vendor that did not answer for a ticker is not
            # a session in which nobody traded, and writing a zero-volume row
            # here would make the two indistinguishable forever.
            continue
        covered += 1
        written += _record_ticker(ticker, rows, _stamp(as_of))

    status = coverage_status(covered, len(tickers))
    detail = f"{covered} of {len(tickers)} tickers returned data"
    if failures:
        detail = f"{detail}; {len(failures)} batch(es) lost: " \
                 + " | ".join(failures[:3])
    pit_store.finish_run(rows_written=written, status=status,
                         error=None if status == "ok" and not failures
                         else detail)
    return {"status": status, "written": written, "covered": covered,
            "requested": len(tickers), "failures": failures}


def bootstrap_history(tickers: List[str], lookback_days: int = 730,
                      as_of: Optional[str] = None) -> Dict[str, Any]:
    """Pull history once so the screen has something to stand on.

    Stamped with today's date, not the sessions' own. That is the whole point:
    this history was learned today, and a simulation standing in 2024 must not
    see it. Useful for computing today's screen; correctly invisible to any
    past-dated question.
    """
    as_of = as_of or _today()
    start = (date.fromisoformat(as_of) - timedelta(days=lookback_days)).isoformat()
    pit_store.start_run("bootstrap_history", as_of_date=as_of)

    try:
        fetched = _fetch_bars(tickers, start=start, end=as_of)
    except Exception as exc:  # noqa: BLE001
        pit_store.finish_run(rows_written=0, status="failed",
                             error=f"{type(exc).__name__}: {exc}")
        return {"status": "failed", "error": str(exc), "written": 0}

    written = 0
    covered = 0
    for ticker in tickers:
        rows = fetched.get(ticker) or []
        if not rows:
            continue
        covered += 1
        written += _record_ticker(ticker, rows, _stamp(as_of))

    status = coverage_status(covered, len(tickers))
    pit_store.finish_run(rows_written=written, status=status,
                         error=None if status == "ok"
                         else f"{covered} of {len(tickers)} returned history")
    return {"status": status, "written": written, "covered": covered}


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

    written = pit_store.record_universe(as_of, entries)
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
        pit_store.record_consensus(
            as_of, row["ticker"], row["fiscal_period"],
            eps_estimate=row.get("eps_estimate"),
            eps_actual=row.get("eps_actual"),
            analyst_count=row.get("analyst_count"))
        written += 1

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


def nightly_tickers(as_of: str, registrants: List[str]) -> List[str]:
    """Everyone eligible, plus this night's slice of everyone else.

    Eligible names are never rotated out. They are what a signal actually
    reads, and a hole in one of those series is a hole where it costs most.
    The slice is keyed on the date so it is deterministic -- a rerun of a given
    night asks for exactly what that night asked for, which is the same
    property the rest of the store is built on.
    """
    eligible = eligible_tickers(as_of)
    keep = list(dict.fromkeys(eligible))
    known = set(keep)
    rest = [t for t in registrants if t not in known]

    room = MAX_NIGHTLY_TICKERS - len(keep)
    if room <= 0 or not rest:
        return keep

    # Ordinal of the date, so consecutive nights take consecutive slices and
    # the cycle length is len(rest) / room nights.
    offset = (date.fromisoformat(as_of).toordinal() * room) % len(rest)
    slice_ = rest[offset:offset + room]
    if len(slice_) < room:
        slice_ += rest[:room - len(slice_)]
    return keep + slice_


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

    asked = nightly_tickers(as_of, registrants)
    results["asked"] = len(asked)

    if bootstrap:
        results["bootstrap"] = bootstrap_history(asked, as_of=as_of)

    results["daily_bars"] = record_daily_bars(asked, as_of=as_of)
    results["universe"] = refresh_universe(as_of=as_of)
    results["consensus"] = record_consensus_snapshots(as_of=as_of)
    return results


# ------------------------------------------------------------- entry point

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
                        help="date to record as (default: today)")
    parser.add_argument("--bootstrap", action="store_true",
                        help="pull history first; needed once on a cold store")
    args = parser.parse_args(argv)

    kwargs: Dict[str, Any] = {"as_of": args.as_of}
    if args.bootstrap:
        kwargs["bootstrap"] = True
    result = run_all(**kwargs)
    print(json.dumps(result, indent=2, default=str))

    if result.get("error"):
        return 1
    stages = [v for k, v in result.items()
              if isinstance(v, dict) and "status" in v]
    return 1 if any(s["status"] == "failed" for s in stages) else 0


if __name__ == "__main__":  # pragma: no cover - exercised via main()
    raise SystemExit(main())
