"""What the automation did, in one screen, for somebody with only a terminal.

The store already holds the answer. `run_log` says what ran and when,
`missing_days` turns that into gaps, and the tables say how much accumulated.
Nothing assembled it, so answering "is this working" meant knowing which SQL to
type into a container, which is not a thing anybody does at 2am.

The distinction this report exists to draw is between **a job that ran and
found nothing** and **a job that did not run**. Both write zero rows. Most
nights the first one is normal and the second one is an outage, and only the
run log separates them -- so this reads the log rather than counting rows.

Four states beyond the obvious, each of which looks like success to a reader
counting rows:

  `never`   - the job has no run at all. On a fresh volume that is every job
              and it is fine. Three weeks in it is an outage. The line between
              is the job's own tolerance, counted from the store's first run:
              a Saturday job is not due on a Tuesday. A store with no runs at
              all has nothing to count from, and is raised.
  `running` - a run started and has not finished, but is younger than the
              longest that job's runs take. It is still working.
  `crashed` - a run started, never finished, and is older than that. The
              process died, and `missing_days` already refuses to count it
              as coverage.
  `stale`   - the last success is further back than the job's own tolerance.
              The scanner refuses outright at five days, so the report has to
              say so before the refusal does.

Exit code 1 when anything needs attention, so the command works in a cron
healthcheck as well as by eye.
"""
from __future__ import annotations

import os
import sqlite3
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from research import pit_store

# Every scheduled job, the module that runs it, and how long it may be silent
# before silence is a problem. The tolerances are the jobs' own schedules plus
# room for one missed run, not round numbers. A job that runs only on weekdays
# counts them in weekdays (see `_age`), so a weekend is not silence.
#
# `max_run_seconds` is how long an unfinished run is still `running`; older,
# it is `crashed`. Without it every run in progress read as a crash (#179). At
# least twice the longest run on the VM by 2026-11-05, and never under ten
# minutes: seed 82.7 min, announce 67.6, daily_bars 3.6, every other nightly
# stage and the fills under one. activist_watch stops itself at 600 s.
#
# `partial_is_normal` marks a job whose steady state is partial coverage.
# `activist_watch` watches a time-bounded slice of the watchlist on purpose, so
# every pass is honestly `partial` and every day is honestly a gap. Both are
# true and neither needs a person. Flagging them would put four alarms an hour
# in front of a reader, which is how the one that matters stops being read.
#
# `testing/test_status.py` checks this list against the compose file, because a
# report that quietly omits a job is worse than no report.
JOBS = (
    {"job": "congress_copy", "module": "research.congress_copy",
     "every": "weekday", "stale_after_days": 2,
     "max_run_seconds": 600, "partial_is_normal": False},
    # QuiverQuant (research/quiver_record.py). Both stages log a run every
    # night, the weekly lists included (0 rows once the week is held), so
    # both are due every calendar day.
    {"job": "quiver_feeds", "module": "research.quiver_record",
     "every": "daily", "stale_after_days": 2,
     "max_run_seconds": 600, "partial_is_normal": False},
    {"job": "quiver_lists", "module": "research.quiver_record",
     "every": "daily", "stale_after_days": 2,
     "max_run_seconds": 600, "partial_is_normal": False},
    {"job": "daily_bars", "module": "research.daily_job",
     "every": "weeknight", "stale_after_days": 2,
     "max_run_seconds": 600,
     "partial_is_normal": False},
    {"job": "newcomers", "module": "research.daily_job",
     "every": "weeknight", "stale_after_days": 2,
     "max_run_seconds": 600,
     "partial_is_normal": False},
    {"job": "universe", "module": "research.daily_job",
     "every": "weeknight", "stale_after_days": 2,
     "max_run_seconds": 600,
     "partial_is_normal": False},
    {"job": "consensus", "module": "research.daily_job",
     "every": "weeknight", "stale_after_days": 2,
     "max_run_seconds": 600,
     "partial_is_normal": False},
    # FINRA publishes short interest twice a month, so most passes find
    # nothing -- but the pass itself runs every night as a stage of
    # research-daily and logs a run either way. A night with no run is a
    # night the job did not run, which is what `stale_after_days` reports.
    {"job": "short_interest", "module": "research.short_interest",
     "every": "weeknight", "stale_after_days": 2,
     "max_run_seconds": 600,
     "partial_is_normal": False},
    {"job": "scan", "module": "research.scanner",
     "every": "weeknight", "stale_after_days": 2,
     "max_run_seconds": 600,
     "partial_is_normal": False},
    {"job": "activist_watch", "module": "research.activist_watch",
     "every": "20 minutes", "stale_after_days": 1,
     # Bounded by a clock, so one pass covers a slice and never the
     # whole watchlist. See `research.activist_watch`. One weekday with
     # no finished pass is already dozens of missed runs. The clock is
     # 600 s; five more minutes cover writing the pass out.
     "max_run_seconds": 900,
     "partial_is_normal": True},
    {"job": "score", "module": "research.scoring",
     "every": "Saturday", "stale_after_days": 10,
     "max_run_seconds": 600,
     "partial_is_normal": False},
    {"job": "announce", "module": "research.announcements",
     "every": "Saturday", "stale_after_days": 10,
     "max_run_seconds": 8100,
     "partial_is_normal": False},
    {"job": "seed", "module": "research.seed_consensus",
     "every": "monthly", "stale_after_days": 40,
     "max_run_seconds": 10800,
     "partial_is_normal": False},
    # Paper fills (issue #116). Optional: they run only once Alpaca paper keys
    # are in .env, so a job that has never run is shown, not raised. Once it
    # has run, a failure or a stall is raised like any other.
    {"job": "fill_submit", "module": "tools.alpaca.fills",
     "every": "weekday", "stale_after_days": 2,
     "max_run_seconds": 600,
     "partial_is_normal": False, "optional": True},
    {"job": "fill_collect", "module": "tools.alpaca.fills",
     "every": "weekday", "stale_after_days": 2,
     "max_run_seconds": 600,
     "partial_is_normal": False, "optional": True},
)

# Reported in full. Anything else in the schema still shows up under `tables`.
HEADLINE_TABLES = ("daily_bar", "universe_snapshot", "consensus_snapshot",
                   "announcement", "paper_order", "activist_filing",
                   "borrow_rate", "short_interest", "run_log")

WEEKDAY_SCHEDULES = ("weeknight", "weekday", "20 minutes")


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _store_exists() -> bool:
    path = pit_store.db_path()
    if not os.path.exists(path):
        return False
    try:
        with pit_store.connect() as conn:
            conn.execute("SELECT 1 FROM run_log LIMIT 1")
        return True
    except sqlite3.DatabaseError:
        return False


def _first_run() -> Optional[str]:
    """The earliest date any job ran for, or None on a store with no runs.

    How long the store has been running, which is what separates a job that
    has not been due yet from one that has stopped. Read from `as_of_date`,
    the same date staleness is measured on, so `--as-of` moves both together.
    """
    with pit_store.connect() as conn:
        row = conn.execute(
            "SELECT min(as_of_date) FROM run_log WHERE as_of_date IS NOT NULL"
        ).fetchone()
    return row[0]


def _last_run(job: str) -> Optional[Dict[str, Any]]:
    with pit_store.connect() as conn:
        row = conn.execute(
            "SELECT * FROM run_log WHERE job = ? ORDER BY run_id DESC LIMIT 1",
            (job,)).fetchone()
    return dict(row) if row else None


def _age(spec: Dict[str, Any], as_of: str, since: str) -> int:
    """Days from `since` to `as_of`, in the unit of the job's tolerance.

    A weekday job counts only the weekdays after `since`, up to and including
    `as_of`: Friday to Monday is one. The noon check runs before the day's
    first pass, so Friday is the latest run a Monday reading can see.
    """
    start, end = date.fromisoformat(since), date.fromisoformat(as_of)
    if spec["every"] not in WEEKDAY_SCHEDULES:
        return (end - start).days
    return sum(1 for n in range(1, (end - start).days + 1)
               if (start + timedelta(days=n)).weekday() < 5)


def _age_unit(spec: Dict[str, Any]) -> str:
    return "weekdays" if spec["every"] in WEEKDAY_SCHEDULES else "days"


def _job_state(spec: Dict[str, Any], as_of: str) -> Dict[str, Any]:
    """One job's line. Reads the log, never the row counts.

    A job that found nothing and a job that did not run both wrote zero rows,
    and the whole point of this report is that they are different facts.
    """
    job = spec["job"]
    out = {"job": job, "module": spec["module"], "every": spec["every"],
           "stale_after_days": spec["stale_after_days"],
           "max_run_seconds": spec["max_run_seconds"],
           "partial_is_normal": spec["partial_is_normal"],
           "optional": spec.get("optional", False),
           "last_run": None, "last_success": None, "state": "never",
           "status": None, "rows_written": None, "error": None,
           "age_days": None, "gaps": None}

    if job == "congress_copy":
        from research.congress_copy import enabled
        if not enabled():
            out["state"] = "off"
            return out

    row = _last_run(job)
    if row is None:
        return out

    out.update({"last_run": row["as_of_date"], "status": row["status"],
                "rows_written": row["rows_written"], "error": row["error"]})

    if not row["finished_at"]:
        # Measured against the wall clock, not `as_of`: whether a process is
        # still alive is a fact about now. `missing_days` refuses to count
        # either as coverage; saying which one turns "no rows last night"
        # into "the process died".
        started = datetime.fromisoformat(row["started_at"].replace("Z", "+00:00"))
        age = (datetime.now(timezone.utc) - started).total_seconds()
        out["state"] = "running" if age <= spec["max_run_seconds"] else "crashed"
        return out

    # The run's own verdict stands. `last_successful_run` counts only 'ok'
    # and 'closed' -- correctly, because the scanner uses it to ask whether the
    # recorder is genuinely working -- so a `partial` job has no recent
    # "success" by that definition. Rewriting it to `failed` on that basis
    # turned every night where two names timed out into an alarm.
    out["state"] = row["status"] or "unknown"
    # Reported for information. NOT what staleness is measured on:
    # `last_successful_run` counts only 'ok' and 'closed', and `daily_bars` is
    # routinely `partial` because two names in 1,565 time out. Keyed on clean
    # successes, that job has no recent success on any night, so it would read
    # as permanently alarming -- and, worse, a `daily_bars` that stopped
    # running altogether would never be called stale, because there was never
    # a success to age from.
    out["last_success"] = pit_store.last_successful_run(job, as_of)

    # Staleness is about whether the job is still running, so it is measured
    # on the last run that finished, whatever it concluded. A job that has
    # stopped is caught regardless of which status word it used.
    if row["as_of_date"]:
        age = _age(spec, as_of, row["as_of_date"])
        out["age_days"] = age
        if age > spec["stale_after_days"]:
            out["state"] = "stale"

    # Gaps in the trailing 14-day window.
    # Clamped to the first date the job ever ran for: before that the job did
    # not exist, so those weekdays are not missed nights (matches
    # `daily_job.coverage_gaps`).
    # Ends on the day before `as_of`: `research-status` runs at midday (12:00
    # UTC) while nightly jobs run at 22:30 UTC, so today is not counted as a
    # gap before its run is due (matches `daily_job.backfill`, "never today").
    first = pit_store.first_run(job)
    window_start = (date.fromisoformat(as_of) - timedelta(days=14)).isoformat()
    start = max(window_start, first) if first else window_start
    yesterday = (date.fromisoformat(as_of) - timedelta(days=1)).isoformat()
    if start > yesterday:
        out["gaps"] = 0
    else:
        out["gaps"] = len(pit_store.missing_days(job, start, yesterday,
                                                 every=spec["every"]))
    return out


def _store_summary() -> Dict[str, Any]:
    path = pit_store.db_path()
    tables: Dict[str, int] = {}
    with pit_store.connect() as conn:
        names = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        for name in names:
            tables[name] = conn.execute(
                f"SELECT count(*) FROM {name}").fetchone()[0]
        span = conn.execute(
            "SELECT min(trade_date), max(trade_date) FROM daily_bar"
        ).fetchone() if "daily_bar" in names else (None, None)
    return {"path": path, "bytes": os.path.getsize(path),
            "tables": tables, "earliest_bar": span[0], "latest_bar": span[1]}


def _book_summary(as_of: str) -> Dict[str, Any]:
    """The last decision on record, and whether it holds any shorts.

    A book with no shorts is the expected state until borrow rates are loaded,
    and saying so here stops it reading as a broken scan.
    """
    with pit_store.connect() as conn:
        latest = conn.execute(
            "SELECT max(as_of_date) FROM paper_order WHERE accepted = 1"
        ).fetchone()[0]
        # Crowding, not cost. Reported beside the borrow count because the two
        # are read together and confused apart: a book with no shorts and no
        # borrow rates is the expected state, and a book with no shorts and a
        # year of short interest is a book that still refuses every short.
        short_interest = conn.execute(
            "SELECT count(*), max(settlement_date) FROM short_interest"
        ).fetchone()
        if latest is None:
            return {"as_of": None, "accepted": 0, "issuers": 0, "shorts": 0,
                    "borrow_rates": conn.execute(
                        "SELECT count(*) FROM borrow_rate").fetchone()[0],
                    "short_interest_rows": short_interest[0],
                    "short_interest_settlement": short_interest[1]}
        rows = conn.execute(
            """SELECT o.ticker, o.side, u.cik FROM paper_order o
                 LEFT JOIN universe_snapshot u ON u.ticker = o.ticker
                WHERE o.accepted = 1 AND o.as_of_date = ?
                GROUP BY o.ticker""", (latest,)).fetchall()
        borrow_rates = conn.execute(
            "SELECT count(*) FROM borrow_rate").fetchone()[0]
    return {"as_of": latest, "accepted": len(rows),
            "issuers": len({r["cik"] for r in rows if r["cik"]}),
            "shorts": sum(1 for r in rows if r["side"] == "short"),
            "borrow_rates": borrow_rates,
            "short_interest_rows": short_interest[0],
            "short_interest_settlement": short_interest[1]}


def collect(as_of: Optional[str] = None) -> Dict[str, Any]:
    """Everything the report shows, as data. `main` only formats it."""
    as_of = as_of or _today()
    jobs = [_job_state(spec, as_of) for spec in JOBS]

    # A job with no run is raised only once the store has been running for
    # longer than that job may be silent. On its second day the store has no
    # Saturday score and no monthly seed, and raising them failed the noon
    # check every weekday until the 1st -- for a known reason, which is how a
    # reader learns to skip it. Before that line the job keeps its `~`.
    first = _first_run()

    attention: List[str] = []
    for job in jobs:
        if job["state"] == "never":
            if job["optional"]:
                continue
            if (first is not None
                    and _age(job, as_of, first) <= job["stale_after_days"]):
                continue
            attention.append(
                f"{job['job']} has never completed a run (runs {job['every']})")
        elif job["state"] == "crashed":
            attention.append(
                f"{job['job']} started a run for {job['last_run']} and never "
                f"finished it")
        elif job["state"] == "stale":
            attention.append(
                f"{job['job']} last ran {job['age_days']} {_age_unit(job)} "
                f"ago, for "
                f"{job['last_run']}")
        elif job["state"] == "failed":
            attention.append(
                f"{job['job']} failed on {job['last_run']}: "
                f"{(job['error'] or '')[:80]}")


    risk = pit_store.risk_state(as_of)
    if risk["halted"]:
        attention.append(
            f"risk switch tripped on {risk['tripped']['as_of_date']}; the scan "
            f"files no orders until it is reset with a reason")

    return {"as_of": as_of, "jobs": jobs, "store": _store_summary(),
            "book": _book_summary(as_of), "risk": risk,
            "gate": pit_store.latest_gate_check(as_of),
            "copy": pit_store.copy_book_summary(as_of),
            "attention": attention}


def _render(report: Dict[str, Any]) -> str:
    """`!` needs a person, `~` is worth a glance, blank is clean."""
    lines = [f"NEMO STATUS   as of {report['as_of']}  "
             f"({datetime.now(timezone.utc).strftime('%H:%M')}Z)", ""]

    lines.append(f"  {'JOB':16} {'EVERY':11} {'LAST':11} {'STATE':9} "
                 f"{'ROWS':>9}  {'GAPS/14d':>8}")
    for job in report["jobs"]:
        # `!` means exactly "named in ATTENTION below". A marker that draws
        # the eye and then offers nothing to do is worse than no marker.
        raised = any(a.startswith(job["job"] + " ") for a in report["attention"])
        mark = "!" if raised else ("~" if job["state"] not in ("ok", "closed", "off")
                                   else " ")
        rows = "-" if job["rows_written"] is None else f"{job['rows_written']:,}"
        # A gap count is meaningless for a job that covers a slice per pass.
        gaps = ("n/a" if job["partial_is_normal"]
                else "-" if job["gaps"] is None else str(job["gaps"]))
        lines.append(
            f"{mark} {job['job']:16} {job['every']:11} "
            f"{job['last_run'] or '-':11} {job['state']:9} {rows:>9}  "
            f"{gaps:>8}")

    store = report["store"]
    lines += ["", f"  STORE  {store['path']}  "
                  f"{store['bytes'] / 1e6:,.0f} MB  "
                  f"bars {store['earliest_bar'] or '-'} to "
                  f"{store['latest_bar'] or '-'}"]
    for name in HEADLINE_TABLES:
        if name in store["tables"]:
            lines.append(f"         {name:20} {store['tables'][name]:>12,}")

    book = report["book"]
    lines += ["", f"  BOOK   {book['accepted']} orders for "
                  f"{book['as_of'] or 'no date yet'}, "
                  f"{book['issuers']} issuers, {book['shorts']} short, "
                  f"{book['borrow_rates']} borrow rates on file"]
    # Two numbers, never one. Borrow rates price a short; short interest says
    # how crowded it is. A book can hold either without the other, and a single
    # "short-side data" line would let a year of crowding read as though the
    # scan could price a short.
    lines.append(
        f"         crowdedness: {book['short_interest_rows']:,} short-interest "
        f"rows, newest settlement "
        f"{book['short_interest_settlement'] or 'none yet'}")

    lines += _risk_lines(report.get("risk"), report.get("gate"))
    lines += _copy_lines(report.get("copy"))

    if report["attention"]:
        lines += ["", "  ATTENTION"]
        lines += [f"    ! {a}" for a in report["attention"]]
    else:
        lines += ["", "  nothing needs attention"]
    return "\n".join(lines)


def _risk_lines(risk: Optional[Dict[str, Any]],
                gate: Optional[Dict[str, Any]]) -> List[str]:
    """The drawdown switch and the go/no-go gate, one line each (#117)."""
    lines = [""]
    latest = (risk or {}).get("latest")
    if not latest:
        lines.append("  RISK   no equity recorded yet; the scan records it "
                     "each night before deciding")
    else:
        switch = (f"switch TRIPPED on {risk['tripped']['as_of_date']}; reset "
                  f"with python -m research.risk --reset --reason"
                  if risk["halted"] else "switch armed")
        unmarked = (f", {latest['unmarked_trades']} unmarked"
                    if latest["unmarked_trades"] else "")
        lines.append(
            f"  RISK   equity {latest['equity_dollars']:+,.0f} "
            f"({latest['closed_trades']} closed, {latest['open_trades']} open"
            f"{unmarked}); drawdown {risk['drawdown_dollars']:,.0f} of a "
            f"{latest['limit_dollars']:,.0f} limit; {switch}")
    if not gate:
        lines.append("  GATE   not checked yet; the weekly score records it")
    else:
        verdict = "PASSED" if gate["passed"] else "not passed"
        lines.append(
            f"  GATE   {gate['measured_trades']} of {gate['min_trades']} trades "
            f"with measured fills; {verdict}: {gate['reason'][:110]}")
    return lines


def _copy_lines(copy: Optional[Dict[str, Any]]) -> List[str]:
    """Congressional copy book, kept off the GATE and RISK lines (#190)."""
    if not copy:
        return ["  COPY   no fills yet"]
    mean = copy.get("mean_return")
    mean_text = (f"; mean return {mean:+.1%}" if mean is not None
                 else "; no closed return yet")
    return [f"  COPY   {copy.get('open', 0)} open, "
            f"{copy.get('closed', 0)} closed{mean_text}"]


def main(argv: Optional[List[str]] = None) -> int:
    """`python -m research.status`. Exit 1 when something needs attention."""
    import argparse
    import json

    from research import discord_post
    elapsed = discord_post.stopwatch()

    parser = argparse.ArgumentParser(
        prog="status",
        description="What the automation did, and what needs attention.")
    parser.add_argument("--as-of", dest="as_of", default=None,
                        help="report as of this date (default: today, UTC)")
    parser.add_argument("--json", action="store_true",
                        help="machine-readable, for scripting")
    args = parser.parse_args(argv)

    if not _store_exists():
        # The first thing anybody runs on a fresh volume, before any job has
        # created the schema. Saying so beats a traceback about a missing
        # table, which reads like a broken install rather than an empty one.
        message = (f"no store at {pit_store.db_path()} yet. Run "
                   f"`research-daily --bootstrap` first.")
        print(json.dumps({"error": message}) if args.json else message)
        return 1

    report = collect(as_of=args.as_of)
    print(json.dumps(report, indent=2, default=str) if args.json
          else _render(report))
    code = 1 if report["attention"] else 0
    return discord_post.finish(
        "research-status", code, report, seconds=elapsed())


if __name__ == "__main__":  # pragma: no cover - exercised via main()
    raise SystemExit(main())
