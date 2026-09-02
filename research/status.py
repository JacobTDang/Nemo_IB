"""What the automation did, in one screen, for somebody with only a terminal.

The store already holds the answer. `run_log` says what ran and when,
`missing_days` turns that into gaps, and the tables say how much accumulated.
Nothing assembled it, so answering "is this working" meant knowing which SQL to
type into a container, which is not a thing anybody does at 2am.

The distinction this report exists to draw is between **a job that ran and
found nothing** and **a job that did not run**. Both write zero rows. Most
nights the first one is normal and the second one is an outage, and only the
run log separates them -- so this reads the log rather than counting rows.

Three states beyond the obvious, each of which looks like success to a reader
counting rows:

  `never`   - the job has no finished run at all. On a fresh volume that is
              every job and it is fine. Three weeks in it is an outage.
  `crashed` - a run started and never finished. The process died, and
              `missing_days` already refuses to count it as coverage.
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
# room for one missed run, not round numbers.
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
    {"job": "daily_bars", "module": "research.daily_job",
     "every": "weeknight", "stale_after_days": 5,
     "partial_is_normal": False},
    {"job": "newcomers", "module": "research.daily_job",
     "every": "weeknight", "stale_after_days": 5,
     "partial_is_normal": False},
    {"job": "universe", "module": "research.daily_job",
     "every": "weeknight", "stale_after_days": 5,
     "partial_is_normal": False},
    {"job": "consensus", "module": "research.daily_job",
     "every": "weeknight", "stale_after_days": 5,
     "partial_is_normal": False},
    {"job": "scan", "module": "research.scanner",
     "every": "weeknight", "stale_after_days": 5,
     "partial_is_normal": False},
    {"job": "activist_watch", "module": "research.activist_watch",
     "every": "20 minutes", "stale_after_days": 2,
     # Bounded by a clock, so one pass covers a slice and never the
     # whole watchlist. See `research.activist_watch`.
     "partial_is_normal": True},
    {"job": "score", "module": "research.scoring",
     "every": "Saturday", "stale_after_days": 10,
     "partial_is_normal": False},
    {"job": "announce", "module": "research.announcements",
     "every": "Saturday", "stale_after_days": 10,
     "partial_is_normal": False},
    {"job": "seed", "module": "research.seed_consensus",
     "every": "monthly", "stale_after_days": 40,
     "partial_is_normal": False},
)

# Reported in full. Anything else in the schema still shows up under `tables`.
HEADLINE_TABLES = ("daily_bar", "universe_snapshot", "consensus_snapshot",
                   "announcement", "paper_order", "activist_filing",
                   "borrow_rate", "run_log")


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


def _last_run(job: str) -> Optional[Dict[str, Any]]:
    with pit_store.connect() as conn:
        row = conn.execute(
            "SELECT * FROM run_log WHERE job = ? ORDER BY run_id DESC LIMIT 1",
            (job,)).fetchone()
    return dict(row) if row else None


def _job_state(spec: Dict[str, Any], as_of: str) -> Dict[str, Any]:
    """One job's line. Reads the log, never the row counts.

    A job that found nothing and a job that did not run both wrote zero rows,
    and the whole point of this report is that they are different facts.
    """
    job = spec["job"]
    out = {"job": job, "module": spec["module"], "every": spec["every"],
           "partial_is_normal": spec["partial_is_normal"],
           "last_run": None, "last_success": None, "state": "never",
           "status": None, "rows_written": None, "error": None,
           "age_days": None, "gaps": None}

    row = _last_run(job)
    if row is None:
        return out

    out.update({"last_run": row["as_of_date"], "status": row["status"],
                "rows_written": row["rows_written"], "error": row["error"]})

    if not row["finished_at"]:
        # `missing_days` already refuses to count this as coverage. Saying so
        # here is what turns "no rows last night" into "the process died".
        out["state"] = "crashed"
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
        age = (date.fromisoformat(as_of)
               - date.fromisoformat(row["as_of_date"])).days
        out["age_days"] = age
        if age > spec["stale_after_days"]:
            out["state"] = "stale"

    start = (date.fromisoformat(as_of) - timedelta(days=14)).isoformat()
    out["gaps"] = len(pit_store.missing_days(job, start, as_of))
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
        if latest is None:
            return {"as_of": None, "accepted": 0, "issuers": 0, "shorts": 0,
                    "borrow_rates": conn.execute(
                        "SELECT count(*) FROM borrow_rate").fetchone()[0]}
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
            "borrow_rates": borrow_rates}


def collect(as_of: Optional[str] = None) -> Dict[str, Any]:
    """Everything the report shows, as data. `main` only formats it."""
    as_of = as_of or _today()
    jobs = [_job_state(spec, as_of) for spec in JOBS]

    attention: List[str] = []
    for job in jobs:
        if job["state"] == "never":
            attention.append(
                f"{job['job']} has never completed a run (runs {job['every']})")
        elif job["state"] == "crashed":
            attention.append(
                f"{job['job']} started a run for {job['last_run']} and never "
                f"finished it")
        elif job["state"] == "stale":
            attention.append(
                f"{job['job']} last ran {job['age_days']} days ago, for "
                f"{job['last_run']}")
        elif job["state"] == "failed":
            attention.append(
                f"{job['job']} failed on {job['last_run']}: "
                f"{(job['error'] or '')[:80]}")


    return {"as_of": as_of, "jobs": jobs, "store": _store_summary(),
            "book": _book_summary(as_of), "attention": attention}


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
        mark = "!" if raised else ("~" if job["state"] not in ("ok", "closed")
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

    if report["attention"]:
        lines += ["", "  ATTENTION"]
        lines += [f"    ! {a}" for a in report["attention"]]
    else:
        lines += ["", "  nothing needs attention"]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    """`python -m research.status`. Exit 1 when something needs attention."""
    import argparse
    import json

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
    return 1 if report["attention"] else 0


if __name__ == "__main__":  # pragma: no cover - exercised via main()
    raise SystemExit(main())
