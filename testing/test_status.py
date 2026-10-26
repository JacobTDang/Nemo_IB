"""One command that answers "is this thing working" from a terminal.

The store already holds everything needed to answer it. `run_log` says what ran
and when, `missing_days` turns that into gaps, and the tables say how much is
in there. Nothing assembled it, so answering the question meant knowing which
SQL to type into a container, which is not a thing anybody does at 2am.

What matters most here is the difference between a job that ran and found
nothing, and a job that did not run. Both write no rows. Only the run log
separates them, and only if the report reads it rather than counting rows.
"""
from __future__ import annotations

import json
import pathlib
from datetime import date, timedelta

import pytest

from research import pit_store, scanner, status

ROOT = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("NEMO_PIT_DB", str(tmp_path / "pit.db"))
    pit_store.init_schema()
    return pit_store


# --- the job list may not drift from what is actually scheduled -------------

def test_every_scheduled_job_is_one_this_report_knows_about():
    """Written by hand, this list goes stale the first time a job is added,
    and a status report that silently omits a job is worse than none."""
    import yaml

    compose = yaml.safe_load(
        (ROOT / "deploy" / "docker-compose.yml").read_text())
    modules = set()
    for service in compose["services"].values():
        entrypoint = service.get("entrypoint") or []
        if isinstance(entrypoint, list) and "-m" in entrypoint:
            module = entrypoint[entrypoint.index("-m") + 1]
            if module.startswith("research.") and module != "research.status":
                modules.add(module)

    covered = {j["module"] for j in status.JOBS}
    assert modules <= covered, (
        f"these are scheduled and the status report does not know them: "
        f"{sorted(modules - covered)}")


def test_the_report_names_a_job_that_has_never_run(store):
    """The case the whole thing exists for. A job that never ran writes no
    rows, and so does a job that ran and found nothing."""
    report = status.collect()

    never = [j for j in report["jobs"] if j["last_run"] is None]
    assert never, "a fresh store should report every job as never run"
    assert all(j["state"] == "never" for j in never)


def test_a_job_that_ran_and_found_nothing_is_not_reported_as_missing(store):
    store.start_run("score", as_of_date="2026-03-03")
    store.finish_run(rows_written=0, status="ok",
                     error="no finished horizons to score")

    row = _job(status.collect(as_of="2026-03-03"), "score")

    assert row["state"] == "ok"
    assert row["rows_written"] == 0
    assert row["last_run"] == "2026-03-03"


def test_a_failed_run_is_reported_as_failed(store):
    store.start_run("consensus", as_of_date="2026-03-03")
    store.finish_run(rows_written=0, status="failed", error="HTTP 503")

    row = _job(status.collect(as_of="2026-03-03"), "consensus")

    assert row["state"] == "failed"
    assert "503" in row["error"]


def test_a_run_that_never_finished_is_not_counted_as_a_run(store):
    """A crashed process leaves a started row with no finish. Reading that as
    coverage is how a job that dies every night looks healthy."""
    store.start_run("scan", as_of_date="2026-03-03")

    row = _job(status.collect(as_of="2026-03-03"), "scan")

    assert row["state"] == "crashed", (
        "a started-and-unfinished run was reported as a completed one")


# --- what needs attention ---------------------------------------------------

def test_a_stale_recorder_is_raised_for_attention(store):
    """The scanner refuses outright once a recorder is 5 days silent, so the
    report has to say so before the refusal does."""
    store.start_run("daily_bars", as_of_date="2026-03-01")
    store.finish_run(rows_written=10, status="ok")

    report = status.collect(as_of="2026-03-20")

    assert any("daily_bars" in a for a in report["attention"])


def test_a_job_that_stops_running_is_caught_even_if_it_was_only_ever_partial(
        store):
    """The hole this closes. Staleness used to be measured on the last *clean*
    success, and `daily_bars` is routinely `partial` because two names in
    1,565 time out. Keyed on clean successes it had none to age from, so a
    recorder that stopped entirely would have sat there reporting `partial`
    from its last night forever."""
    store.start_run("daily_bars", as_of_date="2026-03-01")
    store.finish_run(rows_written=1563, status="partial", error="2 of 1565")

    row = _job(status.collect(as_of="2026-03-20"), "daily_bars")

    assert row["last_success"] is None, "partial is not a clean success"
    assert row["state"] == "stale", "a recorder that stopped was not caught"


def test_a_recent_partial_run_is_not_called_stale(store):
    store.start_run("daily_bars", as_of_date="2026-03-03")
    store.finish_run(rows_written=1563, status="partial", error="2 of 1565")

    row = _job(status.collect(as_of="2026-03-04"), "daily_bars")

    assert row["state"] == "partial"


def test_a_healthy_store_raises_nothing(store):
    for job in [j["job"] for j in status.JOBS]:
        store.start_run(job, as_of_date="2026-03-03")
        store.finish_run(rows_written=1, status="ok")

    report = status.collect(as_of="2026-03-03")

    assert report["attention"] == [], report["attention"]


def test_a_job_that_is_not_due_yet_is_shown_but_not_raised(store, capsys):
    """The store's second day. Score and announce run on Saturday and seed on
    the 1st, so none of them has had a night to run on. Raising them failed
    the noon check every weekday for two weeks for a known reason, which is
    how a reader learns to stop reading it."""
    store.start_run("daily_bars", as_of_date="2026-10-19")
    store.finish_run(rows_written=1565, status="ok")

    rc = status.main(["--as-of", "2026-10-20"])
    out = capsys.readouterr().out

    line = [x for x in out.splitlines() if " score " in x][0]
    assert "never" in line, f"a job that has not run is hidden: {line!r}"
    assert line.startswith("~"), f"expected a quiet mark, got: {line!r}"
    assert rc == 0, "a store one day old fails its healthcheck"


def test_a_job_that_never_ran_is_raised_once_the_store_outlives_its_tolerance(
        store):
    """Measured from the store's first run, with the job's own tolerance.
    Score may be silent for ten days, so day ten is quiet and day eleven is an
    outage -- the same line a job that ran once and stopped would cross."""
    store.start_run("daily_bars", as_of_date="2026-10-09")
    store.finish_run(rows_written=1565, status="ok")

    def raised(as_of, job):
        return any(a.startswith(job + " ") for a in
                   status.collect(as_of=as_of)["attention"])

    assert not raised("2026-10-19", "score")
    assert raised("2026-10-20", "score"), (
        "a weekly job with no run in eleven days was not raised")


def test_each_job_that_never_ran_is_measured_on_its_own_tolerance(store):
    """Six days in, a weeknight job with no run is an outage and a monthly one
    is not due yet. One tolerance for every job gets one of them wrong."""
    store.start_run("daily_bars", as_of_date="2026-10-09")
    store.finish_run(rows_written=1565, status="ok")

    attention = status.collect(as_of="2026-10-15")["attention"]

    assert any(a.startswith("newcomers ") for a in attention)
    assert not any(a.startswith("seed ") for a in attention)


def test_the_exit_code_says_whether_anything_needs_attention(store, capsys):
    """So the command is usable in a cron healthcheck, not only by eye."""
    assert status.main(["--json"]) == 1, "a fresh store needs attention"

    for job in [j["job"] for j in status.JOBS]:
        store.start_run(job, as_of_date="2026-03-03")
        store.finish_run(rows_written=1, status="ok")
    capsys.readouterr()

    assert status.main(["--json", "--as-of", "2026-03-03"]) == 0


def test_the_marker_column_agrees_with_the_attention_list(store, capsys):
    """A `!` beside a job that the ATTENTION list does not mention draws the
    eye and then offers nothing to do. Two of the six jobs did that: a night
    where 2 of 1,565 names timed out is `partial`, which is normal on this
    universe and needs no person.

    So `!` means exactly "this is in the list below", and partial coverage
    gets its own quieter mark.
    """
    store.start_run("daily_bars", as_of_date="2026-03-03")
    store.finish_run(rows_written=1563, status="partial",
                     error="2 of 1565 names returned no data")
    store.start_run("consensus", as_of_date="2026-03-03")
    store.finish_run(rows_written=0, status="failed", error="HTTP 503")

    status.main(["--as-of", "2026-03-03"])
    out = capsys.readouterr().out

    flagged = {line.split()[1] for line in out.splitlines()
               if line.startswith("!")}
    mentioned = {word for line in out.splitlines() if line.strip().startswith("!")
                 for word in line.split()}

    assert "consensus" in flagged, "a failed job carries no marker"
    assert "daily_bars" not in flagged, (
        "a partial night is marked as needing action and then not explained")
    assert "daily_bars" not in mentioned or "partial" in out


def test_a_partial_run_is_still_visible_as_partial(store, capsys):
    """Quieter, not hidden. Two names timing out is worth seeing."""
    store.start_run("daily_bars", as_of_date="2026-03-03")
    store.finish_run(rows_written=1563, status="partial", error="2 of 1565")

    status.main(["--as-of", "2026-03-03"])
    out = capsys.readouterr().out

    line = [x for x in out.splitlines() if "daily_bars" in x][0]
    assert "partial" in line
    assert line.startswith("~"), f"expected a quiet mark, got: {line!r}"


# --- the two output shapes --------------------------------------------------

def test_json_output_parses(store, capsys):
    status.main(["--json"])

    payload = json.loads(capsys.readouterr().out)
    assert "jobs" in payload and "store" in payload and "attention" in payload


def test_the_plain_report_names_every_job(store, capsys):
    status.main([])

    out = capsys.readouterr().out
    for job in [j["job"] for j in status.JOBS]:
        assert job in out, f"{job} is missing from the report"


def test_the_report_survives_a_store_with_no_tables(tmp_path, monkeypatch,
                                                    capsys):
    """The first thing anyone runs on a fresh volume, before any job has
    created the schema. It must say so rather than raise."""
    monkeypatch.setenv("NEMO_PIT_DB", str(tmp_path / "absent.db"))

    rc = status.main([])

    assert rc == 1
    assert "no store" in capsys.readouterr().out.lower()


# --- the store summary ------------------------------------------------------

def test_the_store_summary_counts_rows_and_bytes(store):
    store.record_bars("AAA", [{"trade_date": "2026-03-02", "open": 1.0,
                               "high": 1.0, "low": 1.0, "close": 1.0,
                               "volume": 10}],
                      recorded_at="2026-03-02T21:00:00Z")

    summary = status.collect()["store"]

    assert summary["bytes"] > 0
    assert summary["tables"]["daily_bar"] == 1
    assert summary["path"].endswith(".db")


# --- gap window: clamped to first_run, ending on yesterday ("never today") ---

def test_job_first_run_three_days_ago_reports_no_gaps_from_before_first_run(store):
    """Before its first run the job did not exist, so those weekdays are not
    missed nights. The window must clamp to first_run(job)."""
    # 2026-10-22 is Thursday. 14 days back is 2026-10-08.
    # First run on Monday 2026-10-19 (3 days before 2026-10-22).
    # Ran Monday, Tuesday, Wednesday.
    for day in ["2026-10-19", "2026-10-20", "2026-10-21"]:
        store.start_run("scan", as_of_date=day)
        store.finish_run(rows_written=10, status="ok")

    row = _job(status.collect(as_of="2026-10-22"), "scan")

    assert row["gaps"] == 0


def test_job_that_ran_every_weekday_through_yesterday_reports_zero_gaps_before_todays_run(
        store):
    """research-status runs at midday (12:00 UTC) while nightly jobs run at
    22:30 UTC. The window must end on the day before as_of so today is not
    counted as a gap before its run is due."""
    # 2026-10-22 is Thursday.
    # Job has run every weekday from 2026-10-08 through 2026-10-21.
    weekdays = [
        "2026-10-08", "2026-10-09",
        "2026-10-12", "2026-10-13", "2026-10-14", "2026-10-15", "2026-10-16",
        "2026-10-19", "2026-10-20", "2026-10-21",
    ]
    for day in weekdays:
        store.start_run("scan", as_of_date=day)
        store.finish_run(rows_written=10, status="ok")

    row = _job(status.collect(as_of="2026-10-22"), "scan")

    assert row["gaps"] == 0


def test_weekday_really_missed_inside_window_is_still_counted_as_gap(store):
    """A genuine missed weekday between first_run and yesterday must still
    be counted as a gap."""
    # First run on 2026-10-15 (Thu).
    # Ran 2026-10-15, 2026-10-16 (Fri).
    # Missed 2026-10-19 (Mon).
    # Ran 2026-10-20 (Tue), 2026-10-21 (Wed).
    for day in ["2026-10-15", "2026-10-16", "2026-10-20", "2026-10-21"]:
        store.start_run("scan", as_of_date=day)
        store.finish_run(rows_written=10, status="ok")

    row = _job(status.collect(as_of="2026-10-22"), "scan")

    assert row["gaps"] == 1


# --- staleness measured in weekdays for weekday jobs -----------------------

def test_a_friday_run_read_on_monday_is_not_stale(store):
    """activist_watch runs on weekdays (13-23 UTC) and research-status runs
    at 12:00 UTC. On Monday at 12:00, Friday's run is one weekday old, not
    three calendar days old, so it is not stale."""
    # 2026-10-23 is Friday, 2026-10-26 is Monday.
    store.start_run("activist_watch", as_of_date="2026-10-23")
    store.finish_run(rows_written=10, status="ok")

    row = _job(status.collect(as_of="2026-10-26"), "activist_watch")

    assert row["age_days"] == 1
    assert row["state"] == "ok"


def test_a_weekday_job_that_missed_a_whole_weekday_is_stale(store):
    """A weekday job that missed a whole weekday (e.g. last run Thursday, read
    Monday at 12:00) missed Friday; age is two weekdays and the job is stale."""
    # 2026-10-22 is Thursday, 2026-10-26 is Monday.
    store.start_run("activist_watch", as_of_date="2026-10-22")
    store.finish_run(rows_written=10, status="ok")

    row = _job(status.collect(as_of="2026-10-26"), "activist_watch")

    assert row["age_days"] == 2
    assert row["state"] == "stale"


def test_a_saturday_job_still_uses_calendar_days(store):
    """Score runs on Saturday and its tolerance is calendar days, not
    weekdays. Silent for 11 calendar days (which is only 8 weekdays), it
    must be reported as stale."""
    # 2026-10-10 is Saturday. 2026-10-21 is Wednesday (11 calendar days later,
    # but only 8 weekdays: Oct 12-16 and 19-21). Tolerance is 10 days.
    store.start_run("score", as_of_date="2026-10-10")
    store.finish_run(rows_written=10, status="ok")

    row = _job(status.collect(as_of="2026-10-21"), "score")

    assert row["age_days"] == 11
    assert row["state"] == "stale"


def test_a_weeknight_job_that_missed_one_night_is_not_stale(store):
    """The tolerance is the schedule plus room for one missed run. Thursday's
    run read on Monday missed only Friday night."""
    store.start_run("daily_bars", as_of_date="2026-10-22")
    store.finish_run(rows_written=10, status="ok")

    row = _job(status.collect(as_of="2026-10-26"), "daily_bars")

    assert row["age_days"] == 2
    assert row["state"] == "ok"


def test_a_weeknight_job_that_missed_two_nights_is_stale(store):
    """Wednesday's run read on Monday missed Thursday and Friday nights."""
    store.start_run("daily_bars", as_of_date="2026-10-21")
    store.finish_run(rows_written=10, status="ok")

    report = status.collect(as_of="2026-10-26")

    assert _job(report, "daily_bars")["state"] == "stale"
    assert ("daily_bars last ran 3 weekdays ago, for 2026-10-21"
            in report["attention"])


def test_a_recorder_is_raised_before_the_scanner_refuses_it():
    """The scanner refuses once a recorder's last run is more than
    MAX_RECORDER_GAP_DAYS calendar days old. The report counts weekdays for
    these jobs, so its tolerance must still be crossed first, on every
    weekday it runs and whatever day the last run was."""
    monday = date(2026, 10, 5)
    for spec in [s for s in status.JOBS if s["job"] in scanner.RECORDER_JOBS]:
        for offset in range(14):
            as_of = monday + timedelta(days=offset)
            if as_of.weekday() >= 5:
                continue
            for gap in range(scanner.MAX_RECORDER_GAP_DAYS + 1, 21):
                last = as_of - timedelta(days=gap)
                age = status._age(spec, as_of.isoformat(), last.isoformat())
                assert age > spec["stale_after_days"], (
                    f"{spec['job']} last ran {last}, read {as_of}: the scanner "
                    f"refuses it and the report calls it fresh")


def test_a_weekday_job_that_never_ran_is_not_due_over_a_weekend(store):
    """The store's first run was Friday. On Monday a weeknight job with no run
    has had one weekday to run on, not three days."""
    store.start_run("daily_bars", as_of_date="2026-10-23")
    store.finish_run(rows_written=1565, status="ok")

    attention = status.collect(as_of="2026-10-26")["attention"]

    assert not any(a.startswith("newcomers ") for a in attention), attention


def _job(report, name):
    match = [j for j in report["jobs"] if j["job"] == name]
    assert match, f"{name} not in the report"
    return match[0]
