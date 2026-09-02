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

import pytest

from research import pit_store, status

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


def _job(report, name):
    match = [j for j in report["jobs"] if j["job"] == name]
    assert match, f"{name} not in the report"
    return match[0]
