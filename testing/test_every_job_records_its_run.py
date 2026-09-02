"""Every scheduled job leaves a row saying it ran, or the run log lies.

`pit_store.missing_days` exists to answer one question -- did this job run on
that day -- and states its own reason for existing:

    A gap you cannot see becomes a conclusion; a gap you can see becomes a
    caveat.

It reads `run_log`. Three of the six research jobs never wrote to it.
`research-announce` ran for 27 minutes against the real universe, exited 0,
wrote 26,129 announcement rows, and left no trace at all. `research-score` and
`research-seed` were the same.

The worst of the three is `research-score`, because it is the job that checks
the others. A weekly score run that quietly stops means the book is never
scored, the drift coefficient is never measured, and nothing anywhere says so.

The list below is not written by hand. It is resolved from the compose file, so
a job added tomorrow is covered tomorrow -- the same discipline the tool counts
and the cron lines are already held to.
"""
from __future__ import annotations

import pathlib

import pytest

from research import pit_store

ROOT = pathlib.Path(__file__).resolve().parent.parent
COMPOSE = ROOT / "deploy" / "docker-compose.yml"


def _scheduled_research_modules():
    """Every `research.*` module the compose file runs as a batch job.

    `congress-sync` is excluded on purpose: it writes a separate store and
    carries its own `--status` command, so it answers this question its own
    way rather than through `run_log`.

    `research-status` is excluded because it is the reader. It changes nothing
    and recording its own runs would put a row in the log for every time
    somebody looked at the log.
    """
    reporters = {"research.status"}
    import yaml

    services = yaml.safe_load(COMPOSE.read_text())["services"]
    found = []
    for name, service in services.items():
        entrypoint = service.get("entrypoint") or []
        if not isinstance(entrypoint, list) or "-m" not in entrypoint:
            continue
        module = entrypoint[entrypoint.index("-m") + 1]
        if module.startswith("research.") and module not in reporters:
            found.append((name, module))
    return sorted(found)


SCHEDULED = _scheduled_research_modules()


def test_the_compose_file_still_has_research_jobs_in_it():
    """So a parsing change cannot empty the sweep below into a silent pass."""
    assert len(SCHEDULED) >= 6, (
        f"only found {SCHEDULED}; the compose file has changed shape and this "
        f"sweep is no longer covering what it claims to")


@pytest.mark.parametrize("service,module", SCHEDULED,
                         ids=[s for s, _ in SCHEDULED])
def test_every_scheduled_research_job_records_a_run(service, module):
    source = (ROOT / (module.replace(".", "/") + ".py")).read_text()

    assert "start_run(" in source, (
        f"{service} runs {module}, which never calls start_run. A day it did "
        f"not run is invisible to missing_days, which is the only thing that "
        f"would show it")
    assert "finish_run(" in source, (
        f"{module} starts a run and never finishes one. missing_days counts "
        f"only finished runs, so an unfinished row is not coverage")


# --- and the rows are real, not just the calls ------------------------------

@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("NEMO_PIT_DB", str(tmp_path / "pit.db"))
    pit_store.init_schema()
    return pit_store


def test_a_score_run_is_on_the_record(store):
    from research import scoring

    scoring.score_orders(as_of="2026-03-03", horizon_days=4)

    rows = [dict(r) for r in _runs(store, "score")]
    assert rows, "research-score left no run_log row"
    assert rows[-1]["status"] in ("ok", "partial", "failed")
    assert rows[-1]["finished_at"], "the run was started and never finished"


def test_a_score_run_that_finds_nothing_is_still_on_the_record(store):
    """The case that matters most. An empty book and a job that never ran look
    identical in the data and must not look identical in the log."""
    from research import scoring

    scoring.score_orders(as_of="2026-03-03", horizon_days=4)

    rows = [dict(r) for r in _runs(store, "score")]
    assert len(rows) == 1
    assert rows[0]["rows_written"] == 0


def test_an_announce_run_is_on_the_record(store):
    from research import announcements

    announcements.backfill(tickers=[], as_of="2026-03-03")

    rows = [dict(r) for r in _runs(store, "announce")]
    assert rows, "research-announce left no run_log row"
    assert rows[-1]["finished_at"]


def test_a_seed_run_is_on_the_record(store):
    from research import seed_consensus

    seed_consensus.seed(tickers=[], as_of="2026-03-03")

    rows = [dict(r) for r in _runs(store, "seed")]
    assert rows, "research-seed left no run_log row"
    assert rows[-1]["finished_at"]


def test_a_failed_run_is_recorded_as_failed_not_left_open(store, monkeypatch):
    """A crashed job leaves a started row with no finish, and missing_days
    correctly refuses to count that as coverage. A job that catches its own
    error must not leave the same shape behind."""
    from research import announcements

    def boom(*a, **kw):
        raise RuntimeError("EDGAR said no")

    monkeypatch.setattr(announcements, "for_quarters", boom)

    announcements.backfill(tickers=["AAA"], as_of="2026-03-03")

    rows = [dict(r) for r in _runs(store, "announce")]
    assert rows[-1]["finished_at"], "the run was left open after a failure"


def test_missing_days_can_now_see_a_gap_in_each_of_them(store):
    """The property all of this exists for."""
    from research import scoring

    scoring.score_orders(as_of="2026-03-03", horizon_days=4)

    gaps = store.missing_days("score", "2026-03-02", "2026-03-04")

    assert "2026-03-03" not in gaps, "the day it ran is reported as a gap"
    assert "2026-03-04" in gaps, "a day it did not run is not reported"


def _runs(store, job):
    with store.connect() as conn:
        return conn.execute(
            "SELECT * FROM run_log WHERE job = ? ORDER BY run_id",
            (job,)).fetchall()
