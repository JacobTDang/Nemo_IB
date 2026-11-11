"""Each scheduled job posts from its own ``main()``, to the channel it should.

The unit tests of the messages cannot tell that a job forgot to call
``finish``. Here every ``main()`` runs with its work replaced by a canned
result and a recording transport in place of the HTTP call, so deleting a
``finish`` call, naming the wrong job, or sending to the wrong channel fails
a test by name. The conftest fixture sets both webhooks and replaces the HTTP
call for every other test.
"""
from __future__ import annotations

import sqlite3

import pytest

import research.discord_post as post

OPS = "https://hooks.example/ops"
SIGNALS = "https://hooks.example/signals"


class Recorder:
    """The transport: remembers (channel, payload) for every post."""

    def __init__(self):
        self.posts = []

    def __call__(self, url, body):
        channel = "ops" if url.startswith(OPS) else "signals"
        assert url.startswith((OPS, SIGNALS)), "a post went to an unknown URL"
        self.posts.append((channel, body))

    def texts(self, channel):
        found = []

        def walk(nodes):
            for node in nodes:
                if node.get("type") == post.TEXT:
                    found.append(node["content"])
                walk(node.get("components") or [])

        for name, body in self.posts:
            if name == channel:
                walk(body["components"])
        return "\n".join(found)

    @property
    def channels(self):
        return sorted({channel for channel, _ in self.posts})


@pytest.fixture
def store(tmp_path, monkeypatch):
    from research import pit_store

    monkeypatch.setenv("NEMO_PIT_DB", str(tmp_path / "pit.db"))
    pit_store.init_schema()
    return tmp_path


def _copy(monkeypatch, tmp_path, ok=True):
    from research import congress_copy

    monkeypatch.setattr(congress_copy, "enabled", lambda: False)
    monkeypatch.setattr(congress_copy, "run", lambda: {
        "status": "ok" if ok else "failed", "rows_written": 2, "accepted": 1,
        "error": "" if ok else "stale House sync",
        "preview": [{"ticker": "AAA", "member": "Ada", "dollars": 5000}]})
    return congress_copy.main()


def _scan(monkeypatch, tmp_path, ok=True):
    from research import scanner

    if not ok:
        def boom(**kwargs):
            raise RuntimeError("tape missing")

        monkeypatch.setattr(scanner, "record_scan", boom)
    else:
        monkeypatch.setattr(scanner, "record_scan", lambda **kwargs: {
            "written": 1, "rejected": [], "candidates": [
                {"ticker": "AAA", "side": "long", "sue": 1.2}]})
    return scanner.main([])


def _watch(monkeypatch, tmp_path, ok=True):
    from research import activist_watch

    result = {"status": "partial", "new_events": 1, "events": [{
        "subject_ticker": "INTC", "form": "SC 13D", "filer_name": "Example LP",
        "url": "https://www.sec.gov/Archives/edgar/data/1/x.htm"}]}
    if not ok:
        result = {"status": "failed", "new_events": 0, "events": [],
                  "error": "no eligible universe"}
    monkeypatch.setattr(activist_watch, "watch_pass", lambda **kwargs: dict(result))
    monkeypatch.setattr(activist_watch, "latency_report", lambda **kwargs: {})
    return activist_watch.main([])


def _score(monkeypatch, tmp_path, ok=True):
    from research import scoring

    monkeypatch.setattr(scoring, "score_orders", lambda **kwargs: {
        "sample": 40, "hit_rate": 0.55, "coefficient_usable": True,
        "drift_bps_per_sue": 12.0})
    return scoring.main([])


def _announce(monkeypatch, tmp_path, ok=True):
    from research import announcements

    monkeypatch.setattr(announcements, "backfill", lambda *a, **k: {
        "written": 3, "failed": [], "tickers": 10})
    return announcements.main([])


def _fills(monkeypatch, tmp_path, ok=True):
    from tools.alpaca import fills

    class Broker:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    async def fake_submit(broker, as_of, horizon_days):
        return {"status": "ok", "submitted": 2, "skipped": 0, "rejected": 0,
                "missed": 0, "notes": [],
                "copy": {"status": "ok", "submitted": 0, "skipped": 0,
                         "rejected": 0, "missed": 0}}

    monkeypatch.setattr(fills, "AsyncBroker", Broker)
    monkeypatch.setattr(fills, "submit", fake_submit)
    return fills.main(["submit"])


def _portfolio(monkeypatch, tmp_path, ok=True):
    from tools.alpaca import portfolio

    async def fake_read():
        return {"status": "ok", "equity": 1000.0, "last_equity": 990.0,
                "cash": 1000.0, "positions": [],
                "books": {"Earnings": {"open": 0, "closed": 0,
                                       "mean_return": None}}}

    monkeypatch.setattr(portfolio, "read", fake_read)
    return portfolio.main()


def _daily(monkeypatch, tmp_path, ok=True):
    from research import daily_job

    monkeypatch.setattr(daily_job, "run_all", lambda **kwargs: {
        "as_of": "2026-11-06", "daily_bars": {"status": "ok", "written": 300},
        "gaps": []})
    return daily_job.main([])


def _seed(monkeypatch, tmp_path, ok=True):
    from research import seed_consensus

    monkeypatch.setattr(seed_consensus, "seed", lambda tickers=None: {
        "written": 80, "failed": [], "tickers": 20})
    return seed_consensus.main([])


def _sync(monkeypatch, tmp_path, ok=True):
    from tools.altdata_server import congress_store, congress_sync

    monkeypatch.setenv("NEMO_CONGRESS_DB", str(tmp_path / "congress.db"))
    congress_store.init_schema()
    seen = {"chamber": "house", "year": 2026, "filings_seen": 5,
            "already_held": 2, "filings_parsed": 3, "scans": 0, "errors": 0,
            "remaining": 0, "blocked": None, "complete": True}
    monkeypatch.setattr(congress_sync, "sync_house_ptrs", lambda year, **k: dict(seen))
    return congress_sync.main(["--house", "2026"])


def _backup(monkeypatch, tmp_path, ok=True):
    from research import backup_job

    source = tmp_path / "pit.db"
    sqlite3.connect(source).close()
    monkeypatch.setattr(backup_job, "SOURCE", source)
    monkeypatch.setattr(backup_job, "BACKUPS", tmp_path / "backups")
    return backup_job.main()


def _status(monkeypatch, tmp_path, ok=True):
    from research import status

    monkeypatch.setattr(status, "_store_exists", lambda: True)
    monkeypatch.setattr(status, "collect", lambda as_of=None: {
        "as_of": "2026-11-06", "jobs": [], "store": {}, "book": {},
        "risk": {}, "gate": None, "copy": {}, "attention": []})
    monkeypatch.setattr(status, "_render", lambda report: "ok")
    return status.main([])


# job -> (scenario, channels it posts to, ops text that names the job)
JOBS = {
    "research-copy": (_copy, ["ops", "signals"]),
    "research-scan": (_scan, ["ops", "signals"]),
    "research-watch": (_watch, ["signals"]),
    "research-score": (_score, ["ops", "signals"]),
    "research-announce": (_announce, ["ops", "signals"]),
    "research-fills": (_fills, ["ops", "signals"]),
    "research-portfolio": (_portfolio, ["signals"]),
    "research-daily": (_daily, ["ops"]),
    "research-seed": (_seed, ["ops"]),
    "congress-sync": (_sync, ["ops"]),
    "research-backup": (_backup, ["ops"]),
    "research-status": (_status, ["ops"]),
}


def test_every_job_in_the_table_is_a_job_the_messages_know():
    assert set(JOBS) == set(post.JOB_CHANNELS) - {"docker-prune"}


@pytest.mark.parametrize("job", sorted(JOBS))
def test_a_job_posts_its_own_name_to_the_right_channels(
        job, store, monkeypatch, tmp_path):
    scenario, channels = JOBS[job]
    recorder = Recorder()
    monkeypatch.setattr(post, "_http_send", recorder)
    assert scenario(monkeypatch, tmp_path) == 0
    assert recorder.channels == channels
    for channel in channels:
        assert f"nemo {job}" in recorder.texts(channel)


# The watch posts only a change of state, and the portfolio card is a
# digest: neither posts to #ops when it succeeds.
@pytest.mark.parametrize("job", sorted(set(JOBS) - {"research-watch",
                                                    "research-portfolio"}))
def test_a_job_shows_how_long_it_ran(job, store, monkeypatch, tmp_path):
    scenario, _ = JOBS[job]
    recorder = Recorder()
    monkeypatch.setattr(post, "_http_send", recorder)
    scenario(monkeypatch, tmp_path)
    assert "**Duration**" in recorder.texts("ops")


@pytest.mark.parametrize("job", sorted(JOBS))
def test_a_job_that_cannot_post_exits_75_and_keeps_its_work(
        job, store, monkeypatch, tmp_path):
    scenario, _ = JOBS[job]

    def refuse(url, body):
        raise OSError("network is down")

    monkeypatch.setattr(post, "_http_send", refuse)
    assert scenario(monkeypatch, tmp_path) == post.POST_FAILED


@pytest.mark.parametrize("job", sorted(JOBS))
def test_a_job_with_no_webhook_set_exits_75_and_names_the_setting(
        job, store, monkeypatch, tmp_path, capsys):
    scenario, _ = JOBS[job]
    monkeypatch.delenv(post.OPS_SETTING)
    monkeypatch.delenv(post.SIGNALS_SETTING)
    assert scenario(monkeypatch, tmp_path) == post.POST_FAILED
    assert "NEMO_DISCORD_" in capsys.readouterr().err


FAILING = {"research-copy": _copy, "research-scan": _scan,
           "research-watch": _watch}


@pytest.mark.parametrize("job", sorted(FAILING))
def test_a_failed_job_posts_its_failure_and_keeps_a_nonzero_exit(
        job, store, monkeypatch, tmp_path, capsys):
    recorder = Recorder()
    monkeypatch.setattr(post, "_http_send", recorder)
    code = FAILING[job](monkeypatch, tmp_path, ok=False)
    assert code == 1
    assert "❌" in recorder.texts("ops")
    assert post.REPORTED in capsys.readouterr().err


@pytest.mark.parametrize("job", sorted(FAILING))
def test_a_failed_job_whose_post_fails_keeps_its_own_exit_code(
        job, store, monkeypatch, tmp_path, capsys):
    def refuse(url, body):
        raise OSError("network is down")

    monkeypatch.setattr(post, "_http_send", refuse)
    assert FAILING[job](monkeypatch, tmp_path, ok=False) == 1
    assert post.REPORTED not in capsys.readouterr().err


def test_congress_sync_reports_the_filings_parsed_in_this_run(
        store, monkeypatch, tmp_path):
    recorder = Recorder()
    monkeypatch.setattr(post, "_http_send", recorder)
    _sync(monkeypatch, tmp_path)
    assert "**3** filings parsed" in recorder.texts("ops")


def test_congress_sync_that_parsed_nothing_new_posts_nothing(
        store, monkeypatch, tmp_path):
    from tools.altdata_server import congress_store, congress_sync

    recorder = Recorder()
    monkeypatch.setattr(post, "_http_send", recorder)
    monkeypatch.setenv("NEMO_CONGRESS_DB", str(tmp_path / "congress.db"))
    congress_store.init_schema()
    monkeypatch.setattr(congress_sync, "sync_house_ptrs", lambda year, **k: {
        "filings_parsed": 0, "blocked": None})
    assert congress_sync.main(["--house", "2026"]) == 0
    assert recorder.posts == []


def test_a_failed_backup_posts_the_missing_source(store, monkeypatch, tmp_path,
                                                  capsys):
    from research import backup_job

    recorder = Recorder()
    monkeypatch.setattr(post, "_http_send", recorder)
    monkeypatch.setattr(backup_job, "SOURCE", tmp_path / "missing.db")
    monkeypatch.setattr(backup_job, "BACKUPS", tmp_path / "backups")
    assert backup_job.main() == 1
    assert "missing.db is not there" in recorder.texts("ops")


def test_the_backup_is_named_for_the_utc_date(store, monkeypatch, tmp_path):
    import datetime

    from research import backup_job

    class Clock(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            assert tz is datetime.timezone.utc, "the backup date must be UTC"
            return cls(2026, 11, 7, 23, 30, tzinfo=tz)

    source = tmp_path / "pit.db"
    sqlite3.connect(source).close()
    monkeypatch.setattr(backup_job, "SOURCE", source)
    monkeypatch.setattr(backup_job, "BACKUPS", tmp_path / "backups")
    monkeypatch.setattr(backup_job.datetime, "datetime", Clock)
    monkeypatch.setattr(post, "_http_send", Recorder())
    backup_job.main()
    assert [p.name for p in (tmp_path / "backups").iterdir()] == ["pit-2026-11-07.db"]


def test_a_crash_in_the_copy_job_logs_its_traceback_before_the_json_line(
        store, monkeypatch, tmp_path, capsys):
    from research import congress_copy

    def crash():
        raise KeyError("member_id")

    monkeypatch.setattr(congress_copy, "enabled", lambda: False)
    monkeypatch.setattr(congress_copy, "run", crash)
    recorder = Recorder()
    monkeypatch.setattr(post, "_http_send", recorder)
    assert congress_copy.main() == 1
    captured = capsys.readouterr()
    assert "Traceback (most recent call last)" in captured.err
    assert "KeyError" in captured.err and "crash" in captured.err
    assert '"status": "failed"' in captured.out
    assert "KeyError: 'member_id'" in recorder.texts("ops")
