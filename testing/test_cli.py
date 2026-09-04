"""`nemo`, the one command an operator types to see what the deployment does.

The facts an operator needs at 2am are already produced -- `research.status`
reads the store, `docker compose ps` knows which containers are up -- and
reaching them meant remembering two different incantations, one of which only
works on the host and the other only from the deploy directory. This is the
front door.

What is worth testing here is not the formatting. It is the three decisions
that go wrong silently:

**Where the store is read from.** On the dev Mac the file is on disk. On the
homelab the store lives in a named volume the host user cannot open, so the
report has to run inside a container. Choosing wrong does not fail -- it
reports on the wrong store, or on an empty one -- so the choice is tested from
both sides, and the CLI says on stderr when it went the long way round.

**A server with no container at all.** `docker compose ps` without `-a` simply
omits it, and a table with four rows where five belong reads as fine. A
container that is missing is the single fact the operator most needs, so it
gets a row that says so and a non-zero exit.

**A monitor that dies on one bad refresh.** A loop that raises out of a
collector leaves the screen frozen on a stale reading, which is worse than no
monitor at all, so a failing collector prints in place and the loop goes on.
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import pytest

from research import cli, pit_store

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Captured verbatim from the local stack with
#   docker compose -f deploy/docker-compose.yml --env-file .env ps -a --format json
# and trimmed to the keys the CLI reads. The `research-status` line is a batch
# job that has already exited, which is the whole reason the CLI asks for `-a`:
# without it that line is simply absent and so is every other job.
PS_LINES = [
    {"Service": "research-status", "State": "exited", "Health": "",
     "Status": "Exited (1) 3 seconds ago", "ExitCode": 1, "Publishers": [],
     "Name": "deploy-research-status-run-7329830fa638"},
    {"Service": "altdata", "State": "running", "Health": "healthy",
     "Status": "Up 2 days (healthy)", "ExitCode": 0,
     "Publishers": [{"URL": "127.0.0.1", "TargetPort": 8080,
                     "PublishedPort": 8814, "Protocol": "tcp"}],
     "Name": "nemo-altdata"},
    {"Service": "financial", "State": "running", "Health": "healthy",
     "Status": "Up 2 days (healthy)", "ExitCode": 0,
     "Publishers": [{"URL": "127.0.0.1", "TargetPort": 8080,
                     "PublishedPort": 8811, "Protocol": "tcp"}],
     "Name": "nemo-financial"},
    {"Service": "finnhub", "State": "running", "Health": "healthy",
     "Status": "Up 2 days (healthy)", "ExitCode": 0,
     "Publishers": [{"URL": "127.0.0.1", "TargetPort": 8080,
                     "PublishedPort": 8812, "Protocol": "tcp"}],
     "Name": "nemo-finnhub"},
    {"Service": "fred", "State": "running", "Health": "healthy",
     "Status": "Up 2 days (healthy)", "ExitCode": 0,
     "Publishers": [{"URL": "127.0.0.1", "TargetPort": 8080,
                     "PublishedPort": 8813, "Protocol": "tcp"}],
     "Name": "nemo-fred"},
    {"Service": "sec", "State": "running", "Health": "healthy",
     "Status": "Up 2 days (healthy)", "ExitCode": 0,
     "Publishers": [{"URL": "127.0.0.1", "TargetPort": 8080,
                     "PublishedPort": 8810, "Protocol": "tcp"}],
     "Name": "nemo-sec"},
]


def _ps(rows):
    """What `--format json` actually writes: one object per line."""
    return "".join(json.dumps(row) + "\n" for row in rows)


def _done(returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


class FakeRun:
    """The subprocess seam. Records every command, answers from a queue."""

    def __init__(self, *results):
        self.calls = []
        self.results = list(results)

    def __call__(self, cmd, capture=False):
        self.calls.append(list(cmd))
        self.captured = capture
        return self.results.pop(0) if self.results else _done()

    @property
    def last(self):
        return self.calls[-1]


def never_run(cmd, capture=False):
    """The seam, wired to fail. Nothing in these tests may start a container."""
    raise AssertionError(f"the CLI shelled out: {cmd}")


@pytest.fixture
def no_store(tmp_path, monkeypatch):
    """A host with no readable store: the homelab, where it is in a volume."""
    monkeypatch.setenv("NEMO_PIT_DB", str(tmp_path / "absent.db"))
    return tmp_path / "absent.db"


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A host with the store on disk: the dev Mac."""
    monkeypatch.setenv("NEMO_PIT_DB", str(tmp_path / "pit.db"))
    pit_store.init_schema()
    return tmp_path / "pit.db"


@pytest.fixture
def docker(monkeypatch):
    monkeypatch.setattr(cli, "_which", lambda name: "/usr/local/bin/" + name)


@pytest.fixture
def no_docker(monkeypatch):
    monkeypatch.setattr(cli, "_which", lambda name: None)


# --- the flag the user will actually type ------------------------------------

def test_the_bare_monitor_flag_is_the_monitor_command():
    """`nemo --monitor` is what was asked for, and argparse subcommands do not
    answer to a leading flag. Without the rewrite it is an argument error."""
    assert cli.parse_args(["--monitor"]).command == "monitor"


def test_the_monitor_flag_still_takes_the_interval():
    args = cli.parse_args(["--monitor", "--every", "2"])
    assert (args.command, args.every) == ("monitor", 2)


def test_the_monitor_subcommand_takes_the_interval_too():
    args = cli.parse_args(["monitor", "--every", "5"])
    assert (args.command, args.every) == ("monitor", 5)


def test_the_monitor_refreshes_every_thirty_seconds_unless_told_otherwise():
    assert cli.parse_args(["monitor"]).every == 30


def test_no_subcommand_is_the_help_screen_and_a_non_zero_exit(capsys):
    assert cli.main([]) == 2
    assert "monitor" in capsys.readouterr().err


# --- where the status report is read from ------------------------------------

def test_the_store_on_disk_is_read_in_process(store, docker, monkeypatch):
    """The dev Mac. Starting a container to read a file that is right there
    costs seconds and needs an image that may not be built."""
    seen = []
    monkeypatch.setattr(cli.status, "main", lambda argv: seen.append(argv) or 0)
    run = FakeRun()

    assert cli.main(["status"], run=run) == 0
    assert seen == [[]], "the report did not run in this process"
    assert run.calls == [], f"it also shelled out: {run.calls}"


def test_no_store_on_this_host_means_the_report_runs_in_a_container(
        no_store, docker):
    """The homelab. The store is in the `research-data` volume, which the host
    user cannot open, so the only way to read it is from inside."""
    run = FakeRun(_done(0))

    assert cli.main(["status"], run=run) == 0
    assert run.last == ["docker", "compose",
                        "--env-file", str(ROOT / ".env"),
                        "-f", str(ROOT / "deploy" / "docker-compose.yml"),
                        "run", "--rm", "research-status"]


def test_the_flags_reach_the_container(no_store, docker):
    """`--json` read from a container that was never given it is the same
    report in the wrong shape, which a script parses as an outage."""
    run = FakeRun(_done(0))
    cli.main(["status", "--as-of", "2026-01-02", "--json"], run=run)
    assert run.last[-4:] == ["research-status", "--as-of", "2026-01-02",
                             "--json"]


def test_the_container_exit_code_is_the_command_exit_code(no_store, docker):
    """`research-status` exits 1 when something needs attention. Swallowing
    that turns a cron healthcheck into a cron that never fires."""
    assert cli.main(["status"], run=FakeRun(_done(1))) == 1


def test_going_the_long_way_round_is_said_out_loud(no_store, docker, capsys):
    """A silent fallback here reports on a different store than the reader
    thinks, which is the one failure this command exists to prevent."""
    cli.main(["status"], run=FakeRun(_done(0)))
    said = capsys.readouterr().err.strip().splitlines()
    assert len(said) == 1, f"expected one line on stderr, got {said}"
    assert "docker" in said[0]


def test_via_docker_overrides_a_store_that_is_right_there(store, docker,
                                                          monkeypatch):
    monkeypatch.setattr(cli.status, "main",
                        lambda argv: pytest.fail("--via docker ran in process"))
    run = FakeRun(_done(0))
    assert cli.main(["status", "--via", "docker"], run=run) == 0
    assert "research-status" in run.last


def test_via_local_overrides_a_store_that_is_not_there(no_store, docker,
                                                       monkeypatch):
    seen = []
    monkeypatch.setattr(cli.status, "main", lambda argv: seen.append(argv) or 1)
    run = FakeRun()
    assert cli.main(["status", "--via", "local"], run=run) == 1
    assert seen == [[]] and run.calls == []


def test_neither_the_store_nor_docker_names_both_and_exits_two(
        no_store, no_docker, capsys):
    assert cli.main(["status"], run=never_run) == 2
    said = capsys.readouterr().err
    assert "docker" in said and str(no_store) in said


def test_looking_at_the_store_never_creates_one(no_store, docker):
    """`nemo status` is the first thing anybody types on a fresh volume. A
    front door that built a schema in order to report on it would answer "nine
    jobs have never run" against a database it had just invented.
    `testing/test_every_job_creates_the_store_it_opens.py` exempts this module
    on exactly that ground."""
    cli.main(["status"], run=FakeRun(_done(1)))
    assert not no_store.exists()


def test_a_fresh_host_is_told_there_is_no_store_yet(no_store, docker, capsys):
    """The other half: forced local with nothing there says so, rather than
    a traceback about a missing table -- which reads as a broken install."""
    assert cli.main(["status", "--via", "local"], run=never_run) == 1
    assert "no store" in capsys.readouterr().out
    assert not no_store.exists()


# --- what is running ---------------------------------------------------------

def test_the_table_asks_compose_for_every_container(docker):
    """Without `-a` an exited batch job is simply absent, and so is a server
    whose container died."""
    run = FakeRun(_done(0, _ps(PS_LINES)))
    cli.main(["services"], run=run)
    assert run.last == ["docker", "compose",
                        "--env-file", str(ROOT / ".env"),
                        "-f", str(ROOT / "deploy" / "docker-compose.yml"),
                        "ps", "-a", "--format", "json"]


def test_every_service_in_the_output_gets_a_row(docker, capsys):
    cli.main(["services"], run=FakeRun(_done(0, _ps(PS_LINES))))
    shown = capsys.readouterr().out
    for row in PS_LINES:
        assert row["Service"] in shown, f"{row['Service']} is not on the table"


def test_the_exited_batch_job_is_on_the_table_with_its_status(docker, capsys):
    cli.main(["services"], run=FakeRun(_done(0, _ps(PS_LINES))))
    shown = capsys.readouterr().out
    assert "research-status" in shown
    assert "exited" in shown


def test_the_published_port_is_on_the_table(docker, capsys):
    """`sec` answering on 8810 is how an operator checks a client config."""
    cli.main(["services"], run=FakeRun(_done(0, _ps(PS_LINES))))
    assert "8810" in capsys.readouterr().out


def test_five_healthy_servers_exit_zero(docker):
    assert cli.main(["services"], run=FakeRun(_done(0, _ps(PS_LINES)))) == 0


def test_a_server_with_no_container_gets_a_row_and_a_non_zero_exit(docker,
                                                                   capsys):
    """The fact the operator needs most. `ps` omits it, so the table has to
    put it back."""
    without_sec = [row for row in PS_LINES if row["Service"] != "sec"]
    code = cli.main(["services"], run=FakeRun(_done(0, _ps(without_sec))))
    shown = capsys.readouterr().out

    assert "sec" in shown, "a server with no container vanished from the table"
    assert code == 1, "a missing server is not a healthy stack"


def test_a_server_that_is_up_but_not_healthy_exits_non_zero(docker):
    """`Up` is not `healthy`. The healthcheck probes readiness, and a server
    that is listening and failing its probe serves nothing."""
    sick = [dict(row, Health="unhealthy") if row["Service"] == "fred" else row
            for row in PS_LINES]
    assert cli.main(["services"], run=FakeRun(_done(0, _ps(sick)))) == 1


def test_an_exited_batch_job_is_not_a_reason_to_fail(docker):
    """Every job under the `sync` profile is meant to exit. Only the five
    long-running servers are held to running-and-healthy."""
    assert cli.main(["services"], run=FakeRun(_done(0, _ps(PS_LINES)))) == 0


def test_a_compose_that_fails_is_reported_rather_than_read_as_empty(
        docker, capsys):
    """An empty table from a dead docker daemon reads exactly like a stack
    with nothing running."""
    run = FakeRun(_done(1, "", "Cannot connect to the Docker daemon"))
    assert cli.main(["services"], run=run) == 2
    assert "Cannot connect" in capsys.readouterr().err


# --- logs --------------------------------------------------------------------

def test_logs_builds_the_compose_command(docker):
    run = FakeRun(_done(0))
    cli.main(["logs", "sec"], run=run)
    assert run.last == ["docker", "compose",
                        "--env-file", str(ROOT / ".env"),
                        "-f", str(ROOT / "deploy" / "docker-compose.yml"),
                        "logs", "--tail", "200", "sec"]


def test_logs_takes_a_line_count(docker):
    run = FakeRun(_done(0))
    cli.main(["logs", "sec", "-n", "20"], run=run)
    assert "--tail" in run.last and run.last[run.last.index("--tail") + 1] == "20"


def test_logs_follows_when_asked(docker):
    """Checked at the end of the command rather than anywhere in it: `-f` is
    already there as the compose-file flag, so `"-f" in cmd` is true whether
    or not `--follow` was honoured. Mutation testing found this one passing
    with the rule deleted."""
    run = FakeRun(_done(0))
    cli.main(["logs", "sec", "-f"], run=run)
    assert run.last[-2:] == ["-f", "sec"]


def test_logs_does_not_follow_unless_asked(docker):
    """`-f` on a job that has exited waits forever on a container that will
    never say anything else."""
    run = FakeRun(_done(0))
    cli.main(["logs", "sec"], run=run)
    assert run.last[-3:] == ["--tail", "200", "sec"]


def test_logs_passes_the_exit_code_through(docker):
    assert cli.main(["logs", "sec"], run=FakeRun(_done(7))) == 7


def test_logs_refuses_a_name_that_is_not_a_service(docker, capsys):
    """A typo otherwise reaches docker, which answers `no such service` and
    does not say what the services are."""
    run = FakeRun()
    assert cli.main(["logs", "secc"], run=run) == 2
    said = capsys.readouterr().err
    assert "secc" in said
    assert "sec" in said and "research-daily" in said, (
        f"the refusal did not list the valid names: {said}")
    assert run.calls == [], "it shelled out with a name it had already refused"


# --- the compose file is the source of the names -----------------------------

def test_the_services_are_read_from_the_compose_file_not_from_a_list():
    """A hand-kept list goes stale the first time a job is added, and then the
    CLI refuses a service that exists."""
    import yaml

    declared = set(yaml.safe_load(
        (ROOT / "deploy" / "docker-compose.yml").read_text())["services"])
    assert cli.compose_services() == declared


def test_the_servers_are_the_services_that_are_not_batch_jobs():
    """Held to running-and-healthy. Everything under the `sync` profile is a
    batch job and is meant to exit."""
    import yaml

    services = yaml.safe_load(
        (ROOT / "deploy" / "docker-compose.yml").read_text())["services"]
    expected = {n for n, s in services.items() if "profiles" not in s}
    assert set(cli.servers()) == expected
    assert expected == {"sec", "financial", "finnhub", "fred", "altdata"}


def test_the_default_store_is_anchored_to_the_repository(monkeypatch):
    """`pit_store` defaults `NEMO_PIT_DB` to a relative `db_cache/pit.db`,
    which is a different file in every directory. Unanchored, the same
    `nemo status` reads this host's store from the repository root and falls
    through to a container from a subdirectory -- two stores, no error."""
    monkeypatch.delenv("NEMO_PIT_DB", raising=False)
    assert cli.store_path() == str(ROOT / pit_store._DEFAULT_DB)
    assert pathlib.Path(cli.store_path()).is_absolute()


def test_an_explicit_store_path_is_left_alone(monkeypatch, tmp_path):
    monkeypatch.setenv("NEMO_PIT_DB", str(tmp_path / "pit.db"))
    assert cli.store_path() == str(tmp_path / "pit.db")


def test_the_anchored_path_is_the_one_the_report_reads(monkeypatch, docker):
    """Deciding on one path and reading another is the same bug one step
    further along, and it is invisible: both answers are a valid report."""
    monkeypatch.delenv("NEMO_PIT_DB", raising=False)
    monkeypatch.setattr(cli.status, "_store_exists", lambda: True)
    seen = []
    monkeypatch.setattr(cli.status, "main",
                        lambda argv: seen.append(pit_store.db_path()) or 0)

    assert cli.main(["status"], run=never_run) == 0
    assert seen == [str(ROOT / pit_store._DEFAULT_DB)]


def test_the_environment_is_left_as_it_was_found(monkeypatch, docker):
    """A command that sets a variable and leaves it set changes what every
    later reader in the same process sees."""
    monkeypatch.delenv("NEMO_PIT_DB", raising=False)
    monkeypatch.setattr(cli.status, "_store_exists", lambda: True)
    monkeypatch.setattr(cli.status, "main", lambda argv: 0)

    cli.main(["status"], run=never_run)
    assert "NEMO_PIT_DB" not in os.environ


def test_the_paths_come_from_the_module_not_the_working_directory():
    """`nemo` is typed from wherever the operator happens to be standing."""
    assert cli.ROOT == ROOT
    assert cli.COMPOSE_FILE == ROOT / "deploy" / "docker-compose.yml"
    assert cli.ENV_FILE == ROOT / ".env"
    assert cli.COMPOSE_FILE.exists()


# --- docker itself -----------------------------------------------------------

def test_services_without_docker_says_so_and_exits_two(no_docker, capsys):
    assert cli.main(["services"], run=never_run) == 2
    assert "docker" in capsys.readouterr().err.lower()


def test_logs_without_docker_says_so_and_exits_two(no_docker, capsys):
    assert cli.main(["logs", "sec"], run=never_run) == 2
    assert "docker" in capsys.readouterr().err.lower()


# --- the monitor -------------------------------------------------------------

class Screen:
    def __init__(self):
        self.text = ""

    def write(self, chunk):
        self.text += chunk

    def flush(self):
        pass


class Stop(Exception):
    """Ends the loop the way Ctrl-C does not."""


def _sleeper(stop_after, error=Stop):
    calls = []

    def sleep(seconds):
        calls.append(seconds)
        if len(calls) >= stop_after:
            raise error()

    sleep.calls = calls
    return sleep


def test_every_refresh_re_runs_every_collector():
    """A monitor that renders once and then only sleeps is a screenshot."""
    counted = {"a": 0, "b": 0}

    def first():
        counted["a"] += 1
        return "FIRST"

    def second():
        counted["b"] += 1
        return "SECOND"

    screen = Screen()
    with pytest.raises(Stop):
        cli.monitor(2, (first, second), screen, _sleeper(2), cli._utcnow)

    assert counted == {"a": 2, "b": 2}


def test_the_interval_is_the_one_that_was_asked_for():
    sleep = _sleeper(2)
    with pytest.raises(Stop):
        cli.monitor(5, (lambda: "x",), Screen(), sleep, cli._utcnow)
    assert sleep.calls == [5, 5]


def test_a_collector_that_fails_leaves_its_message_and_the_loop_goes_on():
    """A refresh that raises out of the loop freezes the screen on a stale
    reading, which reads as a working monitor and is not one."""
    tries = []

    def flaky():
        tries.append(1)
        if len(tries) == 1:
            raise RuntimeError("docker daemon went away")
        return "RECOVERED"

    screen = Screen()
    with pytest.raises(Stop):
        cli.monitor(1, (flaky,), screen, _sleeper(2), cli._utcnow)

    assert "docker daemon went away" in screen.text
    assert "RECOVERED" in screen.text, "the loop stopped after one bad refresh"


def test_one_bad_collector_does_not_stop_the_other_one():
    screen = Screen()

    def broken():
        raise RuntimeError("no")

    with pytest.raises(Stop):
        cli.monitor(1, (broken, lambda: "STILL HERE"), screen, _sleeper(1),
                    cli._utcnow)
    assert "STILL HERE" in screen.text


def test_ctrl_c_ends_the_monitor_without_a_traceback():
    """Caught here rather than left to escape, so that removing the rule fails
    this test by name instead of interrupting the whole pytest session."""
    try:
        code = cli.monitor(1, (lambda: "x",), Screen(),
                           _sleeper(1, error=KeyboardInterrupt), cli._utcnow)
    except KeyboardInterrupt:
        pytest.fail("Ctrl-C came out of the monitor as a traceback")
    assert code == 0


def test_ctrl_c_during_a_collector_ends_it_too():
    """The window a refresh spends waiting on docker is most of the interval,
    so this is where Ctrl-C usually lands."""
    def interrupted():
        raise KeyboardInterrupt()

    try:
        code = cli.monitor(1, (interrupted,), Screen(), _sleeper(1),
                           cli._utcnow)
    except KeyboardInterrupt:
        pytest.fail("Ctrl-C in a collector came out as a traceback")
    assert code == 0


def test_the_header_carries_the_clock():
    """A monitor whose screen never changes is indistinguishable from one that
    has hung, so the time it last refreshed is on it."""
    from datetime import datetime, timezone

    screen = Screen()
    frozen = lambda: datetime(2026, 9, 4, 7, 8, 9, tzinfo=timezone.utc)
    with pytest.raises(Stop):
        cli.monitor(1, (lambda: "x",), screen, _sleeper(1), frozen)

    assert "2026-09-04" in screen.text and "07:08:09" in screen.text


def test_the_monitor_command_wires_both_collectors(store, docker, monkeypatch):
    """The two screens the operator asked for: what the jobs did, and what is
    running."""
    monkeypatch.setattr(cli.status, "collect", lambda as_of=None: {"x": 1})
    monkeypatch.setattr(cli.status, "_render", lambda report: "STATUS SCREEN")
    screen = Screen()

    with pytest.raises(Stop):
        cli.main(["--monitor", "--every", "1"],
                 run=FakeRun(_done(0, _ps(PS_LINES))),
                 sleep=_sleeper(1), out=screen)

    assert "STATUS SCREEN" in screen.text
    assert "SERVICE" in screen.text and "8810" in screen.text


# --- the command exists at all ------------------------------------------------

def test_pyproject_declares_the_console_script():
    """`pip install -e .` is what puts `nemo` on PATH."""
    text = (ROOT / "pyproject.toml").read_text()
    assert "[project.scripts]" in text
    assert 'nemo = "research.cli:main"' in text


def test_the_module_runs_as_a_module():
    """`python -m research.cli` has to work wherever the console script is not
    installed, which on a fresh host is every time."""
    done = subprocess.run([sys.executable, "-m", "research.cli", "--help"],
                          cwd=ROOT, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    for command in ("status", "services", "logs", "monitor"):
        assert command in done.stdout
