"""`nemo` -- what the deployment is doing, from wherever the operator stands.

The facts already existed and reaching them did not. `research.status` reads
the run log, `docker compose ps` knows which containers are up, and getting at
either meant remembering an incantation that only works from the deploy
directory, on the right host, with the right `--env-file`. So the answer to
"is this working" was two commands nobody could recall under pressure. This is
the front door: `nemo status`, `nemo services`, `nemo logs`, `nemo --monitor`.

Three decisions here go wrong silently, and each has a test that fails when
the rule is removed.

**Where the store is read from.** On the dev Mac `pit.db` is a file on disk and
reading it in process costs nothing. On the homelab it lives in the
`research-data` volume, which the host user cannot open at all, so the only way
to read it is `docker compose run --rm research-status`. Guessing wrong does
not fail -- it reports on an empty store, or on the wrong one -- so the choice
is made explicitly, `--via` overrides it, and going the long way round is said
on stderr rather than silently.

**A container that is not there.** `docker compose ps` omits it, so a table of
four rows where five belong reads as a healthy stack. A missing server is the
one fact the operator most needs, so it gets a row that says so and a non-zero
exit.

**A monitor that dies on one bad refresh.** A loop that raises out of a
collector leaves a stale screen up, which looks exactly like a working monitor.
A collector that fails prints its error in place and the loop goes on.

Everything reachable by the subprocess seam `run` and the `sleep`/`clock` seams
is injected, so the tests never start a container and production never reads a
fixture.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from research import pit_store, status

# From this module's own path, never from the working directory: `nemo` is
# typed from wherever the operator happens to be standing, which on the
# homelab is a home directory and on the Mac is whichever subdirectory the
# last command left them in.
ROOT = pathlib.Path(__file__).resolve().parent.parent
COMPOSE_FILE = ROOT / "deploy" / "docker-compose.yml"
ENV_FILE = ROOT / ".env"

# `pit_store` defaults `NEMO_PIT_DB` to a *relative* `db_cache/pit.db`, which
# is right for a job started from the repository root and wrong for a command
# typed from wherever the operator is standing. Unanchored, the same
# `nemo status` reads the store from the root and falls through to a container
# from a subdirectory -- two different stores, no error, no clue. The root is
# already known from this module's path, so the default is anchored to it.
DEFAULT_STORE = ROOT / pit_store._DEFAULT_DB

# `research-status` runs the same report inside the image, against the volume.
STATUS_SERVICE = "research-status"

_CLEAR_SCREEN = "\x1b[2J\x1b[3J\x1b[H"


class OperatorError(Exception):
    """Something the operator can fix, said in one sentence, exit 2.

    Distinct from a traceback on purpose. "docker is not on PATH" is an
    instruction; a `FileNotFoundError` from deep inside subprocess is a puzzle.
    """


# --- the two effects, and their real implementations -------------------------

def _run(cmd: Sequence[str], capture: bool = False) -> subprocess.CompletedProcess:
    """The subprocess seam.

    `capture` is the difference between a table this module formats and a
    stream the operator watches: `nemo logs -f` has to reach the terminal as it
    arrives, so its output is not captured and its exit code passes through.
    """
    return subprocess.run(list(cmd), capture_output=capture, text=capture)


def _which(name: str) -> Optional[str]:
    return shutil.which(name)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def store_path() -> str:
    """This host's store, as an absolute path."""
    return os.environ.get("NEMO_PIT_DB") or str(DEFAULT_STORE)


@contextlib.contextmanager
def _store_anchored():
    """`NEMO_PIT_DB` set to the anchored default for the length of a command.

    Set rather than passed, because `research.status` reads it through
    `pit_store.db_path()` and deciding on one path while reading another is
    the same bug one step further along. Restored on the way out so that a
    caller in the same process -- the test suite -- is left as it was found.
    """
    previous = os.environ.get("NEMO_PIT_DB")
    if previous is None:
        os.environ["NEMO_PIT_DB"] = str(DEFAULT_STORE)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("NEMO_PIT_DB", None)


# --- the compose file is the source of the service names ---------------------

_SERVICE_RE = re.compile(r"^  ([A-Za-z][A-Za-z0-9_-]*):\s*$")
_PROFILES_RE = re.compile(r"^    profiles:")


def _service_blocks() -> Dict[str, List[str]]:
    """Every service in the compose file, with its own lines.

    Read from the file rather than kept in a list here. A hand-kept list goes
    stale the first time a job is added, and then `nemo logs` refuses a service
    that exists -- which is worse than no check, because the refusal names the
    valid services and would be lying.
    """
    lines = COMPOSE_FILE.read_text(encoding="utf-8").splitlines()
    try:
        start = lines.index("services:")
    except ValueError:
        raise OperatorError(
            f"{COMPOSE_FILE} has no top-level `services:` block, so there is "
            f"no way to tell which services exist.")

    blocks: Dict[str, List[str]] = {}
    current: Optional[str] = None
    for line in lines[start + 1:]:
        if line and not line[0].isspace():
            break  # the next top-level key ends the services block
        match = _SERVICE_RE.match(line)
        if match:
            current = match.group(1)
            blocks[current] = []
        elif current is not None:
            blocks[current].append(line)
    if not blocks:
        raise OperatorError(f"{COMPOSE_FILE} declares no services.")
    return blocks


def compose_services() -> set:
    """Every service name, batch jobs included."""
    return set(_service_blocks())


def servers() -> List[str]:
    """The long-running services, in the order the compose file declares them.

    A service under a profile is a batch job: `docker compose up` starts none
    of them and every one is meant to exit. Only these five are held to
    running-and-healthy, because only these five are supposed to be up.
    """
    return [name for name, block in _service_blocks().items()
            if not any(_PROFILES_RE.match(line) for line in block)]


def compose_command(*args: str) -> List[str]:
    """Absolute paths on both flags, so the command works from any directory."""
    return ["docker", "compose", "--env-file", str(ENV_FILE),
            "-f", str(COMPOSE_FILE), *args]


def _require_docker() -> None:
    # `_which` is looked up here rather than bound as a default argument, so
    # a test can replace it and so a PATH that changes mid-process is read as
    # it is rather than as it was at import.
    if not _which("docker"):
        raise OperatorError(
            "docker is not on PATH, so there is no way to reach the stack "
            "from here. Run this on the host where the containers are.")


# --- status ------------------------------------------------------------------

def _status_flags(as_of: Optional[str], as_json: bool) -> List[str]:
    flags: List[str] = []
    if as_of:
        flags += ["--as-of", as_of]
    if as_json:
        flags.append("--json")
    return flags


def status_source(via: str) -> str:
    """`local` or `docker`, decided once and never guessed twice.

    `status._store_exists` is the readable test rather than a bare
    `os.path.exists`: a path that is there and unreadable, or there and not a
    database, would otherwise send the report into a traceback about a missing
    table, which reads as a broken install rather than a store somewhere else.
    """
    if via == "local":
        return "local"
    if via == "docker":
        _require_docker()
        return "docker"
    if status._store_exists():
        return "local"
    if _which("docker"):
        return "docker"
    raise OperatorError(
        f"there is no readable store at {store_path()} and docker is "
        f"not on PATH, so this host has no way to read the deployment's "
        f"store.")


def status_text(run: Callable[..., subprocess.CompletedProcess],
                as_of: Optional[str] = None, via: str = "auto") -> str:
    """The status screen as text, for the monitor to paint."""
    if status_source(via) == "local":
        return status._render(status.collect(as_of=as_of))
    done = run(compose_command("run", "--rm", STATUS_SERVICE,
                               *_status_flags(as_of, False)), capture=True)
    # 1 is the report saying something needs attention, which is a reading and
    # not a failure. Anything else is the container failing to run at all.
    if done.returncode not in (0, 1):
        raise OperatorError(
            f"`{STATUS_SERVICE}` exited {done.returncode}: "
            f"{(done.stderr or done.stdout or '').strip()}")
    return (done.stdout or "").rstrip("\n")


def cmd_status(args: argparse.Namespace,
               run: Callable[..., subprocess.CompletedProcess],
               out, err) -> int:
    source = status_source(args.via)
    flags = _status_flags(args.as_of, args.json)
    if source == "local":
        return status.main(flags)

    # Never silently. The two stores are different stores, and a reader who
    # thinks they are looking at this host's is reading the wrong one.
    reason = ("because --via docker was asked for" if args.via == "docker"
              else f"because there is no readable store at {store_path()} "
                   f"on this host")
    err.write(f"reading the store through `docker compose run --rm "
              f"{STATUS_SERVICE}` {reason}\n")
    return run(compose_command("run", "--rm", STATUS_SERVICE, *flags)).returncode


# --- services ----------------------------------------------------------------

def _parse_ps(text: str) -> List[Dict[str, Any]]:
    """`--format json` writes one object per line. Older compose wrote one
    array, and both are read rather than one being a traceback."""
    text = (text or "").strip()
    if not text:
        return []
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _ports(publishers: Optional[Iterable[Dict[str, Any]]]) -> str:
    parts = []
    for published in publishers or ():
        port = published.get("PublishedPort")
        if not port:
            continue
        host = published.get("URL") or ""
        target = published.get("TargetPort")
        protocol = published.get("Protocol") or "tcp"
        head = f"{host}:{port}" if host else str(port)
        parts.append(f"{head}->{target}/{protocol}")
    return ", ".join(parts) or "-"


def _table(rows: List[Tuple[str, ...]]) -> str:
    header = ("SERVICE", "STATE", "HEALTH", "PORTS", "STATUS")
    widths = [max(len(str(row[i])) for row in [header] + rows)
              for i in range(len(header))]
    lines = []
    for row in [header] + rows:
        cells = [str(cell).ljust(widths[i]) for i, cell in enumerate(row)]
        lines.append("  " + "  ".join(cells).rstrip())
    return "\n".join(lines)


def services_report(run: Callable[..., subprocess.CompletedProcess]
                    ) -> Tuple[str, int]:
    """The table, and 0 only when all five servers are running and healthy.

    `-a` is the whole point. Without it a batch job that has already exited is
    absent, and so is a server whose container died -- and a table missing the
    row you needed reads exactly like a table with nothing to report.
    """
    _require_docker()
    done = run(compose_command("ps", "-a", "--format", "json"), capture=True)
    if done.returncode != 0:
        # An empty table from a dead daemon reads as a stack with nothing
        # running, which is a different and much calmer fact than the truth.
        raise OperatorError(
            f"`docker compose ps` exited {done.returncode}: "
            f"{(done.stderr or done.stdout or '').strip()}")

    containers = _parse_ps(done.stdout)
    expected = servers()
    order = {name: index for index, name in enumerate(expected)}

    rows: List[Tuple[str, ...]] = []
    unhealthy: List[str] = []
    for container in sorted(
            containers,
            key=lambda c: (order.get(c.get("Service"), len(order)),
                           c.get("Service") or "", c.get("Name") or "")):
        service = container.get("Service") or "-"
        state = container.get("State") or "-"
        health = container.get("Health") or "-"
        rows.append((service, state, health, _ports(container.get("Publishers")),
                     container.get("Status") or "-"))
        if service in order and not (state == "running" and health == "healthy"):
            unhealthy.append(service)

    seen = {container.get("Service") for container in containers}
    for name in expected:
        if name not in seen:
            # The fact `ps` cannot report: there is no container at all.
            rows.append((name, "missing", "-", "-", "no container"))
            unhealthy.append(name)

    return _table(rows), (1 if unhealthy else 0)


def cmd_services(args: argparse.Namespace,
                 run: Callable[..., subprocess.CompletedProcess],
                 out, err) -> int:
    text, code = services_report(run)
    out.write(text + "\n")
    return code


# --- logs --------------------------------------------------------------------

def cmd_logs(args: argparse.Namespace,
             run: Callable[..., subprocess.CompletedProcess],
             out, err) -> int:
    known = sorted(compose_services())
    if args.service not in known:
        # Checked before docker is, so a typo is answered the same way whether
        # or not this host can reach the stack. `docker compose` answers `no
        # such service` and does not say what the services are.
        raise OperatorError(
            f"`{args.service}` is not a service in {COMPOSE_FILE.name}. "
            f"The services are: {', '.join(known)}.")
    _require_docker()

    cmd = compose_command("logs", "--tail", str(args.tail))
    if args.follow:
        cmd.append("-f")
    cmd.append(args.service)
    return run(cmd).returncode


# --- the monitor -------------------------------------------------------------

def monitor(every: int, collectors: Sequence[Callable[[], str]], out,
            sleep: Callable[[float], None],
            clock: Callable[[], datetime]) -> int:
    """Repaint every `every` seconds until Ctrl-C.

    A collector that raises prints its error where its screen would have been
    and the loop goes on. The alternative -- letting it out -- leaves the last
    good screen frozen on the terminal, which is indistinguishable from a
    monitor that is working, and that is the reading an operator acts on.
    """
    try:
        while True:
            out.write(_CLEAR_SCREEN)
            out.write(f"nemo monitor   "
                      f"{clock().strftime('%Y-%m-%d %H:%M:%S')}Z   "
                      f"refreshing every {every}s   ctrl-c to stop\n\n")
            for collect in collectors:
                try:
                    out.write(collect().rstrip("\n") + "\n")
                except KeyboardInterrupt:
                    raise
                except Exception as problem:
                    out.write(f"  this refresh failed: {problem}\n")
                out.write("\n")
            out.flush()
            sleep(every)
    except KeyboardInterrupt:
        out.write("\n")
        out.flush()
        return 0


def cmd_monitor(args: argparse.Namespace,
                run: Callable[..., subprocess.CompletedProcess],
                out, sleep, clock) -> int:
    collectors = (lambda: status_text(run),
                  lambda: services_report(run)[0])
    return monitor(args.every, collectors, out, sleep, clock)


# --- the command line --------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nemo",
        description="What the Nemo deployment is doing.",
        epilog="`nemo --monitor` is the same as `nemo monitor`.")
    commands = parser.add_subparsers(dest="command")

    report = commands.add_parser(
        "status", help="what every job last did, and what needs attention")
    report.add_argument("--as-of", dest="as_of", default=None,
                        help="report as of this date (default: today, UTC)")
    report.add_argument("--json", action="store_true",
                        help="machine-readable, for scripting")
    report.add_argument("--via", choices=("auto", "local", "docker"),
                        default="auto",
                        help="read the store on this host, or in a container "
                             "(default: whichever is possible)")

    commands.add_parser(
        "services", help="every container, including the jobs that have exited")

    logs = commands.add_parser("logs", help="the log of one service")
    logs.add_argument("service", help="a service name from the compose file")
    logs.add_argument("-n", "--tail", type=int, default=200,
                      help="how many lines (default: 200)")
    logs.add_argument("-f", "--follow", action="store_true",
                      help="keep following the log")

    watch = commands.add_parser(
        "monitor", help="the status screen and the services table, on a loop")
    watch.add_argument("--every", type=int, default=30,
                       help="seconds between refreshes (default: 30)")
    return parser


def _normalise(argv: List[str]) -> List[str]:
    """`nemo --monitor` is what the operator will type.

    argparse subcommands do not answer to a leading flag, and this one is the
    whole reason the command exists, so it is rewritten rather than refused.
    """
    if argv and argv[0] == "--monitor":
        return ["monitor"] + argv[1:]
    return argv


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    return build_parser().parse_args(_normalise(list(argv)))


def main(argv: Optional[Sequence[str]] = None, *, run=None, sleep=None,
         clock=None, out=None, err=None) -> int:
    """`nemo`, and `python -m research.cli`."""
    argv = list(sys.argv[1:] if argv is None else argv)
    run = run or _run
    sleep = sleep or time.sleep
    clock = clock or _utcnow
    out = out if out is not None else sys.stdout
    err = err if err is not None else sys.stderr

    parser = build_parser()
    args = parser.parse_args(_normalise(argv))
    if args.command is None:
        parser.print_help(err)
        return 2

    try:
        with _store_anchored():
            if args.command == "status":
                return cmd_status(args, run, out, err)
            if args.command == "services":
                return cmd_services(args, run, out, err)
            if args.command == "logs":
                return cmd_logs(args, run, out, err)
            return cmd_monitor(args, run, out, sleep, clock)
    except OperatorError as problem:
        err.write(f"{problem}\n")
        return 2


if __name__ == "__main__":  # pragma: no cover - exercised via main()
    raise SystemExit(main())
