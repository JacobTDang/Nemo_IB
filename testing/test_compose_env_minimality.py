"""What each container is handed, against what the code in it actually reads.

`env_file: ../.env` on every service gave all twelve containers the whole file.
Measured on `research-score` -- a weekly job that opens `pit.db`, reads filed
orders and writes a score -- eighteen keys arrived, among them
`ALPACA_LIVE_KEY` and `ALPACA_LIVE_SECRET`. Live broker credentials, in a
container that cannot place an order and has no reason to hold one. A key that
reaches a container it does not need is a key any traceback in that container
can print, and one more copy in `docker inspect`.

The check reads the compose file for what each service declares, then walks the
import graph from the module that service actually runs to find what its code
reads. Deliberately not a hand-written table of permitted keys: that is the
compose file written twice, and it would wave through every future addition to
both.

The direction is one-way on purpose. It asserts declared <= read, never the
reverse. A key the code reads and compose does not set is either a documented
degradation (`CONGRESS_API_KEY`, `FINMIND_TOKEN`) or a default (`NAME`); a key
compose sets and no code reads is a credential handed out for nothing.

Static, not imported. Importing twelve entry points to observe their reads would
run five server constructors and miss every key read inside a function that no
test calls. The AST sees `os.environ.get("X")` wherever it is written.
"""
from __future__ import annotations

import ast
import os
import pathlib
import re

import pytest
import yaml

REPO = pathlib.Path(__file__).resolve().parent.parent
COMPOSE = REPO / "deploy" / "docker-compose.yml"
DEPLOY_README = REPO / "deploy" / "README.md"
ENV_EXAMPLE = REPO / ".env.example"

# Packages that ship in the image. Anything else is a dependency, and a
# dependency reading an environment variable is not this file's business.
FIRST_PARTY = {"tools", "research", "agent", "state", "knowledge"}

# Keys set on a service ahead of the code that would read them.
#
# `tools/filing_cache.py` is the only module that reads either, and it was
# imported by `tools/mcp_http.py` -- the HTTP app -- and by nothing the batch
# jobs run. The jobs drive edgartools into the same 512MB `/root/.edgar` tmpfs
# that filled on the servers, so the cap was declared on them too, ahead of any
# code to honour it.
#
# `research-watch` and `research-announce` now do: both call
# `filing_cache.prune_if_due()` between names, so both have left this list. The
# three below still have no prune call, and until they get one these are
# configuration that nothing reads -- which is exactly what the rest of this
# file exists to forbid, so they are listed here by name rather than waved
# through by a pattern.
DECLARED_AHEAD_OF_THE_CODE = {
    "NEMO_FILING_CACHE_CAP_MB": {
        "congress-sync", "research-daily", "research-scan",
    },
    "NEMO_FILING_CACHE_INTERVAL_S": {
        "congress-sync", "research-daily", "research-scan",
    },
}

# Read by the platform rather than by any module here, so the import walk below
# will never find it and is not the right question to ask about it.
#
# `TZ` is what the C library consults when it turns a timestamp into a local
# one -- `datetime.date.today()`, `time.localtime()` and everything under them.
# It carries no credential and grants no access, so the reasoning that makes an
# unread key dangerous does not apply: an unread key is a secret handed out for
# nothing, and this one is a setting. It is checked by
# `test_every_service_pins_the_clock` instead, which is a stronger question --
# not "does something read it" but "is it the same value everywhere".
#
# Deliberately a named set of one rather than a pattern. The next key that
# wants in has to be argued for here.
READ_BY_THE_PLATFORM = {"TZ"}


# --------------------------------------------------------------- the compose file

@pytest.fixture(scope="module")
def compose() -> dict:
    """Loaded with the YAML merge keys resolved, which is how compose sees it."""
    return yaml.safe_load(COMPOSE.read_text())


@pytest.fixture(scope="module")
def services(compose) -> dict:
    return compose["services"]


def _env_file_names(entry) -> set:
    """The keys an `env_file:` hands over: every name in the file it names.

    Names only. This never reads a value, and the failure messages print names.
    """
    paths = [entry] if isinstance(entry, str) else list(entry or ())
    names: set = set()
    for path in paths:
        if isinstance(path, dict):          # {path: ..., required: ...}
            path = path.get("path", "")
        resolved = (COMPOSE.parent / str(path)).resolve()
        for candidate in (resolved, ENV_EXAMPLE):
            if candidate.exists():
                names |= set(re.findall(r"^\s*([A-Z][A-Z0-9_]*)=",
                                        candidate.read_text(), re.M))
    return names


def _declared(service: dict) -> set:
    """Every environment key this service's container receives."""
    names: set = set()
    env = service.get("environment") or {}
    if isinstance(env, dict):
        names |= set(env)
    else:                                    # ["KEY=value", "KEY"] form
        names |= {item.split("=", 1)[0] for item in env}
    if "env_file" in service:
        names |= _env_file_names(service["env_file"])
    return names


def _module_run_by(service: dict) -> str | None:
    """The `python -m <module>` a service runs, read off the compose file.

    Taken from the file rather than from a table here, so a service pointed at
    a different module is measured against the module it now runs.
    """
    words: list = []
    for key in ("entrypoint", "command"):
        value = service.get(key)
        if isinstance(value, str):
            words += value.split()
        elif isinstance(value, list):
            words += [str(item) for item in value]
    for index, word in enumerate(words):
        if word == "-m" and index + 1 < len(words):
            return words[index + 1]
    return None


# ------------------------------------------------------------- the import graph

def _module_file(module: str) -> pathlib.Path | None:
    direct = REPO.joinpath(*module.split(".")).with_suffix(".py")
    if direct.exists():
        return direct
    package = REPO.joinpath(*module.split("."), "__init__.py")
    return package if package.exists() else None


def _reads_in(tree: ast.AST) -> set:
    """`os.environ.get("X")`, `os.getenv("X")` and `os.environ["X"]`."""
    names: set = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and node.args:
            func = node.func
            looks_like_a_read = (
                isinstance(func, ast.Attribute)
                and (
                    (func.attr == "getenv"
                     and isinstance(func.value, ast.Name)
                     and func.value.id == "os")
                    or (func.attr == "get"
                        and isinstance(func.value, ast.Attribute)
                        and func.value.attr == "environ")
                )
            )
            first = node.args[0]
            if looks_like_a_read and isinstance(first, ast.Constant) \
                    and isinstance(first.value, str):
                names.add(first.value)
        elif isinstance(node, ast.Subscript):
            target = node.value
            if isinstance(target, ast.Attribute) and target.attr == "environ" \
                    and isinstance(node.slice, ast.Constant) \
                    and isinstance(node.slice.value, str):
                names.add(node.slice.value)
    return names


def _imports_in(tree: ast.AST, module: str) -> set:
    """Every module name this one imports, relative imports resolved.

    Walks the whole tree rather than the top level, because a package imported
    inside a function is still shipped and still runs.
    """
    found: set = set()
    package = module.split(".")[:-1]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package[:len(package) - (node.level - 1)] \
                    if node.level > 1 else package
                parts = base + ([node.module] if node.module else [])
            else:
                parts = (node.module or "").split(".")
            target = ".".join(part for part in parts if part)
            if not target:
                continue
            found.add(target)
            found |= {f"{target}.{alias.name}" for alias in node.names}
    return found


def _env_reachable_from(module: str) -> dict:
    """{env name: the module that reads it} over the first-party import graph."""
    reads: dict = {}
    seen: set = set()
    pending = [module]
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        path = _module_file(current)
        if path is None:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
        for name in _reads_in(tree):
            reads.setdefault(name, current)
        pending += [imported for imported in _imports_in(tree, current)
                    if imported.split(".")[0] in FIRST_PARTY
                    and imported not in seen]
    return reads


# ------------------------------------------------------------------------ tests

def test_no_service_is_handed_the_whole_env_file(services):
    """`env_file:` cannot be filtered -- it delivers every line in the file.

    One entry on one service is enough to put every credential in the file into
    that container, so the assertion is that there are none at all rather than
    that the risky ones are gone.
    """
    handed = sorted(name for name, service in services.items()
                    if "env_file" in service)
    assert not handed, (
        f"these services take an unfiltered env_file, so they receive every "
        f"key in it: {handed}")


@pytest.mark.parametrize("name", sorted(yaml.safe_load(
    COMPOSE.read_text())["services"]))
def test_a_service_declares_only_keys_its_own_code_reads(name, services):
    service = services[name]
    declared = _declared(service) - READ_BY_THE_PLATFORM
    if not declared:
        return
    module = _module_run_by(service)
    assert module, (
        f"`{name}` declares {sorted(declared)} and this file cannot tell which "
        f"module it runs, so nothing checks them")

    reads = _env_reachable_from(module)
    pending = {key for key, holders in DECLARED_AHEAD_OF_THE_CODE.items()
               if name in holders}
    unread = sorted(declared - set(reads) - pending)
    assert not unread, (
        f"`{name}` runs {module}, and nothing it imports reads {unread}. A key "
        f"a container does not need is one any traceback in it can print.")


def test_research_score_holds_no_broker_or_model_credential(services):
    """The measured case, named so a regression says which keys came back.

    `research-score` reads filed orders out of pit.db and writes a coefficient.
    It has no upstream, places no order and calls no model.
    """
    declared = _declared(services["research-score"])
    forbidden = {"ALPACA_LIVE_KEY", "ALPACA_LIVE_SECRET", "ALPACA_PAPER_KEY",
                 "ALPACA_PAPER_SECRET", "OPENROUTER_API_KEY", "OPENROUTER_GLM",
                 "OPENROUTER_NEMOTRON", "GROQ_API_KEY", "DISCORD_WEBHOOK_URL"}
    assert not declared & forbidden, (
        f"research-score reads pit.db and nothing else, and is handed "
        f"{sorted(declared & forbidden)}")


def test_a_key_declared_ahead_of_the_code_loses_its_exemption_once_read():
    """The exemption list has to shrink by itself.

    A pending key that the code now reads is an ordinary declaration, and
    leaving it listed here would hide the next one that is not.
    """
    services = yaml.safe_load(COMPOSE.read_text())["services"]
    for key, holders in DECLARED_AHEAD_OF_THE_CODE.items():
        for name in sorted(holders):
            module = _module_run_by(services[name])
            assert module, f"`{name}` runs no python -m module"
            assert key not in _env_reachable_from(module), (
                f"`{name}` now reads {key}; delete it from "
                f"DECLARED_AHEAD_OF_THE_CODE so the list keeps its meaning")


def test_every_operator_supplied_key_the_stack_hands_out_is_documented(services):
    """The deploy page names the secrets the stack needs. It has to name all of
    them.

    It claimed "`FINNHUB_API_KEY`, `FRED_API_KEY`, `SEC_EMAIL`, `NAME`. No model
    credentials of any kind" while `env_file` was handing every container
    `OPENROUTER_API_KEY` and `GROQ_API_KEY`. The claim was about the image,
    which was true, and was read as being about the containers, which was not.
    """
    supplied = set(re.findall(r"^\s*([A-Z][A-Z0-9_]*)=",
                              ENV_EXAMPLE.read_text(), re.M))
    handed = set()
    for service in services.values():
        handed |= _declared(service) & supplied

    page = DEPLOY_README.read_text()
    undocumented = sorted(key for key in handed if f"`{key}`" not in page)
    assert not undocumented, (
        f"the stack hands these to a container and deploy/README.md never "
        f"names them: {undocumented}")


# ------------------------------------------------------- the anchor merge trap

SERVERS = ("sec", "financial", "finnhub", "fred", "altdata")


@pytest.mark.parametrize("name", SERVERS)
def test_a_server_keeps_the_shared_settings_its_own_env_block_would_replace(
        name, services):
    """A YAML merge key is shallow, and this is where that bites.

    `<<: *server` merges one level. A service that then writes its own
    `environment:` replaces the anchor's outright rather than adding to it --
    so a per-service key list silently drops `MCP_AUTH_TOKEN`, and
    `resolve_auth_token` refuses to start the server. The nested `<<:` inside
    `environment:` is what keeps both.
    """
    environment = services[name].get("environment") or {}
    for shared in ("MCP_AUTH_TOKEN", "MCP_LOG_RESPONSES",
                   "NEMO_FILING_CACHE_CAP_MB"):
        assert shared in environment, (
            f"`{name}` lost {shared} to a shallow merge; its own "
            f"`environment:` replaced the anchor's instead of extending it")


# ------------------------------------------------------------ the healthcheck

@pytest.mark.parametrize("name", SERVERS)
def test_the_healthcheck_reads_readiness_rather_than_liveness(name, services):
    """/health answers "the port is bound", which a server with no SEC_EMAIL
    and 48 broken tools answers exactly as well as a working one."""
    probe = " ".join(services[name]["healthcheck"]["test"])
    assert "/ready" in probe, (
        f"`{name}` still health-checks {probe!r}, which cannot fail on a "
        f"missing credential")


def test_the_batch_services_write_nowhere_uncapped(services):
    """A `docker compose run` container's writable layer is host disk, and
    nothing bounds it. `research-score` was the one job without the mounts.
    """
    for name, service in services.items():
        if "ports" in service or _module_run_by(service) is None:
            continue
        mounts = {entry.split(":")[0] for entry in service.get("tmpfs") or ()}
        assert {"/root/.edgar", "/app/db_cache"} <= mounts, (
            f"`{name}` has no tmpfs for {sorted({'/root/.edgar', '/app/db_cache'} - mounts)}, "
            f"so it writes to the container's writable layer on host disk")


# ---------------------------------------------------------------- the clock
#
# Not a credential, and the only key in this file no first-party module reads:
# the C library reads TZ when it turns a timestamp into a local one. It is
# still the compose file's to declare, because nothing else here pins it.
#
# `research/daily_job._today()` is UTC unconditionally, and `python:3.12-slim`
# has no TZ set, so a container's local time is UTC by accident rather than by
# instruction. Issue #28 is what an unpinned clock costs: the cron line runs on
# the host in host-local time, so on an America/New_York box a 22:30 run is
# 02:30 UTC the next day, `as_of` is tomorrow, and every night records zero
# bars and reports `closed`. The code now refuses that rather than filing it,
# and this pins the half the code cannot reach.

def test_every_service_pins_the_clock(services):
    """One value, written out, not `${TZ:-UTC}`.

    An interpolated default is not a pin: it leaves the operator's environment
    able to move the container's clock away from the UTC the code assumes,
    which is the same class of mismatch as the host timezone.
    """
    unpinned = sorted(name for name, service in services.items()
                      if (service.get("environment") or {}).get("TZ") != "UTC")
    assert not unpinned, (
        f"these services do not pin TZ to UTC: {unpinned}. The recorder dates "
        f"its rows off a UTC clock; a container on another one dates them "
        f"wrong and says nothing")
