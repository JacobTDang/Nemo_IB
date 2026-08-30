"""Readiness, as distinct from liveness.

/health proves a process is alive and bound to a port. That is all it can
prove, and it is the right shape for a container healthcheck -- restarting a
container is the only thing a liveness probe's answer can trigger.

It is the wrong thing to read as "this deployment works". A container whose
upstream provider is unreachable, whose API key never made it into the
environment, or which has never once answered a question, returns the same
green /health as a fully working one. An operator or an agent that reads that
green and concludes the deployment is fine is exactly the failure /ready
exists to prevent.

/ready reports the individual facts instead of a verdict with no evidence:
which credentials are absent, whether the MCP layer actually started, when
something last succeeded.

The status code answers a narrower question than the body: can this server do
its job at all. A missing declared credential or an MCP layer that never
started means no, and 503 is what a healthcheck can act on. "Nothing has
succeeded yet" means the container is thirty seconds old, and a probe that
called that unhealthy would call every server unhealthy every morning.
"""
import datetime as dt
import re

import pytest
from starlette.testclient import TestClient

from tools import mcp_http

TOKEN = "s3cret-token-value-long-enough-for-the-minimum"


def _server():
    """A real MCP server object. The session manager only stores it."""
    from mcp.server.lowlevel.server import Server
    return Server("readiness-test")


def _app(*, required_env=(), auth_token=None):
    return mcp_http.build_app(_server(), auth_token=auth_token,
                              required_env=required_env)


def _ready(*, required_env=(), auth_token=None, started=False):
    """The /ready body. `started` runs the lifespan, which is what starts the
    MCP session manager -- the difference between a bound port and a server
    that can answer a call."""
    app = _app(required_env=required_env, auth_token=auth_token)
    if started:
        with TestClient(app) as client:
            return client.get("/ready").json()
    return TestClient(app).get("/ready").json()


@pytest.fixture(autouse=True)
def _nothing_has_succeeded_yet(monkeypatch):
    """The last-success timestamp is module-level, so one test recording a
    success would otherwise decide the outcome of every test after it."""
    monkeypatch.setattr(mcp_http, "_LAST_SUCCESS", None)


# --------------------------------------------------------------- liveness kept

def test_health_still_returns_the_liveness_body_it_always_has():
    """/health is liveness and stays liveness: the process is up and the app
    started. The compose healthcheck moved to /ready, which is a different
    question; this body must not drift into answering it."""
    r = TestClient(_app()).get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "transport": "streamable-http",
                        "stateless": True}


# ------------------------------------------------------------------- reachable

def test_ready_needs_no_token_but_the_mcp_endpoint_still_does():
    """An orchestrator's readiness probe carries no credentials. A /ready
    behind the bearer token cannot be used by one, so it would be dead code.

    The /mcp assertion is what makes the /ready assertion mean anything:
    without it, a 200 could just as well be an app with auth switched off."""
    with TestClient(_app(auth_token=TOKEN)) as client:
        assert client.get("/ready").status_code == 200
        assert client.get("/mcp").status_code == 401


# ----------------------------------------------------------------- credentials

def test_absent_credentials_are_reported_by_name(monkeypatch):
    """"not ready" without naming the variable leaves an operator to guess.
    The name is the entire actionable content of this check."""
    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
    body = _ready(required_env=("FINNHUB_API_KEY",))
    assert body["checks"]["credentials"]["ok"] is False
    assert body["checks"]["credentials"]["missing"] == ["FINNHUB_API_KEY"]


def test_a_blank_credential_counts_as_absent(monkeypatch):
    """`FINNHUB_API_KEY=` in a .env file is how this goes wrong in practice.
    Present-but-empty fails every upstream call exactly as absent does."""
    monkeypatch.setenv("FINNHUB_API_KEY", "   ")
    assert _ready(required_env=("FINNHUB_API_KEY",))["checks"]["credentials"] \
        == {"ok": False, "missing": ["FINNHUB_API_KEY"]}


def test_a_server_that_needs_no_credentials_passes(monkeypatch):
    """FRED and SEC take no key. Declaring none must pass rather than being
    indistinguishable from "declared and missing"."""
    monkeypatch.setenv("FINNHUB_API_KEY", "present-but-not-declared")
    body = _ready(required_env=())
    assert body["checks"]["credentials"] == {"ok": True, "missing": []}


def test_a_configured_credential_passes(monkeypatch):
    monkeypatch.setenv("FINNHUB_API_KEY", "a-real-looking-key")
    body = _ready(required_env=("FINNHUB_API_KEY",))
    assert body["checks"]["credentials"] == {"ok": True, "missing": []}


# ---------------------------------------------------------------- last success

def test_a_server_that_has_never_succeeded_is_not_ready():
    """The failure this endpoint exists to prevent. A container that has never
    answered anything is not ready, however healthy its process looks."""
    body = _ready()
    assert body["checks"]["last_success"] == {
        "ok": False, "at": None, "age_seconds": None}
    assert body["ready"] is False
    assert "last_success" in body["degraded"]


def test_recording_a_success_timestamps_it():
    mcp_http.record_success()
    check = _ready()["checks"]["last_success"]
    assert check["ok"] is True
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", check["at"])
    assert 0 <= check["age_seconds"] < 10


def test_age_seconds_measures_how_long_ago(monkeypatch):
    """The age is the point: "last succeeded at 04:12" means nothing without
    knowing that was six hours ago."""
    monkeypatch.setattr(
        mcp_http, "_LAST_SUCCESS",
        dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=1))
    check = _ready()["checks"]["last_success"]
    assert check["at"].endswith("Z")
    assert 3595 <= check["age_seconds"] <= 3605


def test_a_clock_that_jumped_backwards_does_not_report_a_negative_age(
        monkeypatch):
    """NTP correcting a container's clock must not make the last success look
    like it happens in the future."""
    monkeypatch.setattr(
        mcp_http, "_LAST_SUCCESS",
        dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=1))
    assert _ready()["checks"]["last_success"]["age_seconds"] == 0.0


# ------------------------------------------------------------------- mcp layer

def test_the_mcp_check_fails_before_the_session_manager_starts():
    """uvicorn binds the port whether or not the MCP session manager came up.
    A server in that state answers /health and fails every single tool call,
    which is the exact gap /health cannot see."""
    assert _ready(started=False)["checks"]["mcp"]["ok"] is False


def test_the_mcp_check_passes_once_the_lifespan_has_run():
    assert _ready(started=True)["checks"]["mcp"]["ok"] is True


# ------------------------------------------------------------------ aggregation

def test_ready_is_the_and_of_every_check(monkeypatch):
    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
    mcp_http.record_success()

    everything_passes = _ready(started=True)
    assert everything_passes["ready"] is True
    assert everything_passes["degraded"] == []

    one_thing_fails = _ready(started=True, required_env=("FINNHUB_API_KEY",))
    assert one_thing_fails["ready"] is False


def test_every_failing_check_is_named_in_degraded(monkeypatch):
    """The list is what an operator reads first. A check that fails silently
    outside it is worse than no list at all."""
    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
    body = _ready(required_env=("FINNHUB_API_KEY",))
    assert sorted(body["degraded"]) == ["credentials", "last_success", "mcp"]
    assert body["checks"]["process"]["ok"] is True
    assert "process" not in body["degraded"]


def test_checked_at_is_utc_and_says_so():
    """A bare timestamp with no zone is read as local time by whoever is
    comparing it against their own logs."""
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z",
                        _ready()["checked_at"])


# ------------------------------------------------------------------- robustness

def test_a_check_that_raises_is_reported_rather_than_propagated(monkeypatch):
    """A readiness endpoint that 500s tells an operator nothing about which
    part is broken -- it looks identical to a crashed server."""
    def boom(_required_env):
        raise RuntimeError("environ is unreadable")

    monkeypatch.setattr(mcp_http, "_check_credentials", boom)
    r = TestClient(_app()).get("/ready")
    # 503, not 500: a check that blew up is a failed check, and the body still
    # names which one. A 500 looks identical to a crashed server.
    assert r.status_code == 503
    body = r.json()
    assert body["checks"]["credentials"]["ok"] is False
    assert "environ is unreadable" in body["checks"]["credentials"]["error"]
    assert body["ready"] is False
    assert "credentials" in body["degraded"]
    # The checks that did not raise still reported.
    assert body["checks"]["process"]["ok"] is True


# ----------------------------------------------------------------- status code

def test_a_missing_credential_answers_503():
    """The compose healthcheck reads this endpoint, and a status code is the
    only part of it a healthcheck can act on. An SEC server with no SEC_EMAIL
    came up green with all 48 tools and failed every EDGAR call at runtime."""
    with TestClient(_app(required_env=("DEFINITELY_NOT_SET_ANYWHERE",))) as c:
        r = c.get("/ready")
    assert r.status_code == 503
    assert r.json()["blocking"] == ["credentials"]
    assert r.json()["checks"]["credentials"]["missing"] == [
        "DEFINITELY_NOT_SET_ANYWHERE"]


def test_an_mcp_layer_that_never_started_answers_503():
    """uvicorn binds the port either way. This is the state /health cannot
    see and the reason the healthcheck moved."""
    r = TestClient(_app()).get("/ready")
    assert r.status_code == 503
    assert "mcp" in r.json()["blocking"]


def test_a_server_that_has_not_answered_anything_yet_is_still_200(monkeypatch):
    """The one degradation that must not fail the probe.

    Every container starts here, and a server nobody has queried today sits
    here too. Gating the status code on it would mark all five unhealthy every
    morning, which is a probe nobody reads by the time it means something."""
    monkeypatch.setenv("FINNHUB_API_KEY", "a-real-looking-key")
    app = _app(required_env=("FINNHUB_API_KEY",))
    with TestClient(app) as client:
        r = client.get("/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["blocking"] == []
    assert body["degraded"] == ["last_success"]
    assert body["ready"] is False


def test_a_fully_ready_server_answers_200(monkeypatch):
    monkeypatch.setenv("FINNHUB_API_KEY", "a-real-looking-key")
    mcp_http.record_success()
    app = _app(required_env=("FINNHUB_API_KEY",))
    with TestClient(app) as client:
        r = client.get("/ready")
    assert r.status_code == 200
    assert r.json() == {**r.json(), "ready": True, "degraded": [],
                        "blocking": []}


# ------------------------------------------------------------------ declaration

def test_run_http_declares_which_credentials_the_server_needs(monkeypatch):
    """The declaration is the server's own, not this module's -- only the
    server knows which keys its tools call. run_http is where it is stated."""
    monkeypatch.delenv("MCP_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("MCP_ALLOW_UNAUTHENTICATED", "1")
    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)

    served = {}
    monkeypatch.setattr(mcp_http.uvicorn, "run",
                        lambda app, **kwargs: served.update(app=app))

    mcp_http.run_http(_server(), required_env=("FINNHUB_API_KEY",))

    body = TestClient(served["app"]).get("/ready").json()
    assert body["checks"]["credentials"]["missing"] == ["FINNHUB_API_KEY"]


# --------------------------------------------------- what each server declares
#
# The endpoint above is only worth anything if a server names the keys its tools
# cannot work without. None of the five did: all called `run_http(X().server)`
# with the default `()`, so `credentials` reported ok on every deployment
# regardless. The realistic case was SEC_EMAIL -- nothing builds an SEC identity
# at startup, so the server came up green with all 48 tools and every EDGAR call
# failed at runtime.
#
# Read from the source rather than by importing: the declaration is an argument
# at a call site under `if __name__ == "__main__"`, which importing never runs.

import ast
import pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent

# What each server's tools genuinely cannot work without, and nothing else. A
# key with a working default (NAME falls back to "Investment Analyst") or one
# whose absence is a documented degradation (CONGRESS_API_KEY, FINMIND_TOKEN)
# must not be here: the compose healthcheck reads this endpoint now, so an
# over-declaration marks a working container unhealthy forever.
REQUIRED_ENV = {
    "tools/web_search_server/web_search.py": {"SEC_EMAIL"},
    "tools/news_agregator/finnhub_server.py": {"FINNHUB_API_KEY"},
    "tools/news_agregator/fred_server.py": {"FRED_API_KEY"},
    "tools/financial_modeling_engine/analysis_tools.py": set(),
    "tools/altdata_server/server.py": set(),
}


def _run_http_required_env(relative_path):
    """The literal names passed as `required_env=` at the run_http call site."""
    tree = ast.parse((REPO / relative_path).read_text())
    calls = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call)
             and getattr(node.func, "id", None) == "run_http"]
    assert len(calls) == 1, (
        f"{relative_path} has {len(calls)} run_http call sites; this check "
        f"reads one")
    for keyword in calls[0].keywords:
        if keyword.arg == "required_env":
            return {element.value for element in keyword.value.elts}
    return None


@pytest.mark.parametrize("path", sorted(REQUIRED_ENV))
def test_the_server_declares_the_credentials_its_tools_cannot_work_without(path):
    declared = _run_http_required_env(path)
    assert declared is not None, (
        f"{path} calls run_http without required_env, so /ready reports "
        f"credentials ok whatever the environment holds")
    assert declared == REQUIRED_ENV[path]
