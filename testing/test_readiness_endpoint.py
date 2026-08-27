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
something last succeeded. It always answers 200 -- see
test_a_not_ready_server_still_answers_200 for why a 503 would be worse.
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
    """The compose healthcheck probes /health and nothing else. Enriching it
    with readiness would make a container with a missing API key fail its
    healthcheck and get restarted forever, which fixes nothing."""
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
    client = TestClient(_app(auth_token=TOKEN))
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
    assert r.status_code == 200
    body = r.json()
    assert body["checks"]["credentials"]["ok"] is False
    assert "environ is unreadable" in body["checks"]["credentials"]["error"]
    assert body["ready"] is False
    assert "credentials" in body["degraded"]
    # The checks that did not raise still reported.
    assert body["checks"]["process"]["ok"] is True


def test_a_not_ready_server_still_answers_200():
    """Not 503. Several orchestrators kill a container on a failing readiness
    probe, and "degraded but still serving cached SEC data" is a state worth
    keeping alive. Readiness belongs in the body, where a reader can act on
    the detail instead of on a single status code."""
    r = TestClient(_app(required_env=("DEFINITELY_NOT_SET_ANYWHERE",))).get("/ready")
    assert r.status_code == 200
    assert r.json()["ready"] is False


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
