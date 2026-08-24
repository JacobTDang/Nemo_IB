"""Bearer-token auth on the HTTP transport.

The servers are read-only market data, so a leaked token exposes queries rather
than money -- Alpaca is deliberately not in this image. The one real harm is
SEC identity misuse getting SEC_EMAIL rate-limited. That keeps this
proportionate: a token behind a private network, not an OAuth server for a
single user.

The design decision worth defending is that an unconfigured server REFUSES TO
START. Defaulting to open means one forgotten environment variable silently
publishes every tool to the network, and nothing about the running server would
look wrong.
"""
import pytest
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from tools.mcp_http import BearerAuthMiddleware, resolve_auth_token

TOKEN = "s3cret-token-value-long-enough-for-the-minimum"


def _client(token=TOKEN, exempt=("/health",)):
    async def ok(_request):
        return PlainTextResponse("ok")

    app = Starlette(routes=[Route("/health", ok), Route("/mcp", ok, methods=["GET", "POST"])])
    app.add_middleware(BearerAuthMiddleware, token=token, exempt_paths=exempt)
    return TestClient(app)


def test_correct_token_is_accepted():
    r = _client().get("/mcp", headers={"Authorization": f"Bearer {TOKEN}"})
    assert r.status_code == 200


def test_missing_header_is_rejected():
    r = _client().get("/mcp")
    assert r.status_code == 401


def test_wrong_token_is_rejected():
    r = _client().get("/mcp", headers={"Authorization": "Bearer not-the-token"})
    assert r.status_code == 401


def test_token_as_a_prefix_is_rejected():
    """Guards against a comparison that stops at the shorter string."""
    r = _client().get("/mcp", headers={"Authorization": f"Bearer {TOKEN[:6]}"})
    assert r.status_code == 401


def test_wrong_scheme_is_rejected():
    r = _client().get("/mcp", headers={"Authorization": f"Basic {TOKEN}"})
    assert r.status_code == 401


def test_health_is_exempt_so_the_container_healthcheck_works():
    """Compose probes /health from inside the container and has no token.
    /health reports liveness only, never data."""
    assert _client().get("/health").status_code == 200


def test_rejection_does_not_echo_the_expected_token():
    r = _client().get("/mcp", headers={"Authorization": "Bearer wrong"})
    assert TOKEN not in r.text


# ------------------------------------------------------------ startup posture

def test_absent_token_refuses_to_start(monkeypatch):
    """One forgotten variable must not silently publish every tool."""
    monkeypatch.delenv("MCP_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("MCP_ALLOW_UNAUTHENTICATED", raising=False)
    with pytest.raises(RuntimeError, match="MCP_AUTH_TOKEN"):
        resolve_auth_token()


def test_running_open_requires_an_explicit_opt_in(monkeypatch):
    """Deliberately unauthenticated is a legitimate choice behind a tunnel --
    it just has to be stated, not defaulted into."""
    monkeypatch.delenv("MCP_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("MCP_ALLOW_UNAUTHENTICATED", "1")
    assert resolve_auth_token() is None


def test_configured_token_is_returned(monkeypatch):
    monkeypatch.setenv("MCP_AUTH_TOKEN", TOKEN)
    assert resolve_auth_token() == TOKEN


def test_blank_token_counts_as_absent(monkeypatch):
    """An empty value is how a .env line goes wrong, not a valid secret."""
    monkeypatch.setenv("MCP_AUTH_TOKEN", "   ")
    monkeypatch.delenv("MCP_ALLOW_UNAUTHENTICATED", raising=False)
    with pytest.raises(RuntimeError):
        resolve_auth_token()


def test_short_token_is_refused(monkeypatch):
    """A guessable token is worse than none, because it looks like security."""
    monkeypatch.setenv("MCP_AUTH_TOKEN", "hunter2")
    with pytest.raises(RuntimeError, match="too short"):
        resolve_auth_token()
