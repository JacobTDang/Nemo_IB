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

from tools import mcp_http
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


# --- one token per agent, and a budget each (issue #108) ---------------------
#
# One shared token meant revoking one agent rotated every agent's token. A
# named token per agent can be withdrawn alone, and every request carries the
# name it came in under, for the log. And one runaway agent must not be able to
# spend the SEC identity's rate limit for the rest, so each agent gets a budget
# per minute; the operator's own token does not.

AGENT_A = "a" * 40
AGENT_B = "b" * 40


class _Clock:
    def __init__(self):
        self.now = 1_000.0

    def __call__(self):
        return self.now


def _fleet(agents, rate_limit=None, clock=None):
    seen = {}

    async def ok(request):
        seen["client"] = request.scope.get("state", {}).get("mcp_client")
        return PlainTextResponse("ok")

    app = Starlette(routes=[Route("/health", ok),
                            Route("/mcp", ok, methods=["GET", "POST"])])
    app.add_middleware(BearerAuthMiddleware, token=TOKEN, agents=agents,
                       rate_limit=rate_limit, clock=clock or _Clock())
    return TestClient(app), seen


def _get(client, token):
    return client.get("/mcp", headers={"Authorization": f"Bearer {token}"})


def test_each_agent_gets_in_under_its_own_name():
    client, seen = _fleet({"grok-1": AGENT_A, "grok-2": AGENT_B})

    assert _get(client, AGENT_B).status_code == 200
    assert seen["client"] == "grok-2"


def test_the_operator_token_still_works_and_is_named():
    client, seen = _fleet({"grok-1": AGENT_A})

    assert _get(client, TOKEN).status_code == 200
    assert seen["client"] == "operator"


def test_a_withdrawn_agent_is_refused_and_the_rest_are_not():
    client, _ = _fleet({"grok-2": AGENT_B})

    assert _get(client, AGENT_A).status_code == 401
    assert _get(client, AGENT_B).status_code == 200


def test_the_agent_list_is_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("MCP_AGENT_TOKENS", f"grok-1:{AGENT_A}, grok-2:{AGENT_B}")

    assert mcp_http.resolve_agent_tokens(TOKEN) == {"grok-1": AGENT_A,
                                                    "grok-2": AGENT_B}


def test_no_agent_list_is_no_agents(monkeypatch):
    monkeypatch.delenv("MCP_AGENT_TOKENS", raising=False)

    assert mcp_http.resolve_agent_tokens(TOKEN) == {}


@pytest.mark.parametrize("raw,needle", [
    (f"grok-1{AGENT_A}", "entry 1"),                 # no separator
    (f"grok-1:{AGENT_A},grok-2:short", "entry 2"),   # too short to be a secret
    (f"grok 1:{AGENT_A}", "name"),                   # a name the log cannot hold
    (f"grok-1:{AGENT_A},grok-1:{AGENT_B}", "twice"),  # one name, two tokens
    (f"grok-1:{AGENT_A},grok-2:{AGENT_A}", "reuses"),  # one token, two names
    (f"grok-1:{TOKEN}", "operator"),                 # an agent holding the key
])
def test_a_bad_agent_list_refuses_to_start_without_echoing_a_token(
        monkeypatch, raw, needle):
    monkeypatch.setenv("MCP_AGENT_TOKENS", raw)

    with pytest.raises(RuntimeError) as caught:
        mcp_http.resolve_agent_tokens(TOKEN)

    message = str(caught.value)
    assert needle in message
    for secret in (AGENT_A, AGENT_B, TOKEN):
        assert secret not in message


def test_an_agent_over_its_budget_is_told_when_to_come_back():
    clock = _Clock()
    client, _ = _fleet({"grok-1": AGENT_A, "grok-2": AGENT_B}, rate_limit=3,
                       clock=clock)

    assert [_get(client, AGENT_A).status_code for _ in range(3)] == [200] * 3
    clock.now += 20
    refused = _get(client, AGENT_A)

    assert refused.status_code == 429
    assert refused.headers["Retry-After"] == "40"
    assert _get(client, AGENT_B).status_code == 200, "one agent spent another's"


def test_the_budget_refills_as_the_minute_passes():
    clock = _Clock()
    client, _ = _fleet({"grok-1": AGENT_A}, rate_limit=2, clock=clock)
    _get(client, AGENT_A)
    _get(client, AGENT_A)

    clock.now += 60.5

    assert _get(client, AGENT_A).status_code == 200


def test_the_operator_has_no_budget():
    client, _ = _fleet({"grok-1": AGENT_A}, rate_limit=1)

    assert [_get(client, TOKEN).status_code for _ in range(5)] == [200] * 5


@pytest.mark.parametrize("raw,expected", [("", None), ("120", 120)])
def test_the_budget_is_read_from_the_environment(monkeypatch, raw, expected):
    monkeypatch.setenv("MCP_RATE_LIMIT_PER_MINUTE", raw)

    assert mcp_http.resolve_rate_limit() == expected


@pytest.mark.parametrize("raw", ["0", "-5", "fast"])
def test_a_nonsense_budget_refuses_to_start(monkeypatch, raw):
    monkeypatch.setenv("MCP_RATE_LIMIT_PER_MINUTE", raw)

    with pytest.raises(RuntimeError):
        mcp_http.resolve_rate_limit()


def test_the_server_app_wires_agents_and_budget_from_the_environment(
        monkeypatch):
    from mcp.server.lowlevel.server import Server
    monkeypatch.setenv("MCP_AUTH_TOKEN", TOKEN)
    monkeypatch.setenv("MCP_AGENT_TOKENS", f"grok-1:{AGENT_A}")
    monkeypatch.setenv("MCP_RATE_LIMIT_PER_MINUTE", "1")
    app = mcp_http.build_app(Server("fleet-test"))

    with TestClient(app) as client:
        first = client.get("/ready", headers={})
        a = client.post("/mcp/", headers={"Authorization": f"Bearer {AGENT_A}"})
        b = client.post("/mcp/", headers={"Authorization": f"Bearer {AGENT_A}"})

    assert first.status_code in (200, 503)
    assert a.status_code != 401, "the agent's token was not wired in"
    assert b.status_code == 429, "the budget was not wired in"
