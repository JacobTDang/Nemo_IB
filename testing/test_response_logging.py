"""Optional logging of what the server actually returned.

By default a request shows up as `POST /mcp/ 200 OK` and nothing about the
payload, which is useless when a tool is returning something surprising and you
have shelled into the box to find out why.

Off by default on purpose. Container logs are written to the host disk by
Docker's json-file driver, outside the tmpfs that protects everything else, so
logging every payload unconditionally would reintroduce exactly the unbounded
growth the tmpfs mounts exist to prevent.
"""
import json

import pytest
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from tools.mcp_http import ResponseLoggingMiddleware, resolve_response_logging


def _client(max_chars=4000):
    async def payload(_request):
        return JSONResponse({"ticker": "MSFT", "revenue_base": 331839000000.0})

    app = Starlette(routes=[Route("/mcp", payload, methods=["GET", "POST"]),
                            Route("/health", payload)])
    app.add_middleware(ResponseLoggingMiddleware, max_chars=max_chars,
                       exempt_paths=("/health",))
    return TestClient(app)


def test_response_body_is_logged(capsys):
    _client().post("/mcp")
    err = capsys.readouterr().err
    assert "331839000000" in err, "the payload was not logged"


def test_long_payloads_are_truncated(capsys):
    """An MD&A extract runs to 80KB. Logging it whole floods the log and the
    host disk behind it."""
    _client(max_chars=40).post("/mcp")
    err = capsys.readouterr().err
    assert "truncated" in err.lower()
    assert len(err) < 400


def test_health_is_not_logged(capsys):
    """The healthcheck fires every 30s forever; logging it is pure noise."""
    _client().get("/health")
    assert "331839000000" not in capsys.readouterr().err


def test_status_and_path_are_included(capsys):
    _client().post("/mcp")
    err = capsys.readouterr().err
    assert "/mcp" in err and "200" in err


def test_logging_is_off_unless_enabled(monkeypatch):
    monkeypatch.delenv("MCP_LOG_RESPONSES", raising=False)
    assert resolve_response_logging()[0] is False


def test_logging_can_be_enabled(monkeypatch):
    monkeypatch.setenv("MCP_LOG_RESPONSES", "1")
    enabled, _ = resolve_response_logging()
    assert enabled is True


def test_truncation_limit_is_configurable(monkeypatch):
    monkeypatch.setenv("MCP_LOG_RESPONSES", "1")
    monkeypatch.setenv("MCP_LOG_RESPONSE_CHARS", "120")
    assert resolve_response_logging() == (True, 120)


def test_a_bad_limit_falls_back_rather_than_crashing(monkeypatch):
    """A typo in an env var must not stop the server from starting."""
    monkeypatch.setenv("MCP_LOG_RESPONSES", "1")
    monkeypatch.setenv("MCP_LOG_RESPONSE_CHARS", "not-a-number")
    enabled, limit = resolve_response_logging()
    assert enabled is True and limit > 0
