"""Serve an MCP server over streamable HTTP.

stdio requires the client to spawn the server as a child process, so a stdio
server on a remote host is unreachable by definition. This is what makes the
homelab image a server rather than a build artifact.

Written as a shared entrypoint rather than per-server plumbing: all five
data-source servers construct an `mcp.server.Server`, so they differ only in
which instance they hand over.

Stateless by default. In stateless mode the session manager keeps no
per-client state between requests, which matches how these tools work -- every
call is a self-contained question about the world -- and means a long-lived
container accumulates nothing across requests.
"""
from __future__ import annotations

import contextlib
import hmac
import os
import pathlib
import sys
from typing import Any

import uvicorn
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route

DEFAULT_PORT = 8080
MCP_PATH = "/mcp"
# What clients should register. Without the trailing slash every request
# costs a 307 first.
MCP_PATH_CANONICAL = "/mcp/"


_UNSET = object()


def build_app(mcp_server: Any, *, stateless: bool = True,
              json_response: bool = False, auth_token: Any = _UNSET) -> Starlette:
    """Wrap an MCP server in an ASGI app exposing it at /mcp.

    `stateless` keeps no session state between requests. `json_response`
    returns plain JSON instead of an SSE stream, which is easier to debug with
    curl but gives up server-initiated messages.

    `auth_token` defaults to whatever the environment configures, and
    resolve_auth_token refuses to run silently open. Pass None explicitly only
    in tests.
    """
    manager = StreamableHTTPSessionManager(
        app=mcp_server, stateless=stateless, json_response=json_response)

    async def handle_mcp(scope, receive, send):
        await manager.handle_request(scope, receive, send)

    async def health(_request):
        # Lets a container runtime tell "process alive" from "port bound but
        # the MCP layer never started".
        return JSONResponse({"status": "ok", "transport": "streamable-http",
                             "stateless": stateless})

    @contextlib.asynccontextmanager
    async def lifespan(_app):
        async with manager.run():
            yield

    app = Starlette(
        routes=[Route("/health", health), Mount(MCP_PATH, app=handle_mcp)],
        lifespan=lifespan,
    )

    # Mount only matches "/mcp/" exactly, so a client posting to "/mcp" gets a
    # 307 and pays an extra round trip on every call. Disabling redirects turns
    # that into a 404 instead, so the fix is on the client side: register the
    # URL with its trailing slash. MCP_PATH_CANONICAL is what the docs and the
    # compose healthcheck use.

    token = resolve_auth_token() if auth_token is _UNSET else auth_token
    if token:
        app.add_middleware(BearerAuthMiddleware, token=token)

    log_responses, log_chars = resolve_response_logging()
    if log_responses:
        app.add_middleware(ResponseLoggingMiddleware, max_chars=log_chars)
    return app


def run_http(mcp_server: Any, *, host: str | None = None,
             port: int | None = None, stateless: bool = True) -> None:
    """Serve until interrupted.

    Binds 0.0.0.0 by default because the process runs inside a container; the
    host controls exposure through its port publishing, not through this bind
    address.
    """
    host = host or os.environ.get("MCP_HTTP_HOST", "0.0.0.0")
    port = port or int(os.environ.get("MCP_HTTP_PORT", DEFAULT_PORT))
    # Resolve before binding, so a misconfigured server fails at startup
    # rather than after it is already accepting requests.
    token = resolve_auth_token()
    if token is None:
        print("[mcp_http] WARNING: running unauthenticated "
              "(MCP_ALLOW_UNAUTHENTICATED=1). Anything that reaches this port "
              "can call every tool.", file=sys.stderr, flush=True)

    uvicorn.run(build_app(mcp_server, stateless=stateless, auth_token=token),
                host=host, port=port, log_level="info")


@contextlib.contextmanager
def request_scratch(prefix: str = "mcp-req-"):
    """A temp directory for one request, removed on the way out.

    No tool in this image writes a file today -- none of the five servers even
    accepts a file path. This exists because the safety net that used to cover
    that case is gone: with stdio and --rm, anything a tool left behind died
    with the container. A long-lived HTTP server keeps it.

    Removal is in a `finally`, so it survives an exception. That is the case
    that matters -- a tool failing halfway through writing output is exactly
    how a server that never restarts fills up its disk.
    """
    import shutil
    import tempfile

    path = pathlib.Path(tempfile.mkdtemp(prefix=prefix))
    try:
        yield path
    finally:
        # ignore_errors so a tool that already cleaned up after itself does not
        # turn tidiness into a crash.
        shutil.rmtree(path, ignore_errors=True)


# ---------------------------------------------------------------------------
# Authentication.
#
# These servers are read-only market data -- Alpaca is deliberately not in this
# image -- so a leaked token exposes queries rather than money. The one real
# harm is SEC identity misuse getting SEC_EMAIL rate-limited. That keeps the
# design proportionate: a rotatable token behind a private overlay network,
# not an OAuth server for a single user.
#
# A bearer header is also the only mechanism every MCP client can use. OAuth
# needs client-side implementation and an interactive flow; a header needs
# neither, which is what "any agent could use it" actually requires.
# ---------------------------------------------------------------------------

MIN_TOKEN_LENGTH = 24


def resolve_auth_token() -> str | None:
    """The configured bearer token, or None when explicitly running open.

    Refuses to return quietly when nothing is configured. Defaulting to open
    means one forgotten environment variable silently publishes every tool to
    the network, and nothing about the running server would look wrong.
    Deliberately unauthenticated is a legitimate choice behind an SSH tunnel --
    it just has to be stated.
    """
    token = os.environ.get("MCP_AUTH_TOKEN", "").strip()
    if token:
        if len(token) < MIN_TOKEN_LENGTH:
            raise RuntimeError(
                f"MCP_AUTH_TOKEN is too short ({len(token)} chars, minimum "
                f"{MIN_TOKEN_LENGTH}). A guessable token is worse than none "
                f"because it looks like security. Generate one with: "
                f"openssl rand -hex 32")
        return token

    if os.environ.get("MCP_ALLOW_UNAUTHENTICATED") == "1":
        return None

    raise RuntimeError(
        "MCP_AUTH_TOKEN is not set. This server would otherwise accept any "
        "request that reaches its port. Set a token (openssl rand -hex 32), "
        "or set MCP_ALLOW_UNAUTHENTICATED=1 if it is genuinely reachable only "
        "through a tunnel you control.")


class BearerAuthMiddleware:
    """Reject requests without a matching bearer token.

    /health is exempt: the container healthcheck probes it from inside with no
    token, and it reports liveness only -- never data.
    """

    def __init__(self, app, token: str, exempt_paths: tuple = ("/health",)):
        self.app = app
        self._token = token
        self._exempt = tuple(exempt_paths)

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http" or scope.get("path", "") in self._exempt:
            await self.app(scope, receive, send)
            return

        supplied = ""
        for name, value in scope.get("headers", []):
            if name == b"authorization":
                raw = value.decode("latin-1")
                scheme, _, rest = raw.partition(" ")
                if scheme.lower() == "bearer":
                    supplied = rest.strip()
                break

        # compare_digest so a wrong token cannot be discovered a character at a
        # time from response timing.
        if not supplied or not hmac.compare_digest(supplied, self._token):
            response = JSONResponse(
                {"error": "unauthorized",
                 "detail": "A valid bearer token is required."},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"})
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)


# ---------------------------------------------------------------------------
# Response logging.
#
# By default a request appears as `POST /mcp/ 200 OK` and nothing about the
# payload, which is useless when you have shelled into the box because a tool
# is returning something surprising.
#
# Off unless asked for. Docker's json-file driver writes container logs to the
# HOST disk, outside the tmpfs that bounds everything else, so logging every
# payload unconditionally reintroduces exactly the unbounded growth those
# mounts exist to prevent. Pair this with max-size/max-file in compose.
# ---------------------------------------------------------------------------

DEFAULT_LOG_RESPONSE_CHARS = 4000


def resolve_response_logging() -> tuple:
    """(enabled, max_chars) from the environment.

    A malformed limit falls back to the default rather than refusing to start:
    a typo in an observability setting should not take the server down.
    """
    enabled = os.environ.get("MCP_LOG_RESPONSES") == "1"
    raw = os.environ.get("MCP_LOG_RESPONSE_CHARS", "")
    try:
        limit = int(raw) if raw.strip() else DEFAULT_LOG_RESPONSE_CHARS
        if limit <= 0:
            limit = DEFAULT_LOG_RESPONSE_CHARS
    except ValueError:
        limit = DEFAULT_LOG_RESPONSE_CHARS
    return enabled, limit


class ResponseLoggingMiddleware:
    """Echo response bodies to stderr so `docker logs` shows the payload.

    Bodies are truncated: an MD&A extract runs to 80KB and a filing-history
    response is larger still, so logging them whole floods the log and the host
    disk behind it. /health is exempt because the healthcheck fires every
    thirty seconds forever and its body carries nothing.
    """

    def __init__(self, app, max_chars: int = DEFAULT_LOG_RESPONSE_CHARS,
                 exempt_paths: tuple = ("/health",)):
        self.app = app
        self._max_chars = max_chars
        self._exempt = tuple(exempt_paths)

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http" or scope.get("path", "") in self._exempt:
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        status = {"code": 0}
        chunks: list = []
        captured = {"bytes": 0}

        async def capturing_send(message):
            if message["type"] == "http.response.start":
                status["code"] = message["status"]
            elif message["type"] == "http.response.body":
                body = message.get("body", b"")
                # Stop accumulating past the limit rather than buffering an
                # entire streamed response in memory to throw most of it away.
                if captured["bytes"] < self._max_chars * 2:
                    chunks.append(body)
                    captured["bytes"] += len(body)
            await send(message)

        await self.app(scope, receive, capturing_send)

        text = b"".join(chunks).decode("utf-8", errors="replace").strip()
        if len(text) > self._max_chars:
            text = (f"{text[:self._max_chars]}... "
                    f"[truncated, {captured['bytes']} bytes captured]")
        print(f"[mcp_http] {status['code']} {path} -> {text}",
              file=sys.stderr, flush=True)
