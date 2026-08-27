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

import asyncio
import contextlib
import datetime as dt
import hmac
import os
import pathlib
import sys
from collections.abc import Sequence
from typing import Any

import uvicorn
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route

from tools import filing_cache

DEFAULT_PORT = 8080
MCP_PATH = "/mcp"
# What clients should register. Without the trailing slash every request
# costs a 307 first.
MCP_PATH_CANONICAL = "/mcp/"


_UNSET = object()


async def _prune_filing_cache_forever() -> None:
    """Evict least-recently-used filings for as long as the server runs.

    The cache is capped by a tmpfs and nothing ever removed from it, so the
    mount reached 100% in the deployment and every SEC read failed with
    `[Errno 28] No space left on device`. A cap without eviction is not a
    limit, it is a scheduled outage.

    Runs once at startup because a container restarted onto an already-full
    tmpfs would otherwise wait a full interval before it could serve anything.
    """
    while True:
        filing_cache.prune_and_log()
        await asyncio.sleep(filing_cache.interval_seconds())


def build_app(mcp_server: Any, *, stateless: bool = True,
              json_response: bool = False, auth_token: Any = _UNSET,
              required_env: Sequence[str] = ()) -> Starlette:
    """Wrap an MCP server in an ASGI app exposing it at /mcp.

    `stateless` keeps no session state between requests. `json_response`
    returns plain JSON instead of an SSE stream, which is easier to debug with
    curl but gives up server-initiated messages.

    `auth_token` defaults to whatever the environment configures, and
    resolve_auth_token refuses to run silently open. Pass None explicitly only
    in tests.

    `required_env` names the environment variables this server's tools cannot
    work without. Only the server knows them, so it declares them; /ready
    reports which are absent.
    """
    manager = StreamableHTTPSessionManager(
        app=mcp_server, stateless=stateless, json_response=json_response)

    async def handle_mcp(scope, receive, send):
        await manager.handle_request(scope, receive, send)

    async def health(_request):
        # Liveness only: the process is up and the app started. Deliberately
        # unchanged -- the compose healthcheck reads it, and a healthcheck that
        # fails on a missing API key restarts the container forever without
        # fixing anything. Readiness is /ready.
        return JSONResponse({"status": "ok", "transport": "streamable-http",
                             "stateless": stateless})

    async def ready(_request):
        # Always 200; the verdict is in the body. See _readiness_report.
        return JSONResponse(_readiness_report(manager, required_env))

    @contextlib.asynccontextmanager
    async def lifespan(_app):
        async with manager.run():
            janitor = asyncio.create_task(_prune_filing_cache_forever())
            try:
                yield
            finally:
                janitor.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await janitor

    app = Starlette(
        routes=[Route("/health", health), Route("/ready", ready),
                Mount(MCP_PATH, app=handle_mcp)],
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
             port: int | None = None, stateless: bool = True,
             required_env: Sequence[str] = ()) -> None:
    """Serve until interrupted.

    Binds 0.0.0.0 by default because the process runs inside a container; the
    host controls exposure through its port publishing, not through this bind
    address.

    `required_env` is this server's declaration of the environment variables
    its tools need; /ready reports any that are absent. A missing key is not a
    reason to refuse to start -- most of these servers still answer plenty of
    questions without one -- so it is reported rather than enforced.
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

    uvicorn.run(build_app(mcp_server, stateless=stateless, auth_token=token,
                          required_env=required_env),
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
# Readiness.
#
# /health answers "is this process alive", which is all a liveness probe can
# act on. It is routinely misread as "is this deployment working", and the two
# come apart badly: a container with no FINNHUB_API_KEY, or one whose MCP layer
# never started, or one that has never once answered a question, returns the
# same green /health as a working one.
#
# /ready reports the facts separately rather than a single verdict with no
# evidence behind it, because "not ready" without the reason leaves an operator
# guessing. It always returns 200 -- several orchestrators kill a container on
# a failing readiness probe, and "degraded but still serving cached SEC data"
# is a state worth keeping alive.
# ---------------------------------------------------------------------------

# When something last succeeded against an upstream. Module-level because the
# servers are stateless per request: nothing else outlives a request to hold
# it. None means nothing has ever succeeded, which is not the same as "it has
# been a while" and must not be reported as ready.
_LAST_SUCCESS: dt.datetime | None = None


def record_success(at: dt.datetime | None = None) -> None:
    """Note that a call to an upstream provider succeeded.

    Callers pass nothing; `at` exists so a caller that already timestamped the
    call does not record a second, slightly different time.
    """
    global _LAST_SUCCESS
    _LAST_SUCCESS = at or dt.datetime.now(dt.timezone.utc)


def _utc_iso(moment: dt.datetime) -> str:
    """ISO-8601 in UTC, with a Z rather than +00:00.

    A timestamp with no zone gets read as local time by whoever is comparing it
    against their own logs, which is how a five-minute-old success gets read as
    eight hours stale.
    """
    return (moment.astimezone(dt.timezone.utc)
            .replace(microsecond=0).isoformat().replace("+00:00", "Z"))


def _check_process() -> dict:
    """True by construction: this ran, so the process is serving requests.

    Reported anyway so the check list is the whole picture rather than only the
    parts that can fail.
    """
    return {"ok": True}


def _check_mcp(manager: Any) -> dict:
    """Did the MCP layer actually start?

    uvicorn binds the port whether or not the session manager came up. The
    manager gets its task group in the app's lifespan, and without it every
    single tool call fails -- while /health stays green. That gap is the reason
    this endpoint exists.
    """
    missing = object()
    task_group = getattr(manager, "_task_group", missing)
    if task_group is missing:
        # Reported, not swallowed: the probe can no longer tell, and silently
        # passing would restore exactly the false green this replaces.
        raise RuntimeError(
            "the MCP session manager no longer exposes _task_group; this "
            "readiness probe needs updating for this version of the mcp "
            "package")
    return {"ok": task_group is not None}


def _check_credentials(required_env: Sequence[str]) -> dict:
    """Which declared environment variables are absent or empty.

    Blank counts as absent: `FINNHUB_API_KEY=` in a .env file is how this goes
    wrong in practice, and it fails every upstream call exactly as an unset
    variable does. A server declaring nothing passes -- FRED and SEC take no
    key, and that must not look like "declared and missing".
    """
    missing = [name for name in required_env
               if not os.environ.get(name, "").strip()]
    return {"ok": not missing, "missing": missing}


def _check_last_success() -> dict:
    """When an upstream call last worked, and how long ago.

    The age is the point -- "last succeeded at 04:12" means nothing without
    knowing that was six hours ago.
    """
    if _LAST_SUCCESS is None:
        return {"ok": False, "at": None, "age_seconds": None}
    age = (dt.datetime.now(dt.timezone.utc) - _LAST_SUCCESS).total_seconds()
    # Floored at zero: NTP correcting a container's clock must not make the
    # last success look like it happens in the future.
    return {"ok": True, "at": _utc_iso(_LAST_SUCCESS),
            "age_seconds": round(max(age, 0.0), 3)}


def _run_check(check) -> dict:
    """Run one check, turning a raised exception into a reported failure.

    A readiness endpoint that 500s looks identical to a crashed server and
    tells an operator nothing about which part is broken. A check that blew up
    is a failed check, and its message is the most useful thing in the body.
    """
    try:
        return check()
    except Exception as exc:  # noqa: BLE001 -- reported, never propagated
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _readiness_report(manager: Any, required_env: Sequence[str]) -> dict:
    """The /ready body: every check, the failing ones named, and the AND."""
    checks = {
        "process": _run_check(_check_process),
        "mcp": _run_check(lambda: _check_mcp(manager)),
        "credentials": _run_check(lambda: _check_credentials(required_env)),
        "last_success": _run_check(_check_last_success),
    }
    degraded = [name for name, result in checks.items() if not result["ok"]]
    return {
        "ready": not degraded,
        "checks": checks,
        "degraded": degraded,
        "checked_at": _utc_iso(dt.datetime.now(dt.timezone.utc)),
    }


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

    /health and /ready are exempt: probes carry no credentials -- the container
    healthcheck runs inside the container and an orchestrator's readiness probe
    has no way to hold a token -- and both report the server's own state, never
    data. A readiness endpoint behind the token could not be used by the thing
    it exists for.
    """

    def __init__(self, app, token: str,
                 exempt_paths: tuple = ("/health", "/ready")):
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
