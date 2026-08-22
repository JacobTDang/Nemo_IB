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
import os
import pathlib
from typing import Any

import uvicorn
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route

DEFAULT_PORT = 8080
MCP_PATH = "/mcp"


def build_app(mcp_server: Any, *, stateless: bool = True,
              json_response: bool = False) -> Starlette:
    """Wrap an MCP server in an ASGI app exposing it at /mcp.

    `stateless` keeps no session state between requests. `json_response`
    returns plain JSON instead of an SSE stream, which is easier to debug with
    curl but gives up server-initiated messages.
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

    return Starlette(
        routes=[Route("/health", health), Mount(MCP_PATH, app=handle_mcp)],
        lifespan=lifespan,
    )


def run_http(mcp_server: Any, *, host: str | None = None,
             port: int | None = None, stateless: bool = True) -> None:
    """Serve until interrupted.

    Binds 0.0.0.0 by default because the process runs inside a container; the
    host controls exposure through its port publishing, not through this bind
    address.
    """
    host = host or os.environ.get("MCP_HTTP_HOST", "0.0.0.0")
    port = port or int(os.environ.get("MCP_HTTP_PORT", DEFAULT_PORT))
    uvicorn.run(build_app(mcp_server, stateless=stateless),
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
