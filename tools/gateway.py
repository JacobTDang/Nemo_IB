"""One MCP endpoint in front of the five data servers.

A client that registers the five servers holds five URLs, five copies of a
token and five entries that each fail to connect on their own. This answers at
one URL and forwards each call to the server that serves the tool, over the
compose network.

It is a proxy, not a merge into one process. Each data server is handed only
the keys its own code reads (testing/test_compose_env_minimality.py), and one
process serving all 96 tools would hold every key at once. The gateway holds
none: it reaches the servers with the operator token and nothing else.

It changes nothing about the answers. A server's error result is passed back as
the error it is, and a server that does not answer is named in the error rather
than left out of the list -- a list four servers long reads to a client as a
smaller product, not as an outage.

Run with `python -m tools.gateway http`.
"""
from __future__ import annotations

import os
import pathlib
import sys
from datetime import timedelta
from typing import Any, Dict, List, Optional, Sequence

import httpx
import mcp.types as types
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.server import Server

REPO = pathlib.Path(__file__).resolve().parent.parent

# How long one forwarded call may take. The longest SEC extractions run for
# minutes on a large filing, and a gateway that gave up first would turn a
# slow answer into a failed one.
CALL_TIMEOUT = timedelta(minutes=10)


class Backend:
    """One data server, reached over streamable HTTP.

    A session per request. The servers run stateless, so a session carries
    nothing between calls, and holding one open would only add a connection
    that can go stale between two calls an hour apart.
    """

    def __init__(self, name: str, url: str, token: Optional[str]):
        self.name = name
        self.url = url
        self._headers = {"Authorization": f"Bearer {token}"} if token else {}

    async def _ask(self, ask):
        timeout = httpx.Timeout(30.0, read=CALL_TIMEOUT.total_seconds())
        async with httpx.AsyncClient(headers=self._headers,
                                     timeout=timeout) as client:
            async with streamable_http_client(
                    self.url, http_client=client) as (read, write, _):
                async with ClientSession(
                        read, write,
                        read_timeout_seconds=CALL_TIMEOUT) as session:
                    await session.initialize()
                    return await ask(session)

    async def list_tools(self) -> List[types.Tool]:
        result = await self._ask(lambda session: session.list_tools())
        return list(result.tools)

    async def call_tool(self, tool: str,
                        arguments: Dict[str, Any]) -> types.CallToolResult:
        return await self._ask(lambda session: session.call_tool(tool, arguments))


def resolve_backends(token: Optional[str]) -> List[Backend]:
    """The servers to forward to, from MCP_GATEWAY_BACKENDS.

    Written as "name=url,name=url" in the compose file. An empty or malformed
    list refuses to start: a gateway with no backends serves an empty tool
    list, which a client reads as a server that has no tools.
    """
    raw = os.environ.get("MCP_GATEWAY_BACKENDS", "").strip()
    if not raw:
        raise RuntimeError(
            "MCP_GATEWAY_BACKENDS is not set, so the gateway has no server to "
            "forward to. Set it to name=url pairs, for example "
            "sec=http://sec:8080/mcp/,fred=http://fred:8080/mcp/")
    backends: List[Backend] = []
    for position, entry in enumerate(raw.split(","), start=1):
        entry = entry.strip()
        if not entry:
            continue
        name, sep, url = entry.partition("=")
        name, url = name.strip(), url.strip()
        where = f"MCP_GATEWAY_BACKENDS entry {position}"
        if not sep or not name or not url:
            raise RuntimeError(f"{where} is not in the form name=url")
        if any(b.name == name for b in backends):
            raise RuntimeError(f"{where}: the name {name!r} appears twice")
        # Without the slash the server answers 307, and some HTTP clients drop
        # the Authorization header across the redirect.
        if not url.endswith("/"):
            raise RuntimeError(
                f"{where} ({name}): the URL needs its trailing slash, "
                f"{url}/ not {url}")
        backends.append(Backend(name, url, token))
    return backends


def build_server(backends: Sequence[Any]) -> Server:
    """An MCP server that lists every backend's tools and forwards each call.

    The list is read from every backend before it is answered, and kept once
    all of them have answered: a tool list is fixed for the life of an image,
    and the gateway is rebuilt with the same image as the servers.
    """
    server = Server("nemo")
    owners: Dict[str, Any] = {}
    listed: List[types.Tool] = []

    async def catalogue() -> List[types.Tool]:
        if listed:
            return listed
        tools: List[types.Tool] = []
        owner_of: Dict[str, Any] = {}
        for backend in backends:
            try:
                served = await backend.list_tools()
            except Exception as exc:  # noqa: BLE001 - named and re-raised
                raise RuntimeError(
                    f"the {backend.name} server did not list its tools "
                    f"({type(exc).__name__}: {exc}), so the gateway lists "
                    f"nothing rather than a list without them") from exc
            for tool in served:
                if tool.name in owner_of:
                    raise RuntimeError(
                        f"{tool.name} is served by both "
                        f"{owner_of[tool.name].name} and {backend.name}; one "
                        f"of them would silently answer every call")
                owner_of[tool.name] = backend
                tools.append(tool)
        owners.update(owner_of)
        listed.extend(tools)
        return listed

    @server.list_tools()
    async def list_tools() -> List[types.Tool]:
        return await catalogue()

    # The backend validates its own input, against the schema it declared.
    @server.call_tool(validate_input=False)
    async def call_tool(name: str, arguments: Dict[str, Any]):
        await catalogue()
        backend = owners.get(name)
        if backend is None:
            raise ValueError(f"no server behind this gateway serves {name}")
        try:
            return await backend.call_tool(name, arguments or {})
        except Exception as exc:  # noqa: BLE001 - named and re-raised
            raise RuntimeError(
                f"the {backend.name} server did not answer {name} "
                f"({type(exc).__name__}: {exc})") from exc

    return server


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] != "http":
        print("Usage: python -m tools.gateway http", file=sys.stderr)
        sys.exit(1)

    from tools.mcp_http import resolve_auth_token, run_http

    print("Starting the gateway over streamable HTTP", file=sys.stderr,
          flush=True)
    run_http(build_server(resolve_backends(resolve_auth_token())))
