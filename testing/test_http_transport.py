"""The data-source servers over streamable HTTP.

stdio requires the client to spawn the server, so these tools were unreachable
from anywhere but the spawning machine. These tests are the deploy gate.

/health is deliberately NOT the gate. A server whose MCP layer failed to start
still answers it -- the port is bound by uvicorn either way. Only a real
handshake proves the thing works.

Requires the compose stack:
    docker compose -f deploy/docker-compose.yml up -d
Skipped when it is not up, so the offline suite is unaffected.
"""
import asyncio
import os
import socket

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("SKIP_NETWORK_TESTS") == "1",
    reason="requires the compose stack")

SERVERS = {
    "sec":       (8810, "web_client", "get_revenue_base", {"ticker": "MSFT"}),
    "financial": (8811, "Financial_Analysis", "get_market_data", {"ticker": "NVDA"}),
    "finnhub":   (8812, "finnhub", "get_insider_transactions", {"ticker": "NVDA"}),
    "fred":      (8813, "fred", "get_treasury_yields", {}),
    "altdata":   (8814, "altdata", "get_capex_announcements", {"ticker": "TSM"}),
}


def _up(port: int) -> bool:
    try:
        with socket.create_connection(("localhost", port), timeout=1.5):
            return True
    except OSError:
        return False


def _require(port: int):
    if not _up(port):
        pytest.skip(f"nothing listening on {port}; bring up the compose stack")


async def _session(port: int):
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client
    return streamablehttp_client(f"http://localhost:{port}/mcp"), ClientSession


def _run(coro):
    return asyncio.run(coro)


@pytest.mark.parametrize("name", sorted(SERVERS))
def test_server_completes_a_handshake_and_a_real_call(name):
    port, expected_name, tool, args = SERVERS[name]
    _require(port)

    async def go():
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client
        async with streamablehttp_client(f"http://localhost:{port}/mcp") as (r, w, _):
            async with ClientSession(r, w) as s:
                init = await s.initialize()
                tools = await s.list_tools()
                result = await s.call_tool(tool, args)
                return init.serverInfo.name, len(tools.tools), result

    server_name, tool_count, result = _run(go())
    assert server_name == expected_name
    assert tool_count > 0
    assert result.content, f"{name}: {tool} returned no content"


def test_sec_server_gates_capabilities_over_http_too():
    """Capability gating is applied in list_tools, so it must survive the
    transport swap. Without SearXNG and the RAG stack the SEC server advertises
    39 of its 42 tools; advertising all 42 would mean a caller can invoke
    `search` and get an empty list with no error."""
    port = SERVERS["sec"][0]
    _require(port)

    async def go():
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client
        async with streamablehttp_client(f"http://localhost:{port}/mcp") as (r, w, _):
            async with ClientSession(r, w) as s:
                await s.initialize()
                return {t.name for t in (await s.list_tools()).tools}

    names = _run(go())
    assert "search" not in names, "search advertised without SearXNG"
    assert "rag_search" not in names and "rag_ingest" not in names
    assert "get_revenue_base" in names and "get_share_count_series" in names


def test_repeated_requests_leave_no_residue():
    """The property that replaces --rm.

    Until the transport swap, nothing accumulated because the container died
    with the session. A long-lived server does not get that for free; stateless
    mode is what replaces it, and this is the standing guard on it.
    """
    import subprocess
    port = SERVERS["fred"][0]
    _require(port)

    def snapshot():
        out = subprocess.run(
            ["docker", "exec", "nemo-fred", "sh", "-c",
             "du -s /app/db_cache /tmp /root 2>/dev/null | awk '{print $1}' | tr '\\n' ' ';"
             "ls /proc/1/fd 2>/dev/null | wc -l"],
            capture_output=True, text=True)
        if out.returncode != 0:
            pytest.skip("container nemo-fred not available for inspection")
        return out.stdout.split()

    before = snapshot()

    async def call_once():
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client
        async with streamablehttp_client(f"http://localhost:{port}/mcp") as (r, w, _):
            async with ClientSession(r, w) as s:
                await s.initialize()
                await s.call_tool("get_treasury_yields", {})

    for _ in range(8):
        _run(call_once())

    after = snapshot()
    assert before == after, (
        f"a long-lived container accumulated across 8 requests: "
        f"before={before} after={after}")
