"""One MCP endpoint in front of the five data servers.

A client that registers five servers holds five URLs, five tokens and five
entries that can each fail to connect on their own. The gateway answers at one
URL and forwards each call to the server that serves the tool. It holds no
data-source key: each server keeps only the keys it reads.

What has to hold is that the gateway changes nothing about the answers. Every
tool of every server is listed once, a call reaches the server that owns the
tool, a server's error stays an error, and a server that does not answer is
named rather than left out of the list.
"""
from __future__ import annotations

import socket
import threading
import time

import mcp.types as types
import pytest
import yaml
from mcp.server import Server

from testing.test_readme_counts import SERVERS, _server
from tools import gateway, mcp_http

TOKEN = "t" * 40


def _tool(name: str) -> types.Tool:
    return types.Tool(name=name, description=f"{name} tool",
                      inputSchema={"type": "object", "properties": {}})


def _text(result: types.CallToolResult) -> str:
    return "".join(getattr(block, "text", "") for block in result.content)


class FakeBackend:
    """A data server as the gateway sees it: a name, its tools, and calls."""

    def __init__(self, name, tools, answer=None, down=False):
        self.name = name
        self.tools = [_tool(t) for t in tools]
        self.answer = answer
        self.down = down
        self.calls = []

    async def list_tools(self):
        if self.down:
            raise ConnectionError("connection refused")
        return self.tools

    async def call_tool(self, tool, arguments):
        self.calls.append((tool, arguments))
        if self.answer is not None:
            return self.answer
        return types.CallToolResult(content=[
            types.TextContent(type="text", text=f"{self.name}:{tool}")])


async def _list(server):
    handler = server.request_handlers[types.ListToolsRequest]
    result = await handler(types.ListToolsRequest(method="tools/list"))
    return [t.name for t in result.root.tools]


async def _call(server, name, arguments=None):
    handler = server.request_handlers[types.CallToolRequest]
    result = await handler(types.CallToolRequest(
        method="tools/call",
        params=types.CallToolRequestParams(name=name,
                                           arguments=arguments or {})))
    return result.root


# --- the list -----------------------------------------------------------------

async def test_every_tool_of_every_backend_is_listed_once():
    server = gateway.build_server([FakeBackend("sec", ["a", "b"]),
                                   FakeBackend("fred", ["c"])])
    assert sorted(await _list(server)) == ["a", "b", "c"]


async def test_two_backends_serving_one_name_refuse_to_list():
    """Whichever answered second would silently win every call."""
    server = gateway.build_server([FakeBackend("sec", ["shared"]),
                                   FakeBackend("fred", ["shared"])])
    with pytest.raises(RuntimeError, match=r"shared.*sec.*fred"):
        await _list(server)


async def test_a_backend_that_does_not_answer_is_named_not_left_out():
    """Listing the other four would show a client 48 fewer tools and no
    reason, which reads as a smaller product rather than an outage."""
    server = gateway.build_server([FakeBackend("sec", ["a"], down=True),
                                   FakeBackend("fred", ["c"])])
    with pytest.raises(RuntimeError, match=r"sec"):
        await _list(server)


async def test_the_list_is_asked_again_after_a_backend_was_down():
    sec = FakeBackend("sec", ["a"], down=True)
    server = gateway.build_server([sec, FakeBackend("fred", ["c"])])
    with pytest.raises(RuntimeError):
        await _list(server)

    sec.down = False
    assert sorted(await _list(server)) == ["a", "c"]


# --- the calls ----------------------------------------------------------------

async def test_a_call_reaches_the_backend_that_serves_the_tool():
    sec, fred = FakeBackend("sec", ["a"]), FakeBackend("fred", ["c"])
    server = gateway.build_server([sec, fred])

    result = await _call(server, "c", {"series": "DGS10"})

    assert fred.calls == [("c", {"series": "DGS10"})]
    assert sec.calls == []
    assert _text(result) == "fred:c"
    assert not result.isError


async def test_a_backend_error_stays_an_error():
    failed = types.CallToolResult(isError=True, content=[
        types.TextContent(type="text", text="EDGAR does not map XYZ")])
    server = gateway.build_server([FakeBackend("sec", ["a"], answer=failed)])

    result = await _call(server, "a")

    assert result.isError
    assert "EDGAR does not map XYZ" in _text(result)


async def test_an_unknown_tool_is_an_error_that_says_so():
    server = gateway.build_server([FakeBackend("sec", ["a"])])
    result = await _call(server, "no_such_tool")
    assert result.isError
    assert "no_such_tool" in _text(result)


async def test_a_call_to_a_backend_that_is_down_names_it():
    class Dead(FakeBackend):
        async def call_tool(self, tool, arguments):
            raise ConnectionError("connection refused")

    server = gateway.build_server([Dead("sec", ["a"])])
    result = await _call(server, "a")
    assert result.isError
    assert "sec" in _text(result)


# --- the backend list -----------------------------------------------------------

def test_backends_are_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("MCP_GATEWAY_BACKENDS",
                       "sec=http://sec:8080/mcp/, fred=http://fred:8080/mcp/")
    backends = gateway.resolve_backends(TOKEN)
    assert [(b.name, b.url) for b in backends] == [
        ("sec", "http://sec:8080/mcp/"), ("fred", "http://fred:8080/mcp/")]


@pytest.mark.parametrize("raw, complaint", [
    ("", "not set"),
    ("sec", "name=url"),
    ("sec=http://sec:8080/mcp/,sec=http://other:8080/mcp/", "twice"),
    ("sec=http://sec:8080/mcp", "trailing slash"),
])
def test_a_bad_backend_list_refuses_to_start(monkeypatch, raw, complaint):
    """A gateway that starts with no backends serves an empty list, which a
    client reads as a server with no tools."""
    monkeypatch.setenv("MCP_GATEWAY_BACKENDS", raw)
    with pytest.raises(RuntimeError, match=complaint):
        gateway.resolve_backends(TOKEN)


def test_the_compose_file_puts_every_data_server_behind_the_gateway():
    """A server added to the image and left out of this list would be
    reachable on its own port and missing from the one URL clients use."""
    compose = yaml.safe_load(
        (gateway.REPO / "deploy" / "docker-compose.yml").read_text())
    services = compose["services"]
    raw = services["nemo"]["environment"]["MCP_GATEWAY_BACKENDS"]
    pairs = dict(entry.split("=", 1) for entry in raw.split(","))

    assert sorted(pairs) == sorted(SERVERS)
    for name, url in pairs.items():
        assert url == f"http://{name}:8080/mcp/", (name, url)
        assert name in services, f"{name} is not a compose service"
    assert sorted(services["nemo"]["depends_on"]) == sorted(SERVERS)


def _copied_sources() -> list:
    """Every source path a COPY line in the Dockerfile names, from the repo."""
    sources = []
    for line in (gateway.REPO / "Dockerfile").read_text().splitlines():
        words = line.split()
        if not words or words[0] != "COPY" or any(
                w.startswith("--from") for w in words):
            continue
        sources += words[1:-1]
    return sources


def test_every_module_a_service_runs_is_copied_into_the_image():
    """The Dockerfile copies `tools/` file by file, so a new module run by a
    compose service is missing from the image until it is named there. The
    container then dies with ModuleNotFoundError, which no test that imports
    from the repository can see."""
    services = yaml.safe_load(
        (gateway.REPO / "deploy" / "docker-compose.yml").read_text())["services"]
    sources = _copied_sources()
    for name, block in services.items():
        words = block.get("entrypoint") or str(block.get("command", "")).split()
        if "-m" not in words:
            continue
        path = words[words.index("-m") + 1].replace(".", "/") + ".py"
        assert any(path == src or path.startswith(src.rstrip("/") + "/")
                   for src in sources), (
            f"{name} runs {path}, and no COPY line in the Dockerfile puts it "
            f"in the image")


# --- the five real servers ------------------------------------------------------

class InProcess:
    """A real server object, asked through its own request handlers."""

    def __init__(self, name):
        self.name = name
        self.server = _server(*SERVERS[name])

    async def list_tools(self):
        handler = self.server.request_handlers[types.ListToolsRequest]
        result = await handler(types.ListToolsRequest(method="tools/list"))
        return result.root.tools

    async def call_tool(self, tool, arguments):
        raise AssertionError("this test lists tools only")


async def test_the_gateway_lists_exactly_the_tools_of_the_five_servers():
    backends = [InProcess(name) for name in sorted(SERVERS)]
    expected = []
    for backend in backends:
        expected += [t.name for t in await backend.list_tools()]

    listed = await _list(gateway.build_server(backends))

    assert sorted(listed) == sorted(expected)
    assert len(listed) == len(set(listed))


# --- over HTTP ------------------------------------------------------------------

def _fake_data_server() -> Server:
    server = Server("fake")

    @server.list_tools()
    async def list_tools():
        return [_tool("echo"), _tool("fails")]

    @server.call_tool()
    async def call_tool(name, arguments):
        if name == "fails":
            raise ValueError("the vendor said no")
        return [types.TextContent(type="text", text=f"echo {arguments}")]

    return server


@pytest.fixture
def backend_url():
    """A data server on a real port, behind the same bearer check."""
    import uvicorn

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    app = mcp_http.build_app(_fake_data_server(), auth_token=TOKEN)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port,
                                           log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        assert time.monotonic() < deadline, "the fake server did not start"
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}/mcp/"
    server.should_exit = True
    thread.join(timeout=10)


async def test_a_call_goes_over_http_with_the_operator_token(backend_url):
    backend = gateway.Backend("fake", backend_url, TOKEN)
    server = gateway.build_server([backend])

    assert sorted(await _list(server)) == ["echo", "fails"]
    ok = await _call(server, "echo", {"ticker": "DMC"})
    assert not ok.isError
    assert "DMC" in _text(ok)

    failed = await _call(server, "fails")
    assert failed.isError
    assert "the vendor said no" in _text(failed)


async def test_a_backend_that_refuses_the_token_is_named(backend_url):
    server = gateway.build_server(
        [gateway.Backend("fake", backend_url, "w" * 40)])
    with pytest.raises(RuntimeError, match="fake"):
        await _list(server)
