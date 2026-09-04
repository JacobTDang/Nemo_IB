"""A server can be built, and its registry read, without its credential.

The first CI run failed thirty-two tests on one cause: `FinnhubClient` and
`FredClient` read their key in the constructor, so every test that builds a
server to count its tools, read its manifest or check a label needed a real
key to run. The offline suite is documented as needing no credentials, and
the two constructors were the only reason that was false.

The key is now read on the first request. Nothing about failing loud moves:
the first request without a key raises the same error, `/ready` reports the
missing credential separately (`required_env`), and the compose healthcheck
marks the container unhealthy. What changes is that the registry -- the tool
list, the descriptions, the caveats -- is readable on a machine that has no
key, which is what a documentation test needs and what a data source should
allow.
"""
import asyncio

import pytest

from tools.news_agregator import finnhub_utils, fred_utils


@pytest.fixture
def no_keys(monkeypatch):
    """No key in the environment and no .env to fall back on."""
    for name in ("FINNHUB_API_KEY", "FRED_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(finnhub_utils, "load_dotenv", lambda **kw: False)
    monkeypatch.setattr(fred_utils, "load_dotenv", lambda **kw: False)


@pytest.mark.parametrize("module,cls", [(finnhub_utils, "FinnhubClient"),
                                        (fred_utils, "FredClient")])
def test_a_client_constructs_without_its_key(no_keys, module, cls):
    getattr(module, cls)()


@pytest.mark.parametrize("module,cls,key", [
    (finnhub_utils, "FinnhubClient", "FINNHUB_API_KEY"),
    (fred_utils, "FredClient", "FRED_API_KEY")])
def test_the_first_request_without_a_key_fails_naming_it(no_keys, module, cls,
                                                        key):
    client = getattr(module, cls)()

    with pytest.raises(RuntimeError) as caught:
        asyncio.run(client.get("anything", {}))

    assert key in str(caught.value)


def test_the_finnhub_registry_is_readable_without_a_key(no_keys):
    import mcp.types as types
    from tools.news_agregator.finnhub_server import FinnhubServer

    srv = FinnhubServer().server
    listed = asyncio.run(srv.request_handlers[types.ListToolsRequest](
        types.ListToolsRequest(method="tools/list")))

    assert len(listed.root.tools) > 0


def test_the_fred_registry_is_readable_without_a_key(no_keys):
    import mcp.types as types
    from tools.news_agregator.fred_server import FredServer

    srv = FredServer().server
    listed = asyncio.run(srv.request_handlers[types.ListToolsRequest](
        types.ListToolsRequest(method="tools/list")))

    assert len(listed.root.tools) > 0


def test_a_key_that_is_present_is_read_once_and_kept(monkeypatch):
    monkeypatch.setenv("FINNHUB_API_KEY", "present-for-this-test")
    monkeypatch.setattr(finnhub_utils, "load_dotenv", lambda **kw: False)
    client = finnhub_utils.FinnhubClient()

    first = client.api_key
    monkeypatch.delenv("FINNHUB_API_KEY")

    assert client.api_key is first, "the key must not be re-read per request"
