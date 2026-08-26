"""Recording a success at the one seam every tool call already passes through.

`/ready` reports `last_success: {"ok": false, "at": null}` on all five deployed
servers while they answer questions perfectly well: `record_success()` exists
in `tools/mcp_http.py`, `/ready` reads it, and nothing ever calls it. A probe
pointed at that keeps every container out of rotation forever, which is worse
than the false green it was built to replace.

The recording belongs in `annotating()` -- the dispatcher wrapper every server
already applies -- because that is the single place that sees every one of the
96 tools' results, and a server added next month gets it without remembering
to. What it must not do is record indiscriminately: `/ready` asks "has this
server ever successfully answered anything", so a server returning nothing but
errors is not ready, and a check that cannot tell those apart means nothing.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest
from mcp.types import TextContent

from tools import mcp_http
from tools.response_meta import annotating

REPO_ROOT = Path(__file__).resolve().parents[1]


def _dispatcher(*items):
    """A dispatcher in the shape every server uses: (name, args) -> contents.

    Dicts are serialised the way a tool would; anything else is handed back
    untouched so a test can hand over prose or a JSON list.
    """
    async def call_tool(name, args):
        return [item if isinstance(item, TextContent)
                else TextContent(type="text", text=json.dumps(item))
                for item in items]
    return call_tool


@pytest.fixture
def recorded(monkeypatch):
    """Every call to `record_success`, counted.

    A spy rather than the clock: "the timestamp moved" cannot distinguish one
    recording from five, and requirement five is precisely about the count.
    """
    calls = []
    monkeypatch.setattr(mcp_http, "record_success",
                        lambda *args, **kwargs: calls.append(args))
    return calls


@pytest.fixture(autouse=True)
def _nothing_has_succeeded_yet(monkeypatch):
    """The last-success timestamp is a module global shared by every server.

    A test that records one would otherwise leak into `/ready`'s own tests,
    where "nothing has ever succeeded" is the state under test.
    """
    monkeypatch.setattr(mcp_http, "_LAST_SUCCESS", None)


# ------------------------------------------------------------ what counts

async def test_a_successful_response_records_a_success(recorded):
    """The whole point: a working server must stop reporting itself unready."""
    wrapped = annotating("SEC EDGAR")(_dispatcher({"revenue": 1}))

    await wrapped("get_thing", {})

    assert len(recorded) == 1, f"recorded {len(recorded)} times, expected 1"


async def test_the_real_recorder_is_the_one_that_gets_called():
    """A spy proves the wrapper calls something; it does not prove it calls
    the function `/ready` reads. A wrong module path or a renamed function
    would pass every other test here and leave readiness false in production.
    """
    wrapped = annotating("SEC EDGAR")(_dispatcher({"revenue": 1}))

    await wrapped("get_thing", {})

    assert mcp_http._LAST_SUCCESS is not None
    assert mcp_http._check_last_success()["ok"] is True


@pytest.mark.parametrize("payload", [
    {"success": False, "error": "rate limited"},
    {"error": "upstream 404"},          # success inferred false from the error
    {"success": False, "data": None, "metadata": {"errors": ["no provider"]}},
])
async def test_a_failed_response_records_nothing(recorded, payload):
    """Backwards here makes the check meaningless. `/ready` asks whether this
    server has ever answered anything; a container whose every upstream is
    rejecting its key returns errors happily and would look ready.
    """
    wrapped = annotating("Finnhub")(_dispatcher(payload))

    await wrapped("get_quote", {})

    assert recorded == [], "an error response was recorded as a success"


async def test_a_raising_handler_records_nothing_and_still_raises(recorded):
    """Fail loud. The exception is the server's own failure, and swallowing it
    into a recorded success would report the worst state as the healthiest.
    """
    async def explodes(name, args):
        raise RuntimeError("upstream exploded")

    wrapped = annotating("SEC EDGAR")(explodes)

    with pytest.raises(RuntimeError, match="upstream exploded"):
        await wrapped("get_thing", {})
    assert recorded == []


@pytest.mark.parametrize("item", [
    TextContent(type="text", text="not json at all"),
    [{"headline": "x"}],
    "a bare JSON string",
])
async def test_a_response_that_cannot_carry_provenance_still_records(
        recorded, item):
    """"Could not annotate" is not "failed". These responses completed; they
    merely have nowhere to put provenance -- `annotate()` refuses a list on
    purpose. Treating them as failures would leave `web_search`, which returns
    prose from fifty places, permanently unready.
    """
    wrapped = annotating("Finnhub")(_dispatcher(item))

    await wrapped("get_company_news", {})

    assert len(recorded) == 1


async def test_a_multi_item_response_records_exactly_one_success(recorded):
    """Recording per item would make the count a report on response shape
    rather than on the server, and `/ready` reads a timestamp, not a tally --
    but a wrapper writing it three times per call is doing work per content
    block that belongs per call.
    """
    wrapped = annotating("SEC EDGAR")(
        _dispatcher({"a": 1}, {"b": 2}, {"c": 3}))

    contents = await wrapped("get_things", {})

    assert len(contents) == 3
    assert len(recorded) == 1, f"recorded once per item: {len(recorded)}"


# --------------------------------------------- telemetry never breaks a tool

async def test_a_recorder_that_raises_does_not_break_the_response(monkeypatch):
    """Recording is observability. A clock problem inside `record_success`
    turning a good SEC filing into a failed tool call would be a strictly
    worse bug than the unready flag this fixes.
    """
    def explodes(*args, **kwargs):
        raise OSError("clock went backwards")

    monkeypatch.setattr(mcp_http, "record_success", explodes)
    wrapped = annotating("SEC EDGAR")(_dispatcher({"revenue": 1}))

    contents = await wrapped("get_thing", {})

    assert json.loads(contents[0].text)["revenue"] == 1
    assert json.loads(contents[0].text)["provider"] == "SEC EDGAR"


async def test_an_unavailable_mcp_http_degrades_silently(monkeypatch):
    """`response_meta` is imported by every server, including any future one
    that does not serve over HTTP at all. Recording is not correctness, so its
    absence must cost nothing but the timestamp.
    """
    monkeypatch.setitem(sys.modules, "tools.mcp_http", None)
    wrapped = annotating("FRED")(_dispatcher({"series_id": "GDP"}))

    contents = await wrapped("get_series", {})

    assert json.loads(contents[0].text)["provider"] == "FRED"


def test_response_meta_does_not_import_mcp_http_at_import_time():
    """Every server imports `response_meta`; `mcp_http` imports uvicorn and
    starlette and, in turn, servers. A module-level import would be a real
    circular-import and startup-cost regression, so the dependency has to be
    taken lazily inside the call.

    Checked in a fresh interpreter because this test session has already
    imported `mcp_http` itself.
    """
    probe = ("import tools.response_meta, sys; "
             "print('tools.mcp_http' in sys.modules)")
    result = subprocess.run([sys.executable, "-c", probe],
                            cwd=REPO_ROOT, capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False", (
        "importing response_meta dragged in tools.mcp_http")
