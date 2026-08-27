"""Annotating at the dispatcher, not in ninety-six tool functions.

`web_search` alone returns `TextContent` from fifty separate places. Editing
each one is fifty chances to differ and one chance to be forgotten, and a tool
added next month would silently ship without provenance. Every server instead
exposes exactly one `call_tool(name, args) -> list[TextContent]`, so wrapping
that covers the whole catalogue and cannot be skipped.

What the wrapper must not do is guess. A payload whose period is unstated gets
`data_as_of: null`, which is a true statement about what we know; inferring one
from whatever date-shaped field is lying around would produce confident
provenance that is wrong, which is worse than none.
"""
import json

import pytest
from mcp.types import TextContent

from tools.response_meta import SCHEMA_VERSION, annotating, warning


def _payload(contents, index=0):
    return json.loads(contents[index].text)


async def _run(handler, name="get_thing", args=None):
    return await handler(name, args or {})


@pytest.fixture
def handler():
    """A dispatcher in the shape every server uses."""
    async def call_tool(name, args):
        if name == "raises":
            raise RuntimeError("upstream exploded")
        return [TextContent(type="text", text=json.dumps(args["body"]))]
    return call_tool


async def test_every_response_gains_provenance(handler):
    wrapped = annotating("SEC EDGAR")(handler)
    body = await _run(wrapped, args={"body": {"revenue": 1}})
    out = _payload(body)

    assert out["provider"] == "SEC EDGAR"
    assert out["schema_version"] == SCHEMA_VERSION
    assert out["success"] is True
    assert out["coverage"] == "unknown"
    assert out["retrieved_at"].endswith("Z")
    assert out["revenue"] == 1, "the original payload must survive intact"


async def test_a_per_tool_provider_overrides_the_server_default(handler):
    """altdata reads six different upstreams; one name for all of them lies."""
    wrapped = annotating("altdata", per_tool={
        "get_congress_trades": "US House Clerk / Senate eFD",
    })(handler)

    congress = _payload(await _run(wrapped, "get_congress_trades", {"body": {}}))
    jobs = _payload(await _run(wrapped, "get_job_postings_count", {"body": {}}))

    assert congress["provider"] == "US House Clerk / Senate eFD"
    assert jobs["provider"] == "altdata"


@pytest.mark.parametrize("field,value", [
    ("period_end", "2025-09-28"),
    ("as_of", "2025-12-31"),
    ("data_as_of", "2024-06-30"),
])
async def test_an_unambiguous_period_becomes_data_as_of(handler, field, value):
    wrapped = annotating("SEC EDGAR")(handler)
    out = _payload(await _run(wrapped, args={"body": {field: value}}))
    assert out["data_as_of"] == value


@pytest.mark.parametrize("field", ["timestamp", "date", "filing_date", "filed_date"])
async def test_an_ambiguous_date_is_never_promoted_to_data_as_of(handler, field):
    """`timestamp` is when we asked. `filing_date` is when it was filed.

    Neither is the period the data describes, and a caller comparing two
    observations on a wrong `data_as_of` compares different periods believing
    they match.
    """
    wrapped = annotating("SEC EDGAR")(handler)
    out = _payload(await _run(wrapped, args={"body": {field: "2026-08-25"}}))

    assert out["data_as_of"] is None, (
        f"{field} was promoted to data_as_of; it does not mean the period the "
        f"data covers")
    assert out[field] == "2026-08-25", "the original field must be left alone"


async def test_a_source_url_already_present_is_carried_up(handler):
    wrapped = annotating("SEC EDGAR")(handler)
    out = _payload(await _run(wrapped, args={
        "body": {"source_url": "https://sec.gov/x.htm"}}))
    assert out["source_url"] == "https://sec.gov/x.htm"


async def test_a_list_payload_is_left_untouched_rather_than_guessed_at(handler):
    """annotate() refuses a list on purpose; the wrapper must not force one."""
    wrapped = annotating("Finnhub")(handler)
    body = await _run(wrapped, args={"body": [{"headline": "x"}]})

    assert json.loads(body[0].text) == [{"headline": "x"}]


async def test_non_json_text_is_left_untouched(handler):
    async def prose(name, args):
        return [TextContent(type="text", text="not json at all")]

    wrapped = annotating("SEC EDGAR")(prose)
    assert (await _run(wrapped))[0].text == "not json at all"


async def test_an_error_payload_still_gets_provenance(handler):
    """A failure that loses its provider is unattributable."""
    wrapped = annotating("Finnhub")(handler)
    out = _payload(await _run(wrapped, args={"body": {"error": "rate limited"}}))

    assert out["success"] is False
    assert out["provider"] == "Finnhub"


async def test_a_raising_handler_still_raises(handler):
    """Fail loud: the wrapper must not turn an exception into a response."""
    wrapped = annotating("SEC EDGAR")(handler)
    with pytest.raises(RuntimeError, match="upstream exploded"):
        await _run(wrapped, "raises")


async def test_an_existing_envelope_is_annotated_at_the_top_only(handler):
    """fred and finnhub already wrap; their `data` must not be disturbed."""
    wrapped = annotating("FRED")(handler)
    out = _payload(await _run(wrapped, args={"body": {
        "domain": "macro", "data": {"series_id": "GDP"}, "metadata": {}}}))

    assert out["data"] == {"series_id": "GDP"}
    assert "provider" not in out["data"]
    assert out["provider"] == "FRED"


async def test_multiple_content_items_are_each_annotated(handler):
    async def two(name, args):
        return [TextContent(type="text", text=json.dumps({"a": 1})),
                TextContent(type="text", text=json.dumps({"b": 2}))]

    wrapped = annotating("SEC EDGAR")(two)
    body = await _run(wrapped)
    assert _payload(body, 0)["provider"] == "SEC EDGAR"
    assert _payload(body, 1)["provider"] == "SEC EDGAR"


async def test_static_warnings_ride_along_per_tool(handler):
    """A documented caveat becomes machine-readable where it applies."""
    stale = warning("stale_by_design", "short interest lags 2-3 weeks")
    wrapped = annotating("Finnhub", warnings_per_tool={
        "get_short_interest": [stale]})(handler)

    flagged = _payload(await _run(wrapped, "get_short_interest", {"body": {}}))
    other = _payload(await _run(wrapped, "get_company_news", {"body": {}}))

    assert flagged["warnings"] == [stale]
    assert other["warnings"] == []
