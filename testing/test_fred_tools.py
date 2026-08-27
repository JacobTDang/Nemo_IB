"""Integration test for the FRED MCP server.

This was a script whose only check was that nothing raised -- it printed
"All tests passed!" at the end regardless of what came back, and pytest
collected zero tests from it because nothing inside was named or shaped like
one. The four FRED tools had no coverage at all while the file sat in
`testing/` looking like they did.

The assertions here are the ones the data can actually violate: the curve
and the spreads struck from it must reconcile exactly, the observation count
must match the observations returned, and a tool that recorded an error must
not pass as a success.

Gated on FRED_API_KEY and skipped when SKIP_NETWORK_TESTS=1.

Usage: python -m pytest testing/test_fred_tools.py
"""
import json

import pytest

from tools.news_agregator.fred_server import FredServer
from testing._gates import requires_fred

pytestmark = [pytest.mark.network, requires_fred]

ENVELOPE_KEYS = {"domain", "context", "tool", "timestamp", "data", "metadata"}


@pytest.fixture
async def fred():
    server = FredServer()
    try:
        yield server
    finally:
        await server.client.close()


def _payload(result, tool_name):
    """The envelope every FRED tool returns, checked before it is read."""
    assert result, f"{tool_name} returned nothing"
    body = json.loads(result[0].text)

    missing = ENVELOPE_KEYS - set(body)
    assert not missing, f"{tool_name} envelope is missing {sorted(missing)}"
    assert body["domain"] == "macro", f"{tool_name} domain is {body['domain']!r}"
    assert body["tool"] == tool_name, (
        f"envelope says {body['tool']!r} but {tool_name} was called")

    errors = (body.get("metadata") or {}).get("errors") or []
    assert not errors, f"{tool_name} recorded errors: {errors}"
    return body["data"]


# A level series reports `current`; a price index reports its level and the
# year-over-year change instead, because the level of CPI is not a reading
# anyone acts on. Either shape is fine -- carrying neither is not.
READING_KEYS = ("current", "yoy_pct", "latest_index")


async def test_every_macro_series_carries_a_reading(fred):
    data = _payload(await fred.get_macro_snapshot(), "get_macro_snapshot")
    assert data, "macro snapshot is empty"

    for series_id, row in data.items():
        readings = [k for k in READING_KEYS if row.get(k) is not None]
        assert readings, (
            f"{series_id} carries no reading at all under any of "
            f"{READING_KEYS}: {row}")
        assert row.get("as_of"), f"{series_id} does not say when it was current"
        assert row.get("label"), f"{series_id} has no label to display"


async def test_the_yield_curve_and_its_spreads_reconcile(fred):
    """A spread struck from a curve must equal the curve it was struck from.

    Both come from the same call, so any difference is the tool disagreeing
    with itself rather than two dates being compared.
    """
    data = _payload(await fred.get_treasury_yields(), "get_treasury_yields")
    curve, spreads = data["curve"], data["spreads"]
    assert curve, "the curve is empty"

    for name, spread in spreads.items():
        long_end, short_end = name.split("_")
        if long_end not in curve or short_end not in curve:
            continue
        expected = curve[long_end] - curve[short_end]
        assert spread == pytest.approx(expected, abs=0.011), (
            f"{name} is reported as {spread} but the curve gives "
            f"{curve[long_end]} - {curve[short_end]} = {expected:.2f}")


async def test_the_curve_shape_agrees_with_the_curve(fred):
    data = _payload(await fred.get_treasury_yields(), "get_treasury_yields")
    curve, shape = data["curve"], data["shape"]
    if "10Y" not in curve or "2Y" not in curve:
        pytest.skip("the curve does not carry both tenors this run")

    inverted = curve["10Y"] < curve["2Y"]
    assert inverted == (shape == "inverted"), (
        f"shape says {shape!r} while 10Y={curve['10Y']} and 2Y={curve['2Y']}")


async def test_a_series_returns_the_observations_it_counts(fred):
    data = _payload(await fred.get_fred_series("M2SL", frequency="m"),
                    "get_fred_series")
    observations = data["observations"]
    assert observations, "M2SL came back with no observations"
    assert data["series_id"] == "M2SL"
    assert data["returned"] == len(observations), (
        f"reported {data['returned']} observations but returned "
        f"{len(observations)} -- a caller trusting the count reads past the end")

    for row in observations:
        assert row.get("date"), f"observation has no date: {row}"
        assert row.get("value") is not None, f"observation has no value: {row}"


async def test_search_finds_a_series_that_exists(fred):
    data = _payload(await fred.search_fred("housing starts"), "search_fred")
    results = data["results"]
    assert results, "no FRED series matched 'housing starts'"
    assert data["returned"] == len(results), (
        f"reported {data['returned']} matches but returned {len(results)}")
