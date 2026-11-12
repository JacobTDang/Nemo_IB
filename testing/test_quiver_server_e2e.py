"""Quiver server end-to-end protocol smoke.

Boots tools/quiver_server as a real subprocess via stdio and exercises
every tool through the MCP JSON-RPC framing, against the live
quiverquant.com (public, no-auth data). Guards against the class of bugs
unit tests miss: TextContent serialization, the annotating dispatcher,
async ordering at the protocol layer, and envelope shape (success /
coverage / provider / warnings).

Network calls are kept small (limit=2) and every assertion tolerates the
site's own gaps (e.g. a ticker with no mentions) — the contract under
test is the framing and the honesty envelope, not Quiver's content.
"""
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def _exercise_all_tools() -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "tools.quiver_server.server", "server"],
        env=env,
    )

    results = {}
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            results["__tool_count__"] = len(tools.tools)
            results["__tool_names__"] = sorted(t.name for t in tools.tools)

            r = await session.call_tool("get_quiver_congress_trades", {"limit": 2})
            results["get_quiver_congress_trades"] = json.loads(r.content[0].text)

            r = await session.call_tool("get_quiver_congress_trades", {
                "ticker": "NVDA", "limit": 2})
            results["get_quiver_congress_trades_filtered"] = json.loads(
                r.content[0].text)

            r = await session.call_tool("get_quiver_politician_trades", {
                "politician": "Nancy Pelosi", "limit": 2})
            results["get_quiver_politician_trades"] = json.loads(r.content[0].text)

            r = await session.call_tool("get_quiver_wsb_sentiment", {
                "ticker": "NVDA", "include_history": True})
            results["get_quiver_wsb_sentiment"] = json.loads(r.content[0].text)

            r = await session.call_tool("get_quiver_lobbying", {"limit": 2})
            results["get_quiver_lobbying"] = json.loads(r.content[0].text)

            r = await session.call_tool("get_quiver_gov_contracts",
                                        {"limit": 2})
            results["get_quiver_gov_contracts"] = json.loads(r.content[0].text)

            r = await session.call_tool("get_quiver_ticker_snapshot", {
                "ticker": "NVDA", "trade_limit": 2})
            results["get_quiver_ticker_snapshot"] = json.loads(r.content[0].text)

            # failure vocabulary: unknown entity must be success=false
            # with a machine reason, not an exception over the wire.
            r = await session.call_tool("get_quiver_politician_trades", {
                "politician": "Zzqqxx Nobody"})
            results["unknown_politician"] = json.loads(r.content[0].text)

    return results


def test_quiver_server_e2e():
    results = asyncio.run(_exercise_all_tools())

    assert results["__tool_count__"] == 6, results["__tool_names__"]
    assert "get_quiver_congress_trades" in results["__tool_names__"]

    # Every response shares the envelope contract.
    for name in ["get_quiver_congress_trades", "get_quiver_congress_trades_filtered",
                 "get_quiver_politician_trades", "get_quiver_wsb_sentiment",
                 "get_quiver_lobbying", "get_quiver_gov_contracts",
                 "get_quiver_ticker_snapshot"]:
        body = results[name]
        assert body["success"] is True, (name, body.get("error"))
        assert body["domain"] == "quiver", name
        assert body["provider"], name
        assert isinstance(body["warnings"], list), name

    # Real data actually parsed out of the pages.
    trades = results["get_quiver_congress_trades"]["data"]["trades"]
    assert len(trades) == 2
    assert trades[0]["ticker"], trades
    assert trades[0]["filed_date"], trades

    filtered = results["get_quiver_congress_trades_filtered"]["data"]
    assert all(t["ticker"] == "NVDA" for t in filtered["trades"]), filtered

    pelosi = results["get_quiver_politician_trades"]["data"]
    assert pelosi["total"] > 50, pelosi  # full history, not a page slice
    assert pelosi["trades"][0]["tx_type"], pelosi

    wsb = results["get_quiver_wsb_sentiment"]["data"]
    assert wsb["data_horizon"], wsb  # staleness is always dated
    assert wsb["ticker_row"]["ticker"] == "NVDA", wsb
    assert wsb["history"], wsb  # daily mention series came back

    lobbying = results["get_quiver_lobbying"]["data"]
    assert lobbying["filings"], lobbying
    assert lobbying["top_spenders"], lobbying
    # normalized dates, not "Nov. 9, 2026, 3:10 p.m."
    assert len(lobbying["filings"][0]["date"]) == 10, lobbying["filings"][0]

    awards = results["get_quiver_gov_contracts"]["data"]
    assert awards["awards"], awards
    assert awards["awards"][0]["agency"], awards["awards"][0]

    snap = results["get_quiver_ticker_snapshot"]["data"]
    assert snap["price"]["price_data"]["current_price"], snap
    assert snap["insiders"]["total_trades"] > 0, snap
    assert snap["institutions"], snap
    # bulls_bears is session-gated: present only with a cookie, and when
    # absent it must be named, never silently missing.
    if "bulls_bears" not in snap:
        assert "bulls_bears:no_auth" in snap.get("degraded_sections", []), snap

    # Unknown entity -> success=false with a machine-readable reason.
    miss = results["unknown_politician"]
    assert miss["success"] is False
    assert miss["data"]["reason"] == "unknown_politician", miss


# ---------------------------------------------------------------------------
# Transport honesty. Offline: a scripted transport stands in for the site, so
# these run without a network and without depending on Quiver's rate limiter
# actually firing.
# ---------------------------------------------------------------------------

def _client_with(handler):
    import httpx
    from tools.quiver_server import client as qc
    c = qc.QuiverWebClient()
    c._client = httpx.Client(base_url=qc.BASE_URL,
                             transport=httpx.MockTransport(handler))
    return c


def test_a_rate_limit_is_retried_once_and_then_named():
    """One 429 is a burst, not an outage: retry it, then say what it was."""
    import httpx
    from tools.quiver_server.client import QuiverError

    attempts = []

    def once(request):
        attempts.append(request.url.path)
        if len(attempts) == 1:
            return httpx.Response(429, headers={"retry-after": "0"},
                                  text="slow down")
        return httpx.Response(200, text='{"response": "ok"}')

    c = _client_with(once)
    assert json.loads(c._fetch("/get_current_price/NVDA")) == {"response": "ok"}
    assert len(attempts) == 2, attempts

    # A second 429 in the same call is the site refusing, and the caller is
    # told so instead of being held open for another attempt.
    attempts.clear()
    c = _client_with(lambda request: (
        attempts.append(request.url.path),
        httpx.Response(429, headers={"retry-after": "0"}, text="nope"),
    )[1])
    try:
        c._fetch("/get_current_price/NVDA")
    except QuiverError as exc:
        assert exc.reason == "rate_limited", exc.reason
    else:
        raise AssertionError("a refused request must not return a body")
    assert len(attempts) == 2, attempts


def test_a_rate_limit_is_not_reported_as_an_unknown_ticker():
    """The reason names what happened. A throttled request says nothing about
    whether the ticker exists, and `unknown_ticker` would be a wrong answer."""
    from tools.quiver_server.client import QuiverError

    c = _client_with(lambda request: None)

    def throttled(path, *args, **kwargs):
        raise QuiverError("rate_limited", f"{path} returned HTTP 429")

    c._json = throttled
    try:
        c.get_ticker_snapshot("NVDA")
    except QuiverError as exc:
        assert exc.reason == "rate_limited", exc.reason
    else:
        raise AssertionError("a snapshot with no answering section must raise")

    # A ticker Quiver does not cover is the one case that earns that name.
    def missing(path, *args, **kwargs):
        raise QuiverError("not_found", f"{path} returned HTTP 404")

    c._json = missing
    try:
        c.get_ticker_snapshot("ZZZZZ")
    except QuiverError as exc:
        assert exc.reason == "unknown_ticker", exc.reason
    else:
        raise AssertionError("a ticker with no endpoints must raise")


def test_a_ticker_with_no_data_is_not_returned_as_six_empty_sections():
    """Quiver answers 200 with "N/A" and empty arrays for a ticker it does not
    cover. Handing that back would read as a covered company with no insiders,
    no institutions and no analyst coverage."""
    from tools.quiver_server.client import QuiverError, _snapshot_is_hollow

    c = _client_with(lambda request: None)

    def hollow(path, *args, **kwargs):
        if "bulls_bears" in path:
            return {"response": "empty"}
        if "insiders" in path:
            return {"insider_trades": []}
        if "institutions" in path:
            return {"response": {}}
        if "forecast" in path:
            return {"data": [], "price_targets": {}}
        if "tv_lazy" in path:
            return {"etf_data": []}
        return {"price_data": {"current_price": "N/A"}}

    c._json = hollow
    try:
        c.get_ticker_snapshot("ZZZZZ")
    except QuiverError as exc:
        assert exc.reason == "unknown_ticker", exc.reason
    else:
        raise AssertionError("a hollow snapshot must be named, not returned")

    # A real quote is not hollow even with every holding section empty ...
    assert not _snapshot_is_hollow(
        {"price": {"price_data": {"current_price": "173.73"}}})
    # ... and a covered ticker reports how complete the answer is.
    def quoted(path, *args, **kwargs):
        if "insiders" in path:
            return {"insider_trades": [["X", "Sale", 1]]}
        if "institutions" in path:
            return {"response": {"heatmap_data": [{"Fund": "BlackRock"}]}}
        if "forecast" in path:
            return {"data": [1], "price_targets": {"high": 200}}
        if "tv_lazy" in path:
            return {"etf_data": [{"ticker": "QQQ"}]}
        if "bulls_bears" in path:
            return {"response": "no-auth"}
        return {"price_data": {"current_price": "173.73"}}

    c._json = quoted
    snap = c.get_ticker_snapshot("NVDA")
    assert snap["coverage"] == "partial", snap.get("coverage")
    assert "bulls_bears:no_auth" in snap["degraded_sections"], snap


if __name__ == "__main__":
    test_quiver_server_e2e()
    test_a_rate_limit_is_retried_once_and_then_named()
    test_a_rate_limit_is_not_reported_as_an_unknown_ticker()
    test_a_ticker_with_no_data_is_not_returned_as_six_empty_sections()
    print("quiver server e2e: PASS")
