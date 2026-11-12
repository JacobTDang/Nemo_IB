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


if __name__ == "__main__":
    test_quiver_server_e2e()
    print("quiver server e2e: PASS")
