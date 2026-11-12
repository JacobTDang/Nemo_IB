"""Smoke test: nemo_quiver MCP server over stdio.

Exercises the full stdio path — initialize, list_tools, and one live call
per tool family — the same way a host (Claude Code, Nemo) would.

Run:  .venv/bin/python testing/test_quiver_server_smoke.py
"""
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main() -> int:
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "tools.quiver_server.server", "server"],
        env={**os.environ, "PYTHONPATH": os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))},
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await asyncio.wait_for(session.initialize(), 30)

            tools = await session.list_tools()
            names = [t.name for t in tools.tools]
            print("tools:", names)
            assert "get_quiver_congress_trades" in names
            assert len(names) == 6, f"expected 6 tools, got {names}"

            # 1. congress trades, small limit
            res = await asyncio.wait_for(session.call_tool(
                "get_quiver_congress_trades", {"limit": 5}), 120)
            payload = json.loads(res.content[0].text)
            assert payload["success"], payload.get("error")
            n = payload["data"]["returned"]
            print(f"congress_trades ok: {n} trades, first ticker="
                  f"{payload['data']['trades'][0]['ticker']}")
            assert n == 5

            # 2. politician trades
            res = await asyncio.wait_for(session.call_tool(
                "get_quiver_politician_trades",
                {"politician": "Nancy Pelosi", "limit": 5}), 120)
            payload = json.loads(res.content[0].text)
            assert payload["success"], payload.get("error")
            print(f"politician_trades ok: {payload['data']['total']} total, "
                  f"returned {payload['data']['returned']}")

            # 3. WSB sentiment with history
            res = await asyncio.wait_for(session.call_tool(
                "get_quiver_wsb_sentiment",
                {"ticker": "NVDA", "include_history": True}), 180)
            payload = json.loads(res.content[0].text)
            assert payload["success"], payload.get("error")
            d = payload["data"]
            print(f"wsb ok: mentions={d['ticker_row']['mentions']}, "
                  f"history_days={d.get('history_days')}, "
                  f"horizon={d.get('data_horizon')}")
            assert d.get("history_days", 0) > 0

            # 4. lobbying
            res = await asyncio.wait_for(session.call_tool(
                "get_quiver_lobbying", {"limit": 5}), 120)
            payload = json.loads(res.content[0].text)
            assert payload["success"], payload.get("error")
            print(f"lobbying ok: {payload['data']['total_filings']} filings, "
                  f"top spender {payload['data']['top_spenders'][0]}")

            # 5. gov contract awards
            res = await asyncio.wait_for(session.call_tool(
                "get_quiver_gov_contracts", {"limit": 5}), 120)
            payload = json.loads(res.content[0].text)
            assert payload["success"], payload.get("error")
            print(f"gov_contracts ok: {payload['data']['total_embedded']} "
                  f"embedded, first date "
                  f"{payload['data']['awards'][0]['date_awarded']}")

            # 6. ticker snapshot
            res = await asyncio.wait_for(session.call_tool(
                "get_quiver_ticker_snapshot", {"ticker": "NVDA"}), 180)
            payload = json.loads(res.content[0].text)
            assert payload["success"], payload.get("error")
            d = payload["data"]
            sections = [k for k in d if k not in
                        ("ticker", "degraded_sections")]
            print(f"snapshot ok: sections={sections}, degraded="
                  f"{d.get('degraded_sections')}")
            assert "price" in d

            # 7. unknown politician -> structured failure, not a crash
            res = await asyncio.wait_for(session.call_tool(
                "get_quiver_politician_trades",
                {"politician": "Zzzqqq Nonexistent"}), 120)
            payload = json.loads(res.content[0].text)
            assert payload["success"] is False
            assert payload["error"] == (
                "no congress member matches 'Zzzqqq Nonexistent'")
            print("unknown-politician failure ok:", payload["error"])

    print("\nALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
