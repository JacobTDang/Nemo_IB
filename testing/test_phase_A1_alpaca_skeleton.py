"""Phase A1: Alpaca MCP server skeleton — verify MCP plumbing.

Boots the server as a subprocess via stdio, sends list_tools and
call_tool('ping_alpaca'), asserts the tool is registered and returns
{"status": "pong"}. This guards against MCP framing bugs that unit
tests would miss.
"""
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def _run_skeleton_test() -> dict:
  """Boot the alpaca MCP server, list tools, call ping_alpaca, return results."""
  env = os.environ.copy()
  env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  params = StdioServerParameters(
    command=sys.executable,
    args=["-m", "tools.alpaca.server", "server"],
    env=env,
  )
  async with stdio_client(params) as (read, write):
    async with ClientSession(read, write) as session:
      await session.initialize()
      tools_result = await session.list_tools()
      tool_names = [t.name for t in tools_result.tools]
      ping_result = await session.call_tool("ping_alpaca", {})
      content_text = ping_result.content[0].text if ping_result.content else ""
      try:
        parsed = json.loads(content_text)
      except json.JSONDecodeError:
        parsed = {"raw": content_text}
      return {"tools": tool_names, "ping": parsed}


def test_skeleton_lists_ping_tool():
  result = asyncio.run(_run_skeleton_test())
  assert "ping_alpaca" in result["tools"], \
    f"ping_alpaca not in tools list: {result['tools']}"
  print(f"PASS: server exposes {len(result['tools'])} tool(s): {result['tools']}")


def test_skeleton_ping_returns_pong():
  result = asyncio.run(_run_skeleton_test())
  assert result["ping"].get("status") == "pong", \
    f"ping_alpaca should return status=pong; got {result['ping']!r}"
  print(f"PASS: ping_alpaca returned {result['ping']!r}")


if __name__ == "__main__":
  test_skeleton_lists_ping_tool()
  test_skeleton_ping_returns_pong()
  print("\nAll Phase A1 alpaca skeleton tests passed.")
