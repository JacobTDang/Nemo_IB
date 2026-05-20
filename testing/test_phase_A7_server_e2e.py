"""Phase A7: alpaca MCP server end-to-end protocol smoke.

Boots the server as a real subprocess via stdio and exercises every tool
through the MCP JSON-RPC framing. This guards against a class of bugs the
unit tests miss: TextContent serialization, decorator dispatch, async
ordering at the protocol layer.

For tools that need real Alpaca creds (get_paper_account, place_paper_order,
get_paper_positions broker side, close_paper_position), we don't expect a
successful broker call — we expect the tool to gracefully return an error
dict in the response body. The point is to verify the framing, not the
business logic (covered by A2-A6).
"""
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def _exercise_all_tools() -> dict:
  """Boot the server, call every tool, return results keyed by tool name."""
  env = os.environ.copy()
  env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  # Use fake-but-present creds so Execution_Agent doesn't raise at import;
  # actual Alpaca calls will fail downstream and surface as `error` in body.
  env.setdefault("ALPACA_PAPER_KEY", "e2e_fake_key")
  env.setdefault("ALPACA_PAPER_SECRET", "e2e_fake_secret")

  params = StdioServerParameters(
    command=sys.executable,
    args=["-m", "tools.alpaca.server", "server"],
    env=env,
  )

  results = {}
  async with stdio_client(params) as (read, write):
    async with ClientSession(read, write) as session:
      await session.initialize()

      tools = await session.list_tools()
      results["__tool_count__"] = len(tools.tools)
      results["__tool_names__"] = sorted(t.name for t in tools.tools)

      # ping
      r = await session.call_tool("ping_alpaca", {})
      results["ping_alpaca"] = json.loads(r.content[0].text)

      # account (will error: fake creds)
      r = await session.call_tool("get_paper_account", {})
      results["get_paper_account"] = json.loads(r.content[0].text)

      # positions (broker errors, local part succeeds)
      r = await session.call_tool("get_paper_positions", {})
      results["get_paper_positions"] = json.loads(r.content[0].text)

      # risk_check — does not touch broker; should produce a real decision
      r = await session.call_tool("risk_check_proposed_trade", {
        "ticker": "A7_TEST", "side": "buy", "quantity": 10, "price": 100.0,
        "recommendation": "BUY", "confidence": 0.75,
        "bull_strength": 0.78, "bear_strength": 0.45,
        "position_sizing": "normal", "rationale": "e2e smoke",
      })
      results["risk_check_proposed_trade"] = json.loads(r.content[0].text)

      # place_paper_order with a confidence that triggers risk rejection —
      # broker MUST not be called even though we don't have real creds
      r = await session.call_tool("place_paper_order", {
        "ticker": "A7_TEST", "side": "buy", "quantity": 10, "price": 100.0,
        "recommendation": "BUY", "confidence": 0.30,  # below threshold
        "bull_strength": 0.5, "bear_strength": 0.5,
        "position_sizing": "normal", "rationale": "e2e reject path",
      })
      results["place_paper_order_rejected"] = json.loads(r.content[0].text)

      # close on ticker we know is not open — must return clean error
      r = await session.call_tool("close_paper_position",
                                    {"ticker": "A7_NOTHING",
                                     "reason": "e2e smoke"})
      results["close_paper_position"] = json.loads(r.content[0].text)

  return results


def test_all_six_tools_advertised():
  r = asyncio.run(_exercise_all_tools())
  expected = {'ping_alpaca', 'get_paper_account', 'get_paper_positions',
               'risk_check_proposed_trade', 'place_paper_order',
               'close_paper_position'}
  assert set(r['__tool_names__']) == expected, \
    f"tool set mismatch; got {r['__tool_names__']}"
  print(f"PASS: 6 tools advertised: {r['__tool_names__']}")


def test_ping_alpaca_pong_through_stdio():
  r = asyncio.run(_exercise_all_tools())
  assert r['ping_alpaca'] == {'status': 'pong'}
  print(f"PASS: ping_alpaca returned {r['ping_alpaca']}")


def test_risk_check_returns_real_decision_through_stdio():
  """risk_check is the only tool that doesn't touch Alpaca, so it should
  produce a clean approve/reject regardless of creds."""
  r = asyncio.run(_exercise_all_tools())
  rc = r['risk_check_proposed_trade']
  assert 'approve' in rc, f"risk_check should always include approve key: {rc}"
  assert isinstance(rc['approve'], bool)
  assert 'reasons' in rc
  print(f"PASS: risk_check_proposed_trade returned approve={rc['approve']}")


def test_place_paper_order_rejection_through_stdio():
  """Low confidence -> Risk_Officer rejects -> broker never called. With
  fake creds the request shouldn't even reach the broker; the error path
  must be 'risk_rejected', not 'broker_error' or 'missing_credentials'."""
  r = asyncio.run(_exercise_all_tools())
  out = r['place_paper_order_rejected']
  assert out.get('success') is False
  assert 'risk_rejected' in (out.get('error') or '').lower() or \
    out.get('risk_decision', {}).get('approve') is False
  print(f"PASS: rejected place_paper_order surfaced through stdio "
        f"(error={out.get('error', '')[:80]})")


def test_close_no_position_clean_error_through_stdio():
  r = asyncio.run(_exercise_all_tools())
  out = r['close_paper_position']
  assert out.get('success') is False
  # error is either 'no_open_position' from Execution_Agent OR a credential
  # error if broker init failed; both are graceful
  assert 'error' in out
  print(f"PASS: close on non-existent position returned clean error: "
        f"{out.get('error', '')[:60]}")


def test_get_account_surfaces_error_not_crash():
  """With fake creds, get_account should fail gracefully, not crash the
  server or break the JSON-RPC framing."""
  r = asyncio.run(_exercise_all_tools())
  out = r['get_paper_account']
  # Either error (most likely — fake creds) or success (if creds happen to work)
  assert isinstance(out, dict)
  assert 'paper' in out or 'error' in out
  print(f"PASS: get_paper_account framing OK "
        f"({'error' if 'error' in out else 'success'})")


def test_get_positions_partial_response_through_stdio():
  """Broker side fails but local side should still come back (the
  reconciliation tool returns BOTH views)."""
  r = asyncio.run(_exercise_all_tools())
  out = r['get_paper_positions']
  assert 'local_positions' in out, f"local_positions missing: {sorted(out.keys())}"
  assert 'reconciled' in out
  print(f"PASS: get_paper_positions returned partial response "
        f"(broker={'error' if 'error' in out else 'ok'}, "
        f"local rows={len(out.get('local_positions', []))})")


if __name__ == "__main__":
  test_all_six_tools_advertised()
  test_ping_alpaca_pong_through_stdio()
  test_risk_check_returns_real_decision_through_stdio()
  test_place_paper_order_rejection_through_stdio()
  test_close_no_position_clean_error_through_stdio()
  test_get_account_surfaces_error_not_crash()
  test_get_positions_partial_response_through_stdio()
  print("\nAll Phase A7 server e2e tests passed.")
