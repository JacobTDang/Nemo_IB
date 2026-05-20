"""Phase A2: get_paper_account tool — wraps Execution_Agent.get_account_summary().

In-process unit tests (mock the Alpaca SDK; do not boot the MCP server in a
subprocess — that's covered by A7's e2e protocol test). Verify the tool
method coerces broker types to JSON-safe floats and handles broker errors
gracefully.
"""
import asyncio
import json
import os
import sys
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _fake_alpaca_account(equity="100000.00", cash="100000.00",
                          buying_power="200000.00", portfolio_value="100000.00",
                          status="ACTIVE"):
  acct = MagicMock()
  acct.equity = equity
  acct.cash = cash
  acct.buying_power = buying_power
  acct.portfolio_value = portfolio_value
  acct.status = status
  return acct


def _fake_alpaca_modules(account_obj=None, raises: Exception | None = None):
  """Build a sys.modules patch that fakes the alpaca-py imports the Execution_Agent
  pulls in lazily. Returns the modules dict plus the fake client for assertions."""
  client = MagicMock()
  if raises is not None:
    client.get_account.side_effect = raises
  else:
    client.get_account.return_value = account_obj or _fake_alpaca_account()
  trading_client_module = MagicMock()
  trading_client_module.TradingClient = MagicMock(return_value=client)
  trading_requests = MagicMock()
  trading_enums = MagicMock()
  return {
    'alpaca.trading.client': trading_client_module,
    'alpaca.trading.requests': trading_requests,
    'alpaca.trading.enums': trading_enums,
  }, client


def _call_tool(tool_name: str, arguments: dict) -> dict:
  """Construct the server, invoke its tool method, return parsed JSON body."""
  from tools.alpaca import server as alpaca_srv
  srv = alpaca_srv.AlpacaServer()
  # We dispatch by reading the handler registered on .server. Easier: call
  # the underlying method directly by convention. Phase A2 adds an
  # async method `get_paper_account` we can call directly.
  method = getattr(srv, tool_name)
  result = asyncio.run(method())
  text = result[0].text if result else "{}"
  return json.loads(text)


def test_get_paper_account_returns_floats_under_mock():
  os.environ['ALPACA_PAPER_KEY'] = 'fake_key'
  os.environ['ALPACA_PAPER_SECRET'] = 'fake_secret'
  fake_modules, fake_client = _fake_alpaca_modules()
  with patch.dict('sys.modules', fake_modules):
    # Force re-import of Execution_Agent so the patched sys.modules takes effect
    import importlib
    if 'agent.Execution_Agent' in sys.modules:
      del sys.modules['agent.Execution_Agent']
    parsed = _call_tool('get_paper_account', {})
  assert parsed.get('paper') is True, f"expected paper=True; got {parsed}"
  assert parsed['equity'] == 100000.0
  assert isinstance(parsed['equity'], float)
  assert parsed['cash'] == 100000.0
  assert parsed['buying_power'] == 200000.0
  assert parsed['portfolio_value'] == 100000.0
  assert parsed['status'] == 'ACTIVE'
  print(f"PASS: get_paper_account returns coerced floats ({parsed})")


def test_get_paper_account_surfaces_broker_error():
  os.environ['ALPACA_PAPER_KEY'] = 'fake_key'
  os.environ['ALPACA_PAPER_SECRET'] = 'fake_secret'
  fake_modules, _ = _fake_alpaca_modules(raises=Exception("API down"))
  with patch.dict('sys.modules', fake_modules):
    if 'agent.Execution_Agent' in sys.modules:
      del sys.modules['agent.Execution_Agent']
    parsed = _call_tool('get_paper_account', {})
  assert 'error' in parsed, f"expected error key on broker failure; got {parsed}"
  assert 'API down' in parsed['error'], f"error message should include broker reason; got {parsed['error']}"
  print(f"PASS: broker error surfaced as {{'error': ...}} (no raise)")


def test_get_paper_account_listed_in_tools():
  """The MCP list_tools must advertise get_paper_account."""
  from tools.alpaca import server as alpaca_srv
  srv = alpaca_srv.AlpacaServer()
  # Tool handlers are registered with decorators on srv.server. We can call
  # the registered list_tools by grabbing it from the server's internal map.
  tools = asyncio.run(_list_tools_via_handler(srv))
  names = [t.name for t in tools]
  assert 'get_paper_account' in names, f"get_paper_account missing from list_tools: {names}"
  print(f"PASS: get_paper_account in tools list ({len(names)} total)")


async def _list_tools_via_handler(srv):
  # Pull the registered list_tools handler. The mcp.server.Server stores
  # handlers internally; the simplest path is to call the public ServerSession
  # path, but for unit-test purposes we can rebuild by inspecting srv directly.
  # The cleanest approach: have the AlpacaServer expose a public list_tools_descriptors()
  # method we can call from tests. Add it in the same commit.
  return await srv.list_tools_descriptors()


if __name__ == "__main__":
  test_get_paper_account_listed_in_tools()
  test_get_paper_account_returns_floats_under_mock()
  test_get_paper_account_surfaces_broker_error()
  print("\nAll Phase A2 get_paper_account tests passed.")
