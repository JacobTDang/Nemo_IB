"""Phase A6: close_paper_position — opposing market order to flatten.

No Risk_Officer gate (reducing exposure is always safe), but the `reason`
field is required for the audit trail. Wraps the existing
Execution_Agent.close_position_for().
"""
import asyncio
import json
import os
import sys
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from state.schema import init_schema, get_connection
from state.positions import open_position


def _clean():
  conn = get_connection()
  try:
    conn.execute("DELETE FROM positions WHERE ticker LIKE 'A6_%'")
    conn.execute("DELETE FROM orders WHERE ticker LIKE 'A6_%'")
    conn.commit()
  finally:
    conn.close()


def _make_fake_alpaca(broker_order_id="alp-a6-close-1"):
  fake_order = MagicMock()
  fake_order.id = broker_order_id
  fake_order.status = "accepted"
  client = MagicMock()
  client.submit_order.return_value = fake_order
  trading_client_module = MagicMock()
  trading_client_module.TradingClient = MagicMock(return_value=client)
  trading_requests = MagicMock()
  trading_requests.MarketOrderRequest = MagicMock()
  trading_requests.LimitOrderRequest = MagicMock()
  trading_enums = MagicMock()
  trading_enums.OrderSide.BUY = "buy"
  trading_enums.OrderSide.SELL = "sell"
  trading_enums.TimeInForce.DAY = "day"
  return {
    'alpaca.trading.client': trading_client_module,
    'alpaca.trading.requests': trading_requests,
    'alpaca.trading.enums': trading_enums,
  }, client


def _call_close(args) -> tuple[dict, MagicMock]:
  os.environ['ALPACA_PAPER_KEY'] = 'fake'
  os.environ['ALPACA_PAPER_SECRET'] = 'fake'
  fake_modules, fake_client = _make_fake_alpaca()
  for mod in ('agent.Execution_Agent', 'tools.alpaca.server'):
    if mod in sys.modules:
      del sys.modules[mod]
  with patch.dict('sys.modules', fake_modules):
    from tools.alpaca import server as alpaca_srv
    srv = alpaca_srv.AlpacaServer()
    result = asyncio.run(srv.close_paper_position(args))
  parsed = json.loads(result[0].text)
  return parsed, fake_client


def test_tool_listed():
  from tools.alpaca import server as alpaca_srv
  srv = alpaca_srv.AlpacaServer()
  tools = asyncio.run(srv.list_tools_descriptors())
  names = [t.name for t in tools]
  assert 'close_paper_position' in names, f"missing: {names}"
  print(f"PASS: close_paper_position in descriptors ({len(names)} tools total)")


def test_close_existing_long_submits_sell():
  init_schema(); _clean()
  open_position("A6_AAPL", "long", 10, 195.0, paper=True)
  out, client = _call_close({"ticker": "A6_AAPL", "reason": "stop_loss_hit"})
  assert out.get('success') is True, f"close should succeed: {out}"
  assert out.get('side') == 'sell', f"closing a long should submit sell: {out}"
  assert client.submit_order.call_count == 1
  print(f"PASS: long closed via opposing sell (broker called once)")
  _clean()


def test_close_when_no_open_position_returns_error():
  init_schema(); _clean()
  out, client = _call_close({"ticker": "A6_NONE", "reason": "test"})
  assert out.get('success') is False, f"no-position case should fail gracefully: {out}"
  assert out.get('error') == 'no_open_position' or 'no_open_position' in str(out.get('error', ''))
  assert client.submit_order.call_count == 0
  print(f"PASS: no-position case returned clean error without broker call")


def test_reason_required():
  init_schema(); _clean()
  open_position("A6_REQ", "long", 5, 100.0, paper=True)
  # Missing reason should be rejected
  out, client = _call_close({"ticker": "A6_REQ"})
  assert out.get('success') is False, "reason missing should fail validation"
  assert 'reason' in (out.get('error') or '').lower()
  assert client.submit_order.call_count == 0
  print(f"PASS: missing reason rejected before broker call")
  _clean()


def test_close_short_submits_buy():
  init_schema(); _clean()
  open_position("A6_SHORT", "short", 10, 195.0, paper=True)
  out, client = _call_close({"ticker": "A6_SHORT", "reason": "thesis_broken"})
  assert out.get('success') is True
  assert out.get('side') == 'buy', f"closing a short should submit buy: {out}"
  print(f"PASS: short closed via opposing buy")
  _clean()


def test_broker_failure_handled_gracefully():
  init_schema(); _clean()
  open_position("A6_FAIL", "long", 10, 195.0, paper=True)
  os.environ['ALPACA_PAPER_KEY'] = 'fake'
  os.environ['ALPACA_PAPER_SECRET'] = 'fake'
  fake_modules, fake_client = _make_fake_alpaca()
  fake_client.submit_order.side_effect = Exception("broker down")
  for mod in ('agent.Execution_Agent', 'tools.alpaca.server'):
    if mod in sys.modules:
      del sys.modules[mod]
  with patch.dict('sys.modules', fake_modules):
    from tools.alpaca import server as alpaca_srv
    srv = alpaca_srv.AlpacaServer()
    result = asyncio.run(srv.close_paper_position(
      {"ticker": "A6_FAIL", "reason": "test"}))
  out = json.loads(result[0].text)
  assert out.get('success') is False
  assert 'broker down' in str(out)
  print(f"PASS: broker failure surfaced without raise")
  _clean()


if __name__ == "__main__":
  test_tool_listed()
  test_close_existing_long_submits_sell()
  test_close_when_no_open_position_returns_error()
  test_reason_required()
  test_close_short_submits_buy()
  test_broker_failure_handled_gracefully()
  print("\nAll Phase A6 close_paper_position tests passed.")
