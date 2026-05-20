"""Phase A6: close_paper_position — opposing market order.

Refactored for A9: mocks AsyncBroker.get_open_position + close_position
instead of alpaca-py.
"""
import asyncio
import json
import os
import sys
from unittest.mock import patch

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


def _fake_broker_class(position=None, close_raises=None, close_order=None):
  call_log = []
  class _FakeBroker:
    def __init__(self, *a, **kw): pass
    async def __aenter__(self): return self
    async def __aexit__(self, *a): pass
    async def get_open_position(self, symbol):
      call_log.append(("get_open_position", symbol))
      return position
    async def close_position(self, symbol):
      call_log.append(("close_position", symbol))
      if close_raises:
        raise close_raises
      return close_order or {
        "id": "close-1", "status": "accepted",
        "symbol": symbol, "qty": "10", "side": "sell",
      }
  _FakeBroker.call_log = call_log
  return _FakeBroker


def _call_close(args, broker_class):
  with patch('tools.alpaca.async_broker.AsyncBroker', new=broker_class):
    from tools.alpaca import server as alpaca_srv
    srv = alpaca_srv.AlpacaServer()
    result = asyncio.run(srv.close_paper_position(args))
  return json.loads(result[0].text)


def test_tool_listed():
  from tools.alpaca import server as alpaca_srv
  srv = alpaca_srv.AlpacaServer()
  tools = asyncio.run(srv.list_tools_descriptors())
  assert 'close_paper_position' in [t.name for t in tools]
  print(f"PASS: close_paper_position in descriptors")


def test_close_existing_long_submits_sell():
  init_schema(); _clean()
  open_position("A6_AAPL", "long", 10, 195.0, paper=True)
  broker_class = _fake_broker_class(
    position={"symbol": "A6_AAPL", "qty": 10, "side": "long"},
    close_order={"id": "close-1", "status": "accepted",
                  "symbol": "A6_AAPL", "qty": "10", "side": "sell"},
  )
  out = _call_close({"ticker": "A6_AAPL", "reason": "stop_loss_hit"}, broker_class)
  assert out.get('success') is True, f"close should succeed: {out}"
  assert out.get('side') == 'sell'
  # broker get_open_position + close_position both called
  assert any(c[0] == 'close_position' for c in broker_class.call_log)
  print(f"PASS: long closed via opposing sell")
  _clean()


def test_close_when_no_open_position_returns_error():
  init_schema(); _clean()
  broker_class = _fake_broker_class(position=None)
  out = _call_close({"ticker": "A6_NONE", "reason": "test"}, broker_class)
  assert out.get('success') is False
  assert 'no_open_position' in (out.get('error') or '')
  # close_position should NOT be called when there's no position
  assert not any(c[0] == 'close_position' for c in broker_class.call_log)
  print(f"PASS: no-position case returned error without close call")


def test_reason_required():
  init_schema(); _clean()
  open_position("A6_REQ", "long", 5, 100.0, paper=True)
  broker_class = _fake_broker_class(
    position={"symbol": "A6_REQ", "qty": 5, "side": "long"},
  )
  out = _call_close({"ticker": "A6_REQ"}, broker_class)
  assert out.get('success') is False
  assert 'reason' in (out.get('error') or '').lower()
  # Even broker get_open_position should NOT be called when reason missing
  assert len(broker_class.call_log) == 0
  print(f"PASS: missing reason rejected before any broker call")
  _clean()


def test_close_short_submits_buy():
  init_schema(); _clean()
  open_position("A6_SHORT", "short", 10, 195.0, paper=True)
  broker_class = _fake_broker_class(
    position={"symbol": "A6_SHORT", "qty": 10, "side": "short"},
    close_order={"id": "close-2", "status": "accepted",
                  "symbol": "A6_SHORT", "qty": "10", "side": "buy"},
  )
  out = _call_close({"ticker": "A6_SHORT", "reason": "thesis_broken"}, broker_class)
  assert out.get('success') is True
  assert out.get('side') == 'buy'
  print(f"PASS: short closed via opposing buy")
  _clean()


def test_broker_failure_handled_gracefully():
  init_schema(); _clean()
  open_position("A6_FAIL", "long", 10, 195.0, paper=True)
  broker_class = _fake_broker_class(
    position={"symbol": "A6_FAIL", "qty": 10, "side": "long"},
    close_raises=Exception("broker down"),
  )
  out = _call_close({"ticker": "A6_FAIL", "reason": "test"}, broker_class)
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
