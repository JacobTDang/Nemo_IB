"""Phase A3: get_paper_positions — broker-vs-local reconciliation.

Refactored for A9: mocks AsyncBroker.get_all_positions instead of alpaca-py.
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
    conn.execute("DELETE FROM positions WHERE ticker LIKE 'A3_%'")
    conn.commit()
  finally:
    conn.close()


def _fake_broker_class(positions=None, raises=None):
  class _FakeBroker:
    def __init__(self, *a, **kw):
      self._positions = positions or []
      self._raises = raises
    async def __aenter__(self): return self
    async def __aexit__(self, *a): pass
    async def get_all_positions(self):
      if self._raises:
        raise self._raises
      return self._positions
  return _FakeBroker


def _fake_pos(symbol, qty=10):
  return {"symbol": symbol.upper(), "qty": float(qty), "side": "long",
          "market_value": 1000.0, "avg_entry_price": 100.0}


def _call_get_positions() -> dict:
  from tools.alpaca import server as alpaca_srv
  srv = alpaca_srv.AlpacaServer()
  result = asyncio.run(srv.get_paper_positions())
  return json.loads(result[0].text)


def test_reconciled_when_broker_and_local_match():
  init_schema(); _clean()
  open_position("A3_AAPL", "long", 10, 195.0, paper=True)
  open_position("A3_NVDA", "long", 5, 800.0, paper=True)
  positions = [_fake_pos("A3_AAPL"), _fake_pos("A3_NVDA")]
  with patch('tools.alpaca.async_broker.AsyncBroker', new=_fake_broker_class(positions)):
    out = _call_get_positions()
  assert out.get('reconciled') is True, f"should reconcile; got {out}"
  assert out.get('discrepancies') == []
  assert len(out.get('broker_positions', [])) == 2
  assert len(out.get('local_positions', [])) == 2
  print("PASS: matching sets -> reconciled=True")
  _clean()


def test_discrepancy_when_local_missing_broker_position():
  init_schema(); _clean()
  open_position("A3_AAPL", "long", 10, 195.0, paper=True)
  positions = [_fake_pos("A3_AAPL"), _fake_pos("A3_MSFT")]
  with patch('tools.alpaca.async_broker.AsyncBroker', new=_fake_broker_class(positions)):
    out = _call_get_positions()
  assert out.get('reconciled') is False
  assert any('missing_locally:A3_MSFT' in d for d in out['discrepancies'])
  print(f"PASS: broker-only flagged: {out['discrepancies']}")
  _clean()


def test_discrepancy_when_broker_missing_local_position():
  init_schema(); _clean()
  open_position("A3_AAPL", "long", 10, 195.0, paper=True)
  open_position("A3_GHOST", "long", 1, 50.0, paper=True)
  positions = [_fake_pos("A3_AAPL")]
  with patch('tools.alpaca.async_broker.AsyncBroker', new=_fake_broker_class(positions)):
    out = _call_get_positions()
  assert out.get('reconciled') is False
  assert any('missing_at_broker:A3_GHOST' in d for d in out['discrepancies'])
  print(f"PASS: local-only flagged: {out['discrepancies']}")
  _clean()


def test_empty_broker_and_local_reconcile_clean():
  init_schema(); _clean()
  with patch('tools.alpaca.async_broker.AsyncBroker', new=_fake_broker_class([])):
    out = _call_get_positions()
  assert out.get('reconciled') is True
  assert out.get('broker_positions', []) == []
  print("PASS: empty + empty -> reconciled=True")
  _clean()


def test_broker_error_surfaces_gracefully():
  init_schema(); _clean()
  with patch('tools.alpaca.async_broker.AsyncBroker',
              new=_fake_broker_class(raises=Exception("broker timeout"))):
    out = _call_get_positions()
  assert 'error' in out
  assert 'local_positions' in out  # local side should still come back
  print(f"PASS: broker error surfaced without raise")
  _clean()


def test_tool_listed_in_descriptors():
  from tools.alpaca import server as alpaca_srv
  srv = alpaca_srv.AlpacaServer()
  tools = asyncio.run(srv.list_tools_descriptors())
  assert 'get_paper_positions' in [t.name for t in tools]
  print(f"PASS: get_paper_positions in descriptors")


if __name__ == "__main__":
  test_tool_listed_in_descriptors()
  test_reconciled_when_broker_and_local_match()
  test_discrepancy_when_local_missing_broker_position()
  test_discrepancy_when_broker_missing_local_position()
  test_empty_broker_and_local_reconcile_clean()
  test_broker_error_surfaces_gracefully()
  print("\nAll Phase A3 get_paper_positions tests passed.")
