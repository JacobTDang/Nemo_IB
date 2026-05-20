"""Phase A3: get_paper_positions — broker-vs-local reconciliation.

The MCP tool returns BOTH the broker's view (Alpaca get_all_positions) and
the local view (state.positions.open_positions(paper=True)), plus a
`reconciled` boolean and a `discrepancies` list naming positions that
exist in one but not the other.

This guards against paper-trading divergence between broker state and our
audit log — a real failure mode the Phase 6 audit invariant required.
"""
import asyncio
import json
import os
import sys
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from state.schema import init_schema, get_connection
from state.positions import open_position


def _clean_test_positions():
  conn = get_connection()
  try:
    conn.execute("DELETE FROM positions WHERE ticker LIKE 'A3_%'")
    conn.commit()
  finally:
    conn.close()


def _fake_broker_position(symbol, qty="10", side="long"):
  p = MagicMock()
  p.symbol = symbol
  p.qty = qty
  p.side = side
  p.market_value = "1000.0"
  p.avg_entry_price = "100.0"
  return p


def _fake_alpaca_modules(broker_positions):
  client = MagicMock()
  client.get_all_positions.return_value = broker_positions
  trading_client_module = MagicMock()
  trading_client_module.TradingClient = MagicMock(return_value=client)
  return {
    'alpaca.trading.client': trading_client_module,
    'alpaca.trading.requests': MagicMock(),
    'alpaca.trading.enums': MagicMock(),
  }, client


def _call_get_positions() -> dict:
  os.environ['ALPACA_PAPER_KEY'] = 'fake_key'
  os.environ['ALPACA_PAPER_SECRET'] = 'fake_secret'
  # Force re-import so the env vars + sys.modules patches take effect
  for mod in ('agent.Execution_Agent', 'tools.alpaca.server'):
    if mod in sys.modules:
      del sys.modules[mod]
  from tools.alpaca import server as alpaca_srv
  srv = alpaca_srv.AlpacaServer()
  result = asyncio.run(srv.get_paper_positions())
  return json.loads(result[0].text)


def test_reconciled_when_broker_and_local_match():
  init_schema(); _clean_test_positions()
  open_position("A3_AAPL", "long", 10, 195.0, paper=True)
  open_position("A3_NVDA", "long", 5, 800.0, paper=True)
  broker_pos = [_fake_broker_position("A3_AAPL"), _fake_broker_position("A3_NVDA")]
  fake_modules, _ = _fake_alpaca_modules(broker_pos)
  with patch.dict('sys.modules', fake_modules):
    out = _call_get_positions()
  assert out.get('reconciled') is True, f"should reconcile when sets match; got {out}"
  assert out.get('discrepancies') == [], f"no discrepancies expected; got {out.get('discrepancies')}"
  assert len(out.get('broker_positions', [])) == 2
  assert len(out.get('local_positions', [])) == 2
  print(f"PASS: matching sets -> reconciled=True, 0 discrepancies")
  _clean_test_positions()


def test_discrepancy_when_local_missing_broker_position():
  init_schema(); _clean_test_positions()
  # Local has only AAPL; broker has AAPL + MSFT
  open_position("A3_AAPL", "long", 10, 195.0, paper=True)
  broker_pos = [_fake_broker_position("A3_AAPL"), _fake_broker_position("A3_MSFT")]
  fake_modules, _ = _fake_alpaca_modules(broker_pos)
  with patch.dict('sys.modules', fake_modules):
    out = _call_get_positions()
  assert out.get('reconciled') is False
  discreps = out.get('discrepancies', [])
  assert any('missing_locally:A3_MSFT' in d for d in discreps), \
    f"expected missing_locally:A3_MSFT in discrepancies; got {discreps}"
  print(f"PASS: broker-only position flagged: {discreps}")
  _clean_test_positions()


def test_discrepancy_when_broker_missing_local_position():
  init_schema(); _clean_test_positions()
  # Local has AAPL + GHOST; broker has only AAPL
  open_position("A3_AAPL", "long", 10, 195.0, paper=True)
  open_position("A3_GHOST", "long", 1, 50.0, paper=True)
  broker_pos = [_fake_broker_position("A3_AAPL")]
  fake_modules, _ = _fake_alpaca_modules(broker_pos)
  with patch.dict('sys.modules', fake_modules):
    out = _call_get_positions()
  assert out.get('reconciled') is False
  discreps = out.get('discrepancies', [])
  assert any('missing_at_broker:A3_GHOST' in d for d in discreps), \
    f"expected missing_at_broker:A3_GHOST; got {discreps}"
  print(f"PASS: local-only position flagged: {discreps}")
  _clean_test_positions()


def test_empty_broker_and_local_reconcile_clean():
  init_schema(); _clean_test_positions()
  fake_modules, _ = _fake_alpaca_modules([])
  with patch.dict('sys.modules', fake_modules):
    out = _call_get_positions()
  assert out.get('reconciled') is True
  assert out.get('broker_positions', []) == []
  print(f"PASS: empty broker + empty local -> reconciled=True")
  _clean_test_positions()


def test_broker_error_surfaces_gracefully():
  init_schema(); _clean_test_positions()
  fake_modules, fake_client = _fake_alpaca_modules([])
  fake_client.get_all_positions.side_effect = Exception("broker timeout")
  with patch.dict('sys.modules', fake_modules):
    out = _call_get_positions()
  assert 'error' in out, f"broker error should surface in response; got {out}"
  # local positions should still be reported even when broker fails
  assert 'local_positions' in out
  print(f"PASS: broker error surfaced without raise (error={out['error'][:60]})")
  _clean_test_positions()


def test_tool_listed_in_descriptors():
  from tools.alpaca import server as alpaca_srv
  srv = alpaca_srv.AlpacaServer()
  tools = asyncio.run(srv.list_tools_descriptors())
  names = [t.name for t in tools]
  assert 'get_paper_positions' in names, f"missing from tools: {names}"
  print(f"PASS: get_paper_positions in descriptors ({names})")


if __name__ == "__main__":
  test_tool_listed_in_descriptors()
  test_reconciled_when_broker_and_local_match()
  test_discrepancy_when_local_missing_broker_position()
  test_discrepancy_when_broker_missing_local_position()
  test_empty_broker_and_local_reconcile_clean()
  test_broker_error_surfaces_gracefully()
  print("\nAll Phase A3 get_paper_positions tests passed.")
