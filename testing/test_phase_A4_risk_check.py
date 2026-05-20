"""Phase A4: risk_check_proposed_trade — wraps Risk_Officer.evaluate().

Critical invariant: this tool ONLY evaluates. It must never call Alpaca's
submit_order regardless of input. The tool's role is to expose the
deterministic Python Risk_Officer to Claude Code via MCP so that Claude
can pre-check a trade before calling place_paper_order in A5.

Test coverage mirrors testing/test_phase_06a_risk_officer.py but at the
MCP-tool layer — every rejection path that Risk_Officer enforces must
also surface from this tool.
"""
import asyncio
import json
import os
import sys
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from state.schema import init_schema, get_connection
from state.positions import open_position, close_position


def _clean():
  conn = get_connection()
  try:
    conn.execute("DELETE FROM positions WHERE ticker LIKE 'A4_%'")
    conn.commit()
  finally:
    conn.close()


def _base_args(**overrides):
  d = {
    "ticker": "A4_AAPL",
    "side": "buy",
    "quantity": 10,
    "price": 200.0,
    "recommendation": "BUY",
    "confidence": 0.75,
    "bull_strength": 0.78,
    "bear_strength": 0.45,
    "position_sizing": "normal",
    "rationale": "test",
  }
  d.update(overrides)
  return d


def _call_risk_check(args, fake_submit_order_mock=None) -> dict:
  """Invoke the MCP tool method. If a submit_order mock is provided, attach
  it to the fake Alpaca client so we can assert it's never called."""
  os.environ['ALPACA_PAPER_KEY'] = 'fake'
  os.environ['ALPACA_PAPER_SECRET'] = 'fake'
  fake_client = MagicMock()
  fake_client.submit_order = fake_submit_order_mock or MagicMock()
  trading_client_module = MagicMock()
  trading_client_module.TradingClient = MagicMock(return_value=fake_client)
  fake_modules = {
    'alpaca.trading.client': trading_client_module,
    'alpaca.trading.requests': MagicMock(),
    'alpaca.trading.enums': MagicMock(),
  }
  for mod in ('agent.Execution_Agent', 'tools.alpaca.server'):
    if mod in sys.modules:
      del sys.modules[mod]
  with patch.dict('sys.modules', fake_modules):
    from tools.alpaca import server as alpaca_srv
    srv = alpaca_srv.AlpacaServer()
    result = asyncio.run(srv.risk_check_proposed_trade(args))
  parsed = json.loads(result[0].text)
  return parsed, fake_client


def test_tool_listed():
  from tools.alpaca import server as alpaca_srv
  srv = alpaca_srv.AlpacaServer()
  tools = asyncio.run(srv.list_tools_descriptors())
  names = [t.name for t in tools]
  assert 'risk_check_proposed_trade' in names, f"missing from tools: {names}"
  print(f"PASS: risk_check_proposed_trade in descriptors ({names})")


def test_clean_trade_approves():
  init_schema(); _clean()
  out, client = _call_risk_check(_base_args())
  assert out.get('approve') is True, f"clean trade should approve: {out}"
  client.submit_order.assert_not_called()
  print(f"PASS: clean trade approved (reasons={out.get('reasons')})")


def test_hold_recommendation_rejects():
  init_schema(); _clean()
  out, _ = _call_risk_check(_base_args(recommendation="HOLD"))
  assert out.get('approve') is False
  assert any('HOLD' in r for r in out.get('reasons', []))
  print(f"PASS: HOLD rejected")


def test_low_confidence_rejects():
  init_schema(); _clean()
  out, _ = _call_risk_check(_base_args(confidence=0.5))
  assert out.get('approve') is False
  assert any('confidence' in r.lower() for r in out.get('reasons', []))
  print(f"PASS: conf=0.5 rejected (threshold 0.65)")


def test_close_debate_rejects():
  init_schema(); _clean()
  # spread = |0.7 - 0.65| = 0.05, below 0.20 threshold
  out, _ = _call_risk_check(_base_args(bull_strength=0.7, bear_strength=0.65))
  assert out.get('approve') is False
  assert any('spread' in r.lower() for r in out.get('reasons', []))
  print(f"PASS: close debate rejected")


def test_no_position_sizing_rejects():
  init_schema(); _clean()
  out, _ = _call_risk_check(_base_args(position_sizing="no_position"))
  assert out.get('approve') is False
  print(f"PASS: no_position sizing rejected")


def test_oversized_request_capped():
  init_schema(); _clean()
  # 100 shares @ $200 = $20000 vs 5% of $100k starting = $5000 = 25 shares
  out, _ = _call_risk_check(_base_args(quantity=100, price=200.0))
  assert out.get('approve') is True
  assert out.get('adjusted_quantity') is not None
  assert out['adjusted_quantity'] <= 25
  print(f"PASS: oversized capped to {out['adjusted_quantity']}")


def test_cautious_sizing_halves():
  init_schema(); _clean()
  out, _ = _call_risk_check(_base_args(position_sizing="cautious"))
  assert out.get('approve') is True
  assert out.get('adjusted_quantity') == 5
  print(f"PASS: cautious sizing halves to {out['adjusted_quantity']}")


def test_aggressive_low_conf_downgrades():
  init_schema(); _clean()
  out, _ = _call_risk_check(_base_args(position_sizing="aggressive", confidence=0.70))
  assert out.get('approve') is True
  assert out.get('adjusted_quantity') == 5  # half of 10
  assert any('aggressive' in r.lower() for r in out.get('reasons', []))
  print(f"PASS: aggressive + conf<0.75 downgraded to {out['adjusted_quantity']}")


def test_scale_up_existing_position_approves():
  """Adding to an existing position should approve (round 2 fix). The tool
  must pass open_basket into Risk_Officer.evaluate()."""
  init_schema(); _clean()
  open_position("A4_AAPL", "long", 10, 195.0, paper=True)
  out, _ = _call_risk_check(_base_args(ticker="A4_AAPL"))
  assert out.get('approve') is True, f"scale-up should approve; got {out}"
  print(f"PASS: scale-up approved")
  close_position(1, 195.0, "test cleanup") if False else _clean()


def test_tool_NEVER_calls_submit_order():
  """Critical safety invariant: regardless of approve/reject, this tool
  must never call broker.submit_order. It only evaluates."""
  init_schema(); _clean()
  for args in (
    _base_args(),  # approve path
    _base_args(recommendation="HOLD"),  # reject path
    _base_args(quantity=999, price=500.0),  # cap path
  ):
    out, client = _call_risk_check(args)
    client.submit_order.assert_not_called()
  print(f"PASS: tool never calls submit_order across approve/reject/cap paths")


def test_daily_loss_limit_rejects():
  """Seed today's portfolio with a closed position whose realized loss
  exceeds 2% of start_value, then verify Risk_Officer rejects new trades."""
  init_schema(); _clean()
  pid = open_position("A4_LOSS", "long", 100, 100.0, paper=True)
  close_position(pid, 70.0, "test loss")  # -$3000 = -3% of $100k
  out, _ = _call_risk_check(_base_args())
  assert out.get('approve') is False
  assert any('loss limit' in r.lower() for r in out.get('reasons', []))
  print(f"PASS: daily loss limit rejection works at MCP layer")
  _clean()


if __name__ == "__main__":
  test_tool_listed()
  test_clean_trade_approves()
  test_hold_recommendation_rejects()
  test_low_confidence_rejects()
  test_close_debate_rejects()
  test_no_position_sizing_rejects()
  test_oversized_request_capped()
  test_cautious_sizing_halves()
  test_aggressive_low_conf_downgrades()
  test_scale_up_existing_position_approves()
  test_tool_NEVER_calls_submit_order()
  test_daily_loss_limit_rejects()
  print("\nAll Phase A4 risk_check_proposed_trade tests passed.")
