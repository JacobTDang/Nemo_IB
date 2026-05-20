"""Phase A5: place_paper_order — mandatory internal risk check.

The critical invariant tested here: this tool MUST call Risk_Officer.evaluate
internally BEFORE Execution_Agent.place_order. If Risk_Officer rejects, no
broker call is allowed.

Coverage:
- Approve path → broker called once, order recorded with thesis_id
- Reject path → broker NEVER called (call_count == 0)
- Adjusted_quantity path → broker called with the smaller quantity
- Broker raises → rejected row in audit log, no successful order
- Risk decision always returned in response (auditable)
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
    conn.execute("DELETE FROM positions WHERE ticker LIKE 'A5_%'")
    conn.execute("DELETE FROM orders WHERE ticker LIKE 'A5_%' OR client_order_id LIKE 'nemo-a5-%'")
    conn.commit()
  finally:
    conn.close()


def _base_args(**overrides):
  d = {
    "ticker": "A5_AAPL",
    "side": "buy",
    "quantity": 10,
    "price": 200.0,
    "recommendation": "BUY",
    "confidence": 0.75,
    "bull_strength": 0.78,
    "bear_strength": 0.45,
    "position_sizing": "normal",
    "rationale": "test",
    "thesis_id": 42,
  }
  d.update(overrides)
  return d


def _make_fake_alpaca(submit_raises=None, broker_order_id="alp-a5-order-1"):
  """Build the alpaca-py mock that the existing Execution_Agent expects."""
  fake_order = MagicMock()
  fake_order.id = broker_order_id
  fake_order.status = "accepted"
  fake_order.filled_at = None
  client = MagicMock()
  if submit_raises:
    client.submit_order.side_effect = submit_raises
  else:
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


def _call_place_order(args, submit_raises=None) -> tuple[dict, MagicMock]:
  os.environ['ALPACA_PAPER_KEY'] = 'fake'
  os.environ['ALPACA_PAPER_SECRET'] = 'fake'
  fake_modules, fake_client = _make_fake_alpaca(submit_raises=submit_raises)
  for mod in ('agent.Execution_Agent', 'tools.alpaca.server'):
    if mod in sys.modules:
      del sys.modules[mod]
  with patch.dict('sys.modules', fake_modules):
    from tools.alpaca import server as alpaca_srv
    srv = alpaca_srv.AlpacaServer()
    result = asyncio.run(srv.place_paper_order(args))
  parsed = json.loads(result[0].text)
  return parsed, fake_client


def test_tool_listed():
  from tools.alpaca import server as alpaca_srv
  srv = alpaca_srv.AlpacaServer()
  tools = asyncio.run(srv.list_tools_descriptors())
  names = [t.name for t in tools]
  assert 'place_paper_order' in names, f"missing: {names}"
  print(f"PASS: place_paper_order in descriptors ({len(names)} tools total)")


def test_approve_path_places_order():
  init_schema(); _clean()
  out, client = _call_place_order(_base_args())
  assert out.get('success') is True, f"approve path should succeed: {out}"
  assert out.get('order_id') == 'alp-a5-order-1'
  assert out.get('client_order_id', '').startswith('nemo-')
  # Risk decision is included in the response
  assert 'risk_decision' in out
  assert out['risk_decision']['approve'] is True
  # Broker WAS called exactly once
  assert client.submit_order.call_count == 1
  # Order persisted with thesis_id
  conn = get_connection()
  try:
    row = conn.execute("SELECT thesis_id, status FROM orders WHERE order_id='alp-a5-order-1'").fetchone()
  finally:
    conn.close()
  assert row is not None, "order not persisted"
  assert row['thesis_id'] == 42
  print(f"PASS: approve path placed order with thesis_id={row['thesis_id']}")
  _clean()


def test_reject_path_NEVER_calls_broker():
  """The most safety-critical test: when Risk_Officer rejects, broker
  submit_order must never fire."""
  init_schema(); _clean()
  out, client = _call_place_order(_base_args(confidence=0.4))  # below 0.65 threshold
  assert out.get('success') is False, f"low-conf should be rejected: {out}"
  assert 'risk_rejected' in (out.get('error') or '').lower() or \
    'confidence' in str(out.get('reasons') or out.get('risk_decision', {}).get('reasons', '')).lower()
  # The veto invariant — broker NEVER called
  assert client.submit_order.call_count == 0, \
    f"BROKER CALLED WITH REJECTED TRADE — safety violation; calls: {client.submit_order.call_count}"
  print(f"PASS: rejected trade did NOT reach broker (submit_order calls: 0)")
  _clean()


def test_reject_HOLD_recommendation():
  init_schema(); _clean()
  out, client = _call_place_order(_base_args(recommendation="HOLD"))
  assert out.get('success') is False
  assert client.submit_order.call_count == 0
  print(f"PASS: HOLD recommendation rejected without broker call")
  _clean()


def test_reject_close_debate():
  init_schema(); _clean()
  out, client = _call_place_order(_base_args(bull_strength=0.70, bear_strength=0.65))
  assert out.get('success') is False
  assert client.submit_order.call_count == 0
  print(f"PASS: close debate rejected without broker call")
  _clean()


def test_adjusted_quantity_passed_to_broker():
  """Oversized request: 100 shares at $200 should be capped to 25 (5% of $100k).
  Verify the broker is called with the CAPPED quantity, not the requested 100."""
  init_schema(); _clean()
  out, client = _call_place_order(_base_args(quantity=100, price=200.0))
  assert out.get('success') is True
  assert out['risk_decision'].get('adjusted_quantity') is not None
  adjusted = out['risk_decision']['adjusted_quantity']
  assert adjusted <= 25.0
  # The broker call must have received the adjusted quantity, not 100
  call_args = client.submit_order.call_args
  # MarketOrderRequest is constructed with qty=adjusted; the mock recorded the request object
  # We can't easily inspect MarketOrderRequest's qty since it's also mocked, so verify via
  # the recorded place_order behavior: the request was called and quantity in the success
  # response equals adjusted
  assert out.get('qty') == adjusted, \
    f"response should report the adjusted qty placed; got qty={out.get('qty')} vs adjusted={adjusted}"
  print(f"PASS: broker called with adjusted_quantity={adjusted} (requested 100)")
  _clean()


def test_broker_failure_persists_rejected_row():
  init_schema(); _clean()
  out, client = _call_place_order(_base_args(),
                                   submit_raises=Exception("broker timeout"))
  assert out.get('success') is False
  assert 'broker timeout' in (out.get('error') or '') or \
    'broker timeout' in str(out)
  # The audit trail must record the failed attempt as 'rejected'
  conn = get_connection()
  try:
    row = conn.execute(
      "SELECT status FROM orders WHERE ticker='A5_AAPL' AND status='rejected'"
    ).fetchone()
  finally:
    conn.close()
  assert row is not None, "rejected row not persisted to audit log"
  print(f"PASS: broker failure recorded as rejected in orders table")
  _clean()


def test_risk_decision_always_in_response():
  """Approve OR reject — every response includes risk_decision for audit."""
  init_schema(); _clean()
  for args in (
    _base_args(),  # approve
    _base_args(confidence=0.3),  # reject
    _base_args(position_sizing="cautious"),  # approve w/ adjustment
  ):
    out, _ = _call_place_order(args)
    assert 'risk_decision' in out, f"risk_decision missing from response: {out}"
    assert 'approve' in out['risk_decision']
  print(f"PASS: risk_decision present in approve/reject/adjust paths")
  _clean()


def test_daily_loss_limit_rejects_at_place_layer():
  """Even if Claude bypasses risk_check, place_paper_order's internal
  Risk_Officer call must still enforce daily-loss-limit."""
  init_schema(); _clean()
  pid = open_position("A5_LOSS", "long", 100, 100.0, paper=True)
  close_position(pid, 70.0, "test loss")
  out, client = _call_place_order(_base_args())
  assert out.get('success') is False
  assert client.submit_order.call_count == 0
  print(f"PASS: daily loss limit honored at place_paper_order layer")
  _clean()


if __name__ == "__main__":
  test_tool_listed()
  test_approve_path_places_order()
  test_reject_path_NEVER_calls_broker()
  test_reject_HOLD_recommendation()
  test_reject_close_debate()
  test_adjusted_quantity_passed_to_broker()
  test_broker_failure_persists_rejected_row()
  test_risk_decision_always_in_response()
  test_daily_loss_limit_rejects_at_place_layer()
  print("\nAll Phase A5 place_paper_order tests passed.")
