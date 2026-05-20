"""Phase A5: place_paper_order — mandatory internal risk check.

Refactored for A9: mocks AsyncBroker.submit_market_order instead of alpaca-py.
The critical invariant is the same: Risk_Officer rejection prevents any
broker call.
"""
import asyncio
import json
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from state.schema import init_schema, get_connection
from state.positions import open_position, close_position


def _clean():
  conn = get_connection()
  try:
    conn.execute("DELETE FROM positions WHERE ticker LIKE 'A5_%'")
    conn.execute("DELETE FROM orders WHERE ticker LIKE 'A5_%' OR client_order_id LIKE 'nemo-%'")
    conn.commit()
  finally:
    conn.close()


def _base_args(**overrides):
  d = {
    "ticker": "A5_AAPL", "side": "buy", "quantity": 10, "price": 200.0,
    "recommendation": "BUY", "confidence": 0.75,
    "bull_strength": 0.78, "bear_strength": 0.45,
    "position_sizing": "normal", "rationale": "test", "thesis_id": 42,
  }
  d.update(overrides)
  return d


def _fake_broker_class(submit_raises=None, order_id="alp-a5-order-1"):
  """Returns a class that records every submit_market_order call."""
  call_log = []

  class _FakeBroker:
    def __init__(self, *a, **kw):
      pass
    async def __aenter__(self): return self
    async def __aexit__(self, *a): pass
    async def submit_market_order(self, *, symbol, qty, side, client_order_id, **_):
      call_log.append({"symbol": symbol, "qty": qty, "side": side, "client_order_id": client_order_id})
      if submit_raises:
        raise submit_raises
      return {
        "id": order_id, "client_order_id": client_order_id,
        "status": "accepted", "symbol": symbol, "qty": qty, "side": side,
        "filled_at": None,
      }
  _FakeBroker.call_log = call_log
  return _FakeBroker


def _call_place(args, broker_class):
  with patch('tools.alpaca.async_broker.AsyncBroker', new=broker_class):
    from tools.alpaca import server as alpaca_srv
    srv = alpaca_srv.AlpacaServer()
    result = asyncio.run(srv.place_paper_order(args))
  return json.loads(result[0].text)


def test_tool_listed():
  from tools.alpaca import server as alpaca_srv
  srv = alpaca_srv.AlpacaServer()
  tools = asyncio.run(srv.list_tools_descriptors())
  assert 'place_paper_order' in [t.name for t in tools]
  print(f"PASS: place_paper_order in descriptors")


def test_approve_path_places_order():
  init_schema(); _clean()
  broker_class = _fake_broker_class()
  out = _call_place(_base_args(), broker_class)
  assert out.get('success') is True, f"approve: {out}"
  assert out.get('order_id') == 'alp-a5-order-1'
  assert out.get('client_order_id', '').startswith('nemo-')
  assert out['risk_decision']['approve'] is True
  assert len(broker_class.call_log) == 1, "broker should be called once on approve"
  conn = get_connection()
  try:
    row = conn.execute("SELECT thesis_id, status FROM orders WHERE order_id='alp-a5-order-1'").fetchone()
  finally:
    conn.close()
  assert row['thesis_id'] == 42 and row['status'] == 'pending'
  print(f"PASS: approve path placed order with thesis_id=42")
  _clean()


def test_reject_path_NEVER_calls_broker():
  init_schema(); _clean()
  broker_class = _fake_broker_class()
  out = _call_place(_base_args(confidence=0.4), broker_class)
  assert out.get('success') is False
  assert len(broker_class.call_log) == 0, \
    f"BROKER CALLED WITH REJECTED TRADE — safety violation; calls={broker_class.call_log}"
  print(f"PASS: rejected trade did NOT reach broker (0 calls)")
  _clean()


def test_reject_HOLD_recommendation():
  init_schema(); _clean()
  broker_class = _fake_broker_class()
  out = _call_place(_base_args(recommendation="HOLD"), broker_class)
  assert out.get('success') is False
  assert len(broker_class.call_log) == 0
  print(f"PASS: HOLD rejected without broker call")
  _clean()


def test_reject_close_debate():
  init_schema(); _clean()
  broker_class = _fake_broker_class()
  out = _call_place(_base_args(bull_strength=0.70, bear_strength=0.65), broker_class)
  assert out.get('success') is False
  assert len(broker_class.call_log) == 0
  print(f"PASS: close debate rejected")
  _clean()


def test_adjusted_quantity_passed_to_broker():
  init_schema(); _clean()
  broker_class = _fake_broker_class()
  out = _call_place(_base_args(quantity=100, price=200.0), broker_class)
  assert out.get('success') is True
  assert out['risk_decision'].get('adjusted_quantity') is not None
  adjusted = out['risk_decision']['adjusted_quantity']
  assert adjusted <= 25.0
  # broker must have been called with adjusted quantity, NOT 100
  assert broker_class.call_log[0]['qty'] == adjusted, \
    f"broker received qty={broker_class.call_log[0]['qty']} expected {adjusted}"
  print(f"PASS: broker received adjusted_quantity={adjusted} (requested 100)")
  _clean()


def test_broker_failure_persists_rejected_row():
  init_schema(); _clean()
  broker_class = _fake_broker_class(submit_raises=Exception("broker timeout"))
  out = _call_place(_base_args(), broker_class)
  assert out.get('success') is False
  assert 'broker timeout' in str(out)
  conn = get_connection()
  try:
    row = conn.execute("SELECT status FROM orders WHERE ticker='A5_AAPL' AND status='rejected'").fetchone()
  finally:
    conn.close()
  assert row is not None, "rejected row not persisted"
  print(f"PASS: broker failure recorded as rejected")
  _clean()


def test_risk_decision_always_in_response():
  init_schema(); _clean()
  for args in (
    _base_args(),
    _base_args(confidence=0.3),
    _base_args(position_sizing="cautious"),
  ):
    broker_class = _fake_broker_class()
    out = _call_place(args, broker_class)
    assert 'risk_decision' in out and 'approve' in out['risk_decision']
  print(f"PASS: risk_decision present in all paths")
  _clean()


def test_daily_loss_limit_rejects_at_place_layer():
  init_schema(); _clean()
  pid = open_position("A5_LOSS", "long", 100, 100.0, paper=True)
  close_position(pid, 70.0, "test loss")  # -$3000 = -3% of $100k
  broker_class = _fake_broker_class()
  out = _call_place(_base_args(), broker_class)
  assert out.get('success') is False
  assert len(broker_class.call_log) == 0
  print(f"PASS: daily loss limit honored")
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
