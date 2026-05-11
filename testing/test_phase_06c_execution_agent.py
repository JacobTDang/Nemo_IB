"""Phase 6c: Execution_Agent with Alpaca mocked.

Verifies:
- order submission flow
- idempotency via client_order_id
- error handling (Alpaca exception, missing creds)
- account summary path
- close_position_for path

Real Alpaca paper roundtrip is a calendar-time gate (Phase 6 checkpoint 6.8 —
"run for 1 week, paper only") and is not part of this CI test.
"""
import sys, os
from unittest.mock import patch, MagicMock
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from state.schema import init_schema, get_connection
from state.positions import open_position, get_order, recent_orders


def _clean():
  conn = get_connection()
  try:
    conn.execute("DELETE FROM orders WHERE ticker LIKE 'TST_%'")
    conn.execute("DELETE FROM positions WHERE ticker LIKE 'TST_%'")
    conn.commit()
  finally:
    conn.close()


def _make_alpaca_mock(broker_order_id="alp-order-abc123"):
  """Build a mock that imitates alpaca-py at the level Execution_Agent calls."""
  fake_order = MagicMock()
  fake_order.id = broker_order_id
  fake_order.status = "accepted"
  fake_order.filled_at = None

  fake_account = MagicMock()
  fake_account.equity = "100000.00"
  fake_account.cash = "100000.00"
  fake_account.buying_power = "200000.00"
  fake_account.portfolio_value = "100000.00"
  fake_account.status = "ACTIVE"

  client = MagicMock()
  client.submit_order = MagicMock(return_value=fake_order)
  client.get_account = MagicMock(return_value=fake_account)
  client.get_order_by_id = MagicMock(return_value=fake_order)

  # Mock the alpaca-py module structure
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
    '_client': client, '_order': fake_order, '_account': fake_account,
  }


def test_construction_requires_paper_creds():
  with patch.dict(os.environ, {}, clear=False):
    os.environ.pop('ALPACA_PAPER_KEY', None)
    os.environ.pop('ALPACA_PAPER_SECRET', None)
    from agent.Execution_Agent import Execution_Agent
    try:
      Execution_Agent(paper=True)
      raise AssertionError("should have raised RuntimeError for missing creds")
    except RuntimeError as e:
      assert 'paper' in str(e).lower()
      print(f"PASS: missing paper creds raised RuntimeError: {e}")


def test_construction_requires_live_creds_separately():
  with patch.dict(os.environ, {'ALPACA_PAPER_KEY': 'k', 'ALPACA_PAPER_SECRET': 's'}, clear=False):
    os.environ.pop('ALPACA_LIVE_KEY', None)
    os.environ.pop('ALPACA_LIVE_SECRET', None)
    from agent.Execution_Agent import Execution_Agent
    try:
      Execution_Agent(paper=False)
      raise AssertionError("should have raised RuntimeError for missing live creds")
    except RuntimeError as e:
      assert 'LIVE' in str(e)
      print("PASS: missing live creds raised RuntimeError (live != paper keys)")


def test_place_market_order_success():
  init_schema(); _clean()
  mocks = _make_alpaca_mock()
  with patch.dict(os.environ, {'ALPACA_PAPER_KEY': 'k', 'ALPACA_PAPER_SECRET': 's'}, clear=False), \
       patch.dict('sys.modules', mocks):
    from agent.Execution_Agent import Execution_Agent
    ea = Execution_Agent(paper=True)
    r = ea.place_order(ticker="TST_AAPL", quantity=10, side="buy",
                        order_type="market", thesis_id=42)
  assert r['success']
  assert r['order_id'] == 'alp-order-abc123'
  assert r['client_order_id'].startswith('nemo-42-')
  assert r['paper'] is True
  # Verify DB row was created
  recorded = get_order('alp-order-abc123')
  assert recorded is not None
  assert recorded['ticker'] == 'TST_AAPL'
  assert recorded['status'] == 'pending'
  assert recorded['thesis_id'] == 42
  print(f"PASS: market order placed and recorded (id={r['order_id']}, cli={r['client_order_id']})")


def test_place_limit_order_requires_price():
  mocks = _make_alpaca_mock()
  with patch.dict(os.environ, {'ALPACA_PAPER_KEY': 'k', 'ALPACA_PAPER_SECRET': 's'}, clear=False), \
       patch.dict('sys.modules', mocks):
    from agent.Execution_Agent import Execution_Agent
    ea = Execution_Agent(paper=True)
    r = ea.place_order(ticker="TST_X", quantity=10, side="buy",
                        order_type="limit", limit_price=None)
  assert not r['success']
  assert 'limit_price' in r['error']
  print("PASS: limit order without price rejected")


def test_invalid_side_rejected():
  mocks = _make_alpaca_mock()
  with patch.dict(os.environ, {'ALPACA_PAPER_KEY': 'k', 'ALPACA_PAPER_SECRET': 's'}, clear=False), \
       patch.dict('sys.modules', mocks):
    from agent.Execution_Agent import Execution_Agent
    ea = Execution_Agent(paper=True)
    r = ea.place_order(ticker="TST_X", quantity=10, side="hodl", order_type="market")
  assert not r['success']
  print("PASS: invalid side rejected before broker call")


def test_broker_failure_recorded_as_rejected():
  init_schema(); _clean()
  mocks = _make_alpaca_mock()
  mocks['_client'].submit_order.side_effect = Exception("broker unavailable")
  with patch.dict(os.environ, {'ALPACA_PAPER_KEY': 'k', 'ALPACA_PAPER_SECRET': 's'}, clear=False), \
       patch.dict('sys.modules', mocks):
    from agent.Execution_Agent import Execution_Agent
    ea = Execution_Agent(paper=True)
    r = ea.place_order(ticker="TST_FAIL", quantity=10, side="buy", order_type="market")
  assert not r['success']
  assert 'broker unavailable' in r['error']
  # Verify a rejected row was recorded
  orders = recent_orders(limit=20)
  tst_orders = [o for o in orders if o['ticker'] == 'TST_FAIL']
  assert tst_orders, "rejected order should be recorded for audit"
  assert tst_orders[0]['status'] == 'rejected'
  print(f"PASS: broker failure persisted as rejected for audit trail")


def test_idempotency_via_existing_client_id():
  """If a duplicate client_order_id slips through, the second call must reject."""
  init_schema(); _clean()
  mocks = _make_alpaca_mock()
  with patch.dict(os.environ, {'ALPACA_PAPER_KEY': 'k', 'ALPACA_PAPER_SECRET': 's'}, clear=False), \
       patch.dict('sys.modules', mocks), \
       patch('uuid.uuid4') as mock_uuid:
    # Force the same uuid each time
    mock_uuid.return_value.hex = 'deadbeef12345678'
    from agent.Execution_Agent import Execution_Agent
    ea = Execution_Agent(paper=True)
    r1 = ea.place_order(ticker="TST_IDEM", quantity=10, side="buy",
                         order_type="market", thesis_id=99)
    r2 = ea.place_order(ticker="TST_IDEM", quantity=10, side="buy",
                         order_type="market", thesis_id=99)
  assert r1['success']
  assert not r2['success'], "duplicate client_order_id should reject"
  assert r2['error'] == 'duplicate_client_order_id'
  # submit_order should have been called exactly once
  assert mocks['_client'].submit_order.call_count == 1, \
    f"broker should not have been called twice; was {mocks['_client'].submit_order.call_count}"
  print("PASS: duplicate client_order_id rejected before broker (broker called once only)")


def test_account_summary_returns_floats():
  mocks = _make_alpaca_mock()
  with patch.dict(os.environ, {'ALPACA_PAPER_KEY': 'k', 'ALPACA_PAPER_SECRET': 's'}, clear=False), \
       patch.dict('sys.modules', mocks):
    from agent.Execution_Agent import Execution_Agent
    ea = Execution_Agent(paper=True)
    a = ea.get_account_summary()
  assert a['paper'] is True
  assert a['equity'] == 100_000.0
  assert isinstance(a['equity'], float)
  print(f"PASS: account summary coerces strings to float ({a})")


def test_account_summary_handles_broker_error():
  mocks = _make_alpaca_mock()
  mocks['_client'].get_account.side_effect = Exception("API down")
  with patch.dict(os.environ, {'ALPACA_PAPER_KEY': 'k', 'ALPACA_PAPER_SECRET': 's'}, clear=False), \
       patch.dict('sys.modules', mocks):
    from agent.Execution_Agent import Execution_Agent
    ea = Execution_Agent(paper=True)
    a = ea.get_account_summary()
  assert 'error' in a
  print("PASS: account summary surfaces broker error without raising")


def test_close_position_when_none_open():
  init_schema(); _clean()
  mocks = _make_alpaca_mock()
  with patch.dict(os.environ, {'ALPACA_PAPER_KEY': 'k', 'ALPACA_PAPER_SECRET': 's'}, clear=False), \
       patch.dict('sys.modules', mocks):
    from agent.Execution_Agent import Execution_Agent
    ea = Execution_Agent(paper=True)
    r = ea.close_position_for("TST_NOTHING")
  assert not r['success']
  assert r['error'] == 'no_open_position'
  print("PASS: close_position_for returns clean error when no open position")


def test_close_position_submits_opposing_order():
  """When there's an open long, close_position_for should issue a SELL."""
  init_schema(); _clean()
  open_position("TST_CLS", "long", 10, 200.0, thesis_id=7)
  mocks = _make_alpaca_mock(broker_order_id="alp-close-1")
  with patch.dict(os.environ, {'ALPACA_PAPER_KEY': 'k', 'ALPACA_PAPER_SECRET': 's'}, clear=False), \
       patch.dict('sys.modules', mocks):
    from agent.Execution_Agent import Execution_Agent
    ea = Execution_Agent(paper=True)
    r = ea.close_position_for("TST_CLS", reason="stop_loss")
  assert r['success']
  assert r['side'] == 'sell'
  print(f"PASS: closing a long position submits a SELL order")


if __name__ == "__main__":
  test_construction_requires_paper_creds()
  test_construction_requires_live_creds_separately()
  test_place_market_order_success()
  test_place_limit_order_requires_price()
  test_invalid_side_rejected()
  test_broker_failure_recorded_as_rejected()
  test_idempotency_via_existing_client_id()
  test_account_summary_returns_floats()
  test_account_summary_handles_broker_error()
  test_close_position_when_none_open()
  test_close_position_submits_opposing_order()
  _clean()
  print("\nAll Phase 6c execution agent tests passed.")
