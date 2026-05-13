"""Phase 6b: positions and orders table CRUD + idempotency.

No Alpaca calls. Tests the DB layer that the Execution Agent depends on."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from state.schema import init_schema, get_connection
from state.positions import (
  open_position, close_position, update_position_price,
  open_positions, position_for_ticker, get_position, portfolio_stats,
  record_order, update_order_status, order_exists_for_client_id,
  get_order, recent_orders,
)


def _clean():
  conn = get_connection()
  try:
    conn.execute("DELETE FROM positions WHERE ticker LIKE 'TST_%'")
    conn.execute("DELETE FROM orders WHERE ticker LIKE 'TST_%' OR client_order_id LIKE 'test-%'")
    conn.commit()
  finally:
    conn.close()


# ---- Positions ------------------------------------------------------------

def test_open_position_round_trip():
  init_schema(); _clean()
  pid = open_position("TST_AAPL", "long", 10, 195.0, thesis_id=1,
                       stop_loss=180.0, target_price=220.0)
  assert pid > 0
  pos = get_position(pid)
  assert pos['ticker'] == 'TST_AAPL'
  assert pos['side'] == 'long' and pos['quantity'] == 10
  assert pos['entry_price'] == 195.0
  assert pos['status'] == 'open'
  assert pos['paper'] == 1
  print(f"PASS: open_position round-trip (id={pid})")


def test_position_for_ticker_returns_open_only():
  init_schema(); _clean()
  pid1 = open_position("TST_MSFT", "long", 5, 400.0)
  close_position(pid1, 410.0, 'target_hit')
  pid2 = open_position("TST_MSFT", "long", 8, 405.0)
  pos = position_for_ticker("TST_MSFT")
  assert pos is not None
  assert pos['position_id'] == pid2, "should return the OPEN position, not the closed one"
  print(f"PASS: position_for_ticker returns only open positions")


def test_close_position_computes_pnl():
  init_schema(); _clean()
  pid = open_position("TST_NVDA", "long", 10, 800.0)
  close_position(pid, 900.0, 'target_hit')
  pos = get_position(pid)
  assert pos['status'] == 'closed'
  assert pos['exit_price'] == 900.0
  assert pos['realized_pnl'] == 1000.0, f"(900-800)*10 = 1000, got {pos['realized_pnl']}"
  print(f"PASS: long close P&L correct (+$1000)")


def test_close_short_position_pnl():
  init_schema(); _clean()
  pid = open_position("TST_NVDA", "short", 10, 800.0)
  close_position(pid, 700.0, 'target_hit')
  pos = get_position(pid)
  assert pos['realized_pnl'] == 1000.0, f"short pnl = (800-700)*10 = 1000, got {pos['realized_pnl']}"
  print(f"PASS: short close P&L correct (+$1000 on price drop)")


def test_update_position_price_recomputes_unrealized():
  init_schema(); _clean()
  pid = open_position("TST_NVDA", "long", 10, 800.0)
  update_position_price(pid, 850.0)
  pos = get_position(pid)
  assert pos['current_price'] == 850.0
  assert pos['unrealized_pnl'] == 500.0
  print(f"PASS: unrealized recomputed on price update")


def test_portfolio_stats_aggregation():
  init_schema(); _clean()
  pid1 = open_position("TST_AAPL", "long", 10, 195.0)
  update_position_price(pid1, 200.0)  # +$50
  pid2 = open_position("TST_NVDA", "long", 5, 800.0)
  update_position_price(pid2, 820.0)  # +$100
  closed = open_position("TST_TSLA", "long", 5, 250.0)
  close_position(closed, 260.0, 'manual')  # +$50 realized
  stats = portfolio_stats(paper=True, start_value=100_000)
  assert stats['open_positions_count'] == 2
  assert stats['unrealized_pnl'] == 150.0
  assert stats['realized_pnl_today'] == 50.0
  assert stats['daily_pnl'] == 200.0
  assert stats['positions_opened_today'] == 3
  print(f"PASS: portfolio stats compute correctly: {stats}")


# ---- Orders --------------------------------------------------------------

def test_portfolio_stats_isolates_paper_from_live():
  """Realized P&L today must filter by paper/live. Pre-fix, the today's-closes
  query had no paper filter, so a live close would pollute paper stats."""
  init_schema(); _clean()
  # Paper close: +$50 realized
  paper_pid = open_position("TST_PAPER", "long", 10, 100.0, paper=True)
  close_position(paper_pid, 105.0, "test")  # (105-100)*10 = $50
  # Live close: +$1000 realized — must NOT show up in paper stats
  live_pid = open_position("TST_LIVE", "long", 10, 100.0, paper=False)
  close_position(live_pid, 200.0, "test")  # (200-100)*10 = $1000

  paper_stats = portfolio_stats(paper=True, start_value=100_000)
  assert paper_stats['realized_pnl_today'] == 50.0, \
    f"paper stats must exclude live close; got {paper_stats['realized_pnl_today']} (expected 50)"

  live_stats = portfolio_stats(paper=False, start_value=100_000)
  assert live_stats['realized_pnl_today'] == 1000.0, \
    f"live stats must exclude paper close; got {live_stats['realized_pnl_today']}"
  print(f"PASS: portfolio_stats isolates paper ({paper_stats['realized_pnl_today']}) "
        f"from live ({live_stats['realized_pnl_today']})")


def test_record_order_idempotency():
  init_schema(); _clean()
  record_order(order_id="oid-1", client_order_id="test-cli-1",
               ticker="TST_AAPL", side="buy", order_type="market",
               quantity=10, status='pending', thesis_id=42)
  # Second insert with same client_order_id should be ignored
  record_order(order_id="oid-2", client_order_id="test-cli-1",
               ticker="TST_AAPL", side="buy", order_type="market",
               quantity=10, status='pending', thesis_id=42)
  conn = get_connection()
  try:
    count = conn.execute(
      "SELECT COUNT(*) c FROM orders WHERE client_order_id='test-cli-1'"
    ).fetchone()['c']
  finally:
    conn.close()
  assert count == 1, f"expected 1 row (idempotent), got {count}"
  print("PASS: record_order is idempotent on client_order_id")


def test_order_exists_for_client_id():
  init_schema(); _clean()
  assert not order_exists_for_client_id("test-not-here")
  record_order(order_id="oid-X", client_order_id="test-here",
               ticker="TST_X", side="buy", order_type="market", quantity=1)
  assert order_exists_for_client_id("test-here")
  print("PASS: order_exists_for_client_id discriminates correctly")


def test_update_order_status():
  init_schema(); _clean()
  record_order(order_id="oid-fill", client_order_id="test-fill",
               ticker="TST_AAPL", side="buy", order_type="market", quantity=10)
  update_order_status("oid-fill", "filled", filled_at="2026-05-10T10:30:00")
  o = get_order("oid-fill")
  assert o['status'] == 'filled'
  assert o['filled_at'] == '2026-05-10T10:30:00'
  print("PASS: order status update persists")


def test_recent_orders_returns_newest_first():
  init_schema(); _clean()
  for i in range(3):
    record_order(order_id=f"oid-r{i}", client_order_id=f"test-r{i}",
                 ticker=f"TST_X{i}", side='buy', order_type='market', quantity=1)
  orders = [o for o in recent_orders(limit=10) if o['client_order_id'].startswith('test-r')]
  assert len(orders) == 3
  assert orders[0]['client_order_id'] == 'test-r2', "newest first"
  print("PASS: recent_orders ordering correct")


if __name__ == "__main__":
  test_open_position_round_trip()
  test_position_for_ticker_returns_open_only()
  test_close_position_computes_pnl()
  test_close_short_position_pnl()
  test_update_position_price_recomputes_unrealized()
  test_portfolio_stats_aggregation()
  test_portfolio_stats_isolates_paper_from_live()
  test_record_order_idempotency()
  test_order_exists_for_client_id()
  test_update_order_status()
  test_recent_orders_returns_newest_first()
  _clean()
  print("\nAll Phase 6b positions/orders tests passed.")
