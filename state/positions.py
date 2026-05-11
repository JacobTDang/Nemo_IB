"""CRUD for positions and orders tables (Phase 6).

A position is opened when an order fills and closed when its counter-order
fills (or stop-loss/target triggers). All P&L is computed from these rows;
the broker is the source of truth for execution, this table is the audit log.
"""
import json
from datetime import datetime
from typing import Optional, List, Dict, Any

from state.schema import get_connection


# ---- Positions ------------------------------------------------------------

def open_position(
  ticker: str, side: str, quantity: float, entry_price: float,
  thesis_id: Optional[int] = None,
  stop_loss: Optional[float] = None,
  target_price: Optional[float] = None,
  paper: bool = True,
) -> int:
  """Open a new position. Returns position_id."""
  conn = get_connection()
  try:
    cur = conn.execute("""
      INSERT INTO positions
        (ticker, side, quantity, entry_price, entry_date, current_price,
         unrealized_pnl, thesis_id, stop_loss, target_price, status, paper)
      VALUES (?,?,?,?,?,?,?,?,?,?,'open',?)
    """, (ticker.upper(), side, quantity, entry_price,
          datetime.now().isoformat(), entry_price, 0.0,
          thesis_id, stop_loss, target_price, 1 if paper else 0))
    conn.commit()
    return cur.lastrowid
  finally:
    conn.close()


def close_position(position_id: int, exit_price: float, exit_reason: str) -> None:
  """Mark a position closed and compute realized P&L."""
  conn = get_connection()
  try:
    row = conn.execute("SELECT * FROM positions WHERE position_id = ?",
                       (position_id,)).fetchone()
    if not row:
      raise ValueError(f"position {position_id} not found")
    side = row['side']
    qty = row['quantity']
    entry = row['entry_price']
    realized = (exit_price - entry) * qty if side == 'long' else (entry - exit_price) * qty
    conn.execute("""
      UPDATE positions SET
        status='closed', closed_at=?, exit_price=?, realized_pnl=?, exit_reason=?,
        current_price=?, unrealized_pnl=0
      WHERE position_id=?
    """, (datetime.now().isoformat(), exit_price, realized, exit_reason,
          exit_price, position_id))
    conn.commit()
  finally:
    conn.close()


def update_position_price(position_id: int, current_price: float) -> None:
  """Refresh the current price + unrealized P&L for an open position."""
  conn = get_connection()
  try:
    row = conn.execute("SELECT side, quantity, entry_price FROM positions "
                       "WHERE position_id = ?", (position_id,)).fetchone()
    if not row:
      return
    side, qty, entry = row['side'], row['quantity'], row['entry_price']
    unrealized = (current_price - entry) * qty if side == 'long' else (entry - current_price) * qty
    conn.execute("UPDATE positions SET current_price=?, unrealized_pnl=? "
                 "WHERE position_id=?", (current_price, unrealized, position_id))
    conn.commit()
  finally:
    conn.close()


def open_positions(paper: Optional[bool] = None) -> List[Dict[str, Any]]:
  """All currently open positions, optionally filtered by paper/live."""
  conn = get_connection()
  try:
    if paper is None:
      rows = conn.execute("SELECT * FROM positions WHERE status='open'").fetchall()
    else:
      rows = conn.execute("SELECT * FROM positions WHERE status='open' AND paper=?",
                          (1 if paper else 0,)).fetchall()
    return [dict(r) for r in rows]
  finally:
    conn.close()


def position_for_ticker(ticker: str, paper: bool = True) -> Optional[Dict[str, Any]]:
  """Returns the open position for a ticker, or None."""
  conn = get_connection()
  try:
    row = conn.execute(
      "SELECT * FROM positions WHERE ticker=? AND status='open' AND paper=?",
      (ticker.upper(), 1 if paper else 0)
    ).fetchone()
    return dict(row) if row else None
  finally:
    conn.close()


def get_position(position_id: int) -> Optional[Dict[str, Any]]:
  conn = get_connection()
  try:
    row = conn.execute("SELECT * FROM positions WHERE position_id=?",
                       (position_id,)).fetchone()
    return dict(row) if row else None
  finally:
    conn.close()


def portfolio_stats(paper: bool = True, start_value: float = 100_000.0) -> Dict[str, Any]:
  """Aggregate stats used by Risk_Officer to enforce daily limits.

  start_value is the assumed starting paper-trading account balance — required
  because we don't pull live broker equity in this function.
  """
  conn = get_connection()
  try:
    # Open positions
    open_rows = conn.execute(
      "SELECT * FROM positions WHERE status='open' AND paper=?",
      (1 if paper else 0,)
    ).fetchall()
    unrealized = sum(r['unrealized_pnl'] or 0 for r in open_rows)
    open_capital = sum((r['current_price'] or r['entry_price']) * r['quantity']
                       for r in open_rows)

    # Realized P&L since EOD prior trading day (use start of today midnight)
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    todays_closes = conn.execute(
      "SELECT realized_pnl FROM positions WHERE status='closed' AND closed_at >= ?",
      (today.isoformat(),)
    ).fetchall()
    realized_today = sum(r['realized_pnl'] or 0 for r in todays_closes)

    # Positions opened today
    todays_opens = conn.execute(
      "SELECT COUNT(*) c FROM positions WHERE entry_date >= ? AND paper=?",
      (today.isoformat(), 1 if paper else 0)
    ).fetchone()['c']

    total_value = start_value + realized_today + unrealized
    daily_pnl = realized_today + unrealized
    daily_pnl_pct = daily_pnl / start_value if start_value > 0 else 0
  finally:
    conn.close()

  return {
    'total_value': round(total_value, 2),
    'starting_value': start_value,
    'open_positions_count': len(open_rows),
    'open_capital_at_risk': round(open_capital, 2),
    'unrealized_pnl': round(unrealized, 2),
    'realized_pnl_today': round(realized_today, 2),
    'daily_pnl': round(daily_pnl, 2),
    'daily_pnl_pct': round(daily_pnl_pct, 4),
    'positions_opened_today': todays_opens,
  }


# ---- Orders ---------------------------------------------------------------

def record_order(
  order_id: str, client_order_id: str,
  ticker: str, side: str, order_type: str, quantity: float,
  limit_price: Optional[float] = None,
  status: str = 'pending',
  thesis_id: Optional[int] = None,
  arbiter_verdict_id: Optional[int] = None,
  paper: bool = True,
) -> None:
  """Insert an order audit log row. Idempotent on client_order_id."""
  conn = get_connection()
  try:
    conn.execute("""
      INSERT OR IGNORE INTO orders
        (order_id, client_order_id, ticker, side, order_type, quantity,
         limit_price, status, created_at, thesis_id, arbiter_verdict_id, paper)
      VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
    """, (order_id, client_order_id, ticker.upper(), side, order_type, quantity,
          limit_price, status, datetime.now().isoformat(),
          thesis_id, arbiter_verdict_id, 1 if paper else 0))
    conn.commit()
  finally:
    conn.close()


def update_order_status(order_id: str, status: str,
                         filled_at: Optional[str] = None) -> None:
  conn = get_connection()
  try:
    if filled_at:
      conn.execute("UPDATE orders SET status=?, filled_at=? WHERE order_id=?",
                   (status, filled_at, order_id))
    else:
      conn.execute("UPDATE orders SET status=? WHERE order_id=?", (status, order_id))
    conn.commit()
  finally:
    conn.close()


def order_exists_for_client_id(client_order_id: str) -> bool:
  conn = get_connection()
  try:
    r = conn.execute("SELECT 1 FROM orders WHERE client_order_id=?",
                     (client_order_id,)).fetchone()
    return r is not None
  finally:
    conn.close()


def get_order(order_id: str) -> Optional[Dict[str, Any]]:
  conn = get_connection()
  try:
    r = conn.execute("SELECT * FROM orders WHERE order_id=?",
                     (order_id,)).fetchone()
    return dict(r) if r else None
  finally:
    conn.close()


def recent_orders(limit: int = 50, paper: Optional[bool] = None) -> List[Dict[str, Any]]:
  conn = get_connection()
  try:
    if paper is None:
      rows = conn.execute("SELECT * FROM orders ORDER BY created_at DESC LIMIT ?",
                          (limit,)).fetchall()
    else:
      rows = conn.execute("SELECT * FROM orders WHERE paper=? ORDER BY created_at DESC LIMIT ?",
                          (1 if paper else 0, limit)).fetchall()
    return [dict(r) for r in rows]
  finally:
    conn.close()
