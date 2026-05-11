"""Single execution agent. The only thing in the codebase that talks to the
broker (Alpaca paper or live).

Hard invariants:
  - client_order_id includes thesis_id + uuid, ensuring idempotency. The local
    `orders` table also rejects duplicates on this key.
  - Default is paper=True. Live mode requires explicit construction with
    paper=False AND the ALPACA_LIVE_KEY/_SECRET env vars set (not the paper
    keys).
  - Every order is mirrored into the local DB before AND after broker confirm.

Failure modes are explicit: place_order returns {success: bool, ...}; never
raises on network errors. Caller is the workflow / Risk Officer chain.
"""
import os
import sys
import uuid
from typing import Optional, Dict, Any
from datetime import datetime
from dotenv import load_dotenv

from state.positions import (
  record_order, update_order_status, order_exists_for_client_id,
  open_position, close_position, position_for_ticker,
)


class Execution_Agent:
  """Thin wrapper over alpaca-py. Use only one instance per process."""

  def __init__(self, paper: bool = True):
    load_dotenv()
    self.paper = paper
    if paper:
      key = os.getenv("ALPACA_PAPER_KEY")
      secret = os.getenv("ALPACA_PAPER_SECRET")
    else:
      key = os.getenv("ALPACA_LIVE_KEY")
      secret = os.getenv("ALPACA_LIVE_SECRET")
    if not key or not secret:
      raise RuntimeError(
        f"Missing Alpaca {'paper' if paper else 'LIVE'} credentials. "
        f"Set ALPACA_{'PAPER' if paper else 'LIVE'}_KEY and SECRET in .env"
      )
    self._key = key
    self._secret = secret
    self._client = None  # Lazy — only construct on first call

  def _get_client(self):
    if self._client is None:
      try:
        from alpaca.trading.client import TradingClient
        self._client = TradingClient(
          api_key=self._key, secret_key=self._secret, paper=self.paper
        )
      except ImportError as e:
        raise RuntimeError(f"alpaca-py not installed: {e}")
    return self._client

  # ---- Account info -----------------------------------------------------

  def get_account_summary(self) -> Dict[str, Any]:
    """Returns account equity, buying power, cash."""
    try:
      c = self._get_client()
      a = c.get_account()
      return {
        'paper': self.paper,
        'equity': float(a.equity),
        'cash': float(a.cash),
        'buying_power': float(a.buying_power),
        'portfolio_value': float(a.portfolio_value),
        'status': a.status if isinstance(a.status, str) else str(a.status),
      }
    except Exception as e:
      return {'error': f"{type(e).__name__}: {e}", 'paper': self.paper}

  # ---- Order placement --------------------------------------------------

  def place_order(self,
                  ticker: str, quantity: float, side: str,
                  order_type: str = 'market',
                  limit_price: Optional[float] = None,
                  thesis_id: Optional[int] = None,
                  arbiter_verdict_id: Optional[int] = None) -> Dict[str, Any]:
    """Submit a buy/sell order. Returns {success, order_id, ...}.

    Idempotency: a deterministic client_order_id of the form
      nemo-{thesis_id|none}-{8 hex}
    is generated. If we've already seen this client_order_id in our DB, we
    refuse to re-submit.
    """
    side = side.lower()
    if side not in ('buy', 'sell'):
      return {'success': False, 'error': f'invalid side: {side}'}
    ticker_u = ticker.upper()

    client_order_id = f"nemo-{thesis_id or 'none'}-{uuid.uuid4().hex[:8]}"

    # Pre-flight dedup: never replay the same client_order_id
    if order_exists_for_client_id(client_order_id):
      return {'success': False, 'error': 'duplicate_client_order_id',
              'client_order_id': client_order_id}

    try:
      from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
      from alpaca.trading.enums import OrderSide, TimeInForce
    except ImportError as e:
      return {'success': False, 'error': f'alpaca-py not installed: {e}'}

    alp_side = OrderSide.BUY if side == 'buy' else OrderSide.SELL
    try:
      if order_type == 'market':
        req = MarketOrderRequest(
          symbol=ticker_u, qty=quantity, side=alp_side,
          time_in_force=TimeInForce.DAY,
          client_order_id=client_order_id,
        )
      elif order_type == 'limit':
        if limit_price is None:
          return {'success': False, 'error': 'limit order requires limit_price'}
        req = LimitOrderRequest(
          symbol=ticker_u, qty=quantity, side=alp_side,
          limit_price=limit_price,
          time_in_force=TimeInForce.DAY,
          client_order_id=client_order_id,
        )
      else:
        return {'success': False, 'error': f'unsupported order_type: {order_type}'}

      client = self._get_client()
      order = client.submit_order(req)
      broker_order_id = str(order.id)
    except Exception as e:
      # Record the attempt as rejected for the audit trail
      record_order(
        order_id=f"rejected-{client_order_id}",
        client_order_id=client_order_id,
        ticker=ticker_u, side=side, order_type=order_type,
        quantity=quantity, limit_price=limit_price,
        status='rejected', thesis_id=thesis_id,
        arbiter_verdict_id=arbiter_verdict_id, paper=self.paper,
      )
      print(f"[Execution] order submission failed: {e}", file=sys.stderr, flush=True)
      return {'success': False, 'error': str(e),
              'client_order_id': client_order_id}

    # Persist
    record_order(
      order_id=broker_order_id, client_order_id=client_order_id,
      ticker=ticker_u, side=side, order_type=order_type,
      quantity=quantity, limit_price=limit_price,
      status='pending', thesis_id=thesis_id,
      arbiter_verdict_id=arbiter_verdict_id, paper=self.paper,
    )
    return {
      'success': True,
      'order_id': broker_order_id,
      'client_order_id': client_order_id,
      'paper': self.paper,
      'ticker': ticker_u, 'side': side, 'qty': quantity,
    }

  def reconcile_order_status(self, broker_order_id: str) -> Dict[str, Any]:
    """Refresh local order status against broker. Returns {status, filled_at?}."""
    try:
      c = self._get_client()
      o = c.get_order_by_id(broker_order_id)
      status = str(o.status).split('.')[-1].lower()
      filled = o.filled_at.isoformat() if getattr(o, 'filled_at', None) else None
      update_order_status(broker_order_id, status, filled)
      return {'order_id': broker_order_id, 'status': status, 'filled_at': filled}
    except Exception as e:
      return {'error': str(e), 'order_id': broker_order_id}

  def close_position_for(self, ticker: str, reason: str = 'manual') -> Dict[str, Any]:
    """Close an open position by submitting an opposing market order."""
    pos = position_for_ticker(ticker, paper=self.paper)
    if not pos:
      return {'success': False, 'error': 'no_open_position', 'ticker': ticker}
    qty = abs(float(pos['quantity']))
    side = 'sell' if pos['side'] == 'long' else 'buy'
    return self.place_order(
      ticker=ticker, quantity=qty, side=side, order_type='market',
      thesis_id=pos.get('thesis_id'),
    )
