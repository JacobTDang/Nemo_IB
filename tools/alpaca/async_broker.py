"""Async wrapper around Alpaca's REST API for paper trading.

alpaca-py 0.43.2 ships a sync TradingClient backed by httpx-sync. Inside
the MCP stdio subprocess + asyncio.to_thread, sync httpx calls were
observed to take 30-60s vs 0.6s standalone — likely an event-loop /
anyio contention issue specific to Windows + MCP framing.

This module bypasses alpaca-py entirely and talks to Alpaca's documented
REST endpoints directly via httpx.AsyncClient. Reference:
https://docs.alpaca.markets/reference/getaccount-1

Endpoints used:
  - GET /v2/account
  - GET /v2/positions
  - POST /v2/orders                  (market/limit)
  - GET /v2/orders/{order_id}
  - DELETE /v2/positions/{symbol}    (close position via opposing market)

Auth headers:
  - APCA-API-KEY-ID
  - APCA-API-SECRET-KEY
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv


_PAPER_BASE = "https://paper-api.alpaca.markets"
_LIVE_BASE = "https://api.alpaca.markets"
_DEFAULT_TIMEOUT = httpx.Timeout(10.0, connect=5.0)


class AsyncBrokerError(RuntimeError):
  """Raised when the broker returns a non-2xx response."""


class AsyncBroker:
  """Minimal async client for Alpaca paper trading.

  Construct with `paper=True` (default) for paper, `False` for live.
  Paper credentials are read from env in priority order:
    1. `ALPACA_PAPER_KEY` + `ALPACA_PAPER_SECRET` (preferred — explicit)
    2. `ALPACA_API_KEY` + `ALPACA_SECRET` (legacy fallback — Alpaca's default
       env-var names; only honored for paper mode)
  Live mode requires `ALPACA_LIVE_KEY` + `ALPACA_LIVE_SECRET` and never
  falls back to legacy names — that prevents an unintended live-trade
  configuration from any account whose credentials happen to live under
  the generic `ALPACA_API_KEY` pair.

  Always use via `async with AsyncBroker() as broker:` so the underlying
  httpx.AsyncClient is closed properly.
  """

  def __init__(self, paper: bool = True, timeout: Optional[httpx.Timeout] = None):
    load_dotenv()
    self.paper = paper
    if paper:
      self.key = os.getenv("ALPACA_PAPER_KEY") or os.getenv("ALPACA_API_KEY")
      self.secret = os.getenv("ALPACA_PAPER_SECRET") or os.getenv("ALPACA_SECRET")
    else:
      self.key = os.getenv("ALPACA_LIVE_KEY")
      self.secret = os.getenv("ALPACA_LIVE_SECRET")
    if not self.key or not self.secret:
      if paper:
        hint = ("Set ALPACA_PAPER_KEY + ALPACA_PAPER_SECRET (preferred) or "
                "ALPACA_API_KEY + ALPACA_SECRET (legacy) in .env")
      else:
        hint = "Set ALPACA_LIVE_KEY + ALPACA_LIVE_SECRET in .env"
      raise RuntimeError(
        f"Missing Alpaca {'paper' if paper else 'LIVE'} credentials. {hint}"
      )
    self.base_url = _PAPER_BASE if paper else _LIVE_BASE
    self._timeout = timeout or _DEFAULT_TIMEOUT
    self._client: Optional[httpx.AsyncClient] = None

  async def __aenter__(self) -> "AsyncBroker":
    self._client = httpx.AsyncClient(
      base_url=self.base_url,
      timeout=self._timeout,
      headers={
        "APCA-API-KEY-ID": self.key,
        "APCA-API-SECRET-KEY": self.secret,
        "accept": "application/json",
      },
    )
    return self

  async def __aexit__(self, *exc) -> None:
    if self._client is not None:
      await self._client.aclose()
      self._client = None

  # --- Account ---------------------------------------------------------

  async def get_account(self) -> Dict[str, Any]:
    """Returns the trading account summary."""
    resp = await self._client.get("/v2/account")
    self._raise_for_status(resp)
    data = resp.json()
    return {
      "paper": self.paper,
      "equity": float(data.get("equity", 0)),
      "cash": float(data.get("cash", 0)),
      "buying_power": float(data.get("buying_power", 0)),
      "portfolio_value": float(data.get("portfolio_value", 0)),
      "status": data.get("status", "UNKNOWN"),
    }

  # --- Positions -------------------------------------------------------

  async def get_all_positions(self) -> List[Dict[str, Any]]:
    """All open positions at the broker."""
    resp = await self._client.get("/v2/positions")
    self._raise_for_status(resp)
    out = []
    for p in resp.json() or []:
      out.append({
        "symbol": str(p.get("symbol", "")).upper(),
        "qty": float(p.get("qty", 0) or 0),
        "side": p.get("side", "long"),
        "market_value": float(p.get("market_value", 0) or 0),
        "avg_entry_price": float(p.get("avg_entry_price", 0) or 0),
      })
    return out

  async def get_open_position(self, symbol: str) -> Optional[Dict[str, Any]]:
    resp = await self._client.get(f"/v2/positions/{symbol.upper()}")
    if resp.status_code == 404:
      return None
    self._raise_for_status(resp)
    p = resp.json()
    return {
      "symbol": str(p.get("symbol", "")).upper(),
      "qty": float(p.get("qty", 0) or 0),
      "side": p.get("side", "long"),
    }

  # --- Orders ----------------------------------------------------------

  async def submit_market_order(
    self,
    symbol: str,
    qty: float,
    side: str,
    *,
    client_order_id: str,
    time_in_force: str = "day",
  ) -> Dict[str, Any]:
    """Submit a market order. `side` is 'buy' or 'sell'."""
    payload = {
      "symbol": symbol.upper(),
      "qty": str(qty),
      "side": side,
      "type": "market",
      "time_in_force": time_in_force,
      "client_order_id": client_order_id,
    }
    resp = await self._client.post("/v2/orders", json=payload)
    self._raise_for_status(resp)
    o = resp.json()
    return {
      "id": o.get("id"),
      "client_order_id": o.get("client_order_id"),
      "status": o.get("status"),
      "symbol": o.get("symbol"),
      "qty": float(o.get("qty", 0) or 0),
      "side": o.get("side"),
      "filled_at": o.get("filled_at"),
    }

  async def get_order_by_id(self, order_id: str) -> Dict[str, Any]:
    resp = await self._client.get(f"/v2/orders/{order_id}")
    self._raise_for_status(resp)
    return resp.json()

  async def close_position(self, symbol: str) -> Dict[str, Any]:
    """Submit an opposing market order to flatten the position. Returns
    the broker's order response."""
    resp = await self._client.delete(f"/v2/positions/{symbol.upper()}")
    self._raise_for_status(resp)
    return resp.json()

  # --- Internals -------------------------------------------------------

  @staticmethod
  def _raise_for_status(resp: httpx.Response) -> None:
    if resp.status_code >= 400:
      try:
        body = resp.json()
        msg = body.get("message", resp.text)
      except Exception:
        msg = resp.text
      raise AsyncBrokerError(f"HTTP {resp.status_code}: {msg}")
