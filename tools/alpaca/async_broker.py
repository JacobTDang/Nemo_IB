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
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv

# Imported rather than mirrored: common/secret.py imports nothing, so it
# crosses no boundary and cannot drag a layer in behind it. The same import
# sits at the head of tools/alpaca_server/alpaca_server.py, which is this
# module's sibling on the sync path.
from common.secret import Secret


_DOTENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=_DOTENV_PATH)

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

  The credentials are held as `Secret`, not as strings. This object outlives
  the constructor that read them, so anything that walks it — a debugger, a
  crash reporter, pytest's --showlocals on a frame holding the broker — would
  otherwise find a live broker key in the instance dict (issue #64).
  """

  def __init__(self, paper: bool = True, timeout: Optional[httpx.Timeout] = None):
    self.paper = paper
    # Wrapped inside the expression that reads the environment, not after it.
    # An intermediate `key = os.getenv(...)` is a local, and --showlocals, a
    # debugger and a crash reporter render a local exactly as they render a
    # parameter — so the value would be disclosed before the wrapper was ever
    # reached. The `or` chains are the env-var precedence
    # testing/test_phase_B3a_alpaca_env_fallback.py pins, kept whole inside
    # the wrapper so no step of the fallback lands on a name of its own.
    if paper:
      self._key = Secret(os.getenv("ALPACA_PAPER_KEY")
                         or os.getenv("ALPACA_API_KEY") or "")
      self._secret = Secret(os.getenv("ALPACA_PAPER_SECRET")
                            or os.getenv("ALPACA_SECRET") or "")
    else:
      self._key = Secret(os.getenv("ALPACA_LIVE_KEY") or "")
      self._secret = Secret(os.getenv("ALPACA_LIVE_SECRET") or "")
    if not self._key or not self._secret:
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

  # `key` and `secret` were the raw attributes this class used to store, and
  # testing/test_phase_A9_async_broker.py builds its own mock httpx client out
  # of them. They stay, as reveals under the old name rather than as stored
  # values: a property is not in the instance dict, so everything that renders
  # an object automatically — --showlocals, a debugger, a crash reporter —
  # now finds a `Secret` and nothing else. New code should say
  # `self._key.reveal()`, which greps as the deliberate act it is.
  async def __aenter__(self) -> "AsyncBroker":
    self._client = httpx.AsyncClient(
      base_url=self.base_url,
      timeout=self._timeout,
      headers={
        # Revealed at the point of use and never bound to a name, which is the
        # whole discipline: the value exists only for as long as httpx takes
        # to copy it into its own header store.
        "APCA-API-KEY-ID": self._key.reveal(),
        "APCA-API-SECRET-KEY": self._secret.reveal(),
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

  def _raise_for_status(self, resp: httpx.Response) -> None:
    """Turn a non-2xx into AsyncBrokerError, with the credentials taken out.

    An instance method rather than a static one because scrubbing needs the
    credentials to remove. The text matters more here than in a log line:
    `tools/alpaca/server.py` puts `str(e)` from this exception straight into
    the `error` field of the MCP result it returns, so a provider that echoed
    the offending key back would send it out of the process entirely. Alpaca's
    401 body does not quote the key today; nothing here should depend on it
    never starting to.

    Scrubbed in the same expression that reads the body, for the same reason
    the credentials are wrapped at the read: a `msg = resp.text` cleaned only
    on the next line has already put the untouched text on a local, where
    --showlocals prints it.
    """
    if resp.status_code >= 400:
      try:
        detail = self._scrub(str(resp.json().get("message", resp.text)))
      except Exception:
        detail = self._scrub(resp.text)
      raise AsyncBrokerError(f"HTTP {resp.status_code}: {detail}")

  def _scrub(self, text: str) -> str:
    """`text` with either credential replaced by the placeholder."""
    return self._secret.scrub(self._key.scrub(text))
