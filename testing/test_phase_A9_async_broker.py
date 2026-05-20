"""Phase A9: AsyncBroker unit tests using httpx.MockTransport.

Validates the response-parsing and URL-routing without making live calls.
Live coverage is in A8.
"""
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx
from tools.alpaca.async_broker import AsyncBroker, AsyncBrokerError


def _mock_transport(handler):
  return httpx.MockTransport(handler)


def _set_fake_creds():
  os.environ["ALPACA_PAPER_KEY"] = "fake_key"
  os.environ["ALPACA_PAPER_SECRET"] = "fake_secret"


async def _with_mock_broker(handler, fn):
  """Build an AsyncBroker that uses MockTransport for httpx instead of real network."""
  _set_fake_creds()
  broker = AsyncBroker(paper=True)
  # Build the client manually with mock transport
  broker._client = httpx.AsyncClient(
    base_url=broker.base_url,
    headers={
      "APCA-API-KEY-ID": broker.key,
      "APCA-API-SECRET-KEY": broker.secret,
    },
    transport=_mock_transport(handler),
  )
  try:
    return await fn(broker)
  finally:
    await broker._client.aclose()


def test_credentials_required():
  for k in ("ALPACA_PAPER_KEY", "ALPACA_PAPER_SECRET"):
    os.environ.pop(k, None)
  try:
    AsyncBroker(paper=True)
    raise AssertionError("should have raised on missing creds")
  except RuntimeError as e:
    assert "ALPACA_PAPER_KEY" in str(e)
  print("PASS: missing creds raises RuntimeError")


def test_get_account_parses_response():
  def handler(request):
    assert request.url.path == "/v2/account"
    assert request.headers.get("APCA-API-KEY-ID") == "fake_key"
    return httpx.Response(200, json={
      "equity": "100000.50", "cash": "99000.25",
      "buying_power": "200000.50", "portfolio_value": "100000.50",
      "status": "ACTIVE",
    })
  async def run(broker):
    return await broker.get_account()
  result = asyncio.run(_with_mock_broker(handler, run))
  assert result["equity"] == 100000.5
  assert result["status"] == "ACTIVE"
  assert result["paper"] is True
  print(f"PASS: get_account returned {result}")


def test_get_all_positions_parses_response():
  def handler(request):
    return httpx.Response(200, json=[
      {"symbol": "aapl", "qty": "10", "side": "long",
       "market_value": "2000", "avg_entry_price": "195.0"},
      {"symbol": "TSLA", "qty": "5", "side": "long",
       "market_value": "1500", "avg_entry_price": "300.0"},
    ])
  async def run(broker):
    return await broker.get_all_positions()
  result = asyncio.run(_with_mock_broker(handler, run))
  assert len(result) == 2
  assert result[0]["symbol"] == "AAPL"  # uppercased
  assert result[0]["qty"] == 10.0
  assert isinstance(result[1]["market_value"], float)
  print(f"PASS: get_all_positions parsed {len(result)} positions")


def test_get_open_position_404_returns_none():
  def handler(request):
    return httpx.Response(404, json={"message": "position not found"})
  async def run(broker):
    return await broker.get_open_position("UNKNOWN")
  result = asyncio.run(_with_mock_broker(handler, run))
  assert result is None
  print("PASS: 404 -> None for get_open_position")


def test_submit_market_order_builds_payload():
  captured = {}
  def handler(request):
    captured["body"] = json.loads(request.content)
    captured["path"] = request.url.path
    return httpx.Response(200, json={
      "id": "alpaca-order-xyz", "client_order_id": captured["body"]["client_order_id"],
      "status": "accepted", "symbol": "F", "qty": "1", "side": "buy",
      "filled_at": None,
    })
  async def run(broker):
    return await broker.submit_market_order(
      "F", qty=1, side="buy", client_order_id="nemo-test-cli-1"
    )
  result = asyncio.run(_with_mock_broker(handler, run))
  assert captured["path"] == "/v2/orders"
  assert captured["body"]["symbol"] == "F"
  assert captured["body"]["type"] == "market"
  assert captured["body"]["side"] == "buy"
  assert captured["body"]["client_order_id"] == "nemo-test-cli-1"
  assert result["id"] == "alpaca-order-xyz"
  print(f"PASS: market order payload + response parsed correctly")


def test_close_position_calls_delete():
  captured = {}
  def handler(request):
    captured["method"] = request.method
    captured["path"] = request.url.path
    return httpx.Response(200, json={
      "id": "close-order-abc", "status": "accepted", "symbol": "F",
      "qty": "1", "side": "sell",
    })
  async def run(broker):
    return await broker.close_position("F")
  result = asyncio.run(_with_mock_broker(handler, run))
  assert captured["method"] == "DELETE"
  assert captured["path"] == "/v2/positions/F"
  assert result["id"] == "close-order-abc"
  print("PASS: close_position issued DELETE /v2/positions/F")


def test_broker_error_on_4xx_5xx():
  def handler(request):
    return httpx.Response(403, json={"message": "forbidden"})
  async def run(broker):
    try:
      await broker.get_account()
      return None
    except AsyncBrokerError as e:
      return str(e)
  err = asyncio.run(_with_mock_broker(handler, run))
  assert err is not None
  assert "403" in err and "forbidden" in err
  print(f"PASS: 403 raised AsyncBrokerError: {err[:60]}")


if __name__ == "__main__":
  test_credentials_required()
  test_get_account_parses_response()
  test_get_all_positions_parses_response()
  test_get_open_position_404_returns_none()
  test_submit_market_order_builds_payload()
  test_close_position_calls_delete()
  test_broker_error_on_4xx_5xx()
  print("\nAll Phase A9 async broker tests passed.")
