"""Phase A2: get_paper_account tool.

Refactored for A9: mocks the AsyncBroker.get_account at the module level
instead of the alpaca-py SDK. The mock pattern is simpler than the
sys.modules patching the previous version used.
"""
import asyncio
import json
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _FakeBroker:
  def __init__(self, account=None, raises=None):
    self.account = account or {
      "paper": True, "equity": 100000.0, "cash": 100000.0,
      "buying_power": 200000.0, "portfolio_value": 100000.0, "status": "ACTIVE",
    }
    self.raises = raises

  def __init__(self, *args, **kwargs):
    # Default values - allows AsyncBroker(paper=True) signature
    self.account = {
      "paper": True, "equity": 100000.0, "cash": 100000.0,
      "buying_power": 200000.0, "portfolio_value": 100000.0, "status": "ACTIVE",
    }
    self.raises = None

  async def __aenter__(self):
    return self

  async def __aexit__(self, *a):
    pass

  async def get_account(self):
    if self.raises:
      raise self.raises
    return self.account


def _fake_broker_class(raises=None, account_override=None):
  def factory(*args, **kwargs):
    fb = _FakeBroker()
    if account_override is not None:
      fb.account = account_override
    fb.raises = raises
    return fb
  return factory


async def _call_get_account():
  from tools.alpaca import server as alpaca_srv
  srv = alpaca_srv.AlpacaServer()
  result = await srv.get_paper_account()
  return json.loads(result[0].text)


def test_get_paper_account_returns_floats():
  with patch('tools.alpaca.async_broker.AsyncBroker', new=_fake_broker_class()):
    parsed = asyncio.run(_call_get_account())
  assert parsed.get('paper') is True
  assert parsed['equity'] == 100000.0 and isinstance(parsed['equity'], float)
  assert parsed['status'] == 'ACTIVE'
  print(f"PASS: get_paper_account returns {parsed}")


def test_get_paper_account_surfaces_broker_error():
  with patch('tools.alpaca.async_broker.AsyncBroker',
              new=_fake_broker_class(raises=Exception("API down"))):
    parsed = asyncio.run(_call_get_account())
  assert 'error' in parsed
  assert 'API down' in parsed['error']
  print(f"PASS: broker error surfaced as error key")


def test_get_paper_account_listed_in_tools():
  from tools.alpaca import server as alpaca_srv
  srv = alpaca_srv.AlpacaServer()
  tools = asyncio.run(srv.list_tools_descriptors())
  names = [t.name for t in tools]
  assert 'get_paper_account' in names, f"missing: {names}"
  print(f"PASS: get_paper_account in tools list ({len(names)} total)")


if __name__ == "__main__":
  test_get_paper_account_listed_in_tools()
  test_get_paper_account_returns_floats()
  test_get_paper_account_surfaces_broker_error()
  print("\nAll Phase A2 get_paper_account tests passed.")
