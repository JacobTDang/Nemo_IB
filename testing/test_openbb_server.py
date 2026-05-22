"""Unit + smoke tests for tools/openbb_server/server.py.

Three layers:
  1. Schema: list_tools registers exactly 4 tools with correct schemas
  2. Smoke (live): invoke each tool with a known-good ticker, assert
     non-empty data field on success
  3. Error path: invalid ticker returns envelope with success=false

Run:
  .venv\\Scripts\\python.exe testing\\test_openbb_server.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.openbb_server.server import OpenBBServer


_results = {'pass': 0, 'fail': 0, 'failures': []}


def _check(name: str, condition: bool, hint: str = '') -> None:
  if condition:
    _results['pass'] += 1
    print(f"  PASS  {name}")
  else:
    _results['fail'] += 1
    _results['failures'].append((name, hint))
    print(f"  FAIL  {name}  --  {hint}")


# ---------------------------------------------------------------------------
# Schema layer (no SDK calls)
# ---------------------------------------------------------------------------

async def _list_tools(srv: OpenBBServer):
  # The decorator-registered handler is the one we wired in _setup_handlers.
  # Pull it from the server's request handlers.
  from mcp.types import ListToolsRequest
  handler = srv.server.request_handlers[ListToolsRequest]
  req = ListToolsRequest(method='tools/list', params=None)
  resp = await handler(req)
  return resp.root.tools


def test_lists_4_tools():
  print("\n== schema: list_tools returns 4 tools with valid shapes ==")
  srv = OpenBBServer()
  tools = asyncio.run(_list_tools(srv))
  _check("4 tools registered", len(tools) == 4, f"got {len(tools)}")
  expected = {
    'obb_insider_trading', 'obb_news_company',
    'obb_options_chain', 'obb_analyst_consensus',
  }
  actual = {t.name for t in tools}
  _check("expected tool names", actual == expected, str(actual))

  for t in tools:
    _check(f"{t.name}: has description", bool(t.description) and len(t.description) > 20,
           f"len={len(t.description or '')}")
    schema = t.inputSchema
    _check(f"{t.name}: schema has ticker required",
           'ticker' in (schema.get('required') or []),
           str(schema.get('required')))
    _check(f"{t.name}: schema has ticker property",
           'ticker' in (schema.get('properties') or {}),
           str(schema.get('properties', {}).keys()))


# ---------------------------------------------------------------------------
# Smoke layer (live SDK calls)
# ---------------------------------------------------------------------------

async def _invoke(srv: OpenBBServer, tool_name: str, args: dict):
  from mcp.types import CallToolRequest, CallToolRequestParams
  handler = srv.server.request_handlers[CallToolRequest]
  params = CallToolRequestParams(name=tool_name, arguments=args)
  req = CallToolRequest(method='tools/call', params=params)
  resp = await handler(req)
  text = resp.root.content[0].text
  return json.loads(text)


def test_smoke_insider_trading():
  print("\n== smoke: obb_insider_trading AAPL ==")
  srv = OpenBBServer()
  env = asyncio.run(_invoke(srv, 'obb_insider_trading',
                            {'ticker': 'AAPL', 'limit': 5}))
  _check("envelope.success == True", env.get('success') is True,
         str(env.get('metadata', {}).get('errors')))
  _check("envelope.ticker == 'AAPL'", env.get('ticker') == 'AAPL')
  _check("envelope.data is a non-empty list",
         isinstance(env.get('data'), list) and len(env['data']) > 0,
         f"type={type(env.get('data')).__name__}")


def test_smoke_news_company():
  print("\n== smoke: obb_news_company AAPL ==")
  srv = OpenBBServer()
  env = asyncio.run(_invoke(srv, 'obb_news_company',
                            {'ticker': 'AAPL', 'limit': 5}))
  _check("envelope.success == True", env.get('success') is True,
         str(env.get('metadata', {}).get('errors')))
  _check("envelope.data is a non-empty list",
         isinstance(env.get('data'), list) and len(env['data']) > 0,
         f"len={len(env.get('data') or [])}")


def test_smoke_options_chain():
  print("\n== smoke: obb_options_chain AAPL ==")
  srv = OpenBBServer()
  env = asyncio.run(_invoke(srv, 'obb_options_chain', {'ticker': 'AAPL'}))
  _check("envelope.success == True", env.get('success') is True,
         str(env.get('metadata', {}).get('errors')))
  data = env.get('data') or {}
  _check("data has total_rows > 0",
         isinstance(data, dict) and data.get('total_rows', 0) > 0,
         f"got {data}")
  _check("data['rows'] returned (capped at 200)",
         isinstance(data.get('rows'), list) and 0 < len(data['rows']) <= 200,
         f"len={len(data.get('rows', []))}")


def test_smoke_analyst_consensus():
  print("\n== smoke: obb_analyst_consensus AAPL ==")
  srv = OpenBBServer()
  env = asyncio.run(_invoke(srv, 'obb_analyst_consensus',
                            {'ticker': 'AAPL'}))
  _check("envelope.success == True", env.get('success') is True,
         str(env.get('metadata', {}).get('errors')))
  _check("envelope.data is a non-empty list",
         isinstance(env.get('data'), list) and len(env['data']) > 0,
         f"len={len(env.get('data') or [])}")


# ---------------------------------------------------------------------------
# Error path
# ---------------------------------------------------------------------------

def test_missing_ticker_returns_error_envelope():
  print("\n== error: empty-string ticker returns envelope with success=false ==")
  # Schema-required validation happens at the MCP framework layer, so passing
  # {} would never reach our handler. Pass ticker='' to exercise the in-handler
  # validation path that returns a structured error envelope.
  srv = OpenBBServer()
  env = asyncio.run(_invoke(srv, 'obb_insider_trading', {'ticker': ''}))
  _check("success == False", env.get('success') is False,
         f"success={env.get('success')}")
  _check("errors list populated",
         bool(env.get('metadata', {}).get('errors')),
         str(env.get('metadata', {}).get('errors')))


def test_invalid_ticker_does_not_crash():
  print("\n== error: invalid ticker ZZZZZZ returns envelope (no crash) ==")
  srv = OpenBBServer()
  env = asyncio.run(_invoke(srv, 'obb_insider_trading',
                            {'ticker': 'ZZZZZZ'}))
  # Either success=True with empty data OR success=False with error message;
  # both are acceptable -- the contract is no exception, envelope returned.
  _check("returned envelope (no crash)",
         isinstance(env, dict) and 'success' in env,
         str(env)[:200])


# ---------------------------------------------------------------------------

def main() -> int:
  print("\nOpenBB MCP server tests\n")
  test_lists_4_tools()
  test_smoke_insider_trading()
  test_smoke_news_company()
  test_smoke_options_chain()
  test_smoke_analyst_consensus()
  test_missing_ticker_returns_error_envelope()
  test_invalid_ticker_does_not_crash()

  print(f"\n== Summary ==\n  PASS: {_results['pass']}\n  FAIL: {_results['fail']}")
  for n, h in _results['failures']:
    print(f"  - {n}: {h}")
  return 0 if _results['fail'] == 0 else 1


if __name__ == "__main__":
  sys.exit(main())
