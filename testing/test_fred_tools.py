"""
Test FRED MCP tools via direct server instantiation.

Usage: python -m testing.test_fred_tools
"""
import asyncio
import json
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.news_agregator.fred_server import FredServer
from mcp.types import TextContent


def pretty(label: str, result: list[TextContent]):
  """Pretty-print a tool result."""
  print(f"\n{'='*70}")
  print(f"  {label}")
  print(f"{'='*70}")
  for tc in result:
    data = json.loads(tc.text)
    print(json.dumps(data, indent=2, default=str)[:3000])
  print()


async def main():
  server = FredServer()

  try:
    # 1. Macro Snapshot
    print("\n[1/4] Testing get_macro_snapshot...")
    result = await server.get_macro_snapshot()
    pretty("MACRO SNAPSHOT", result)

    # 2. Treasury Yields
    print("[2/4] Testing get_treasury_yields...")
    result = await server.get_treasury_yields()
    pretty("TREASURY YIELDS", result)

    # 3. Generic Series (M2 Money Supply)
    print("[3/4] Testing get_fred_series (M2SL)...")
    result = await server.get_fred_series("M2SL", frequency="m")
    pretty("FRED SERIES: M2SL", result)

    # 4. Search
    print("[4/4] Testing search_fred ('housing starts')...")
    result = await server.search_fred("housing starts")
    pretty("FRED SEARCH: housing starts", result)

    print("\nAll tests passed!")

  finally:
    await server.client.close()


if __name__ == "__main__":
  asyncio.run(main())
