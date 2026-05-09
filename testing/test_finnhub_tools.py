"""
Standalone test for the Finnhub MCP server.
Spins up the server via MCPConnectionManager, calls each tool with AAPL,
and verifies the envelope structure.

Usage: python -m testing.test_finnhub_tools
"""
import asyncio
import sys
import json
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, ".")

from agent.MCP_manager import MCPConnectionManager


SHOW_ENVELOPES = True  # Set to False to hide full JSON dumps


async def test_envelope(result: dict, tool_name: str):
  """Verify the standard envelope shape."""
  assert result.get("domain") == "market_intel", f"{tool_name}: missing domain"
  assert "ticker" in result, f"{tool_name}: missing ticker"
  assert "timestamp" in result, f"{tool_name}: missing timestamp"
  assert "data" in result, f"{tool_name}: missing data"
  assert "metadata" in result, f"{tool_name}: missing metadata"
  # Not an error response
  if isinstance(result["data"], dict) and "error" in result["data"]:
    print(f"  WARNING: {tool_name} returned error: {result['data']['error']}")
    return False
  return True


def dump_envelope(result: dict, tool_name: str):
  """Print the full envelope JSON so you can see what the LLM receives."""
  if not SHOW_ENVELOPES:
    return
  print(f"\n  --- {tool_name} envelope ---")
  print(f"  {json.dumps(result, indent=2, default=str)}")
  print(f"  --- end {tool_name} ---\n")


async def main():
  ticker = "AAPL"
  today = datetime.now().strftime("%Y-%m-%d")
  thirty_days_ago = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
  seven_days_out = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")

  passed = 0
  failed = 0

  # Only connect to finnhub server for this test
  async with MCPConnectionManager() as mcp:
    # Override: only connect to finnhub
    # (MCPConnectionManager connects all by default, which is fine for integration)

    tools = await mcp.list_tools()
    finnhub_tools = {name: info for name, info in tools.items() if info.get("server") == "finnhub"}
    print(f"\nFinnhub tools registered: {len(finnhub_tools)}")
    for name in sorted(finnhub_tools):
      print(f"  - {name}")
    print()

    # Test 1: get_company_news (slimmed articles)
    print("Testing get_company_news...")
    result = await mcp.call_tool("get_company_news", {
      "ticker": ticker, "from_date": thirty_days_ago, "to_date": today
    })
    if await test_envelope(result, "get_company_news"):
      articles = result["data"]
      if isinstance(articles, list):
        assert len(articles) <= 20, f"Expected <= 20 articles, got {len(articles)}"
        if articles:
          keys = set(articles[0].keys())
          expected = {"headline", "summary", "source", "datetime", "url"}
          assert keys == expected, f"Article keys mismatch: {keys} != {expected}"
          # Verify no bloat fields leaked through
          assert "id" not in articles[0], "id should be stripped"
          assert "image" not in articles[0], "image should be stripped"
        print(f"  OK: {len(articles)} articles (capped at 20, slimmed to 5 fields)")
      else:
        print(f"  OK: non-list data (possible error response)")
      dump_envelope(result, "get_company_news")
      passed += 1
    else:
      failed += 1

    # Test 2: get_market_news (slimmed articles)
    print("Testing get_market_news...")
    result = await mcp.call_tool("get_market_news", {"category": "general"})
    if await test_envelope(result, "get_market_news"):
      articles = result["data"]
      if isinstance(articles, list):
        assert len(articles) <= 20, f"Expected <= 20 articles, got {len(articles)}"
        if articles:
          assert "headline" in articles[0], "Missing headline field"
          assert "image" not in articles[0], "image should be stripped"
        print(f"  OK: {len(articles)} articles (capped at 20, slimmed)")
      else:
        print(f"  OK: non-list data")
      dump_envelope(result, "get_market_news")
      passed += 1
    else:
      failed += 1

    # Test 3: get_insider_transactions (condensed)
    print("Testing get_insider_transactions...")
    result = await mcp.call_tool("get_insider_transactions", {"ticker": ticker})
    if await test_envelope(result, "get_insider_transactions"):
      data = result["data"]
      assert isinstance(data, dict), f"Expected condensed dict, got {type(data)}"
      # Verify condensed structure
      for key in ["total_bought", "total_sold", "net_shares", "buy_count",
                   "sell_count", "top_insiders", "recent_30d", "recent_90d", "signal"]:
        assert key in data, f"Missing condensed key: {key}"
      assert data["signal"] in ("net_buying", "net_selling", "neutral"), f"Bad signal: {data['signal']}"
      assert isinstance(data["top_insiders"], list), "top_insiders should be a list"
      assert len(data["top_insiders"]) <= 5, "top_insiders should be capped at 5"
      # Verify recency buckets have expected keys
      for bucket in ["recent_30d", "recent_90d"]:
        for k in ["bought", "sold", "net"]:
          assert k in data[bucket], f"Missing {k} in {bucket}"
      print(f"  OK: condensed insider data")
      print(f"    signal={data['signal']}, net_shares={data['net_shares']:,}")
      print(f"    buys={data['buy_count']}, sells={data['sell_count']}")
      print(f"    top_insiders: {[i['name'] for i in data['top_insiders']]}")
      print(f"    30d net={data['recent_30d']['net']:,}, 90d net={data['recent_90d']['net']:,}")
      dump_envelope(result, "get_insider_transactions")
      passed += 1
    else:
      failed += 1

    # Test 4: get_earnings_calendar (condensed)
    print("Testing get_earnings_calendar...")
    result = await mcp.call_tool("get_earnings_calendar", {
      "from_date": today, "to_date": seven_days_out
    })
    if await test_envelope(result, "get_earnings_calendar"):
      data = result["data"]
      assert isinstance(data, dict), f"Expected condensed dict, got {type(data)}"
      for key in ["total_companies", "by_date", "events"]:
        assert key in data, f"Missing condensed key: {key}"
      assert isinstance(data["by_date"], list), "by_date should be a list"
      assert isinstance(data["events"], list), "events should be a list"
      assert len(data["events"]) <= 15, f"Events should be capped at 15, got {len(data['events'])}"
      print(f"  OK: condensed earnings calendar")
      print(f"    total_companies={data['total_companies']}, dates={len(data['by_date'])}, events_shown={len(data['events'])}")
      for d in data["by_date"]:
        print(f"    {d['date']}: {d['count']} companies reporting")
      dump_envelope(result, "get_earnings_calendar")
      passed += 1
    else:
      failed += 1

    # Test 5: get_analyst_recommendations (condensed)
    print("Testing get_analyst_recommendations...")
    result = await mcp.call_tool("get_analyst_recommendations", {"ticker": ticker})
    if await test_envelope(result, "get_analyst_recommendations"):
      data = result["data"]
      assert isinstance(data, dict), f"Expected condensed dict, got {type(data)}"
      for key in ["latest", "prior", "consensus", "trend", "total_analysts"]:
        assert key in data, f"Missing condensed key: {key}"
      assert data["consensus"] in ("strong_buy", "buy", "hold", "sell", "strong_sell", "unknown"), \
        f"Bad consensus: {data['consensus']}"
      assert data["trend"] in ("upgrading", "downgrading", "stable", "unknown"), \
        f"Bad trend: {data['trend']}"
      # Verify latest period structure
      if data["latest"]:
        for k in ["strong_buy", "buy", "hold", "sell", "strong_sell", "period"]:
          assert k in data["latest"], f"Missing {k} in latest period"
      print(f"  OK: condensed analyst data")
      print(f"    consensus={data['consensus']}, trend={data['trend']}, analysts={data['total_analysts']}")
      if data["latest"]:
        l = data["latest"]
        print(f"    latest: SB={l['strong_buy']} B={l['buy']} H={l['hold']} S={l['sell']} SS={l['strong_sell']}")
      dump_envelope(result, "get_analyst_recommendations")
      passed += 1
    else:
      failed += 1

    # Test 6: get_company_peers (unchanged)
    print("Testing get_company_peers...")
    result = await mcp.call_tool("get_company_peers", {"ticker": ticker})
    if await test_envelope(result, "get_company_peers"):
      peers = result["data"]
      print(f"  OK: peers = {peers if isinstance(peers, list) else 'N/A'}")
      dump_envelope(result, "get_company_peers")
      passed += 1
    else:
      failed += 1

    # Test 7: get_basic_financials (condensed to IB-essential metrics)
    print("Testing get_basic_financials...")
    result = await mcp.call_tool("get_basic_financials", {"ticker": ticker})
    if await test_envelope(result, "get_basic_financials"):
      data = result["data"]
      assert isinstance(data, dict), f"Expected condensed dict, got {type(data)}"
      assert "metric" in data, "Missing metric key"
      assert "metric_count" in data, "Missing metric_count key"
      assert "series" not in data, "series should be dropped"
      metrics = data["metric"]
      print(f"  OK: {data['metric_count']} key metrics (filtered from 132)")
      print(f"    Valuation: P/E={metrics.get('peTTM')}, EV/EBITDA={metrics.get('evEbitdaTTM')}, PEG={metrics.get('pegTTM')}")
      print(f"    Margins: Gross={metrics.get('grossMarginTTM')}%, Op={metrics.get('operatingMarginTTM')}%, Net={metrics.get('netProfitMarginTTM')}%")
      print(f"    Growth: EPS TTM YoY={metrics.get('epsGrowthTTMYoy')}%, Rev TTM YoY={metrics.get('revenueGrowthTTMYoy')}%")
      print(f"    Size: MktCap={metrics.get('marketCapitalization')}, EV={metrics.get('enterpriseValue')}")
      dump_envelope(result, "get_basic_financials")
      passed += 1
    else:
      failed += 1

  print(f"\n{'='*40}")
  print(f"Results: {passed} passed, {failed} failed out of 7 tools")
  print(f"{'='*40}")

  return failed == 0


if __name__ == "__main__":
  success = asyncio.run(main())
  sys.exit(0 if success else 1)
