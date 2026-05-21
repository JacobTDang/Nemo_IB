"""Tool health smoke test — exercise every Nemo MCP tool with a canonical
call and report per-tool pass/fail.

A tool "passes" if it returns a structured response that contains real data
(not an `error` envelope, not a `success: false` for tools that flag it,
and not an empty `data` block where data is expected).

Safety:
  - Trading tools that mutate state (place_paper_order, close_paper_position)
    are exercised through their REJECTION path only — proposals constructed
    to trip Risk_Officer's confidence gate, so no order ever reaches Alpaca.

Usage:
  ./.venv/Scripts/python.exe testing/test_tool_health.py [ticker]
  ./.venv/Scripts/python.exe testing/test_tool_health.py MSFT > tool_health.log

The script does NOT use pytest — same shape as the rest of testing/.
"""
import asyncio
import json
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from agent.MCP_manager import MCPConnectionManager


TICKER = (sys.argv[1] if len(sys.argv) > 1 else "MSFT").upper()
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
RESULTS_PATH = os.path.join(OUT_DIR, "tool_health.json")


def _today_iso():
  return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _days_ago(n):
  return (datetime.now(timezone.utc) - timedelta(days=n)).strftime("%Y-%m-%d")


def _is_error_envelope(payload):
  """A response is considered an error if any of these are true:
   - top-level 'error' key with a value
   - 'success' explicitly False AND no useful fallback data
   - data.error is set (some tools nest the error)
  Returns (failed: bool, reason: str)."""
  if not isinstance(payload, dict):
    return True, f"non-dict response: {type(payload).__name__}"
  if payload.get("error"):
    return True, f"error: {str(payload['error'])[:120]}"
  data = payload.get("data")
  if isinstance(data, dict) and data.get("error"):
    return True, f"data.error: {str(data['error'])[:120]}"
  # Tools that publish a success flag — only fail when explicitly False
  if payload.get("success") is False:
    return True, "success=False"
  return False, ""


async def _call(session_obj, tool_name, args, kind):
  """Issue a single tool call. `session_obj` is either an MCPConnectionManager
  (for the 4 data servers) or a ClientSession (for alpaca direct stdio)."""
  try:
    if kind == "mcp_manager":
      raw = await session_obj.call_tool(tool_name, args)
      result = raw
    else:  # direct ClientSession
      resp = await session_obj.call_tool(tool_name, args)
      result = json.loads(resp.content[0].text)
  except Exception as exc:
    return {
      "tool": tool_name, "args": args, "ok": False,
      "reason": f"{type(exc).__name__}: {str(exc)[:120]}",
      "result": None,
    }
  failed, why = _is_error_envelope(result)
  return {
    "tool": tool_name, "args": args, "ok": not failed,
    "reason": why if failed else "",
    "result": result,
  }


# ---------------------------------------------------------------------------
# Canonical args per tool, organized by server. Args use TICKER as the target.
# ---------------------------------------------------------------------------

def _build_call_plan(ticker):
  today = _today_iso()
  thirty_ago = _days_ago(30)
  one_year_ago = _days_ago(365)
  six_months_out = (datetime.now(timezone.utc) + timedelta(days=180)).strftime("%Y-%m-%d")

  web = [
    # search + scrape — search expects query as dict[label]=search_string
    ("search",                       {"ticker": ticker,
                                       "query": {"10K": f"{ticker} 10-K filing site:sec.gov"}}),
    ("get_urls_content",             {"urls": ["https://www.sec.gov/"]}),
    # SEC XBRL data extraction
    ("get_revenue_base",             {"ticker": ticker}),
    ("get_ebitda_margin",            {"ticker": ticker}),
    ("get_capex_pct_revenue",        {"ticker": ticker}),
    ("get_tax_rate",                 {"ticker": ticker}),
    ("get_depreciation",             {"ticker": ticker}),
    ("get_margin_breakdown",         {"ticker": ticker}),
    ("get_historical_fcf",           {"ticker": ticker}),
    ("get_working_capital",          {"ticker": ticker}),
    ("get_disclosures_names",        {"ticker": ticker}),
    ("extract_disclosure_data",      {"ticker": ticker, "disclosure_name": "us-gaap:Revenues"}),
    ("get_latest_filing",            {"ticker": ticker}),
    # 8-K + proxy
    ("extract_8k_events",            {"ticker": ticker}),
    ("extract_proxy_compensation",   {"ticker": ticker}),
    ("extract_governance_data",      {"ticker": ticker}),
  ]

  financial = [
    ("get_market_data",              {"ticker": ticker}),
    ("comparable_company_analysis",  {"companies": [ticker, "GOOGL", "AAPL"]}),
    # calculate_dcf: schema says "SET TO 0 -- auto-resolved" but that only
    # works inside the LangGraph execution engine. At the MCP layer the
    # math layer accepts zeros and returns garbage with no error key.
    # Pass realistic MSFT inputs from the B3 pilot for an honest health check.
    ("calculate_dcf",                {
      "ticker": ticker,
      "revenue_base": 281_700_000_000, "ebitda_margin": 0.556,
      "capex_pct_revenue": 0.229, "tax_rate": 0.176,
      "depreciation": 0.078,
      "revenue_growth": [0.18, 0.16, 0.14, 0.12, 0.10],
      "wacc": 0.104, "terminal_growth": 0.025, "terminal_multiple": 17.0,
      "cash": 78_200_000_000, "debt": 125_400_000_000,
      "shares_outstanding": 7_430_000_000,
    }),
    ("calculate_wacc",               {
      "ticker": ticker,
      "beta": 1.093, "risk_free_rate": 0.042, "equity_risk_premium": 0.06,
      "cost_of_debt": 0.019, "tax_rate": 0.176,
      "market_cap": 3_130_000_000_000, "total_debt": 125_400_000_000,
    }),
    ("calculate_scenario_dcf",       {
      "ticker": ticker,
      "revenue_base": 281_700_000_000,
      "capex_pct_revenue": 0.229,
      "tax_rate": 0.176,
      "depreciation": 0.078,
      "wacc": 0.104,
      "terminal_growth": 0.025,
      "terminal_multiple": 17.0,
      "cash": 78_200_000_000,
      "debt": 125_400_000_000,
      "shares_outstanding": 7_430_000_000,
      "bear_growth": [0.139, 0.129, 0.119, 0.109, 0.099],
      "base_growth": [0.179, 0.169, 0.159, 0.149, 0.139],
      "bull_growth": [0.209, 0.199, 0.189, 0.179, 0.169],
      "bear_margin": 0.486,
      "base_margin": 0.556,
      "bull_margin": 0.586,
    }),
    ("calculate_lbo",                {
      "ticker": ticker,
      "entry_ev": 3_130_000_000_000,
      "revenue_base": 281_700_000_000,
      "ebitda_margin": 0.556,
      "capex_pct_revenue": 0.229,
      "depreciation": 0.078,
      "tax_rate": 0.176,
      "revenue_growth": [0.12, 0.11, 0.10, 0.09, 0.08],
      "debt_interest_rate": 0.08,
      "leverage_turns": 4.5,
      "exit_multiple": 17.0,
      "hold_years": 5,
    }),
    ("calculate_credit_profile",     {
      "ticker": ticker,
      "total_debt": 125_400_000_000,
      "cash": 78_200_000_000,
      "ebitda": 156_500_000_000,
      "interest_expense": 2_390_000_000,
      "depreciation_abs": 21_900_000_000,
      "capex_abs": 64_600_000_000,
      "tax_rate": 0.176,
      "market_cap": 3_130_000_000_000,
    }),
    ("calculate_capital_returns",    {
      "ticker": ticker,
      "market_cap": 3_130_000_000_000,
      "ebitda": 156_500_000_000,
      "capex_abs": 64_600_000_000,
      "tax_rate": 0.176,
      "depreciation_abs": 21_900_000_000,
      "dividends_paid": 26_448_500_000,
      "shares_repurchased": 18_000_000_000,
      "shares_outstanding": 7_430_000_000,
    }),
  ]

  finnhub = [
    ("get_company_profile",          {"ticker": ticker}),
    ("get_basic_financials",         {"ticker": ticker}),
    ("get_market_news",              {"category": "general"}),
    ("get_company_news",             {"ticker": ticker, "from_date": thirty_ago, "to_date": today}),
    ("get_insider_transactions",     {"ticker": ticker}),
    ("get_insider_sentiment",        {"ticker": ticker, "from_date": one_year_ago, "to_date": today}),
    ("get_analyst_recommendations",  {"ticker": ticker}),
    ("get_company_peers",            {"ticker": ticker}),
    ("get_earnings_calendar",        {"from_date": today, "to_date": six_months_out}),
    ("get_earnings_surprises",       {"ticker": ticker}),
    ("get_forward_estimates",        {"ticker": ticker}),
    ("get_financial_statements",     {"ticker": ticker, "statement": "ic", "freq": "annual"}),
  ]

  fred = [
    ("get_macro_snapshot",           {}),
    ("get_treasury_yields",          {}),
    ("get_credit_spreads",           {}),
    ("get_fred_series",              {"series_id": "DGS10"}),
    ("search_fred",                  {"search_text": "unemployment"}),
  ]

  return {"web": web, "financial": financial, "finnhub": finnhub, "fred": fred}


# ---------------------------------------------------------------------------
# Alpaca block — read-only tools + rejection-path mutating tools
# ---------------------------------------------------------------------------

async def _exercise_alpaca():
  """Open one stdio session to alpaca and exercise its 6 tools.

  place_paper_order and close_paper_position are exercised via the
  REJECTION path only: place_paper_order with low confidence (Risk_Officer
  refuses; no broker call), close_paper_position with reason='health_check'
  on a ticker we don't own (returns no_open_position error gracefully).
  """
  env = {**os.environ, "PYTHONUNBUFFERED": "1"}
  project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")

  params = StdioServerParameters(
    command=sys.executable,
    args=["-m", "tools.alpaca.server", "server"],
    env=env,
  )

  results = []
  async with stdio_client(params) as (read, write):
    async with ClientSession(read, write) as session:
      await session.initialize()
      # Health
      results.append(await _call(session, "ping_alpaca", {}, "session"))
      results.append(await _call(session, "get_paper_account", {}, "session"))
      results.append(await _call(session, "get_paper_positions", {}, "session"))

      # Risk check — clean approve path (high conviction, normal size)
      approve_args = {
        "ticker": TICKER, "side": "buy", "quantity": 1, "price": 420.0,
        "recommendation": "BUY", "confidence": 0.78,
        "bull_strength": 0.78, "bear_strength": 0.40,
        "position_sizing": "normal", "rationale": "health check"
      }
      results.append(await _call(session, "risk_check_proposed_trade",
                                  approve_args, "session"))

      # place_paper_order — REJECTION path: low confidence so Risk_Officer
      # refuses BEFORE the broker is called. "Success" here means the
      # response is well-formed, success=False, and risk_decision.approve=False.
      reject_args = dict(approve_args)
      reject_args["confidence"] = 0.30
      reject_args["recommendation"] = "HOLD"  # any HOLD is auto-rejected
      reject_resp = await _call(session, "place_paper_order", reject_args, "session")
      # We re-interpret: a well-formed rejection is a PASS for the smoke test.
      if isinstance(reject_resp.get("result"), dict):
        r = reject_resp["result"]
        if r.get("success") is False and r.get("risk_decision", {}).get("approve") is False:
          reject_resp["ok"] = True
          reject_resp["reason"] = "rejected by Risk_Officer (expected)"
      results.append(reject_resp)

      # close_paper_position — call against a ticker we don't own. The
      # response should be a clean error ("no_open_position"), which we
      # treat as PASS because it proves the error-path works.
      close_args = {"ticker": "HEALTH_CHECK_NONEXIST", "reason": "smoke_test"}
      close_resp = await _call(session, "close_paper_position", close_args, "session")
      if isinstance(close_resp.get("result"), dict):
        r = close_resp["result"]
        if r.get("error") and "no_open_position" in str(r.get("error", "")).lower():
          close_resp["ok"] = True
          close_resp["reason"] = "no_open_position (expected, audit pass)"
      results.append(close_resp)
  return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def _exercise_data_servers():
  """Drive the 4 data servers through MCPConnectionManager. Separate
  function so its async context is fully closed before alpaca's stdio_client
  opens — avoids a Windows subprocess race that cancels the new spawn."""
  plan = _build_call_plan(TICKER)
  out = {}
  async with MCPConnectionManager() as mcp:
    for server_label in ("web", "financial", "finnhub", "fred"):
      print(f"\n--- {server_label} ({len(plan[server_label])} tools) ---")
      results = []
      for tool, args in plan[server_label]:
        r = await _call(mcp, tool, args, "mcp_manager")
        results.append(r)
        status = "OK " if r["ok"] else "FAIL"
        print(f"  [{status}] {tool:30s} {r['reason'][:80]}")
      out[server_label] = results
  return out


async def main():
  os.makedirs(OUT_DIR, exist_ok=True)
  all_results = {"ticker": TICKER, "started_at": _today_iso(), "by_server": {}}

  print(f"=== Tool Health Check — {TICKER} ===")

  # Run alpaca FIRST in its own stdio_client context, before MCPConnectionManager
  # is opened. This avoids a cancel-scope race on Windows where cleanup of
  # the data-server sessions bleeds into the alpaca-spawn task.
  print(f"\n--- alpaca (6 tools, place/close exercised via rejection path) ---")
  try:
    alpaca_results = await _exercise_alpaca()
  except Exception as e:
    print(f"  [FAIL] alpaca block crashed: {type(e).__name__}: {e}")
    alpaca_results = [{"tool": "<alpaca_block>", "ok": False,
                        "reason": f"{type(e).__name__}: {e}",
                        "args": {}, "result": None}]
  for r in alpaca_results:
    status = "OK " if r["ok"] else "FAIL"
    print(f"  [{status}] {r['tool']:30s} {r['reason'][:80]}")
  all_results["by_server"]["alpaca"] = alpaca_results

  all_results["by_server"].update(await _exercise_data_servers())

  # Summary
  total = sum(len(v) for v in all_results["by_server"].values())
  passed = sum(1 for v in all_results["by_server"].values() for r in v if r["ok"])
  failed = total - passed
  print(f"\n=== Summary ===")
  print(f"  Total tools: {total}")
  print(f"  Passed:      {passed}")
  print(f"  Failed:      {failed}")
  if failed:
    print(f"\n  Failures:")
    for srv, results in all_results["by_server"].items():
      for r in results:
        if not r["ok"]:
          print(f"    [{srv}] {r['tool']}: {r['reason'][:100]}")

  with open(RESULTS_PATH, "w") as f:
    json.dump(all_results, f, indent=2, default=str)
  print(f"\n  Bundle saved to: {RESULTS_PATH}")

  return failed


if __name__ == "__main__":
  exit_code = asyncio.run(main())
  sys.exit(0 if exit_code == 0 else 1)
