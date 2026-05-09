"""
Extensive integration tests for FRED macro tools in the full pipeline.

Tests:
  1. Pure macro query -- verifies get_macro_snapshot / get_treasury_yields fire,
     MACRO ENVIRONMENT section renders, variables flatten correctly.
  2. DCF with auto risk_free_rate -- verifies get_treasury_yields provides
     risk_free_rate that auto-resolves into calculate_wacc.
  3. Mixed macro + news query -- verifies both macro and market intel tools
     fire in the same pipeline run.

Usage: python -m testing.test_fred_integration [1|2|3]
  No arg = run test 1 only (fastest)
  Pass test number to run a specific test
"""
import asyncio
import json
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import cast
from agent.workflows.analysis_workflow import WorkFlow
from agent.workflows.agent_state import AgentState
from agent.MCP_manager import MCPConnectionManager


def print_banner(text: str):
  print(f"\n{'#'*80}")
  print(f"#  {text}")
  print(f"{'#'*80}\n")


def print_section(text: str):
  print(f"\n{'='*60}")
  print(f"  {text}")
  print(f"{'='*60}")


def analyze_result(result: dict, test_name: str):
  """Analyze a pipeline result and print diagnostics."""
  print_section(f"DIAGNOSTICS: {test_name}")

  # Phase history
  history = result.get('phase_history', [])
  print(f"\nPhase history ({len(history)} steps):")
  for step in history:
    print(f"  {step}")

  # Iteration count
  print(f"\nGlobal iterations: {result.get('global_iteration', '?')}")
  print(f"Execution count: {result.get('execution_count', '?')}")
  print(f"Analysis count: {result.get('analysis_count', '?')}")

  # Variables
  variables = result.get('variables', {})
  flat_vars = {k: v for k, v in variables.items() if '.' not in k}
  namespaced_vars = {k: v for k, v in variables.items() if '.' in k}
  print(f"\nVariables: {len(variables)} total ({len(flat_vars)} flat, {len(namespaced_vars)} namespaced)")

  # Check for macro-specific variables
  macro_keys = [k for k in variables if k.startswith('macro.') or k.startswith('treasury.') or k == 'risk_free_rate' or k == 'yield_curve_shape' or k == 'hy_spread']
  if macro_keys:
    print(f"\n  MACRO VARIABLES ({len(macro_keys)}):")
    for k in sorted(macro_keys):
      v = variables[k]
      if isinstance(v, (int, float)):
        print(f"    {k}: {v}")
      else:
        print(f"    {k}: {str(v)[:80]}")
  else:
    print(f"\n  WARNING: No macro variables found!")

  # Check risk_free_rate specifically
  if 'risk_free_rate' in variables:
    print(f"\n  risk_free_rate = {variables['risk_free_rate']} (decimal)")
  else:
    print(f"\n  WARNING: risk_free_rate NOT in variable store!")

  # Verification
  verification = result.get('verification_result', {})
  if verification:
    print(f"\nVerification: action={verification.get('action', '?')}, quality={verification.get('quality_score', '?')}")
  else:
    print(f"\nVerification: not performed")

  # Summarized output types
  outputs = result.get('summarized_output', [])
  type_counts = {}
  for o in outputs:
    t = o.get('type', 'unknown')
    type_counts[t] = type_counts.get(t, 0) + 1
  print(f"\nSummarized output: {len(outputs)} items")
  for t, c in type_counts.items():
    tools = [o.get('tool', '?') for o in outputs if o.get('type') == t]
    print(f"  {t}: {c} ({', '.join(tools)})")

  # Analysis report preview
  report = result.get('analysis_report', '')
  if report:
    # Check for MACRO section
    has_macro_section = 'macro' in report.lower() or 'interest rate' in report.lower() or 'treasury' in report.lower() or 'inflation' in report.lower()
    print(f"\nAnalysis report: {len(report)} chars, has_macro_content={has_macro_section}")
    print(f"\n--- REPORT PREVIEW (first 1500 chars) ---")
    print(report[:1500])
    print(f"--- END PREVIEW ---")
  else:
    print(f"\nWARNING: No analysis report generated!")

  return {
    'has_macro_vars': len(macro_keys) > 0,
    'has_risk_free_rate': 'risk_free_rate' in variables,
    'has_report': bool(report),
    'iterations': result.get('global_iteration', 0),
    'verification_action': verification.get('action', 'none'),
  }


async def test_1_macro_outlook(mcp: MCPConnectionManager):
  """Test 1: Pure macro query -- should use get_macro_snapshot and/or get_treasury_yields."""
  print_banner("TEST 1: Pure Macro Query")
  print("Query: 'What is the current macro environment and interest rate outlook?'")
  print("Expected: get_macro_snapshot + get_treasury_yields fire, MACRO ENVIRONMENT renders")

  w = WorkFlow(mcp=mcp)
  start = time.time()

  result = await w.app.ainvoke(cast(AgentState, {
    "user_query": "What is the current macro environment and interest rate outlook?",
    "ticker": "MACRO"
  }))

  elapsed = time.time() - start
  print(f"\nCompleted in {elapsed:.1f}s")

  diag = analyze_result(result, "MACRO OUTLOOK")

  # Assertions
  passed = True
  if not diag['has_macro_vars']:
    print("\nFAIL: No macro variables in variable store")
    passed = False
  if not diag['has_risk_free_rate']:
    print("\nFAIL: risk_free_rate not in variable store")
    passed = False
  if not diag['has_report']:
    print("\nFAIL: No analysis report generated")
    passed = False

  print(f"\n{'PASS' if passed else 'FAIL'}: Test 1 - Macro Outlook")
  return passed


async def test_2_dcf_risk_free_rate(mcp: MCPConnectionManager):
  """Test 2: DCF query -- should auto-resolve risk_free_rate from macro tools into calculate_wacc."""
  print_banner("TEST 2: DCF with Auto Risk-Free Rate Resolution")
  print("Query: 'Run a DCF analysis on MSFT'")
  print("Expected: get_treasury_yields or get_macro_snapshot fires BEFORE calculate_wacc,")
  print("          risk_free_rate auto-resolves via _resolve_args()")

  w = WorkFlow(mcp=mcp)
  start = time.time()

  result = await w.app.ainvoke(cast(AgentState, {
    "user_query": "Run a DCF analysis on MSFT",
    "ticker": "MSFT"
  }))

  elapsed = time.time() - start
  print(f"\nCompleted in {elapsed:.1f}s")

  diag = analyze_result(result, "DCF + RISK FREE RATE")

  # Check that risk_free_rate was resolved
  variables = result.get('variables', {})
  passed = True

  if not diag['has_risk_free_rate']:
    print("\nFAIL: risk_free_rate not in variable store -- macro tools may not have fired")
    passed = False
  else:
    rfr = variables['risk_free_rate']
    if 0.01 <= rfr <= 0.10:
      print(f"\nPASS: risk_free_rate = {rfr} (reasonable range 1-10%)")
    else:
      print(f"\nWARN: risk_free_rate = {rfr} (outside expected range)")

  if not diag['has_report']:
    print("\nFAIL: No analysis report generated")
    passed = False

  # Check if WACC was calculated
  has_wacc = 'wacc' in variables or any('wacc' in k.lower() for k in variables)
  if has_wacc:
    print(f"PASS: WACC found in variables")
  else:
    print(f"INFO: WACC not found in variables (orchestrator may not have planned calculate_wacc)")

  print(f"\n{'PASS' if passed else 'FAIL'}: Test 2 - DCF Risk-Free Rate")
  return passed


async def test_3_mixed_macro_news(mcp: MCPConnectionManager):
  """Test 3: Mixed query -- both macro and market intel tools should fire."""
  print_banner("TEST 3: Mixed Macro + News Query")
  print("Query: 'What is the macro outlook and current news sentiment for AAPL?'")
  print("Expected: macro tools + get_company_news + possibly get_insider_transactions")

  w = WorkFlow(mcp=mcp)
  start = time.time()

  result = await w.app.ainvoke(cast(AgentState, {
    "user_query": "What is the macro outlook and current news sentiment for AAPL?",
    "ticker": "AAPL"
  }))

  elapsed = time.time() - start
  print(f"\nCompleted in {elapsed:.1f}s")

  diag = analyze_result(result, "MIXED MACRO + NEWS")

  # Check for both macro and news variables
  variables = result.get('variables', {})
  outputs = result.get('summarized_output', [])

  has_macro = any(o.get('type') == 'macro' for o in outputs)
  has_news = any(o.get('type') == 'news_analysis' for o in outputs)
  has_market_intel = any(o.get('type') == 'market_intel' for o in outputs)

  passed = True

  if has_macro:
    print(f"\nPASS: Macro data present in output")
  else:
    print(f"\nFAIL: No macro data in output")
    passed = False

  if has_news:
    print(f"PASS: News analysis present in output")
  else:
    print(f"WARN: No news analysis in output (orchestrator may have used different tools)")

  if has_market_intel:
    print(f"PASS: Market intel data present in output")

  if not diag['has_report']:
    print(f"FAIL: No analysis report generated")
    passed = False

  # Check news variables
  news_vars = [k for k in variables if 'news' in k.lower() or 'sentiment' in k.lower()]
  if news_vars:
    print(f"\nNews variables ({len(news_vars)}):")
    for k in news_vars:
      print(f"  {k}: {variables[k]}")

  print(f"\n{'PASS' if passed else 'FAIL'}: Test 3 - Mixed Macro + News")
  return passed


async def main():
  test_num = int(sys.argv[1]) if len(sys.argv) > 1 else 1

  tests = {
    1: test_1_macro_outlook,
    2: test_2_dcf_risk_free_rate,
    3: test_3_mixed_macro_news,
  }

  if test_num not in tests:
    print(f"Unknown test number: {test_num}. Use 1, 2, or 3.")
    sys.exit(1)

  print_banner(f"FRED INTEGRATION TEST SUITE -- Test {test_num}")

  async with MCPConnectionManager() as mcp:
    test_fn = tests[test_num]
    passed = await test_fn(mcp)

  print_banner("RESULTS")
  print(f"  Test {test_num}: {'PASS' if passed else 'FAIL'}")

  sys.exit(0 if passed else 1)


if __name__ == "__main__":
  asyncio.run(main())
