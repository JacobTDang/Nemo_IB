"""Test Fix #3: comparable_company_analysis is injected for relative-valuation queries."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.workflows.analysis_workflow import WorkFlow


def _empty_plan():
  return {'task_type': 'test', 'ticker': 'AAPL', 'reasoning': 'test', 'tools_sequence': []}


def test_comps_injected_on_comprehensive_query():
  plan = _empty_plan()
  result = WorkFlow._inject_missing_intent_tools(
    "Run a comprehensive analysis on AAPL", plan, "AAPL", already_run=set(), variables={}
  )
  tools = [t['tool'] for t in result['tools_sequence']]
  assert 'comparable_company_analysis' in tools, f"comps tool not injected: {tools}"
  assert 'get_company_peers' in tools, f"peers prereq not injected: {tools}"
  print(f"PASS: comprehensive query injects {tools}")


def test_comps_injected_on_comps_query():
  plan = _empty_plan()
  result = WorkFlow._inject_missing_intent_tools(
    "How does AAPL trade vs its peers?", plan, "AAPL", already_run=set(), variables={}
  )
  tools = [t['tool'] for t in result['tools_sequence']]
  assert 'comparable_company_analysis' in tools, f"comps tool not injected: {tools}"
  print(f"PASS: peer query injects comps")


def test_not_injected_on_sentiment_query():
  plan = _empty_plan()
  result = WorkFlow._inject_missing_intent_tools(
    "What's the news sentiment for AAPL?", plan, "AAPL", already_run=set(), variables={}
  )
  tools = [t['tool'] for t in result['tools_sequence']]
  assert 'comparable_company_analysis' not in tools, f"sentiment query should NOT inject comps: {tools}"
  print(f"PASS: sentiment query does not inject comps")


def test_idempotent_when_peers_already_have():
  # Simulate peers already in variable store
  plan = _empty_plan()
  result = WorkFlow._inject_missing_intent_tools(
    "Run a comprehensive analysis on AAPL",
    plan, "AAPL", already_run=set(),
    variables={'company_peers': ['MSFT', 'GOOGL', 'META']}
  )
  tools = [t['tool'] for t in result['tools_sequence']]
  # get_company_peers should NOT be added (already have them)
  assert tools.count('get_company_peers') == 0, "should not add get_company_peers when peers already present"
  assert 'comparable_company_analysis' in tools
  # The comps args should include the actual peer list
  comps_step = next(t for t in result['tools_sequence'] if t['tool'] == 'comparable_company_analysis')
  assert comps_step['arguments']['companies'] == ['MSFT', 'GOOGL', 'META']
  print(f"PASS: existing peers used directly, no duplicate get_company_peers")


def test_no_double_injection():
  # Already in already_run -- don't re-inject
  plan = _empty_plan()
  result = WorkFlow._inject_missing_intent_tools(
    "comprehensive analysis on AAPL",
    plan, "AAPL",
    already_run={'comparable_company_analysis'}, variables={}
  )
  tools = [t['tool'] for t in result['tools_sequence']]
  assert tools.count('comparable_company_analysis') == 0, "should not re-add when already ran"
  print(f"PASS: skipped when already in already_run")


if __name__ == "__main__":
  test_comps_injected_on_comprehensive_query()
  test_comps_injected_on_comps_query()
  test_not_injected_on_sentiment_query()
  test_idempotent_when_peers_already_have()
  test_no_double_injection()
  print("\nAll tests passed.")
