"""Phase 3c: End-to-end debate wired into analyze_node.

This test mocks the analyst (so we don't pay for a 30s analyst call) and
verifies the debate wiring around it:
  - bull and bear run in parallel after the analyst
  - arbiter synthesizes
  - the persisted thesis uses the arbiter's recommendation, not the analyst's
  - the markdown returned to the user includes a DEBATE section
"""
import sys, os
import asyncio
from unittest.mock import patch, MagicMock
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.Financial_Analysis_Agent import AnalysisReport
from agent.Bull_Agent import BullCase
from agent.Bear_Agent import BearCase
from agent.Arbiter_Agent import ArbiterVerdict
from state.schema import init_schema, get_connection
from state.theses import latest_thesis


def _make_analyst_report():
  return AnalysisReport(
    executive_summary="AAPL fundamentals strong, services growth 14% YoY.",
    recommendation="BUY",  # Analyst's call
    signal="bullish",
    valuation="DCF $200 vs $195 price.",
    financial_performance="Revenue $94B, ops margin 30.8%",
    risks=["China iPhone -19%"],
    assumptions=["Services growth 12-14%"],
    data_gaps=[],
    confidence=0.72,
    conclusion="Buy on services-driven re-rating.",
  )


def _fake_bull():
  return BullCase(
    thesis="Services inflection drives 25% upside.",
    catalysts=["Q3 earnings"],
    upside_targets=["$230 if Services >$28B"],
    refutation_of_bear="China is small relative to Services.",
    conviction=0.78,
  )


def _fake_bear():
  return BearCase(
    thesis="Decel + valuation = 20% downside.",
    risks=["P/E 31x", "Services decel 22% to 14%"],
    downside_targets=["$160"],
    refutation_of_bull="Services decel is real.",
    conviction=0.72,
  )


def _fake_arbiter_hold():
  """Arbiter overrides analyst BUY with HOLD."""
  return ArbiterVerdict(
    final_recommendation="HOLD",  # Different from analyst's BUY
    confidence=0.61,
    bull_strength=0.65, bear_strength=0.62,
    decisive_factors=["Both sides cite real evidence; spread is small."],
    acknowledged_risks=["China weakness"],
    conditions_to_change_mind=["Services >18%", "Services <10%"],
    position_sizing_guidance="cautious",
    rationale="Close debate — HOLD with cautious sizing pending next earnings.",
  )


def _clean_aapl_theses():
  conn = get_connection()
  try:
    conn.execute("DELETE FROM theses WHERE ticker = 'AAPL_E2E_TEST'")
    conn.commit()
  finally:
    conn.close()


async def _run_analyze_with_mocked_components():
  """Drive analyze_node end-to-end with everything mocked except the debate
  pieces, so we can assert the wiring without making 4 real LLM calls."""
  from agent.workflows.analysis_workflow import WorkFlow
  from agent.MCP_manager import MCPConnectionManager

  # Patch agent constructors so WorkFlow init doesn't make API calls
  with patch('agent.workflows.analysis_workflow.Master_Orchestrator'), \
       patch('agent.workflows.analysis_workflow.Probing_Agent'), \
       patch('agent.workflows.analysis_workflow.Orchestrator_Agent'), \
       patch('agent.workflows.analysis_workflow.Search_Summarizer_Agent'), \
       patch('agent.workflows.analysis_workflow.Financial_Analysis_Agent') as MockFA, \
       patch('agent.workflows.analysis_workflow.Financial_Modeling_Agent'), \
       patch('agent.workflows.analysis_workflow.Plan_Verifier_Agent'), \
       patch('agent.workflows.analysis_workflow.News_Processing_Agent'), \
       patch('agent.workflows.analysis_workflow.Verification_Agent') as MockVer, \
       patch('agent.workflows.analysis_workflow.Bull_Agent') as MockBull, \
       patch('agent.workflows.analysis_workflow.Bear_Agent') as MockBear, \
       patch('agent.workflows.analysis_workflow.Arbiter_Agent') as MockArb:

    analyst_report = _make_analyst_report()
    MockFA.return_value.analyze.return_value = (
      analyst_report.render_markdown(), analyst_report
    )
    MockVer.return_value.verify.return_value = {
      'action': 'approve', 'quality_score': 0.85, 'feedback': ''
    }
    MockBull.return_value.argue.return_value = _fake_bull()
    MockBear.return_value.argue.return_value = _fake_bear()
    MockArb.return_value.judge.return_value = _fake_arbiter_hold()

    mcp = MagicMock(spec=MCPConnectionManager)
    wf = WorkFlow(mcp)

    state = {
      'user_query': 'Analyze AAPL',
      'ticker': 'AAPL_E2E_TEST',
      'summarized_output': [{'tool': 'fake', 'result': 'fake'}],
      'execution_plan': {'tools_sequence': []},
      'analytical_considerations': [],
      'variables': {
        'revenue_ttm': 388e9, 'services_growth_yoy': 0.14, 'iphone_growth_yoy': -0.02,
        'operating_margin': 0.308, 'pe_trailing': 31, 'current_price': 195.0,
        'market_cap': 3e12, 'beta': 1.21,
      },
      'plan_verification': None,
      'revision_context': None,
      'model_outputs': {},
      'analysis_count': 0,
    }
    result = await wf.analyze_node(state)
    return result, analyst_report


def test_debate_wires_in_after_analysis():
  init_schema(); _clean_aapl_theses()
  result, analyst_report = asyncio.run(_run_analyze_with_mocked_components())

  assert result['current_phase'] == 'analyzed'
  vars_after = result['variables']

  # The effective recommendation should match the arbiter's (HOLD), not the
  # analyst's (BUY). This proves arbiter took precedence.
  assert vars_after['analysis.recommendation'] == 'HOLD', \
    f"expected HOLD (arbiter), got {vars_after['analysis.recommendation']}"
  assert vars_after['analysis.confidence'] == 0.61, \
    f"expected arbiter confidence 0.61, got {vars_after['analysis.confidence']}"
  assert vars_after['analysis.bull_strength'] == 0.65
  assert vars_after['analysis.bear_strength'] == 0.62
  assert vars_after['analysis.position_sizing'] == 'cautious'
  print("PASS: arbiter verdict overrides analyst recommendation in state")


def test_debate_section_appears_in_returned_markdown():
  init_schema(); _clean_aapl_theses()
  result, _ = asyncio.run(_run_analyze_with_mocked_components())
  md = result['analysis_report']
  assert "## DEBATE" in md, "DEBATE section missing from markdown"
  assert "Bull (conviction" in md, "Bull section header missing"
  assert "Bear (conviction" in md, "Bear section header missing"
  assert "Arbiter Verdict: HOLD" in md, "Arbiter verdict label missing"
  assert "Decisive factors" in md or "decisive" in md.lower()
  print("PASS: returned markdown contains DEBATE section with both sides + verdict")


def test_persisted_thesis_uses_arbiter_recommendation():
  init_schema(); _clean_aapl_theses()
  result, _ = asyncio.run(_run_analyze_with_mocked_components())
  thesis = latest_thesis('AAPL_E2E_TEST')
  assert thesis is not None, "thesis was not persisted"
  assert thesis['recommendation'] == 'HOLD', \
    f"persisted thesis should use arbiter's HOLD, got {thesis['recommendation']}"
  assert thesis['confidence'] == 0.61
  print(f"PASS: persisted thesis #{thesis['thesis_id']} uses arbiter recommendation (HOLD)")


def test_debate_skipped_for_info_recommendation():
  """INFO queries (factual lookups) skip the debate entirely."""
  init_schema(); _clean_aapl_theses()
  from agent.workflows.analysis_workflow import WorkFlow
  from agent.MCP_manager import MCPConnectionManager

  info_report = AnalysisReport(
    executive_summary="Apple is a tech company.",
    recommendation="INFO", signal="n/a",
    valuation="N/A", financial_performance="N/A",
    risks=[], assumptions=[], data_gaps=[], confidence=0.5,
    conclusion="Factual lookup.",
  )

  with patch('agent.workflows.analysis_workflow.Master_Orchestrator'), \
       patch('agent.workflows.analysis_workflow.Probing_Agent'), \
       patch('agent.workflows.analysis_workflow.Orchestrator_Agent'), \
       patch('agent.workflows.analysis_workflow.Search_Summarizer_Agent'), \
       patch('agent.workflows.analysis_workflow.Financial_Analysis_Agent') as MockFA, \
       patch('agent.workflows.analysis_workflow.Financial_Modeling_Agent'), \
       patch('agent.workflows.analysis_workflow.Plan_Verifier_Agent'), \
       patch('agent.workflows.analysis_workflow.News_Processing_Agent'), \
       patch('agent.workflows.analysis_workflow.Verification_Agent') as MockVer, \
       patch('agent.workflows.analysis_workflow.Bull_Agent') as MockBull, \
       patch('agent.workflows.analysis_workflow.Bear_Agent') as MockBear, \
       patch('agent.workflows.analysis_workflow.Arbiter_Agent') as MockArb:

    MockFA.return_value.analyze.return_value = (info_report.render_markdown(), info_report)
    MockVer.return_value.verify.return_value = {'action': 'approve', 'quality_score': 0.7, 'feedback': ''}
    MockBull.return_value.argue.return_value = _fake_bull()  # Should not be called

    mcp = MagicMock(spec=MCPConnectionManager)
    wf = WorkFlow(mcp)

    state = {
      'user_query': 'When was Apple founded?',
      'ticker': 'AAPL_E2E_TEST',
      'summarized_output': [{'tool': 'fake', 'result': 'fake'}],
      'execution_plan': {'tools_sequence': []},
      'analytical_considerations': [],
      'variables': {
        'company_name': 'Apple Inc', 'founded': '1976',
        'sector': 'Technology', 'market_cap': 3e12, 'revenue_ttm': 388e9,
      },
      'plan_verification': None, 'revision_context': None,
      'model_outputs': {}, 'analysis_count': 0,
    }
    result = asyncio.run(wf.analyze_node(state))

    # Bull should NOT have been called for INFO recommendation
    MockBull.return_value.argue.assert_not_called()
    MockBear.return_value.argue.assert_not_called()
    MockArb.return_value.judge.assert_not_called()
    # No DEBATE section in markdown
    assert "## DEBATE" not in result['analysis_report']
  print("PASS: INFO recommendation skips the debate entirely")


if __name__ == "__main__":
  test_debate_wires_in_after_analysis()
  test_debate_section_appears_in_returned_markdown()
  test_persisted_thesis_uses_arbiter_recommendation()
  test_debate_skipped_for_info_recommendation()
  _clean_aapl_theses()
  print("\nAll Phase 3c e2e debate tests passed.")
