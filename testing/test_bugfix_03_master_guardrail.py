"""Bug 3: master_node forces `done` when verifier approves and plan_verify is complete.

Closes the re-plan loop bug where the LLM saw data gaps in state summary and
routed `analyzed -> plan` despite both validators saying we're done.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from unittest.mock import MagicMock
from agent.workflows.analysis_workflow import WorkFlow


def _state(qc_action='approve', plan_verify_complete=True, execution_count=3):
  return {
    'current_phase': 'analyzed',
    'execution_count': execution_count,
    'analysis_count': 1,
    'global_iteration': 5,
    'phase_history': ['init', 'probed', 'planned', 'executed', 'modeled', 'analyzed'],
    'variables': {
      'analysis.qc_action': qc_action,
      'analysis.data_gaps': ['some gap'],  # data gaps present but irrelevant when approve+complete
    },
    'plan_verification': {'complete': plan_verify_complete, 'summary': 'ok', 'gaps': []},
    'query_complexity': 'complex',
  }


def _wf_with_llm_picking(action: str):
  """Construct a WorkFlow with a mocked master LLM that picks the given action."""
  wf = WorkFlow.__new__(WorkFlow)  # bypass __init__ — no MCP needed for unit test
  wf.master_orchestrator = MagicMock()
  wf.master_orchestrator.decide.return_value = {
    'next_action': action,
    'reasoning': 'mock',
    'query_complexity': 'complex',
    'revision_context': {'type': 'missing_data', 'feedback': 'x'} if action == 'plan' else None,
  }
  return wf


def test_guardrail_forces_done_on_approve():
  wf = _wf_with_llm_picking('plan')  # LLM tries to re-plan
  result = wf.master_node(_state(qc_action='approve', plan_verify_complete=True))
  assert result['next_action'] == 'done', \
    f"guardrail failed, got {result['next_action']}"
  print("PASS: approve + complete forces 'plan' -> 'done'")


def test_guardrail_silent_when_verifier_revised():
  wf = _wf_with_llm_picking('plan')
  result = wf.master_node(_state(qc_action='revise', plan_verify_complete=True))
  # Verifier said revise -> LLM choice 'plan' should pass through
  assert result['next_action'] == 'plan', \
    f"guardrail should not fire when verifier revised, got {result['next_action']}"
  print("PASS: guardrail silent when verifier revised")


def test_guardrail_silent_when_plan_verify_incomplete():
  wf = _wf_with_llm_picking('plan')
  result = wf.master_node(_state(qc_action='approve', plan_verify_complete=False))
  assert result['next_action'] == 'plan', \
    f"guardrail should not fire when plan_verify incomplete, got {result['next_action']}"
  print("PASS: guardrail silent when plan_verify incomplete")


def test_done_passes_through_unchanged():
  wf = _wf_with_llm_picking('done')
  result = wf.master_node(_state(qc_action='approve', plan_verify_complete=True))
  assert result['next_action'] == 'done'
  print("PASS: LLM's own 'done' passes through unchanged")


if __name__ == "__main__":
  test_guardrail_forces_done_on_approve()
  test_guardrail_silent_when_verifier_revised()
  test_guardrail_silent_when_plan_verify_incomplete()
  test_done_passes_through_unchanged()
  print("\nAll tests passed.")
