from .openrouter_template import OpenRouterModel
from .workflows.agent_state import AgentState
from .workflows.constants import MAX_ITERATIONS, MAX_EXECUTIONS, MAX_ANALYSES
from pydantic import BaseModel
from typing import Dict, Any, Optional
import json
import sys


class MasterDecision(BaseModel):
  next_action: str
  reasoning: str
  query_complexity: str
  revision_context: Optional[Dict[str, Any]] = None


class Master_Orchestrator(OpenRouterModel):
  """
  Hub orchestrator for the hub-and-spoke workflow architecture.
  Reads full state, decides which spoke to route to next.
  Stateless per invocation -- clears conversation history each call.
  """
  response_schema = MasterDecision
  REASONING_EFFORT = None  # No reasoning -- just output the routing decision JSON

  def __init__(self, model_name: str = "nvidia/nemotron-3-nano-30b-a3b:free"):
    super().__init__(model_name=model_name, api_key_env="OPENROUTER_NEMOTRON")

  def _build_state_summary(self, state: AgentState) -> str:
    """Build a concise summary of current state for the LLM prompt."""
    current_phase = state.get('current_phase', 'init')
    query_complexity = state.get('query_complexity', '')
    user_query = state.get('user_query', '')
    ticker = state.get('ticker', '')

    summary = f"""CURRENT STATE:
- User Query: {user_query}
- Ticker: {ticker}
- Current Phase: {current_phase}
- Query Complexity: {query_complexity or 'NOT SET (you must classify this)'}
- Phase History: {state.get('phase_history', [])}
"""

    # Add what data is available based on phase
    if state.get('data_requirements'):
      summary += f"- Data Requirements: {len(state['data_requirements'])} identified\n"
    if state.get('analytical_considerations'):
      summary += f"- Analytical Considerations: {len(state['analytical_considerations'])} for analysis agent\n"

    if state.get('execution_plan'):
      plan = state['execution_plan']
      tool_count = len(plan.get('tools_sequence', []))
      summary += f"- Execution Plan: {plan.get('task_type', 'N/A')} with {tool_count} tools\n"

    if state.get('summarized_output'):
      output_count = len(state['summarized_output'])
      summary += f"- Execution Results: {output_count} data items collected\n"
      stats = state.get('execution_stats', {})
      if stats:
        summary += f"  Successes: {stats.get('successes', '?')}, Errors: {stats.get('errors', '?')}\n"
        error_rate = stats.get('errors', 0) / max(stats.get('total', 1), 1)
        if error_rate > 0.5:
          summary += f"  WARNING: {error_rate:.0%} tool failure rate. Data likely insufficient.\n"

    model_outputs = state.get('model_outputs', {})
    if model_outputs and model_outputs.get('models_run'):
      models_run = model_outputs.get('models_run', [])
      summary += f"- Model Outputs: {models_run}\n"
      if 'scenario_dcf' in model_outputs:
        pr = model_outputs['scenario_dcf'].get('price_range', {})
        summary += (f"  Scenario DCF: Bear=${pr.get('low', 0):.2f} | "
                    f"Base=${pr.get('mid', 0):.2f} | Bull=${pr.get('high', 0):.2f}\n")
      if 'credit_profile' in model_outputs:
        cp = model_outputs['credit_profile']
        summary += (f"  Credit Profile: {cp.get('credit_label', 'N/A')} | "
                    f"Net Debt/EBITDA: {cp.get('net_debt_ebitda', 0):.1f}x\n")
      if 'capital_returns' in model_outputs:
        cr = model_outputs['capital_returns']
        summary += f"  Total Shareholder Yield: {cr.get('total_shareholder_yield_pct', 0):.1f}%\n"
      if 'lbo' in model_outputs:
        lb = model_outputs['lbo']
        summary += f"  LBO: IRR={lb.get('irr_pct', 0):.1f}% | MOIC={lb.get('moic', 0):.2f}x\n"

    if state.get('analysis_report'):
      report = state['analysis_report']
      variables = state.get('variables', {})
      summary += f"- Analysis Report: generated ({len(report)} chars)\n"
      signal = variables.get('analysis.signal', '')
      if signal:
        summary += f"  Signal: {signal}\n"
      conclusion = variables.get('analysis.conclusion', '')
      if conclusion:
        summary += f"  Conclusion: {conclusion[:200]}\n"
      data_gaps = variables.get('analysis.data_gaps', [])
      if data_gaps:
        summary += f"  DATA GAPS ({len(data_gaps)}): {data_gaps}\n"

    plan_verification = state.get('plan_verification', {})
    if plan_verification:
      complete = plan_verification.get('complete', True)
      pv_summary = plan_verification.get('summary', '')
      gaps = plan_verification.get('gaps', [])
      critical = [g for g in gaps if g.get('priority') == 'critical']
      status_str = "COMPLETE" if complete else f"INCOMPLETE ({len(critical)} critical gaps)"
      summary += f"- Plan Verification: {status_str}\n"
      if pv_summary:
        summary += f"  {pv_summary}\n"
      for g in critical:
        summary += f"  - [CRITICAL] {g['description']} -> {g['recommended_tool']}\n"

    if state.get('revision_context'):
      summary += f"- Revision Context: {json.dumps(state['revision_context'], default=str)[:300]}\n"

    # Per-phase counters (shown against limits so model knows remaining budget)
    summary += f"- Execution Count: {state.get('execution_count', 0)} / {MAX_EXECUTIONS} max\n"
    summary += f"- Analysis Count: {state.get('analysis_count', 0)} / {MAX_ANALYSES} max\n"
    summary += f"- Global Iteration: {state.get('global_iteration', 0)} / {MAX_ITERATIONS} max\n"

    # Show gathered variables (flat keys only, skip namespaced)
    variables = state.get('variables', {})
    if variables:
      flat_vars = {k: v for k, v in variables.items() if '.' not in k}
      if flat_vars:
        summary += f"\nGATHERED DATA ({len(flat_vars)} variables):\n"
        for k, v in flat_vars.items():
          summary += f"  {k}: {v}\n"

    # Show notes from previous iterations
    master_notes = state.get('master_notes', [])
    if master_notes:
      summary += f"\nYOUR NOTES FROM PREVIOUS ITERATIONS:\n"
      for note in master_notes:
        summary += f"  - {note}\n"

    return summary

  def _build_system_prompt(self) -> str:
    """Build the master orchestrator system prompt."""
    return f"""You are the Master Orchestrator in a financial analysis workflow. You decide which sub-agent to invoke next based on the current state.

AVAILABLE SUB-AGENTS:
1. probe - Generates strategic research questions before analysis. USE when: first pass on standard/complex queries. SKIP when: simple queries, already probed.
2. plan - Creates a tool execution plan (which MCP tools to call and in what order). USE when: need to gather data, or need a revised plan after plan_verification flagged critical gaps. SKIP when: plan already exists and no revision needed.
3. execute - Runs the tools from the execution plan. USE when: plan exists and needs execution. SKIP when: no plan exists.
4. model - Runs financial models (scenario DCF, credit profile, capital returns, LBO) using the gathered variable store. USE when: plan_verification says COMPLETE and data has been collected. Runs between execute and analyze.
5. analyze - Performs financial analysis on gathered data + model outputs. USE when: model phase is complete (phase is "modeled"). SKIP when: model phase hasn't run yet.
6. done - Return the analysis to the user. USE when: analysis is complete.

NOTE: After every execution, Plan Verification runs automatically (DeepSeek R1). It checks if all
critical data is present. If INCOMPLETE, the system guardrails will re-route to plan automatically.
You should route to model only when Plan Verification shows COMPLETE.

QUERY COMPLEXITY RULES (classify on first pass, then persist):
- "simple": Factual lookups ("what is AAPL revenue?") -> plan -> execute -> done (skip probe)
- "standard": Standard analysis ("run DCF on AAPL") -> probe -> plan -> execute -> model -> analyze -> done
- "complex": Multi-faceted analysis ("compare AAPL vs MSFT valuation with sensitivity") -> probe -> plan -> execute -> model -> analyze -> done (may need multiple data-gathering loops before model)

ROUTING LOGIC:
- Phase "init" + no complexity set -> classify query, then route to probe (standard/complex) or plan (simple)
- Phase "probed" -> route to plan
- Phase "planned" -> route to execute
- Phase "executed" -> route to model (standard/complex) or done (simple, if query answered by data alone)
  Note: Plan Verification guardrails will intercept and re-route to plan if critical gaps remain.
- Phase "modeled" -> route to analyze
- Phase "analyzed" -> done (analysis is the final output, no separate verification step)
  Exception: route to plan with revision_context if data gaps in analysis are severe and execution budget remains.

REVISION HANDLING:
When routing to plan for data gaps, populate revision_context:
- {"type": "missing_data", "feedback": <description of what is missing and which tools to use>}

GUARDRAILS (you cannot override these, the system enforces them):
- Maximum {MAX_ITERATIONS} global iterations
- Maximum {MAX_EXECUTIONS} executions
- Maximum {MAX_ANALYSES} analyses

RULES:
- next_action must be one of: probe, plan, execute, model, analyze, done
- query_complexity must be set on every call (persist the same value after first classification)
- revision_context is null unless routing to plan for data gaps

OUTPUT FORMAT - you must output exactly this JSON structure, no other fields:
{
  "next_action": "<one of: probe, plan, execute, model, analyze, done>",
  "reasoning": "<1-2 sentence explanation of why you chose this action>",
  "query_complexity": "<simple, standard, or complex>",
  "revision_context": null
}

Example when re-planning for gaps:
{
  "next_action": "plan",
  "reasoning": "Analysis missing WACC inputs. Fetching market data and macro rates before re-running DCF.",
  "query_complexity": "standard",
  "revision_context": {"type": "missing_data", "feedback": "WACC inputs missing: fetch get_market_data and get_macro_snapshot"}
}"""

  def decide(self, state: AgentState) -> Dict[str, Any]:
    """
    Read full state, decide which spoke to route to next.
    Clears conversation history each call (stateless).

    Returns:
      Dict with keys: next_action, reasoning, query_complexity, revision_context
    """
    # Stateless: clear history each invocation
    self.conversatoin_history = []

    state_summary = self._build_state_summary(state)
    system_prompt = self._build_system_prompt()

    user_prompt = f"""{state_summary}

Based on the current state above, decide what should happen next."""

    print(f"\n{'='*60}", file=sys.stderr, flush=True)
    print(f"MASTER ORCHESTRATOR - Deciding next action...", file=sys.stderr, flush=True)
    print(f"Current phase: {state.get('current_phase', 'init')}", file=sys.stderr, flush=True)
    print(f"{'='*60}\n", file=sys.stderr, flush=True)

    # First attempt
    response = self.generate_response(prompt=user_prompt, system_prompt=system_prompt)

    try:
      decision = self.parse_response(response)
      result = decision.model_dump()

      # Validate next_action
      valid_actions = {'probe', 'plan', 'execute', 'model', 'analyze', 'done'}
      if result['next_action'] not in valid_actions:
        raise ValueError(f"Invalid next_action: {result['next_action']}")

      self._log_decision(result)
      return result
    except Exception as e:
      print(f"\nMaster Orchestrator: First attempt failed: {e}. Retrying...", file=sys.stderr, flush=True)

    # Retry with shorter prompt
    self.conversatoin_history = []

    retry_prompt = f"""Current phase: {state.get('current_phase', 'init')}
Query: {state.get('user_query', '')}
Query complexity: {state.get('query_complexity', 'standard')}

Decide the next action."""

    response = self.generate_response(prompt=retry_prompt, system_prompt="You are a workflow router. Decide the next action.")

    try:
      decision = self.parse_response(response)
      result = decision.model_dump()
      self._log_decision(result)
      return result
    except Exception as e:
      # Both attempts failed -- use a deterministic phase-based fallback rather than crashing.
      phase = state.get('current_phase', 'init')
      fallback_map = {
        'init': 'probe',
        'probed': 'plan',
        'planned': 'execute',
        'executed': 'model',
        'modeled': 'analyze',
        'analyzed': 'done',
      }
      fallback_action = fallback_map.get(phase, 'analyze')
      print(
        f"\nWARNING: Master Orchestrator LLM failed twice (phase={phase}, error={e}). "
        f"Using deterministic fallback: {fallback_action}",
        file=sys.stderr, flush=True
      )
      result = {
        'next_action': fallback_action,
        'reasoning': f'LLM parse failure after 2 attempts. Deterministic fallback for phase={phase}.',
        'query_complexity': state.get('query_complexity', 'standard'),
        'revision_context': None,
      }
      self._log_decision(result)
      return result

  def _log_decision(self, decision: Dict[str, Any]):
    """Log the master's routing decision to stderr."""
    print(f"\n{'='*60}", file=sys.stderr, flush=True)
    print(f"MASTER DECISION", file=sys.stderr, flush=True)
    print(f"  Next Action: {decision.get('next_action', 'N/A')}", file=sys.stderr, flush=True)
    print(f"  Complexity:  {decision.get('query_complexity', 'N/A')}", file=sys.stderr, flush=True)
    print(f"  Reasoning:   {decision.get('reasoning', 'N/A')}", file=sys.stderr, flush=True)
    if decision.get('revision_context'):
      print(f"  Revision:    {json.dumps(decision['revision_context'], default=str)[:200]}", file=sys.stderr, flush=True)
    print(f"{'='*60}\n", file=sys.stderr, flush=True)
