from .ollama_template import OllamaModel
from .workflows.agent_state import AgentState
from pydantic import BaseModel
from typing import Dict, Any, Optional
import json
import sys


class MasterDecision(BaseModel):
  next_action: str
  reasoning: str
  query_complexity: str
  revision_context: Optional[Dict[str, Any]] = None
  notes: Optional[str] = None


class Master_Orchestrator(OllamaModel):
  """
  Hub orchestrator for the hub-and-spoke workflow architecture.
  Reads full state, decides which spoke to route to next.
  Stateless per invocation -- clears conversation history each call.
  """
  response_schema = MasterDecision

  def __init__(self, model_name: str = "orchestrator:latest"):
    super().__init__(model_name=model_name)

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
- Global Iteration: {state.get('global_iteration', 0)}
- Phase History: {state.get('phase_history', [])}
"""

    # Add what data is available based on phase
    if state.get('research_questions'):
      questions = state['research_questions']
      summary += f"- Probing Questions: {len(questions)} generated\n"

    if state.get('execution_plan'):
      plan = state['execution_plan']
      tool_count = len(plan.get('tools_sequence', []))
      summary += f"- Execution Plan: {plan.get('task_type', 'N/A')} with {tool_count} tools\n"

    if state.get('summarized_output'):
      output_count = len(state['summarized_output'])
      summary += f"- Execution Results: {output_count} data items collected\n"

    if state.get('analysis_report'):
      report_len = len(state['analysis_report'])
      summary += f"- Analysis Report: generated ({report_len} chars)\n"

    if state.get('verification_result'):
      verification = state['verification_result']

      if 'error' in verification:
        parse_failures = state.get('verify_parse_failures', 0)
        raw = verification.get('raw_response', '')[:400]
        summary += f"- Verification: PARSE FAILED ({parse_failures} consecutive failures) - verifier output was not valid JSON\n"
        summary += f"  Raw verifier feedback: {raw}\n"
      else:
        action = verification.get('action', 'unknown')
        score = verification.get('quality_score', 'N/A')
        summary += f"- Verification: action={action}, quality_score={score}\n"
        summary += f"  Reasoning: {verification.get('action_reasoning', 'N/A')}\n"

        if action == 'revise' or action == 'reject':
          weaknesses = verification.get('weaknesses', [])
          missing = verification.get('missing_components', [])
          summary += f"  Weaknesses: {weaknesses}\n"
          summary += f"  Missing Components: {missing}\n"

    if state.get('revision_context'):
      summary += f"- Revision Context: {json.dumps(state['revision_context'], default=str)[:300]}\n"

    # Per-phase counters
    summary += f"- Execution Count: {state.get('execution_count', 0)}\n"
    summary += f"- Analysis Count: {state.get('analysis_count', 0)}\n"

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
    return """You are the Master Orchestrator in a financial analysis workflow. You decide which sub-agent to invoke next based on the current state.

AVAILABLE SUB-AGENTS:
1. probe - Generates strategic research questions before analysis. USE when: first pass on standard/complex queries. SKIP when: simple queries, already probed.
2. plan - Creates a tool execution plan (which MCP tools to call and in what order). USE when: need to gather data, or need a revised plan after verification feedback. SKIP when: plan already exists and no revision needed.
3. execute - Runs the tools from the execution plan. USE when: plan exists and needs execution, or re-execution with additional tools. SKIP when: no plan exists.
4. analyze - Performs financial analysis on gathered data. USE when: data has been collected and needs analysis, or re-analysis with verifier feedback. SKIP when: no data collected.
5. verify - Quality checks the analysis output. USE when: analysis is complete and needs verification. SKIP when: simple queries, already verified and approved.

QUERY COMPLEXITY RULES (classify on first pass, then persist):
- "simple": Factual lookups ("what is AAPL revenue?") -> plan -> execute -> done (skip probe, verify)
- "standard": Standard analysis ("run DCF on AAPL") -> probe -> plan -> execute -> analyze -> verify -> done
- "complex": Multi-faceted analysis ("compare AAPL vs MSFT valuation with sensitivity") -> probe -> plan -> execute -> analyze -> verify -> done (may need multiple revision loops)

ROUTING LOGIC:
- Phase "init" + no complexity set -> classify query, then route to probe (standard/complex) or plan (simple)
- Phase "probed" -> route to plan
- Phase "planned" -> route to execute
- Phase "executed" -> route to analyze (standard/complex) or done (simple, if query is answered by data alone)
- Phase "analyzed" -> route to verify (standard/complex) or done (simple)
- Phase "verified" + action "approve" -> done
- Phase "verified" + action "revise" -> look at weaknesses:
    - Missing data/components -> route to plan (create new tool sequence for gaps)
    - Poor analysis quality -> route to analyze (same data, with feedback)
    - Both -> route to plan first

REVISION HANDLING:
When routing after a failed verification, you MUST populate revision_context with:
- For plan: {"type": "missing_data", "feedback": <verifier weaknesses and missing components>}
- For analyze: {"type": "poor_analysis", "feedback": <verifier weaknesses and improvement instructions>}

GUARDRAILS (you cannot override these, the system enforces them):
- Must verify before done for standard/complex queries
- Maximum 12 global iterations
- Maximum 3 executions
- Maximum 3 analyses

NOTES (scratchpad for yourself):
- The "notes" field lets you leave a message for your future self.
- Each invocation clears your conversation history, but notes persist in state.
- Use notes to record observations, flag concerns, or track decisions across iterations.
- Example: "WACC came back unusually high (18%), may need to verify inputs" or "Revenue data only goes back 2 years, used shorter history"
- Keep notes concise (1-2 sentences). Leave null if nothing noteworthy.

RULES:
- next_action must be one of: probe, plan, execute, analyze, verify, done
- query_complexity must be set on every call (persist the same value after first classification)
- revision_context is null unless routing after a failed verification
- notes is optional -- use it when you have observations worth preserving"""

  def decide(self, state: AgentState) -> Dict[str, Any]:
    """
    Read full state, decide which spoke to route to next.
    Clears conversation history each call (stateless).

    Returns:
      Dict with keys: next_action, reasoning, query_complexity, revision_context

    Raises:
      RuntimeError: If LLM fails to produce valid output after retry.
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

    # First attempt -- structured output guarantees valid JSON
    response = self.generate_response(prompt=user_prompt, system_prompt=system_prompt)

    try:
      decision = self.parse_response(response)
      result = decision.model_dump()

      # Validate next_action
      valid_actions = {'probe', 'plan', 'execute', 'analyze', 'verify', 'done'}
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
      error_context = {
        "error": "Master Orchestrator failed to produce valid output after 2 attempts",
        "last_error": str(e),
        "raw_response": response[:500],
        "current_phase": state.get('current_phase', 'init'),
        "user_query": state.get('user_query', ''),
        "global_iteration": state.get('global_iteration', 0),
        "phase_history": state.get('phase_history', [])
      }
      raise RuntimeError(
        f"Master Orchestrator LLM parse failure. The workflow cannot continue without a valid routing decision.\n"
        f"Context: {json.dumps(error_context, indent=2, default=str)}\n"
        f"To recover: manually specify the next action or abort the workflow."
      )

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
