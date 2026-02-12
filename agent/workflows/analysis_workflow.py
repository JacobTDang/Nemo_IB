"""
Hub-and-spoke workflow for financial analysis.
Master orchestrator is the hub; probe, plan, execute, analyze, verify are spokes.
Every spoke returns to master. Master reads state and routes to the next spoke.
"""
from langgraph.graph import StateGraph, END
from .agent_state import AgentState
from ..Financial_Analysis_Agent import Financial_Analysis_Agent
from ..Orchestrator_Agent import Orchestrator_Agent
from ..Probing_Agent import Probing_Agent
from ..Verification_Agent import Verification_Agent
from ..Search_Summarizer_Agent import Search_Summarizer_Agent
from ..Master_Orchestrator import Master_Orchestrator
from ..MCP_manager import MCPConnectionManager
from ..cache import Session_Cache
from .execution_engine import run_tools
from typing import cast
import asyncio
import json, sys

# Guardrail constants
MAX_ITERATIONS = 12   # Hard cap on total master invocations
MAX_EXECUTIONS = 3    # Max times execution_node can run
MAX_ANALYSES = 3      # Max times analyze_node can run


class WorkFlow:
  def __init__(self, mcp: MCPConnectionManager):
    self.master_orchestrator = Master_Orchestrator("orchestrator:latest")
    self.prober = Probing_Agent("llama3.1:8b")
    self.orchestrator = Orchestrator_Agent("orchestrator:latest")
    self.search_summarizer = Search_Summarizer_Agent("llama3.1:8b")
    self.financial_analyst = Financial_Analysis_Agent("DeepSeek-R1-Distill-Llama-8B:latest")
    self.verification_agent = Verification_Agent("DeepSeek-R1-Distill-Llama-8B:latest")
    self.workflow = StateGraph(AgentState)
    self.mcp = mcp
    self.cache = Session_Cache()

    self.app = self.setup_graph()

  def master_node(self, state: AgentState):
    global_iteration = state.get('global_iteration', 0) + 1
    phase_history = list(state.get('phase_history', []))

    # Hard stop guardrail
    if global_iteration > MAX_ITERATIONS:
      print(f"\nWARNING: Hit MAX_ITERATIONS ({MAX_ITERATIONS}). Returning current results.", file=sys.stderr, flush=True)
      print(f"Phase history: {phase_history}", file=sys.stderr, flush=True)
      return {
        'global_iteration': global_iteration,
        'phase_history': phase_history,
        'next_action': 'done',
        'master_reasoning': f'Hard stop: exceeded {MAX_ITERATIONS} iterations. Returning best available results.'
      }

    # Execution count guardrail
    execution_count = state.get('execution_count', 0)
    if execution_count >= MAX_EXECUTIONS and state.get('current_phase') == 'planned':
      print(f"\nWARNING: Hit MAX_EXECUTIONS ({MAX_EXECUTIONS}). Proceeding to analyze.", file=sys.stderr, flush=True)
      phase_history.append(f"master->analyze (exec limit)")
      return {
        'global_iteration': global_iteration,
        'phase_history': phase_history,
        'next_action': 'analyze',
        'master_reasoning': f'Execution limit reached ({MAX_EXECUTIONS}). Proceeding with available data.',
        'query_complexity': state.get('query_complexity', 'standard')
      }

    # Analysis count guardrail
    analysis_count = state.get('analysis_count', 0)
    if analysis_count >= MAX_ANALYSES and state.get('current_phase') == 'verified':
      print(f"\nWARNING: Hit MAX_ANALYSES ({MAX_ANALYSES}). Returning best analysis.", file=sys.stderr, flush=True)
      phase_history.append(f"master->done (analysis limit)")
      return {
        'global_iteration': global_iteration,
        'phase_history': phase_history,
        'next_action': 'done',
        'master_reasoning': f'Analysis limit reached ({MAX_ANALYSES}). Returning best analysis so far.',
        'query_complexity': state.get('query_complexity', 'standard')
      }

    # Verify parse failure guardrail: verifier output was not valid JSON
    verification = state.get('verification_result', {})
    if state.get('current_phase') == 'verified' and 'error' in verification:
      parse_failures = state.get('verify_parse_failures', 0)
      if parse_failures >= 2:
        # Tried twice, verifier can't produce valid JSON -- finish with what we have
        print(f"\nWARNING: Verification parse failed {parse_failures} times. Returning analysis as-is.", file=sys.stderr, flush=True)
        phase_history.append(f"master->done (verify parse limit)")
        return {
          'global_iteration': global_iteration,
          'phase_history': phase_history,
          'next_action': 'done',
          'master_reasoning': f'Verification agent failed to produce valid JSON {parse_failures} times. Returning analysis without verification.',
          'query_complexity': state.get('query_complexity', 'standard')
        }
      else:
        # First failure -- re-analyze so verifier gets different input
        print(f"\nWARNING: Verification parse failed (attempt {parse_failures + 1}). Re-analyzing to give verifier different input.", file=sys.stderr, flush=True)
        phase_history.append(f"master->analyze (verify parse failed)")
        return {
          'global_iteration': global_iteration,
          'phase_history': phase_history,
          'next_action': 'analyze',
          'master_reasoning': f'Verification output was malformed JSON. Re-analyzing to produce different output for verification.',
          'query_complexity': state.get('query_complexity', 'standard'),
          'revision_context': {'type': 'poor_analysis', 'feedback': 'Previous analysis could not be verified due to output format issues. Produce a clearer, more structured analysis.'}
        }

    # Call the master orchestrator LLM to decide
    decision = self.master_orchestrator.decide(state)

    next_action = decision['next_action']
    query_complexity = decision.get('query_complexity', state.get('query_complexity', 'standard'))
    revision_context = decision.get('revision_context')

    # Append master notes if provided
    master_notes = list(state.get('master_notes', []))
    new_note = decision.get('notes')
    if new_note:
      master_notes.append(f"[iter {global_iteration}] {new_note}")

    # Guardrail: must execute after planning -- LLM sometimes skips this
    if state.get('current_phase') == 'planned' and next_action != 'execute':
      print(f"Guardrail: Phase is 'planned' but LLM chose '{next_action}'. Overriding to execute.", file=sys.stderr, flush=True)
      next_action = 'execute'

    # Guardrail: verified + approve = done -- LLM sometimes re-verifies despite approval
    if state.get('current_phase') == 'verified':
      verification = state.get('verification_result', {})
      if verification.get('action') == 'approve' and next_action != 'done':
        print(f"Guardrail: Verification approved but LLM chose '{next_action}'. Overriding to done.", file=sys.stderr, flush=True)
        next_action = 'done'

    # Guardrail: must verify before done for standard/complex queries
    # Exception: if verification was attempted but parse failed repeatedly, let it through
    if next_action == 'done' and query_complexity in ('standard', 'complex'):
      verification = state.get('verification_result', {})
      verified_action = verification.get('action', '')
      verify_attempted = state.get('verify_parse_failures', 0) > 0 or verified_action != ''
      if verified_action != 'approve' and not verify_attempted:
        print(f"Guardrail: Must verify before done for {query_complexity} queries. Overriding to verify.", file=sys.stderr, flush=True)
        next_action = 'verify'
        decision['reasoning'] = f"Override: {query_complexity} query requires verification before completion."

    phase_history.append(f"master->{next_action}")

    return {
      'global_iteration': global_iteration,
      'phase_history': phase_history,
      'next_action': next_action,
      'master_reasoning': decision.get('reasoning', ''),
      'query_complexity': query_complexity,
      'revision_context': revision_context,  # None unless master explicitly sets it
      'master_notes': master_notes
    }

  def probe_node(self, state: AgentState):
    user_query = state['user_query']
    ticker = state.get("ticker")

    result = self.prober.probe(user_query=user_query, ticker=ticker)

    if not result:
      raise RuntimeError(f"Probing result is None: {result}")

    return {
      "ticker": result.get('ticker'),
      "research_questions": result.get('probing_questions', []),
      "critical_assumptions": result.get('critical_assumptions_to_validate', "No critical assumptions"),
      "potential_data_gaps": result.get("potential_data_gaps", "No potential data gaps"),
      "current_phase": "probed"
    }

  async def plan_node(self, state: AgentState):
    user_query = state['user_query']
    tool_list = await self.mcp.list_tools()
    variables = state.get('variables', {})

    # Build flat variable summary (exclude namespaced keys for readability)
    gathered = {k: v for k, v in variables.items() if '.' not in k} if variables else {}

    # Check if we have revision context (feedback from verifier)
    revision_context = state.get('revision_context')

    if revision_context and revision_context.get('type') == 'missing_data':
      # Re-plan with feedback about what's missing
      feedback = {
        "previous_plan": state.get('execution_plan'),
        "issues": [revision_context.get('feedback', '')],
        "missing_data": [],
        "recommendations": [],
        "reasoning": revision_context.get('feedback', ''),
        "already_gathered": gathered
      }
      plan = self.orchestrator.create_plan(
        user_query=user_query,
        tool_list=tool_list,
        revision_feedback=feedback,
        gathered_variables=gathered
      )
    else:
      # First-pass planning, use probing questions if available
      probing_questions = state.get('research_questions', [])
      plan = self.orchestrator.create_plan(
        user_query=user_query,
        tool_list=tool_list,
        previous_node=probing_questions,
        clarification_context={}
      )

    if not plan:
      raise RuntimeError(f"Plan result is None: {plan}")

    return {
      'execution_plan': plan,
      'plan_reasoning': plan['reasoning'],
      'current_phase': 'planned',
      'revision_context': None,       # Consumed, clear it
      'verification_result': None,    # Old verification no longer relevant
      'analysis_report': None         # Old analysis is stale after replan
    }

  async def execution_node(self, state: AgentState):
    execution_plan = state['execution_plan']
    tool_sequence = list(execution_plan['tools_sequence'])
    ticker = state.get('ticker', 'UNKNOWN')

    # Merge additional_tools from revision context if present
    additional_tools = state.get('additional_tools', [])
    if additional_tools:
      print(f"  [Execute] Merging {len(additional_tools)} additional tools from revision", file=sys.stderr, flush=True)
      tool_sequence.extend(additional_tools)

    # Append to existing results instead of replacing
    existing = list(state.get('summarized_output', []) or [])
    existing_vars = dict(state.get('variables', {}) or {})
    new_results, updated_vars = await run_tools(
      tool_sequence, ticker, self.mcp, self.search_summarizer, self.cache,
      variables=existing_vars
    )
    results = existing + new_results

    execution_count = state.get('execution_count', 0) + 1

    return {
      'summarized_output': results,
      'variables': updated_vars,
      'current_phase': 'executed',
      'execution_count': execution_count
    }

  async def analyze_node(self, state: AgentState):
    user_query = state['user_query']
    summarized_output = state['summarized_output']
    execution_plan = state['execution_plan']
    research_questions = state.get('research_questions', [])

    print(f"\n[DEBUG: SUMMARIZED OUTPUT]:", file=sys.stderr, flush=True)
    print(json.dumps(summarized_output, indent=2, default=str)[:2000], file=sys.stderr, flush=True)

    # Check for revision context with verifier feedback
    revision_context = state.get('revision_context')
    revision_prefix = ""
    if revision_context and revision_context.get('type') == 'poor_analysis':
      feedback = revision_context.get('feedback', '')
      revision_prefix = f"""
                          REVISION REQUIRED - Previous analysis was reviewed and needs improvement.
                          VERIFIER FEEDBACK: {feedback}

                          Fix the issues identified above. Use the SAME data but improve your analysis quality.
                          """

    # Prepend revision feedback to user query if present
    effective_query = f"{revision_prefix}\n{user_query}" if revision_prefix else user_query

    variables = state.get('variables', {})

    result = self.financial_analyst.analyze(
      user_query=effective_query,
      execution_plan=execution_plan,
      tools_results=summarized_output,
      research_questions=research_questions,
      variables=variables
    )

    analysis_count = state.get('analysis_count', 0) + 1

    return {
      'analysis_report': result,
      'current_phase': 'analyzed',
      'analysis_count': analysis_count,
      'revision_context': None  # Consumed, clear it
    }

  async def verify_node(self, state: AgentState):
    user_query = state['user_query']
    analysis_report = state['analysis_report']
    execution_plan = state['execution_plan']

    result = self.verification_agent.verify(
      user_query=user_query,
      analysis_output=analysis_report,
      execution_plan=execution_plan
    )

    # Track consecutive parse failures so master can break the loop
    if 'error' in result:
      parse_failures = state.get('verify_parse_failures', 0) + 1
      print(f"Verify parse failure #{parse_failures}", file=sys.stderr, flush=True)
    else:
      parse_failures = 0

    return {
      "verification_result": result,
      "current_phase": "verified",
      "verify_parse_failures": parse_failures
    }

  def setup_graph(self):
    # Register nodes
    self.workflow.add_node('master', self.master_node)
    self.workflow.add_node('probe', self.probe_node)
    self.workflow.add_node('plan', self.plan_node)
    self.workflow.add_node('execute', self.execution_node)
    self.workflow.add_node('analyze', self.analyze_node)
    self.workflow.add_node('verify', self.verify_node)

    # Entry point is always master
    self.workflow.set_entry_point('master')

    # Master routes via conditional edge
    def route_from_master(state: AgentState):
      return state.get('next_action', 'done')

    self.workflow.add_conditional_edges('master', route_from_master, {
      'probe': 'probe',
      'plan': 'plan',
      'execute': 'execute',
      'analyze': 'analyze',
      'verify': 'verify',
      'done': END
    })

    # Every spoke returns to master
    for node in ['probe', 'plan', 'execute', 'analyze', 'verify']:
      self.workflow.add_edge(node, 'master')
    return self.workflow.compile()


if __name__ == "__main__":
  async def main():
    async with MCPConnectionManager() as mcp:
      # Create workflow
      w = WorkFlow(mcp=mcp)

      # Run the workflow
      result = await w.app.ainvoke(cast(AgentState, {
        "user_query": "Run DCF on AAPL",
        "ticker": "AAPL"
      }))

      # Print the final analysis
      print("\n" + "="*80)
      print("FINAL ANALYSIS")
      print("="*80)
      print(result.get("analysis_report", "No analysis generated"))

  asyncio.run(main())
