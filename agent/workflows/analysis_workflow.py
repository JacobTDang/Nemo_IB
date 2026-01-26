"""
this is going to be the main workflow for financial analysis work
"""
from langgraph.graph import StateGraph, END
from .agent_state import AgentState
from ..Financial_Analysis_Agent import Financial_Analysis_Agent
from ..Orchestrator_Agent import Orchestrator_Agent
from ..Probing_Agent import Probing_Agent
from ..Plan_Validator_Agent import Plan_Validator_Agent
from ..Verification_Agent import Verification_Agent
from ..Search_Summarizer_Agent import Search_Summarizer_Agent
from ..MCP_manager import MCPConnectionManager
from langchain_core.runnables import RunnableConfig
from typing import cast
import asyncio
import json, sys

class WorkFlow:
  def __init__(self, mcp: MCPConnectionManager):
    self.prober = Probing_Agent("llama3.1:8b")
    self.orchestrator = Orchestrator_Agent("orchestrator:latest")
    self.plan_validator = Plan_Validator_Agent("llama3.1:8b")
    self.search_summarizer = Search_Summarizer_Agent("llama3.1:8b")
    self.financial_analyst = Financial_Analysis_Agent("DeepSeek-R1-Distill-Llama-8B:latest")
    self.verification_agent = Verification_Agent("DeepSeek-R1-Distill-Llama-8B:latest")
    self.workflow = StateGraph(AgentState)
    self.mcp = mcp

    self.app = self.setup_graph()

  def probe_node(self, state: AgentState):
    # what if the user doesn't mention ticker? -> model should automatically handle that
    # read from the state
    user_query = state['user_query']
    ticker = state.get("ticker")

    result = self.prober.probe(user_query=user_query, ticker=ticker)

    # stop teh program if probing result is none
    if not result: raise RuntimeError(f"Probing result is None: {result}")

    return{
      "ticker": result.get('ticker'),
      "research_questions": result.get('probing_questions', []),
      "critical_assumptions": result.get('critical_assumptions_to_validate', "No critical assumptions"),
      "potential_data_gaps": result.get("potential_data_gaps", "No potential data gaps")
    }

  async def orchestrate_node(self, state: AgentState):
    user_query = state['user_query']

    # check if we're coming back from validation
    plan_validation = state.get('plan_validation')
    tool_list = await self.mcp.list_tools()

    if plan_validation:
      action = plan_validation.get('action', '').lower()

      if action == 'revise':
        # pass validator feedback to orchestrator for revision
        feedback = {
          "previous_plan": state.get('execution_plan'),
          "issues": plan_validation.get('issues', []),
          "missing_data": plan_validation.get('missing_critical_data', []),
          "recommendations": plan_validation.get('recommendations', []),
          "reasoning": plan_validation.get('action_reasoning', '')
        }

        plan = self.orchestrator.create_plan(
          user_query=user_query,
          tool_list=tool_list,
          revision_feedback=feedback
        )

      elif action == 'clarify':
        # handle the unclear request, use clarifying questions as context
        clarifications = {
          "ambiguities": plan_validation.get('request_clarity', {}).get('ambiguities', []),
          "questions": plan_validation.get('request_clarity', {}).get('clarifying_questions', []),
        }

        plan = self.orchestrator.create_plan(
          user_query=user_query,
          tool_list=tool_list,
          clarification_context=clarifications
        )
      else:
        # shouldn't reach here if conditional edges work correctly
        raise RuntimeError(f"Unexpected action in orchestrate_node: {action}")
    else:
      # first run, use probing questions
      probing_questions = state.get('research_questions', [])

      plan = self.orchestrator.create_plan(
        user_query=user_query,
        tool_list=tool_list,
        previous_node=probing_questions,
        clarification_context={} # there arent any
      )

    if not plan:
      raise RuntimeError(f"Plan result is None: {plan}")

    # increment revision counter to prevent infinite loops
    return_count = state.get('orchestrate_return_count', 0) + 1

    return {
      'execution_plan': plan,
      'plan_reasoning': plan['reasoning'],
      'orchestrate_return_count': return_count
    }

  async def plan_validate_node(self, state: AgentState):
    user_query = state['user_query']
    execution_plan = state['execution_plan']
    plan_reasoning = state['plan_reasoning']

    result = self.plan_validator.validate_plan(user_query=user_query, execution_plan=execution_plan, plan_reasoning=plan_reasoning)

    return{
      'plan_validation': result
    }

  async def execution_node(self, state: AgentState):
    execution_plan = state['execution_plan']
    tool_sequence = execution_plan['tools_sequence']
    results = []

    print(f"\n{'='*60}", flush=True)
    print(f"EXECUTION PHASE - Running {len(tool_sequence)} tools", flush=True)
    print(f"{'='*60}\n", flush=True)

    for idx, tool in enumerate(tool_sequence, 1):
      if tool.get('tool') and tool.get('arguments'):
        tool_name = tool['tool']
        arguments = tool['arguments']
        print(f"[Step {idx}/{len(tool_sequence)}] Executing: {tool_name}", flush=True)
        print(f"  Arguments: {json.dumps(arguments, indent=2)}", flush=True)

        tool_result = await self.mcp.call_tool(tool_name, arguments)

        # Debug: Print what the tool returned
        print(f"  Result preview: {str(tool_result)[:500]}...\n", flush=True)

        results.append({
          'tool': tool_name,
          'arguments': arguments,
          'result': tool_result,
          'success': True
        })

        # AUTO-INJECT: If this was a search, automatically scrape the URLs
        if tool_name == 'search' and tool_result.get('search_result'):
          search_results = tool_result.get('search_result', [])
          urls = [item['link'] for item in search_results[:3] if item.get('link')]  # Top 3 URLs

          if urls:
            print(f"  [Auto-inject] Scraping {len(urls)} URLs from search results...", flush=True)
            try:
              scrape_result = await self.mcp.call_tool('get_urls_content', {'urls': urls})
              print(f"  [Auto-inject] Scraped content preview: {str(scrape_result)[:300]}...\n", flush=True)

              results.append({
                'tool': 'get_urls_content (auto-injected)',
                'arguments': {'urls': urls},
                'result': scrape_result,
                'success': True
              })
            except Exception as e:
              print(f"  [Auto-inject] Failed to scrape URLs: {e}\n", flush=True)

      else:
        raise KeyError(f"Unable to find tool and arguments in step {idx}: {tool}")

    print(f"\n{'='*60}", flush=True)
    print(f"EXECUTION COMPLETE - {len(results)} tools executed", flush=True)
    print(f"{'='*60}\n", flush=True)

    return {
      'tool_output': results
    }

  def summarize_node(self, state: AgentState):
    """Summarize and filter tool outputs to reduce context bloat before analysis"""
    tool_output = state['tool_output']
    ticker = state.get('ticker', 'UNKNOWN')

    # Run the summarizer
    summarized = self.search_summarizer.summarize_tool_outputs(tool_output, ticker)

    return {
      'summarized_output': summarized
    }

  async def verification_node(self, state: AgentState):
    # need to get the final analysis output and return it. compare it against the user request?
    # what other context do I need?
    user_query = state['user_query']
    analysis_report = state['analysis_report']
    execution_plan = state['execution_plan']

    result = self.verification_agent.verify(user_query=user_query, analysis_output=analysis_report, execution_plan=execution_plan)

    return{
      "verification_result": result
    }

  async def final_analysis(self, state:AgentState):
    user_query = state['user_query']
    summarized_output = state['summarized_output']

    print(f"\n[DEBUG: SUMMARIZED OUTPUT]:", file=sys.stderr, flush=True)
    print(json.dumps(summarized_output, indent=2, default=str)[:2000], file=sys.stderr, flush=True)

    execution_plan = state['execution_plan']
    research_questions = state['research_questions']

    # increment the final_analysis count
    final_analysis_iteration = state.get('final_analysis_return_count', 0) + 1

    # Pass summarized data instead of raw tool output
    result = self.financial_analyst.analyze(
      user_query=user_query,
      execution_plan=execution_plan,
      tools_results=summarized_output,
      research_questions=research_questions
    )

    return{
      'analysis_report': result,
      "final_analysis_return_count": final_analysis_iteration
    }


  def setup_graph_orchestrator(self):

    # use a main orchestrator node, and connect it to sub agents
    


    # final compile
    self.workflow.compile()


  def setup_graph(self):
    # --- initializing node keys---
    self.workflow.add_node('probe', self.probe_node)
    self.workflow.add_node('orchestrate', self.orchestrate_node)
    self.workflow.add_node('plan_validation', self.plan_validate_node)
    self.workflow.add_node('execution', self.execution_node)
    self.workflow.add_node('summarize', self.summarize_node)
    self.workflow.add_node('final_analysis', self.final_analysis)
    self.workflow.add_node('final_verification', self.verification_node)

    # set starting point at probe node to first develop research questoins
    self.workflow.set_entry_point("probe")

    # --- connecting nodes with edges ---
    self.workflow.add_edge('probe', 'orchestrate')
    self.workflow.add_edge('orchestrate', 'plan_validation')

    def check_plan(state: AgentState):
      # iteration, from plan_validation to orchestration node, check to prevent infinite loops
      iteration_num = state.get('orchestrate_return_count', 0)
      if iteration_num >= 10:
        return "Reject"

      # check state to cheak if clear is valid
      plan_action = state['plan_validation']['action']
      if plan_action.lower() == 'approve':
        return 'Accept'
      elif plan_action.lower() == "revise":
        return 'Revise'
      elif plan_action.lower() == "reject":
        return "Reject"
      else:
        return "Clarify"

    self.workflow.add_conditional_edges('plan_validation', check_plan,
                                        {'Accept': 'execution', # if plan is clear then continue to execution node
                                        'Revise': 'orchestrate', # else ...
                                        'Clarify': 'orchestrate',
                                        'Reject': END
                                        })

    # execution → summarize → final_analysis → verification
    self.workflow.add_edge('execution', 'summarize')
    self.workflow.add_edge('summarize', 'final_analysis')
    self.workflow.add_edge('final_analysis', 'final_verification')

    def check_final_analysis(state: AgentState):
      # iteration check
      final_iteration_num = state.get('final_analysis_return_count', 0)
      if final_iteration_num >= 5:
        return 'approve'  # Force end after 5 iterations

      # Handle case where verification failed to parse JSON
      verification = state.get('verification_result', {})
      if 'error' in verification or 'action' not in verification:
        print(f"Warning: Verification parse failed, approving by default", file=sys.stderr)
        return 'approve'  # Default to approve if verification failed

      # check the action in verification result
      action = verification['action'].lower()
      if action == "approve":
        return 'approve'
      elif action == "revise":
        return 'revise'
      else:
        return "reject"

    self.workflow.add_conditional_edges(
      'final_verification', check_final_analysis,
      {
        'approve': END,
        'revise': 'final_analysis',
        'reject': END
      }
    )


    return self.workflow.compile()


if __name__ == "__main__":
  async def main():
    async with MCPConnectionManager() as mcp:
      # Create workflow
      w = WorkFlow(mcp=mcp)

      # Run the workflow
      result = await w.app.ainvoke(cast(AgentState,{
        "user_query": "Run DCF on AAPL",
        "ticker": "AAPL"
      }))

      # Print the final analysis
      print("\n" + "="*80)
      print("FINAL ANALYSIS")
      print("="*80)
      print(result.get("analysis_report", "No analysis generated"))

  asyncio.run(main())
