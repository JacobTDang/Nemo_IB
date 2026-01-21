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
from ..MCP_manager import MCPConnectionManager
from langchain_core.runnables import RunnableConfig
from typing import cast
import asyncio
import json

class WorkFlow:
  def __init__(self, mcp: MCPConnectionManager):
    self.prober = Probing_Agent("llama3.1:8b")
    self.orchestrator = Orchestrator_Agent("orchestrator:latest")
    self.plan_validator = Plan_Validator_Agent("llama3.1:8b")
    self.financial_analyst = Financial_Analysis_Agent("DeepSeek-R1-Distill-Llama-8B:latest")
    self.verification_agent = Verification_Agent("")
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
    return_count = state.get('return_count', 0) + 1

    return {
      'execution_plan': plan,
      'plan_reasoning': plan['reasoning'],
      'return_count': return_count
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
    tool_output = state['tool_output']
    execution_plan = state['execution_plan']
    research_questions = state['research_questions']

    result = self.financial_analyst.analyze(user_query=user_query, execution_plan=execution_plan, tools_results=tool_output, research_questions=research_questions)

    return{
      'analysis_report': result
    }

  def setup_graph(self):
    # --- initializing node keys---
    self.workflow.add_node('probe', self.probe_node)
    self.workflow.add_node('orchestrate', self.orchestrate_node)
    self.workflow.add_node('plan_validation', self.plan_validate_node)
    self.workflow.add_node('execution', self.execution_node)  # Fixed method name
    self.workflow.add_node('final_analysis', self.final_analysis)

    # set starting point at probe node to first develop research questoins
    self.workflow.set_entry_point("probe")

    # --- connecting nodes with edges ---
    self.workflow.add_edge('probe', 'orchestrate')
    self.workflow.add_edge('orchestrate', 'plan_validation')

    def check_plan(state: AgentState):
      # iteration, from plan_validation to orchestration node, check to prevent infinite loops
      iteration_num = state.get('return_count', 0)
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

    self.workflow.add_edge('execution', 'final_analysis')
    self.workflow.add_edge('final_analysis', END)

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
