"""
this is going to be the main workflow for financial analysis work
"""
from langgraph.graph import StateGraph, END
from agent_state import AgentState
from ..Financial_Analysis_Agent import Financial_Analysis_Agent
from ..Orchestrator_Agent import Orchestrator_Agent
from ..Probing_Agent import Probing_Agent
from ..Plan_Validator_Agent import Plan_Validator_Agent
from ..Verification_Agent import Verification_Agent
from ..MCP_manager import MCPConnectionManager
from langchain_core.runnables import RunnableConfig
from typing import Any, Dict, Optional
import asyncio

class WorkFlow:
  def __init__(self, mcp: MCPConnectionManager):
    self.prober = Probing_Agent("llama3.1:8b")
    self.orchestrator = Orchestrator_Agent("orchestrator:latest")
    self.plan_validator = Plan_Validator_Agent("llama3.1:8b")
    self.financial_analyst = Financial_Analysis_Agent("DeepSeek-R1-Distill-Llama-8B:latest")
    self.final_approval = Verification_Agent("")

    self.workflow = StateGraph(AgentState)
    self.mcp = mcp

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

    actions = ['approve', 'revise', 'reject', 'clarify']
    been_reviewed = True if state.get('action', None) in actions else False
    # we only want to bring use probing questions prompt once, should not if plan needs revising or reject
    probing_questions = state.get('research_questions', []) if not been_reviewed else []

    # need to check the state
    tool_list = await self.mcp.list_tools()
    plan = self.orchestrator.create_plan(user_query=user_query, tool_list=tool_list, previous_node=probing_questions)

    # stop the program if plan result is none
    if not plan: raise RuntimeError(f"Plan results is None: {plan}")

    return{
      'execution_plan': plan['tools_sequence'],
      'plan_reasoning': plan['reasoning']
    }

  async def plan_validate_node(self, state: AgentState):
    user_query = state['user_query']
    execution_plan = state['execution_plan']
    plan_reasoning = state['plan_reasoning']

    result = self.plan_validator.validate_plan(user_query=user_query, execution_plan=execution_plan, plan_reasoning=plan_reasoning)

    return{
      'is_clear': result['request_clarity']['is_clear'],
      'plan_validation': result
    }

  async def excution_node(self, state: AgentState):
    execution_plan = state['execution_plan']
    tool_sequence = execution_plan['tool_sequence']

    for tool in tool_sequence:
      if tool['tool'] and tool['arguments']:
        tool_name = tool['tool']
        arguments = tool['arguments']
        await self.mcp.call_tool(tool_name, arguments)
      else:
        raise KeyError("Unable to find tool and argumetns")

if __name__ == "__main__":
  async def main():
    async with MCPConnectionManager() as mcp:

      w = WorkFlow(mcp=mcp)

  asyncio.run(main())
