"""
this will be a shared state schema with type definitions, meant to be
reusable across workflows
"""
from typing import TypedDict, List, Dict, Any
class AgentState(TypedDict):
  user_query: str
  ticker: str

  research_questions: List
  critical_assumptions: str
  potential_data_gaps: str

  action: List[str]

  execution_plan: Dict[str, Any]
  plan_reasoning: str


  tools_results : List[Dict[str, Any]]

  is_clear: bool | str
  plan_validation: Dict[str, Any]

  tool_output: List[Any]
  data_sufficient: bool

  missing_data: List
  return_count: int

  analysis_report: str

  token_count: int
