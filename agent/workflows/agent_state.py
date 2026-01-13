"""
this will be a shared state schema with type definitions, meant to be
reusable across workflows
"""
from typing import TypedDict, List, Dict, Any
class AgentState(TypedDict):
  user_query: str
  ticker: str

  research_questions: List[str]

  execution_plan: Dict[str, Any]

  tools_results : List[Dict[str, Any]]

  data_sufficent: bool
  missing_data: List[str]
  return_count: int

  final_analysis: str

  token_count: int
