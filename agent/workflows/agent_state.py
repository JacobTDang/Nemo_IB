"""
Shared state schema for the hub-and-spoke workflow architecture.
The master orchestrator reads and writes to this state to route between spokes.
"""
from typing import TypedDict, List, Dict, Any, Optional


class AgentState(TypedDict):
  # User input
  user_query: str
  ticker: str

  # Probing phase outputs
  data_requirements: List[Dict[str, str]]       # Concrete data to fetch (for orchestrator)
  analytical_considerations: List[Dict[str, str]] # Strategic guidance (for analysis agent)
  recommended_approach: str

  # Planning phase outputs
  execution_plan: Dict[str, Any]
  plan_reasoning: str

  # Execution phase outputs
  summarized_output: List[Dict[str, Any]]
  variables: Dict[str, Any]               # Shared variable store - flat k/v from tool results

  # Modeling phase outputs (Financial_Modeling_Agent results)
  model_outputs: Optional[Dict[str, Any]]   # scenario_dcf, credit_profile, capital_returns, lbo

  # Analysis phase outputs
  analysis_report: str

  # Pre-analysis plan verification (set by plan_verify_node after each execution)
  plan_verification: Dict[str, Any]  # {"complete": bool, "summary": str, "gaps": [...]}

  # Master orchestrator routing
  current_phase: str              # "init"|"probed"|"planned"|"executed"|"analyzed"
  next_action: str                # "probe"|"plan"|"execute"|"analyze"|"done"
  master_reasoning: str           # Why the master chose this action
  query_complexity: str           # "simple"|"standard"|"complex" - set once on first pass

  # Global iteration tracking
  global_iteration: int           # Total master invocations (hard cap at 12)
  phase_history: List[str]        # Audit trail: ["master->probe", "master->plan", ...]

  # Feedback forwarding
  revision_context: Dict[str, Any]      # Feedback from verifier to pass to next agent
  additional_tools: List[Dict[str, Any]] # Extra tools for re-execution

  # Execution quality tracking
  execution_stats: Dict[str, int]  # {"errors": N, "successes": N, "total": N}

  # Per-phase iteration counters
  execution_count: int
  analysis_count: int

  # Master orchestrator scratchpad -- persists across stateless invocations
  master_notes: List[str]
