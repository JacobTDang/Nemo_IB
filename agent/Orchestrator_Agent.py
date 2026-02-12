from .ollama_template import OllamaModel
from pydantic import BaseModel
from typing import Dict, Optional, List, Any
import sys
import json


class ToolCall(BaseModel):
  tool: str
  arguments: Dict[str, Any]


class ExecutionPlan(BaseModel):
  task_type: str
  ticker: Optional[str] = None
  reasoning: str
  tools_sequence: List[ToolCall]


class Orchestrator_Agent(OllamaModel):
  """Creates execution plans -- which MCP tools to call and in what order."""
  response_schema = ExecutionPlan

  def __init__(self, model_name: str = 'orchestrator:latest'):
    super().__init__(model_name=model_name)

  def build_orchestrator_prompt(self,
                                  tool_list: Dict[str, Dict],
                                  previous_node: Optional[List[Dict[str, Any]]] = None,
                                  revision_feedback: Optional[Dict[str, Any]] = None,
                                  clarification_context: Optional[Dict[str, Any]] = None,
                                  gathered_variables: Optional[Dict[str, Any]] = None) -> str:
    """Build complete system prompt with reasoning framework and available tools"""
    from datetime import datetime
    current_date = datetime.now().strftime("%B %d, %Y")

    prompt = f"""You are an AI Task Orchestrator for financial analysis. Today: {current_date}

YOUR ROLE:
Given a user's request and available tools, create an execution plan by reasoning about:
1. What information is needed to answer the question?
2. Which tools can provide that information?
3. What's missing or unclear? (Use web search for current data, context, or missing pieces)
4. Do tools depend on each other? (Some tools output data needed by other tools)

REASONING FRAMEWORK:
- Historical financial data -> Use SEC filing tools (revenue, margins, etc.)
- Current market data (beta, shares, market cap, debt) -> Use get_market_data
- WACC calculation -> Use get_market_data (beta, market cap, debt) + get_tax_rate + search (risk-free rate, ERP) + calculate_wacc
- DCF valuation -> Gather all inputs first, then use calculate_dcf
- Current market context, analyst opinions -> Use search
- Company strategy/decisions unclear -> Use disclosures or search
- Tool output contains IDs/names/URLs -> Next tool might need those as input

EXAMPLES OF DYNAMIC THINKING:

"Run a DCF on AAPL":
- Need: Revenue (get_revenue_base)
- Need: Profitability (get_ebitda_margin, get_tax_rate)
- Need: CapEx & D&A (get_capex_pct_revenue, get_depreciation)
- Need: Market data (get_market_data -> beta, cash, debt, shares outstanding, interest expense)
- Need: WACC inputs (search "10-year Treasury yield {datetime.now().year}", search "equity risk premium {datetime.now().year}")
- Compute: calculate_wacc(beta, risk_free_rate, erp, cost_of_debt, tax_rate, market_cap, total_debt)
- Compute: calculate_dcf(revenue, margins, growth, wacc, cash, debt, shares)

"Is TSLA a good buy right now?":
- Need: Current valuation multiples (search "TSLA P/E ratio")
- Need: Recent financial performance (get_revenue_base, get_ebitda_margin)
- Need: Market sentiment (search "TSLA analyst ratings {datetime.now().year}")
- Need: Recent material events (extract_8k_events)
- Need: Peer comparison (comparable_company_analysis)

TWO-PART TOOLS (output -> input):
- search gives URLs -> get_urls_content needs those URLs (use placeholder "FROM_SEARCH")
- get_disclosures_names gives list -> extract_disclosure_data needs specific name

PRINCIPLES:
- Be flexible and adaptive
- Use search liberally when you need current/contextual data
- Chain tools when outputs feed inputs
- Don't rigidly follow patterns - reason about the actual need

Available tools:

"""

    # Add dynamic tool list
    for tool_name, tool_info in tool_list.items():
      schema = tool_info['parameters']
      params = []

      if 'properties' in schema:
        required = set(schema.get('required', []))
        for param_name, param_info in schema['properties'].items():
          param_type = param_info.get('type', 'any')
          if param_name in required:
            params.append(f'{param_name}: {param_type}')
          else:
            params.append(f'{param_name}?: {param_type}')

      params_str = ", ".join(params)
      prompt += f"- {tool_name}({params_str}): {tool_info['description']}\n"

    # Add probing questions from first run
    if previous_node and len(previous_node) > 0:
      prompt += "\n\nPROBING QUESTIONS TO CONSIDER:\n"
      for question in previous_node:
        if isinstance(question, dict):
          prompt += f"- {question.get('category', '')}: {question.get('question', '')} ({question.get('rationale', '')})\n"
        else:
          prompt += f"- {question}\n"

    # Add revision feedback if this is a plan revision
    if revision_feedback:
      prompt += f"""

REVISION REQUIRED:
Your previous plan was reviewed and needs improvement.

Previous Plan:
{json.dumps(revision_feedback.get('previous_plan', []), indent=2)}

Issues Identified:
{json.dumps(revision_feedback.get('issues', []), indent=2)}

Missing Critical Data:
{json.dumps(revision_feedback.get('missing_data', []), indent=2)}

Recommendations:
{json.dumps(revision_feedback.get('recommendations', []), indent=2)}

Validator Feedback:
{revision_feedback.get('reasoning', 'No additional feedback')}

Create an IMPROVED plan that addresses ALL of these issues.
"""

    # Show what data has already been gathered (so orchestrator doesn't re-fetch)
    if gathered_variables:
      prompt += "\n\nDATA ALREADY GATHERED (do NOT re-fetch these):\n"
      for key, value in gathered_variables.items():
        prompt += f"- {key}: {value}\n"
      prompt += "\nOnly plan tools for data you DON'T already have.\n"

    # Add clarification context if request was unclear
    if clarification_context:
      prompt += f"""

REQUEST NEEDS INTERPRETATION:
The user's request has ambiguities. Make reasonable assumptions and proceed.

Ambiguities: {json.dumps(clarification_context.get('ambiguities', []), indent=2)}
Questions: {json.dumps(clarification_context.get('questions', []), indent=2)}

Create a plan based on the most reasonable interpretation.
Document your assumptions in the "reasoning" field.
"""

    prompt += """

SPECIAL TOOL FORMATS:

search tool - query parameter must be a dict with keys and search strings as values:
  CORRECT: {"ticker": "MSFT", "query": {"q1": "MSFT WACC 2026", "q2": "MSFT revenue forecast"}}

get_urls_content tool - urls must be a list of strings:
  CORRECT: {"urls": ["https://example.com"]}
  Use "FROM_SEARCH" placeholder if URLs come from previous search tool

RULES:
- Tool names must match available tools exactly
- For two-part tools: use "FROM_SEARCH" for URLs
- Reasoning should explain your thinking, not just list tools
- Sequence matters - think about dependencies
- Default form_type to "10-K" unless specified"""

    return prompt

  def create_plan(self,
                   user_query: str,
                   tool_list: Dict[str, Dict],
                   previous_node: Optional[List[Dict[str, Any]]] = None,
                   revision_feedback: Optional[Dict[str, Any]] = None,
                   clarification_context: Optional[Dict[str, Any]] = None,
                   gathered_variables: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """
    Create execution plan for user query.

    Returns:
      dict: Execution plan with task_type, ticker, reasoning, tools_sequence
      or None if planning fails
    """
    if revision_feedback:
      mode = "REVISING PLAN"
    elif clarification_context:
      mode = "PLANNING (with clarifications)"
    else:
      mode = "CREATING INITIAL PLAN"

    print(f"\n{'='*60}", file=sys.stderr, flush=True)
    print(f"Orchestrator: {mode}", file=sys.stderr, flush=True)
    print(f"{'='*60}\n", file=sys.stderr, flush=True)

    system_prompt = self.build_orchestrator_prompt(
      tool_list=tool_list,
      previous_node=previous_node,
      revision_feedback=revision_feedback,
      clarification_context=clarification_context,
      gathered_variables=gathered_variables
    )

    response = self.generate_response(prompt=user_query, system_prompt=system_prompt)

    try:
      plan = self.parse_response(response)
      result = plan.model_dump()

      print(f"\nParsed plan successfully:", file=sys.stderr, flush=True)
      print(f"  Task Type: {result['task_type']}", file=sys.stderr, flush=True)
      print(f"  Ticker: {result.get('ticker', 'N/A')}", file=sys.stderr, flush=True)
      print(f"  Tools: {len(result['tools_sequence'])}", file=sys.stderr, flush=True)
      print(f"  Reasoning: {result['reasoning']}", file=sys.stderr, flush=True)

      return result
    except Exception as e:
      print(f"Plan parse failed: {e}", file=sys.stderr, flush=True)
      return None
