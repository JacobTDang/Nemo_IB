from .openrouter_template import OpenRouterModel
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


class Orchestrator_Agent(OpenRouterModel):
  """Creates execution plans -- which MCP tools to call and in what order."""
  response_schema = ExecutionPlan
  MAX_OUTPUT_TOKENS = 8192  # Large prompt (28 tools) + reasoning overhead needs room
  REASONING_EFFORT = None   # No reasoning -- just output the structured plan

  def __init__(self, model_name: str = 'nvidia/nemotron-3-nano-30b-a3b:free'):
    super().__init__(model_name=model_name, api_key_env="OPENROUTER_NEMOTRON")

  def build_orchestrator_prompt(self,
                                  tool_list: Dict[str, Dict],
                                  data_requirements: Optional[List[Dict[str, str]]] = None,
                                  revision_feedback: Optional[Dict[str, Any]] = None,
                                  gathered_variables: Optional[Dict[str, Any]] = None) -> str:
    """Build complete system prompt with reasoning framework and available tools"""
    from datetime import datetime
    current_date = datetime.now().strftime("%B %d, %Y")

    prompt = f"""You are an AI Task Orchestrator for financial analysis. Today: {current_date}

YOUR ROLE:
Given a user's request and available tools, create an execution plan. You receive DATA REQUIREMENTS from the probing agent telling you exactly what data to fetch and which tool category to use.

TOOL PRIORITY (STRICT ORDER):
1. SEC filing tools FIRST -- for historical financial data (revenue, margins, tax, capex, depreciation, disclosures, 8-K)
2. Financial tools SECOND -- for market data (beta, market cap, debt, cash) and calculations (WACC, DCF, comps)
3. Macro tools THIRD -- for interest rates, inflation, GDP, unemployment, yield curve (FRED data)
   - get_macro_snapshot for broad macro context (includes risk-free rate, inflation, employment, GDP)
   - get_treasury_yields for full yield curve (3M-30Y) with spreads and curve shape
   - get_fred_series for any specific FRED series by ID
   - search_fred to discover FRED series IDs
4. Market intelligence tools FOURTH -- for news sentiment, insider activity, analyst ratings, earnings calendar, peer comparison, key financials
   - get_company_news / get_market_news for news articles (NOT search)
   - get_insider_transactions for insider buying/selling
   - get_analyst_recommendations for consensus ratings
   - get_basic_financials for key financial metrics
   - get_company_peers for peer comparison
   - get_earnings_calendar for upcoming earnings
5. Search LAST -- ONLY when the above tools cannot provide the data (e.g. qualitative research, specific analyst reports)

DO NOT plan search tools for data that SEC, financial, macro, or market intelligence tools can provide.
DO NOT use search for news -- use get_company_news or get_market_news instead.
DO NOT use search for risk-free rate, Treasury yields, inflation, or GDP -- use macro tools instead.

TOOL DEPENDENCIES:
- get_market_data provides: beta, market cap, debt, cash, shares, interest expense
- get_macro_snapshot / get_treasury_yields provides: risk_free_rate (auto-resolved into calculate_wacc)
- calculate_wacc needs: beta, risk_free_rate, equity_risk_premium, cost_of_debt, tax_rate, market_cap, total_debt
- calculate_dcf needs: revenue, ebitda_margin, capex, depreciation, tax_rate, wacc, cash, debt, shares
- The execution engine auto-resolves values between tools. Use 0 as placeholder for values that will come from earlier tools.

TWO-PART TOOLS (output -> input):
- search gives URLs -> get_urls_content needs those URLs (use placeholder "FROM_SEARCH")
- get_disclosures_names gives list -> extract_disclosure_data needs specific name

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

    # Add data requirements from probing agent
    if data_requirements and len(data_requirements) > 0:
      prompt += "\n\nDATA REQUIREMENTS (from probing agent -- plan tools to fetch these):\n"
      for req in data_requirements:
        if isinstance(req, dict):
          prompt += f"- {req.get('data_needed', '')} -> {req.get('tool_hint', '')} ({req.get('rationale', '')})\n"
        else:
          prompt += f"- {req}\n"

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
                   data_requirements: Optional[List[Dict[str, str]]] = None,
                   revision_feedback: Optional[Dict[str, Any]] = None,
                   gathered_variables: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """
    Create execution plan for user query.

    Returns:
      dict: Execution plan with task_type, ticker, reasoning, tools_sequence
      or None if planning fails
    """
    self.conversatoin_history = []
    mode = "REVISING PLAN" if revision_feedback else "CREATING INITIAL PLAN"

    print(f"\n{'='*60}", file=sys.stderr, flush=True)
    print(f"Orchestrator: {mode}", file=sys.stderr, flush=True)
    print(f"{'='*60}\n", file=sys.stderr, flush=True)

    system_prompt = self.build_orchestrator_prompt(
      tool_list=tool_list,
      data_requirements=data_requirements,
      revision_feedback=revision_feedback,
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
