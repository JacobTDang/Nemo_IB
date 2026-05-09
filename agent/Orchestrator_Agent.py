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
  MAX_OUTPUT_TOKENS = 8192  # Large prompt (37 tools) + reasoning overhead needs room
  REASONING_EFFORT = "high"  # Enabled: Nemotron needs reasoning to select tools correctly

  def __init__(self, model_name: str = 'nvidia/nemotron-3-nano-30b-a3b:free'):
    super().__init__(model_name=model_name, api_key_env="OPENROUTER_NEMOTRON")

  # Tool categories for grouping in the prompt
  _SEC_TOOLS = {
    'get_revenue_base', 'get_ebitda_margin', 'get_capex_pct_revenue', 'get_depreciation',
    'get_tax_rate', 'get_latest_filing', 'get_disclosures_names', 'extract_disclosure_data',
    'extract_8k_events', 'extract_proxy_compensation', 'extract_governance_data',
  }
  _FINANCIAL_TOOLS = {
    'get_market_data', 'calculate_wacc', 'calculate_dcf', 'comparable_company_analysis',
  }
  _MACRO_TOOLS = {
    'get_macro_snapshot', 'get_treasury_yields', 'get_credit_spreads',
    'get_fred_series', 'search_fred',
  }
  _MARKET_INTEL_TOOLS = {
    'get_company_news', 'get_market_news', 'get_insider_transactions', 'get_insider_sentiment',
    'get_analyst_recommendations', 'get_earnings_calendar', 'get_earnings_surprises',
    'get_forward_estimates', 'get_company_peers', 'get_basic_financials',
    'get_financial_statements', 'get_company_profile',
  }
  _SEARCH_TOOLS = {'search', 'get_urls_content'}

  def _format_tool_section(self, tool_list: Dict[str, Dict], names: set) -> str:
    """Format a subset of tools from the tool_list into a numbered parameter string."""
    lines = []
    for tool_name, tool_info in tool_list.items():
      if tool_name not in names:
        continue
      schema = tool_info['parameters']
      params = []
      if 'properties' in schema:
        required = set(schema.get('required', []))
        for param_name, param_info in schema['properties'].items():
          param_type = param_info.get('type', 'any')
          marker = '' if param_name in required else '?'
          params.append(f'{param_name}{marker}: {param_type}')
      params_str = ', '.join(params)
      lines.append(f"  - {tool_name}({params_str})\n    {tool_info['description']}\n")
    return '\n'.join(lines)

  def build_orchestrator_prompt(self,
                                  tool_list: Dict[str, Dict],
                                  data_requirements: Optional[List[Dict[str, str]]] = None,
                                  revision_feedback: Optional[Dict[str, Any]] = None,
                                  gathered_variables: Optional[Dict[str, Any]] = None,
                                  execution_count: int = 0) -> str:
    """Build complete system prompt with reasoning framework and available tools"""
    from datetime import datetime
    current_date = datetime.now().strftime("%B %d, %Y")

    sec_tools_str      = self._format_tool_section(tool_list, self._SEC_TOOLS)
    financial_tools_str = self._format_tool_section(tool_list, self._FINANCIAL_TOOLS)
    macro_tools_str    = self._format_tool_section(tool_list, self._MACRO_TOOLS)
    intel_tools_str    = self._format_tool_section(tool_list, self._MARKET_INTEL_TOOLS)
    search_tools_str   = self._format_tool_section(tool_list, self._SEARCH_TOOLS)

    pass_label = "FIRST PASS (no data gathered yet)" if execution_count == 0 else f"REVISION PASS {execution_count}"

    prompt = f"""You are an AI Task Orchestrator for financial analysis. Today: {current_date}
Execution pass: {pass_label}

YOUR ROLE:
Given a user's financial analysis request, create a comprehensive execution plan by systematically
evaluating EVERY available tool and deciding whether it is needed to fully answer the request.
Do not pick only the obvious tools -- think like a senior investment banker who leaves no stone unturned.

HOW EXECUTION WORKS:
The execution engine automatically runs tools in two waves:
1. PARALLEL wave first: all data-fetch tools (SEC, market data, macro, market intel, search) run concurrently.
2. SEQUENTIAL wave after: calculation tools (calculate_wacc, calculate_dcf) run after the parallel wave
   completes, so they see a fully-populated variable store.

This means you CAN include calculate_wacc and calculate_dcf in the same plan as the data tools that feed them.
Use 0 as a placeholder for arguments that will be auto-resolved from earlier tools in the same plan.
The engine resolves: ebitda_margin from get_ebitda_margin, beta from get_market_data, risk_free_rate from get_macro_snapshot, etc.

{'REVISION GUIDANCE: Data already gathered is shown below. Only plan tools for gaps and any calculations whose inputs are now confirmed.' if execution_count > 0 else ''}

PLANNING APPROACH -- follow these steps in order:

Step 1: Understand the request
  - What is the user asking for? (valuation, sentiment, macro context, news, etc.)
  - Which company/ticker?
  - What depth of analysis is required?

Step 2: Walk through EVERY tool category and ask "Is this needed?"
  For each tool listed below, explicitly ask yourself:
    "Would including this tool meaningfully improve the analysis or answer a part of the request?"
  If yes -- include it. If no -- skip it and move on.
  A comprehensive IB analysis typically uses 8-15 tools across multiple categories.

Step 3: Order by dependencies
  - Fetch data before calculations that depend on it
  - SEC / Financial / Macro / Market Intel before DCF/WACC
  - Search last (only for gaps no structured tool covers)

TOOL DEPENDENCIES:
- get_market_data provides: beta, market cap, debt, cash, shares, interest expense
- get_macro_snapshot / get_treasury_yields provides: risk_free_rate (auto-resolved into calculate_wacc)
- calculate_wacc needs: beta, risk_free_rate, equity_risk_premium, cost_of_debt, tax_rate, market_cap, total_debt
- calculate_dcf needs: revenue, ebitda_margin, capex, depreciation, tax_rate, wacc, cash, debt, shares
- The execution engine auto-resolves values from earlier tools. Use 0 as a placeholder for auto-resolved args.

calculate_dcf SPECIAL ARGUMENTS (auto-handled when set to 0):
- revenue_growth: MUST be a list e.g. [0.12, 0.10, 0.08, 0.07, 0.06]. Set to [0,0,0,0,0] or 0 to
  auto-build from get_basic_financials (revenueGrowthTTMYoy field). Always run get_basic_financials
  BEFORE calculate_dcf. Without it, revenue_growth stays 0 and DCF fails schema validation.
- terminal_growth: set to 0 to auto-default to 0.025 (2.5% GDP-match perpetuity growth).
- terminal_multiple: set to 0 to use perpetuity growth method only (no exit multiple).
  Set to a real EV/EBITDA (e.g. 15.0 for large-cap tech) to use min(perpetuity, exit multiple).

TWO-PART TOOLS:
- search gives URLs -> get_urls_content scrapes them (use "FROM_SEARCH" as placeholder for urls)
- get_disclosures_names lists disclosures -> extract_disclosure_data extracts a specific one

DO NOT use search for data that structured tools provide (news, rates, financials, macros).

---
CATEGORY 1: SEC FILING TOOLS -- historical financial data from XBRL filings
Ask: Does this request need historical revenue, margins, capex, depreciation, tax rate, or specific SEC disclosures?

{sec_tools_str}
---
CATEGORY 2: FINANCIAL MODELING TOOLS -- market data and quantitative calculations
Ask: Does this request need market cap / beta / debt? Does it involve a DCF, WACC, or peer comp?

{financial_tools_str}
---
CATEGORY 3: MACRO / FRED TOOLS -- interest rates, inflation, GDP, credit spreads, yield curve
Ask: Does this request need macro context? Does it involve valuation (needs risk-free rate), credit risk, or economic backdrop?
Note: get_macro_snapshot covers rates + inflation + GDP in one call. Add get_treasury_yields for full curve detail.
      Add get_credit_spreads if the request involves credit risk, spreads, or financing costs.

{macro_tools_str}
---
CATEGORY 4: MARKET INTELLIGENCE TOOLS -- forward-looking signals, sentiment, analyst views
Ask for EACH of the following:
  - News/sentiment needed? -> get_company_news (always include for any "should I buy" or "current situation" query)
  - Insider buying/selling signal? -> get_insider_transactions + get_insider_sentiment (MSPR)
  - Analyst ratings / upgrades / downgrades? -> get_analyst_recommendations
  - Earnings beat/miss history? -> get_earnings_surprises
  - Forward EPS / Revenue / EBITDA consensus? -> get_forward_estimates
  - Upcoming earnings catalyst? -> get_earnings_calendar
  - Peer tickers for comp? -> get_company_peers
  - Key ratios snapshot (PE, EV/EBITDA, margins, beta)? -> get_basic_financials
  - Historical standardized financials? -> get_financial_statements
  - Company profile / sector / employees? -> get_company_profile

{intel_tools_str}
---
CATEGORY 5: SEARCH / SCRAPE -- unstructured web research
Ask: Is there qualitative information (management commentary, specific reports, strategic context) that NONE of the above tools can provide?
Only include if genuinely needed -- structured tools are always preferred.

{search_tools_str}
---
"""

    # Add data requirements from probing agent
    if data_requirements and len(data_requirements) > 0:
      prompt += "\nDATA REQUIREMENTS (from probing agent -- make sure your plan covers all of these):\n"
      for req in data_requirements:
        if isinstance(req, dict):
          prompt += f"  - {req.get('data_needed', '')} [{req.get('tool_hint', '')}]: {req.get('rationale', '')}\n"
        else:
          prompt += f"  - {req}\n"
      prompt += "\n"

    # Add revision feedback if this is a plan revision
    if revision_feedback:
      prompt += f"""
REVISION REQUIRED:
Your previous plan had gaps. Create an improved plan that addresses ALL issues below.
Do NOT re-fetch data that is already gathered (see ALREADY GATHERED section below).

Issues Identified:
{json.dumps(revision_feedback.get('issues', []), indent=2)}

Missing Critical Data:
{json.dumps(revision_feedback.get('missing_data', []), indent=2)}

Validator Feedback:
{revision_feedback.get('reasoning', 'No additional feedback')}

"""

    # Show what data has already been gathered (so orchestrator doesn't re-fetch)
    if gathered_variables:
      prompt += "DATA ALREADY GATHERED (do NOT re-fetch these -- only plan tools for what's missing):\n"
      for key, value in gathered_variables.items():
        prompt += f"  {key}: {value}\n"
      prompt += "\n"

    prompt += """SPECIAL TOOL FORMATS:

search -- query must be a dict of labeled search strings:
  {"ticker": "AAPL", "query": {"q1": "AAPL revenue growth 2026", "q2": "AAPL competitive moat"}}

get_urls_content -- urls must be a list:
  {"urls": ["FROM_SEARCH"]}

RULES:
- Tool names must match the available tools exactly (no hallucinated tool names)
- Use 0 as placeholder for numeric args that will be auto-resolved from earlier tools
- reasoning must explain WHY each category was included or skipped, not just list tool names
- Default form_type to "10-K" for SEC tools unless the query specifies otherwise"""

    return prompt

  def create_plan(self,
                   user_query: str,
                   tool_list: Dict[str, Dict],
                   data_requirements: Optional[List[Dict[str, str]]] = None,
                   revision_feedback: Optional[Dict[str, Any]] = None,
                   gathered_variables: Optional[Dict[str, Any]] = None,
                   execution_count: int = 0) -> Optional[Dict[str, Any]]:
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
      gathered_variables=gathered_variables,
      execution_count=execution_count
    )

    response = self.generate_response(prompt=user_query, system_prompt=system_prompt)

    try:
      plan = self.parse_response(response)
      result = plan.model_dump()

      print(f"\nParsed plan successfully:", file=sys.stderr, flush=True)
      print(f"  Task Type: {result['task_type']}", file=sys.stderr, flush=True)
      print(f"  Ticker: {result.get('ticker', 'N/A')}", file=sys.stderr, flush=True)
      print(f"  Tools: {len(result['tools_sequence'])}", file=sys.stderr, flush=True)
      print(f"  Reasoning: {result['reasoning'][:200]}", file=sys.stderr, flush=True)

      return result
    except Exception as e:
      print(f"Plan parse failed: {e}", file=sys.stderr, flush=True)
      return None
