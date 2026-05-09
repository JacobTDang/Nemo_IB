from .openrouter_template import OpenRouterModel
from pydantic import BaseModel
from typing import Dict, Optional, List, Any
import sys
import json


class DataGap(BaseModel):
  description: str
  recommended_tool: str
  suggested_arguments: Dict[str, Any]   # best-guess args; use 0 for auto-resolved values
  priority: str                          # "critical" | "helpful"


class PlanVerificationResult(BaseModel):
  complete: bool
  summary: str
  gaps: List[DataGap]


_DEEPSEEK_MODEL = 'deepseek/deepseek-r1-0528:free'


class Plan_Verifier_Agent(OpenRouterModel):
  """
  Pre-analysis data completeness verifier.

  Runs after every execution pass, before analysis, to check whether
  all data required for a meaningful analysis has been gathered. Returns
  a structured list of critical and helpful gaps with concrete tool
  recommendations so the orchestrator can fill them in a targeted re-plan.

  Primary: GLM-4.5 Air (fast, low latency).
  Stream-failure fallback: GLM-4.5 Air (OpenRouterModel base class default).
  Parse-failure fallback: DeepSeek R1 (manually retried in verify()).
  """
  response_schema = PlanVerificationResult
  MAX_OUTPUT_TOKENS = 4096
  REASONING_EFFORT = "low"   # GLM Air is fast; deep thinking not needed for a checklist task
  # FALLBACK_MODEL intentionally not set: stream failures fall back to GLM (base class default).
  # parse failures fall back to DeepSeek manually in verify() below.

  def __init__(self, model_name: str = 'z-ai/glm-4.5-air:free'):
    super().__init__(model_name=model_name)

  def _build_system_prompt(self) -> str:
    from datetime import datetime
    current_date = datetime.now().strftime("%B %d, %Y")

    return f"""You are a pre-analysis data completeness auditor for an investment banking AI system.
Today: {current_date}

YOUR ROLE:
Before expensive DeepSeek analysis runs, you verify that all critical data has been gathered.
You compare what was actually fetched against what is needed to answer the user's request,
then report specific gaps with exact tool recommendations to fill them.

AVAILABLE TOOLS (37 total):

SEC FILING TOOLS (historical fundamentals from XBRL):
  get_revenue_base          - Revenue from 10-K/10-Q filings
  get_ebitda_margin         - EBITDA margin % from SEC filings
  get_capex_pct_revenue     - CapEx as % of revenue from SEC filings
  get_depreciation          - D&A as % of revenue from SEC filings
  get_tax_rate              - Effective tax rate from SEC filings
  get_latest_filing         - Metadata for the latest 10-K or 10-Q
  get_disclosures_names     - List available disclosures in a filing
  extract_disclosure_data   - Extract a specific disclosure by name
  extract_8k_events         - Material corporate events from 8-K filings
  extract_proxy_compensation - Executive compensation from DEF 14A
  extract_governance_data   - Board composition and independence data

FINANCIAL MODELING TOOLS:
  get_market_data           - Market cap, beta, debt, cash, shares, interest expense
  calculate_wacc            - Weighted average cost of capital (needs beta, rates, market data)
  calculate_dcf             - Discounted cash flow valuation (needs all fundamentals + WACC)
  comparable_company_analysis - Peer multiple analysis (EV/EBITDA, P/E, EV/Revenue, etc.)

MACRO / FRED TOOLS:
  get_macro_snapshot        - Fed rate, inflation, GDP, unemployment in one call
  get_treasury_yields       - Full Treasury yield curve (2Y, 5Y, 10Y, 30Y)
  get_credit_spreads        - Investment grade and high yield credit spreads
  get_fred_series           - Any specific FRED time series by ID
  search_fred               - Search FRED for relevant economic data series

MARKET INTELLIGENCE TOOLS (forward-looking / sentiment):
  get_company_news          - Recent news articles + LLM sentiment analysis
  get_market_news           - Broad market / sector news
  get_insider_transactions  - Recent insider buy/sell transactions
  get_insider_sentiment     - MSPR insider sentiment score (6-month signal)
  get_analyst_recommendations - Buy/hold/sell consensus + trend
  get_earnings_calendar     - Upcoming earnings dates
  get_earnings_surprises    - Historical beat/miss history
  get_forward_estimates     - Consensus forward EPS/Revenue/EBITDA estimates
  get_company_peers         - Peer ticker list for comp analysis
  get_basic_financials      - Key ratios: PE, EV/EBITDA, margins, beta
  get_financial_statements  - Standardized income statement / balance sheet
  get_company_profile       - Sector, industry, employees, description

SEARCH / SCRAPE TOOLS:
  search                    - DuckDuckGo web search (last resort; use structured tools first)
  get_urls_content          - Scrape and summarize specific URLs

---

DATA REQUIREMENTS BY ANALYSIS TYPE:

DCF valuation requires ALL of:
  revenue (get_revenue_base), ebitda_margin_percent (get_ebitda_margin),
  capex_pct_revenue (get_capex_pct_revenue), d&a_pct (get_depreciation),
  effective_tax_rate (get_tax_rate), wacc (calculate_wacc),
  totalCash (get_market_data), sharesOutstanding (get_market_data),
  financials.revenueGrowthTTMYoy (get_basic_financials) -- REQUIRED for revenue growth auto-resolution.
    If financials.revenueGrowthTTMYoy is absent from variables, revenue_growth stays as 0 and DCF fails.
    Always flag get_basic_financials as CRITICAL when DCF is needed and this key is missing.

WACC calculation requires:
  beta (get_market_data), marketCap (get_market_data), totalDebt (get_market_data),
  risk_free_rate (get_macro_snapshot or get_treasury_yields), effective_tax_rate (get_tax_rate)

Comparable company analysis requires:
  peer tickers (get_company_peers) OR pass explicit peer list

Comprehensive buy/sell analysis requires at minimum:
  market data (get_market_data), news sentiment (get_company_news),
  analyst view (get_analyst_recommendations), key ratios (get_basic_financials)

---

RULES FOR FLAGGING GAPS:

"critical": Data whose absence would make the analysis hollow or outright wrong.
  Examples: DCF asked for but wacc=0 or revenue missing; no market data at all for a valuation.

"helpful": Data that would enrich the analysis but its absence is acceptable.
  Examples: Macro context missing for a pure equity story; insider sentiment not fetched for a DCF.

IMPORTANT RULES:
- If WACC was calculated but showed 0%, that is a critical gap -- the inputs were missing.
- If DCF was planned but not in summarized_output, that is a critical gap.
- If DCF ran but enterprise_value or equity_value is not in variables, inputs were likely 0.
- If DCF ran and pv_terminal_value is 0.0 while terminal_value_perpetuity is positive,
  the DCF computed a broken terminal value (terminal_multiple=0 bug). Flag as critical: re-run calculate_dcf.
  Terminal value should be the largest component of enterprise value (typically 60-80% of EV).
  A DCF with pv_terminal_value=0 is NOT a valid valuation -- it must be re-run.
- If financials.revenueGrowthTTMYoy is absent from variables and DCF is needed,
  flag get_basic_financials as critical (without it, revenue_growth stays 0 and DCF fails).
- If a tool errored with "ACCESS DENIED (HTTP 403)", do NOT flag it as critical or recommend retrying.
  403 means the API key lacks access to that endpoint (free tier restriction). Retrying is pointless.
  Treat the data as permanently unavailable and note it as a limitation, not a gap to fill.
- If a tool errored with a transient error (connection timeout, rate limit, parse error), flag it as
  critical only if its data is genuinely needed and another run would likely succeed.
- If a tool ran and errored with a non-403 error, its outputs are missing -- flag if those outputs are critical.
- Mark complete=True if: all critical dependencies are present OR the request is qualitative enough
  that missing quantitative data is acceptable (e.g. "what is management saying about X?").
- Do NOT flag helpful tools as critical. Only flag what makes the analysis fundamentally broken.
- Do NOT flag data that no tool can provide (e.g. private company data, future events).
- In suggested_arguments: use 0 for numeric args that the execution engine auto-resolves from
  the variable store (e.g. WACC inputs are auto-resolved; DCF inputs are auto-resolved).

OUTPUT FORMAT:
Respond with ONLY valid JSON matching this exact schema. No text before or after.
{{
  "complete": true | false,
  "summary": "one-sentence explanation of the completeness verdict",
  "gaps": [
    {{
      "description": "what is missing and why it matters",
      "recommended_tool": "exact tool name",
      "suggested_arguments": {{"ticker": "AAPL", "form_type": "10-K"}},
      "priority": "critical" | "helpful"
    }}
  ]
}}
gaps should be an empty list [] when complete=true.
Do not add or omit any fields from the schema."""

  def _build_user_prompt(self,
                          user_query: str,
                          variables: Dict[str, Any],
                          summarized_output: List[Dict[str, Any]],
                          execution_plan: Dict[str, Any],
                          ticker: str,
                          execution_count: int) -> str:
    # Show what tools were planned
    planned_tools = []
    if execution_plan and 'tools_sequence' in execution_plan:
      for tool in execution_plan['tools_sequence']:
        planned_tools.append(tool.get('tool', ''))

    # Show what tools actually ran (from summarized_output)
    ran_tools = []
    for item in summarized_output:
      tool_name = item.get('tool', item.get('type', ''))
      if tool_name:
        ran_tools.append(tool_name)

    # Show gathered variables (exclude namespaced keys for readability)
    flat_vars = {k: v for k, v in variables.items() if '.' not in k}

    prompt = f"""USER REQUEST: {user_query}
TICKER: {ticker or 'N/A'}
EXECUTION PASS: {execution_count}

WHAT WAS PLANNED (tools in execution plan):
{json.dumps(planned_tools, indent=2)}

WHAT ACTUALLY RAN (tools in results):
{json.dumps(ran_tools, indent=2)}

VARIABLES GATHERED (flat key-value store):
"""
    if flat_vars:
      for k, v in sorted(flat_vars.items()):
        # Truncate long values
        val_str = str(v)
        if len(val_str) > 120:
          val_str = val_str[:117] + '...'
        prompt += f"  {k}: {val_str}\n"
    else:
      prompt += "  (none)\n"

    # Flag any tools that ran with errors -- split by error type
    access_denied_tools = []   # 403 Forbidden: API tier restriction, cannot retry
    transient_error_tools = [] # Other errors: may be transient, could retry
    for item in summarized_output:
      data = item.get('data', {})
      if isinstance(data, dict) and data.get('error'):
        tool_label = item.get('tool', '?')
        err_msg = str(data['error'])
        if '403' in err_msg or 'Forbidden' in err_msg or 'forbidden' in err_msg:
          access_denied_tools.append(f"{tool_label}: {err_msg[:120]}")
        else:
          transient_error_tools.append(f"{tool_label}: {err_msg[:120]}")

    if access_denied_tools:
      prompt += "\nTOOLS WITH ACCESS DENIED (HTTP 403 - API tier restriction, cannot retry):\n"
      for e in access_denied_tools:
        prompt += f"  {e}\n"

    if transient_error_tools:
      prompt += "\nTOOLS THAT ERRORED (may be transient, could retry):\n"
      for e in transient_error_tools:
        prompt += f"  {e}\n"

    prompt += f"""
Now audit: Is all critical data present to give {user_query!r} a complete, defensible answer?
List any critical gaps with exact tool recommendations. If data is sufficient, set complete=true."""

    return prompt

  def verify(self,
              user_query: str,
              variables: Dict[str, Any],
              summarized_output: List[Dict[str, Any]],
              execution_plan: Dict[str, Any],
              ticker: str = '',
              execution_count: int = 0) -> PlanVerificationResult:
    """
    Check data completeness before analysis phase.

    Returns PlanVerificationResult with complete=True if data is sufficient,
    or complete=False with a list of critical gaps to fill.

    Fails open: on parse failure returns complete=True to avoid blocking
    analysis indefinitely on a transient LLM error.
    """
    self.conversatoin_history = []  # stateless per call

    system_prompt = self._build_system_prompt()
    user_prompt = self._build_user_prompt(
      user_query, variables, summarized_output, execution_plan, ticker, execution_count
    )

    print(f"\n[Plan Verifier] Checking data completeness (pass {execution_count})...", file=sys.stderr, flush=True)

    response = self.generate_response(prompt=user_prompt, system_prompt=system_prompt)

    try:
      result = self.parse_response(response)
      print(f"[Plan Verifier] Complete: {result.complete} | Gaps: {len(result.gaps)}", file=sys.stderr, flush=True)
      for gap in result.gaps:
        print(f"  [{gap.priority.upper()}] {gap.description} -> {gap.recommended_tool}", file=sys.stderr, flush=True)
      return result
    except Exception as e_primary:
      print(f"[Plan Verifier] GLM parse failed: {e_primary}. Retrying with DeepSeek R1...", file=sys.stderr, flush=True)

    # Fallback: retry with DeepSeek R1 (slower but produces cleaner structured JSON)
    self.conversatoin_history = []
    _original_model = self.model_name
    self.model_name = _DEEPSEEK_MODEL
    try:
      response = self.generate_response(prompt=user_prompt, system_prompt=system_prompt)
      result = self.parse_response(response)
      print(f"[Plan Verifier] DeepSeek fallback OK. Complete: {result.complete} | Gaps: {len(result.gaps)}", file=sys.stderr, flush=True)
      for gap in result.gaps:
        print(f"  [{gap.priority.upper()}] {gap.description} -> {gap.recommended_tool}", file=sys.stderr, flush=True)
      return result
    except Exception as e_fallback:
      print(f"[Plan Verifier] DeepSeek fallback also failed: {e_fallback}. Failing open.", file=sys.stderr, flush=True)
      return PlanVerificationResult(
        complete=True,
        summary=f"Plan verification parse failed after both models ({e_fallback}). Proceeding with available data.",
        gaps=[]
      )
    finally:
      self.model_name = _original_model


if __name__ == "__main__":
  pass
