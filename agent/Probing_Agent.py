from .openrouter_template import OpenRouterModel
from pydantic import BaseModel
from typing import Dict, Optional, List
import sys
from datetime import datetime


class DataRequirement(BaseModel):
  data_needed: str
  tool_hint: str
  rationale: str


class AnalyticalConsideration(BaseModel):
  topic: str
  guidance: str


class ProbingResult(BaseModel):
  analysis_type: str
  ticker: Optional[str] = None
  data_requirements: List[DataRequirement]
  analytical_considerations: List[AnalyticalConsideration]
  recommended_approach: str


class Probing_Agent(OpenRouterModel):
  response_schema = ProbingResult
  MAX_OUTPUT_TOKENS = 4096  # Large JSON: 10+ data requirements + considerations + approach
  REASONING_EFFORT = None   # No reasoning -- just output structured data requirements

  def __init__(self, model_name: str = 'nvidia/nemotron-3-nano-30b-a3b:free'):
    super().__init__(model_name=model_name, api_key_env="OPENROUTER_NEMOTRON")

  def build_prompt(self, user_query: str, modeling_tools_context: str = '') -> str:
    """Build system prompt for strategic probing.

    modeling_tools_context: dynamically injected from MCP tool schemas at runtime.
    Lists modeling-phase tools + their data requirements so the probe can surface
    those requirements without hardcoding them here.
    """
    current_date = datetime.now().strftime("%B %d, %Y")

    prompt = f"""You are a Senior Research Analyst at Goldman Sachs with 15+ years of experience. Today is {current_date}.

YOUR ROLE:
Before any financial analysis begins, you identify what DATA needs to be fetched and what ANALYTICAL CONSIDERATIONS the analysis agent should reason about. These are two distinct outputs.

You must also surface data requirements for the MODELING PHASE -- the financial models that will run after data gathering. These models have specific data needs listed below.

STEP 1 — CHOOSE THE RIGHT ANALYSIS TYPE:
Read the user's query and match it to one of these types. This determines which tools to include.

- "dcf": User explicitly asks for DCF / intrinsic value / fair value
- "comps": User asks how a stock compares to peers, what multiples it trades at, relative valuation
- "comprehensive": User asks "should I buy/sell?", "advanced analysis", "deep dive", "investment thesis"
  NOTE: This is the ONLY query type that warrants DCF data gathering automatically.
- "sentiment": User asks about news, what's happening, recent events, market sentiment
- "factual": User asks for a specific data point (revenue, earnings, EBITDA, stock price)

DO NOT include DCF data requirements for "sentiment" or simple "comps" queries.

DATA GATHERING TOOLS (execution phase):
- SEC filing tools: get_revenue_base, get_ebitda_margin, get_tax_rate, get_capex_pct_revenue, get_depreciation, get_disclosures_names, extract_disclosure_data, extract_8k_events, extract_proxy_compensation, extract_governance_data, get_latest_filing
- Financial tools: get_market_data, calculate_wacc, calculate_dcf, comparable_company_analysis
- Macro tools (FRED): get_macro_snapshot, get_treasury_yields, get_credit_spreads, get_fred_series, search_fred
- Market intelligence (Finnhub): get_company_news, get_market_news, get_insider_transactions, get_insider_sentiment, get_analyst_recommendations, get_company_peers, get_basic_financials, get_earnings_surprises, get_forward_estimates, get_financial_statements, get_company_profile
- Web search (last resort): search, get_urls_content

MODELING PHASE TOOLS (run after data gathering -- Financial Modeling Agent reads the variable store):
{modeling_tools_context if modeling_tools_context else "(modeling tools context not provided)"}

IMPORTANT: When the user's query implies a modeling-phase analysis (e.g. "full valuation", "DCF", "buyout analysis", "credit profile", "capital returns"), you must include ALL data requirements for that model in your data_requirements list. Read the modeling tool descriptions above to know exactly what data each model needs.

OUTPUT 1: DATA REQUIREMENTS
Concrete, fetchable data points. Each must map to a specific tool.

BAD (too vague): "financial health" | GOOD: "EBITDA margin from SEC filing (get_ebitda_margin)"
BAD: "market conditions" | GOOD: "10Y Treasury yield for WACC risk-free rate (get_treasury_yields)"

OUTPUT 2: ANALYTICAL CONSIDERATIONS
Strategic framing for the analysis agent. NOT data fetching. 3-5 items max.

RULES:
1. Choose analysis_type FIRST
2. Generate 5-15 data requirements -- be thorough, include modeling-phase inputs when relevant
3. Generate 3-5 analytical considerations -- strategic framing only
4. Prioritize SEC and structured tools; search is last resort
5. Include calculate_wacc and calculate_dcf as requirements only for "dcf" and "comprehensive" types"""

    return prompt

  def probe(self, user_query: str, ticker: Optional[str] = None,
            modeling_tools_context: str = '') -> Dict:
    """
    Analyze the user query and output data requirements + analytical considerations.

    modeling_tools_context: dynamically injected from MCP tool schemas (analysis_workflow.py).
    Lets the probe surface modeling-phase data requirements without hardcoding them.

    Returns:
        Dict containing probing result (analysis_type, data_requirements, analytical_considerations)
    """
    self.conversatoin_history = []
    system_prompt = self.build_prompt(user_query, modeling_tools_context)

    user_prompt = f"""User Request: {user_query}"""
    if ticker:
        user_prompt += f"\nTicker: {ticker}"

    user_prompt += "\n\nIdentify data requirements and analytical considerations for this analysis."

    print(f"Probing Agent analyzing request...", file=sys.stderr, flush=True)

    response = self.generate_response(
        prompt=user_prompt,
        system_prompt=system_prompt
    )

    try:
      result = self.parse_response(response)
      parsed = result.model_dump()

      print(f"\n{'='*60}", file=sys.stderr, flush=True)
      print(f"PROBING COMPLETE", file=sys.stderr, flush=True)
      print(f"{'='*60}", file=sys.stderr, flush=True)
      print(f"Data requirements: {len(parsed.get('data_requirements', []))}", file=sys.stderr, flush=True)
      print(f"Analytical considerations: {len(parsed.get('analytical_considerations', []))}", file=sys.stderr, flush=True)
      print(f"Analysis type: {parsed.get('analysis_type', 'N/A')}", file=sys.stderr, flush=True)
      print(f"{'='*60}\n", file=sys.stderr, flush=True)

      return parsed
    except Exception as e:
      print(f"Warning: Probing parse failed: {e}", file=sys.stderr, flush=True)
      return {
          "error": f"Failed to parse probing response: {e}",
          "raw_response": response[:500]
      }


if __name__ == "__main__":
    agent = Probing_Agent(model_name='llama3.1:8b')
    user_query = "Run a DCF analysis on AAPL"
    ticker = "AAPL"

    print(f"\n{'='*80}")
    print(f"TESTING PROBING AGENT")
    print(f"{'='*80}")
    print(f"Query: {user_query}")
    print(f"Ticker: {ticker}")
    print(f"{'='*80}\n")

    result = agent.probe(user_query=user_query, ticker=ticker)

    if 'error' in result:
        print(f"\nERROR: {result['error']}")
        print(f"Raw response: {result.get('raw_response', '')[:500]}")
    else:
        print(f"\n{'='*80}")
        print(f"PROBING RESULTS")
        print(f"{'='*80}")
        print(f"Analysis Type: {result.get('analysis_type', 'N/A')}")
        print(f"Ticker: {result.get('ticker', 'N/A')}")

        print(f"\n\nDATA REQUIREMENTS:")
        print(f"{'-'*80}")
        for idx, req in enumerate(result.get('data_requirements', []), 1):
            print(f"\n{idx}. {req['data_needed']}")
            print(f"   Tool hint: {req['tool_hint']}")
            print(f"   Why: {req['rationale']}")

        print(f"\n\nANALYTICAL CONSIDERATIONS:")
        print(f"{'-'*80}")
        for idx, con in enumerate(result.get('analytical_considerations', []), 1):
            print(f"\n{idx}. [{con['topic']}]")
            print(f"   {con['guidance']}")

        print(f"\n\nRECOMMENDED APPROACH:")
        print(f"  {result.get('recommended_approach', 'N/A')}")

        print(f"\n{'='*80}\n")
