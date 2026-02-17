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
  REASONING_EFFORT = None  # No reasoning -- just output structured data requirements

  def __init__(self, model_name: str = 'nvidia/nemotron-3-nano-30b-a3b:free'):
    super().__init__(model_name=model_name, api_key_env="OPENROUTER_NEMOTRON")

  def build_prompt(self, user_query: str) -> str:
    """build specialized system prompt for strategic probing"""
    current_date = datetime.now().strftime("%B %d, %Y")

    prompt = f"""You are a Senior Research Analyst at Goldman Sachs with 15+ years of experience. Today is {current_date}.

YOUR ROLE:
Before any financial analysis begins, you identify what DATA needs to be fetched and what ANALYTICAL CONSIDERATIONS the analysis agent should reason about. These are two distinct outputs.

OUTPUT 1: DATA REQUIREMENTS (what tools need to fetch)
These are concrete, fetchable data points. Each one should map to a tool or data source.

Tool categories available:
- SEC filing tools: revenue, EBITDA margin, tax rate, capex, depreciation, disclosures, 8-K events
- Financial tools: market data (beta, market cap, debt, cash), WACC calculation, DCF, comparable company analysis
- Macro tools (FRED): interest rates, inflation, GDP, unemployment, yield curve (get_macro_snapshot, get_treasury_yields, get_fred_series, search_fred)
- Market intelligence tools (Finnhub): news articles (get_company_news, get_market_news), insider transactions (get_insider_transactions), analyst recommendations (get_analyst_recommendations), key financials (get_basic_financials), peer companies (get_company_peers), earnings calendar (get_earnings_calendar)
- Web search: LAST RESORT only -- use for qualitative research, specific analyst reports, or data not available from the above tools

EXAMPLES OF GOOD DATA REQUIREMENTS:
For "Run a DCF on AAPL" (DCF needs ALL of these):
- data_needed: "Revenue base from latest 10-K", tool_hint: "SEC (get_revenue_base)"
- data_needed: "EBITDA margin", tool_hint: "SEC (get_ebitda_margin)"
- data_needed: "CapEx as % of revenue", tool_hint: "SEC (get_capex_pct_revenue)"
- data_needed: "Effective tax rate", tool_hint: "SEC (get_tax_rate)"
- data_needed: "Depreciation & amortization as % of revenue", tool_hint: "SEC (get_depreciation)"
- data_needed: "Beta, market cap, total debt, cash, shares outstanding", tool_hint: "financial (get_market_data)"
- data_needed: "Risk-free rate and macro context", tool_hint: "macro (get_treasury_yields)"
- data_needed: "WACC calculation", tool_hint: "financial (calculate_wacc) -- needs beta, risk-free rate, ERP, cost of debt, tax rate, market cap, total debt"
- data_needed: "DCF valuation", tool_hint: "financial (calculate_dcf) -- needs revenue, margins, growth, wacc, cash, debt, shares"

For "Is TSLA a good buy?":
- data_needed: "Recent 8-K material events", tool_hint: "SEC (extract_8k_events)"
- data_needed: "Revenue and profitability", tool_hint: "SEC (get_revenue_base, get_ebitda_margin)"
- data_needed: "Peer valuation multiples", tool_hint: "financial (comparable_company_analysis)"
- data_needed: "Current analyst ratings", tool_hint: "market_intel (get_analyst_recommendations)"
- data_needed: "Key financial metrics", tool_hint: "market_intel (get_basic_financials)"

For "What is the news sentiment for AAPL?":
- data_needed: "Recent company news articles", tool_hint: "market_intel (get_company_news)"
- data_needed: "Broad market news", tool_hint: "market_intel (get_market_news)"
- data_needed: "Insider buying/selling activity", tool_hint: "market_intel (get_insider_transactions)"

BAD DATA REQUIREMENTS (too vague):
- "What's the company's financial health?" (not a specific data point)
- "Market conditions" (what specifically?)

OUTPUT 2: ANALYTICAL CONSIDERATIONS (for the analysis agent to reason about)
These are strategic questions and framing guidance that the ANALYSIS AGENT should think about when synthesizing data. These do NOT require tool calls -- they guide interpretation.

EXAMPLES:
For "Run a DCF on AAPL":
- topic: "Revenue mix", guidance: "Model hardware vs services separately -- services has higher margins and growth"
- topic: "Terminal growth", guidance: "Mature mega-cap; terminal growth should reflect GDP-like rates (2-3%)"

For "Is TSLA a good buy?":
- topic: "Valuation framework", guidance: "Traditional auto multiples vs tech/growth multiples -- justify which to use"
- topic: "Optionality", guidance: "Consider autonomous driving and energy as call options on the base business"

RULES:
1. Generate 5-10 data requirements -- be thorough, list EVERY data point needed
2. Generate 3-5 analytical considerations -- strategic framing, not data fetching
3. Prioritize SEC and financial tools over search in data requirements
4. Only include search requirements when structured tools genuinely cannot provide the data
5. Data requirements must be concrete and fetchable, not analytical questions
6. For valuation analyses (DCF, comps), include the CALCULATION tools (calculate_wacc, calculate_dcf, comparable_company_analysis) as data requirements -- not just the inputs"""

    return prompt

  def probe(self, user_query: str, ticker: Optional[str] = None) -> Dict:
    """
    Generate strategic probing questions before analysis.

    Returns:
        Dict containing probing questions and recommendations
    """
    self.conversatoin_history = []
    system_prompt = self.build_prompt(user_query)

    user_prompt = f"""User Request: {user_query}"""
    if ticker:
        user_prompt += f"\nTicker: {ticker}"

    user_prompt += "\n\nGenerate strategic probing questions for this analysis."

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
