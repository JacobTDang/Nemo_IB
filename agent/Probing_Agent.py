from .ollama_template import OllamaModel
from pydantic import BaseModel
from typing import Dict, Optional, List
import sys
import datetime


class ProbingQuestion(BaseModel):
  category: str
  question: str
  rationale: str


class ProbingResult(BaseModel):
  analysis_type: str
  ticker: Optional[str] = None
  probing_questions: List[ProbingQuestion]
  critical_assumptions_to_validate: List[str]
  potential_data_gaps: List[str]
  recommended_approach: str


class Probing_Agent(OllamaModel):
  response_schema = ProbingResult

  def __init__(self, model_name = 'llama3.1:8b'):
    super().__init__(model_name = model_name)

  def build_prompt(self, user_query: str) -> str:
    """build specialized system prompt for strategic probing"""
    current_date = datetime.datetime.now().strftime("%B %d, %Y")

    prompt = f"""You are a Senior Research Analyst at Goldman Sachs with 15+ years of experience. Today is {current_date}.

YOUR ROLE:
Before any financial analysis begins, you conduct thorough pre-analysis probing to ensure the analysis is comprehensive, well-founded, and answers the right questions. Think like the best investment bankers - leave no stone unturned.

PROBING FRAMEWORK (9 Categories):

1. CLARIFICATION - What decision does this support? Time horizon? Audience?
2. COMPANY UNDERSTANDING - Business model, revenue drivers, competitive position, growth stage
3. SECTOR & MARKET CONTEXT - Sector trends, competitors, TAM, regulatory, macro conditions
4. RECENT EVENTS - Earnings surprises, 8-K filings, management changes, M&A, guidance
5. DATA REQUIREMENTS - Historical financials needed, forward estimates, WACC components, data sources
6. METHODOLOGY SELECTION - Appropriate valuation method, multiple methods for triangulation
7. KEY ASSUMPTIONS & SENSITIVITIES - Critical assumptions, sensitivity analysis, scenarios (base/bull/bear)
8. RISKS & LIMITATIONS - Business/financial/market risks, what we're NOT considering
9. OUTPUT REQUIREMENTS - Format, supporting materials, confidence level

EXAMPLES OF GOOD PROBING QUESTIONS:

For "Run a DCF on AAPL":
- "How should we model the hardware vs services revenue split given their different growth trajectories?"
- "Given Apple's mature market position, what terminal growth rate is appropriate?"

For "Is TSLA a good buy?":
- "What's the investment horizon - trading opportunity or long-term hold?"
- "How sensitive is the thesis to autonomous vehicle timeline assumptions?"

BAD PROBING QUESTIONS (Too Generic):
- "What's Apple's revenue?" (data gathering, not probing)
- "Is Tesla profitable?" (too basic)

RULES:
1. Generate 5-10 strategic probing questions
2. Cover at least 5 different categories
3. Questions must be specific to the company/situation, not generic
4. Focus on what could materially change the analysis outcome
5. Questions should guide what tools to use and what data to gather"""

    return prompt

  def probe(self, user_query: str, ticker: Optional[str] = None) -> Dict:
    """
    Generate strategic probing questions before analysis.

    Returns:
        Dict containing probing questions and recommendations
    """
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
      print(f"Generated {len(parsed.get('probing_questions', []))} strategic questions", file=sys.stderr, flush=True)
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

        print(f"\n\nSTRATEGIC PROBING QUESTIONS:")
        print(f"{'-'*80}")
        for idx, q in enumerate(result.get('probing_questions', []), 1):
            print(f"\n{idx}. [{q['category'].upper()}]")
            print(f"   Q: {q['question']}")
            print(f"   Why: {q['rationale']}")

        print(f"\n\nCRITICAL ASSUMPTIONS TO VALIDATE:")
        for assumption in result.get('critical_assumptions_to_validate', []):
            print(f"  - {assumption}")

        print(f"\n\nPOTENTIAL DATA GAPS:")
        for gap in result.get('potential_data_gaps', []):
            print(f"  - {gap}")

        print(f"\n\nRECOMMENDED APPROACH:")
        print(f"  {result.get('recommended_approach', 'N/A')}")

        print(f"\n{'='*80}\n")
