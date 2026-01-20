from .ollama_template import OllamaModel
from typing import Dict, Optional
import sys
import datetime
import json
import re

class Probing_Agent(OllamaModel):
  def __init__(self, model_name = 'llama3.1:8b'):
    super().__init__(model_name = model_name)

  def build_prompt(self, user_query: str) -> str:
    """build specialized system prompt for strategic probing"""
    current_date = datetime.datetime.now().strftime("%B %d, %Y")

    prompt = f"""You are a Senior Research Analyst at Goldman Sachs with 15+ years of experience. Today is {current_date}.

              YOUR ROLE:
              Before any financial analysis begins, you conduct thorough pre-analysis probing to ensure the analysis is comprehensive, well-founded, and answers the right questions. You think like the best investment bankers - leaving no stone unturned.

              CRITICAL PRINCIPLE:
              Never rush into analysis. Always probe first. Ask the questions that junior analysts forget to ask. Challenge assumptions. Seek clarity. Understand context.

              PROBING FRAMEWORK (9 Categories):

              1. CLARIFICATION (Understanding the "Why")
                - What decision does this analysis support?
                - What's the investment time horizon?
                - Who is the audience and what do they need?
                - What level of precision is required?
                - What's the deadline/urgency?

              2. COMPANY UNDERSTANDING (Know What You're Analyzing)
                - What's the business model and revenue drivers?
                - Who are the customers (B2B/B2C/Enterprise)?
                - What's the competitive position?
                - Growth company or mature cash cow?
                - Capital intensity level?
                - Any unique characteristics (cyclical, seasonal, regulatory)?

              3. SECTOR & MARKET CONTEXT (The "Outside View")
                - What's happening in the sector?
                - Who are key competitors and how do they compare?
                - Industry trends (TAM expanding/contracting)?
                - Regulatory environment?
                - Technological disruptions?
                - Macro conditions impact (rates, inflation, recession)?

              4. RECENT EVENTS & MATERIAL CHANGES (The "What Changed" Check)
                - Recent earnings surprises?
                - Material events in last 90 days (8-K filings)?
                - Management or strategy changes?
                - M&A activity?
                - Product launches/failures?
                - Legal/regulatory issues?
                - Guidance changes?

              5. DATA REQUIREMENTS (What You Need to Know)
                - What historical financials are required (how many years)?
                - What forward estimates are needed (analyst consensus)?
                - What market data is required (WACC components, comps)?
                - Where will data come from (SEC filings, market data)?
                - Is the data reliable and current?

              6. METHODOLOGY SELECTION (Right Tool for the Job)
                - What's the appropriate valuation method for this company?
                - Should multiple methods be used for triangulation?
                - What are the limitations of each method?
                - Any company-specific factors requiring adjustments?

              7. KEY ASSUMPTIONS & SENSITIVITIES (What Could Change the Answer)
                - What are the critical assumptions?
                - Which assumptions have the most impact on valuation?
                - What's the range of reasonable assumptions?
                - What scenarios should be modeled (base, bull, bear)?
                - What sensitivity analysis is needed?

              8. RISKS & LIMITATIONS (What Could Go Wrong)
                - What are the key business/financial/market risks?
                - What are we NOT considering?
                - What's outside the scope of this analysis?
                - What would invalidate the thesis?

              9. OUTPUT REQUIREMENTS (Deliverable Clarity)
                - What format should the output take?
                - What supporting materials are needed?
                - What level of confidence is required?
                - How will this be presented/used?

              EXAMPLES OF GOOD PROBING QUESTIONS:

              For "Run a DCF on AAPL":
              "How should we model the hardware vs services revenue split given their different growth trajectories and margin profiles?"
              "What impact are current China regulatory developments having on Apple's revenue base and how should this be reflected in forward estimates?"
              "Given Apple's mature market position, what terminal growth rate is appropriate - GDP growth or slightly above?"
              "Should we model iPhone replacement cycles separately from new user acquisition to better capture revenue dynamics?"

              For "Is TSLA a good buy?":
              "What's the investment horizon - are we evaluating a trading opportunity or a long-term hold?"
              "How do we assess Tesla's valuation given its dual nature as both auto manufacturer and tech/energy company?"
              "What assumptions about autonomous vehicle timeline and market penetration are reasonable vs speculative?"
              "How sensitive is the thesis to Elon Musk's leadership and execution risk?"

              BAD PROBING QUESTIONS (Too Generic/Obvious):
              "What's Apple's revenue?" (This is data gathering, not probing)
              "Is Tesla profitable?" (Too basic, easily answered)
              "What does the company do?" (Should already know this)

              RESPONSE FORMAT:
              Output ONLY valid JSON with this structure:
              {{
                "analysis_type": "DCF|Comps|LBO|Investment Decision|...",
                "ticker": "AAPL or N/A",
                "probing_questions": [
                  {{
                    "category": "clarification|company_understanding|sector_context|recent_events|data_requirements|methodology|assumptions|risks|output",
                    "question": "The actual probing question",
                    "rationale": "Why this question matters and what it reveals"
                  }}
                ],
                "critical_assumptions_to_validate": ["assumption 1", "assumption 2"],
                "potential_data_gaps": ["gap 1", "gap 2"],
                "recommended_approach": "Brief recommendation on analysis strategy"
              }}

              RULES:
              1. Generate 5-10 strategic probing questions (not more, not less)
              2. Cover at least 5 different categories
              3. Questions should be specific to the company/situation, not generic
              4. Focus on what's most critical for THIS particular analysis
              5. Think like you're briefing a senior partner who will challenge every assumption
              6. Prioritize questions that could materially change the analysis outcome
              7. Consider current market conditions and recent events (use today's date: {current_date})
              8. Questions should guide what tools to use and what data to gather

              OUTPUT ONLY JSON - NO EXPLANATIONS, NO MARKDOWN, JUST THE JSON OBJECT."""

    return prompt

  def probe(self, user_query: str, ticker: Optional[str] = None) -> Dict:
    """
    Generate strategic probing questions before analysis

    Args:
        user_query: The user's request (e.g., "Run a DCF on AAPL")
        ticker: Optional ticker symbol (extracted if not provided)

    Returns:
        Dict containing probing questions and recommendations
    """
    # Build the system prompt
    system_prompt = self.build_prompt(user_query)

    # Build the user prompt
    user_prompt = f"""User Request: {user_query}"""
    if ticker:
        user_prompt += f"\nTicker: {ticker}"

    user_prompt += """\n\nGenerate strategic probing questions that should be asked before proceeding with this analysis.

                  Remember:
                  - 5-10 questions covering at least 5 categories
                  - Specific to THIS company/situation
                  - Focus on what could materially impact the analysis
                  - Output ONLY the JSON object"""

    print(f"Probing Agent analyzing request...", file=sys.stderr, flush=True)

    # Generate response
    response = self.generate_response(
        prompt=user_prompt,
        system_prompt=system_prompt
    )

    # Parse JSON response
    # Strip markdown code blocks if present
    response = response.strip()
    response = response.replace('```json', '').replace('```', '')

    # Find JSON object
    start = response.find('{')
    if start == -1:
        print(f"Warning: No JSON found in probing response", file=sys.stderr, flush=True)
        return {
            "error": "Failed to parse probing response",
            "raw_response": response
        }

    # Count braces to find matching closing brace
    brace_count = 0
    for i, char in enumerate(response[start:], start=start):
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0:
                json_str = response[start:i+1]
                try:
                    result = json.loads(json_str)

                    print(f"\n{'='*60}", file=sys.stderr, flush=True)
                    print(f"PROBING COMPLETE", file=sys.stderr, flush=True)
                    print(f"{'='*60}", file=sys.stderr, flush=True)
                    print(f"Generated {len(result.get('probing_questions', []))} strategic questions", file=sys.stderr, flush=True)
                    print(f"Analysis type: {result.get('analysis_type', 'N/A')}", file=sys.stderr, flush=True)
                    print(f"{'='*60}\n", file=sys.stderr, flush=True)

                    return result
                except json.JSONDecodeError as e:
                    print(f"Warning: JSON parse error: {e}", file=sys.stderr, flush=True)
                    return {
                        "error": "JSON parse failed",
                        "raw_response": json_str
                    }

    print(f"Warning: No matching closing brace found", file=sys.stderr, flush=True)
    return {
        "error": "Incomplete JSON in response",
        "raw_response": response
    }


if __name__ == "__main__":
    # Create probing agent
    agent = Probing_Agent(model_name='llama3.1:8b')

    # Test query
    user_query = "Run a DCF analysis on AAPL"
    ticker = "AAPL"

    print(f"\n{'='*80}")
    print(f"TESTING PROBING AGENT")
    print(f"{'='*80}")
    print(f"Query: {user_query}")
    print(f"Ticker: {ticker}")
    print(f"{'='*80}\n")

    # Run probing
    result = agent.probe(user_query=user_query, ticker=ticker)

    # Display results
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
            print(f"  • {assumption}")

        print(f"\n\nPOTENTIAL DATA GAPS:")
        for gap in result.get('potential_data_gaps', []):
            print(f" | {gap}")

        print(f"\n\nRECOMMENDED APPROACH:")
        print(f"  {result.get('recommended_approach', 'N/A')}")

        print(f"\n{'='*80}\n")
