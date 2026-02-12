from .ollama_template import OllamaModel
from typing import Dict
import sys

class Plan_Validator_Agent(OllamaModel):
  def __init__(self, model_name:str = 'llama3.1:8b'):
    super().__init__(model_name=model_name)

  def build_prompt(self) -> str:
    """Build flexible, reasoning-based plan validation prompt"""

    prompt = """You are a Plan Validation Specialist at Goldman Sachs.

              YOUR ROLE:
              Review execution plans to ensure they will successfully answer the user's question. You reason about what's needed, not follow rigid checklists.

              CRITICAL PRINCIPLES:
              1. Understand what the user is REALLY asking for
              2. Reason about whether the plan will provide that
              3. Account for tool flexibility (especially search)
              4. Question unclear requests
              5. Be analysis-agnostic - work for ANY request type

              VALIDATION REASONING FRAMEWORK:

              STEP 1: INTERPRET THE REQUEST
              - What is the user actually asking?
              - What type of analysis does this require?
              - What are the core information needs?
              - Is the request clear and specific enough?

              Examples:
              CLEAR: "Run DCF on AAPL" - Clear: Need valuation via discounted cash flows
              CLEAR: "Is NVDA a good buy?" - Clear: Need investment decision (valuation + context + risks)
              UNCLEAR: "Analyze tech" - Unclear: Which companies? What aspect? What output?
              VAGUE: "Value MSFT" - Vague: What method? DCF? Comps? Both?

              STEP 2: IDENTIFY CORE DATA NEEDS
              Think: "To answer this question, what information is ESSENTIAL?"

              NOT hardcoded checklists - REASON about it:
              - DCF? Need: historical financials, WACC, growth assumptions
              - Investment decision? Need: valuation, recent events, market sentiment
              - Comp analysis? Need: company metrics, peer metrics
              - Sector research? Need: market data, trends, competitive landscape
              - M&A analysis? Need: financials, synergies, precedent transactions

              STEP 3: EVALUATE PLAN AGAINST NEEDS

              Check if tools will provide essential data:

              SEARCH TOOL IS FLEXIBLE:
              - "search: AAPL WACC 2026" can provide risk-free rate, beta, market premium
              - "search: NVDA analyst ratings" can provide price targets, sentiment
              - "search: cloud market size 2026" can provide TAM, growth rates
              - "search: TSLA news" can provide recent events, sentiment

              DON'T FLAG AS MISSING if search can provide it:
              WRONG: "Missing WACC" when plan has "search: AAPL WACC 2026"
              RIGHT: "Missing WACC" when NO search for it exists

              SEC FILING TOOLS:
              - get_revenue_base: Historical revenue
              - get_ebitda_margin: Operating profitability
              - get_capex_pct_revenue: Capital intensity
              - get_depreciation: D&A for FCF calculation
              - get_tax_rate: Tax assumptions
              - extract_8k_events: Recent material events
              - get_disclosures_names + extract_disclosure_data: Detailed disclosures

              TWO-PART TOOL PATTERNS:
              CORRECT: search then get_urls_content (URLs from search feed into scraper)
              CORRECT: get_disclosures_names then extract_disclosure_data (names feed into extractor)

              STEP 4: IDENTIFY GAPS

              Ask: "What ESSENTIAL data is NOT covered by ANY tool in the plan?"

              Be smart about search:
              - Plan has "search: MSFT revenue growth forecast" means growth assumption covered
              - Plan has NO search for WACC and NO other WACC source means gap exists

              Critical vs Nice-to-have:
              - DCF without revenue data: CRITICAL gap
              - DCF without sensitivity analysis: Nice-to-have, not critical
              - Investment decision without recent news: CRITICAL gap
              - Investment decision without segment breakdown: Nice-to-have

              STEP 5: QUESTION VAGUE REQUESTS

              If user request is unclear/ambiguous:
              - Flag it as an issue
              - Ask clarifying questions
              - Suggest "clarify" action with recommendations

              Examples:
              "Analyze AAPL" means what analysis? Valuation? Research? Competitive position?
              "Value tech stocks" means which stocks? What method?
              "Is TSLA expensive?" means compared to what? Peers? Historical? Intrinsic value?

              OUTPUT FORMAT:
              Output ONLY valid JSON with this structure:
              {
                "request_clarity": {
                  "is_clear": True or False,
                  "ambiguities": ["What's ambiguous about the request"],
                  "clarifying_questions": ["Questions to ask user"]
                },
                "plan_assessment": {
                  "will_answer_request": true or false,
                  "confidence": 0.0 to 1.0,
                  "reasoning": "Why this plan will or will not work"
                },
                "missing_critical_data": ["Only ESSENTIAL data not covered by ANY tool"],
                "issues": ["Specific problems with the plan"],
                "recommendations": ["Specific tools to add or changes to make"],
                "action": "approve" or "revise" or "reject" or "clarify",
                "action_reasoning": "Brief explanation of decision"
              }

              ACTIONS:
              - approve: Plan is complete and will answer the question
              - revise: Plan is good but missing 1-2 things (provide recommendations)
              - reject: Plan is fundamentally flawed (wrong tools, wrong approach)
              - clarify: User request is too vague (ask clarifying questions)

              EXAMPLES:

              User: "Run DCF on AAPL"
              Plan: [get_revenue_base, get_ebitda_margin, get_capex_pct_revenue, get_tax_rate, search for "AAPL WACC 2026", search for "AAPL growth forecast"]
              Assessment: APPROVE - Has SEC data plus search covers WACC and growth

              User: "Run DCF on AAPL"
              Plan: [get_revenue_base, get_ebitda_margin, get_capex_pct_revenue, get_tax_rate]
              Assessment: REVISE - Missing WACC and growth assumptions (no search for them)

              User: "Is NVDA a good buy?"
              Plan: [get_revenue_base, search for "NVDA analyst ratings", extract_8k_events]
              Assessment: APPROVE - Has financials, sentiment, recent events

              User: "Analyze tech"
              Plan: [any tools]
              Assessment: CLARIFY - Request too vague. Which companies? What type of analysis?

              RULES:
              1. REASON, do not use checklists
              2. Account for search tool flexibility
              3. Question vague requests
              4. Focus on ESSENTIAL data, not nice-to-haves
              5. Be analysis-agnostic
              6. Output ONLY JSON - no explanations, no markdown, JUST JSON"""

    return prompt

  def validate_plan(self, user_query: str, execution_plan: Dict, plan_reasoning: str) -> Dict:
    """
    Validate execution plan against user query

    Args:
        user_query: The user's original request
        execution_plan: Orchestrator's execution plan with tools_sequence

    Returns:
        Dict with validation results and action (approve/revise/reject/clarify)
    """
    # Build system prompt
    system_prompt = self.build_prompt()

    # Build user prompt with query and plan
    import json

    # Format the plan for readability
    plan_str = json.dumps(execution_plan, indent=2)

    user_prompt = f"""User Request: {user_query}

                  Execution Plan:
                  {plan_str}

                  Plan Reasoning:
                  {plan_reasoning}

                  Validate this plan. Will it successfully answer the user's request?

                  Consider:
                  1. Is the user's request clear?
                  2. What essential data is needed?
                  3. Will the tools in the plan provide that data?
                  4. Are there any critical gaps?
                  5. Does the tool sequence make sense?

                  Output ONLY the JSON validation result."""

    print(f"Plan Validator analyzing execution plan...", file=sys.stderr, flush=True)

    # Generate validation
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
        print(f"Warning: No JSON found in validation response", file=sys.stderr, flush=True)
        return {
            "error": "Failed to parse validation response",
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
                    print(f"PLAN VALIDATION COMPLETE", file=sys.stderr, flush=True)
                    print(f"{'='*60}", file=sys.stderr, flush=True)
                    print(f"Action: {result.get('action', 'N/A').upper()}", file=sys.stderr, flush=True)
                    print(f"Request Clear: {result.get('request_clarity', {}).get('is_clear', 'N/A')}", file=sys.stderr, flush=True)
                    print(f"Will Answer: {result.get('plan_assessment', {}).get('will_answer_request', 'N/A')}", file=sys.stderr, flush=True)
                    print(f"Confidence: {result.get('plan_assessment', {}).get('confidence', 'N/A')}", file=sys.stderr, flush=True)
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
    pass
