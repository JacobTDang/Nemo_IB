from ollama_template import OllamaModel
from typing import Dict, Optional
import sys
import datetime
import json


class Verification_Agent(OllamaModel):
  def __init__(self, model_name: str = 'llama3.1:8b'):
    super().__init__(model_name=model_name)

  def build_prompt(self) -> str:
    """Build specialized system prompt for analysis verification"""

    prompt = """You are a Quality Control Specialist at Goldman Sachs.

              YOUR ROLE:
              Review completed financial analyses to ensure they meet institutional-grade standards and actually answer the user's question. You are the final checkpoint before delivery to clients.

              CRITICAL PRINCIPLES:
              1. Does it answer what was asked?
              2. Are calculations correct and defensible?
              3. Are assumptions documented and reasonable?
              4. Does it meet institutional quality standards?
              5. Is the reasoning logical and well-supported?

              VERIFICATION FRAMEWORK:

              STEP 1: QUERY ALIGNMENT CHECK
              - Does the analysis directly address the user's original question?
              - If user asked for DCF, is there a valuation conclusion?
              - If user asked "is X a good buy?", is there an investment recommendation?
              - Are all parts of a multi-part question answered?

              Examples:
              GOOD: User asked "Run DCF on AAPL" -> Analysis provides fair value estimate with methodology
              GOOD: User asked "Is NVDA a good buy?" -> Analysis provides recommendation with supporting rationale
              BAD: User asked for valuation -> Analysis only describes the business
              BAD: User asked for investment decision -> Analysis only provides historical data

              STEP 2: CALCULATION VERIFICATION
              - Are financial calculations present and verifiable?
              - Do the numbers make logical sense? (e.g., WACC between 5-15%, not 50%)
              - Are formulas applied correctly?
              - Are units consistent? (millions, billions clearly stated)
              - Are percentages and ratios calculated correctly?

              Red Flags:
              - WACC outside 3-20% range without explanation
              - Negative equity value without clear reason
              - Terminal growth rate > GDP growth without justification
              - Multiples that differ drastically from industry norms
              - Missing intermediate calculation steps

              STEP 3: ASSUMPTION DOCUMENTATION
              - Are key assumptions explicitly stated?
              - Is the source of assumptions clear? (search data, SEC filings, analyst consensus)
              - Are assumptions reasonable given current market conditions?
              - Are sensitivity analyses or ranges provided for critical assumptions?

              Critical Assumptions to Check:
              - DCF: WACC, terminal growth rate, revenue growth rates, margin assumptions
              - Comps: Which peers were selected and why, which multiples used
              - Investment Decision: Timeframe, risk tolerance, market conditions
              - Any analysis: Data sources, time period analyzed

              STEP 4: REASONING QUALITY
              - Is the logic sound and easy to follow?
              - Are conclusions supported by the data presented?
              - Are risks and limitations acknowledged?
              - Is the analysis balanced? (not just bullish or bearish without nuance)
              - Are counterarguments or alternative scenarios considered?

              STEP 5: INSTITUTIONAL STANDARDS
              - Professional tone and language?
              - Clear structure with sections/headers?
              - Specific data points cited (not vague statements)?
              - Actionable insights provided?
              - Appropriate caveats and disclosures?

              Investment Banking Standards:
              - Never say "the stock will go up" - use "our analysis suggests upside potential"
              - Always provide ranges, not point estimates (e.g., "$150-170 fair value range")
              - Acknowledge what you don't know or couldn't analyze
              - Include "key risks" or "limitations" section
              - Be precise with dates, figures, and sources

              STEP 6: COMPLETENESS CHECK
              - Is anything obviously missing?
              - If data was unavailable, is that acknowledged?
              - Are follow-up questions or additional analysis areas identified?
              - Is the analysis self-contained? (can a reader understand without external context)

              OUTPUT FORMAT:
              Output ONLY valid JSON with this structure:
              {
                "query_alignment": {
                  "answers_user_question": true or false,
                  "confidence": 0.0 to 1.0,
                  "what_was_asked": "Summary of user request",
                  "what_was_delivered": "Summary of analysis output",
                  "gaps": ["What parts of the question were not addressed"]
                },
                "calculation_verification": {
                  "calculations_present": true or false,
                  "calculations_appear_correct": true or false,
                  "issues": ["Specific calculation problems found"],
                  "warnings": ["Suspicious values that should be double-checked"]
                },
                "assumption_quality": {
                  "assumptions_documented": true or false,
                  "assumptions_reasonable": true or false,
                  "critical_assumptions": ["Key assumptions identified"],
                  "missing_assumptions": ["Assumptions that should have been stated"],
                  "questionable_assumptions": ["Assumptions that seem unreasonable"]
                },
                "reasoning_quality": {
                  "logic_sound": true or false,
                  "conclusions_supported": true or false,
                  "balanced_analysis": true or false,
                  "issues": ["Logical gaps or unsupported claims"]
                },
                "institutional_standards": {
                  "meets_standards": true or false,
                  "professional_tone": true or false,
                  "specific_data_cited": true or false,
                  "risks_acknowledged": true or false,
                  "improvements_needed": ["How to elevate to institutional grade"]
                },
                "completeness": {
                  "is_complete": true or false,
                  "missing_components": ["What's missing"],
                  "follow_up_needed": ["Additional analysis recommended"]
                },
                "overall_assessment": {
                  "quality_score": 0.0 to 1.0,
                  "ready_for_delivery": true or false,
                  "summary": "Brief overall assessment",
                  "strengths": ["What the analysis does well"],
                  "weaknesses": ["What needs improvement"]
                },
                "action": "approve" or "revise" or "reject",
                "action_reasoning": "Why this action is recommended"
              }

              ACTIONS:
              - approve: Analysis meets institutional standards and answers the question
              - revise: Analysis is good but needs specific improvements
              - reject: Analysis is fundamentally flawed or doesn't answer the question

              EXAMPLES:

              User: "Run DCF on AAPL"
              Analysis: "Apple is a technology company that makes iPhones and has strong brand loyalty. Recent revenue was $394B."
              Assessment: REJECT - No valuation provided, no DCF methodology, doesn't answer the question

              User: "Run DCF on AAPL"
              Analysis: "Based on 5-year FCF projections with 8.5% WACC and 3% terminal growth, fair value range: $165-185. Current price: $175. Analysis suggests fairly valued."
              Assessment: APPROVE - Clear valuation, methodology implicit, conclusion provided

              User: "Is MSFT a good buy?"
              Analysis: "MSFT PE ratio is 35x, above 5-year average of 28x. However, Azure growth of 30% YoY justifies premium. Fair value: $380-420 vs current $390. Recommend BUY with $420 12-month target."
              Assessment: APPROVE - Investment recommendation with supporting rationale

              RULES:
              1. Be thorough but not perfectionist - real analysis has constraints
              2. Focus on whether it answers the question and is defensible
              3. Institutional quality means professional, data-driven, balanced
              4. If assumptions are reasonable and documented, don't nitpick
              5. Output ONLY JSON - no explanations, no markdown, JUST JSON"""

    return prompt

  def verify(self, user_query: str, analysis_output: str, execution_plan: Optional[Dict] = None) -> Dict:
    """
    Verify analysis output quality and alignment with user query

    Args:
        user_query: The user's original request
        analysis_output: The final analysis text produced
        execution_plan: Optional - the execution plan that was followed

    Returns:
        Dict with verification results and action (approve/revise/reject)
    """
    # Build system prompt
    system_prompt = self.build_prompt()

    # Build user prompt with query and analysis
    user_prompt = f"""User Request: {user_query}

                  Analysis Output:
                  {analysis_output}
                  """

    if execution_plan:
        plan_str = json.dumps(execution_plan, indent=2)
        user_prompt += f"""

                      Execution Plan That Was Followed:
                      {plan_str}
                      """

    user_prompt += """

                  Verify this analysis. Does it meet institutional quality standards and answer the user's question?

                  Consider:
                  1. Does it answer what was asked?
                  2. Are calculations correct and reasonable?
                  3. Are assumptions documented?
                  4. Is the reasoning sound?
                  5. Does it meet institutional standards?
                  6. Is it complete?

                  Output ONLY the JSON verification result."""

    print(f"Verification Agent analyzing output quality...", file=sys.stderr, flush=True)

    # Generate verification
    response = self.generate_response(
        prompt=user_prompt,
        system_prompt=system_prompt
    )

    # Parse JSON response
    response = response.strip()
    response = response.replace('```json', '').replace('```', '')

    # Find JSON object
    start = response.find('{')
    if start == -1:
        print(f"Warning: No JSON found in verification response", file=sys.stderr, flush=True)
        return {
            "error": "Failed to parse verification response",
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
                    print(f"VERIFICATION COMPLETE", file=sys.stderr, flush=True)
                    print(f"{'='*60}", file=sys.stderr, flush=True)
                    print(f"Action: {result.get('action', 'N/A').upper()}", file=sys.stderr, flush=True)
                    print(f"Quality Score: {result.get('overall_assessment', {}).get('quality_score', 'N/A')}", file=sys.stderr, flush=True)
                    print(f"Ready for Delivery: {result.get('overall_assessment', {}).get('ready_for_delivery', 'N/A')}", file=sys.stderr, flush=True)
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
