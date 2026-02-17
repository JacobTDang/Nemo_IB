from .openrouter_template import OpenRouterModel
from pydantic import BaseModel
from typing import Dict, Optional, List
import sys
import json


class VerificationResult(BaseModel):
  action: str
  action_reasoning: str
  quality_score: float
  answers_question: bool
  strengths: List[str]
  weaknesses: List[str]
  missing_components: List[str]


class Verification_Agent(OpenRouterModel):
  response_schema = VerificationResult
  MAX_OUTPUT_TOKENS = 4096  # Needs room for thinking + JSON output

  def __init__(self, model_name: str = 'deepseek/deepseek-r1-0528:free'):
    super().__init__(model_name=model_name)

  def build_prompt(self) -> str:
    """Build specialized system prompt for analysis verification"""
    from datetime import datetime
    current_date = datetime.now().strftime("%B %d, %Y")

    prompt = f"""You are a Quality Control Specialist at Goldman Sachs. You verify financial analyses before delivery to clients.

TODAY'S DATE: {current_date}
IMPORTANT: Dates in {datetime.now().year} are CURRENT, not future dates. Do NOT flag them as invalid.

CHECK THESE 5 THINGS:
1. Does it answer what the user asked? (DCF needs a valuation, "good buy?" needs a recommendation)
2. Are calculations correct? (WACC 5-15% typical, no nonsense numbers, show work)
3. Are assumptions stated? (growth rate, WACC, terminal value -- sourced or marked as assumptions)
4. Is reasoning sound? (conclusions supported by data, risks acknowledged)
5. Is anything missing? (what data gaps exist, what would improve it)

ACTIONS:
- "approve": Answers the question, calculations defensible, assumptions documented
- "revise": Partially answers but needs specific fixes (you MUST say what to fix)
- "reject": Fundamentally wrong or doesn't answer the question at all

EXAMPLES:
User: "Run DCF on AAPL" -> Analysis gives fair value range with methodology -> APPROVE
User: "Run DCF on AAPL" -> Analysis describes business but no valuation -> REJECT
User: "Is MSFT a good buy?" -> PE analysis with price target -> APPROVE

RULES:
- BIAS TOWARD APPROVE: If the analysis answers the question and the data is reasonable, APPROVE it. Minor stylistic issues, rounding differences, or data that shows "top N" instead of all entries are NOT reasons to revise. Only revise for material errors that would mislead an investor.
- Do NOT flag data as invalid just because you cannot verify the source. The data was gathered by automated tools and is presumed correct.

OUTPUT FORMAT:
You MUST respond with ONLY valid JSON matching this exact schema. No text before or after the JSON.
{{
  "action": "approve" | "revise" | "reject",
  "action_reasoning": "string explaining why you chose this action",
  "quality_score": 0.0 to 1.0,
  "answers_question": true or false,
  "strengths": ["list", "of", "strengths"],
  "weaknesses": ["list", "of", "weaknesses"],
  "missing_components": ["list", "of", "missing", "items"]
}}
ALL 7 fields are REQUIRED. Do not add or omit any fields."""

    return prompt

  def verify(self, user_query: str, analysis_output: str, execution_plan: Optional[Dict] = None) -> Dict:
    """
    Verify analysis output quality and alignment with user query.

    Returns:
        Dict with verification results and action (approve/revise/reject).
        On parse failure, returns error dict with raw_response for master to use.
    """
    system_prompt = self.build_prompt()

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

    user_prompt += "\nVerify this analysis."

    print(f"Verification Agent analyzing output quality...", file=sys.stderr, flush=True)

    response = self.generate_response(
        prompt=user_prompt,
        system_prompt=system_prompt
    )

    try:
      result = self.parse_response(response)
      verification = result.model_dump()

      print(f"\n{'='*60}", file=sys.stderr, flush=True)
      print(f"VERIFICATION COMPLETE", file=sys.stderr, flush=True)
      print(f"{'='*60}", file=sys.stderr, flush=True)
      print(f"Action: {verification['action'].upper()}", file=sys.stderr, flush=True)
      print(f"Quality Score: {verification['quality_score']}", file=sys.stderr, flush=True)
      print(f"{'='*60}\n", file=sys.stderr, flush=True)

      return verification
    except Exception as e:
      print(f"Warning: Verification parse failed: {e}", file=sys.stderr, flush=True)
      return {"error": f"Verification parse failed: {e}", "raw_response": response[:500]}


if __name__ == "__main__":
    pass
