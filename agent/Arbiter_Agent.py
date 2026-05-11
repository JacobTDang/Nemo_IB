"""Arbiter. Reads the analyst report, the bull case, and the bear case, and
synthesizes a final verdict. Uses a DIFFERENT model family from Bull/Bear so
the synthesis isn't biased by the side-pickers' habits.

The arbiter's verdict — not the analyst's — is what gets persisted as the
recommendation in the thesis.
"""
from .groq_template import GroqModel
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import json
import sys


class ArbiterVerdict(BaseModel):
  final_recommendation: str = Field(
    description="One of: BUY, HOLD, SELL, NEUTRAL"
  )
  confidence: float = Field(description="0-1 confidence in this final call.")
  bull_strength: float = Field(description="0-1 — how strong the bull case actually was.")
  bear_strength: float = Field(description="0-1 — how strong the bear case actually was.")
  decisive_factors: List[str] = Field(
    default_factory=list,
    description="What tipped the scale. Cite specific evidence."
  )
  acknowledged_risks: List[str] = Field(
    default_factory=list,
    description="Risks the verdict acknowledges but discounts."
  )
  conditions_to_change_mind: List[str] = Field(
    default_factory=list,
    description="Explicit thesis-breakers — events that would flip the call."
  )
  position_sizing_guidance: str = Field(
    description="One of: aggressive, normal, cautious, no_position"
  )
  rationale: str = Field(description="<= 600 chars explaining how the verdict was reached.")


SYSTEM_PROMPT = """You are the Chief Investment Officer. Two analysts have
made one-sided cases for the same stock — one bull, one bear. Your job is to
weigh which case is stronger based on the EVIDENCE each cites, not the
confidence each claims. Pick the recommendation, set position sizing, and
name the conditions that would change your mind.

DECISION GUIDE:
  - Bull conviction without specific evidence < bear with specific evidence.
  - If both cases cite real, conflicting evidence, lean toward HOLD with
    cautious sizing.
  - If one side meaningfully refutes the other ("they say X but the data says
    Y"), that side wins.
  - Confidence should reflect the spread between bull_strength and
    bear_strength. Close debates get lower confidence.
  - Position sizing:
    - aggressive: high conviction + low risk → larger position
    - normal: typical conviction
    - cautious: high uncertainty or close debate
    - no_position: structurally unclear or recommendation=HOLD/NEUTRAL

  - conditions_to_change_mind MUST be concrete. "If earnings miss" is OK;
    "if things go bad" is not.

OUTPUT: ONLY a JSON object matching the schema.

EXAMPLE OUTPUT:
{"final_recommendation": "BUY",
 "confidence": 0.68,
 "bull_strength": 0.75, "bear_strength": 0.45,
 "decisive_factors": ["Services growth >14% YoY confirmed in 2 prior prints", "Cash return policy supports floor at $170"],
 "acknowledged_risks": ["China iPhone unit decline could accelerate"],
 "conditions_to_change_mind": ["Services growth <10% in next earnings", "EU DMA ruling forces App Store fee cut >50%"],
 "position_sizing_guidance": "normal",
 "rationale": "Bull cited specific Services growth evidence; bear's China concern is real but smaller than Services tailwind."}"""


class Arbiter_Agent(GroqModel):
  response_schema = ArbiterVerdict
  MAX_OUTPUT_TOKENS = 2048
  # Different model family from Bull/Bear (which use llama-3.3-70b) to reduce
  # same-model bias in synthesis.
  FALLBACK_MODEL = 'llama-3.3-70b-versatile'

  def __init__(self, model_name: str = "qwen/qwen3-32b"):
    super().__init__(model_name=model_name)

  def judge(self,
            analyst_report_md: str,
            bull_case,  # BullCase Pydantic instance
            bear_case,  # BearCase Pydantic instance
            variables: Optional[Dict[str, Any]] = None,
            ticker: str = "") -> Optional[ArbiterVerdict]:
    self.conversatoin_history = []
    flat_vars = {k: v for k, v in (variables or {}).items() if '.' not in k}
    bull_dict = bull_case.model_dump() if bull_case else {}
    bear_dict = bear_case.model_dump() if bear_case else {}
    prompt = (
      f"TICKER: {ticker}\n\n"
      f"ANALYST DRAFT REPORT:\n{analyst_report_md[:5000]}\n\n"
      f"BULL CASE:\n{json.dumps(bull_dict, indent=2)}\n\n"
      f"BEAR CASE:\n{json.dumps(bear_dict, indent=2)}\n\n"
      f"GROUNDING VARIABLES:\n{json.dumps(flat_vars, default=str)[:3000]}\n\n"
      "Synthesize a final verdict."
    )
    try:
      resp = self.generate_response(prompt=prompt, system_prompt=SYSTEM_PROMPT)
      return self.parse_response(resp)
    except Exception as e:
      print(f"[Arbiter] parse failed: {e}", file=sys.stderr, flush=True)
      return None
