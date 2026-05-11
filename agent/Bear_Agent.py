"""Bear-side advocate. Mirror of Bull_Agent — argues the strongest downside
case, identifies risks the analyst may have downplayed, sets downside targets.
"""
from .groq_template import GroqModel
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import json
import sys


class BearCase(BaseModel):
  thesis: str = Field(description="1-paragraph max-bear thesis. Conviction language OK.")
  risks: List[str] = Field(
    default_factory=list,
    description="Specific named risks with explanation, not generic 'macro risk'."
  )
  downside_targets: List[str] = Field(
    default_factory=list,
    description="Price/valuation targets with conditions."
  )
  refutation_of_bull: str = Field(
    description="Where the bull case fails. Cite specific assumptions."
  )
  conviction: float = Field(description="0-1 self-rated conviction in this bear case.")


SYSTEM_PROMPT = """You are a senior short-side analyst tasked with making the
STRONGEST CASE that this stock is overvalued and headed lower. You are not
neutral. Your job is to identify every reason this stock could disappoint,
miss numbers, or de-rate, using the data provided.

GROUND RULES:
  - Use only facts present in the provided context. Do NOT invent numbers.
  - Identify specific risks, not vague 'macro headwinds'. "Apple Services
    growth has decelerated from 27% to 14% over six quarters" is concrete;
    "growth might slow" is weak.
  - Set downside targets with explicit conditions. "Stock to $130 if iPhone
    units fall >8% YoY" beats "stock could go lower."
  - Preemptively refute the bull case. The bulls will say X. Why is X wrong?
  - High conviction (>0.7) requires concrete evidence; reserve 0.9+ for
    overwhelming cases.

OUTPUT: ONLY a JSON object matching the schema.

EXAMPLE OUTPUT:
{"thesis": "Hardware unit decline plus services growth deceleration crushes the multiple — 20% downside.",
 "risks": ["iPhone China revenue -19% YoY trend", "Services growth decelerated 8pts in 4 quarters", "App Store DMA pressure"],
 "downside_targets": ["$160 if iPhone -6% YoY", "$140 if Services growth slips below 8%"],
 "refutation_of_bull": "Bulls cite Services as growing >14% — but two-year stack shows clear deceleration that the bull narrative ignores.",
 "conviction": 0.65}"""


class Bear_Agent(GroqModel):
  response_schema = BearCase
  MAX_OUTPUT_TOKENS = 1536

  def __init__(self, model_name: str = "llama-3.3-70b-versatile"):
    super().__init__(model_name=model_name)

  def argue(self,
            analyst_report_md: str,
            variables: Optional[Dict[str, Any]] = None,
            model_outputs: Optional[Dict[str, Any]] = None,
            ticker: str = "") -> Optional[BearCase]:
    self.conversatoin_history = []
    flat_vars = {k: v for k, v in (variables or {}).items() if '.' not in k}
    prompt = (
      f"TICKER: {ticker}\n\n"
      f"ANALYST DRAFT REPORT (your starting point — do not just restate it):\n"
      f"{analyst_report_md[:8000]}\n\n"
      f"GATHERED VARIABLES:\n"
      f"{json.dumps(flat_vars, default=str)[:3500]}\n\n"
      f"MODEL OUTPUTS:\n"
      f"{json.dumps(model_outputs or {}, default=str)[:3000]}\n\n"
      "Build the strongest bear case using these facts."
    )
    try:
      resp = self.generate_response(prompt=prompt, system_prompt=SYSTEM_PROMPT)
      return self.parse_response(resp)
    except Exception as e:
      print(f"[Bear] parse failed: {e}", file=sys.stderr, flush=True)
      return None
