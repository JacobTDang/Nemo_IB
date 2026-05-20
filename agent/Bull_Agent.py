"""Bull-side advocate. Reads the analyst's report and the data, argues the
strongest upside case. Single-sided — its job is NOT to be balanced.

Pairs with Bear_Agent and Arbiter_Agent for adversarial debate before the
final recommendation is persisted as a thesis.
"""
from .groq_template import GroqModel
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import json
import sys


class BullCase(BaseModel):
  thesis: str = Field(description="1-paragraph max-bull thesis. Conviction language OK.")
  catalysts: List[str] = Field(
    default_factory=list,
    description="Specific upcoming or near-term events that could trigger upside."
  )
  upside_targets: List[str] = Field(
    default_factory=list,
    description="Price/valuation targets with conditions, e.g. '$200 if Services hits $30B'."
  )
  refutation_of_bear: str = Field(
    description="Where a bear case would be wrong. Pre-empt the bear before they speak."
  )
  conviction: float = Field(description="0-1 self-rated conviction in this bull case.")


SYSTEM_PROMPT = """You are a senior buy-side analyst tasked with making the
STRONGEST CASE for owning this stock. You are not neutral. Your job is to
identify and articulate every reason this stock could go significantly up,
using the data provided.

GROUND RULES:
  - Use only facts present in the provided context (variables, model outputs,
    analyst report). Do NOT invent numbers.
  - Identify specific catalysts with timing. Vague "growth potential" is weak;
    "Q3 earnings on Aug 1 with Services likely to print +14% YoY" is strong.
  - CATALYST DATES MUST come from the CATALYST CALENDAR below. Do NOT invent
    earnings dates, FOMC dates, ex-dividend dates, or anything else dated.
    If the calendar is empty, refer to events by name without specific dates.
  - Set upside targets with explicit conditions. "Stock to $250 IF iPhone
    units stabilize" beats "stock could go higher."
  - Preemptively refute the bear case. What would a smart short-seller argue
    against this stock? Why are they wrong?
  - High conviction (>0.7) requires concrete evidence; reserve 0.9+ for
    overwhelming cases.

OUTPUT: ONLY a JSON object matching the schema.

EXAMPLE OUTPUT:
{"thesis": "Services revenue inflection + iPhone refresh super-cycle creates 25% upside in 18 months.",
 "catalysts": ["Q3 earnings Aug 1 — Services growth >14% likely", "iPhone 17 launch September"],
 "upside_targets": ["$210 if Services >$28B run rate", "$240 if iPhone units +4% YoY"],
 "refutation_of_bear": "Bears point to China weakness, but Services revenue is now larger and growing faster than the China iPhone unit drag.",
 "conviction": 0.75}"""


class Bull_Agent(GroqModel):
  response_schema = BullCase
  MAX_OUTPUT_TOKENS = 1536

  def __init__(self, model_name: str = "llama-3.3-70b-versatile"):
    super().__init__(model_name=model_name)

  @staticmethod
  def _format_calendar(catalysts: Optional[List[Dict[str, Any]]]) -> str:
    """Render the catalyst calendar inline. Empty list -> explicit empty note
    so the model knows it must not invent dates."""
    if not catalysts:
      return ("(no upcoming dated catalysts available — do NOT invent specific "
              "dates; refer to events by name only)")
    lines = []
    for c in catalysts[:25]:
      d = c.get('date', '')
      typ = c.get('type', '')
      desc = c.get('description', '')
      lines.append(f"  {d} | {typ} | {desc}")
    return "\n".join(lines)

  def argue(self,
            analyst_report_md: str,
            variables: Optional[Dict[str, Any]] = None,
            model_outputs: Optional[Dict[str, Any]] = None,
            ticker: str = "",
            catalysts: Optional[List[Dict[str, Any]]] = None) -> Optional[BullCase]:
    self.conversatoin_history = []
    flat_vars = {k: v for k, v in (variables or {}).items() if '.' not in k}
    cal_block = self._format_calendar(catalysts)
    prompt = (
      f"TICKER: {ticker}\n\n"
      f"ANALYST DRAFT REPORT (your starting point — do not just restate it):\n"
      f"{analyst_report_md[:8000]}\n\n"
      f"GATHERED VARIABLES (facts you may cite):\n"
      f"{json.dumps(flat_vars, default=str)[:3500]}\n\n"
      f"MODEL OUTPUTS (DCF/scenario/LBO/credit summaries):\n"
      f"{json.dumps(model_outputs or {}, default=str)[:3000]}\n\n"
      f"CATALYST CALENDAR — the ONLY dated events you may cite as catalysts:\n"
      f"{cal_block}\n\n"
      "Build the strongest bull case using these facts."
    )
    try:
      resp = self.generate_response(prompt=prompt, system_prompt=SYSTEM_PROMPT)
      return self.parse_response(resp)
    except Exception as e:
      print(f"[Bull] parse failed: {e}", file=sys.stderr, flush=True)
      return None
