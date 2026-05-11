"""Pre-mortem agent.

Imagine it's 6 months from now and this recommendation lost 30%. What happened?
Forces the system to enumerate specific failure modes BEFORE committing capital.

Output is attached to each thesis and surfaced on the dashboard ticker page.
"""
from .groq_template import GroqModel
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import json
import sys


class PreMortemReport(BaseModel):
  failure_modes: List[str] = Field(
    description="3-5 specific scenarios that would lead to a 30% loss in 6 months. "
                "Each must reference a concrete thing (a number, an event, a competitor)."
  )
  early_warnings: List[str] = Field(
    description="Specific observable signals that any of those failures is in progress."
  )
  hedge_or_exit: List[str] = Field(
    description="Concrete actions to take when early warnings fire."
  )
  worst_case_loss_pct: float = Field(
    description="Estimated downside in the worst plausible 6-month scenario, as a percent."
  )


SYSTEM_PROMPT = """You are a contrarian risk officer. The team has reached
a recommendation. Your job: assume the trade WILL go badly. Imagine it lost
30% over the next 6 months. Enumerate the SPECIFIC, CONCRETE scenarios that
caused it — not generic 'macro headwinds', but 'Services growth fell from
14% to 6% because EU forced App Store fee cuts'.

GROUND RULES:
  - Use facts from the provided thesis and data. Don't invent companies or
    events.
  - 3-5 failure modes, each one paragraph or less.
  - Early warnings must be OBSERVABLE — a metric in a future earnings report,
    a regulatory filing, a competitor announcement.
  - Hedge/exit actions must be ACTIONABLE — "reduce position 50% if Services
    growth prints below 10%" is good; "monitor closely" is not.
  - worst_case_loss_pct is your honest worst-plausible-case, not click-bait.

OUTPUT: ONLY a JSON object matching the schema.

EXAMPLE OUTPUT:
{"failure_modes": [
   "EU DMA enforcement cuts App Store revenue 40%, dropping Services growth to single digits in 2 quarters.",
   "China iPhone unit decline accelerates from -19% to -35% YoY as Huawei regains premium share.",
   "Recession forces capex cuts at hyperscalers, reducing Services demand for AI infrastructure tie-ins."
 ],
 "early_warnings": [
   "Services revenue growth prints below 10% in any quarter",
   "iPhone China unit decline > 25% YoY",
   "Two consecutive hyperscalers cut capex guidance"
 ],
 "hedge_or_exit": [
   "Trim 50% on first Services growth print < 10%",
   "Full exit if China iPhone declines > 30% in any single quarter"
 ],
 "worst_case_loss_pct": 28.0}"""


class Pre_Mortem_Agent(GroqModel):
  response_schema = PreMortemReport
  MAX_OUTPUT_TOKENS = 1536

  def __init__(self, model_name: str = "llama-3.3-70b-versatile"):
    super().__init__(model_name=model_name)

  def envision(self,
               ticker: str,
               recommendation: str,
               analyst_report_md: str,
               variables: Optional[Dict[str, Any]] = None) -> Optional[PreMortemReport]:
    self.conversatoin_history = []
    flat_vars = {k: v for k, v in (variables or {}).items() if '.' not in k}
    prompt = (
      f"TICKER: {ticker}\n"
      f"RECOMMENDATION: {recommendation}\n\n"
      f"ANALYST REPORT:\n{analyst_report_md[:6000]}\n\n"
      f"GROUNDING DATA:\n{json.dumps(flat_vars, default=str)[:3000]}\n\n"
      "Imagine this trade lost 30% over the next 6 months. "
      "Enumerate the specific failure modes, early warnings, and hedge/exit triggers."
    )
    try:
      resp = self.generate_response(prompt=prompt, system_prompt=SYSTEM_PROMPT)
      return self.parse_response(resp)
    except Exception as e:
      print(f"[PreMortem] failed: {e}", file=sys.stderr, flush=True)
      return None
