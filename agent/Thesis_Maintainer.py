"""Evaluate whether a new material event breaks, weakens, or leaves intact an
existing thesis on a ticker.

Triggered from the event router (Phase 2 wiring) when a material event arrives
for a ticker that has an active thesis. Output drives:
- alert severity (intact -> none, weakened -> amber, broken -> red)
- whether to trigger a full re-analysis (recommended_action='trigger_reanalysis')

Uses Groq Llama 70B for fast, cheap structured reasoning.
"""
from .groq_template import GroqModel
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import json
import sys


class ThesisVerdict(BaseModel):
  status: str = Field(description="One of: intact, weakened, broken")
  reasoning: str = Field(description="<= 400 chars why the event affects (or doesn't) the thesis")
  affected_assumptions: List[str] = Field(
    default_factory=list,
    description="Which of the thesis's key_assumptions are now in question"
  )
  recommended_action: str = Field(
    description="One of: no_action, alert_only, trigger_reanalysis"
  )
  severity: str = Field(
    description="Alert severity for downstream routing: intact|weakened|broken"
  )


SYSTEM_PROMPT = """You are a financial analyst evaluating whether a new news
event affects an existing investment thesis. You will be given:
  - An existing thesis on a ticker (recommendation, signal, key assumptions)
  - A new material event (headline, body, classification)

Decide whether the event:
  - INTACT: doesn't move the thesis; ignore. action=no_action
  - WEAKENED: chips at one or two assumptions but the thesis still holds.
    action=alert_only
  - BROKEN: invalidates a core assumption or makes the recommendation
    obviously wrong. action=trigger_reanalysis

DECISION GUIDE:
  - A bullish thesis BUY met with a meaningful earnings miss -> broken
  - A bullish thesis BUY with an unrelated CEO comment -> intact
  - A bullish thesis BUY with negative macro that hits the sector -> weakened
  - A bearish thesis SELL met with an earnings beat -> broken
  - Use the thesis's own key_assumptions list — name which ones are affected

OUTPUT: ONLY a JSON object matching the schema. Severity must equal status.

EXAMPLE OUTPUT:
{"status": "broken", "reasoning": "Earnings miss + Q3 guide down directly contradicts the bullish growth assumption underpinning the BUY.",
 "affected_assumptions": ["iPhone revenue 10%+ growth", "Services margin expansion"],
 "recommended_action": "trigger_reanalysis", "severity": "broken"}"""


class Thesis_Maintainer(GroqModel):
  response_schema = ThesisVerdict
  MAX_OUTPUT_TOKENS = 1024

  def __init__(self, model_name: str = "llama-3.3-70b-versatile"):
    super().__init__(model_name=model_name)

  def evaluate(self, thesis: Dict[str, Any], event: Dict[str, Any]) -> Optional[ThesisVerdict]:
    """Given an active thesis dict and an event dict, return a verdict."""
    self.conversatoin_history = []
    thesis_block = (
      f"Ticker: {thesis.get('ticker')}\n"
      f"Recommendation: {thesis.get('recommendation')}\n"
      f"Signal: {thesis.get('signal')}\n"
      f"Confidence: {thesis.get('confidence')}\n"
      f"Target price: {thesis.get('target_price')}\n"
      f"Stop loss: {thesis.get('stop_loss')}\n"
      f"Summary: {thesis.get('analysis_summary')}\n"
      f"Key assumptions: {json.dumps(thesis.get('key_assumptions', []))[:1500]}\n"
      f"Data gaps: {json.dumps(thesis.get('data_gaps', []))[:600]}\n"
      f"Thesis date: {thesis.get('thesis_date')}"
    )
    event_block = (
      f"Headline: {event.get('headline')}\n"
      f"Source: {event.get('source')}\n"
      f"Published: {event.get('published_at')}\n"
      f"Materiality: {event.get('materiality')} ({event.get('category')})\n"
      f"Direction: {event.get('directional_signal')}, urgency {event.get('urgency')}\n"
      f"Body: {(event.get('body') or '')[:1500]}\n"
      f"Classifier reason: {event.get('classifier_reason')}"
    )
    prompt = (
      "EXISTING THESIS:\n"
      f"{thesis_block}\n\n"
      "NEW MATERIAL EVENT:\n"
      f"{event_block}\n\n"
      "Evaluate the impact of this event on the thesis."
    )
    try:
      response = self.generate_response(prompt=prompt, system_prompt=SYSTEM_PROMPT)
      verdict = self.parse_response(response)
      # Coerce severity to match status if model put something else
      if verdict.severity not in ('intact', 'weakened', 'broken'):
        verdict.severity = verdict.status
      return verdict
    except Exception as e:
      print(f"[ThesisMaintainer] failed for {thesis.get('ticker')}: {e}",
            file=sys.stderr, flush=True)
      return None
