"""Classify a single news headline + summary for materiality.

Uses Groq's fast Llama-8B-Instant (cheap; ~50 tokens/sec) since this fires
hundreds of times per day from the news watcher daemon. Outputs structured
JSON so downstream handlers can route by category/urgency/signal.
"""
from .groq_template import GroqModel
from pydantic import BaseModel, Field
from typing import List, Optional
import sys


class MaterialityResult(BaseModel):
  is_material: bool = Field(description="True if this article warrants further attention.")
  category: str = Field(description="One of: earnings, m_and_a, regulatory, exec_change, product_launch, supply_chain, macro, legal, sector, other")
  affected_tickers: List[str] = Field(default_factory=list, description="Ticker symbols mentioned that could move on this news. Empty if unclear.")
  primary_ticker: Optional[str] = Field(default=None, description="The single most-affected ticker, if any.")
  directional_signal: str = Field(description="One of: bullish, bearish, neutral, mixed")
  urgency: str = Field(description="One of: immediate (intraday move), hours, days, watch (no near-term move)")
  confidence: float = Field(description="0.0 to 1.0 confidence in this classification.")
  one_line_reason: str = Field(description="<= 200 chars explaining why this is/isn't material.")


SYSTEM_PROMPT = """You are a financial news triage classifier. You read a single
news headline + body and decide whether it's material for equity investors.

MATERIALITY HEURISTICS:
- material=true for: earnings beats/misses, M&A, FDA decisions, regulatory action,
  C-suite changes, major product launches, supply chain disruptions, macro
  catalysts (Fed, jobs, CPI), legal verdicts, dividend/buyback changes
- material=false for: routine coverage, opinion pieces without new facts,
  product reviews, "stock up X% today" without a catalyst, paywalled previews

CATEGORY (pick ONE):
- earnings: earnings releases, guidance, pre-announcements
- m_and_a: deals, takeovers, divestitures
- regulatory: SEC actions, FDA approvals, antitrust
- exec_change: CEO/CFO/board changes
- product_launch: new products, major releases
- supply_chain: shortages, factory issues, logistics
- macro: Fed, CPI, NFP, GDP, rates
- legal: court rulings, settlements, class actions
- sector: industry-wide news affecting a group of tickers
- other: anything else material

URGENCY:
- immediate: stock will move today/tonight (earnings drop, M&A announcement)
- hours: priced in within hours (Fed decision, FDA approval)
- days: priced in over days (analyst upgrade waves, sector rotation)
- watch: long-tail story worth tracking but no immediate move

DIRECTIONAL_SIGNAL:
- bullish for the named ticker(s); bearish; neutral; mixed (some up, some down)

TICKERS:
- Extract ALL US-listed ticker symbols mentioned (case-insensitive match)
- primary_ticker = the one most affected; null if no clear primary
- DO NOT invent tickers not present in the text

OUTPUT: ONLY a JSON OBJECT (not a schema definition). No explanation outside the JSON.
Do NOT include "properties", "$defs", "title", or "type" keys. Return ONLY the
classification fields.

EXAMPLE OUTPUT (this is the shape you must return, not a schema):
{"is_material": true, "category": "earnings", "affected_tickers": ["AAPL"],
 "primary_ticker": "AAPL", "directional_signal": "bullish", "urgency": "immediate",
 "confidence": 0.92, "one_line_reason": "EPS beat by 7% with strong iPhone growth."}"""


class Materiality_Classifier(GroqModel):
  response_schema = MaterialityResult
  MAX_OUTPUT_TOKENS = 512

  def __init__(self, model_name: str = "llama-3.3-70b-versatile"):
    # 70B-versatile is reliable for structured output. 8B-instant echoed the
    # schema definition back instead of producing data, costing us correctness.
    # On Groq free tier 70B is still fast (~600 tok/s) and free for our volume.
    super().__init__(model_name=model_name)

  def classify(self, headline: str, summary: str = "", source: str = "") -> Optional[MaterialityResult]:
    """Return structured classification, or None on parse failure."""
    self.conversatoin_history = []
    user_prompt = (
      f"HEADLINE: {headline}\n"
      f"SOURCE: {source}\n"
      f"SUMMARY: {summary[:1500] if summary else '(none)'}\n\n"
      "Classify this article."
    )
    try:
      response = self.generate_response(prompt=user_prompt, system_prompt=SYSTEM_PROMPT)
      result = self.parse_response(response)
      return result
    except Exception as e:
      print(f"[Materiality] parse failed for headline '{headline[:60]}': {e}",
            file=sys.stderr, flush=True)
      return None
