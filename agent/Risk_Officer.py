"""Deterministic risk officer with HARD-CODED limits.

No LLM. Decisions are reproducible. Risk officer has veto power on every
trade — even high-conviction arbiter recommendations are blocked if they
violate any limit below.

Limit table is intentionally conservative. Tune via class attributes only
after backtest expectancy is positive over >50 trades.
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from agent.Arbiter_Agent import ArbiterVerdict


@dataclass
class RiskDecision:
  approve: bool
  reasons: List[str]
  adjusted_quantity: Optional[float] = None
  adjusted_dollar_size: Optional[float] = None


class Risk_Officer:
  # --- hard limits ---
  MAX_POSITION_PCT          = 0.05  # 5% of starting portfolio per name
  MAX_SECTOR_PCT            = 0.30  # 30% per sector (Phase 9 — placeholder)
  MAX_DAILY_LOSS_PCT        = 0.02  # halt new positions if down 2% today
  MAX_NEW_POSITIONS_PER_DAY = 3
  MIN_CONFIDENCE_TO_TRADE   = 0.65  # arbiter confidence floor
  REQUIRE_DEBATE_SPREAD     = 0.20  # |bull - bear| spread must exceed this

  def __init__(self,
               max_position_pct: Optional[float] = None,
               min_confidence: Optional[float] = None,
               max_new_per_day: Optional[int] = None):
    if max_position_pct is not None:
      self.MAX_POSITION_PCT = max_position_pct
    if min_confidence is not None:
      self.MIN_CONFIDENCE_TO_TRADE = min_confidence
    if max_new_per_day is not None:
      self.MAX_NEW_POSITIONS_PER_DAY = max_new_per_day

  def evaluate(self,
               proposed_quantity: float,
               proposed_price: float,
               arbiter_verdict: ArbiterVerdict,
               portfolio: Dict[str, Any]) -> RiskDecision:
    """Returns approve=True or False with reasons.

    On approve, may return adjusted_quantity if size was capped by limits.
    """
    reasons: List[str] = []

    # 1. Arbiter must recommend a directional action
    if arbiter_verdict.final_recommendation not in ('BUY', 'SELL'):
      return RiskDecision(approve=False, reasons=[
        f"arbiter recommendation is {arbiter_verdict.final_recommendation}; "
        "no trade for HOLD/NEUTRAL"
      ])

    # 2. Confidence floor
    if arbiter_verdict.confidence < self.MIN_CONFIDENCE_TO_TRADE:
      return RiskDecision(approve=False, reasons=[
        f"arbiter confidence {arbiter_verdict.confidence:.2f} < threshold "
        f"{self.MIN_CONFIDENCE_TO_TRADE}"
      ])

    # 3. Debate spread — close debates should not result in trades
    spread = abs(arbiter_verdict.bull_strength - arbiter_verdict.bear_strength)
    if spread < self.REQUIRE_DEBATE_SPREAD:
      return RiskDecision(approve=False, reasons=[
        f"bull/bear strength spread {spread:.2f} < required {self.REQUIRE_DEBATE_SPREAD}; "
        "debate too close to act"
      ])

    # 4. Sizing guidance from arbiter is honored
    if arbiter_verdict.position_sizing_guidance == 'no_position':
      return RiskDecision(approve=False, reasons=[
        "arbiter explicit sizing guidance = no_position"
      ])

    # 5. Daily loss limit
    daily_pnl_pct = portfolio.get('daily_pnl_pct', 0)
    if daily_pnl_pct < -self.MAX_DAILY_LOSS_PCT:
      return RiskDecision(approve=False, reasons=[
        f"daily P&L {daily_pnl_pct*100:.2f}% breaches loss limit "
        f"-{self.MAX_DAILY_LOSS_PCT*100:.1f}%; no new positions today"
      ])

    # 6. Position count limit
    if portfolio.get('positions_opened_today', 0) >= self.MAX_NEW_POSITIONS_PER_DAY:
      return RiskDecision(approve=False, reasons=[
        f"{self.MAX_NEW_POSITIONS_PER_DAY} new positions already opened today"
      ])

    # 7. Position size cap (5% of starting portfolio)
    starting_value = portfolio.get('starting_value', 100_000.0)
    max_dollar = starting_value * self.MAX_POSITION_PCT
    requested_dollar = proposed_quantity * proposed_price
    adjusted_qty: Optional[float] = None
    adjusted_dollar: Optional[float] = None

    if requested_dollar > max_dollar:
      adjusted_qty = max(1.0, max_dollar / proposed_price) if proposed_price > 0 else 0
      adjusted_dollar = adjusted_qty * proposed_price
      reasons.append(
        f"size capped from ${requested_dollar:.0f} to ${adjusted_dollar:.0f} "
        f"({self.MAX_POSITION_PCT*100:.0f}% of starting capital)"
      )

    # 8. Aggressive sizing requires high confidence
    if (arbiter_verdict.position_sizing_guidance == 'aggressive'
        and arbiter_verdict.confidence < 0.75):
      reasons.append("aggressive sizing requested but confidence <0.75; "
                     "downgrading to normal sizing")
      # Cap to 50% of approved size
      if adjusted_qty is not None:
        adjusted_qty = adjusted_qty * 0.5
      else:
        adjusted_qty = proposed_quantity * 0.5
      adjusted_dollar = adjusted_qty * proposed_price

    # 9. Cautious sizing applies a 50% reduction
    if arbiter_verdict.position_sizing_guidance == 'cautious':
      reasons.append("cautious sizing -> 50% of approved size")
      base_qty = adjusted_qty if adjusted_qty is not None else proposed_quantity
      adjusted_qty = base_qty * 0.5
      adjusted_dollar = adjusted_qty * proposed_price

    return RiskDecision(
      approve=True, reasons=reasons,
      adjusted_quantity=adjusted_qty,
      adjusted_dollar_size=adjusted_dollar,
    )
