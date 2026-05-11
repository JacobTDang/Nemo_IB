"""Phase 3b: Arbiter agent — given fixture bull+bear, produce a synthesized
verdict that picks the stronger side and articulates decisive factors.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.Bull_Agent import BullCase
from agent.Bear_Agent import BearCase
from agent.Arbiter_Agent import Arbiter_Agent, ArbiterVerdict


VARIABLES = {
  "ticker": "AAPL", "services_growth_yoy": 0.14, "iphone_growth_yoy": -0.02,
  "iphone_china_growth_yoy": -0.19, "operating_margin": 0.308, "roic": 0.28,
  "pe_trailing": 31, "dcf_fair_value": 172.0, "current_price": 195.0,
}
ANALYST_REPORT = "AAPL Q2: rev $94.8B, Services $23.9B (+14%), iPhone -2%. HOLD with caveats."


def _bull_strong():
  return BullCase(
    thesis="Services revenue inflection drives multiple expansion — 18 month target $230 (+18%).",
    catalysts=["Q3 earnings Aug 1 — Services likely >$25B run rate",
                "iPhone 17 launch Sept with India/EM uplift",
                "Buyback authorization $90B remaining"],
    upside_targets=["$210 if Services >$28B Q3 run rate",
                     "$230 if iPhone units stabilize FY27"],
    refutation_of_bear="Bears cite China iPhone -19% but Services is now 25% of total revenue with 75% gross margins — China unit drag is mathematically smaller than Services tailwind.",
    conviction=0.78,
  )


def _bear_weak():
  return BearCase(
    thesis="Stock is overvalued at 31x given growth deceleration.",
    risks=["valuation", "macro risk"],
    downside_targets=["$170 (DCF)"],
    refutation_of_bull="Services growth might slow.",
    conviction=0.45,
  )


def _bull_weak():
  return BullCase(
    thesis="Apple is a great company.",
    catalysts=["future products"],
    upside_targets=["higher"],
    refutation_of_bear="Bears are too negative.",
    conviction=0.5,
  )


def _bear_strong():
  return BearCase(
    thesis="Two-year stack shows clear Services deceleration that contradicts the bull narrative — 20% downside to DCF fair value.",
    risks=["Services growth decelerated from 27% to 14% over 6 quarters",
            "China iPhone unit -19% YoY — structural, not cyclical",
            "EU DMA fee cut on App Store reduces Services margin",
            "31x P/E unjustified at <6% revenue growth"],
    downside_targets=["$172 (DCF fair value)",
                       "$155 if Services slips below 10% growth",
                       "$140 if EU DMA forces 50%+ App Store fee cut"],
    refutation_of_bull="Bulls anchor on +14% Services growth but ignore that this is decelerating fast; on a 2-year stack the trend points to single digits.",
    conviction=0.78,
  )


def test_arbiter_picks_stronger_side_bull():
  """When the bull case is concrete and bear is weak, arbiter should lean bullish."""
  arbiter = Arbiter_Agent()
  verdict = arbiter.judge(ANALYST_REPORT, _bull_strong(), _bear_weak(),
                           variables=VARIABLES, ticker="AAPL")
  assert verdict is not None, "Arbiter returned None"
  assert isinstance(verdict, ArbiterVerdict)
  print(f"  rec={verdict.final_recommendation} conf={verdict.confidence:.2f} "
        f"bull={verdict.bull_strength:.2f} bear={verdict.bear_strength:.2f} "
        f"sizing={verdict.position_sizing_guidance}")
  assert verdict.bull_strength > verdict.bear_strength, \
    f"strong bull + weak bear should yield bull_strength > bear_strength " \
    f"({verdict.bull_strength} vs {verdict.bear_strength})"
  assert verdict.final_recommendation in ('BUY', 'HOLD'), \
    f"strong bull, weak bear should land BUY or HOLD, not {verdict.final_recommendation}"
  print("PASS: strong-bull / weak-bear -> bull wins strength contest")


def test_arbiter_picks_stronger_side_bear():
  """Mirror: concrete bear, weak bull."""
  arbiter = Arbiter_Agent()
  verdict = arbiter.judge(ANALYST_REPORT, _bull_weak(), _bear_strong(),
                           variables=VARIABLES, ticker="AAPL")
  assert verdict is not None
  print(f"  rec={verdict.final_recommendation} conf={verdict.confidence:.2f} "
        f"bull={verdict.bull_strength:.2f} bear={verdict.bear_strength:.2f}")
  assert verdict.bear_strength > verdict.bull_strength, \
    f"weak bull + strong bear should yield bear_strength > bull_strength"
  assert verdict.final_recommendation in ('SELL', 'HOLD'), \
    f"strong bear, weak bull should land SELL or HOLD, not {verdict.final_recommendation}"
  print("PASS: weak-bull / strong-bear -> bear wins strength contest")


def test_arbiter_decisive_factors_are_concrete():
  arbiter = Arbiter_Agent()
  verdict = arbiter.judge(ANALYST_REPORT, _bull_strong(), _bear_strong(),
                           variables=VARIABLES, ticker="AAPL")
  assert verdict is not None
  print(f"  decisive: {verdict.decisive_factors}")
  print(f"  conditions: {verdict.conditions_to_change_mind}")
  assert len(verdict.decisive_factors) >= 1, "must name at least one decisive factor"
  # At least one decisive factor should be > 25 chars (not "yes" or "data")
  meaty = [d for d in verdict.decisive_factors if len(d.strip()) > 25]
  assert meaty, f"decisive_factors too short: {verdict.decisive_factors}"
  print(f"PASS: arbiter names {len(meaty)} substantive decisive factor(s)")


def test_arbiter_close_debate_yields_lower_confidence():
  """When bull and bear are both strong, confidence should be < high-confidence threshold."""
  arbiter = Arbiter_Agent()
  verdict = arbiter.judge(ANALYST_REPORT, _bull_strong(), _bear_strong(),
                           variables=VARIABLES, ticker="AAPL")
  assert verdict is not None
  print(f"  Close debate confidence: {verdict.confidence:.2f}, "
        f"spread={abs(verdict.bull_strength - verdict.bear_strength):.2f}")
  spread = abs(verdict.bull_strength - verdict.bear_strength)
  # If spread is small, confidence should be moderate (< 0.85)
  if spread < 0.2:
    assert verdict.confidence < 0.85, \
      f"Close debate (spread={spread:.2f}) should yield confidence <0.85, got {verdict.confidence}"
    print(f"PASS: close-debate confidence is appropriately calibrated")
  else:
    print(f"SKIP: debate not close (spread={spread:.2f}), can't test calibration")


def test_arbiter_conditions_to_change_mind_present():
  arbiter = Arbiter_Agent()
  verdict = arbiter.judge(ANALYST_REPORT, _bull_strong(), _bear_strong(),
                           variables=VARIABLES, ticker="AAPL")
  assert verdict is not None
  assert len(verdict.conditions_to_change_mind) >= 1, \
    "arbiter must name at least one condition that would change the verdict"
  print(f"PASS: arbiter names {len(verdict.conditions_to_change_mind)} change-mind condition(s)")


def test_arbiter_position_sizing_aligned_with_recommendation():
  """no_position guidance implies HOLD/NEUTRAL recommendation."""
  arbiter = Arbiter_Agent()
  v = arbiter.judge(ANALYST_REPORT, _bull_strong(), _bear_weak(),
                     variables=VARIABLES, ticker="AAPL")
  assert v
  if v.position_sizing_guidance == 'no_position':
    assert v.final_recommendation in ('HOLD', 'NEUTRAL'), \
      f"no_position sizing should pair with HOLD/NEUTRAL, got {v.final_recommendation}"
  if v.position_sizing_guidance == 'aggressive':
    assert v.confidence >= 0.6, f"aggressive sizing should require conf>=0.6, got {v.confidence}"
  print(f"PASS: sizing/recommendation alignment OK ({v.position_sizing_guidance}/{v.final_recommendation})")


if __name__ == "__main__":
  test_arbiter_picks_stronger_side_bull()
  test_arbiter_picks_stronger_side_bear()
  test_arbiter_decisive_factors_are_concrete()
  test_arbiter_close_debate_yields_lower_confidence()
  test_arbiter_conditions_to_change_mind_present()
  test_arbiter_position_sizing_aligned_with_recommendation()
  print("\nAll Phase 3b arbiter tests passed.")
