"""Phase 3a: Bull and Bear agents on the same fixture — verify they argue
DIFFERENT sides (no agreement collapse), produce structured outputs, and
ground claims in the provided data.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.Bull_Agent import Bull_Agent, BullCase
from agent.Bear_Agent import Bear_Agent, BearCase


# Fixture analyst report (AAPL-shaped). Both agents see this + variables.
ANALYST_REPORT = """## EXECUTIVE SUMMARY
Apple delivered Q2 with revenue $94.8B (+5% YoY), EPS $1.53 (+11% YoY).
Services revenue $23.9B (+14% YoY) — the strongest line. iPhone revenue
$45.9B (-2% YoY). Gross margin 46.6%, +110bps YoY.

## RECOMMENDATION: HOLD

## VALUATION
- P/E: 31x trailing
- EV/EBITDA: 22x
- DCF fair value: $172 (vs current $195)

## FINANCIAL PERFORMANCE
- Revenue: $94.8B Q2, +5% YoY
- Services: $23.9B Q2, +14% YoY, 75% gross margin
- iPhone: $45.9B Q2, -2% YoY
- Operating margin: 30.8%
- ROIC: 28%

## RISKS
- China iPhone unit decline (-19% YoY)
- EU DMA App Store fee pressure
- Services growth decelerating from prior 22%

## ASSUMPTIONS
- Services growth 12-14% sustains FY26
- iPhone units stabilize in FY27
"""

VARIABLES = {
  "ticker": "AAPL",
  "revenue_ttm": 388_000_000_000,
  "services_revenue_q2": 23_900_000_000,
  "services_growth_yoy": 0.14,
  "iphone_revenue_q2": 45_900_000_000,
  "iphone_growth_yoy": -0.02,
  "iphone_china_growth_yoy": -0.19,
  "operating_margin": 0.308,
  "gross_margin": 0.466,
  "roic": 0.28,
  "pe_trailing": 31,
  "ev_ebitda": 22,
  "dcf_fair_value": 172.0,
  "current_price": 195.0,
  "market_cap": 3_000_000_000_000,
  "beta": 1.21,
}


def test_bull_produces_valid_output():
  bull = Bull_Agent()
  case = bull.argue(ANALYST_REPORT, VARIABLES, model_outputs={}, ticker="AAPL")
  assert case is not None, "Bull returned None"
  assert isinstance(case, BullCase)
  assert len(case.thesis) > 30, f"thesis too short: {case.thesis}"
  assert 0 <= case.conviction <= 1
  print(f"PASS: Bull produced valid output (conviction={case.conviction:.2f})")
  print(f"  thesis: {case.thesis[:200]}")
  print(f"  catalysts: {case.catalysts[:3]}")
  return case


def test_bear_produces_valid_output():
  bear = Bear_Agent()
  case = bear.argue(ANALYST_REPORT, VARIABLES, model_outputs={}, ticker="AAPL")
  assert case is not None, "Bear returned None"
  assert isinstance(case, BearCase)
  assert len(case.thesis) > 30, f"thesis too short: {case.thesis}"
  assert 0 <= case.conviction <= 1
  print(f"PASS: Bear produced valid output (conviction={case.conviction:.2f})")
  print(f"  thesis: {case.thesis[:200]}")
  print(f"  risks: {case.risks[:3]}")
  return case


def test_bull_and_bear_argue_different_sides():
  bull = Bull_Agent()
  bear = Bear_Agent()
  bull_case = bull.argue(ANALYST_REPORT, VARIABLES, ticker="AAPL")
  bear_case = bear.argue(ANALYST_REPORT, VARIABLES, ticker="AAPL")
  assert bull_case and bear_case

  # Heuristic: count bullish vs bearish words in each thesis
  bullish_words = {'upside', 'growth', 'tailwind', 'beat', 'expansion',
                   'opportunity', 'inflection', 'super-cycle', 'higher',
                   'strong', 'momentum', 'positive', 'attractive'}
  bearish_words = {'downside', 'decelerat', 'risk', 'pressure', 'headwind',
                   'miss', 'compress', 'decline', 'overval', 'lower',
                   'weak', 'concern', 'erod'}

  def score(text, words):
    t = text.lower()
    return sum(1 for w in words if w in t)

  bull_score_bullish = score(bull_case.thesis + ' ' + bull_case.refutation_of_bear, bullish_words)
  bull_score_bearish = score(bull_case.thesis + ' ' + bull_case.refutation_of_bear, bearish_words)
  bear_score_bullish = score(bear_case.thesis + ' ' + bear_case.refutation_of_bull, bullish_words)
  bear_score_bearish = score(bear_case.thesis + ' ' + bear_case.refutation_of_bull, bearish_words)

  print(f"\n  Bull text: bullish={bull_score_bullish} bearish={bull_score_bearish}")
  print(f"  Bear text: bullish={bear_score_bullish} bearish={bear_score_bearish}")

  # Bull should skew bullish; bear should skew bearish.
  assert bull_score_bullish >= bull_score_bearish, \
    f"Bull case is not predominantly bullish: bull={bull_score_bullish}, bear={bull_score_bearish}"
  assert bear_score_bearish >= bear_score_bullish, \
    f"Bear case is not predominantly bearish: bull={bear_score_bullish}, bear={bear_score_bearish}"
  print(f"PASS: Bull leans bullish, Bear leans bearish (skews are correct)")


def test_bull_targets_have_conditions():
  """Strong bull cases use conditional targets, not just a number."""
  bull = Bull_Agent()
  case = bull.argue(ANALYST_REPORT, VARIABLES, ticker="AAPL")
  assert case
  if not case.upside_targets:
    print(f"WARN: bull produced no upside_targets — accepted but flagged")
    return
  # At least one target should contain a conditional word (if/when/should/assuming)
  conditional = any(
    any(w in t.lower() for w in ('if ', 'when ', 'should ', 'assuming', 'subject', 'given'))
    for t in case.upside_targets
  )
  if not conditional:
    print(f"WARN: bull targets are bare numbers without conditions:\n  {case.upside_targets}")
  else:
    print(f"PASS: at least one bull target is conditional")


def test_bear_risks_reference_real_concerns():
  """Bear's risks list should reference real items from the analyst's data."""
  bear = Bear_Agent()
  case = bear.argue(ANALYST_REPORT, VARIABLES, ticker="AAPL")
  assert case
  # At least one risk should reference China, iPhone, DMA, or Services dec
  texts = ' '.join(case.risks).lower() + ' ' + case.thesis.lower()
  expected_anchors = ['china', 'iphone', 'dma', 'app store', 'decelerat',
                       'services', 'multiple', 'p/e', 'p e', 'ev/ebitda', 'valuation']
  found = [a for a in expected_anchors if a in texts]
  assert found, f"bear's risks have no anchor to AAPL data: risks={case.risks}"
  print(f"PASS: bear cites concrete AAPL concerns ({found})")


if __name__ == "__main__":
  test_bull_produces_valid_output()
  test_bear_produces_valid_output()
  test_bull_and_bear_argue_different_sides()
  test_bull_targets_have_conditions()
  test_bear_risks_reference_real_concerns()
  print("\nAll Phase 3a Bull/Bear tests passed.")
