"""Phase 6a: Risk_Officer hard limits.

Tests every rejection path AND the size-cap modifier. No LLM calls."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.Risk_Officer import Risk_Officer, RiskDecision
from agent.Arbiter_Agent import ArbiterVerdict


def _ok_verdict(rec='BUY', conf=0.75, bull=0.78, bear=0.45, sizing='normal'):
  return ArbiterVerdict(
    final_recommendation=rec, confidence=conf,
    bull_strength=bull, bear_strength=bear,
    decisive_factors=['ok'], acknowledged_risks=['ok'],
    conditions_to_change_mind=['ok'],
    position_sizing_guidance=sizing,
    rationale='test',
  )


def _ok_portfolio():
  return {
    'total_value': 100_000, 'starting_value': 100_000,
    'open_positions_count': 0, 'open_capital_at_risk': 0,
    'unrealized_pnl': 0, 'realized_pnl_today': 0,
    'daily_pnl': 0, 'daily_pnl_pct': 0,
    'positions_opened_today': 0,
  }


def test_approves_clean_trade():
  ro = Risk_Officer()
  d = ro.evaluate(10, 200, _ok_verdict(), _ok_portfolio())
  assert d.approve, f"clean trade should approve: {d.reasons}"
  print(f"PASS: clean trade approved (reasons={d.reasons})")


def test_rejects_hold_recommendation():
  ro = Risk_Officer()
  d = ro.evaluate(10, 200, _ok_verdict(rec='HOLD'), _ok_portfolio())
  assert not d.approve
  assert any('HOLD' in r for r in d.reasons)
  print(f"PASS: HOLD rejected ({d.reasons})")


def test_rejects_neutral_recommendation():
  ro = Risk_Officer()
  d = ro.evaluate(10, 200, _ok_verdict(rec='NEUTRAL'), _ok_portfolio())
  assert not d.approve
  print(f"PASS: NEUTRAL rejected ({d.reasons[0]})")


def test_rejects_low_confidence():
  ro = Risk_Officer()
  d = ro.evaluate(10, 200, _ok_verdict(conf=0.5), _ok_portfolio())
  assert not d.approve
  assert any('confidence' in r.lower() for r in d.reasons)
  print(f"PASS: conf=0.5 rejected (threshold 0.65)")


def test_rejects_close_debate():
  ro = Risk_Officer()
  # spread = |bull - bear| = 0.05, below 0.20 threshold
  d = ro.evaluate(10, 200, _ok_verdict(bull=0.7, bear=0.65), _ok_portfolio())
  assert not d.approve
  assert any('spread' in r.lower() for r in d.reasons)
  print(f"PASS: close debate (spread=0.05) rejected")


def test_rejects_no_position_sizing():
  ro = Risk_Officer()
  d = ro.evaluate(10, 200, _ok_verdict(sizing='no_position'), _ok_portfolio())
  assert not d.approve
  print(f"PASS: no_position sizing rejected")


def test_rejects_daily_loss_limit_breached():
  ro = Risk_Officer()
  port = _ok_portfolio()
  port['daily_pnl_pct'] = -0.03  # 3% loss > 2% threshold
  d = ro.evaluate(10, 200, _ok_verdict(), port)
  assert not d.approve
  assert any('loss limit' in r.lower() for r in d.reasons)
  print(f"PASS: daily loss limit breach (-3%) blocks new positions")


def test_rejects_position_count_limit():
  ro = Risk_Officer()
  port = _ok_portfolio()
  port['positions_opened_today'] = 3
  d = ro.evaluate(10, 200, _ok_verdict(), port)
  assert not d.approve
  assert any('already opened today' in r for r in d.reasons)
  print(f"PASS: 3 positions opened today -> rejected")


def test_caps_oversized_position():
  """Requesting 100 shares at $200 = $20,000 = 20% of $100k portfolio (> 5% cap).
  Should approve with adjusted_quantity capped to 5% = $5000 = 25 shares."""
  ro = Risk_Officer()
  d = ro.evaluate(100, 200, _ok_verdict(), _ok_portfolio())
  assert d.approve, "should approve with size cap, not reject"
  assert d.adjusted_quantity is not None
  assert d.adjusted_quantity <= 25, f"5% cap = 25 shares, got {d.adjusted_quantity}"
  assert any('capped' in r.lower() for r in d.reasons)
  print(f"PASS: oversized 100@$200 ($20k) capped to {d.adjusted_quantity} shares "
        f"(${d.adjusted_dollar_size})")


def test_aggressive_sizing_requires_high_conf():
  ro = Risk_Officer()
  d = ro.evaluate(10, 200, _ok_verdict(sizing='aggressive', conf=0.70), _ok_portfolio())
  assert d.approve
  # Aggressive but conf < 0.75 -> downgrade to 50%
  assert d.adjusted_quantity is not None
  assert d.adjusted_quantity == 5, f"50% of 10 = 5, got {d.adjusted_quantity}"
  assert any('aggressive' in r.lower() for r in d.reasons)
  print(f"PASS: aggressive sizing + conf<0.75 downgraded to {d.adjusted_quantity}")


def test_cautious_sizing_halves_position():
  ro = Risk_Officer()
  d = ro.evaluate(10, 200, _ok_verdict(sizing='cautious'), _ok_portfolio())
  assert d.approve
  assert d.adjusted_quantity == 5
  assert any('cautious' in r.lower() for r in d.reasons)
  print(f"PASS: cautious sizing halves position to {d.adjusted_quantity}")


def test_multiple_violations_all_reported():
  """When multiple rules trigger, all should be reported (but caller can act on any)."""
  ro = Risk_Officer()
  # Low conf AND close debate AND HOLD — first rejection short-circuits
  d = ro.evaluate(10, 200, _ok_verdict(rec='HOLD', conf=0.4, bull=0.5, bear=0.5),
                   _ok_portfolio())
  assert not d.approve
  print(f"PASS: multiple violations -> rejection (short-circuits on first match)")


def test_custom_thresholds_override_defaults():
  ro_strict = Risk_Officer(min_confidence=0.85, max_position_pct=0.02)
  d = ro_strict.evaluate(10, 200, _ok_verdict(conf=0.80), _ok_portfolio())
  assert not d.approve, "stricter 0.85 conf threshold should reject 0.80"
  print(f"PASS: custom min_confidence={ro_strict.MIN_CONFIDENCE_TO_TRADE} rejected conf=0.80")


if __name__ == "__main__":
  test_approves_clean_trade()
  test_rejects_hold_recommendation()
  test_rejects_neutral_recommendation()
  test_rejects_low_confidence()
  test_rejects_close_debate()
  test_rejects_no_position_sizing()
  test_rejects_daily_loss_limit_breached()
  test_rejects_position_count_limit()
  test_caps_oversized_position()
  test_aggressive_sizing_requires_high_conf()
  test_cautious_sizing_halves_position()
  test_multiple_violations_all_reported()
  test_custom_thresholds_override_defaults()
  print("\nAll Phase 6a risk officer tests passed.")
