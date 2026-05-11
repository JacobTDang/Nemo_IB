"""Test Fix #4-7: signal-based modeling prompt expansion (beat rate, curve, NFCI, profile)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.Financial_Modeling_Agent import Financial_Modeling_Agent


def test_user_prompt_includes_new_signals():
  agent = Financial_Modeling_Agent()
  variables = {
    'revenue_base': 400_000_000_000,
    'ebitda_margin': 0.30,
    'earnings_quality.beat_rate_pct': 85.0,
    'yield_curve_shape': 'inverted',
    'macro.NFCI': 0.7,
    'profile.finnhubIndustry': 'Technology',
    'insider_sentiment.signal': 'net_buying',
  }
  prompt = agent._build_user_prompt("Analyze AAPL", variables)

  assert 'earnings_quality.beat_rate_pct: 85.0' in prompt, "beat rate missing from prompt"
  assert 'yield_curve_shape: inverted' in prompt, "yield_curve_shape missing"
  assert 'macro.NFCI: 0.7' in prompt, "NFCI missing"
  assert 'profile.finnhubIndustry: Technology' in prompt, "industry missing"
  assert 'insider_sentiment.signal: net_buying' in prompt, "insider sentiment missing"
  print("PASS: all new signals appear in user prompt")


def test_system_prompt_has_signal_rules():
  agent = Financial_Modeling_Agent()
  system = agent._build_system_prompt("(dummy modeling context)")

  assert 'SIGNAL-BASED ADJUSTMENTS' in system, "signal-based adjustments block missing"
  assert 'beat_rate_pct > 75' in system, "beat-rate-high rule missing"
  assert 'beat_rate_pct < 40' in system, "beat-rate-low rule missing"
  assert 'yield_curve_shape == "inverted"' in system, "yield curve rule missing"
  assert 'macro.NFCI > 0' in system, "NFCI rule missing"
  assert 'insider_sentiment.signal' in system, "insider sentiment rule missing"
  assert 'profile.finnhubIndustry' in system, "industry-informed margin rule missing"
  print("PASS: system prompt contains all signal-based adjustment rules")


if __name__ == "__main__":
  test_user_prompt_includes_new_signals()
  test_system_prompt_has_signal_rules()
  print("\nAll tests passed.")
