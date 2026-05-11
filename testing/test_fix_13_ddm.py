"""Test Fix #13: Dividend Discount Model (Gordon + two-stage)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.financial_modeling_engine.analysis_tools import _ddm_math


def test_gordon_growth():
  # DPS=$1, Ke=9%, g=3% -> D_1 = 1.03, price = 1.03 / 0.06 = $17.17
  result = _ddm_math(current_dps=1.0, cost_of_equity=0.09, terminal_growth=0.03)
  assert result['method'] == 'gordon_growth'
  assert abs(result['intrinsic_value_per_share'] - 17.17) < 0.05, f"expected ~$17.17, got ${result['intrinsic_value_per_share']}"
  print(f"PASS: Gordon DDM DPS=$1 Ke=9% g=3% -> ${result['intrinsic_value_per_share']:.2f}")


def test_two_stage():
  # 5 yrs at 10% growth, then 3% perpetuity, Ke=9%
  result = _ddm_math(current_dps=1.0, cost_of_equity=0.09, terminal_growth=0.03,
                     high_growth_rate=0.10, high_growth_years=5)
  assert result['method'] == 'two_stage'
  # Two-stage should give a higher value than simple Gordon at 3%
  gordon = _ddm_math(current_dps=1.0, cost_of_equity=0.09, terminal_growth=0.03)
  assert result['intrinsic_value_per_share'] > gordon['intrinsic_value_per_share'], \
    "two-stage with high growth phase should exceed simple Gordon"
  print(f"PASS: two-stage ${result['intrinsic_value_per_share']:.2f} > gordon ${gordon['intrinsic_value_per_share']:.2f}")


def test_terminal_growth_exceeds_cost_of_equity():
  # Invalid: g >= Ke
  result = _ddm_math(current_dps=1.0, cost_of_equity=0.05, terminal_growth=0.06)
  assert result.get('success') is False
  assert 'must exceed' in result.get('error', '').lower()
  print(f"PASS: invalid g>=Ke handled: {result['error']}")


def test_high_dps():
  # Realistic mature company: KO-like, DPS=$1.94, Ke=8%, g=3%
  result = _ddm_math(current_dps=1.94, cost_of_equity=0.08, terminal_growth=0.03)
  # 1.94 * 1.03 / 0.05 = 39.96
  assert 35 < result['intrinsic_value_per_share'] < 45
  print(f"PASS: KO-like inputs -> ${result['intrinsic_value_per_share']:.2f}")


if __name__ == "__main__":
  test_gordon_growth()
  test_two_stage()
  test_terminal_growth_exceeds_cost_of_equity()
  test_high_dps()
  print("\nAll tests passed.")
