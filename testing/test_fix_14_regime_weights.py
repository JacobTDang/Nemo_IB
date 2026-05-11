"""Test Fix #14: regime-weighted scenario probability."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.Financial_Modeling_Agent import Financial_Modeling_Agent, ScenarioParams


def _base_variables():
  return {
    'revenue_base': 400_000_000_000,
    'ebitda_margin': 0.30,
    'capex_pct_revenue': 0.04,
    'tax_rate': 0.21,
    'depreciation': 0.03,
    'wacc': 0.09,
    'terminal_growth': 0.025,
    'terminal_multiple': 15.0,
    'totalCash': 80_000_000_000,
    'totalDebt': 110_000_000_000,
    'sharesOutstanding': 16_000_000_000,
    'ticker': 'TEST',
  }


def _params():
  return ScenarioParams(
    bear_growth_y1=0.02, bear_growth_long_run=0.01,
    base_growth_y1=0.06, base_growth_long_run=0.04,
    bull_growth_y1=0.12, bull_growth_long_run=0.08,
    bear_margin_adj=-0.02, bull_margin_adj=0.02,
    reasoning='test'
  )


def test_expansion_weights_bull_higher():
  agent = Financial_Modeling_Agent()
  variables = _base_variables()
  variables['yield_curve_shape'] = 'normal'
  variables['macro.NFCI'] = -0.3
  result = agent._run_scenario_dcf(variables, _params())
  rw = result['regime_weighted']
  assert rw['regime'] == 'expansion'
  assert rw['weights']['bull'] > rw['weights']['bear']
  # Expansion: 30% bull, 20% bear -> weighted > mid (base case mid is the middle price)
  print(f"PASS: expansion regime weights={rw['weights']} expected=${rw['expected_price']:.2f}")


def test_inverted_curve_weights_bear_higher():
  agent = Financial_Modeling_Agent()
  variables = _base_variables()
  variables['yield_curve_shape'] = 'inverted'
  variables['macro.NFCI'] = 0.8
  result = agent._run_scenario_dcf(variables, _params())
  rw = result['regime_weighted']
  assert rw['regime'] == 'late-cycle / tightening'
  assert rw['weights']['bear'] > rw['weights']['bull']
  assert rw['weights']['bear'] == 0.50
  print(f"PASS: late-cycle/tightening weights={rw['weights']} expected=${rw['expected_price']:.2f}")


def test_inverted_only_weights():
  agent = Financial_Modeling_Agent()
  variables = _base_variables()
  variables['yield_curve_shape'] = 'inverted'
  variables['macro.NFCI'] = -0.2  # NOT tightening
  result = agent._run_scenario_dcf(variables, _params())
  rw = result['regime_weighted']
  assert rw['regime'] == 'late-cycle'
  assert rw['weights']['bear'] == 0.40
  print(f"PASS: late-cycle weights={rw['weights']}")


def test_expected_price_between_bear_and_bull():
  agent = Financial_Modeling_Agent()
  variables = _base_variables()
  variables['yield_curve_shape'] = 'normal'
  result = agent._run_scenario_dcf(variables, _params())
  rw = result['regime_weighted']
  px = result['price_range']
  assert px['low'] <= rw['expected_price'] <= px['high'], \
    f"weighted {rw['expected_price']} should be between {px['low']} and {px['high']}"
  print(f"PASS: weighted ${rw['expected_price']:.2f} between bear ${px['low']:.2f} and bull ${px['high']:.2f}")


if __name__ == "__main__":
  test_expansion_weights_bull_higher()
  test_inverted_curve_weights_bear_higher()
  test_inverted_only_weights()
  test_expected_price_between_bear_and_bull()
  print("\nAll tests passed.")
