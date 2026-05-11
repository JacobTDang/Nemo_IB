"""Test Fix #1: LBO HY spread lookup reads the correct flattened key."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.Financial_Modeling_Agent import Financial_Modeling_Agent, LBOParams


def _base_variables():
  return {
    'marketCap': 1_000_000_000,
    'totalDebt': 200_000_000,
    'totalCash': 50_000_000,
    'revenue_base': 500_000_000,
    'ebitda_margin': 0.20,
    'capex_pct_revenue': 0.05,
    'depreciation': 0.03,
    'tax_rate': 0.21,
    'macro.DGS10': 4.5,  # 4.5%
    'financials.revenueGrowthTTMYoy': 8.0,
    'financials.revenueGrowth5Y': 6.0,
  }


def test_hy_spread_from_credit_spread_bps():
  agent = Financial_Modeling_Agent()
  variables = _base_variables()
  variables['credit_spread.BAMLH0A0HYM2'] = 350.0  # 350 bps -> 0.035 decimal

  params = LBOParams(entry_premium=0.30, leverage_turns=4.5, exit_multiple=10.0, hold_years=5, reasoning='test')
  result = agent._run_lbo(variables, params)

  assert result is not None, "LBO result should not be None"
  comp = result['debt_rate_composition']
  assert abs(comp['hy_spread'] - 0.035) < 0.0001, f"hy_spread should be 0.035, got {comp['hy_spread']}"
  assert abs(comp['all_in_rate'] - (0.045 + 0.035)) < 0.0001, f"all_in_rate should be ~0.08, got {comp['all_in_rate']}"
  print(f"PASS: hy_spread={comp['hy_spread']:.4f} all_in_rate={comp['all_in_rate']:.4f}")


def test_hy_spread_from_flat_credit_spread_hy():
  agent = Financial_Modeling_Agent()
  variables = _base_variables()
  variables['credit_spread_hy'] = 425.0  # 425 bps -> 0.0425

  params = LBOParams(entry_premium=0.30, leverage_turns=4.5, exit_multiple=10.0, hold_years=5, reasoning='test')
  result = agent._run_lbo(variables, params)
  comp = result['debt_rate_composition']
  assert abs(comp['hy_spread'] - 0.0425) < 0.0001, f"hy_spread should be 0.0425, got {comp['hy_spread']}"
  print(f"PASS: flat-key fallback hy_spread={comp['hy_spread']:.4f}")


def test_hy_spread_fallback_when_missing():
  agent = Financial_Modeling_Agent()
  variables = _base_variables()
  # No credit spread keys at all

  params = LBOParams(entry_premium=0.30, leverage_turns=4.5, exit_multiple=10.0, hold_years=5, reasoning='test')
  result = agent._run_lbo(variables, params)
  comp = result['debt_rate_composition']
  assert abs(comp['hy_spread'] - 0.035) < 0.0001, f"fallback should be 0.035, got {comp['hy_spread']}"
  print(f"PASS: fallback hy_spread={comp['hy_spread']:.4f}")


if __name__ == "__main__":
  test_hy_spread_from_credit_spread_bps()
  test_hy_spread_from_flat_credit_spread_hy()
  test_hy_spread_fallback_when_missing()
  print("\nAll tests passed.")
