"""Bug 2: credit profile and capital returns refuse to compute on missing fundamentals.

Live run produced "Net Debt/EBITDA: 16,203,997,184.0x" because revenue_base was
absent → ebitda = 0 → safe_ebitda fell back to 1.0 → ratio came out as raw $.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.Financial_Modeling_Agent import Financial_Modeling_Agent, LBOParams
from tools.financial_modeling_engine.analysis_tools import (
  _credit_profile_math, _capital_returns_math,
)


def test_run_credit_profile_skips_when_revenue_missing():
  agent = Financial_Modeling_Agent()
  variables = {'ebitda_margin': 0.31, 'totalDebt': 5e9, 'totalCash': 1e9}
  result = agent._run_credit_profile(variables)
  assert result is None, f"should return None when revenue missing, got {result}"
  print("PASS: credit profile skipped when revenue missing")


def test_run_credit_profile_skips_when_margin_zero():
  agent = Financial_Modeling_Agent()
  variables = {'revenue_base': 400e9, 'ebitda_margin': 0, 'totalDebt': 5e9, 'totalCash': 1e9}
  result = agent._run_credit_profile(variables)
  assert result is None
  print("PASS: credit profile skipped when margin is zero")


def test_credit_math_rejects_zero_ebitda():
  result = _credit_profile_math(
    total_debt=5e9, cash=1e9, ebitda=0,
    interest_expense=1e9, depreciation_abs=1e9, capex_abs=1e9, tax_rate=0.21,
  )
  assert result.get('error'), f"should return error dict, got {result}"
  assert 'ebitda' in result['error'].lower()
  print(f"PASS: credit math rejects ebitda=0 -> {result['error']}")


def test_credit_math_normal_inputs():
  result = _credit_profile_math(
    total_debt=110e9, cash=80e9, ebitda=120e9,
    interest_expense=4e9, depreciation_abs=12e9, capex_abs=15e9, tax_rate=0.21,
    market_cap=3e12,
  )
  assert not result.get('error'), f"normal inputs should not error: {result}"
  ratio = result['net_debt_ebitda']
  assert 0 < ratio < 10, f"leverage should be reasonable, got {ratio}"
  print(f"PASS: credit math with normal inputs -> Net Debt/EBITDA={ratio:.2f}x")


def test_capital_returns_skips_when_market_cap_missing():
  agent = Financial_Modeling_Agent()
  variables = {'revenue_base': 400e9, 'ebitda_margin': 0.31}  # no marketCap
  result = agent._run_capital_returns(variables)
  assert result is None
  print("PASS: capital returns skipped when market_cap missing")


def test_capital_returns_math_rejects_zero_market_cap():
  result = _capital_returns_math(
    market_cap=0, ebitda=120e9, capex_abs=15e9, tax_rate=0.21, depreciation_abs=12e9,
  )
  assert result.get('error'), f"should return error: {result}"
  assert 'market_cap' in result['error'].lower()
  print(f"PASS: capital returns math rejects market_cap=0 -> {result['error']}")


def test_lbo_skips_when_revenue_missing():
  agent = Financial_Modeling_Agent()
  variables = {'marketCap': 1e9, 'totalDebt': 2e8, 'totalCash': 5e7}  # no revenue
  params = LBOParams(entry_premium=0.30, leverage_turns=4.5, exit_multiple=10.0, hold_years=5, reasoning='test')
  result = agent._run_lbo(variables, params)
  assert result is None
  print("PASS: LBO skipped when revenue missing")


def test_full_normal_run():
  agent = Financial_Modeling_Agent()
  variables = {
    'revenue_base': 400e9, 'ebitda_margin': 0.31,
    'depreciation': 0.03, 'capex_pct_revenue': 0.04, 'tax_rate': 0.21,
    'totalDebt': 110e9, 'totalCash': 80e9, 'interestExpense': 4e9,
    'marketCap': 3e12,
  }
  result = agent._run_credit_profile(variables)
  assert result is not None, "normal inputs should produce a result"
  assert 0 < result['net_debt_ebitda'] < 10, f"leverage = {result['net_debt_ebitda']}"
  print(f"PASS: full normal run -> Net Debt/EBITDA={result['net_debt_ebitda']:.2f}x")


if __name__ == "__main__":
  test_run_credit_profile_skips_when_revenue_missing()
  test_run_credit_profile_skips_when_margin_zero()
  test_credit_math_rejects_zero_ebitda()
  test_credit_math_normal_inputs()
  test_capital_returns_skips_when_market_cap_missing()
  test_capital_returns_math_rejects_zero_market_cap()
  test_lbo_skips_when_revenue_missing()
  test_full_normal_run()
  print("\nAll tests passed.")
