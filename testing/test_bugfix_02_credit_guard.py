"""Bug 2: the credit and capital-returns math refuses to compute on missing fundamentals.

Live run produced "Net Debt/EBITDA: 16,203,997,184.0x" because revenue_base was
absent → ebitda = 0 → safe_ebitda fell back to 1.0 → ratio came out as raw $.

The five tests that drove this through `Financial_Modeling_Agent._run_*` went
with that module when the LangGraph/OpenRouter layer was retired. The guard
itself lives in `tools/financial_modeling_engine/analysis_tools.py` and is
reached by the shipped `calculate_credit_profile` and `calculate_capital_returns`
tools, so it is still tested -- directly, at the layer that ships.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.financial_modeling_engine.analysis_tools import (
  _credit_profile_math, _capital_returns_math,
)


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


def test_capital_returns_math_rejects_zero_market_cap():
  result = _capital_returns_math(
    market_cap=0, ebitda=120e9, capex_abs=15e9, tax_rate=0.21, depreciation_abs=12e9,
  )
  assert result.get('error'), f"should return error: {result}"
  assert 'market_cap' in result['error'].lower()
  print(f"PASS: capital returns math rejects market_cap=0 -> {result['error']}")


if __name__ == "__main__":
  test_credit_math_rejects_zero_ebitda()
  test_credit_math_normal_inputs()
  test_capital_returns_math_rejects_zero_market_cap()
  print("\nAll tests passed.")
