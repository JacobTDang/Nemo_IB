"""Item 4: pre-flight dependency contract on modeling tools."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.Financial_Modeling_Agent import Financial_Modeling_Agent as FMA


def test_missing_inputs_empty_variables():
  """Empty variables -> every required input is missing."""
  missing = FMA.missing_inputs_for('scenario_dcf', {})
  assert set(missing) == set(FMA.REQUIRED_INPUTS['scenario_dcf']), \
    f"unexpected missing set: {missing}"
  print(f"PASS: empty variables -> all {len(missing)} scenario_dcf inputs flagged missing")


def test_missing_inputs_full_variables():
  """All required keys present -> nothing missing."""
  variables = {
    'revenue_base': 400e9, 'ebitda_margin': 0.31, 'capex_pct_revenue': 0.04,
    'tax_rate': 0.21, 'depreciation': 0.03, 'wacc': 0.09, 'sharesOutstanding': 16e9,
  }
  missing = FMA.missing_inputs_for('scenario_dcf', variables)
  assert missing == [], f"expected nothing missing, got {missing}"
  print("PASS: full variables -> nothing missing")


def test_missing_inputs_zero_treated_as_missing():
  """Zero in the store -> treated as missing (not legitimately zero)."""
  variables = {'revenue_base': 0, 'ebitda_margin': 0.31}
  missing = FMA.missing_inputs_for('credit_profile', variables)
  assert 'revenue_base' in missing, f"zero not flagged: {missing}"
  print(f"PASS: zero treated as missing")


def test_tools_to_fill_gaps_dedupes():
  """Multiple missing keys mapping to same tool dedupe in result."""
  missing = ['marketCap', 'beta', 'totalDebt', 'totalCash', 'sharesOutstanding']
  # All five map to get_market_data
  tools = FMA.tools_to_fill_gaps(missing)
  assert tools == ['get_market_data'], f"expected single dedupe, got {tools}"
  print(f"PASS: 5 keys -> 1 deduped tool: {tools}")


def test_tools_to_fill_gaps_order_preserved():
  """Order of missing keys is preserved in the returned tool list."""
  missing = ['revenue_base', 'ebitda_margin', 'capex_pct_revenue']
  tools = FMA.tools_to_fill_gaps(missing)
  assert tools == ['get_revenue_base', 'get_ebitda_margin', 'get_capex_pct_revenue']
  print(f"PASS: tool list order preserved: {tools}")


def test_ddm_required_inputs():
  """DDM is special — needs dividendsPaid which most queries skip."""
  missing = FMA.missing_inputs_for('ddm', {})
  assert 'dividendsPaid' in missing
  assert FMA.INPUT_TO_TOOL['dividendsPaid'] == 'get_financial_statements'
  print("PASS: DDM declares dividendsPaid; maps to get_financial_statements")


def test_unknown_model_returns_empty():
  """Asking about a model that doesn't exist -> empty list (no crash)."""
  missing = FMA.missing_inputs_for('alien_model', {})
  assert missing == []
  print("PASS: unknown model returns empty missing list")


if __name__ == "__main__":
  test_missing_inputs_empty_variables()
  test_missing_inputs_full_variables()
  test_missing_inputs_zero_treated_as_missing()
  test_tools_to_fill_gaps_dedupes()
  test_tools_to_fill_gaps_order_preserved()
  test_ddm_required_inputs()
  test_unknown_model_returns_empty()
  print("\nAll tests passed.")
