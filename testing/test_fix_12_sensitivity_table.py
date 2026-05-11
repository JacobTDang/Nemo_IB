"""Test Fix #12: 2D sensitivity table (WACC x terminal growth) -> price."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.financial_modeling_engine.analysis_tools import _sensitivity_table_math


def _base_inputs():
  return {
    'revenue_base': 400_000_000_000,
    'ebitda_margin': 0.30,
    'capex_pct_revenue': 0.04,
    'tax_rate': 0.21,
    'depreciation': 0.03,
    'revenue_growth': [0.08, 0.07, 0.06, 0.05, 0.04],
    'wacc': 0.10,
    'terminal_growth': 0.025,
    'terminal_multiple': 0,
    'cash': 80_000_000_000,
    'debt': 110_000_000_000,
    'shares_outstanding': 16_000_000_000,
    'ticker': 'AAPL',
  }


def test_default_grid():
  result = _sensitivity_table_math(_base_inputs())
  assert result['cells_filled'] >= 20, f"should fill most of 25 cells, got {result['cells_filled']}"
  assert result['min_price'] > 0
  assert result['max_price'] > result['min_price']
  assert result['mid_price'] is not None, "mid (base WACC, 2.5% g) should be filled"
  print(f"PASS: {result['cells_filled']} cells, range ${result['min_price']:.2f}-${result['max_price']:.2f}, mid=${result['mid_price']:.2f}")


def test_unstable_cells_skipped():
  # Set tiny WACC so the bottom row collides with terminal_growth
  inputs = _base_inputs()
  result = _sensitivity_table_math(inputs, wacc_range=[0.02, 0.05, 0.10])
  # 0.02 - 0.035 < 0.005 -> None
  # 0.02 - 0.030 < 0.005 -> None
  # 0.02 - 0.025 < 0.005 -> None
  bottom_row = result['table']['0.0200']
  none_count = sum(1 for v in bottom_row.values() if v is None)
  assert none_count >= 3, f"low-WACC row should have several None cells, got {none_count}"
  print(f"PASS: {none_count} unstable cells correctly skipped")


def test_custom_ranges():
  result = _sensitivity_table_math(_base_inputs(),
                                   wacc_range=[0.08, 0.09, 0.10],
                                   tg_range=[0.02, 0.025, 0.03])
  assert len(result['wacc_range']) == 3
  assert len(result['tg_range']) == 3
  assert result['cells_filled'] == 9
  print(f"PASS: custom 3x3 grid filled {result['cells_filled']} cells")


if __name__ == "__main__":
  test_default_grid()
  test_unstable_cells_skipped()
  test_custom_ranges()
  print("\nAll tests passed.")
