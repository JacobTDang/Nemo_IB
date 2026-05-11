"""Test Fix #8/#9: get_margin_breakdown extracts gross/SG&A/R&D from XBRL."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.web_search_server.sec_utils import get_margin_breakdown


def test_aapl_margin_breakdown():
  result = get_margin_breakdown('AAPL')
  print(f"AAPL result: {result}")
  assert result.get('success') is True, f"AAPL extraction failed: {result.get('error')}"
  assert result.get('gross_margin_pct', 0) > 30, f"Apple gross margin should be >30%, got {result.get('gross_margin_pct')}"
  assert result.get('rnd_pct_revenue', 0) > 5, f"Apple R&D should be >5% of revenue, got {result.get('rnd_pct_revenue')}"
  print(f"PASS: AAPL gross={result['gross_margin_pct']:.1f}% rnd={result['rnd_pct_revenue']:.1f}%")


def test_jpm_no_gross_margin():
  # Banks don't have a GrossProfit XBRL concept; the function should still return success with that field absent
  result = get_margin_breakdown('JPM')
  print(f"JPM result keys: {list(result.keys())}")
  assert result.get('success') is True, f"JPM extraction failed: {result.get('error')}"
  # gross_margin_pct may legitimately be absent for banks
  assert 'revenue' in result, "revenue should still be extracted"
  print(f"PASS: JPM revenue extracted (gross_margin_pct present={('gross_margin_pct' in result)})")


if __name__ == "__main__":
  test_aapl_margin_breakdown()
  test_jpm_no_gross_margin()
  print("\nAll tests passed.")
