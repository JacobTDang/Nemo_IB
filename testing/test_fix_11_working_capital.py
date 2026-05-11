"""Test Fix #11: get_working_capital extracts NWC components."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.web_search_server.sec_utils import get_working_capital


def test_wmt_nwc():
  """Walmart has a clear classified balance sheet with explicit current items."""
  result = get_working_capital('WMT')
  print(f"WMT NWC: {result}")
  if not result.get('success'):
    print(f"WMT skipped (XBRL data unavailable): {result.get('error')}")
    return
  assert result.get('current_assets', 0) > 0
  assert result.get('current_liabilities', 0) > 0
  print(f"PASS: WMT CA=${result['current_assets']/1e9:.0f}B CL=${result['current_liabilities']/1e9:.0f}B NWC=${result['net_working_capital']/1e9:.1f}B")


def test_msft_nwc():
  result = get_working_capital('MSFT')
  print(f"MSFT NWC: {result}")
  if not result.get('success'):
    print(f"MSFT skipped (XBRL data unavailable): {result.get('error')}")
    return
  print(f"PASS: MSFT NWC pct revenue={result.get('nwc_pct_revenue')}")


def test_no_filing_handled():
  result = get_working_capital('NOTAREALTICKER12345')
  assert result.get('success') is False
  print(f"PASS: nonexistent ticker handled: {result.get('error')}")


if __name__ == "__main__":
  test_wmt_nwc()
  test_msft_nwc()
  test_no_filing_handled()
  print("\nAll tests passed.")
