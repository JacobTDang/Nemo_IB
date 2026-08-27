"""Test Fix #11: get_working_capital extracts NWC components.

Every test here calls live EDGAR, so every test here is gated. Ungated, all
three passed on a machine with no SEC_EMAIL: get_working_capital returned
`success: False` with "SEC_EMAIL is not set" and each test read that as its
own skip condition. test_no_filing_handled was the worst of the three -- it
asserts `success is False` for a nonexistent ticker, which the credential
error satisfies for free, so it claimed to prove a bad ticker is handled
while proving only that the request never left the machine.

test_msft_nwc additionally had no assertion at all after its early return: it
printed the ratio and returned None whatever the number was.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.web_search_server.sec_utils import get_working_capital
from testing._gates import requires_sec


@requires_sec
def test_wmt_nwc():
  """Walmart has a clear classified balance sheet with explicit current items."""
  result = get_working_capital('WMT')
  print(f"WMT NWC: {result}")
  assert result.get('success'), f"WMT working capital failed: {result.get('error')}"
  assert result.get('current_assets', 0) > 0
  assert result.get('current_liabilities', 0) > 0
  print(f"PASS: WMT CA=${result['current_assets']/1e9:.0f}B CL=${result['current_liabilities']/1e9:.0f}B NWC=${result['net_working_capital']/1e9:.1f}B")


@requires_sec
def test_msft_nwc():
  """NWC % of revenue has to be a number, and a plausible one."""
  result = get_working_capital('MSFT')
  print(f"MSFT NWC: {result}")
  assert result.get('success'), f"MSFT working capital failed: {result.get('error')}"
  pct = result.get('nwc_pct_revenue')
  assert isinstance(pct, (int, float)), f"nwc_pct_revenue is {pct!r}, not a number"
  # NWC is a balance-sheet stock over an income-statement flow, so it is not
  # bounded by 100%, but a ratio outside +/-5x revenue means the two figures
  # came from different scales.
  assert -500 < pct < 500, f"nwc_pct_revenue={pct} is off any plausible scale"
  print(f"PASS: MSFT NWC pct revenue={pct}")


@requires_sec
def test_no_filing_handled():
  result = get_working_capital('NOTAREALTICKER12345')
  assert result.get('success') is False
  error = str(result.get('error') or '')
  assert 'SEC_EMAIL' not in error, (
    "the failure is a missing credential, not an unknown ticker; this test "
    f"proves nothing about ticker handling: {error}")
  print(f"PASS: nonexistent ticker handled: {error}")


if __name__ == "__main__":
  test_wmt_nwc()
  test_msft_nwc()
  test_no_filing_handled()
  print("\nAll tests passed.")
