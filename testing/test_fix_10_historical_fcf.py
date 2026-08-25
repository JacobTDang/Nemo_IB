"""Test Fix #10: get_historical_fcf extracts OCF, capex, FCF from XBRL."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from dotenv import load_dotenv

# These reach live EDGAR, which now requires a real SEC_EMAIL rather than a
# placeholder. pytest does not read .env, so this module does -- the same
# thing testing/test_sec_series.py does for the same reason.
load_dotenv()

from tools.web_search_server.sec_utils import get_historical_fcf

# Every test here calls live SEC EDGAR. Without this the offline run was not
# offline: it spent real requests on EDGAR and could fail for network reasons
# in a run that is supposed to be hermetic.
pytestmark = [pytest.mark.network,
              pytest.mark.skipif(os.environ.get("SKIP_NETWORK_TESTS") == "1",
                                 reason="live EDGAR test")]


def test_aapl_fcf():
  result = get_historical_fcf('AAPL')
  print(f"AAPL FCF: {result}")
  assert result.get('success') is True, f"failed: {result.get('error')}"
  assert result.get('operating_cash_flow', 0) > 80_000_000_000, "AAPL OCF should be >$80B"
  assert result.get('capex', 0) > 0, "capex should be normalized to positive"
  assert result.get('free_cash_flow', 0) > 0, "FCF should be positive"
  fcf_margin = result.get('fcf_margin_pct') or 0
  assert 15 < fcf_margin < 35, f"AAPL FCF margin should be 15-35%, got {fcf_margin}"
  print(f"PASS: AAPL OCF=${result['operating_cash_flow']/1e9:.1f}B FCF=${result['free_cash_flow']/1e9:.1f}B margin={fcf_margin:.1f}%")


def test_msft_fcf():
  result = get_historical_fcf('MSFT')
  print(f"MSFT FCF: {result}")
  assert result.get('success') is True, f"failed: {result.get('error')}"
  fcf_margin = result.get('fcf_margin_pct') or 0
  assert 15 < fcf_margin < 50, f"MSFT FCF margin should be 15-50%, got {fcf_margin}"
  print(f"PASS: MSFT FCF margin={fcf_margin:.1f}%")


if __name__ == "__main__":
  test_aapl_fcf()
  test_msft_fcf()
  print("\nAll tests passed.")
