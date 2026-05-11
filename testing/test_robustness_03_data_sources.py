"""Item 3: polymorphic data-source fetchers (SEC -> yfinance fallback)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import patch
from data.sources import (
  get_revenue, get_ebitda_margin_pct, get_market_cap, get_beta,
  get_total_debt, get_cash, get_shares_outstanding,
)


def test_cache_hit_short_circuits():
  """When variables has the value, no fetch happens."""
  variables = {'revenue_base': 400_000_000_000}
  with patch('data.sources._sec_revenue') as mock_sec, \
       patch('data.sources._yfinance_data') as mock_yf:
    v, src = get_revenue('AAPL', variables=variables)
    assert v == 400_000_000_000.0
    assert src == 'cached'
    mock_sec.assert_not_called()
    mock_yf.assert_not_called()
  print("PASS: cache hit short-circuits both sources")


def test_sec_fails_yfinance_recovers():
  """When SEC errors, fall through to yfinance."""
  with patch('data.sources._sec_revenue', return_value={'success': False, 'error': 'sim'}), \
       patch('data.sources._yfinance_data', return_value={'revenue': 250_000_000_000}):
    v, src = get_revenue('JPM')
    assert v == 250_000_000_000.0
    assert src == 'yfinance'
  print("PASS: SEC failure -> yfinance recovery")


def test_both_sources_fail():
  """When both sources return nothing, get (None, None) — no exception."""
  with patch('data.sources._sec_revenue', return_value={'success': False}), \
       patch('data.sources._yfinance_data', return_value={'revenue': None}):
    v, src = get_revenue('NONEXISTENT')
    assert v is None
    assert src is None
  print("PASS: both sources fail returns (None, None)")


def test_yfinance_only_metrics():
  """Market data has no SEC source — yfinance is the only authority."""
  variables = {'marketCap': 3_000_000_000_000}
  v, src = get_market_cap('AAPL', variables=variables)
  assert v == 3_000_000_000_000.0
  assert src == 'cached'
  print("PASS: market cap from cache")

  with patch('data.sources._yfinance_data', return_value={'beta': 1.25}):
    v, src = get_beta('AAPL')
    assert v == 1.25
    assert src == 'yfinance'
  print("PASS: beta from yfinance")


def test_ebitda_margin_derived_from_yfinance_when_sec_fails():
  """SEC margin fails -> derive from yfinance EBITDA/revenue."""
  with patch('data.sources._sec_ebitda_margin', return_value={'success': False}), \
       patch('data.sources._yfinance_data', return_value={'EBITDA': 100_000_000_000, 'revenue': 400_000_000_000}):
    v, src = get_ebitda_margin_pct('AAPL')
    assert v == 25.0, f"expected 25.0%, got {v}"
    assert src == 'yfinance'
  print(f"PASS: ebitda_margin derived from yfinance = {v}%")


def test_zero_values_treated_as_missing():
  """If a cached value is zero, fall through to next source."""
  variables = {'revenue_base': 0}  # zero is "missing", not legitimate
  with patch('data.sources._sec_revenue', return_value={'success': True, 'revenue_base': 100_000_000_000}):
    v, src = get_revenue('TICKER', variables=variables)
    assert v == 100_000_000_000.0
    assert src == 'sec'
  print("PASS: zero in cache treated as missing, falls through")


if __name__ == "__main__":
  test_cache_hit_short_circuits()
  test_sec_fails_yfinance_recovers()
  test_both_sources_fail()
  test_yfinance_only_metrics()
  test_ebitda_margin_derived_from_yfinance_when_sec_fails()
  test_zero_values_treated_as_missing()
  print("\nAll tests passed.")
