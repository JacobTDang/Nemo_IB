"""Polymorphic data-source layer.

Each public function tries multiple sources in priority order and returns
`(value, source_name)` on success or `(None, None)` on total failure.

Source priority is set per-metric based on data authority:
  - Fundamentals (revenue, margins): SEC XBRL first, yfinance fallback
  - Market data (cap, beta, debt): yfinance only (real-time)
  - Rates / time series: not handled here (use FRED tools directly)

The first arg is always `ticker`. The optional `variables` dict lets callers
short-circuit when the variable store already has the value — return early
with `source='cached'`.

Usage:
    from data.sources import get_revenue, get_market_cap
    revenue, src = get_revenue('AAPL', variables=current_store)
    if revenue is None:
        # both sources failed
        ...
"""
from typing import Optional, Tuple
import sys

from tools.web_search_server.sec_utils import (
  get_revenue_base as _sec_revenue,
  get_ebitda_margin as _sec_ebitda_margin,
  get_capex_pct_revenue as _sec_capex,
  get_tax_rate as _sec_tax,
  get_depreciation as _sec_depreciation,
)
from tools.financial_modeling_engine.utils import get_data as _yfinance_data

DataValue = Tuple[Optional[float], Optional[str]]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _from_sec(fn, ticker: str, key: str) -> DataValue:
  """Run a SEC fetcher and extract `key` on success. (None, None) on failure."""
  try:
    r = fn(ticker)
    if r and r.get('success'):
      v = r.get(key)
      if v is not None and v != 0:
        return float(v), 'sec'
  except Exception as e:
    print(f"[sources] SEC {fn.__name__} failed: {e}", file=sys.stderr, flush=True)
  return None, None


def _from_yfinance(ticker: str, key: str) -> DataValue:
  """Read `key` from yfinance get_data. (None, None) on failure."""
  try:
    d = _yfinance_data(ticker)
    v = d.get(key)
    if v is not None and v != 0:
      return float(v), 'yfinance'
  except Exception as e:
    print(f"[sources] yfinance {key} failed: {e}", file=sys.stderr, flush=True)
  return None, None


def _from_cache(variables: Optional[dict], *keys: str) -> DataValue:
  """First non-zero value among `keys` in variables. Empty result if cache miss."""
  if not variables:
    return None, None
  for k in keys:
    v = variables.get(k)
    if v is not None and v != 0:
      try:
        return float(v), 'cached'
      except (TypeError, ValueError):
        continue
  return None, None


# ---------------------------------------------------------------------------
# Fundamentals — SEC primary, yfinance fallback
# ---------------------------------------------------------------------------

def get_revenue(ticker: str, variables: dict = None) -> DataValue:
  """Total annual revenue in dollars. SEC (10-K) -> yfinance (TTM)."""
  v, s = _from_cache(variables, 'revenue_base', 'revenue')
  if v is not None: return v, s
  v, s = _from_sec(_sec_revenue, ticker, 'revenue_base')
  if v is not None: return v, s
  return _from_yfinance(ticker, 'revenue')


def get_ebitda_margin_pct(ticker: str, variables: dict = None) -> DataValue:
  """EBITDA as a percentage of revenue (e.g. 31.5 for 31.5%)."""
  v, s = _from_cache(variables, 'ebitda_margin_percent')
  if v is not None: return v, s
  v, s = _from_sec(_sec_ebitda_margin, ticker, 'ebitda_margin_percent')
  if v is not None: return v, s
  try:
    d = _yfinance_data(ticker)
    ebitda = d.get('EBITDA')
    revenue = d.get('revenue')
    if ebitda and revenue and revenue > 0:
      return (ebitda / revenue) * 100, 'yfinance'
  except Exception as e:
    print(f"[sources] yfinance ebitda_margin failed: {e}", file=sys.stderr, flush=True)
  return None, None


def get_capex_pct(ticker: str, variables: dict = None) -> DataValue:
  """CapEx as percentage of revenue (SEC only — yfinance doesn't expose this cleanly)."""
  v, s = _from_cache(variables, 'capex_pct_revenue')
  if v is not None: return v, s
  return _from_sec(_sec_capex, ticker, 'capex_pct_revenue')


def get_tax_rate_pct(ticker: str, variables: dict = None) -> DataValue:
  """Effective tax rate as percentage (e.g. 21.0 for 21%)."""
  v, s = _from_cache(variables, 'effective_tax_rate')
  if v is not None: return v, s
  return _from_sec(_sec_tax, ticker, 'effective_tax_rate')


def get_depreciation_pct(ticker: str, variables: dict = None) -> DataValue:
  """D&A as percentage of revenue."""
  v, s = _from_cache(variables, 'd&a_pct')
  if v is not None: return v, s
  return _from_sec(_sec_depreciation, ticker, 'd&a_pct')


# ---------------------------------------------------------------------------
# Market data — yfinance authoritative (real-time)
# ---------------------------------------------------------------------------

def get_market_cap(ticker: str, variables: dict = None) -> DataValue:
  v, s = _from_cache(variables, 'marketCap', 'market_cap')
  if v is not None: return v, s
  return _from_yfinance(ticker, 'marketCap')


def get_beta(ticker: str, variables: dict = None) -> DataValue:
  v, s = _from_cache(variables, 'beta')
  if v is not None: return v, s
  return _from_yfinance(ticker, 'beta')


def get_total_debt(ticker: str, variables: dict = None) -> DataValue:
  v, s = _from_cache(variables, 'totalDebt', 'total_debt')
  if v is not None: return v, s
  return _from_yfinance(ticker, 'totalDebt')


def get_cash(ticker: str, variables: dict = None) -> DataValue:
  v, s = _from_cache(variables, 'totalCash', 'cash')
  if v is not None: return v, s
  return _from_yfinance(ticker, 'cash')


def get_shares_outstanding(ticker: str, variables: dict = None) -> DataValue:
  v, s = _from_cache(variables, 'sharesOutstanding', 'shares_outstanding')
  if v is not None: return v, s
  return _from_yfinance(ticker, 'sharesOutstanding')


def get_interest_expense(ticker: str, variables: dict = None) -> DataValue:
  v, s = _from_cache(variables, 'interestExpense', 'interest_expense')
  if v is not None: return v, s
  return _from_yfinance(ticker, 'interestExpense')
