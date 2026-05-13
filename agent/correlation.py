"""Correlation analysis for portfolio risk.

When the Risk Officer evaluates a new BUY, this module checks whether the
proposed position would push portfolio correlation above the configured
threshold — i.e., we'd just be doubling the same bet.

Uses 90-day daily returns from yfinance. Cached at function level (LRU) to
avoid hammering Yahoo on each evaluation.
"""
import sys
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from functools import lru_cache


@lru_cache(maxsize=1)
def _daily_returns_panel_cached(tickers_tuple: tuple, days: int = 90):
  """Pull daily close % changes for a tuple of tickers. Cached for 1 call."""
  try:
    import yfinance as yf
    import pandas as pd
    end = datetime.now()
    start = end - timedelta(days=days * 2)  # padding for weekends
    df = yf.download(list(tickers_tuple), start=start.date(), end=end.date(),
                     progress=False, auto_adjust=True)
    if df.empty:
      return None
    # yfinance multi-ticker returns multi-index columns: ('Close', 'AAPL'), etc.
    if 'Close' in df.columns.get_level_values(0):
      closes = df['Close']
    else:
      closes = df  # single-ticker fallback
    returns = closes.pct_change().dropna()
    return returns
  except Exception as e:
    print(f"[correlation] yfinance failed: {e}", file=sys.stderr, flush=True)
    return None


def correlation_matrix(tickers: List[str], days: int = 90):
  """Returns a pandas DataFrame correlation matrix, or None on failure."""
  if not tickers:
    return None
  returns = _daily_returns_panel_cached(tuple(sorted(set(tickers))), days)
  if returns is None or returns.empty:
    return None
  return returns.corr()


def avg_correlation_to_basket(candidate: str, basket: List[str],
                               days: int = 90) -> Optional[float]:
  """Average pairwise correlation between candidate and each ticker in basket.

  Returns None when correlation data can't be fetched."""
  if not basket:
    return 0.0  # no existing positions to be correlated with
  if candidate in basket:
    return 1.0  # already in basket
  all_tickers = [candidate] + list(basket)
  corr = correlation_matrix(all_tickers, days)
  if corr is None or candidate not in corr.columns:
    return None
  pairs = [float(corr.loc[candidate, b]) for b in basket
           if b in corr.index and b != candidate]
  if not pairs:
    return None
  return sum(pairs) / len(pairs)


def correlation_decision(candidate: str, basket: List[str],
                          threshold: float = 0.7,
                          days: int = 90) -> Dict[str, any]:
  """Decide whether candidate is too correlated to the existing basket.

  Returns:
    {ok: bool, avg_correlation: float, threshold, reason: str}

  Special case: if `candidate` is already in `basket`, this is a scale-up of
  an existing position — not a new concentration. Short-circuit and approve
  without computing a meaningless self-correlation.
  """
  if candidate and basket and candidate.upper() in {b.upper() for b in basket}:
    return {
      'ok': True, 'avg_correlation': None, 'threshold': threshold,
      'reason': f'scale_up_existing_position ({candidate})',
    }
  avg = avg_correlation_to_basket(candidate, basket, days)
  if avg is None:
    return {
      'ok': True, 'avg_correlation': None, 'threshold': threshold,
      'reason': 'correlation_unavailable_skipping_check',
    }
  return {
    'ok': avg < threshold,
    'avg_correlation': round(avg, 3),
    'threshold': threshold,
    'reason': (
      f"avg corr {avg:.2f} >= threshold {threshold:.2f}; "
      f"adding {candidate} would concentrate the basket"
      if avg >= threshold
      else f"avg corr {avg:.2f} < threshold {threshold:.2f}"
    ),
  }
