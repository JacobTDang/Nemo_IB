"""Alternative data sources beyond fundamentals + news.

Phase 5c: Google Trends via pytrends (free, rate-limited by Google).

Maps a ticker to product/brand query terms via TICKER_QUERIES, pulls 12-month
search interest, and computes momentum.
"""
import sys
from typing import List, Dict, Any, Optional


# Mapping of tickers to representative consumer-product/brand search terms.
# Add new tickers here as the watchlist grows.
TICKER_QUERIES: Dict[str, List[str]] = {
  'AAPL': ['iPhone', 'AirPods', 'Apple Vision Pro'],
  'MSFT': ['Microsoft Copilot', 'Xbox', 'Azure'],
  'GOOGL': ['Google search', 'YouTube', 'Gemini AI'],
  'AMZN': ['Amazon Prime', 'AWS'],
  'META': ['Instagram', 'WhatsApp', 'Threads'],
  'NVDA': ['Nvidia H100', 'Nvidia B200', 'CUDA'],
  'TSLA': ['Tesla', 'Cybertruck', 'Model Y'],
  'NFLX': ['Netflix'],
  'DIS':  ['Disney+', 'Disneyland'],
  'KO':   ['Coca-Cola'],
  'JPM':  ['JPMorgan Chase'],
  'XOM':  ['Exxon'],
  'WMT':  ['Walmart'],
  'COST': ['Costco'],
}


def google_trends_for(ticker: str, query_terms: Optional[List[str]] = None,
                       timeframe: str = 'today 12-m') -> Dict[str, Any]:
  """Pull Google Trends interest-over-time for the ticker's query terms.

  Returns:
    {
      'ticker': str,
      'queries': list[str],
      'series': {query: list[(date_str, value)]},
      'latest_avg': dict[query, float],
      'momentum_3m_vs_12m_pct': dict[query, float],  # > 0 = recent > average
    }
  pytrends fails opaquely on rate-limits; we catch and return error info.
  """
  ticker_u = ticker.upper()
  queries = query_terms or TICKER_QUERIES.get(ticker_u)
  if not queries:
    return {'error': 'no_queries_mapped', 'ticker': ticker_u,
            'hint': f'add {ticker_u} to TICKER_QUERIES'}

  try:
    from pytrends.request import TrendReq
  except ImportError as e:
    return {'error': f'pytrends not installed: {e}'}

  try:
    pytrends = TrendReq(hl='en-US', tz=0, timeout=(10, 25), retries=2, backoff_factor=0.3)
    # pytrends caps at 5 queries per request
    pytrends.build_payload(queries[:5], timeframe=timeframe, geo='US')
    df = pytrends.interest_over_time()
  except Exception as e:
    return {'error': f'pytrends_failed: {type(e).__name__}: {e}',
            'ticker': ticker_u, 'queries': queries}

  if df is None or df.empty:
    return {'error': 'empty_dataframe', 'ticker': ticker_u, 'queries': queries}

  if 'isPartial' in df.columns:
    df = df[df['isPartial'] == False]

  series = {}
  latest_avg = {}
  momentum = {}
  for q in queries[:5]:
    if q not in df.columns:
      continue
    col = df[q].astype(float)
    series[q] = [(str(idx.date()), float(v)) for idx, v in col.items()]
    if len(col) == 0:
      continue
    full_avg = col.mean()
    recent_avg = col.tail(13).mean()  # last ~3 months (weekly data)
    latest_avg[q] = round(float(recent_avg), 2)
    momentum[q] = round(float((recent_avg - full_avg) / full_avg * 100), 2) \
                  if full_avg > 0 else 0.0

  return {
    'ticker': ticker_u,
    'queries': queries[:5],
    'series': series,
    'latest_avg': latest_avg,
    'momentum_3m_vs_12m_pct': momentum,
    'timeframe': timeframe,
  }
