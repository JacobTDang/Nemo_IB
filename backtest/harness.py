"""Replay recorded theses against realized returns and compute hit rate +
expectancy.

A "thesis" is a row in the theses table with a recommendation, confidence, and
thesis_date. We measure the forward N-day return from that date and classify
the outcome as a win or loss based on the recommendation:

  BUY    -> win if forward return > 0
  SELL   -> win if forward return < 0
  HOLD   -> win if |forward return| < 5%  (small move)
  NEUTRAL -> excluded from stats (no directional view)
  INFO   -> excluded from stats (factual lookup)

The price source is injectable so tests can use a deterministic fake.
"""
import sys
from datetime import datetime, timedelta
from typing import Callable, List, Dict, Any, Optional, Tuple
from statistics import mean

from state.theses import thesis_history
from state.schema import get_connection


PriceFetcher = Callable[[str, str, str], Optional[Tuple[float, float]]]
# (ticker, start_iso, end_iso) -> (start_price, end_price) or None


# Percent-magnitude threshold for HOLD outcome classification:
#   |realized_return_pct| < HOLD_SMALL_MOVE_PCT  -> win (market stayed put)
#   |realized_return_pct| >= HOLD_SMALL_MOVE_PCT -> loss
# The reward bonus on a HOLD win shrinks linearly toward this threshold:
#   bonus = max(0, HOLD_SMALL_MOVE_PCT - |ret|)
# Keep both the win classification and the bonus computation referencing this
# constant — they must move together.
HOLD_SMALL_MOVE_PCT: float = 5.0


def _default_yfinance_fetcher(ticker: str, start_iso: str, end_iso: str
                               ) -> Optional[Tuple[float, float]]:
  """Default price source: yfinance daily closes."""
  try:
    import yfinance as yf
    start = datetime.fromisoformat(start_iso)
    end = datetime.fromisoformat(end_iso)
    # Pad both sides by 1 day to ensure we get a quote even on weekends
    hist = yf.Ticker(ticker.upper()).history(
      start=(start - timedelta(days=2)).date(),
      end=(end + timedelta(days=2)).date(),
      auto_adjust=True,
    )
    if hist.empty:
      return None
    # Pick closest-on-or-after each target date
    closes = hist['Close']
    start_price = None
    end_price = None
    for ts, px in closes.items():
      d = ts.date()
      if start_price is None and d >= start.date():
        start_price = float(px)
      if d >= end.date() and end_price is None:
        end_price = float(px)
        break
    if start_price is None or end_price is None:
      return None
    return start_price, end_price
  except Exception as e:
    print(f"[backtest] yfinance failed for {ticker}: {e}", file=sys.stderr, flush=True)
    return None


def backtest_thesis(thesis: Dict[str, Any], forward_days: int = 30,
                     price_fetcher: Optional[PriceFetcher] = None) -> Dict[str, Any]:
  """Measure realized return over the forward window. Returns:
    {ticker, thesis_date, recommendation, confidence,
     start_price, end_price, realized_return_pct, win, included}
  `included=False` for NEUTRAL/INFO (no directional outcome).
  """
  fetch = price_fetcher or _default_yfinance_fetcher
  ticker = thesis['ticker']
  rec = thesis.get('recommendation', '').upper()
  start_iso = thesis.get('thesis_date', '')
  if not start_iso:
    return {'error': 'missing_thesis_date', 'ticker': ticker}

  start = datetime.fromisoformat(start_iso)
  end = start + timedelta(days=forward_days)
  if end > datetime.now():
    return {'error': 'forward_window_in_future', 'ticker': ticker,
            'thesis_date': start_iso, 'window_end': end.isoformat()}

  prices = fetch(ticker, start.isoformat(), end.isoformat())
  if prices is None:
    return {'error': 'price_unavailable', 'ticker': ticker,
            'thesis_date': start_iso}
  start_price, end_price = prices
  if start_price <= 0:
    return {'error': 'invalid_start_price', 'ticker': ticker}

  ret_pct = (end_price - start_price) / start_price * 100

  included = True
  if rec == 'BUY':
    win = ret_pct > 0
  elif rec == 'SELL':
    win = ret_pct < 0
  elif rec == 'HOLD':
    win = abs(ret_pct) < HOLD_SMALL_MOVE_PCT  # see constant docstring
  else:
    included = False
    win = False

  return {
    'ticker': ticker,
    'thesis_id': thesis.get('thesis_id'),
    'thesis_date': start_iso,
    'recommendation': rec,
    'confidence': thesis.get('confidence', 0.0),
    'forward_days': forward_days,
    'start_price': round(start_price, 2),
    'end_price': round(end_price, 2),
    'realized_return_pct': round(ret_pct, 2),
    'win': win,
    'included': included,
  }


def backtest_all_theses(forward_days: int = 30,
                         price_fetcher: Optional[PriceFetcher] = None,
                         min_date: Optional[str] = None) -> List[Dict[str, Any]]:
  """Replay every thesis that has fully aged past forward_days."""
  conn = get_connection()
  try:
    cutoff = (datetime.now() - timedelta(days=forward_days)).isoformat()
    if min_date:
      rows = conn.execute(
        "SELECT * FROM theses WHERE thesis_date <= ? AND thesis_date >= ?",
        (cutoff, min_date)
      ).fetchall()
    else:
      rows = conn.execute(
        "SELECT * FROM theses WHERE thesis_date <= ?", (cutoff,)
      ).fetchall()
  finally:
    conn.close()
  out = []
  for r in rows:
    t = dict(r)
    out.append(backtest_thesis(t, forward_days, price_fetcher))
  return out


def aggregate_stats(results: List[Dict[str, Any]]) -> Dict[str, Any]:
  """Compute hit rate + expectancy from a list of backtest_thesis outputs."""
  valid = [r for r in results if 'error' not in r and r.get('included')]
  total = len(valid)
  if total == 0:
    return {'error': 'no_valid_theses', 'total': 0}

  wins = [r for r in valid if r['win']]
  losses = [r for r in valid if not r['win']]
  hit_rate = len(wins) / total

  # Returns by side
  buy_returns = [r['realized_return_pct'] for r in valid if r['recommendation'] == 'BUY']
  sell_returns = [r['realized_return_pct'] for r in valid if r['recommendation'] == 'SELL']

  # Win/loss magnitude — depends on direction.
  # HOLD asymmetry: a "win" means the price stayed put, so reward is a capped
  # bonus that shrinks as the move grows toward the 5% threshold. A "loss" is
  # the full magnitude of the unexpected move.
  def signed_return(r):
    rec = r['recommendation']
    ret = r['realized_return_pct']
    if rec == 'BUY':
      return ret
    if rec == 'SELL':
      return -ret
    if rec == 'HOLD':
      if r.get('win'):
        return max(0.0, HOLD_SMALL_MOVE_PCT - abs(ret))
      return -abs(ret)
    return 0

  win_sizes = [signed_return(r) for r in wins]
  loss_sizes = [signed_return(r) for r in losses]
  avg_win = mean(win_sizes) if win_sizes else 0
  avg_loss = mean(loss_sizes) if loss_sizes else 0
  expectancy = hit_rate * avg_win + (1 - hit_rate) * avg_loss

  return {
    'total_theses': total,
    'wins': len(wins), 'losses': len(losses),
    'hit_rate': round(hit_rate, 4),
    'avg_win_signed_pct': round(avg_win, 3),
    'avg_loss_signed_pct': round(avg_loss, 3),
    'expectancy_pct_per_thesis': round(expectancy, 3),
    'buys_evaluated': len(buy_returns),
    'sells_evaluated': len(sell_returns),
    'buy_avg_return_pct': round(mean(buy_returns), 3) if buy_returns else 0,
    'sell_avg_return_pct': round(mean(sell_returns), 3) if sell_returns else 0,
    'errors': sum(1 for r in results if 'error' in r),
  }
