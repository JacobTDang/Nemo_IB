"""Phase 8: backtest harness + calibration.

Uses an injected price fetcher so we don't depend on yfinance network calls."""
import sys, os
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.harness import (
  backtest_thesis, aggregate_stats, backtest_all_theses,
  _default_yfinance_fetcher,
)
from backtest.calibration import calibration_table


def _fake_fetcher(price_map: Dict[Tuple[str, str], Tuple[float, float]]):
  """Build a deterministic price fetcher from a (ticker, start_iso) -> (start, end) map."""
  def f(ticker, start_iso, end_iso):
    return price_map.get((ticker, start_iso[:10]))
  return f


def _thesis(ticker, rec, conf, days_ago, thesis_id=1):
  d = (datetime.now() - timedelta(days=days_ago)).isoformat()
  return {'thesis_id': thesis_id, 'ticker': ticker, 'recommendation': rec,
          'confidence': conf, 'thesis_date': d}


# ---- backtest_thesis core --------------------------------------------------

def test_buy_with_positive_return_is_win():
  t = _thesis('AAPL', 'BUY', 0.75, days_ago=60)
  start_date = t['thesis_date'][:10]
  fetcher = _fake_fetcher({('AAPL', start_date): (180.0, 200.0)})  # +11%
  r = backtest_thesis(t, forward_days=30, price_fetcher=fetcher)
  assert 'error' not in r, r
  assert r['win'] is True
  assert r['realized_return_pct'] == 11.11
  print(f"PASS: BUY + return={r['realized_return_pct']}% -> win")


def test_buy_with_negative_return_is_loss():
  t = _thesis('XYZ', 'BUY', 0.65, days_ago=45)
  start_date = t['thesis_date'][:10]
  fetcher = _fake_fetcher({('XYZ', start_date): (100.0, 90.0)})
  r = backtest_thesis(t, forward_days=30, price_fetcher=fetcher)
  assert r['win'] is False
  print(f"PASS: BUY with -10% -> loss")


def test_sell_with_negative_return_is_win():
  t = _thesis('TSLA', 'SELL', 0.7, days_ago=60)
  start_date = t['thesis_date'][:10]
  fetcher = _fake_fetcher({('TSLA', start_date): (300.0, 250.0)})  # -17%
  r = backtest_thesis(t, forward_days=30, price_fetcher=fetcher)
  assert r['win'] is True
  print(f"PASS: SELL + price drop -> win")


def test_hold_small_move_is_win():
  t = _thesis('KO', 'HOLD', 0.5, days_ago=60)
  start_date = t['thesis_date'][:10]
  fetcher = _fake_fetcher({('KO', start_date): (60.0, 61.0)})  # +1.6%
  r = backtest_thesis(t, forward_days=30, price_fetcher=fetcher)
  assert r['win'] is True, "HOLD with <5% move should be a win"
  print(f"PASS: HOLD with small move -> win")


def test_hold_large_move_is_loss():
  t = _thesis('KO', 'HOLD', 0.5, days_ago=60)
  start_date = t['thesis_date'][:10]
  fetcher = _fake_fetcher({('KO', start_date): (60.0, 80.0)})  # +33%
  r = backtest_thesis(t, forward_days=30, price_fetcher=fetcher)
  assert r['win'] is False
  print(f"PASS: HOLD with 33% move -> loss")


def test_info_thesis_excluded():
  t = _thesis('AAPL', 'INFO', 0.3, days_ago=60)
  start_date = t['thesis_date'][:10]
  fetcher = _fake_fetcher({('AAPL', start_date): (100.0, 110.0)})
  r = backtest_thesis(t, forward_days=30, price_fetcher=fetcher)
  assert r['included'] is False
  print(f"PASS: INFO thesis excluded from win/loss")


def test_future_window_returns_error():
  t = _thesis('AAPL', 'BUY', 0.7, days_ago=0)  # today
  fetcher = lambda *_: (100.0, 110.0)
  r = backtest_thesis(t, forward_days=30, price_fetcher=fetcher)
  assert 'error' in r
  assert 'future' in r['error']
  print("PASS: not-yet-aged thesis returns error not crash")


def test_default_fetcher_handles_single_ticker_history():
  """yfinance Ticker.history() always returns a flat-Index DataFrame.
  The fetcher must extract start/end Close prices correctly without crashing
  on the multi-ticker code path."""
  from unittest.mock import patch, MagicMock
  import pandas as pd
  import numpy as np
  from datetime import datetime, timedelta

  # Build a mock that mirrors yfinance Ticker.history() return shape exactly.
  idx = pd.date_range('2026-04-01', periods=40, freq='B')
  fake_hist = pd.DataFrame({
    'Open':   np.linspace(190, 210, 40),
    'High':   np.linspace(192, 212, 40),
    'Low':    np.linspace(189, 209, 40),
    'Close':  np.linspace(191, 211, 40),
    'Volume': [1_000_000] * 40,
  }, index=idx)

  fake_ticker = MagicMock()
  fake_ticker.history.return_value = fake_hist
  fake_yf = MagicMock()
  fake_yf.Ticker.return_value = fake_ticker

  with patch.dict('sys.modules', {'yfinance': fake_yf}):
    start_iso = idx[5].to_pydatetime().isoformat()
    end_iso   = idx[35].to_pydatetime().isoformat()
    prices = _default_yfinance_fetcher('AAPL', start_iso, end_iso)

  assert prices is not None, "default fetcher should return a (start, end) tuple"
  start_p, end_p = prices
  assert isinstance(start_p, float) and isinstance(end_p, float)
  # Start at row 5 (price ~193.6), end at row 35 (price ~209) — monotonic up
  assert end_p > start_p, f"expected upward path, got start={start_p}, end={end_p}"
  print(f"PASS: single-ticker history extracted start=${start_p:.2f} end=${end_p:.2f}")


def test_missing_price_returns_error():
  t = _thesis('XYZ', 'BUY', 0.7, days_ago=60)
  fetcher = lambda *_: None
  r = backtest_thesis(t, forward_days=30, price_fetcher=fetcher)
  assert 'error' in r
  print("PASS: missing price returns error")


# ---- aggregate_stats -------------------------------------------------------

def test_aggregate_stats_correct_hit_rate():
  # 7 wins out of 10
  results = []
  for i in range(7):
    results.append({'ticker': f'W{i}', 'recommendation': 'BUY', 'confidence': 0.7,
                    'realized_return_pct': 5.0, 'win': True, 'included': True})
  for i in range(3):
    results.append({'ticker': f'L{i}', 'recommendation': 'BUY', 'confidence': 0.6,
                    'realized_return_pct': -4.0, 'win': False, 'included': True})
  stats = aggregate_stats(results)
  assert stats['total_theses'] == 10
  assert stats['hit_rate'] == 0.7
  assert stats['wins'] == 7 and stats['losses'] == 3
  assert stats['expectancy_pct_per_thesis'] > 0
  print(f"PASS: hit_rate={stats['hit_rate']}, expectancy={stats['expectancy_pct_per_thesis']}%")


def test_aggregate_stats_hold_wins_positive_avg_win():
  """HOLD wins should contribute POSITIVE signed return (small-move bonus),
  not -abs(ret). Pre-fix, the code returned -abs(ret) for ALL HOLD positions
  including wins, so avg_win was negative and expectancy was corrupted."""
  results = []
  # 5 HOLD wins: small moves (1% each) — bonus = 5 - 1 = 4 per win
  for i in range(5):
    results.append({'ticker': f'H{i}', 'recommendation': 'HOLD', 'confidence': 0.5,
                    'realized_return_pct': 1.0, 'win': True, 'included': True})
  # 3 HOLD losses: moves at 6% (just past threshold) — penalty = -6 per loss
  for i in range(3):
    results.append({'ticker': f'L{i}', 'recommendation': 'HOLD', 'confidence': 0.4,
                    'realized_return_pct': 6.0, 'win': False, 'included': True})
  stats = aggregate_stats(results)
  assert stats['avg_win_signed_pct'] > 0, \
    f"HOLD wins should give positive avg_win, got {stats['avg_win_signed_pct']}"
  assert stats['expectancy_pct_per_thesis'] > 0, \
    f"high hit rate + small losses should yield positive expectancy, " \
    f"got {stats['expectancy_pct_per_thesis']}"
  print(f"PASS: HOLD wins -> positive avg_win ({stats['avg_win_signed_pct']}), "
        f"expectancy ({stats['expectancy_pct_per_thesis']})")


def test_aggregate_excludes_info_and_errors():
  results = [
    {'recommendation': 'INFO', 'confidence': 0.5, 'win': False, 'included': False, 'realized_return_pct': 0},
    {'recommendation': 'BUY', 'confidence': 0.7, 'win': True, 'included': True, 'realized_return_pct': 5},
    {'error': 'price_unavailable'},
  ]
  stats = aggregate_stats(results)
  assert stats['total_theses'] == 1
  assert stats['errors'] == 1
  print(f"PASS: aggregate skips INFO + errors (counted 1 valid, {stats['errors']} errors)")


def test_aggregate_empty_returns_error():
  s = aggregate_stats([])
  assert 'error' in s
  print("PASS: empty results -> error")


# ---- Calibration -----------------------------------------------------------

def test_calibration_well_calibrated_when_conf_matches_hit():
  """0.8 conf -> 80% hit rate => well calibrated."""
  results = []
  # 10 theses at conf 0.8, 8 wins
  for i in range(8):
    results.append({'confidence': 0.80, 'win': True, 'included': True,
                     'recommendation': 'BUY', 'realized_return_pct': 5})
  for i in range(2):
    results.append({'confidence': 0.80, 'win': False, 'included': True,
                     'recommendation': 'BUY', 'realized_return_pct': -3})
  cal = calibration_table(results)
  assert cal['quality'] == 'well_calibrated'
  print(f"PASS: 0.8 conf + 80% hit -> well_calibrated (error={cal['overall_calibration_error']})")


def test_calibration_miscalibrated_when_overconfident():
  """0.9 conf but 40% hit rate -> badly miscalibrated."""
  results = []
  for i in range(4):
    results.append({'confidence': 0.90, 'win': True, 'included': True,
                     'recommendation': 'BUY', 'realized_return_pct': 5})
  for i in range(6):
    results.append({'confidence': 0.90, 'win': False, 'included': True,
                     'recommendation': 'BUY', 'realized_return_pct': -3})
  cal = calibration_table(results)
  assert cal['quality'] == 'badly_miscalibrated', \
    f"expected miscalibration alarm, got {cal['quality']}"
  print(f"PASS: 0.9 conf + 40% hit -> {cal['quality']} (error={cal['overall_calibration_error']})")


def test_calibration_bins_distinct():
  results = [
    {'confidence': 0.55, 'win': True,  'included': True, 'recommendation': 'BUY', 'realized_return_pct': 2},
    {'confidence': 0.70, 'win': True,  'included': True, 'recommendation': 'BUY', 'realized_return_pct': 3},
    {'confidence': 0.80, 'win': False, 'included': True, 'recommendation': 'BUY', 'realized_return_pct': -3},
    {'confidence': 0.90, 'win': True,  'included': True, 'recommendation': 'BUY', 'realized_return_pct': 8},
  ]
  cal = calibration_table(results)
  populated = [b for b in cal['bins'] if b['n'] > 0]
  assert len(populated) >= 3, f"results span 4 bins, expected at least 3 populated, got {len(populated)}"
  print(f"PASS: calibration binning split into {len(populated)} populated bins")


def test_calibration_empty_returns_error():
  cal = calibration_table([])
  assert 'error' in cal
  print("PASS: empty input -> error")


if __name__ == "__main__":
  test_buy_with_positive_return_is_win()
  test_buy_with_negative_return_is_loss()
  test_sell_with_negative_return_is_win()
  test_hold_small_move_is_win()
  test_hold_large_move_is_loss()
  test_info_thesis_excluded()
  test_future_window_returns_error()
  test_default_fetcher_handles_single_ticker_history()
  test_missing_price_returns_error()
  test_aggregate_stats_correct_hit_rate()
  test_aggregate_stats_hold_wins_positive_avg_win()
  test_aggregate_excludes_info_and_errors()
  test_aggregate_empty_returns_error()
  test_calibration_well_calibrated_when_conf_matches_hit()
  test_calibration_miscalibrated_when_overconfident()
  test_calibration_bins_distinct()
  test_calibration_empty_returns_error()
  print("\nAll Phase 8 backtest + calibration tests passed.")
