"""Confidence calibration.

If the system says confidence=0.8, is it right 80% of the time? Bins the
backtested theses by confidence and computes hit rate per bin. The
calibration error is the average |confidence - hit_rate| per bin.

Well-calibrated: error < 0.10. Drifted: > 0.20.
"""
from typing import List, Dict, Any
from statistics import mean


CONFIDENCE_BINS = [
  (0.0, 0.5),   # low
  (0.5, 0.65),  # medium-low (below trade threshold)
  (0.65, 0.75), # tradeable normal
  (0.75, 0.85), # high
  (0.85, 1.01), # very high (1.01 to include 1.0)
]


def calibration_table(backtest_results: List[Dict[str, Any]]) -> Dict[str, Any]:
  """Bin backtest results by confidence and compute hit rate per bin."""
  valid = [r for r in backtest_results if 'error' not in r and r.get('included')]
  if not valid:
    return {'error': 'no_valid_results'}

  bins_out = []
  for low, high in CONFIDENCE_BINS:
    in_bin = [r for r in valid if low <= (r.get('confidence') or 0) < high]
    if not in_bin:
      bins_out.append({
        'range': [low, high], 'n': 0, 'hit_rate': None,
        'avg_confidence': None, 'error': None,
      })
      continue
    wins = sum(1 for r in in_bin if r['win'])
    hr = wins / len(in_bin)
    avg_conf = mean(r['confidence'] for r in in_bin)
    bins_out.append({
      'range': [low, high],
      'n': len(in_bin),
      'hit_rate': round(hr, 3),
      'avg_confidence': round(avg_conf, 3),
      'error': round(abs(avg_conf - hr), 3),
    })

  # Overall calibration error = weighted mean |conf - hit_rate| across non-empty bins
  weighted = [(b['n'], b['error']) for b in bins_out if b['n'] > 0]
  total_n = sum(n for n, _ in weighted)
  cal_err = (sum(n * e for n, e in weighted) / total_n) if total_n else None

  if cal_err is None:
    quality = 'unknown'
  elif cal_err < 0.10:
    quality = 'well_calibrated'
  elif cal_err < 0.20:
    quality = 'drifted'
  else:
    quality = 'badly_miscalibrated'

  return {
    'bins': bins_out,
    'overall_calibration_error': round(cal_err, 3) if cal_err is not None else None,
    'quality': quality,
    'sample_size': total_n,
  }
