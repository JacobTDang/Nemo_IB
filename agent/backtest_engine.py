"""Lightweight backtest engine.

Given a signal definition + a universe of tickers + a date range, evaluate
how the signal would have performed historically. The engine is designed
for *single-asset signal validation* — not portfolio-level backtesting —
which is the appropriate granularity for thesis-pattern validation.

Two signal definition modes:

  1. Threshold rule (deterministic): e.g.,
       {"metric": "rsi_14", "op": "<", "value": 30, "hold_days": 30}
     "When RSI(14) crosses below 30, hold long for 30 trading days."

  2. Multi-condition rule: e.g.,
       {"and": [
           {"metric": "ttm_revenue_growth", "op": ">", "value": 20},
           {"metric": "fcf_yield", "op": ">", "value": 0.03},
       ], "hold_days": 252}

  3. Callable: pass a Python function (ticker, date, price_history) -> bool
     for arbitrary logic (used by tests).

Output metrics per trigger:
  - entry_date, entry_price, exit_date, exit_price
  - holding_period_days, return_pct
Plus aggregate:
  - n_trades, hit_rate, mean_return, median_return,
    max_drawdown, sharpe (simplified), worst_trade, best_trade
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


# ---------------------------------------------------------------------------
# Signal definition types
# ---------------------------------------------------------------------------

# A signal is either a callable, a single rule, or an {and: [...]} / {or: [...]}
SignalCallable = Callable[[str, Any, Any], bool]
SignalRule = Dict[str, Any]


# ---------------------------------------------------------------------------
# Metric computation — operates on a daily-bar DataFrame from yfinance
# ---------------------------------------------------------------------------

def _rsi(closes, period: int = 14) -> List[Optional[float]]:
  """Wilder's RSI on a pandas Series. Returns python list (None for warmup)."""
  vals = closes.tolist() if hasattr(closes, 'tolist') else list(closes)
  if len(vals) < period + 1:
    return [None] * len(vals)
  out: List[Optional[float]] = [None] * len(vals)
  gains = []
  losses = []
  for i in range(1, len(vals)):
    diff = vals[i] - vals[i - 1]
    gains.append(max(diff, 0))
    losses.append(max(-diff, 0))
  # Initial avg
  avg_gain = sum(gains[:period]) / period
  avg_loss = sum(losses[:period]) / period
  for i in range(period, len(vals)):
    if i > period:
      avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
      avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period
    rs = avg_gain / avg_loss if avg_loss > 0 else float('inf')
    rsi = 100 - (100 / (1 + rs))
    out[i] = rsi
  return out


def _sma(closes, period: int) -> List[Optional[float]]:
  vals = closes.tolist() if hasattr(closes, 'tolist') else list(closes)
  out: List[Optional[float]] = [None] * len(vals)
  for i in range(period - 1, len(vals)):
    out[i] = sum(vals[i - period + 1: i + 1]) / period
  return out


def _drawdown_from_peak(closes) -> List[float]:
  vals = closes.tolist() if hasattr(closes, 'tolist') else list(closes)
  out = []
  peak = -float('inf')
  for v in vals:
    if v > peak:
      peak = v
    out.append((v / peak - 1) * 100 if peak > 0 else 0)
  return out


def compute_indicators(history) -> Dict[str, List[Optional[float]]]:
  """Given a pandas DataFrame with Open/High/Low/Close/Volume columns
  (as returned by yfinance), compute a panel of common indicators.

  Returns a dict keyed by indicator name, each value a list aligned
  with the input rows (None for warmup periods)."""
  close = history['Close']
  return {
    'close':        close.tolist() if hasattr(close, 'tolist') else list(close),
    'rsi_14':       _rsi(close, 14),
    'sma_20':       _sma(close, 20),
    'sma_50':       _sma(close, 50),
    'sma_200':      _sma(close, 200),
    'drawdown_pct': _drawdown_from_peak(close),
  }


# ---------------------------------------------------------------------------
# Rule evaluation
# ---------------------------------------------------------------------------

_OPS = {
  '>':  lambda a, b: a is not None and a > b,
  '<':  lambda a, b: a is not None and a < b,
  '>=': lambda a, b: a is not None and a >= b,
  '<=': lambda a, b: a is not None and a <= b,
  '==': lambda a, b: a is not None and abs(a - b) < 1e-9,
  '!=': lambda a, b: a is not None and abs(a - b) >= 1e-9,
}


def _eval_rule(rule: SignalRule, indicators: Dict[str, List], i: int) -> bool:
  """Evaluate a rule at row index i."""
  if 'and' in rule:
    return all(_eval_rule(r, indicators, i) for r in rule['and'])
  if 'or' in rule:
    return any(_eval_rule(r, indicators, i) for r in rule['or'])
  metric = rule.get('metric')
  op = rule.get('op')
  value = rule.get('value')
  if metric is None or op not in _OPS or value is None:
    return False
  series = indicators.get(metric)
  if series is None or i >= len(series):
    return False
  v = series[i]
  return _OPS[op](v, value)


# ---------------------------------------------------------------------------
# Backtest core
# ---------------------------------------------------------------------------

@dataclass
class Trade:
  ticker:           str
  entry_idx:        int
  entry_date:       str
  entry_price:      float
  exit_idx:         int
  exit_date:        str
  exit_price:       float
  hold_days:        int
  return_pct:       float
  max_dd_in_trade:  float  # worst drawdown during the holding period


@dataclass
class BacktestResult:
  ticker:           str
  signal_name:      str
  n_trades:         int
  hit_rate:         Optional[float]    # % of trades with return > 0
  mean_return:      Optional[float]
  median_return:    Optional[float]
  best_trade:       Optional[float]
  worst_trade:      Optional[float]
  mean_hold_days:   Optional[float]
  max_drawdown_pct: Optional[float]
  sharpe_simple:    Optional[float]    # mean / std of returns (annualized)
  trades:           List[Trade] = field(default_factory=list)
  date_range:       Optional[str] = None
  warning:          Optional[str] = None


def backtest_signal(
  ticker: str,
  signal: Union[SignalRule, SignalCallable],
  hold_days: int = 30,
  start_date: Optional[str] = None,
  end_date: Optional[str] = None,
  cooldown_days: int = 0,
  signal_name: str = "unnamed",
) -> BacktestResult:
  """Run a single-asset backtest.

  - `signal`: a rule dict (e.g. {"metric": "rsi_14", "op": "<", "value": 30})
    or a callable (ticker, idx, indicators) -> bool
  - `hold_days`: how many trading days to hold after a signal fires
  - `cooldown_days`: minimum bars between entries (prevents overlap)

  Fetches price history via yfinance. If yfinance is unreachable, returns
  a result with `warning` set.
  """
  try:
    import yfinance as yf
  except ImportError:
    return BacktestResult(
      ticker=ticker, signal_name=signal_name, n_trades=0,
      hit_rate=None, mean_return=None, median_return=None, best_trade=None,
      worst_trade=None, mean_hold_days=None, max_drawdown_pct=None,
      sharpe_simple=None, warning='yfinance not installed',
    )

  try:
    t = yf.Ticker(ticker)
    kwargs = {}
    if start_date and end_date:
      kwargs['start'] = start_date
      kwargs['end'] = end_date
    else:
      kwargs['period'] = '5y'
    history = t.history(auto_adjust=True, **kwargs)
  except Exception as e:
    return BacktestResult(
      ticker=ticker, signal_name=signal_name, n_trades=0,
      hit_rate=None, mean_return=None, median_return=None, best_trade=None,
      worst_trade=None, mean_hold_days=None, max_drawdown_pct=None,
      sharpe_simple=None, warning=f'history fetch failed: {type(e).__name__}: {e}',
    )

  if history is None or history.empty or len(history) < 200:
    return BacktestResult(
      ticker=ticker, signal_name=signal_name, n_trades=0,
      hit_rate=None, mean_return=None, median_return=None, best_trade=None,
      worst_trade=None, mean_hold_days=None, max_drawdown_pct=None,
      sharpe_simple=None, warning='insufficient history (<200 bars)',
    )

  indicators = compute_indicators(history)
  closes = indicators['close']
  date_index = [d.strftime('%Y-%m-%d') for d in history.index]

  # Resolve signal evaluator
  if callable(signal):
    is_triggered = lambda idx: signal(ticker, idx, indicators)
  else:
    is_triggered = lambda idx: _eval_rule(signal, indicators, idx)

  trades: List[Trade] = []
  i = 0
  while i < len(closes) - hold_days:
    if is_triggered(i):
      entry_idx = i
      exit_idx = min(i + hold_days, len(closes) - 1)
      entry_price = closes[entry_idx]
      exit_price = closes[exit_idx]
      if entry_price is None or exit_price is None or entry_price == 0:
        i += 1
        continue
      ret = ((exit_price / entry_price) - 1) * 100
      # Intra-trade drawdown
      sub = closes[entry_idx:exit_idx + 1]
      peak = sub[0]
      max_dd = 0.0
      for v in sub:
        if v > peak:
          peak = v
        dd = (v / peak - 1) * 100 if peak > 0 else 0
        if dd < max_dd:
          max_dd = dd
      trades.append(Trade(
        ticker=ticker,
        entry_idx=entry_idx,
        entry_date=date_index[entry_idx],
        entry_price=float(entry_price),
        exit_idx=exit_idx,
        exit_date=date_index[exit_idx],
        exit_price=float(exit_price),
        hold_days=exit_idx - entry_idx,
        return_pct=round(ret, 3),
        max_dd_in_trade=round(max_dd, 2),
      ))
      # Skip ahead past the holding window + cooldown so trades don't overlap
      i += hold_days + max(0, cooldown_days)
    else:
      i += 1

  if not trades:
    return BacktestResult(
      ticker=ticker, signal_name=signal_name, n_trades=0,
      hit_rate=None, mean_return=None, median_return=None, best_trade=None,
      worst_trade=None, mean_hold_days=None, max_drawdown_pct=None,
      sharpe_simple=None, trades=[],
      date_range=f'{date_index[0]} -> {date_index[-1]}',
    )

  returns = [t.return_pct for t in trades]
  hold_days_list = [t.hold_days for t in trades]
  hit_rate = sum(1 for r in returns if r > 0) / len(returns) * 100
  mean_ret = sum(returns) / len(returns)
  sorted_r = sorted(returns)
  median_ret = sorted_r[len(sorted_r) // 2]

  # Sharpe-ish: mean return / std deviation, annualized by sqrt(252/hold_days)
  if len(returns) > 1:
    mean = mean_ret
    var = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    std = math.sqrt(var) if var > 0 else 0
    avg_hold = sum(hold_days_list) / len(hold_days_list)
    sharpe = (mean / std) * math.sqrt(252 / max(avg_hold, 1)) if std > 0 else None
  else:
    sharpe = None

  return BacktestResult(
    ticker=ticker,
    signal_name=signal_name,
    n_trades=len(trades),
    hit_rate=round(hit_rate, 1),
    mean_return=round(mean_ret, 3),
    median_return=round(median_ret, 3),
    best_trade=round(max(returns), 3),
    worst_trade=round(min(returns), 3),
    mean_hold_days=round(sum(hold_days_list) / len(hold_days_list), 1),
    max_drawdown_pct=round(min(t.max_dd_in_trade for t in trades), 2),
    sharpe_simple=round(sharpe, 3) if sharpe is not None else None,
    trades=trades,
    date_range=f'{date_index[0]} -> {date_index[-1]}',
  )


# ---------------------------------------------------------------------------
# Convenience: built-in named signals
# ---------------------------------------------------------------------------

NAMED_SIGNALS = {
  'oversold_rsi':       {'metric': 'rsi_14', 'op': '<', 'value': 30},
  'overbought_rsi':     {'metric': 'rsi_14', 'op': '>', 'value': 70},
  'golden_cross':       {'and': [
                          {'metric': 'sma_50', 'op': '>', 'value': 0},
                          {'metric': 'sma_200', 'op': '>', 'value': 0},
                        ]},  # placeholder — proper golden cross would compare sma_50 vs sma_200
  'big_drawdown':       {'metric': 'drawdown_pct', 'op': '<', 'value': -20},
}
