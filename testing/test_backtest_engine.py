"""Tests for the backtest engine — correctness of indicators, rule eval,
and trade construction. No network calls in pure-Python tests; the
realworld test uses yfinance with a known liquid ticker."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.backtest_engine import (
    _rsi, _sma, _drawdown_from_peak, _eval_rule,
    backtest_signal, compute_indicators, NAMED_SIGNALS,
)


_results = {'pass': 0, 'fail': 0, 'failures': []}


def _check(name, cond, hint=''):
  if cond:
    _results['pass'] += 1
    print(f"  PASS  {name}")
  else:
    _results['fail'] += 1
    _results['failures'].append((name, hint))
    print(f"  FAIL  {name}  --  {hint}")


def _section(t):
  print(f"\n=== {t} ===")


def test_indicators():
  _section("1. Indicator math")
  # Constant series — RSI should be ~50 (no gains, no losses), but math
  # actually returns 0 or 100 depending on numerator sign. Test with a
  # known trend.
  closes = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
            111, 112, 113, 114, 115]  # steady uptrend
  rsi = _rsi(closes, 14)
  _check("  RSI in steady uptrend ~ 100", rsi[-1] is not None and rsi[-1] > 95,
         f"got {rsi[-1]}")

  closes_down = [100 - i for i in range(20)]
  rsi = _rsi(closes_down, 14)
  _check("  RSI in steady downtrend ~ 0", rsi[-1] is not None and rsi[-1] < 5,
         f"got {rsi[-1]}")

  sma = _sma([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 3)
  _check("  SMA(3) of [1..10] last value = 9", sma[-1] == 9, f"got {sma[-1]}")
  _check("  SMA(3) warmup is None", sma[0] is None and sma[1] is None)

  dd = _drawdown_from_peak([100, 110, 120, 100, 80, 90])
  _check("  drawdown from 120 to 80 ~ -33.33%", abs(dd[4] - -33.33) < 0.5,
         f"got {dd[4]}")


def test_rule_eval():
  _section("2. Rule evaluation")
  ind = {'rsi_14': [None, None, 25, 35, 45], 'close': [100, 105, 99, 102, 108]}

  rule = {'metric': 'rsi_14', 'op': '<', 'value': 30}
  _check("  RSI<30 at i=2 (value 25)", _eval_rule(rule, ind, 2) is True)
  _check("  RSI<30 at i=3 (value 35)", _eval_rule(rule, ind, 3) is False)
  _check("  RSI<30 at i=0 (None)", _eval_rule(rule, ind, 0) is False)

  # Compound AND
  rule2 = {'and': [
    {'metric': 'rsi_14', 'op': '<', 'value': 50},
    {'metric': 'close',  'op': '>', 'value': 100},
  ]}
  _check("  AND: rsi<50 and close>100 at i=3", _eval_rule(rule2, ind, 3) is True)
  _check("  AND fails when one cond fails at i=2",
         _eval_rule(rule2, ind, 2) is False)

  # OR
  rule3 = {'or': [
    {'metric': 'rsi_14', 'op': '<', 'value': 30},
    {'metric': 'close',  'op': '>', 'value': 107},
  ]}
  _check("  OR at i=4 (close 108 > 107)", _eval_rule(rule3, ind, 4) is True)


def test_backtest_realworld():
  _section("3. Real-world backtest (MSFT, RSI<30)")
  r = backtest_signal('MSFT', NAMED_SIGNALS['oversold_rsi'], hold_days=30,
                      cooldown_days=10, signal_name='oversold_rsi')
  _check("  ran without warning", r.warning is None,
         f"warning: {r.warning}")
  _check("  produced at least 1 trade", r.n_trades >= 1,
         f"got {r.n_trades}")
  _check("  hit_rate is a percentage", r.hit_rate is None or 0 <= r.hit_rate <= 100)
  _check("  date_range populated", bool(r.date_range))
  if r.trades:
    t = r.trades[0]
    _check("  trade entry_price > 0", t.entry_price > 0)
    _check("  trade has hold_days > 0", t.hold_days > 0)
    _check("  trade has matching dates", bool(t.entry_date) and bool(t.exit_date))


def test_no_signal_case():
  _section("4. Signal that never fires")
  # Impossible rule: RSI < -100
  impossible = {'metric': 'rsi_14', 'op': '<', 'value': -100}
  r = backtest_signal('MSFT', impossible, hold_days=10, signal_name='impossible')
  _check("  impossible signal -> 0 trades", r.n_trades == 0,
         f"got {r.n_trades}")
  _check("  no_trades hit_rate is None", r.hit_rate is None)


def main():
  print("\nBacktest Engine — tests\n")
  test_indicators()
  test_rule_eval()
  test_backtest_realworld()
  test_no_signal_case()
  print(f"\n=== Summary ===\n  PASS: {_results['pass']}\n  FAIL: {_results['fail']}")
  if _results['failures']:
    for n, h in _results['failures']:
      print(f"  - {n}: {h}")
  return 0 if _results['fail'] == 0 else 1


if __name__ == "__main__":
  sys.exit(main())
