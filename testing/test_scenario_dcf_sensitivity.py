"""Unit tests for the terminal-multiple sensitivity sweep in _scenario_dcf_math.

The main bear/base/bull output uses the conservative min(perpetuity, exit_multiple)
terminal value. The new terminal_sensitivity field strips the perpetuity floor and
shows pure exit-multiple PT across five multiples per scenario, so the analyst can
see how load-bearing the terminal multiple assumption is.

Run:
  .venv\\Scripts\\python.exe testing\\test_scenario_dcf_sensitivity.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.financial_modeling_engine.analysis_tools import _scenario_dcf_math


_results = {'pass': 0, 'fail': 0, 'failures': []}


def _check(name: str, condition: bool, hint: str = '') -> None:
  if condition:
    _results['pass'] += 1
    print(f"  PASS  {name}")
  else:
    _results['fail'] += 1
    _results['failures'].append((name, hint))
    print(f"  FAIL  {name}  --  {hint}")


def _base_inputs(terminal_multiple=15.0):
  return dict(
    revenue_base=100.0,
    capex_pct_revenue=0.05,
    tax_rate=0.21,
    depreciation=0.04,
    wacc=0.10,
    terminal_growth=0.025,
    terminal_multiple=terminal_multiple,
    cash=20.0,
    debt=10.0,
    shares_outstanding=10.0,
    ticker='TEST',
  )


def _run(terminal_multiple=15.0):
  return _scenario_dcf_math(
    _base_inputs(terminal_multiple=terminal_multiple),
    bear_growth=[0.05, 0.04, 0.03, 0.02, 0.02],
    base_growth=[0.10, 0.08, 0.06, 0.05, 0.04],
    bull_growth=[0.15, 0.12, 0.10, 0.08, 0.06],
    bear_margin=0.18, base_margin=0.22, bull_margin=0.26,
  )


def test_sensitivity_present_when_exit_multiple_positive():
  print("\n== sensitivity present when terminal_multiple > 0 ==")
  result = _run(terminal_multiple=15.0)
  _check("terminal_sensitivity key exists",
         'terminal_sensitivity' in result)
  _check("terminal_sensitivity_base_multiple == 15.0",
         result.get('terminal_sensitivity_base_multiple') == 15.0)


def test_sensitivity_shape():
  print("\n== sensitivity is a 3-scenario x 5-multiple table ==")
  result = _run(terminal_multiple=15.0)
  s = result['terminal_sensitivity']
  _check("has bear / base / bull keys",
         set(s.keys()) == {'bear', 'base', 'bull'},
         hint=str(set(s.keys())))
  expected_multiples = {'11.0x', '13.0x', '15.0x', '17.0x', '19.0x'}
  for case in ('bear', 'base', 'bull'):
    _check(f"{case} row has 5 multiples {expected_multiples}",
           set(s[case].keys()) == expected_multiples,
           hint=str(set(s[case].keys())))


def test_sensitivity_monotonic_in_multiple():
  print("\n== sensitivity is monotonically increasing in multiple ==")
  result = _run(terminal_multiple=15.0)
  for case in ('bear', 'base', 'bull'):
    row = result['terminal_sensitivity'][case]
    px = [row['11.0x'], row['13.0x'], row['15.0x'], row['17.0x'], row['19.0x']]
    is_sorted = all(px[i] <= px[i + 1] for i in range(len(px) - 1))
    spread = max(px) - min(px)
    _check(f"{case}: prices increase with multiple",
           is_sorted, hint=f"px={px}")
    _check(f"{case}: spread > 0 (multiple is doing work)",
           spread > 0, hint=f"spread={spread}")


def test_sensitivity_uses_pure_exit_multiple_not_min():
  print("\n== sensitivity strips perpetuity floor ==")
  # With this profile the perpetuity floor binds in the main output (the
  # _dcf_math min() picks the lower of growth-perpetuity and exit-multiple).
  # The sensitivity row at the base multiple should therefore be >= the main
  # base PT, and meaningfully different from it.
  result = _run(terminal_multiple=15.0)
  main_base = result['base']['price_per_share']
  sens_base_at_base = result['terminal_sensitivity']['base']['15.0x']
  _check("sensitivity@15x >= main base PT (no perpetuity floor)",
         sens_base_at_base >= main_base,
         hint=f"main={main_base} sens@15x={sens_base_at_base}")
  _check("sensitivity@15x meaningfully differs from main base PT",
         abs(sens_base_at_base - main_base) > 0.5,
         hint=f"diff={abs(sens_base_at_base - main_base):.2f}")


def test_perpetuity_only_mode_skips_sensitivity():
  print("\n== perpetuity-only mode (terminal_multiple=0) skips sensitivity ==")
  result = _run(terminal_multiple=0)
  _check("terminal_sensitivity NOT present when terminal_multiple == 0",
         'terminal_sensitivity' not in result)
  _check("terminal_sensitivity_base_multiple NOT present",
         'terminal_sensitivity_base_multiple' not in result)


def test_existing_keys_still_present():
  print("\n== additive change: existing return shape unchanged ==")
  result = _run(terminal_multiple=15.0)
  for k in ('bear', 'base', 'bull', 'price_range'):
    _check(f"key '{k}' still present", k in result)
  for k in ('low', 'mid', 'high'):
    _check(f"price_range.{k} still present", k in result['price_range'])
  for case in ('bear', 'base', 'bull'):
    for sub in ('price_per_share', 'enterprise_value', 'equity_value',
                'pv_terminal_value', 'revenue_growth_y1_pct', 'ebitda_margin_pct'):
      _check(f"{case}.{sub} still present", sub in result[case])


def main() -> int:
  print("\nScenario DCF terminal-multiple sensitivity tests\n")
  test_sensitivity_present_when_exit_multiple_positive()
  test_sensitivity_shape()
  test_sensitivity_monotonic_in_multiple()
  test_sensitivity_uses_pure_exit_multiple_not_min()
  test_perpetuity_only_mode_skips_sensitivity()
  test_existing_keys_still_present()

  print(f"\n== Summary ==\n  PASS: {_results['pass']}\n  FAIL: {_results['fail']}")
  for n, h in _results['failures']:
    print(f"  - {n}: {h}")
  return 0 if _results['fail'] == 0 else 1


if __name__ == "__main__":
  sys.exit(main())
