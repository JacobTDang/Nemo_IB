"""Unit tests for the terminal-multiple sensitivity sweep in _scenario_dcf_math.

Every price in the payload -- the headline bear/base/bull PT and every cell of
the terminal_sensitivity grid -- is built on the same conservative
min(perpetuity, exit_multiple) terminal value, so the grid is a real
sensitivity of the headline and the cell at the base multiple reproduces it.

The grid used to strip the perpetuity floor and price the pure exit multiple
instead, which put two prices from two methods side by side with nothing to
tell them apart: NVDA's base case read 151.96 beside a 25x cell of 303.54.
See testing/test_scenario_grid_matches_its_headline.py for that defect.

The sweep still answers how load-bearing the terminal multiple is. Where the
perpetuity binds across the whole sweep the row is flat, which is the true
answer -- the multiple is setting nothing -- and terminal_sensitivity_floor_note
says so.

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
  print("\n== sensitivity is monotonically non-decreasing in multiple ==")
  result = _run(terminal_multiple=15.0)
  for case in ('bear', 'base', 'bull'):
    row = result['terminal_sensitivity'][case]
    px = [row['11.0x'], row['13.0x'], row['15.0x'], row['17.0x'], row['19.0x']]
    is_sorted = all(px[i] <= px[i + 1] for i in range(len(px) - 1))
    _check(f"{case}: prices never fall as the multiple rises",
           is_sorted, hint=f"px={px}")
    # A flat row is an answer, not a bug: the perpetuity is binding at every
    # multiple in the sweep, so the exit multiple sets nothing. It has to say
    # so rather than look like a broken table.
    if max(px) - min(px) == 0:
      _check(f"{case}: a flat row is explained",
             bool(result.get('terminal_sensitivity_floor_note')),
             hint="flat row with no floor note")


def test_sensitivity_agrees_with_the_headline_at_the_base_multiple():
  print("\n== sensitivity is a sensitivity of the headline ==")
  # The one cell whose answer is already known. Priced on a different terminal
  # method it came out at twice the headline; priced on the same one it has to
  # reproduce it exactly.
  result = _run(terminal_multiple=15.0)
  for case in ('bear', 'base', 'bull'):
    main = result[case]['price_per_share']
    cell = result['terminal_sensitivity'][case]['15.0x']
    _check(f"{case}: sensitivity@15x == headline PT",
           abs(cell - main) <= 0.02,
           hint=f"main={main} sens@15x={cell}")
  _check("the grid states the rule it applies",
         'min' in (result.get('terminal_sensitivity_method') or '').lower(),
         hint=str(result.get('terminal_sensitivity_method'))[:80])


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
  test_sensitivity_agrees_with_the_headline_at_the_base_multiple()
  test_perpetuity_only_mode_skips_sensitivity()
  test_existing_keys_still_present()

  print(f"\n== Summary ==\n  PASS: {_results['pass']}\n  FAIL: {_results['fail']}")
  for n, h in _results['failures']:
    print(f"  - {n}: {h}")
  return 0 if _results['fail'] == 0 else 1


if __name__ == "__main__":
  sys.exit(main())
