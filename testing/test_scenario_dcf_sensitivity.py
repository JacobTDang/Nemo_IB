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

Why these are asserts and not prints
------------------------------------
Every check below used to route through a local `_check(name, condition,
hint)` helper that incremented a counter and printed PASS or FAIL. It never
raised. Under pytest each `test_*` function ran to completion and returned
None, so the file reported six passing tests no matter what the code did --
forty-odd checks that could not fail, wearing the appearance of a gate. It was
silently green through a change that broke four of them; that was only found
by running the file as a script and reading the summary line, which nothing in
CI does.

A gate that cannot fail is not a gate. The hint strings the helper printed are
kept as assertion messages, so a failure still says which multiple, which
scenario and which two numbers disagreed -- the diagnostic value survives, the
silence does not.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.financial_modeling_engine.analysis_tools import _scenario_dcf_math


SCENARIOS = ('bear', 'base', 'bull')
EXPECTED_MULTIPLES = {'11.0x', '13.0x', '15.0x', '17.0x', '19.0x'}
SWEEP_ORDER = ('11.0x', '13.0x', '15.0x', '17.0x', '19.0x')


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


@pytest.fixture(scope='module')
def result():
  """The one payload every check below reads, priced at a 15x exit multiple."""
  return _run(terminal_multiple=15.0)


def test_sensitivity_present_when_exit_multiple_positive(result):
  assert 'terminal_sensitivity' in result, (
    "no terminal_sensitivity grid was returned for terminal_multiple=15.0")
  assert result.get('terminal_sensitivity_base_multiple') == 15.0, (
    "the grid does not state the multiple it is centred on: "
    f"{result.get('terminal_sensitivity_base_multiple')!r}")


def test_sensitivity_shape(result):
  """A 3-scenario x 5-multiple table, or the sweep is not a sweep."""
  s = result['terminal_sensitivity']
  assert set(s.keys()) == set(SCENARIOS), str(set(s.keys()))
  for case in SCENARIOS:
    assert set(s[case].keys()) == EXPECTED_MULTIPLES, (
      f"{case} row is not the {sorted(EXPECTED_MULTIPLES)} sweep: "
      f"{sorted(s[case].keys())}")


@pytest.mark.parametrize('case', SCENARIOS)
def test_sensitivity_monotonic_in_multiple(result, case):
  """A higher exit multiple can never price the same company lower."""
  row = result['terminal_sensitivity'][case]
  px = [row[k] for k in SWEEP_ORDER]
  assert all(px[i] <= px[i + 1] for i in range(len(px) - 1)), (
    f"{case}: price falls as the multiple rises, px={px}")


@pytest.mark.parametrize('case', SCENARIOS)
def test_a_flat_row_says_why_it_is_flat(result, case):
  """A flat row is an answer, not a bug -- but only if it says so.

  The perpetuity floor binding at every multiple in the sweep means the exit
  multiple is setting nothing, which is the true answer. Unexplained it reads
  as a broken table, and the reader's repair is to distrust the grid rather
  than the assumption.
  """
  row = result['terminal_sensitivity'][case]
  px = [row[k] for k in SWEEP_ORDER]
  if max(px) - min(px) != 0:
    pytest.skip(f"{case} row is not flat: px={px}")
  assert result.get('terminal_sensitivity_floor_note'), (
    f"{case}: flat row with no floor note, px={px}")


@pytest.mark.parametrize('case', SCENARIOS)
def test_sensitivity_agrees_with_the_headline_at_the_base_multiple(result, case):
  """The one cell whose answer is already known.

  Priced on a different terminal method it came out at twice the headline;
  priced on the same one it has to reproduce it exactly.
  """
  main = result[case]['price_per_share']
  cell = result['terminal_sensitivity'][case]['15.0x']
  assert abs(cell - main) <= 0.02, (
    f"{case}: sensitivity@15x != headline PT -- main={main} sens@15x={cell}")


def test_the_grid_states_the_rule_it_applies(result):
  method = result.get('terminal_sensitivity_method') or ''
  assert 'min' in method.lower(), (
    "the grid does not name the min(perpetuity, exit_multiple) rule it "
    f"applies: {str(result.get('terminal_sensitivity_method'))[:80]}")


def test_perpetuity_only_mode_skips_sensitivity():
  """terminal_multiple=0 means there is no multiple to sweep."""
  result = _run(terminal_multiple=0)
  assert 'terminal_sensitivity' not in result, (
    "a sensitivity grid was returned for a model with no exit multiple")
  assert 'terminal_sensitivity_base_multiple' not in result, (
    "a base multiple was reported for a model with no exit multiple")


@pytest.mark.parametrize('key', ('bear', 'base', 'bull', 'price_range'))
def test_existing_top_level_keys_still_present(result, key):
  """The grid was an additive change; the existing return shape is unchanged."""
  assert key in result, f"key '{key}' disappeared from the payload"


@pytest.mark.parametrize('key', ('low', 'mid', 'high'))
def test_price_range_keys_still_present(result, key):
  assert key in result['price_range'], f"price_range.{key} disappeared"


@pytest.mark.parametrize('case', SCENARIOS)
@pytest.mark.parametrize('sub', ('price_per_share', 'enterprise_value',
                                 'equity_value', 'pv_terminal_value',
                                 'revenue_growth_y1_pct', 'ebitda_margin_pct'))
def test_scenario_keys_still_present(result, case, sub):
  assert sub in result[case], f"{case}.{sub} disappeared"
