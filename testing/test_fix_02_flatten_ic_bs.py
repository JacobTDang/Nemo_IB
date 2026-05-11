"""Test Fix #2: get_financial_statements flattens IC and BS, not just CF."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.workflows.execution_engine import _flatten_market_intel


def test_flatten_cf():
  variables = {}
  data = {
    'statement': 'cf', 'freq': 'annual',
    'periods': [{
      'dividendsPaid': -15_000_000_000,
      'repurchaseOfCapitalStock': -50_000_000_000,
      'operatingCashFlow': 100_000_000_000,
      'capitalExpenditures': -10_000_000_000,
      'freeCashFlow': 90_000_000_000,
    }],
  }
  _flatten_market_intel(variables, 'get_financial_statements', data)
  assert variables['cf.dividendsPaid'] == -15_000_000_000
  assert variables['cf.operatingCashFlow'] == 100_000_000_000
  assert variables['dividendsPaid'] == -15_000_000_000
  print("PASS: CF flatten works")


def test_flatten_ic():
  variables = {}
  data = {
    'statement': 'ic', 'freq': 'annual',
    'periods': [{
      'revenue': 400_000_000_000,
      'costOfRevenue': 200_000_000_000,
      'grossProfit': 200_000_000_000,
      'operatingIncome': 120_000_000_000,
      'netIncome': 100_000_000_000,
      'eps': 6.50,
    }],
  }
  _flatten_market_intel(variables, 'get_financial_statements', data)
  assert variables['ic.revenue'] == 400_000_000_000
  assert variables['ic.grossProfit'] == 200_000_000_000
  assert variables['ic.netIncome'] == 100_000_000_000
  assert abs(variables['ic.grossMargin'] - 0.50) < 0.0001, f"grossMargin should be 0.50, got {variables['ic.grossMargin']}"
  assert abs(variables['ic.netMargin'] - 0.25) < 0.0001, f"netMargin should be 0.25, got {variables['ic.netMargin']}"
  print("PASS: IC flatten works with derived margins")


def test_flatten_bs():
  variables = {'totalDebt': 99_999_999, 'cashAndEquivalents': 88_888_888}  # pre-populated from get_market_data
  data = {
    'statement': 'bs', 'freq': 'annual',
    'periods': [{
      'totalAssets': 350_000_000_000,
      'totalLiabilities': 280_000_000_000,
      'totalEquity': 70_000_000_000,
      'totalDebt': 110_000_000_000,  # should NOT overwrite flat key
      'longTermDebt': 95_000_000_000,
      'goodwill': 50_000_000_000,
      'cashAndEquivalents': 30_000_000_000,  # should NOT overwrite flat key
    }],
  }
  _flatten_market_intel(variables, 'get_financial_statements', data)
  assert variables['bs.totalAssets'] == 350_000_000_000
  assert variables['bs.totalEquity'] == 70_000_000_000
  assert variables['bs.totalDebt'] == 110_000_000_000  # namespaced version IS written
  # Flat keys are preserved (no overwrite)
  assert variables['totalDebt'] == 99_999_999, "flat totalDebt should NOT be overwritten"
  assert variables['cashAndEquivalents'] == 88_888_888, "flat cashAndEquivalents should NOT be overwritten"
  # Other fields DO get flat keys
  assert variables['totalAssets'] == 350_000_000_000
  print("PASS: BS flatten works with collision protection")


def test_empty_periods():
  variables = {}
  _flatten_market_intel(variables, 'get_financial_statements', {'statement': 'ic', 'periods': []})
  assert variables == {}, "empty periods should write nothing"
  print("PASS: empty periods handled gracefully")


if __name__ == "__main__":
  test_flatten_cf()
  test_flatten_ic()
  test_flatten_bs()
  test_empty_periods()
  print("\nAll tests passed.")
