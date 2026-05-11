"""Test Fix #6 sub: NFCI and consumer sentiment flatten from macro snapshot."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.workflows.execution_engine import _flatten_macro


def test_nfci_and_umich_flatten():
  variables = {}
  data = {
    'DGS10': {'current': 4.5},
    'NFCI': {'current': 0.5},
    'UMCSENT': {'current': 65.0},
  }
  _flatten_macro(variables, 'get_macro_snapshot', data)

  assert variables['macro.NFCI'] == 0.5, f"macro.NFCI should be 0.5, got {variables.get('macro.NFCI')}"
  assert variables['NFCI'] == 0.5, "flat NFCI should also be set"
  assert variables['macro.consumer_sentiment'] == 65.0, f"consumer sentiment wrong: {variables.get('macro.consumer_sentiment')}"
  print(f"PASS: macro.NFCI={variables['macro.NFCI']} consumer_sentiment={variables['macro.consumer_sentiment']}")


def test_missing_nfci_no_error():
  variables = {}
  data = {'DGS10': {'current': 4.5}}  # no NFCI
  _flatten_macro(variables, 'get_macro_snapshot', data)
  assert 'macro.NFCI' not in variables
  print("PASS: missing NFCI handled without error")


if __name__ == "__main__":
  test_nfci_and_umich_flatten()
  test_missing_nfci_no_error()
  print("\nAll tests passed.")
