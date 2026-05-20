"""Round 4 item 2: insider transactions self-documenting window.

The tool's existing aggregation is correct (totals, per-insider net_shares,
recent_30d/90d). The problem found during the AAPL fact-check is that the
response has no explicit `period_start`/`period_end`, so the Bear agent
fabricated qualifiers like "since Q1 2024" without any way to verify.

Fix: include the actual date range of the transactions present in the data
so callers (Bull/Bear) can cite the right window.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.news_agregator.finnhub_server import _condense_insider_data


def test_response_includes_period_start_and_end():
  """Synthetic transactions across a known date range. Verify the condenser
  exposes period_start/period_end so downstream agents don't fabricate
  qualifiers."""
  raw = {"data": [
    {"name": "COOK", "change": -50000, "transactionCode": "S",
     "transactionDate": "2025-12-15"},
    {"name": "COOK", "change": -60000, "transactionCode": "S",
     "transactionDate": "2026-04-01"},
    {"name": "LEVINSON", "change": -340000, "transactionCode": "S",
     "transactionDate": "2025-09-30"},
  ]}
  out = _condense_insider_data(raw)
  assert 'period_start' in out, \
    f"missing period_start key — agents will fabricate qualifiers: {sorted(out.keys())}"
  assert 'period_end' in out, f"missing period_end key: {sorted(out.keys())}"
  assert out['period_start'] == '2025-09-30', \
    f"period_start should be earliest tx date; got {out['period_start']!r}"
  assert out['period_end'] == '2026-04-01', \
    f"period_end should be latest tx date; got {out['period_end']!r}"
  print(f"PASS: window labeled {out['period_start']} -> {out['period_end']}")


def test_existing_aggregation_unchanged():
  """Regression: the previously-correct totals must still compute."""
  raw = {"data": [
    {"name": "COOK", "change": -50000, "transactionCode": "S", "transactionDate": "2025-12-15"},
    {"name": "COOK", "change": -60000, "transactionCode": "S", "transactionDate": "2026-04-01"},
    {"name": "LEVINSON", "change": -340000, "transactionCode": "S", "transactionDate": "2025-09-30"},
  ]}
  out = _condense_insider_data(raw)
  assert out['total_sold'] == 450000
  assert out['total_bought'] == 0
  assert out['net_shares'] == -450000
  assert out['sell_count'] == 3
  cook = next(i for i in out['top_insiders'] if i['name'] == 'COOK')
  assert cook['net_shares'] == -110000
  assert cook['transaction_count'] == 2
  print("PASS: aggregation totals unchanged (regression guard)")


def test_empty_input_returns_empty_window():
  out = _condense_insider_data({"data": []})
  assert out['period_start'] is None
  assert out['period_end'] is None
  assert out['total_bought'] == 0 and out['total_sold'] == 0
  print("PASS: empty input -> period_start/end = None")


def test_malformed_dates_dont_break_window():
  raw = {"data": [
    {"name": "X", "change": -100, "transactionCode": "S", "transactionDate": "not-a-date"},
    {"name": "Y", "change": -200, "transactionCode": "S", "transactionDate": "2025-12-01"},
  ]}
  out = _condense_insider_data(raw)
  # Should ignore the malformed date and still report the valid one
  assert out['period_start'] == '2025-12-01'
  assert out['period_end'] == '2025-12-01'
  print("PASS: malformed dates ignored gracefully")


def test_window_label_documents_recent_buckets():
  """recent_30d / recent_90d should remain as named buckets with no
  ambiguity about what 'recent' means. The data window covered may be
  much longer than 90 days — the explicit period_start/end disambiguates."""
  raw = {"data": [
    {"name": "X", "change": -100, "transactionCode": "S", "transactionDate": "2025-01-15"},
    {"name": "X", "change": -200, "transactionCode": "S", "transactionDate": "2026-05-01"},
  ]}
  out = _condense_insider_data(raw)
  assert 'recent_30d' in out and 'recent_90d' in out
  assert out['period_start'] == '2025-01-15'
  assert out['period_end'] == '2026-05-01'
  # The whole window is ~16 months. Recent_30/90 should only contain the 2026 tx.
  print(f"PASS: full window {out['period_start']}->{out['period_end']} "
        "coexists with recent_30d/90d windows")


if __name__ == "__main__":
  test_response_includes_period_start_and_end()
  test_existing_aggregation_unchanged()
  test_empty_input_returns_empty_window()
  test_malformed_dates_dont_break_window()
  test_window_label_documents_recent_buckets()
  print("\nAll Round 4 / item 2 insider fidelity tests passed.")
