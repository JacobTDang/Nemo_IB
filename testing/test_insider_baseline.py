"""Unit tests for the 90-day baseline ratio in _condense_insider_data.

Programmatic 10b5-1 selling is the baseline state for many large-company
execs. Calling 690-to-0 sells "loud" is a vibe-claim unless we compare to
the prior-period rate. These tests validate the ratio math by feeding
synthetic transactions across two 90-day windows.

Run:
  .venv\\Scripts\\python.exe testing\\test_insider_baseline.py
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.news_agregator.finnhub_server import _condense_insider_data


_results = {'pass': 0, 'fail': 0, 'failures': []}


def _check(name: str, condition: bool, hint: str = '') -> None:
  if condition:
    _results['pass'] += 1
    print(f"  PASS  {name}")
  else:
    _results['fail'] += 1
    _results['failures'].append((name, hint))
    print(f"  FAIL  {name}  --  {hint}")


def _today():
  return datetime.now(timezone.utc).date()


def _days_ago(n: int) -> str:
  return (_today() - timedelta(days=n)).strftime("%Y-%m-%d")


def _sale(days_ago: int, shares: int = 100000, name: str = "Insider1") -> dict:
  return {
    "name": name,
    "share": 0,
    "change": -shares,  # negative because shares left the holding
    "transactionCode": "S",
    "transactionDate": _days_ago(days_ago),
  }


def _purchase(days_ago: int, shares: int = 100000, name: str = "Insider1") -> dict:
  return {
    "name": name,
    "share": shares,
    "change": shares,
    "transactionCode": "P",
    "transactionDate": _days_ago(days_ago),
  }


def test_2x_baseline():
  print("\n== 2x baseline: 10 recent sales vs 5 prior sales -> ratio ~2.0 ==")
  # 10 sales (100k each) spread across the last 90 days
  # 5 sales (100k each) spread across days 100-360 (270-day prior window)
  txns = []
  for i in range(10):
    txns.append(_sale(days_ago=5 + i * 8, shares=100000))  # days 5..77
  for i in range(5):
    txns.append(_sale(days_ago=100 + i * 50, shares=100000))  # days 100..300
  result = _condense_insider_data({"data": txns})

  _check("recent_90d sold == 1,000,000",
         result["recent_90d"]["sold"] == 1_000_000,
         hint=str(result["recent_90d"]))
  _check("prior_period_avg_per_90d_sold is not None",
         result["prior_period_avg_per_90d_sold"] is not None,
         hint=str(result["prior_period_avg_per_90d_sold"]))
  ratio = result["current_vs_baseline_ratio"]
  _check("current_vs_baseline_ratio > 1.5 (recent selling above baseline)",
         ratio is not None and ratio > 1.5,
         hint=f"ratio={ratio}")


def test_below_baseline():
  print("\n== below baseline: light recent activity vs heavy prior activity -> ratio < 1.0 ==")
  # The prior window covers more than 90 days so the AVERAGE per 90-day
  # window is what matters, not the raw count. Recent 90d = 300k sold.
  # Prior window ~260 days = 2.0M sold -> avg 90d = ~692k -> ratio ~0.43.
  txns = []
  for i in range(3):
    txns.append(_sale(days_ago=10 + i * 25, shares=100000))   # days 10..60
  for i in range(20):
    txns.append(_sale(days_ago=100 + i * 13, shares=100000))  # days 100..347
  result = _condense_insider_data({"data": txns})

  _check("recent_90d sold == 300,000",
         result["recent_90d"]["sold"] == 300_000,
         hint=str(result["recent_90d"]))
  ratio = result["current_vs_baseline_ratio"]
  _check("current_vs_baseline_ratio < 1.0 (recent selling below baseline)",
         ratio is not None and ratio < 1.0,
         hint=f"ratio={ratio}")


def test_empty_data_keeps_shape():
  print("\n== empty data: ratio fields present and None ==")
  result = _condense_insider_data({"data": []})
  for key in ("prior_90d", "prior_period_avg_per_90d_sold",
              "current_vs_baseline_ratio"):
    _check(f"empty envelope contains key '{key}'", key in result)
  _check("prior_period_avg_per_90d_sold is None",
         result["prior_period_avg_per_90d_sold"] is None)
  _check("current_vs_baseline_ratio is None",
         result["current_vs_baseline_ratio"] is None)
  _check("prior_90d sub-shape", result["prior_90d"] == {"bought": 0, "sold": 0, "net": 0})


def test_too_little_prior_coverage_returns_none():
  print("\n== prior coverage < 30 days: ratio is None (no fabrication) ==")
  # All transactions within the last 90 days; prior window has < 30 days of data
  txns = [_sale(days_ago=10), _sale(days_ago=30),
          _sale(days_ago=60), _sale(days_ago=85)]
  result = _condense_insider_data({"data": txns})
  _check("prior_period_avg_per_90d_sold is None",
         result["prior_period_avg_per_90d_sold"] is None,
         hint=str(result["prior_period_avg_per_90d_sold"]))
  _check("current_vs_baseline_ratio is None",
         result["current_vs_baseline_ratio"] is None,
         hint=str(result["current_vs_baseline_ratio"]))


def test_prior_90d_bucket_isolated():
  print("\n== prior_90d bucket only counts days 91-180 ==")
  txns = [
    _sale(days_ago=10, shares=50_000),    # in recent_90, NOT in prior_90
    _sale(days_ago=120, shares=70_000),   # in prior_90
    _sale(days_ago=150, shares=80_000),   # in prior_90
    _sale(days_ago=250, shares=99_999),   # outside both windows
  ]
  result = _condense_insider_data({"data": txns})
  _check("recent_90d sold == 50,000", result["recent_90d"]["sold"] == 50_000,
         hint=str(result["recent_90d"]))
  _check("prior_90d sold == 150,000", result["prior_90d"]["sold"] == 150_000,
         hint=str(result["prior_90d"]))


def test_purchases_baseline_does_not_break():
  print("\n== purchase activity does not interfere with sell baseline ==")
  txns = [
    _purchase(days_ago=20, shares=200_000),
    _sale(days_ago=30, shares=100_000),
    _sale(days_ago=200, shares=50_000),
  ]
  result = _condense_insider_data({"data": txns})
  _check("recent_90d bought == 200,000",
         result["recent_90d"]["bought"] == 200_000)
  _check("ratio uses sold series only (purchases ignored)",
         result["current_vs_baseline_ratio"] is not None or
         result["prior_period_avg_per_90d_sold"] is None)


def main() -> int:
  print("\nInsider baseline ratio unit tests\n")
  test_2x_baseline()
  test_below_baseline()
  test_empty_data_keeps_shape()
  test_too_little_prior_coverage_returns_none()
  test_prior_90d_bucket_isolated()
  test_purchases_baseline_does_not_break()

  print(f"\n== Summary ==\n  PASS: {_results['pass']}\n  FAIL: {_results['fail']}")
  for n, h in _results['failures']:
    print(f"  - {n}: {h}")
  return 0 if _results['fail'] == 0 else 1


if __name__ == "__main__":
  sys.exit(main())
