"""Dividends and splits.

Zero coverage previously, and its absence makes historical per-share
comparisons silently wrong. Comparing NVDA's 2023 EPS to its 2025 EPS without
knowing about the 10:1 split in June 2024 produces a conclusion off by an order
of magnitude, with no error to signal it.
"""
import os
from datetime import datetime

import pandas as pd
import pytest

from tools.financial_modeling_engine import corporate_actions as ca

SKIP_NETWORK = os.environ.get("SKIP_NETWORK_TESTS") == "1"


def network(func):
  """Apply the real `network` marker plus the offline skip.

  This name used to be bound to a bare pytest.mark.skipif. A skipif is not
  a registered marker, so `-m network` and `-m "not network"` collected
  nothing here -- the tests were selectable only by file path.
  """
  func = pytest.mark.network(func)
  return pytest.mark.skipif(SKIP_NETWORK, reason="live yfinance test")(func)


def _series(pairs):
    if not pairs:
        return pd.Series(dtype="float64")
    index = pd.to_datetime([d for d, _ in pairs], utc=True)
    return pd.Series([v for _, v in pairs], index=index)


class _FakeTicker:
    def __init__(self, dividends=None, splits=None):
        self.dividends = _series(dividends or [])
        self.splits = _series(splits or [])


def test_split_ratio_and_date_are_reported(monkeypatch):
    monkeypatch.setattr(ca, "_ticker",
                        lambda t: _FakeTicker(splits=[("2024-06-10", 10.0)]))
    result = ca.get_corporate_actions("NVDA")
    assert result["success"] is True
    assert result["latest_split_ratio"] == 10.0
    assert result["latest_split_date"].startswith("2024-06-10")


def test_most_recent_split_wins(monkeypatch):
    monkeypatch.setattr(ca, "_ticker", lambda t: _FakeTicker(
        splits=[("2021-07-20", 4.0), ("2024-06-10", 10.0)]))
    result = ca.get_corporate_actions("NVDA")
    assert result["latest_split_ratio"] == 10.0


def test_no_splits_reports_none_not_a_ratio_of_one(monkeypatch):
    """A ratio of 1.0 would read as 'a split happened that changed nothing'."""
    monkeypatch.setattr(ca, "_ticker", lambda t: _FakeTicker())
    result = ca.get_corporate_actions("BRK-A")
    assert result["latest_split_ratio"] is None
    assert result["latest_split_date"] is None
    assert result["split_count"] == 0


def test_dividends_are_summed_over_the_trailing_year(monkeypatch):
    this_year = datetime.now().year
    monkeypatch.setattr(ca, "_ticker", lambda t: _FakeTicker(dividends=[
        (f"{this_year}-01-15", 0.25), (f"{this_year}-04-15", 0.25),
        (f"{this_year - 5}-01-15", 0.10),
    ]))
    result = ca.get_corporate_actions("KO")
    assert result["ttm_dividend"] == pytest.approx(0.50)


def test_non_payer_reports_zero_dividends_and_says_so(monkeypatch):
    monkeypatch.setattr(ca, "_ticker", lambda t: _FakeTicker())
    result = ca.get_corporate_actions("GOOGL")
    assert result["pays_dividend"] is False
    assert result["ttm_dividend"] == 0.0


def test_years_window_excludes_older_actions(monkeypatch):
    monkeypatch.setattr(ca, "_ticker", lambda t: _FakeTicker(
        splits=[("1999-01-04", 2.0), ("2024-06-10", 10.0)]))
    result = ca.get_corporate_actions("NVDA", years=5)
    assert result["split_count"] == 1


def test_failure_is_reported_not_swallowed(monkeypatch):
    def explode(_):
        raise RuntimeError("yfinance unavailable")
    monkeypatch.setattr(ca, "_ticker", explode)
    result = ca.get_corporate_actions("BOOM")
    assert result["success"] is False
    assert "yfinance unavailable" in result["error"]


# ------------------------------------------------------------- live golden set

@network
def test_nvda_ten_for_one_split_is_found():
    """NVDA split 10:1 effective 2024-06-10. An exact, checkable fact."""
    result = ca.get_corporate_actions("NVDA", years=6)
    assert result["success"] is True
    splits = {s["date"][:10]: s["ratio"] for s in result["splits"]}
    assert "2024-06-10" in splits, f"10:1 split missing; found {splits}"
    assert splits["2024-06-10"] == pytest.approx(10.0)


@network
def test_dividend_payer_is_identified():
    result = ca.get_corporate_actions("KO", years=3)
    assert result["success"] is True
    assert result["pays_dividend"] is True
    assert result["ttm_dividend"] > 0
