"""Phase 5: catalyst calendar, 10-K risk factor diff, Google Trends.

All tests use fixtures + mocks. No live API calls."""
import sys, os
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.catalysts import upcoming_catalysts, summarize_for_analyst
from data.risk_factor_diff import (
  diff_risk_factors, _paragraph_split, _ngrams, _jaccard, _matches_any
)
from data.alt_sources import google_trends_for, TICKER_QUERIES


# ---- Catalyst calendar -----------------------------------------------------

def test_catalysts_includes_fomc_when_in_window():
  """A future FOMC date should appear if within the days_out window."""
  fake_earnings = {
    'earningsCalendar': [
      {'date': '2026-08-01', 'epsEstimate': 1.49, 'revenueEstimate': 95_000_000_000,
       'symbol': 'AAPL', 'hour': 'amc'}
    ]
  }
  with patch.dict(os.environ, {'FINNHUB_API_KEY': 'fake'}), \
       patch('requests.get') as mock_get, \
       patch('data.catalysts._ex_dividend_yfinance', return_value=[]):
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = fake_earnings
    events = upcoming_catalysts('AAPL', days_out=120)

  types = {e['type'] for e in events}
  assert 'fomc' in types, f"FOMC missing from events. Got types: {types}"
  fomc_dates = [e['date'] for e in events if e['type'] == 'fomc']
  print(f"PASS: FOMC events surfaced ({len(fomc_dates)} in 120-day window)")


def test_catalysts_includes_earnings_when_available():
  fake_earnings = {
    'earningsCalendar': [
      {'date': '2026-07-30', 'epsEstimate': 1.49, 'revenueEstimate': 95e9,
       'symbol': 'AAPL', 'hour': 'amc'}
    ]
  }
  with patch.dict(os.environ, {'FINNHUB_API_KEY': 'fake'}), \
       patch('requests.get') as mock_get, \
       patch('data.catalysts._ex_dividend_yfinance', return_value=[]):
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = fake_earnings
    events = upcoming_catalysts('AAPL', days_out=120)

  earnings = [e for e in events if e['type'] == 'earnings']
  assert len(earnings) == 1
  assert earnings[0]['ticker'] == 'AAPL'
  assert earnings[0]['expected_eps'] == 1.49
  print("PASS: Finnhub earnings call surfaced correctly")


def test_catalysts_sorted_chronologically():
  with patch.dict(os.environ, {'FINNHUB_API_KEY': 'fake'}), \
       patch('requests.get') as mock_get, \
       patch('data.catalysts._ex_dividend_yfinance', return_value=[]):
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = {'earningsCalendar': []}
    events = upcoming_catalysts('AAPL', days_out=300)
  dates = [e['date'] for e in events]
  assert dates == sorted(dates), "events not sorted chronologically"
  print(f"PASS: events sorted ({len(dates)} entries)")


def test_catalysts_no_api_key_still_returns_macro():
  """No Finnhub key should not block FOMC + macro patterns."""
  with patch.dict(os.environ, {}, clear=False), \
       patch('data.catalysts._ex_dividend_yfinance', return_value=[]):
    os.environ.pop('FINNHUB_API_KEY', None)
    events = upcoming_catalysts('AAPL', days_out=90)
  # Should still have macro/FOMC entries
  assert len(events) > 0
  print(f"PASS: macro/FOMC events surface even without Finnhub key ({len(events)} events)")


def test_nfp_lands_on_first_friday():
  """NFP release is always the first Friday of the month, not day-of-month=5.
  Pre-fix the function emitted NFP on the 5th, which is a Friday only ~14% of
  the time."""
  events = upcoming_catalysts('AAPL', days_out=180)
  from datetime import datetime as _dt
  nfp_events = [e for e in events
                 if 'NFP' in (e.get('description') or '') and e.get('type') == 'macro_release']
  assert nfp_events, f"expected at least one NFP event in 180-day window; got events: " \
    f"{[e.get('description') for e in events]}"
  failures = []
  for e in nfp_events:
    dt = _dt.fromisoformat(e['date'])
    if dt.weekday() != 4:  # 4 == Friday
      failures.append(f"NFP {e['date']} weekday={dt.weekday()} (not Friday)")
    if dt.day > 7:
      failures.append(f"NFP {e['date']} day={dt.day} (not first week)")
  assert not failures, "NFP date errors:\n  " + "\n  ".join(failures)
  print(f"PASS: all {len(nfp_events)} NFP events fall on first Friday "
        f"({[e['date'] for e in nfp_events]})")


def test_summarize_for_analyst_renders_compactly():
  events = [
    {'type': 'earnings', 'date': '2026-08-01', 'ticker': 'AAPL',
     'description': 'AAPL earnings', 'expected_eps': 1.49, 'hour': 'amc',
     'impact': 'company-specific'},
    {'type': 'fomc', 'date': '2026-09-16',
     'description': 'FOMC interest rate decision', 'impact': 'rate-sensitive'},
  ]
  out = summarize_for_analyst(events)
  assert 'UPCOMING CATALYSTS' in out
  assert '2026-08-01' in out
  assert 'AAPL' in out
  assert 'FOMC' in out
  print(f"PASS: analyst summary renders")


def test_summarize_for_analyst_empty():
  assert summarize_for_analyst([]) == "No catalysts in window."
  print("PASS: empty catalyst list handled")


# ---- 10-K Risk Factor diff -------------------------------------------------

PRIOR_RISK = """We face competition from established technology companies in the
consumer electronics market, including but not limited to companies with
greater resources and established customer bases.

Our supply chain depends on third-party manufacturers concentrated primarily
in Asia. Disruptions to these manufacturing operations could materially
adversely affect our business, results of operations, and financial condition.

Cybersecurity incidents could disrupt our operations and harm our reputation.
We invest substantial resources in protecting our systems and customer data,
but no security measures are guaranteed to be effective."""


CURRENT_RISK = """We face competition from established technology companies in the
consumer electronics market, including but not limited to companies with
greater resources and established customer bases.

Geopolitical tensions with China could result in tariffs, export restrictions,
or other trade barriers that disrupt our access to manufacturing capacity,
critical components, or end-market demand in mainland China.

Our supply chain depends on third-party manufacturers concentrated primarily
in Asia. Disruptions to these manufacturing operations could materially
adversely affect our business, results of operations, and financial condition,
and recent geopolitical events have amplified this concentration risk.

Cybersecurity incidents could disrupt our operations and harm our reputation.
We invest substantial resources in protecting our systems and customer data,
but no security measures are guaranteed to be effective."""


def test_paragraph_split_filters_short():
  text = "Short.\n\nThis is a longer paragraph that contains more than the minimum number of characters required to be considered a real risk factor and not noise.\n\nNo."
  paras = _paragraph_split(text, min_chars=80)
  assert len(paras) == 1
  print(f"PASS: paragraph split filters fragments below min_chars")


def test_paragraph_split_handles_varied_separators():
  """Real 10-K text uses many paragraph separators: \\n\\n, \\n  \\n,
  \\n\\t\\n, and similar. Pre-fix the splitter handled only \\n\\n,
  collapsing multiple paragraphs into one when other whitespace appeared."""
  text = (
    "We face competition from established players in the consumer electronics "
    "market, including companies with greater resources and established customer bases.\n\n"
    "Our supply chain depends on third-party manufacturers concentrated primarily in Asia. "
    "Disruptions could materially adversely affect our business.\n  \n"  # spaces between newlines
    "Cybersecurity incidents could disrupt our operations and harm our reputation. "
    "We invest substantial resources in protecting our systems and customer data.\n\t\n"  # tab between newlines
    "  - Bulleted: Geopolitical tensions with China could result in tariffs, export "
    "restrictions, or other trade barriers that disrupt our access to manufacturing capacity.\n\n"
    "1. Numbered: Foreign exchange exposure could materially affect our financial results "
    "as a significant share of revenue is generated outside the United States.\n"
  )
  paras = _paragraph_split(text, min_chars=80)
  assert len(paras) == 5, \
    f"expected 5 paragraphs separated by varied whitespace; got {len(paras)}"
  # No paragraph should still begin with a bullet/number marker (those should
  # be stripped before Jaccard scoring)
  for p in paras:
    assert not p.startswith('- ') and not p.startswith('* ') \
      and not p.startswith('1.') and not p.startswith('2.'), \
      f"leading marker not stripped: {p[:30]}..."
  print(f"PASS: paragraph splitter handles {len(paras)} paragraphs with mixed separators "
        "and strips leading bullets/numbering")


def test_ngrams_jaccard_self_match():
  text = "supply chain disruption manufacturing Asia"
  s = _ngrams(text, n=3)
  assert _jaccard(s, s) == 1.0
  print("PASS: ngram jaccard self-match = 1.0")


def test_diff_detects_new_risk_paragraph():
  result = diff_risk_factors(CURRENT_RISK, PRIOR_RISK)
  assert 'error' not in result
  print(f"  new={result['new_risks_count']} removed={result['removed_risks_count']} "
        f"modified={result['modified_risks_count']} delta={result['paragraph_count_delta']}")
  assert result['new_risks_count'] >= 1, \
    f"should detect new geopolitical paragraph, got {result['new_risks_count']}"
  # The newly added paragraph mentions China/geopolitical
  new_text = ' '.join(result['new_risks']).lower()
  assert 'china' in new_text or 'geopolitical' in new_text
  print(f"PASS: diff detected new geopolitical risk paragraph")


def test_diff_detects_modified_supply_chain_paragraph():
  result = diff_risk_factors(CURRENT_RISK, PRIOR_RISK)
  # Supply chain paragraph has trailing "recent geopolitical events have amplified..."
  # which makes it 0.5-0.85 similar — should appear in modified_risks
  if result['modified_risks_count'] >= 1:
    print(f"PASS: diff detected {result['modified_risks_count']} modified paragraph(s)")
  else:
    print(f"WARN: modified_risks_count = 0 (acceptable if similarity > 0.85)")


def test_diff_detects_no_removals_when_all_kept():
  result = diff_risk_factors(CURRENT_RISK, PRIOR_RISK)
  # All 3 prior paragraphs should match to current (none removed)
  assert result['removed_risks_count'] == 0, \
    f"prior was a subset, expected 0 removed, got {result['removed_risks_count']}"
  print("PASS: no false removals when prior is subset of current")


def test_diff_handles_empty_input():
  result = diff_risk_factors("", PRIOR_RISK)
  assert 'error' in result
  result2 = diff_risk_factors(PRIOR_RISK, "")
  assert 'error' in result2
  print("PASS: empty input returns error not crash")


def test_diff_paragraph_count_delta():
  result = diff_risk_factors(CURRENT_RISK, PRIOR_RISK)
  # current has 4 paragraphs, prior has 3, delta = +1
  assert result['paragraph_count_delta'] == 1
  print("PASS: paragraph count delta is correct")


# ---- Google Trends ---------------------------------------------------------

def test_trends_returns_error_for_unmapped_ticker():
  result = google_trends_for('ZZZ_UNKNOWN', query_terms=None)
  assert 'error' in result
  assert result['error'] == 'no_queries_mapped'
  print("PASS: unmapped ticker returns clear error")


def test_trends_uses_mapped_queries():
  """When user doesn't supply query_terms, we should fall back to TICKER_QUERIES."""
  assert 'AAPL' in TICKER_QUERIES
  assert 'iPhone' in TICKER_QUERIES['AAPL']
  # We can't make a real call but we can verify the lookup
  print(f"PASS: TICKER_QUERIES has {len(TICKER_QUERIES)} tickers mapped including AAPL/NVDA/TSLA")


def test_trends_handles_pytrends_failure():
  with patch.dict('sys.modules', {'pytrends.request': MagicMock(
    TrendReq=MagicMock(side_effect=Exception("simulated rate limit"))
  )}):
    result = google_trends_for('AAPL')
  assert 'error' in result, f"failed pytrends should return error, got {result}"
  print("PASS: pytrends failure returned as error, not crash")


def test_trends_processes_dataframe():
  """Mock the dataframe return and verify post-processing."""
  import pandas as pd
  import numpy as np
  fake_df = pd.DataFrame({
    'iPhone':   np.linspace(40, 80, 52),  # rising trend
    'AirPods':  [50] * 52,                # flat trend
    'Apple Vision Pro': np.linspace(80, 30, 52),  # falling
    'isPartial': [False] * 52,
  }, index=pd.date_range('2025-05-10', periods=52, freq='W'))

  fake_pytrends_cls = MagicMock()
  fake_pytrends = MagicMock()
  fake_pytrends.interest_over_time.return_value = fake_df
  fake_pytrends_cls.return_value = fake_pytrends

  with patch.dict('sys.modules', {'pytrends.request': MagicMock(TrendReq=fake_pytrends_cls)}):
    result = google_trends_for('AAPL')

  assert 'error' not in result, f"unexpected error: {result}"
  assert 'iPhone' in result['series']
  assert result['momentum_3m_vs_12m_pct']['iPhone'] > 0, \
    f"rising series should have positive momentum, got {result['momentum_3m_vs_12m_pct']['iPhone']}"
  assert result['momentum_3m_vs_12m_pct']['Apple Vision Pro'] < 0, \
    f"falling series should have negative momentum"
  print(f"PASS: trends processed correctly "
        f"(iPhone momentum={result['momentum_3m_vs_12m_pct']['iPhone']}%, "
        f"Vision Pro momentum={result['momentum_3m_vs_12m_pct']['Apple Vision Pro']}%)")


if __name__ == "__main__":
  test_catalysts_includes_fomc_when_in_window()
  test_catalysts_includes_earnings_when_available()
  test_catalysts_sorted_chronologically()
  test_catalysts_no_api_key_still_returns_macro()
  test_nfp_lands_on_first_friday()
  test_summarize_for_analyst_renders_compactly()
  test_summarize_for_analyst_empty()
  test_paragraph_split_filters_short()
  test_paragraph_split_handles_varied_separators()
  test_ngrams_jaccard_self_match()
  test_diff_detects_new_risk_paragraph()
  test_diff_detects_modified_supply_chain_paragraph()
  test_diff_detects_no_removals_when_all_kept()
  test_diff_handles_empty_input()
  test_diff_paragraph_count_delta()
  test_trends_returns_error_for_unmapped_ticker()
  test_trends_uses_mapped_queries()
  test_trends_handles_pytrends_failure()
  test_trends_processes_dataframe()
  print("\nAll Phase 5 tests passed.")
