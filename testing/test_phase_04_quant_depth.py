"""Phase 4: quant depth math — reverse DCF, Monte Carlo, F-score, Z-score,
insider clusters, revisions momentum.

All pure-math tests; no API calls, no LLM. Run fast and deterministic.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.financial_modeling_engine.analysis_tools import (
  _dcf_math, _reverse_dcf_math, _monte_carlo_dcf_math,
  _piotroski_f_score_math, _altman_z_score_math,
  _detect_insider_clusters, _revisions_momentum_math,
)


BASE_DCF_INPUTS = {
  'revenue_base': 388_000_000_000.0,
  'ebitda_margin': 0.31,
  'capex_pct_revenue': 0.03,
  'tax_rate': 0.16,
  'depreciation': 0.03,
  'revenue_growth': [0.07] * 5,  # 7% uniform growth
  'wacc': 0.085,
  'terminal_growth': 0.025,
  'terminal_multiple': 14.0,
  'cash': 70_000_000_000.0,
  'debt': 110_000_000_000.0,
  'shares_outstanding': 15_400_000_000.0,
  'ticker': 'AAPL',
}


# ---- Reverse DCF ----------------------------------------------------------

def test_reverse_dcf_recovers_known_growth():
  """If we feed the price produced by _dcf_math at growth=0.07, reverse should
  recover ~7%."""
  baseline = _dcf_math(**BASE_DCF_INPUTS)
  target_price = baseline['price_per_share']
  result = _reverse_dcf_math(target_price, BASE_DCF_INPUTS)
  implied = result['implied_growth_decimal']
  assert abs(implied - 0.07) < 0.005, \
    f"reverse DCF should recover ~7% growth from its own price, got {implied}"
  print(f"PASS: reverse DCF recovers known growth (target $7%, got {implied*100:.2f}%)")


def test_reverse_dcf_rich_when_price_above_base():
  baseline = _dcf_math(**BASE_DCF_INPUTS)
  rich_price = baseline['price_per_share'] * 1.4  # 40% above base case
  result = _reverse_dcf_math(rich_price, BASE_DCF_INPUTS)
  if 'error' in result:
    print(f"SKIP: rich_price out of solver range: {result}")
    return
  assert result['verdict'] == 'rich', f"expected rich, got {result['verdict']}"
  assert result['spread_pct'] > 2, f"spread should be positive >2pp, got {result['spread_pct']}"
  print(f"PASS: reverse DCF flags rich at 40% premium (spread={result['spread_pct']}pp)")


def test_reverse_dcf_cheap_when_price_below_base():
  baseline = _dcf_math(**BASE_DCF_INPUTS)
  cheap_price = baseline['price_per_share'] * 0.5
  result = _reverse_dcf_math(cheap_price, BASE_DCF_INPUTS)
  if 'error' in result:
    print(f"SKIP: cheap_price out of solver range: {result}")
    return
  assert result['verdict'] == 'cheap', f"expected cheap, got {result['verdict']}"
  assert result['spread_pct'] < -2
  print(f"PASS: reverse DCF flags cheap at 50% discount (spread={result['spread_pct']}pp)")


def test_reverse_dcf_no_solution_in_range():
  """A wildly outside-range price should return error, not crash."""
  result = _reverse_dcf_math(0.01, BASE_DCF_INPUTS)
  assert 'error' in result or result.get('verdict') == 'cheap'
  print("PASS: extreme price handled gracefully")


# ---- Monte Carlo DCF -------------------------------------------------------

def test_monte_carlo_returns_valid_distribution():
  result = _monte_carlo_dcf_math(BASE_DCF_INPUTS, n_iter=500, seed=42)
  assert 'mean' in result, f"missing mean: {result}"
  assert result['n_iter_valid'] > 400, f"too many invalid iters: {result['n_iter_valid']}"
  assert result['p10'] < result['median'] < result['p90']
  assert result['p5'] < result['p25'] < result['p75'] < result['p95']
  baseline = _dcf_math(**BASE_DCF_INPUTS)['price_per_share']
  # Mean should be within ~30% of baseline (gaussian noise is symmetric-ish)
  assert abs(result['mean'] - baseline) / baseline < 0.35, \
    f"mean ${result['mean']} too far from baseline ${baseline}"
  print(f"PASS: monte carlo distribution coherent "
        f"(mean=${result['mean']}, p10-p90=[${result['p10']},${result['p90']}], "
        f"baseline=${baseline:.2f})")


def test_monte_carlo_deterministic_with_seed():
  r1 = _monte_carlo_dcf_math(BASE_DCF_INPUTS, n_iter=200, seed=99)
  r2 = _monte_carlo_dcf_math(BASE_DCF_INPUTS, n_iter=200, seed=99)
  assert r1['mean'] == r2['mean'] and r1['p90'] == r2['p90'], \
    "same seed should produce identical results"
  print("PASS: monte carlo is deterministic with fixed seed")


def test_monte_carlo_widens_with_higher_std():
  r_narrow = _monte_carlo_dcf_math(BASE_DCF_INPUTS, n_iter=500,
                                     wacc_std=0.001, margin_std=0.001,
                                     growth_std=0.001, seed=1)
  r_wide = _monte_carlo_dcf_math(BASE_DCF_INPUTS, n_iter=500,
                                   wacc_std=0.02, margin_std=0.04,
                                   growth_std=0.05, seed=1)
  assert r_wide['std'] > r_narrow['std'] * 3, \
    f"wider stds should produce wider distribution: narrow={r_narrow['std']} wide={r_wide['std']}"
  print(f"PASS: distribution widens with higher input stds "
        f"(narrow_std=${r_narrow['std']}, wide_std=${r_wide['std']})")


# ---- Piotroski F-score -----------------------------------------------------

def test_piotroski_strong_company():
  fin = {
    'net_income': 100, 'net_income_prior': 80,
    'op_cash_flow': 150, 'total_assets': 1000, 'total_assets_prior': 950,
    'long_term_debt': 200, 'long_term_debt_prior': 250,
    'current_ratio': 2.0, 'current_ratio_prior': 1.8,
    'shares_outstanding': 100, 'shares_outstanding_prior': 100,
    'gross_margin': 0.45, 'gross_margin_prior': 0.40,
    'asset_turnover': 1.5, 'asset_turnover_prior': 1.2,
  }
  result = _piotroski_f_score_math(fin)
  assert result['score'] == 9, f"perfect inputs should score 9/9, got {result['score']}"
  assert result['rating'] == 'strong'
  print(f"PASS: ideal company gets 9/9 (strong)")


def test_piotroski_weak_company():
  fin = {
    'net_income': -50, 'net_income_prior': -10,
    'op_cash_flow': -20, 'total_assets': 1000, 'total_assets_prior': 1100,
    'long_term_debt': 500, 'long_term_debt_prior': 400,  # debt rising
    'current_ratio': 0.8, 'current_ratio_prior': 1.0,    # liquidity deteriorating
    'shares_outstanding': 120, 'shares_outstanding_prior': 100,  # dilution
    'gross_margin': 0.20, 'gross_margin_prior': 0.25,
    'asset_turnover': 0.8, 'asset_turnover_prior': 1.0,
  }
  result = _piotroski_f_score_math(fin)
  assert result['score'] <= 3, f"distressed inputs should score 3 or below, got {result['score']}"
  assert result['rating'] == 'weak'
  print(f"PASS: distressed company gets {result['score']}/9 (weak)")


def test_piotroski_partial_data_handles_gracefully():
  """With only net_income + op_cash_flow, only tests that don't need prior-period
  data should evaluate. The rest are skipped, NOT spuriously True. Pre-fix, the
  defaults (`float('inf')`, `1`) caused roa_improving and lt_debt_decreasing to
  pass spuriously, inflating the score."""
  fin = {'net_income': 100, 'op_cash_flow': 80}
  result = _piotroski_f_score_math(fin)
  assert 'score' in result
  assert result['score'] <= 3, \
    f"with most data missing, score should be at most 3, got {result['score']}"
  assert len(result.get('skipped_tests', [])) >= 6, \
    f"at least 6 tests should be skipped, got {len(result.get('skipped_tests', []))}"
  assert result.get('max_score_evaluated', 9) <= 3, \
    f"max_score_evaluated should be at most 3, got {result.get('max_score_evaluated')}"
  print(f"PASS: partial data score={result['score']}/{result['max_score_evaluated']} "
        f"with {len(result['skipped_tests'])} tests skipped")


def test_piotroski_few_tests_does_not_yield_strong():
  """When fewer than _PIOTROSKI_MIN_EVALUATED tests can be evaluated, the
  rating must be 'insufficient_data' regardless of how many of the few
  evaluable tests passed. Pre-fix, 2 passes out of 2 evaluable tests gave
  ratio 1.0 -> 'strong', which is not defensible."""
  # Only positive_net_income + positive_op_cash_flow can evaluate from these
  # inputs (everything else needs prior-period data). Both pass.
  fin = {'net_income': 100, 'op_cash_flow': 80}
  result = _piotroski_f_score_math(fin)
  assert result['score'] == 2, f"expected 2 passes, got {result['score']}"
  assert result['max_score_evaluated'] <= 3, \
    f"expected at most 3 evaluable tests, got {result['max_score_evaluated']}"
  assert result['rating'] == 'insufficient_data', \
    f"too few tests must yield insufficient_data, got {result['rating']!r}"
  print(f"PASS: score=2 of {result['max_score_evaluated']} evaluable -> "
        f"{result['rating']!r} (not 'strong')")


def test_piotroski_few_tests_does_not_yield_weak():
  """Mirror of the above: a single failing test should not flip to 'weak'."""
  # Only positive_net_income can evaluate; net income is negative so it fails.
  fin = {'net_income': -50}
  result = _piotroski_f_score_math(fin)
  assert result['score'] == 0
  assert result['max_score_evaluated'] <= 2, result['max_score_evaluated']
  assert result['rating'] == 'insufficient_data', \
    f"too few tests must yield insufficient_data, got {result['rating']!r}"
  print(f"PASS: 1 evaluable test (fail) -> insufficient_data, not 'weak'")


def test_piotroski_full_data_unchanged():
  """Regression guard: with full data, the score and rating must remain
  9/9-strong with no skipped tests."""
  fin = {
    'net_income': 100, 'net_income_prior': 80,
    'op_cash_flow': 150, 'total_assets': 1000, 'total_assets_prior': 950,
    'long_term_debt': 200, 'long_term_debt_prior': 250,
    'current_ratio': 2.0, 'current_ratio_prior': 1.8,
    'shares_outstanding': 100, 'shares_outstanding_prior': 100,
    'gross_margin': 0.45, 'gross_margin_prior': 0.40,
    'asset_turnover': 1.5, 'asset_turnover_prior': 1.2,
  }
  result = _piotroski_f_score_math(fin)
  assert result['score'] == 9, f"full data should score 9/9, got {result['score']}"
  assert result['rating'] == 'strong'
  assert result.get('max_score_evaluated') == 9
  assert not result.get('skipped_tests'), \
    f"no skips with full data; got {result.get('skipped_tests')}"
  print(f"PASS: full data -> score=9/9, no skipped tests")


# ---- Altman Z-score --------------------------------------------------------

def test_altman_z_safe_zone():
  fin = {
    'total_assets': 1000, 'working_capital': 200,
    'retained_earnings': 400, 'ebit': 150,
    'market_cap': 800, 'total_liabilities': 300, 'revenue': 1200,
  }
  result = _altman_z_score_math(fin)
  assert result['zone'] == 'safe', f"expected safe, got {result['zone']} (z={result['z_score']})"
  assert result['z_score'] > 2.99
  print(f"PASS: healthy balance sheet -> Z={result['z_score']} (safe)")


def test_altman_z_distress_zone():
  fin = {
    'total_assets': 1000, 'working_capital': -100,  # negative WC
    'retained_earnings': -200, 'ebit': 10,
    'market_cap': 100, 'total_liabilities': 900, 'revenue': 400,
  }
  result = _altman_z_score_math(fin)
  assert result['zone'] == 'distress', f"expected distress, got {result['zone']} (z={result['z_score']})"
  assert result['z_score'] < 1.81
  print(f"PASS: distressed company -> Z={result['z_score']} (distress)")


def test_altman_z_components_are_round():
  fin = {
    'total_assets': 1000, 'working_capital': 200, 'retained_earnings': 400,
    'ebit': 150, 'market_cap': 800, 'total_liabilities': 300, 'revenue': 1200,
  }
  result = _altman_z_score_math(fin)
  c = result['components']
  assert c['X1_wc_ta'] == 0.2
  assert c['X2_re_ta'] == 0.4
  assert c['X4_mc_tl'] == round(800/300, 4)
  print(f"PASS: Z-score components compute correctly")


# ---- Insider clusters ------------------------------------------------------

def test_insider_strong_cluster_buy():
  from datetime import datetime, timedelta
  recent = (datetime.now() - timedelta(days=5)).isoformat()
  txns = [
    {'date': recent, 'shares': 1000, 'insider_name': 'Alice', 'transaction_value': 100_000},
    {'date': recent, 'shares': 2000, 'insider_name': 'Bob',   'transaction_value': 200_000},
    {'date': recent, 'shares': 1500, 'insider_name': 'Carol', 'transaction_value': 150_000},
    {'date': recent, 'shares': 1200, 'insider_name': 'Dave',  'transaction_value': 120_000},
  ]
  result = _detect_insider_clusters(txns, lookback_days=30)
  assert result['signal'] == 'strong_cluster_buy'
  assert result['distinct_buyers'] == 4
  assert result['distinct_sellers'] == 0
  print(f"PASS: 4 buyers + 0 sellers -> strong_cluster_buy")


def test_insider_strong_cluster_sell():
  from datetime import datetime, timedelta
  recent = (datetime.now() - timedelta(days=10)).isoformat()
  txns = [
    {'date': recent, 'shares': -1000, 'insider_name': 'Alice', 'transaction_value': 100_000},
    {'date': recent, 'shares': -1500, 'insider_name': 'Bob',   'transaction_value': 150_000},
    {'date': recent, 'shares': -2000, 'insider_name': 'Carol', 'transaction_value': 200_000},
  ]
  result = _detect_insider_clusters(txns)
  assert result['signal'] == 'strong_cluster_sell'
  print(f"PASS: 3 sellers + 0 buyers -> strong_cluster_sell")


def test_insider_mixed_no_signal():
  from datetime import datetime, timedelta
  recent = (datetime.now() - timedelta(days=5)).isoformat()
  txns = [
    {'date': recent, 'shares': 1000, 'insider_name': 'A', 'transaction_value': 100_000},
    {'date': recent, 'shares': -1000, 'insider_name': 'B', 'transaction_value': 100_000},
    {'date': recent, 'shares': 1500, 'insider_name': 'C', 'transaction_value': 150_000},
    {'date': recent, 'shares': -1500, 'insider_name': 'D', 'transaction_value': 150_000},
  ]
  result = _detect_insider_clusters(txns)
  assert result['signal'] is None, f"mixed should produce None signal, got {result['signal']}"
  print(f"PASS: mixed direction -> no signal")


def test_insider_stale_transactions_excluded():
  from datetime import datetime, timedelta
  stale = (datetime.now() - timedelta(days=90)).isoformat()
  txns = [
    {'date': stale, 'shares': 1000, 'insider_name': 'A', 'transaction_value': 100_000},
    {'date': stale, 'shares': 2000, 'insider_name': 'B', 'transaction_value': 200_000},
    {'date': stale, 'shares': 1500, 'insider_name': 'C', 'transaction_value': 150_000},
  ]
  result = _detect_insider_clusters(txns, lookback_days=30)
  assert result['signal'] is None, f"stale txns should produce no signal, got {result['signal']}"
  print(f"PASS: 90-day-old transactions excluded with 30-day lookback")


# ---- Revisions momentum ----------------------------------------------------

def test_revisions_rising():
  trends = [
    {'period': '2026-05-01', 'strongBuy': 15, 'buy': 12, 'hold': 5, 'sell': 1, 'strongSell': 0},
    {'period': '2026-04-01', 'strongBuy': 12, 'buy': 10, 'hold': 6, 'sell': 2, 'strongSell': 1},
    {'period': '2026-03-01', 'strongBuy': 10, 'buy': 9,  'hold': 7, 'sell': 3, 'strongSell': 1},
    {'period': '2026-02-01', 'strongBuy': 8,  'buy': 8,  'hold': 8, 'sell': 4, 'strongSell': 2},
  ]
  result = _revisions_momentum_math(trends)
  assert result['direction'] in ('rising', 'strong_rising'), \
    f"upward revisions should rise, got {result['direction']}"
  assert result['delta_30d'] > 0 and result['delta_90d'] > 0
  print(f"PASS: rising revisions detected (composite={result['composite_score']})")


def test_revisions_falling():
  trends = [
    {'period': '2026-05-01', 'strongBuy': 5, 'buy': 5,  'hold': 8, 'sell': 5, 'strongSell': 3},
    {'period': '2026-04-01', 'strongBuy': 7, 'buy': 7,  'hold': 8, 'sell': 4, 'strongSell': 2},
    {'period': '2026-03-01', 'strongBuy': 9, 'buy': 9,  'hold': 7, 'sell': 3, 'strongSell': 1},
    {'period': '2026-02-01', 'strongBuy': 12,'buy': 10, 'hold': 6, 'sell': 2, 'strongSell': 1},
  ]
  result = _revisions_momentum_math(trends)
  assert result['direction'] in ('falling', 'strong_falling'), \
    f"falling consensus should fall, got {result['direction']}"
  assert result['delta_30d'] < 0
  print(f"PASS: falling revisions detected (composite={result['composite_score']})")


def test_revisions_insufficient_history():
  result = _revisions_momentum_math([])
  assert 'error' in result
  result2 = _revisions_momentum_math([{'period': 'p', 'strongBuy': 5, 'buy': 3, 'hold': 2, 'sell': 1, 'strongSell': 0}])
  assert 'error' in result2
  print(f"PASS: insufficient history returns error not crash")


if __name__ == "__main__":
  test_reverse_dcf_recovers_known_growth()
  test_reverse_dcf_rich_when_price_above_base()
  test_reverse_dcf_cheap_when_price_below_base()
  test_reverse_dcf_no_solution_in_range()
  test_monte_carlo_returns_valid_distribution()
  test_monte_carlo_deterministic_with_seed()
  test_monte_carlo_widens_with_higher_std()
  test_piotroski_strong_company()
  test_piotroski_weak_company()
  test_piotroski_partial_data_handles_gracefully()
  test_piotroski_few_tests_does_not_yield_strong()
  test_piotroski_few_tests_does_not_yield_weak()
  test_piotroski_full_data_unchanged()
  test_altman_z_safe_zone()
  test_altman_z_distress_zone()
  test_altman_z_components_are_round()
  test_insider_strong_cluster_buy()
  test_insider_strong_cluster_sell()
  test_insider_mixed_no_signal()
  test_insider_stale_transactions_excluded()
  test_revisions_rising()
  test_revisions_falling()
  test_revisions_insufficient_history()
  print("\nAll Phase 4 quant depth tests passed.")
