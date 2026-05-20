"""Round 4 item 3: analyst recommendations source attribution.

Investigation showed the tool ALREADY returns latest+prior snapshots
correctly (not cumulative as the plan assumed). The 39/54 strong-buy-or-buy
claim in the AAPL report is accurate per Finnhub's data. The external
discrepancy with Yahoo/TipRanks ('~28 analysts') is Finnhub's methodology
including all firms on file rather than de-duplicated distinct analysts.

Fix: add a `source` and `methodology_note` to the response so downstream
Bull/Bear agents have proper context and don't conflict with other
analyst-count sources.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.news_agregator.finnhub_server import _condense_recommendations


def test_response_includes_source_and_methodology():
  raw = [
    {"strongBuy": 15, "buy": 24, "hold": 13, "sell": 2, "strongSell": 0,
     "period": "2026-05-01"},
    {"strongBuy": 14, "buy": 23, "hold": 15, "sell": 2, "strongSell": 0,
     "period": "2026-04-01"},
  ]
  out = _condense_recommendations(raw)
  assert 'source' in out, \
    f"missing source attribution; agents will conflict with Yahoo numbers: {sorted(out.keys())}"
  assert 'finnhub' in out['source'].lower()
  assert 'methodology_note' in out, \
    f"missing methodology_note; agents won't know counts may differ from other sources"
  # Note must mention that counts can exceed distinct analyst counts
  assert 'distinct' in out['methodology_note'].lower() or \
    'differ' in out['methodology_note'].lower() or \
    'aggreg' in out['methodology_note'].lower(), \
    f"methodology_note should warn about double-counting; got {out['methodology_note']!r}"
  print(f"PASS: source='{out['source']}', methodology_note set")


def test_aapl_fixture_matches_report_claim():
  """The 'fact-check found 39/54 — but report's 39/54 was actually right'
  realization: lock in the invariant that for AAPL-shaped data, the tool's
  latest snapshot sums to total_analysts and matches buy_count = strong_buy + buy."""
  raw = [
    {"strongBuy": 15, "buy": 24, "hold": 13, "sell": 2, "strongSell": 0,
     "period": "2026-05-01"},
  ]
  out = _condense_recommendations(raw)
  latest = out['latest']
  assert latest is not None
  total = latest['strong_buy'] + latest['buy'] + latest['hold'] + \
          latest['sell'] + latest['strong_sell']
  assert out['total_analysts'] == total, \
    f"total_analysts ({out['total_analysts']}) must equal sum of latest buckets ({total})"
  assert latest['period'] == '2026-05-01'
  # Specifically: 15 strong_buy + 24 buy = 39
  assert latest['strong_buy'] + latest['buy'] == 39
  print(f"PASS: latest snapshot self-consistent (39 buy+/{total} total)")


def test_existing_latest_prior_structure_unchanged():
  """Regression guard: the previously-correct latest/prior/consensus/trend
  fields must still compute."""
  raw = [
    {"strongBuy": 15, "buy": 24, "hold": 13, "sell": 2, "strongSell": 0,
     "period": "2026-05-01"},
    {"strongBuy": 14, "buy": 23, "hold": 15, "sell": 2, "strongSell": 0,
     "period": "2026-04-01"},
  ]
  out = _condense_recommendations(raw)
  assert out['latest']['period'] == '2026-05-01'
  assert out['prior']['period'] == '2026-04-01'
  assert out['consensus'] in ('buy', 'strong_buy', 'hold', 'sell', 'strong_sell')
  # Trend: upgrading (latest has 39 buy+ vs prior's 37)
  assert out['trend'] in ('upgrading', 'downgrading', 'stable', 'unknown')
  print(f"PASS: latest/prior/consensus/trend unchanged (consensus={out['consensus']}, trend={out['trend']})")


def test_empty_input_handles_gracefully():
  out = _condense_recommendations([])
  assert out['latest'] is None
  assert out['prior'] is None
  assert out['total_analysts'] == 0
  # source/methodology must still be present even on empty input
  assert 'source' in out and 'methodology_note' in out
  print("PASS: empty input -> structured response with source/methodology preserved")


if __name__ == "__main__":
  test_response_includes_source_and_methodology()
  test_aapl_fixture_matches_report_claim()
  test_existing_latest_prior_structure_unchanged()
  test_empty_input_handles_gracefully()
  print("\nAll Round 4 / item 3 analyst fidelity tests passed.")
