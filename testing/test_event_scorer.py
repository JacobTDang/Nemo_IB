"""Unit tests for agent.event_scorer.

Verifies each component (novelty / magnitude / reliability / market_awareness /
thesis_relevance) responds correctly to event metadata, and the overall_score
correctly orders high-signal vs low-signal events. Magnitude-gate ensures
low-importance events don't inflate just from novelty + reliability.

Run:
  .venv\\Scripts\\python.exe testing\\test_event_scorer.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.event_scorer import score_event


_results = {'pass': 0, 'fail': 0, 'failures': []}


def _check(name: str, condition: bool, hint: str = '') -> None:
  if condition:
    _results['pass'] += 1
    print(f"  PASS  {name}")
  else:
    _results['fail'] += 1
    _results['failures'].append((name, hint))
    print(f"  FAIL  {name}  --  {hint}")


def _ev(**overrides):
  """Default event dict; override any field."""
  base = {
    'event_id':       'unit_test_evt',
    'source':         'finnhub',
    'category':       'general_news',
    'materiality':    'medium',
    'urgency':        'medium',
    'primary_ticker': 'XYZ',
    'published_at':   '2026-05-22T10:00:00+00:00',
  }
  base.update(overrides)
  return base


def test_output_shape():
  print("\n== output shape ==")
  s = score_event(_ev())
  required = {'novelty', 'magnitude', 'reliability', 'market_awareness',
              'thesis_relevance', 'overall_score'}
  _check("returns all 6 components", set(s.keys()) >= required,
         f"missing: {required - set(s.keys())}")
  for k in required:
    v = s.get(k)
    _check(f"  {k} in [0.0, 1.0]", isinstance(v, (int, float)) and 0.0 <= v <= 1.0,
           f"got {v}")


def test_magnitude_responds_to_materiality():
  print("\n== magnitude responds to materiality ==")
  low = score_event(_ev(materiality='low'))['magnitude']
  med = score_event(_ev(materiality='medium'))['magnitude']
  high = score_event(_ev(materiality='high'))['magnitude']
  _check("low < medium", low < med, f"low={low} medium={med}")
  _check("medium < high", med < high, f"med={med} high={high}")
  _check("high <= 1.0", high <= 1.0, f"got {high}")


def test_8k_bonus():
  print("\n== 8-K category bonus ==")
  generic = score_event(_ev(category='general_news', materiality='high'))['magnitude']
  eight_k = score_event(_ev(category='8-K', materiality='high'))['magnitude']
  _check("8-K outscores generic news at same materiality",
         eight_k > generic, f"8-K={eight_k} generic={generic}")


def test_reliability_by_source():
  print("\n== reliability by source ==")
  sec = score_event(_ev(source='sec_edgar'))['reliability']
  finnhub = score_event(_ev(source='finnhub'))['reliability']
  scraped = score_event(_ev(source='scraped'))['reliability']
  _check("sec_edgar > finnhub > scraped",
         sec > finnhub > scraped, f"sec={sec} finn={finnhub} scraped={scraped}")
  _check("sec_edgar = 1.0", sec == 1.0, f"got {sec}")


def test_overall_orders_correctly():
  print("\n== overall_score orders signals ==")
  # High-impact 8-K event
  high_signal = score_event(_ev(
    source='sec_edgar', category='8-K', materiality='high', urgency='breaking',
  ))['overall_score']
  # Low-impact generic finnhub news
  low_signal = score_event(_ev(
    source='finnhub', category='general_news', materiality='low', urgency='low',
  ))['overall_score']
  _check("high_signal > 0.60", high_signal > 0.60, f"got {high_signal}")
  _check("low_signal < 0.50 (filtered out)", low_signal < 0.50, f"got {low_signal}")
  _check("high_signal > low_signal", high_signal > low_signal,
         f"high={high_signal} low={low_signal}")


def test_magnitude_gate_suppresses_noise():
  print("\n== magnitude gate suppresses noise ==")
  # Event with all other components inflated but low magnitude — should still
  # be filtered. This is the bug the magnitude gate fixed.
  noisy_low_mag = score_event(_ev(
    source='sec_edgar',          # reliability 1.0
    category='general_news',     # no bonus
    materiality='low',           # magnitude 0.30
    urgency='low',               # market_awareness LOW → (1-aware) HIGH
    primary_ticker='UNDISCOVERED',  # likely first event for this ticker → novelty 1.0
  ))['overall_score']
  _check("low-magnitude + high other-components stays < 0.50",
         noisy_low_mag < 0.50, f"got {noisy_low_mag}")


def test_age_affects_market_awareness():
  print("\n== older events have higher market awareness ==")
  fresh = score_event(_ev(
    published_at='2026-05-22T10:00:00+00:00',  # ~today
    urgency='medium',
  ))['market_awareness']
  old = score_event(_ev(
    published_at='2026-05-17T10:00:00+00:00',  # 5 days ago
    urgency='medium',
  ))['market_awareness']
  _check("older event has higher market_awareness",
         old > fresh, f"old={old} fresh={fresh}")


def test_missing_ticker_handled():
  print("\n== missing ticker doesn't crash ==")
  s = score_event(_ev(primary_ticker=None, ticker=None))
  _check("returns valid score for ticker-less event",
         0.0 <= s['overall_score'] <= 1.0, f"got {s}")


def main():
  print("\nEvent scorer unit tests\n")
  test_output_shape()
  test_magnitude_responds_to_materiality()
  test_8k_bonus()
  test_reliability_by_source()
  test_overall_orders_correctly()
  test_magnitude_gate_suppresses_noise()
  test_age_affects_market_awareness()
  test_missing_ticker_handled()

  print(f"\n== Summary ==\n  PASS: {_results['pass']}\n  FAIL: {_results['fail']}")
  for n, h in _results['failures']:
    print(f"  - {n}: {h}")
  return 0 if _results['fail'] == 0 else 1


if __name__ == "__main__":
  sys.exit(main())
