"""Stress tests for the falsifier evaluator.

Tests organized by category:
  1. Numeric condition parsing — does it extract op/value/unit correctly?
  2. Numeric evaluation — does it correctly fire on thresholds?
  3. Text matching — token overlap, proper-noun boost, polarity
  4. Adversarial — partial matches, negation traps, similar-but-different
  5. End-to-end — synthetic falsifiers against synthetic evidence pools
  6. Performance — N theses x M falsifiers x E evidence

Run via:
  ./.venv/Scripts/python.exe testing/test_falsifier_evaluator.py
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.falsifier_evaluator import (
    parse_numeric_conditions,
    evaluate_numeric_condition,
    has_negation,
    score_evidence_against_falsifier,
    evaluate_falsifier,
    NumericCondition,
)


# ===========================================================================
# Test infrastructure — bare-bones assertions with category counters
# ===========================================================================

_results = {'pass': 0, 'fail': 0, 'failures': []}


def _check(name: str, condition, hint: str = ''):
  if condition:
    _results['pass'] += 1
    print(f"  PASS  {name}")
  else:
    _results['fail'] += 1
    _results['failures'].append((name, hint))
    print(f"  FAIL  {name}  --  {hint}")


def _section(title: str):
  print(f"\n=== {title} ===")


# ===========================================================================
# 1. Numeric condition parsing
# ===========================================================================

def test_numeric_parsing():
  _section("1. Numeric condition parsing")

  cases = [
    # (falsifier, expected_count, expected_first_op, expected_first_value, expected_unit)
    ("10Y treasury > 5.25%",          1, 'gt', 5.25, '%'),
    ("HY OAS > 400bps",               1, 'gt', 400,  'bps'),
    ("Azure revenue growth < 18% YoY", 1, 'lt', 18,  '%'),
    ("net debt/EBITDA above 3.5x",    1, 'gt', 3.5,  'x'),
    ("EPS below $4.10",               1, 'lt', 4.10, ''),
    ("Days to cover >= 5",            1, 'gte', 5,   ''),
    # No conditions
    ("Hood replaced as CFO",          0, None, None, None),
    # Multiple conditions
    ("10Y > 5.25% AND HY OAS > 400bps", 2, 'gt', 5.25, '%'),
  ]

  for falsifier, exp_count, exp_op, exp_val, exp_unit in cases:
    conds = parse_numeric_conditions(falsifier)
    _check(
      f"  count of conditions in {falsifier!r}",
      len(conds) == exp_count,
      f"got {len(conds)} expected {exp_count}",
    )
    if exp_count > 0 and conds:
      _check(
        f"  first op in {falsifier!r}",
        conds[0].op == exp_op,
        f"got {conds[0].op} expected {exp_op}",
      )
      _check(
        f"  first value in {falsifier!r}",
        abs(conds[0].value - exp_val) < 0.01,
        f"got {conds[0].value} expected {exp_val}",
      )


# ===========================================================================
# 2. Numeric evaluation
# ===========================================================================

def test_numeric_evaluation():
  _section("2. Numeric condition evaluation")

  # gt
  c = NumericCondition('10Y treasury', 'gt', 5.25, '%', '> 5.25%')
  _check("  5.5 > 5.25 -> True",  evaluate_numeric_condition(c, 5.5)  is True)
  _check("  5.0 > 5.25 -> False", evaluate_numeric_condition(c, 5.0)  is False)
  _check("  5.25 > 5.25 -> False (strict)", evaluate_numeric_condition(c, 5.25) is False)
  _check("  None -> None", evaluate_numeric_condition(c, None) is None)

  # lt
  c2 = NumericCondition('Azure growth', 'lt', 18.0, '%', '< 18%')
  _check("  15 < 18 -> True", evaluate_numeric_condition(c2, 15.0) is True)
  _check("  20 < 18 -> False", evaluate_numeric_condition(c2, 20.0) is False)

  # gte
  c3 = NumericCondition('coverage', 'gte', 5.0, '', '>= 5')
  _check("  5.0 >= 5.0 -> True", evaluate_numeric_condition(c3, 5.0) is True)
  _check("  4.99 >= 5.0 -> False", evaluate_numeric_condition(c3, 4.99) is False)


# ===========================================================================
# 3. Text matching basics
# ===========================================================================

def test_text_matching():
  _section("3. Text matching")

  # Identical
  r = score_evidence_against_falsifier(
    "Azure cloud revenue growth slowed below twenty percent",
    "Azure cloud revenue growth slowed below twenty percent",
  )
  _check("  identical text scores high (>=0.5)", r['score'] >= 0.5,
         f"score={r['score']}")

  # Totally unrelated
  r = score_evidence_against_falsifier(
    "Azure cloud revenue growth slowed",
    "Federal Reserve discusses inflation outlook",
  )
  _check("  unrelated text scores low (<0.2)", r['score'] < 0.2,
         f"score={r['score']}")

  # Proper-noun match boost
  r = score_evidence_against_falsifier(
    "CFO Amy Hood replaced unexpectedly",
    "Microsoft announces management change involving Amy Hood",
  )
  _check("  proper-noun boost lifts ambiguous match >= 0.25",
         r['score'] >= 0.25, f"score={r['score']}")
  _check("  proper noun captured", 'Amy Hood' in r['proper_nouns_matched'],
         f"got {r['proper_nouns_matched']}")

  # Polarity mismatch
  r = score_evidence_against_falsifier(
    "Hood NOT replaced as CFO",
    "Hood replaced as CFO effective immediately",
  )
  _check("  polarity mismatch detected",
         r['falsifier_negates'] is True and r['evidence_negates'] is False,
         f"got falsifier_negates={r['falsifier_negates']} evidence_negates={r['evidence_negates']}")


def test_negation():
  _section("3b. Negation detection")
  _check("  'not present'", has_negation("the CFO has not been replaced") is True)
  _check("  'no significant change'", has_negation("no significant change in guidance") is True)
  _check("  'guidance affirmed'", has_negation("guidance affirmed at midpoint") is False)
  _check("  empty string", has_negation("") is False)


# ===========================================================================
# 4. Adversarial — false-positive traps
# ===========================================================================

def test_adversarial():
  _section("4. Adversarial / false-positive guards")

  # Trap 1: Same tokens, opposite meaning (negation)
  r = evaluate_falsifier(
    "CEO Nadella sells more than $50M in stock within 60 days of an earnings miss",
    [{'text': "CEO Nadella is highly committed to Microsoft; has no plans to sell stock"}],
  )
  _check("  trap 1: negation prevents false trigger on CEO sale falsifier",
         not r.triggered, f"score={r.score} reason={r.reason}")

  # Trap 2: Generic token overlap with no proper noun
  r = evaluate_falsifier(
    "Azure revenue growth drops below 18% YoY for two consecutive quarters",
    [{'text': "growth across many sectors of the economy has remained robust"}],
  )
  _check("  trap 2: generic-word overlap doesn't trigger Azure-specific falsifier",
         not r.triggered, f"score={r.score} reason={r.reason}")

  # Trap 3: Right entity, wrong action
  r = evaluate_falsifier(
    "CFO Amy Hood replaced",
    [{'text': "Amy Hood reiterated guidance for fiscal 2026 on the earnings call"}],
  )
  # Without proper-noun boost, scores could match — proper noun gives boost
  # of 0.15 + small token overlap. The threshold (0.35) should prevent firing
  # unless evidence strongly says "replaced". Acceptable: not triggered.
  _check("  trap 3: same-entity benign mention doesn't trigger 'replaced' falsifier",
         not r.triggered, f"score={r.score} reason={r.reason}")

  # Trap 4: Numeric condition with no observed value should NOT trigger
  r = evaluate_falsifier(
    "10Y treasury exceeds 5.25%",
    [{'text': "the 10-year treasury yield is in focus this week"}],
    observed_values=None,
  )
  _check("  trap 4: numeric falsifier without observed data does not trigger",
         not r.triggered, f"score={r.score} reason={r.reason}")


# ===========================================================================
# 5. End-to-end true-positive cases
# ===========================================================================

def test_true_positives():
  _section("5. True-positive scenarios — should TRIGGER")

  # Case 1: Numeric breach via observed_values
  r = evaluate_falsifier(
    "10Y treasury > 5.25%",
    [],
    observed_values={'10Y treasury': 5.45},
  )
  _check("  numeric breach 10Y 5.45 > 5.25 -> triggered",
         r.triggered and r.reason == 'numeric_threshold_breached',
         f"triggered={r.triggered} reason={r.reason}")

  # Case 2: Strong text + proper-noun match
  r = evaluate_falsifier(
    "Amy Hood replaced as CFO of Microsoft",
    [{'text': 'Microsoft announces that Amy Hood will be replaced as Chief Financial Officer effective immediately. CEO Satya Nadella thanked her for her service.',
      'source': '8-K Item 5.02', 'event_id': 'test-1'}],
  )
  _check("  exact-match scenario triggers (Amy Hood replaced)",
         r.triggered, f"score={r.score} reason={r.reason}")
  _check("  best_evidence is populated",
         r.best_evidence is not None and r.best_evidence.get('event_id') == 'test-1')

  # Case 3: Multiple evidence — pick the best
  r = evaluate_falsifier(
    "Azure cloud revenue growth drops below 18% YoY",
    [
      {'text': 'Microsoft announced a new dividend', 'event_id': 'noise-1'},
      {'text': 'Azure cloud revenue grew 15% year over year, below the 18% threshold the company had previously guided', 'event_id': 'signal-1'},
      {'text': 'Federal Reserve held rates steady', 'event_id': 'noise-2'},
    ],
  )
  _check("  picks signal-1 from noisy pool", r.triggered,
         f"triggered={r.triggered} score={r.score}")
  _check("  best_evidence is signal-1",
         r.best_evidence and r.best_evidence.get('event_id') == 'signal-1',
         f"got {r.best_evidence}")


# ===========================================================================
# 6. Performance
# ===========================================================================

def test_performance():
  _section("6. Performance — 100 falsifiers x 50 evidence items")

  falsifiers = [
    "Azure revenue growth drops below 18% YoY",
    "CFO Amy Hood replaced",
    "10Y treasury > 5.25%",
    "HY OAS > 400bps",
    "Capex grows above 28% of revenue",
    "Operating margin compresses below 40%",
    "Insider net sales above $200M in 60 days",
    "Material adverse event in cloud segment",
    "Major customer terminates contract",
    "Regulatory investigation announced",
  ] * 10  # 100 falsifiers

  evidence_pool = [
    {'text': f"Sample news event {i}: Microsoft reported solid quarterly results", 'event_id': f'ev-{i}'}
    for i in range(50)
  ]
  # Inject one true-positive
  evidence_pool[27] = {
    'text': 'Microsoft Azure cloud revenue grew only 15% year-over-year, below the 18% threshold',
    'event_id': 'ev-27',
  }

  t0 = time.time()
  trigger_count = 0
  for f in falsifiers:
    r = evaluate_falsifier(f, evidence_pool)
    if r.triggered:
      trigger_count += 1
  elapsed = time.time() - t0

  total_ops = len(falsifiers) * len(evidence_pool)
  _check(f"  100 falsifiers x 50 evidence ({total_ops} ops) under 5 seconds",
         elapsed < 5.0, f"elapsed={elapsed:.3f}s")
  _check("  exactly one falsifier should trigger (Azure)",
         trigger_count == 10, f"got {trigger_count}; expected 10 (Azure falsifier repeated 10 times)")
  print(f"  -- {total_ops} scored, {elapsed:.3f}s ({total_ops / elapsed:.0f} ops/sec)")


# ===========================================================================
# 7. Real-world data integration test
# ===========================================================================

def test_realworld_falsifiers():
  _section("7. Real-world falsifier samples")

  # These are the actual falsifiers we used in the MSFT thesis earlier
  falsifiers = [
    "Azure revenue growth drops below 18% YoY for 2 consecutive quarters",
    "Hood replaced as CFO",
    "10Y treasury > 5.25% with HY OAS > 400bps",
    "Any insider open-market sell of >$50M by Nadella or Hood within 60 days of an earnings miss",
    "Azure cloud-revenue growth print under 20% YoY with operating margin compression vs. FY25",
  ]

  # Compose a synthetic event pool with some real triggers and a lot of noise
  events = [
    {'text': 'Microsoft reports record cloud revenue', 'event_id': 'real-1'},
    {'text': 'Federal Reserve minutes from May meeting', 'event_id': 'real-2'},
    {'text': 'Apple announces new product lineup', 'event_id': 'real-3'},
    {'text': 'Tech sector rally continues into Q2', 'event_id': 'real-4'},
  ]
  observed = {'10Y treasury': 4.67, 'HY OAS': 286}  # current macro

  print(f"  testing {len(falsifiers)} real falsifiers against {len(events)} noisy events")
  triggered = []
  for f in falsifiers:
    r = evaluate_falsifier(f, events, observed_values=observed)
    if r.triggered:
      triggered.append((f, r.reason, r.score))
      print(f"    TRIGGERED: {f[:60]}... ({r.reason}, score={r.score})")
  _check("  no false positives on noisy benign events",
         len(triggered) == 0, f"got {len(triggered)} triggers; expected 0")


# ===========================================================================
# Main
# ===========================================================================

def main():
  print("\nFalsifier Evaluator — stress test suite\n")
  test_numeric_parsing()
  test_numeric_evaluation()
  test_text_matching()
  test_negation()
  test_adversarial()
  test_true_positives()
  test_performance()
  test_realworld_falsifiers()

  print(f"\n=== Summary ===")
  print(f"  PASS: {_results['pass']}")
  print(f"  FAIL: {_results['fail']}")
  if _results['failures']:
    print("\nFailures:")
    for name, hint in _results['failures']:
      print(f"  - {name}: {hint}")
  return 0 if _results['fail'] == 0 else 1


if __name__ == "__main__":
  sys.exit(main())
