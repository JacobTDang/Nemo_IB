"""Phase 2b: Thesis_Maintainer real-call evaluation.

Five hand-crafted thesis+event pairs with expected verdicts. The agent should
match human judgment on at least 4 of 5 (allowing one edge-case disagreement).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.Thesis_Maintainer import Thesis_Maintainer, ThesisVerdict


# (label, thesis, event, expected_status)
SCENARIOS = [
  ("earnings_miss_breaks_buy",
   {
     "ticker": "AAPL", "recommendation": "BUY", "signal": "bullish",
     "confidence": 0.8, "target_price": 215.0, "stop_loss": 175.0,
     "analysis_summary": "Strong services growth + iPhone refresh cycle drives EPS expansion.",
     "key_assumptions": ["iPhone revenue +5% YoY",
                          "Services growth +12% YoY",
                          "Operating margin holds at 30%+"],
     "data_gaps": [],
     "thesis_date": "2026-04-01T10:00:00",
   },
   {
     "headline": "Apple posts surprise revenue miss, guides Q3 below consensus",
     "source": "test:wsj",
     "published_at": "2026-05-01T16:30:00",
     "materiality": "high", "category": "earnings",
     "directional_signal": "bearish", "urgency": "immediate",
     "body": "Apple (AAPL) reported revenue of $80B vs $85B consensus. "
             "iPhone revenue fell 8% YoY. CFO guided Q3 to ~$78B vs $84B consensus. "
             "Stock down 6% after-hours.",
     "classifier_reason": "Q2 miss + Q3 guide down"
   },
   "broken"),

  ("unrelated_event_intact",
   {
     "ticker": "AAPL", "recommendation": "BUY", "signal": "bullish",
     "confidence": 0.7, "target_price": 215.0, "stop_loss": 175.0,
     "analysis_summary": "Services + iPhone cycle drives growth.",
     "key_assumptions": ["iPhone revenue +5% YoY", "Services +12% YoY"],
     "data_gaps": [], "thesis_date": "2026-04-01T10:00:00",
   },
   {
     "headline": "Tim Cook gives speech at university graduation",
     "source": "test:bloomberg",
     "published_at": "2026-05-09T10:00:00",
     "materiality": "low", "category": "other",
     "directional_signal": "neutral", "urgency": "watch",
     "body": "Apple CEO Tim Cook delivered the commencement address at a university, "
             "discussing his career path and the importance of education.",
     "classifier_reason": "non-business speech"
   },
   "intact"),

  ("macro_weakens_sector_thesis",
   {
     "ticker": "NVDA", "recommendation": "BUY", "signal": "bullish",
     "confidence": 0.75, "target_price": 950.0, "stop_loss": 750.0,
     "analysis_summary": "Hyperscaler AI capex expansion drives accelerator demand.",
     "key_assumptions": ["Hyperscaler capex +30% YoY",
                          "No major China export restriction expansion",
                          "Power infrastructure keeps up with demand"],
     "data_gaps": [], "thesis_date": "2026-04-15T10:00:00",
   },
   {
     "headline": "Major hyperscaler cuts 2026 capex guidance, cites macro uncertainty",
     "source": "test:reuters",
     "published_at": "2026-05-10T08:00:00",
     "materiality": "high", "category": "sector",
     "directional_signal": "bearish", "urgency": "hours",
     "body": "A major US hyperscaler reduced its 2026 capex guidance from $80B to $65B, "
             "citing macro uncertainty. The company specifically called out AI infrastructure "
             "as 'on a more measured pace' than prior plans.",
     "classifier_reason": "capex guide down"
   },
   "weakened"),

  ("earnings_beat_breaks_sell",
   {
     "ticker": "TSLA", "recommendation": "SELL", "signal": "bearish",
     "confidence": 0.7, "target_price": 150.0, "stop_loss": 280.0,
     "analysis_summary": "Demand weakness and margin compression justify SELL.",
     "key_assumptions": ["Vehicle margins compress further",
                          "Delivery growth flat to negative",
                          "No new product catalyst"],
     "data_gaps": [], "thesis_date": "2026-04-01T10:00:00",
   },
   {
     "headline": "Tesla smashes Q2 deliveries, automotive gross margin expands 200bps",
     "source": "test:reuters",
     "published_at": "2026-05-08T16:00:00",
     "materiality": "high", "category": "earnings",
     "directional_signal": "bullish", "urgency": "immediate",
     "body": "Tesla (TSLA) reported Q2 deliveries of 480K vs 425K expected. "
             "Automotive gross margin expanded 200bps QoQ to 18.5%. "
             "Stock up 12% after-hours.",
     "classifier_reason": "Deliveries beat + margin expansion"
   },
   "broken"),

  ("ceo_change_weakens_or_breaks",
   {
     "ticker": "JPM", "recommendation": "BUY", "signal": "bullish",
     "confidence": 0.65, "target_price": 230.0, "stop_loss": 180.0,
     "analysis_summary": "Dimon-era operating leverage + capital return discipline supports BUY.",
     "key_assumptions": ["Jamie Dimon leadership continuity",
                          "ROTCE ~20% sustained",
                          "Buyback authorization maintained"],
     "data_gaps": [], "thesis_date": "2026-04-01T10:00:00",
   },
   {
     "headline": "Jamie Dimon to retire by year-end; succession plan not yet announced",
     "source": "test:wsj",
     "published_at": "2026-05-09T07:00:00",
     "materiality": "high", "category": "exec_change",
     "directional_signal": "neutral", "urgency": "days",
     "body": "Jamie Dimon, CEO of JPMorgan Chase since 2005, announced he will retire "
             "by year-end. The board has not yet identified a successor.",
     "classifier_reason": "key-person risk realized"
   },
   ("weakened", "broken")),  # either is defensible
]


def test_thesis_maintainer_scenarios():
  maintainer = Thesis_Maintainer()
  failures = []
  for label, thesis, event, expected in SCENARIOS:
    print(f"\n--- {label} ---")
    v = maintainer.evaluate(thesis, event)
    assert v is not None, f"{label}: maintainer returned None (parse failure)"
    assert isinstance(v, ThesisVerdict)
    print(f"  status={v.status} action={v.recommended_action} severity={v.severity}")
    print(f"  reasoning: {v.reasoning[:140]}")
    print(f"  affected: {v.affected_assumptions}")
    # Accept tuple of allowed verdicts
    expected_set = expected if isinstance(expected, tuple) else (expected,)
    if v.status not in expected_set:
      failures.append(f"{label}: expected status in {expected_set}, got {v.status}")
    # Severity must match status
    if v.severity != v.status:
      failures.append(f"{label}: severity ({v.severity}) != status ({v.status})")
  # Tolerance: at most 1 disagreement out of 5 (matches "30% disagreement stop condition")
  if len(failures) > 1:
    raise AssertionError(
      f"Thesis_Maintainer failed {len(failures)} of {len(SCENARIOS)} scenarios:\n  "
      + "\n  ".join(failures)
    )
  if failures:
    print(f"\nWARN: 1 edge-case disagreement (acceptable):\n  " + "\n  ".join(failures))
  print(f"\nPASS: Thesis_Maintainer matched expected verdicts on "
        f"{len(SCENARIOS) - len(failures)}/{len(SCENARIOS)} scenarios.")


def test_broken_status_triggers_reanalysis_action():
  """Whenever status=broken, action must be trigger_reanalysis."""
  m = Thesis_Maintainer()
  # Reuse the earnings_miss scenario
  thesis = SCENARIOS[0][1]; event = SCENARIOS[0][2]
  v = m.evaluate(thesis, event)
  assert v is not None
  if v.status == 'broken':
    assert v.recommended_action == 'trigger_reanalysis', \
      f"broken status should trigger reanalysis, got action={v.recommended_action}"
    print("PASS: broken -> trigger_reanalysis invariant holds")
  else:
    print(f"SKIP: scenario returned status={v.status}, not broken — invariant not exercised")


if __name__ == "__main__":
  test_thesis_maintainer_scenarios()
  test_broken_status_triggers_reanalysis_action()
  print("\nAll Phase 2b thesis maintainer tests passed.")
