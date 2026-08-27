"""End-to-end simulation of the /sentry-tick decision logic.

This test does NOT invoke Claude or the MCP layer — those are tested
elsewhere. It exercises the deterministic glue: queue read → cooldown
check → simulated skill verdicts → eval log writes → budget consumption
→ queue state transitions.

Each scenario sets up a synthetic candidate and then walks the
decision tree as the skill body would, asserting the expected
side-effects.

Scenarios:
  A. Cooldown skip (recent eval blocks re-research)
  B. Read-through inconclusive → research_synthesis card
  C. Research budget exhausted → tick stops
  D. Portfolio-fit rejects → research_synthesis (no trade)
  E. Risk_Officer rejects → trade_rejected (no order)
  F. Daily new-position cap hit → trade_rejected
  G. Happy path: all gates pass → trade_placed, eval+budget recorded

Why the scenarios are named test_scenario_*
-------------------------------------------
They used to be named scenario_A_..., which pytest does not collect. The file
sat in testing/ with a _check helper and twenty-four checks in it and pytest
ran none of them -- the seven scenarios only executed if someone remembered to
run the file as a script. Each scenario calls _cleanup() on entry, so they are
order-independent and safe to collect individually.

Run:
  .venv\\Scripts\\python.exe testing\\test_sentry_tick_e2e.py
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from state.schema import init_schema, get_connection
from state import sentry_queue, sentry_eval_log
from agent import sentry_budget as B


_results = {'pass': 0, 'fail': 0, 'failures': []}
_TICKER_PREFIX = 'E2E_'


def _check(name: str, condition: bool, hint: str = '') -> None:
  """Fail the test when `condition` is false.

  This helper used to increment a counter and print PASS/FAIL, and nothing
  else. Under pytest -- which never calls main() -- no one read the counter,
  so every test_* function in this file ran to completion, returned None and
  was reported green no matter what the code did. A gate that cannot fail is
  not a gate.

  The counters and the printed lines are kept so the summary still reads the
  same; the raise is what makes pytest see a wrong value.
  """
  if condition:
    _results['pass'] += 1
    print(f"  PASS  {name}")
  else:
    _results['fail'] += 1
    _results['failures'].append((name, hint))
    print(f"  FAIL  {name}  --  {hint}")
    raise AssertionError(f"{name}: {hint}" if hint else name)


def _cleanup():
  conn = get_connection()
  try:
    conn.execute("DELETE FROM sentry_queue WHERE ticker LIKE ?", (f"{_TICKER_PREFIX}%",))
    conn.execute("DELETE FROM sentry_evaluation_log WHERE ticker LIKE ?", (f"{_TICKER_PREFIX}%",))
    conn.execute("DELETE FROM sentry_daily_actions WHERE day = ?", (B._today_et(),))
    conn.commit()
  finally:
    conn.close()


def _setup_candidate(ticker: str, score: float = 0.75) -> int:
  """Inject a pending queue row for `ticker`. Returns queue_id."""
  return sentry_queue.enqueue(ticker, score, triggered_by='event_score',
                              source_event_id='e2e_synthetic_event')


# ----------------------------------------------------------------------------
# Scenario A: Cooldown skip
# ----------------------------------------------------------------------------
def test_scenario_A_cooldown_skip():
  print("\n== Scenario A: cooldown skip ==")
  _cleanup()
  ticker = f"{_TICKER_PREFIX}COOL"

  # Pre-record an 'acted' eval — sets 7-day cooldown
  sentry_eval_log.record_eval(
    ticker, decision='acted', triggered_by='manual',
    verdict='buy', confidence=0.72, sizing='cautious',
  )

  # Now queue a candidate for the same ticker
  qid = _setup_candidate(ticker)
  _check("queue row created", qid is not None)

  # Skill checks should_skip — should return True
  skip, reason = sentry_eval_log.should_skip(ticker)
  _check("should_skip returns True due to cooldown", skip, f"reason: {reason}")

  # Skill would record skipped eval + mark queue as 'completed' (or 'dropped')
  sentry_eval_log.record_eval(
    ticker, decision='skipped_cooldown', triggered_by='event_score',
    skip_reason=reason,
  )
  sentry_queue.mark_status(qid, 'completed', notes_append='skipped_cooldown')

  # Verify eval log has 2 rows (the original 'acted' + the 'skipped')
  evals = sentry_eval_log.recent_evals_for_ticker(ticker)
  _check("eval log has 2 rows after skip", len(evals) == 2)
  _check("most recent eval is skipped_cooldown",
         evals[0]['decision'] == 'skipped_cooldown')


# ----------------------------------------------------------------------------
# Scenario B: Read-through inconclusive → research_synthesis
# ----------------------------------------------------------------------------
def test_scenario_B_inconclusive_readthrough():
  print("\n== Scenario B: read-through inconclusive ==")
  _cleanup()
  ticker = f"{_TICKER_PREFIX}INCONC"
  qid = _setup_candidate(ticker)

  # Skill runs /cross-company-readthrough, gets verdict='inconclusive', conf=0.40
  # Simulates the skipping branch.
  #
  # contradiction_check_passed and factor_buckets are mandatory for
  # researched/no_position -- _validate_audit_fields in state/sentry_eval_log.py
  # raises ValueError without them. This call omitted both, so it had been
  # simulating a record_eval that production would reject outright. The
  # scenario never ran under pytest, so nothing said so. SKILL.md line 135
  # shows the real skill body passing them; this now matches.
  sentry_eval_log.record_eval(
    ticker, decision='researched', triggered_by='event_score',
    verdict='no_position', confidence=0.40, sizing='no_position',
    skip_reason='cross-company-readthrough inconclusive',
    contradiction_check_passed=True,
    factor_buckets=['ai_capex_long'],
  )
  sentry_queue.mark_status(qid, 'completed')

  # Verify
  evals = sentry_eval_log.recent_evals_for_ticker(ticker)
  _check("eval recorded as researched with no_position", evals[0]['verdict'] == 'no_position')
  _check("queue row marked completed",
         sentry_queue.get_by_id(qid)['status'] == 'completed')

  # Cooldown: no_position → 30 days, so next_review_at should be ~30d out
  _check("30-day cooldown set", evals[0]['next_review_at'] is not None)


# ----------------------------------------------------------------------------
# Scenario C: Research budget exhausted
# ----------------------------------------------------------------------------
def test_scenario_C_research_budget_exhausted():
  print("\n== Scenario C: research budget exhausted mid-tick ==")
  _cleanup()
  ticker = f"{_TICKER_PREFIX}BUDGET"
  qid = _setup_candidate(ticker)

  # Burn the research budget
  for _ in range(B.MAX_RESEARCH_RUNS_PER_DAY):
    B.record_action(B.ACTION_RESEARCH)

  # Skill checks can_research before invoking deep-research
  ok, reason = B.can_research()
  _check("can_research returns False after cap hit", not ok)

  # Skill resets the queue row to pending (will retry tomorrow) — does NOT
  # record an eval (the row is just postponed)
  sentry_queue.mark_status(qid, 'pending', notes_append=f"deferred: {reason}")

  # Verify queue row is back to pending (will be retried next tick / day)
  row = sentry_queue.get_by_id(qid)
  _check("queue row reset to pending", row['status'] == 'pending')


# ----------------------------------------------------------------------------
# Scenario D: Portfolio-fit rejects → research recorded, no trade
# ----------------------------------------------------------------------------
def test_scenario_D_portfolio_fit_reject():
  print("\n== Scenario D: portfolio-fit rejects ==")
  _cleanup()
  ticker = f"{_TICKER_PREFIX}FIT"
  qid = _setup_candidate(ticker)

  # Simulate deep-research happened (record budget)
  B.record_action(B.ACTION_RESEARCH)

  # /portfolio-fit returned verdict='reject' (e.g., book already 28% AI)
  # Skill skips the trade gate, records research with downgraded sizing
  sentry_eval_log.record_eval(
    ticker, decision='researched', triggered_by='event_score',
    verdict='watchlist', confidence=0.70, sizing='watchlist',
    skip_reason='portfolio-fit reject: AI capex bucket at 28%',
    contradiction_check_passed=True,
    factor_buckets=['ai_capex_long'],
  )
  sentry_queue.mark_status(qid, 'completed')

  evals = sentry_eval_log.recent_evals_for_ticker(ticker)
  _check("eval verdict downgraded to watchlist", evals[0]['verdict'] == 'watchlist')
  _check("factor_buckets recorded as list",
         evals[0].get('factor_buckets') == ['ai_capex_long'])
  _check("research budget consumed", B.summary()['research_runs'].startswith('1/'))


# ----------------------------------------------------------------------------
# Scenario E: Risk_Officer rejects → trade_rejected card, no order placed
# ----------------------------------------------------------------------------
def test_scenario_E_risk_officer_rejects():
  print("\n== Scenario E: Risk_Officer rejects the proposal ==")
  _cleanup()
  ticker = f"{_TICKER_PREFIX}RISK"
  qid = _setup_candidate(ticker)

  B.record_action(B.ACTION_RESEARCH)

  # Simulate risk_check_proposed_trade returning approve=False
  # Skill records skipped eval; no place_paper_order call
  # researched/buy is the strictest row the validator knows: all four
  # discipline audit fields are required. See scenario B for why they were
  # missing here.
  sentry_eval_log.record_eval(
    ticker, decision='researched', triggered_by='event_score',
    verdict='buy', confidence=0.68, sizing='cautious',
    skip_reason='risk_officer rejected: max position size exceeded',
    analogue_considered='2018 memory downcycle',
    terminal_sensitivity_ran=True,
    contradiction_check_passed=True,
    factor_buckets=['memory_cycle'],
  )
  sentry_queue.mark_status(qid, 'completed')

  # No paper_orders counter incremented (the order was never placed)
  _check("paper_orders counter NOT incremented when risk_officer rejects",
         B.summary()['paper_orders'] == '0/5')

  # But research counter IS incremented
  _check("research counter incremented", B.summary()['research_runs'].startswith('1/'))


# ----------------------------------------------------------------------------
# Scenario F: Daily new-position cap hit → trade rejected
# ----------------------------------------------------------------------------
def test_scenario_F_new_position_cap_hit():
  print("\n== Scenario F: daily new-position cap hit ==")
  _cleanup()
  ticker = f"{_TICKER_PREFIX}CAP"
  qid = _setup_candidate(ticker)

  B.record_action(B.ACTION_RESEARCH)

  # Pre-burn the new-position cap
  for _ in range(B.MAX_NEW_POSITIONS_PER_DAY):
    B.record_action(B.ACTION_PAPER_ORDER)
    B.record_action(B.ACTION_NEW_POSITION)

  # Skill checks can_open_new_position before placing
  ok, reason = B.can_open_new_position()
  _check("can_open_new_position blocks at cap", not ok)

  # Skill records skipped eval
  sentry_eval_log.record_eval(
    ticker, decision='researched', triggered_by='event_score',
    verdict='buy', confidence=0.75, sizing='cautious',
    skip_reason=f'budget cap: {reason}',
    analogue_considered='none',
    terminal_sensitivity_ran=True,
    contradiction_check_passed=True,
    factor_buckets=['ai_capex_long'],
  )
  sentry_queue.mark_status(qid, 'completed')

  # paper_orders still at cap (2), not 3 (we didn't place this one)
  _check("paper_orders unchanged after cap-blocked attempt",
         B.summary()['paper_orders'] == f'{B.MAX_NEW_POSITIONS_PER_DAY}/5')


# ----------------------------------------------------------------------------
# Scenario G: Happy path — all gates pass, trade placed, full audit trail
# ----------------------------------------------------------------------------
def test_scenario_G_happy_path():
  print("\n== Scenario G: happy path — all gates pass ==")
  _cleanup()
  ticker = f"{_TICKER_PREFIX}HAPPY"
  qid = _setup_candidate(ticker)

  # Simulate full path: research, all gates pass, risk_check approves, order placed
  B.record_action(B.ACTION_RESEARCH)
  # ... portfolio_fit OK, factor_exposure OK, kill_switch intact, risk_officer approves ...
  B.record_action(B.ACTION_PAPER_ORDER)
  B.record_action(B.ACTION_NEW_POSITION)
  B.record_action(B.ACTION_SLACK)   # trade_placed card

  sentry_eval_log.record_eval(
    ticker, decision='acted', triggered_by='event_score',
    verdict='buy', confidence=0.74, sizing='cautious',
    factor_buckets=['memory_cycle'], notes='order_id=PAPER_12345',
  )
  sentry_queue.mark_status(qid, 'completed')

  # Verify audit trail
  evals = sentry_eval_log.recent_evals_for_ticker(ticker)
  _check("eval decision is 'acted'", evals[0]['decision'] == 'acted')
  _check("eval verdict is 'buy'", evals[0]['verdict'] == 'buy')
  _check("7-day cooldown set", evals[0]['next_review_at'] is not None)
  _check("queue row completed", sentry_queue.get_by_id(qid)['status'] == 'completed')

  # Verify budget consumption
  s = B.summary()
  _check("research counter +1", s['research_runs'].startswith('1/'))
  _check("paper_orders counter +1", s['paper_orders'].startswith('1/'))
  _check("new_positions counter +1", s['new_positions'].startswith('1/'))
  _check("slack counter +1", s['slack_messages'].startswith('1/'))


def main():
  init_schema()
  print("\n/sentry-tick decision-logic end-to-end simulation\n")
  test_scenario_A_cooldown_skip()
  test_scenario_B_inconclusive_readthrough()
  test_scenario_C_research_budget_exhausted()
  test_scenario_D_portfolio_fit_reject()
  test_scenario_E_risk_officer_rejects()
  test_scenario_F_new_position_cap_hit()
  test_scenario_G_happy_path()
  _cleanup()

  print(f"\n== Summary ==\n  PASS: {_results['pass']}\n  FAIL: {_results['fail']}")
  for n, h in _results['failures']:
    print(f"  - {n}: {h}")
  return 0 if _results['fail'] == 0 else 1


if __name__ == "__main__":
  sys.exit(main())
