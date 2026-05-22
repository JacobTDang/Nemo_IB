"""Unit tests for agent.sentry_budget.

Verifies:
  - Each cap correctly blocks after threshold
  - Caps are independent (research cap doesn't block slack, etc.)
  - paper_orders cap covers both new and add scenarios
  - Day rollover gives a fresh budget
  - record_action increments the right counter
  - summary() shape

Run:
  .venv\\Scripts\\python.exe testing\\test_sentry_budget.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from state.schema import init_schema, get_connection
from agent import sentry_budget as B


_results = {'pass': 0, 'fail': 0, 'failures': []}


def _check(name: str, condition: bool, hint: str = '') -> None:
  if condition:
    _results['pass'] += 1
    print(f"  PASS  {name}")
  else:
    _results['fail'] += 1
    _results['failures'].append((name, hint))
    print(f"  FAIL  {name}  --  {hint}")


def _clear_today():
  conn = get_connection()
  try:
    conn.execute("DELETE FROM sentry_daily_actions WHERE day = ?", (B._today_et(),))
    conn.commit()
  finally:
    conn.close()


def test_fresh_day_permits_all():
  print("\n== fresh day: all 4 permits granted ==")
  _clear_today()
  ok, _ = B.can_research(); _check("can_research", ok)
  ok, _ = B.can_post_slack(); _check("can_post_slack", ok)
  ok, _ = B.can_open_new_position(); _check("can_open_new_position", ok)
  ok, _ = B.can_add_to_existing(); _check("can_add_to_existing", ok)


def test_research_cap():
  print("\n== research cap ==")
  _clear_today()
  for _ in range(B.MAX_RESEARCH_RUNS_PER_DAY):
    B.record_action(B.ACTION_RESEARCH)
  ok, reason = B.can_research()
  _check("research blocked at cap", not ok, f"reason: {reason}")
  # Slack should still be allowed
  ok2, _ = B.can_post_slack()
  _check("slack independent of research cap", ok2)


def test_slack_cap():
  print("\n== slack cap ==")
  _clear_today()
  for _ in range(B.MAX_SLACK_MESSAGES_PER_DAY):
    B.record_action(B.ACTION_SLACK)
  ok, reason = B.can_post_slack()
  _check("slack blocked at cap", not ok, f"reason: {reason}")
  ok2, _ = B.can_research()
  _check("research independent of slack cap", ok2)


def test_new_position_cap():
  print("\n== new-position cap blocks adds independently ==")
  _clear_today()
  # Fill the new-position cap (2) without using up all 5 paper orders
  for _ in range(B.MAX_NEW_POSITIONS_PER_DAY):
    B.record_action(B.ACTION_PAPER_ORDER)
    B.record_action(B.ACTION_NEW_POSITION)
  ok, reason = B.can_open_new_position()
  _check("new-position blocked at cap", not ok, f"reason: {reason}")
  # Adds should still be allowed (only 2 of 5 paper orders used)
  ok_add, _ = B.can_add_to_existing()
  _check("adds still permitted with budget remaining", ok_add)


def test_total_paper_order_cap():
  print("\n== total paper-order cap blocks both new + add ==")
  _clear_today()
  for _ in range(B.MAX_PAPER_ORDERS_PER_DAY):
    B.record_action(B.ACTION_PAPER_ORDER)
    B.record_action(B.ACTION_ADD_TRIM)
  ok_new, reason_new = B.can_open_new_position()
  ok_add, reason_add = B.can_add_to_existing()
  _check("new blocked when paper-order cap hit", not ok_new, f"reason: {reason_new}")
  _check("add blocked when paper-order cap hit", not ok_add, f"reason: {reason_add}")


def test_day_rollover():
  print("\n== day rollover: today is independent of yesterday ==")
  _clear_today()
  # Compute yesterday dynamically — hardcoded dates risk collision with
  # today's ET date when the test runs across UTC midnight
  from datetime import datetime, timezone, timedelta
  today_et = B._today_et()
  yesterday_et = (
      datetime.strptime(today_et, '%Y-%m-%d') - timedelta(days=1)
  ).strftime('%Y-%m-%d')

  # Fake a yesterday row that is full
  conn = get_connection()
  conn.execute(
    """INSERT OR REPLACE INTO sentry_daily_actions
       (day, research_runs, slack_messages, paper_orders, new_positions)
       VALUES (?, 999, 999, 999, 999)""",
    (yesterday_et,),
  )
  conn.commit()
  conn.close()

  # Today should be unaffected by yesterday's full row
  ok_r, _ = B.can_research()
  ok_s, _ = B.can_post_slack()
  ok_n, _ = B.can_open_new_position()
  _check("today research independent of yesterday", ok_r,
         f"today={today_et} yesterday={yesterday_et}")
  _check("today slack independent of yesterday", ok_s)
  _check("today new-position independent of yesterday", ok_n)

  # Cleanup yesterday's fake row
  conn = get_connection()
  conn.execute("DELETE FROM sentry_daily_actions WHERE day = ?", (yesterday_et,))
  conn.commit()
  conn.close()


def test_record_action_invalid_raises():
  print("\n== record_action validates action_type ==")
  raised = False
  try:
    B.record_action('not_a_valid_action')
  except ValueError:
    raised = True
  _check("record_action raises ValueError on invalid type", raised)


def test_summary_shape():
  print("\n== summary returns expected keys ==")
  _clear_today()
  B.record_action(B.ACTION_RESEARCH)
  B.record_action(B.ACTION_SLACK)
  s = B.summary()
  expected_keys = {'day', 'research_runs', 'slack_messages', 'paper_orders',
                   'new_positions', 'adds_or_trims', 'first_action_at', 'last_action_at'}
  _check("summary has all expected keys", set(s.keys()) >= expected_keys,
         f"missing: {expected_keys - set(s.keys())}")
  _check("research_runs format is X/Y", '/' in str(s['research_runs']),
         f"got {s['research_runs']}")
  _check("first_action_at set after recording", s['first_action_at'] is not None)
  _check("last_action_at set after recording", s['last_action_at'] is not None)


def main():
  init_schema()
  print("Sentry budget unit tests\n")
  test_fresh_day_permits_all()
  test_research_cap()
  test_slack_cap()
  test_new_position_cap()
  test_total_paper_order_cap()
  test_day_rollover()
  test_record_action_invalid_raises()
  test_summary_shape()
  _clear_today()

  print(f"\n== Summary ==\n  PASS: {_results['pass']}\n  FAIL: {_results['fail']}")
  for n, h in _results['failures']:
    print(f"  - {n}: {h}")
  return 0 if _results['fail'] == 0 else 1


if __name__ == "__main__":
  sys.exit(main())
