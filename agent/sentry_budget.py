"""Daily budget gate for the Sentry action layer.

Caps per ET day (America/New_York):
  - 10 research runs            (cost / cadence control)
  - 50 Slack messages           (spam control)
  - 5 paper orders              (sanity)
  - 2 NEW positions             (subset of paper orders — adds/trims unlimited within size cap)
  - 30s per MCP call timeout    (not enforced here; documented for the skill)

The budget is persisted in `sentry_daily_actions` table. Each /sentry-tick
invocation reads the current day's counters, decides whether the next
action is permitted, and increments after the action completes. Day
rollover is automatic — a new row is created on the first action of the
next ET day.
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Tuple

from state.schema import get_connection


# -- Cap constants -----------------------------------------------------------
MAX_RESEARCH_RUNS_PER_DAY    = 10
MAX_SLACK_MESSAGES_PER_DAY   = 50
MAX_PAPER_ORDERS_PER_DAY     = 5
MAX_NEW_POSITIONS_PER_DAY    = 2
MAX_MCP_CALL_S               = 30

# Action types (must match column names in sentry_daily_actions for clarity)
ACTION_RESEARCH         = 'research_runs'
ACTION_SLACK            = 'slack_messages'
ACTION_PAPER_ORDER      = 'paper_orders'
ACTION_NEW_POSITION     = 'new_positions'
ACTION_ADD_TRIM         = 'adds_or_trims'


# Offset from UTC to America/New_York. The ET timezone shifts between EST
# (UTC-5) and EDT (UTC-4) for daylight saving — for budget rollover precision
# is unnecessary, so we use a fixed approximation: ET = UTC-5. Worst case the
# day rolls over an hour early or late twice a year on the DST transition,
# which is acceptable for budget purposes.
_ET_OFFSET_HOURS = -5


def _today_et() -> str:
  """Return today's date string in ET as YYYY-MM-DD."""
  return (datetime.now(timezone.utc) + timedelta(hours=_ET_OFFSET_HOURS)).strftime('%Y-%m-%d')


def _ensure_row(day: str) -> None:
  """Insert a zero-counters row for `day` if it doesn't exist."""
  conn = get_connection()
  try:
    conn.execute(
      "INSERT OR IGNORE INTO sentry_daily_actions (day) VALUES (?)",
      (day,),
    )
    conn.commit()
  finally:
    conn.close()


def _read_counters(day: str | None = None) -> dict:
  """Return current counters for the day. Creates the row if missing."""
  if day is None:
    day = _today_et()
  _ensure_row(day)
  conn = get_connection()
  try:
    row = conn.execute(
      "SELECT * FROM sentry_daily_actions WHERE day = ?", (day,)
    ).fetchone()
    return dict(row) if row else {}
  finally:
    conn.close()


def can_research() -> Tuple[bool, str | None]:
  """Permit a /equity-deep-research run? Counts against MAX_RESEARCH_RUNS_PER_DAY."""
  counters = _read_counters()
  current = counters.get('research_runs', 0)
  if current >= MAX_RESEARCH_RUNS_PER_DAY:
    return (False, f"daily research cap hit ({current}/{MAX_RESEARCH_RUNS_PER_DAY})")
  return (True, None)


def can_post_slack() -> Tuple[bool, str | None]:
  """Permit a Slack notification? Counts against MAX_SLACK_MESSAGES_PER_DAY."""
  counters = _read_counters()
  current = counters.get('slack_messages', 0)
  if current >= MAX_SLACK_MESSAGES_PER_DAY:
    return (False, f"daily slack cap hit ({current}/{MAX_SLACK_MESSAGES_PER_DAY})")
  return (True, None)


def can_open_new_position() -> Tuple[bool, str | None]:
  """Permit place_paper_order for a NEW position (no existing position in this ticker)?

  Subject to TWO caps: total paper_orders AND new_positions specifically.
  """
  counters = _read_counters()
  orders = counters.get('paper_orders', 0)
  new_pos = counters.get('new_positions', 0)
  if orders >= MAX_PAPER_ORDERS_PER_DAY:
    return (False, f"daily paper-order cap hit ({orders}/{MAX_PAPER_ORDERS_PER_DAY})")
  if new_pos >= MAX_NEW_POSITIONS_PER_DAY:
    return (False, f"daily new-position cap hit ({new_pos}/{MAX_NEW_POSITIONS_PER_DAY})")
  return (True, None)


def can_add_to_existing() -> Tuple[bool, str | None]:
  """Permit place_paper_order for an ADD/TRIM to existing position?

  Subject only to the total paper_orders cap — adds/trims have unlimited
  count within that. The skill must still respect the per-position size
  cap (1% of book) separately via Risk_Officer.
  """
  counters = _read_counters()
  orders = counters.get('paper_orders', 0)
  if orders >= MAX_PAPER_ORDERS_PER_DAY:
    return (False, f"daily paper-order cap hit ({orders}/{MAX_PAPER_ORDERS_PER_DAY})")
  return (True, None)


def record_action(action_type: str, *, day: str | None = None) -> None:
  """Increment the named counter. `action_type` must be one of the ACTION_* constants.

  Order placements MUST record BOTH `paper_orders` and either
  `new_positions` or `adds_or_trims` — call this twice for clarity.
  """
  if action_type not in (ACTION_RESEARCH, ACTION_SLACK, ACTION_PAPER_ORDER,
                         ACTION_NEW_POSITION, ACTION_ADD_TRIM):
    raise ValueError(f"unknown action_type: {action_type}")

  if day is None:
    day = _today_et()
  _ensure_row(day)

  now_iso = datetime.now(timezone.utc).isoformat()
  conn = get_connection()
  try:
    conn.execute(
      f"""UPDATE sentry_daily_actions
          SET {action_type} = {action_type} + 1,
              first_action_at = COALESCE(first_action_at, ?),
              last_action_at = ?
          WHERE day = ?""",
      (now_iso, now_iso, day),
    )
    conn.commit()
  finally:
    conn.close()


def summary(day: str | None = None) -> dict:
  """Return a human-readable summary of today's budget consumption."""
  counters = _read_counters(day)
  return {
    'day':              counters.get('day'),
    'research_runs':    f"{counters.get('research_runs', 0)}/{MAX_RESEARCH_RUNS_PER_DAY}",
    'slack_messages':   f"{counters.get('slack_messages', 0)}/{MAX_SLACK_MESSAGES_PER_DAY}",
    'paper_orders':     f"{counters.get('paper_orders', 0)}/{MAX_PAPER_ORDERS_PER_DAY}",
    'new_positions':    f"{counters.get('new_positions', 0)}/{MAX_NEW_POSITIONS_PER_DAY}",
    'adds_or_trims':    counters.get('adds_or_trims', 0),
    'first_action_at':  counters.get('first_action_at'),
    'last_action_at':   counters.get('last_action_at'),
  }
