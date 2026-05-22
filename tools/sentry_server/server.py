"""nemo_sentry MCP server -- exposes Sentry state-layer operations as MCP
tools so skills can read/mutate state without shelling out to `python -c`.

19 tools across:
  - queue (get, mark status, reset stuck, trim)
  - events (get, validate candidate, recent for ticker)
  - eval log (record, recent, should skip)
  - budget (summary, can act, record action)
  - portfolio state read (watchlist, positions, theses, latest thesis)
  - discovery admin (today's run status)
  - falsifier alerts (active alerts)

NOT a daemon. Started on demand by Claude Code via stdio transport when
the user has registered it via `claude mcp add -s user nemo_sentry`.

Register:
  claude mcp add -s user nemo_sentry -- \\
    "<repo>/.venv/Scripts/python.exe" -m tools.sentry_server.server
"""
from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from state.schema import get_connection, get_watchlist, init_schema
from state import sentry_queue, sentry_eval_log, events_store, positions, theses
from agent import sentry_budget

from tools.sentry_server._validation import validate_candidate


# ---------------------------------------------------------------------------
# Envelope helpers
# ---------------------------------------------------------------------------

def build_envelope(
  data: Any,
  tool: str,
  errors: Optional[List[str]] = None,
) -> Dict[str, Any]:
  return {
    'domain': 'sentry_state',
    'tool': tool,
    'timestamp': datetime.now(timezone.utc).isoformat(),
    'success': not bool(errors),
    'data': data,
    'metadata': {'errors': errors or []},
  }


def _ok(tool: str, data: Any) -> List[TextContent]:
  return [TextContent(
    type='text',
    text=json.dumps(build_envelope(data, tool), default=str),
  )]


def _err(tool: str, msg: str) -> List[TextContent]:
  return [TextContent(
    type='text',
    text=json.dumps(build_envelope(None, tool, errors=[msg]), default=str),
  )]


def _parse_affected(raw):
  if not raw:
    return []
  if isinstance(raw, list):
    return raw
  try:
    return json.loads(raw)
  except (json.JSONDecodeError, TypeError):
    return []


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

class SentryServer:
  def __init__(self):
    self.server = Server('sentry')
    init_schema()
    self._setup_handlers()

  # -- tool registry ------------------------------------------------------

  def _setup_handlers(self):
    parent = self

    @self.server.list_tools()
    async def list_tools() -> List[Tool]:
      return [
        # ----- Queue ops --------------------------------------------------
        Tool(
          name='sentry_get_queue',
          description=(
            'Read pending Sentry queue rows in score order, with the '
            'source event detail joined in and recent evaluations folded '
            'for each ticker. Replaces the multi-step bash-python pull '
            'in /sentry-tick. Use this to see what is next in the queue.'
          ),
          inputSchema={
            'type': 'object',
            'properties': {
              'limit': {'type': 'integer', 'default': 10,
                        'description': 'Max rows to return.'},
              'include_event_detail': {'type': 'boolean', 'default': True,
                                       'description': 'Join the source event row.'},
              'include_recent_evals': {'type': 'boolean', 'default': True,
                                       'description': 'Include recent eval_log rows per ticker (last 30 days).'},
            },
          },
        ),
        Tool(
          name='sentry_mark_queue_status',
          description=(
            'Update a queue row status to one of pending / processing / '
            'completed / dropped. Sets processed_at automatically when '
            'leaving pending. Optionally appends to the notes column.'
          ),
          inputSchema={
            'type': 'object',
            'required': ['queue_id', 'status'],
            'properties': {
              'queue_id': {'type': 'integer'},
              'status':   {'type': 'string',
                           'enum': ['pending', 'processing', 'completed', 'dropped']},
              'notes_append': {'type': 'string',
                               'description': 'Optional text to append to the notes column.'},
            },
          },
        ),
        Tool(
          name='sentry_reset_stuck_queue',
          description=(
            'Find queue rows stuck in processing for longer than '
            'stuck_minutes and flip them back to pending. Call at the '
            'start of /sentry-tick to recover from crashed prior ticks.'
          ),
          inputSchema={
            'type': 'object',
            'properties': {
              'stuck_minutes': {'type': 'integer', 'default': 15},
            },
          },
        ),
        Tool(
          name='sentry_trim_queue',
          description=(
            'Drop the lowest-scored pending rows until count <= max_pending. '
            'Useful for keeping the queue bounded after a flood of events.'
          ),
          inputSchema={
            'type': 'object',
            'properties': {
              'max_pending': {'type': 'integer', 'default': 20},
            },
          },
        ),

        # ----- Events -----------------------------------------------------
        Tool(
          name='sentry_get_event',
          description=(
            'Fetch a single row from the events table by event_id. Parses '
            'the JSON affected_tickers column. Returns null when not found.'
          ),
          inputSchema={
            'type': 'object',
            'required': ['event_id'],
            'properties': {
              'event_id': {'type': 'string',
                           'description': '16-char hex event id.'},
            },
          },
        ),
        Tool(
          name='sentry_validate_candidate',
          description=(
            'Pre-flight structural check on a queue row before the skill '
            'spends the deep-research budget. Returns ok=true when the '
            'queue.ticker is corroborated by the event (primary_ticker '
            'matches OR ticker appears in affected_tickers), the event '
            'is fresh, and the same (ticker, event_id) was not previously '
            'rejected as misclassified. Otherwise returns ok=false with '
            'a suggested_action (drop_misclassified, drop_stale, etc.).'
          ),
          inputSchema={
            'type': 'object',
            'required': ['queue_id'],
            'properties': {
              'queue_id': {'type': 'integer'},
            },
          },
        ),
        Tool(
          name='sentry_recent_events_for_ticker',
          description=(
            'Events touching a ticker (in affected_tickers or as primary) '
            'within the last N hours. Useful for /sentry-tick to gather '
            'the recent news context for a candidate.'
          ),
          inputSchema={
            'type': 'object',
            'required': ['ticker'],
            'properties': {
              'ticker': {'type': 'string'},
              'hours':  {'type': 'integer', 'default': 24},
            },
          },
        ),

        # ----- Eval log ---------------------------------------------------
        Tool(
          name='sentry_record_evaluation',
          description=(
            'Insert an eval row recording the outcome of a Sentry '
            'evaluation. Computes next_review_at from cooldown rules. '
            'Call ONCE per candidate per tick. '
            'When decision=researched AND verdict in (buy, short): the '
            'discipline audit fields (analogue_considered, '
            'terminal_sensitivity_ran, contradiction_check_passed, '
            'factor_buckets) are REQUIRED. The validator will reject '
            'the insert if any is missing. For watchlist / no_position '
            'verdicts only contradiction_check_passed and factor_buckets '
            'are required (the heavy checks are vacuous for decisions '
            'that do not commit capital).'
          ),
          inputSchema={
            'type': 'object',
            'required': ['ticker', 'decision', 'triggered_by'],
            'properties': {
              'ticker':            {'type': 'string'},
              'decision':          {'type': 'string',
                                    'description': 'researched / skipped_cooldown / skipped_recent_verdict / acted / skipped_budget / skipped_misclassified'},
              'triggered_by':      {'type': 'string',
                                    'description': 'event_score / theme_flow / insider_cluster / catalyst_calendar / manual'},
              'trigger_event_id':  {'type': 'string'},
              'verdict':           {'type': 'string',
                                    'description': 'buy / short / watchlist / avoid / no_position'},
              'confidence':        {'type': 'number'},
              'sizing':            {'type': 'string',
                                    'description': 'aggressive / normal / cautious / no_position'},
              'factor_buckets':    {'type': 'array',
                                    'items': {'type': 'string'},
                                    'description': 'Factor tags from factor-exposure-check. Required for any researched verdict.'},
              'skip_reason':       {'type': 'string'},
              'notes':             {'type': 'string'},
              'analogue_considered':       {'type': 'string',
                                            'description': 'Name of historical analogue considered, or the literal "none" if checked and none applied. Required for researched buy/short.'},
              'terminal_sensitivity_ran':  {'type': 'boolean',
                                            'description': 'True if calculate_scenario_dcf produced a terminal_sensitivity table. Required for researched buy/short.'},
              'contradiction_check_passed':{'type': 'boolean',
                                            'description': 'Result of the Step 19a red-team sub-agent. Required for any researched verdict.'},
              'provenance_filing_count':   {'type': 'integer',
                                            'description': 'Count of [filing:...] tags in Sections 5/6.'},
              'provenance_press_count':    {'type': 'integer',
                                            'description': 'Count of [press-reported] tags in Sections 5/6.'},
            },
          },
        ),
        Tool(
          name='sentry_recent_evals',
          description=(
            'Last N days of eval rows for a ticker, newest first. '
            'Used to remind Claude of prior verdicts before re-evaluating.'
          ),
          inputSchema={
            'type': 'object',
            'required': ['ticker'],
            'properties': {
              'ticker': {'type': 'string'},
              'days':   {'type': 'integer', 'default': 30},
            },
          },
        ),
        Tool(
          name='sentry_should_skip',
          description=(
            'Combined cooldown + falsifier-bypass gate. Returns '
            '{skip: bool, reason: str|null}. Call before each '
            'candidate to decide whether to spend the research budget.'
          ),
          inputSchema={
            'type': 'object',
            'required': ['ticker'],
            'properties': {
              'ticker':   {'type': 'string'},
              'event_id': {'type': 'string',
                           'description': 'Optional source event id; used for falsifier bypass.'},
            },
          },
        ),

        # ----- Budget -----------------------------------------------------
        Tool(
          name='sentry_budget_summary',
          description=(
            "Human-readable summary of today's budget consumption: research "
            'runs, slack messages, paper orders, new positions. Use to log '
            'tick state and to decide whether further actions are permitted.'
          ),
          inputSchema={
            'type': 'object',
            'properties': {
              'day': {'type': 'string',
                      'description': 'Optional YYYY-MM-DD; defaults to today ET.'},
            },
          },
        ),
        Tool(
          name='sentry_can_act',
          description=(
            'Permit gate for an action. Returns {permit: bool, '
            'reason: str|null}. action_type must be one of: research, '
            'slack, new_position, add_to_existing. Call before each '
            'budget-spending operation.'
          ),
          inputSchema={
            'type': 'object',
            'required': ['action_type'],
            'properties': {
              'action_type': {'type': 'string',
                              'enum': ['research', 'slack', 'new_position', 'add_to_existing']},
            },
          },
        ),
        Tool(
          name='sentry_record_action',
          description=(
            'Increment a daily budget counter after the action completes. '
            'action_type maps to the column: research_runs, slack_messages, '
            'paper_orders, new_positions, adds_or_trims. Order placements '
            'should record BOTH paper_orders and either new_positions or '
            'adds_or_trims (two calls).'
          ),
          inputSchema={
            'type': 'object',
            'required': ['action_type'],
            'properties': {
              'action_type': {'type': 'string',
                              'enum': ['research_runs', 'slack_messages',
                                       'paper_orders', 'new_positions',
                                       'adds_or_trims']},
            },
          },
        ),

        # ----- Portfolio state read --------------------------------------
        Tool(
          name='sentry_get_watchlist',
          description='List of watched ticker symbols at or above min_priority.',
          inputSchema={
            'type': 'object',
            'properties': {
              'min_priority': {'type': 'integer', 'default': 1},
            },
          },
        ),
        Tool(
          name='sentry_get_open_positions',
          description=(
            'All currently open positions, optionally filtered by paper. '
            'Defaults to paper=true (the only mode this system supports).'
          ),
          inputSchema={
            'type': 'object',
            'properties': {
              'paper': {'type': 'boolean', 'default': True},
            },
          },
        ),
        Tool(
          name='sentry_get_active_theses',
          description=(
            'Active (non-superseded) theses across the watchlist. Parses '
            'JSON columns key_assumptions / data_gaps / falsifiers.'
          ),
          inputSchema={
            'type': 'object',
            'properties': {
              'limit': {'type': 'integer', 'default': 100},
            },
          },
        ),
        Tool(
          name='sentry_latest_thesis',
          description=(
            'Most recent active thesis for a ticker, or null. Use to '
            'check current conviction before re-evaluating.'
          ),
          inputSchema={
            'type': 'object',
            'required': ['ticker'],
            'properties': {
              'ticker': {'type': 'string'},
            },
          },
        ),

        # ----- Discovery admin -------------------------------------------
        Tool(
          name='sentry_get_discovery_status',
          description=(
            "Today's sentry_discovery_runs row with per-channel "
            'enqueue counts. Use to confirm discovery has fired for '
            'the current ET day, and to inspect channel-level activity.'
          ),
          inputSchema={
            'type': 'object',
            'properties': {
              'day': {'type': 'string',
                      'description': 'Optional YYYY-MM-DD; defaults to today ET.'},
            },
          },
        ),

        # ----- Falsifier alerts ------------------------------------------
        Tool(
          name='sentry_active_falsifier_alerts',
          description=(
            'Recent falsifier_alerts rows (last 7 days), newest first. '
            'Joins against theses so each alert carries the ticker. '
            'Returns [] gracefully if the table does not exist yet.'
          ),
          inputSchema={
            'type': 'object',
            'properties': {
              'limit': {'type': 'integer', 'default': 20},
            },
          },
        ),
      ]

    @self.server.call_tool()
    async def call_tool(name: str, args: Dict[str, Any]):
      try:
        if name == 'sentry_get_queue':
          return parent.tool_get_queue(args)
        if name == 'sentry_mark_queue_status':
          return parent.tool_mark_queue_status(args)
        if name == 'sentry_reset_stuck_queue':
          return parent.tool_reset_stuck_queue(args)
        if name == 'sentry_trim_queue':
          return parent.tool_trim_queue(args)
        if name == 'sentry_get_event':
          return parent.tool_get_event(args)
        if name == 'sentry_validate_candidate':
          return parent.tool_validate_candidate(args)
        if name == 'sentry_recent_events_for_ticker':
          return parent.tool_recent_events_for_ticker(args)
        if name == 'sentry_record_evaluation':
          return parent.tool_record_evaluation(args)
        if name == 'sentry_recent_evals':
          return parent.tool_recent_evals(args)
        if name == 'sentry_should_skip':
          return parent.tool_should_skip(args)
        if name == 'sentry_budget_summary':
          return parent.tool_budget_summary(args)
        if name == 'sentry_can_act':
          return parent.tool_can_act(args)
        if name == 'sentry_record_action':
          return parent.tool_record_action(args)
        if name == 'sentry_get_watchlist':
          return parent.tool_get_watchlist(args)
        if name == 'sentry_get_open_positions':
          return parent.tool_get_open_positions(args)
        if name == 'sentry_get_active_theses':
          return parent.tool_get_active_theses(args)
        if name == 'sentry_latest_thesis':
          return parent.tool_latest_thesis(args)
        if name == 'sentry_get_discovery_status':
          return parent.tool_get_discovery_status(args)
        if name == 'sentry_active_falsifier_alerts':
          return parent.tool_active_falsifier_alerts(args)
        return _err(name, f'unknown tool: {name}')
      except Exception as exc:
        return _err(name, f'{type(exc).__name__}: {str(exc)[:300]}')

  # ----- Tool implementations -------------------------------------------

  def tool_get_queue(self, args):
    limit = int(args.get('limit', 10))
    include_event = bool(args.get('include_event_detail', True))
    include_evals = bool(args.get('include_recent_evals', True))

    rows = sentry_queue.dequeue_top(n=limit)

    if include_event or include_evals:
      conn = get_connection()
      try:
        for r in rows:
          if include_event and r.get('source_event_id'):
            ev = conn.execute(
              """SELECT event_id, source, ticker, primary_ticker,
                        affected_tickers, headline, body, category,
                        materiality, directional_signal, urgency,
                        published_at, ingested_at, url, classifier_reason
                 FROM events WHERE event_id = ?""",
              (r['source_event_id'],),
            ).fetchone()
            if ev:
              ev_d = dict(ev)
              ev_d['affected_tickers'] = _parse_affected(ev_d.get('affected_tickers'))
              r['event'] = ev_d
            else:
              r['event'] = None
          if include_evals and r.get('ticker'):
            r['recent_evals'] = sentry_eval_log.recent_evals_for_ticker(
              r['ticker'], days=30,
            )
      finally:
        conn.close()

    return _ok('sentry_get_queue', {'candidates': rows, 'count': len(rows)})

  def tool_mark_queue_status(self, args):
    queue_id = int(args['queue_id'])
    status = args['status']
    notes_append = args.get('notes_append')
    sentry_queue.mark_status(queue_id, status, notes_append=notes_append)
    row = sentry_queue.get_by_id(queue_id)
    return _ok('sentry_mark_queue_status', {'updated': row})

  def tool_reset_stuck_queue(self, args):
    stuck_minutes = int(args.get('stuck_minutes', 15))
    cutoff = (datetime.now(timezone.utc).timestamp() - stuck_minutes * 60)
    cutoff_iso = datetime.fromtimestamp(cutoff, tz=timezone.utc).isoformat()
    conn = get_connection()
    try:
      cur = conn.execute(
        """UPDATE sentry_queue SET status = 'pending', processed_at = NULL
           WHERE status = 'processing'
             AND COALESCE(processed_at, queued_at) < ?""",
        (cutoff_iso,),
      )
      conn.commit()
      return _ok('sentry_reset_stuck_queue', {'reset_count': cur.rowcount,
                                              'cutoff': cutoff_iso})
    finally:
      conn.close()

  def tool_trim_queue(self, args):
    max_pending = int(args.get('max_pending', 20))
    dropped = sentry_queue.trim_to_cap(max_pending)
    return _ok('sentry_trim_queue', {'dropped': dropped, 'max_pending': max_pending})

  def tool_get_event(self, args):
    event_id = args.get('event_id')
    if not event_id:
      return _err('sentry_get_event', 'event_id is required')
    conn = get_connection()
    try:
      row = conn.execute('SELECT * FROM events WHERE event_id = ?',
                         (event_id,)).fetchone()
      if not row:
        return _ok('sentry_get_event', None)
      d = dict(row)
      d['affected_tickers'] = _parse_affected(d.get('affected_tickers'))
      return _ok('sentry_get_event', d)
    finally:
      conn.close()

  def tool_validate_candidate(self, args):
    queue_id = int(args['queue_id'])
    conn = get_connection()
    try:
      result = validate_candidate(conn, queue_id)
      return _ok('sentry_validate_candidate', result)
    finally:
      conn.close()

  def tool_recent_events_for_ticker(self, args):
    ticker = args['ticker']
    hours = int(args.get('hours', 24))
    rows = events_store.recent_events_for_ticker(ticker, hours=hours)
    for r in rows:
      r['affected_tickers'] = _parse_affected(r.get('affected_tickers'))
    return _ok('sentry_recent_events_for_ticker',
               {'events': rows, 'count': len(rows)})

  def tool_record_evaluation(self, args):
    ticker = args['ticker']
    decision = args['decision']
    triggered_by = args['triggered_by']
    eval_id = sentry_eval_log.record_eval(
      ticker, decision,
      triggered_by=triggered_by,
      trigger_event_id=args.get('trigger_event_id'),
      verdict=args.get('verdict'),
      confidence=args.get('confidence'),
      sizing=args.get('sizing'),
      factor_buckets=args.get('factor_buckets'),
      skip_reason=args.get('skip_reason'),
      notes=args.get('notes'),
      analogue_considered=args.get('analogue_considered'),
      terminal_sensitivity_ran=args.get('terminal_sensitivity_ran'),
      contradiction_check_passed=args.get('contradiction_check_passed'),
      provenance_filing_count=args.get('provenance_filing_count'),
      provenance_press_count=args.get('provenance_press_count'),
    )
    return _ok('sentry_record_evaluation',
               {'eval_id': eval_id, 'ticker': ticker.upper(),
                'decision': decision})

  def tool_recent_evals(self, args):
    ticker = args['ticker']
    days = int(args.get('days', 30))
    rows = sentry_eval_log.recent_evals_for_ticker(ticker, days=days)
    return _ok('sentry_recent_evals', {'evals': rows, 'count': len(rows)})

  def tool_should_skip(self, args):
    ticker = args['ticker']
    event_id = args.get('event_id')
    event = None
    if event_id:
      conn = get_connection()
      try:
        row = conn.execute('SELECT * FROM events WHERE event_id = ?',
                           (event_id,)).fetchone()
        if row:
          event = dict(row)
          event['affected_tickers'] = _parse_affected(event.get('affected_tickers'))
      finally:
        conn.close()
    skip, reason = sentry_eval_log.should_skip(ticker, event=event)
    return _ok('sentry_should_skip', {'skip': skip, 'reason': reason,
                                       'ticker': ticker.upper()})

  def tool_budget_summary(self, args):
    day = args.get('day')
    return _ok('sentry_budget_summary', sentry_budget.summary(day=day))

  def tool_can_act(self, args):
    action = args['action_type']
    if action == 'research':
      permit, reason = sentry_budget.can_research()
    elif action == 'slack':
      permit, reason = sentry_budget.can_post_slack()
    elif action == 'new_position':
      permit, reason = sentry_budget.can_open_new_position()
    elif action == 'add_to_existing':
      permit, reason = sentry_budget.can_add_to_existing()
    else:
      return _err('sentry_can_act', f'unknown action_type: {action}')
    return _ok('sentry_can_act', {'permit': permit, 'reason': reason,
                                   'action_type': action})

  def tool_record_action(self, args):
    action = args['action_type']
    sentry_budget.record_action(action)
    return _ok('sentry_record_action', {'recorded': action})

  def tool_get_watchlist(self, args):
    min_priority = int(args.get('min_priority', 1))
    tickers = get_watchlist(min_priority=min_priority)
    return _ok('sentry_get_watchlist', {'tickers': tickers, 'count': len(tickers)})

  def tool_get_open_positions(self, args):
    paper = args.get('paper', True)
    rows = positions.open_positions(paper=paper)
    return _ok('sentry_get_open_positions', {'positions': rows, 'count': len(rows)})

  def tool_get_active_theses(self, args):
    limit = int(args.get('limit', 100))
    rows = theses.active_theses(limit=limit)
    return _ok('sentry_get_active_theses', {'theses': rows, 'count': len(rows)})

  def tool_latest_thesis(self, args):
    ticker = args['ticker']
    row = theses.latest_thesis(ticker)
    return _ok('sentry_latest_thesis', row)

  def tool_get_discovery_status(self, args):
    day = args.get('day')
    if not day:
      from datetime import timedelta as _td
      day = (datetime.now(timezone.utc) + _td(hours=-5)).strftime('%Y-%m-%d')
    conn = get_connection()
    try:
      row = conn.execute(
        'SELECT * FROM sentry_discovery_runs WHERE day = ?',
        (day,),
      ).fetchone()
      return _ok('sentry_get_discovery_status',
                 dict(row) if row else None)
    finally:
      conn.close()

  def tool_active_falsifier_alerts(self, args):
    limit = int(args.get('limit', 20))
    from datetime import timedelta as _td
    cutoff = (datetime.now(timezone.utc) - _td(days=7)).isoformat()
    conn = get_connection()
    try:
      try:
        rows = conn.execute(
          """SELECT a.*, t.ticker
             FROM falsifier_alerts a
             LEFT JOIN theses t ON t.thesis_id = a.thesis_id
             WHERE a.fired_at >= ?
             ORDER BY a.fired_at DESC LIMIT ?""",
          (cutoff, limit),
        ).fetchall()
        return _ok('sentry_active_falsifier_alerts',
                   {'alerts': [dict(r) for r in rows], 'count': len(rows)})
      except sqlite3.OperationalError:
        # Table not yet created by falsifier_watcher; return empty gracefully
        return _ok('sentry_active_falsifier_alerts',
                   {'alerts': [], 'count': 0,
                    'note': 'falsifier_alerts table does not exist yet'})
    finally:
      conn.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main():
  srv = SentryServer()
  async with stdio_server() as (read, write):
    await srv.server.run(read, write,
                         srv.server.create_initialization_options())


if __name__ == '__main__':
  asyncio.run(main())
