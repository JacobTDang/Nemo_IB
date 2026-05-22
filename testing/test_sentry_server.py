"""Unit + smoke tests for tools/sentry_server/server.py.

Three layers:
  1. Schema: list_tools registers 19 tools with valid shapes
  2. Per-tool behavior: each tool against a controlled DB fixture
  3. validate_candidate: explicit 4-case coverage of the structural
     pre-flight check (proceed / misclassified / stale / repeat)

Tests call the server's tool methods directly (not through MCP stdio)
so they're fast and deterministic.

Run:
  .venv\\Scripts\\python.exe testing\\test_sentry_server.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from state.schema import get_connection, init_schema
from state import sentry_queue, sentry_eval_log, events_store
from tools.sentry_server.server import SentryServer
from tools.sentry_server._validation import validate_candidate, STALE_AGE_DAYS


_results = {'pass': 0, 'fail': 0, 'failures': []}


def _check(name: str, condition: bool, hint: str = '') -> None:
  if condition:
    _results['pass'] += 1
    print(f"  PASS  {name}")
  else:
    _results['fail'] += 1
    _results['failures'].append((name, hint))
    print(f"  FAIL  {name}  --  {hint}")


_TICKER_PREFIX = 'SRVTST_'
_EVENT_PREFIX = 'SRVTSTEVT_'


def _cleanup():
  conn = get_connection()
  try:
    conn.execute("DELETE FROM events WHERE event_id LIKE ?",
                 (f"{_EVENT_PREFIX}%",))
    conn.execute("DELETE FROM sentry_queue WHERE ticker LIKE ?",
                 (f"{_TICKER_PREFIX}%",))
    conn.execute("DELETE FROM sentry_evaluation_log WHERE ticker LIKE ?",
                 (f"{_TICKER_PREFIX}%",))
    conn.commit()
  finally:
    conn.close()


def _inject_event(event_id, ticker, affected_tickers=None, primary_ticker=None,
                  published_at=None, ingested_at=None, headline=None,
                  materiality='high', category='corporate_event'):
  """Inject an event row with explicit event_id (bypasses the hash dedup)."""
  if published_at is None:
    published_at = datetime.now(timezone.utc).isoformat()
  if ingested_at is None:
    ingested_at = published_at
  if headline is None:
    headline = f"{ticker} test headline"
  conn = get_connection()
  try:
    conn.execute(
      """INSERT INTO events
         (event_id, source, ticker, headline, body, url, published_at, ingested_at,
          materiality, category, affected_tickers, primary_ticker,
          directional_signal, urgency, classifier_reason, processed)
         VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,0)""",
      (event_id, 'test', ticker, headline, 'body', 'http://x',
       published_at, ingested_at, materiality, category,
       json.dumps(affected_tickers or []), primary_ticker,
       'neutral', 'hours', 'test'),
    )
    conn.commit()
  finally:
    conn.close()


def _read_envelope(text_content_list):
  """Tool methods return List[TextContent]; pull out the JSON dict."""
  return json.loads(text_content_list[0].text)


# ===========================================================================
# 1. Schema layer
# ===========================================================================

async def _list_tools(srv):
  from mcp.types import ListToolsRequest
  handler = srv.server.request_handlers[ListToolsRequest]
  req = ListToolsRequest(method='tools/list', params=None)
  resp = await handler(req)
  return resp.root.tools


def test_lists_19_tools():
  print("\n== schema: list_tools returns 19 tools with valid shapes ==")
  srv = SentryServer()
  tools = asyncio.run(_list_tools(srv))
  _check("19 tools registered", len(tools) == 19, f"got {len(tools)}")
  for t in tools:
    _check(f"{t.name}: has description (>20 chars)",
           bool(t.description) and len(t.description) > 20,
           f"len={len(t.description or '')}")
    _check(f"{t.name}: has inputSchema with properties",
           isinstance(t.inputSchema, dict)
           and isinstance(t.inputSchema.get('properties'), dict),
           str(t.inputSchema)[:100])


# ===========================================================================
# 2. validate_candidate (4 cases)
# ===========================================================================

def test_validate_proceed_when_primary_matches():
  print("\n== validate: primary_ticker set, queue.ticker matches -> ok ==")
  _cleanup()
  eid = f'{_EVENT_PREFIX}1'
  ticker = f'{_TICKER_PREFIX}A'
  _inject_event(eid, ticker,
                affected_tickers=[ticker], primary_ticker=ticker)
  qid = sentry_queue.enqueue(ticker, 0.7, triggered_by='event_score',
                             source_event_id=eid)
  conn = get_connection()
  try:
    result = validate_candidate(conn, qid)
  finally:
    conn.close()
  _check("ok = True", result['ok'] is True, result.get('reason'))
  _check("suggested_action = proceed",
         result['suggested_action'] == 'proceed',
         result.get('suggested_action'))


def test_validate_proceed_when_in_affected():
  print("\n== validate: primary null, queue.ticker in affected -> ok ==")
  _cleanup()
  eid = f'{_EVENT_PREFIX}2'
  ticker = f'{_TICKER_PREFIX}B'
  _inject_event(eid, 'NOISE', affected_tickers=[ticker, 'OTHER'],
                primary_ticker=None)
  qid = sentry_queue.enqueue(ticker, 0.7, triggered_by='event_score',
                             source_event_id=eid)
  conn = get_connection()
  try:
    result = validate_candidate(conn, qid)
  finally:
    conn.close()
  _check("ok = True (ticker corroborated by affected)",
         result['ok'] is True, result.get('reason'))


def test_validate_reject_misclassified():
  print("\n== validate: primary null, queue.ticker NOT in affected -> reject ==")
  _cleanup()
  eid = f'{_EVENT_PREFIX}3'
  ticker = f'{_TICKER_PREFIX}C'
  # The AAPL-in-ECB-payments-story shape: queue.ticker is something the
  # classifier did not flag as affected.
  _inject_event(eid, ticker,
                affected_tickers=['V', 'MA'], primary_ticker=None)
  qid = sentry_queue.enqueue(ticker, 0.7, triggered_by='event_score',
                             source_event_id=eid)
  conn = get_connection()
  try:
    result = validate_candidate(conn, qid)
  finally:
    conn.close()
  _check("ok = False", result['ok'] is False,
         f"unexpectedly ok={result['ok']}")
  _check("suggested_action = drop_misclassified",
         result['suggested_action'] == 'drop_misclassified',
         result.get('suggested_action'))
  _check("reason mentions affected_tickers",
         'affected_tickers' in (result.get('reason') or ''),
         result.get('reason'))


def test_validate_reject_stale():
  print("\n== validate: event older than STALE_AGE_DAYS -> reject ==")
  _cleanup()
  eid = f'{_EVENT_PREFIX}4'
  ticker = f'{_TICKER_PREFIX}D'
  stale = (datetime.now(timezone.utc) - timedelta(days=STALE_AGE_DAYS + 2)).isoformat()
  _inject_event(eid, ticker, affected_tickers=[ticker],
                primary_ticker=ticker,
                published_at=stale, ingested_at=stale)
  qid = sentry_queue.enqueue(ticker, 0.7, triggered_by='event_score',
                             source_event_id=eid)
  conn = get_connection()
  try:
    result = validate_candidate(conn, qid)
  finally:
    conn.close()
  _check("ok = False", result['ok'] is False, result.get('reason'))
  _check("suggested_action = drop_stale",
         result['suggested_action'] == 'drop_stale',
         result.get('suggested_action'))


def test_validate_reject_repeat_misclassify():
  print("\n== validate: previously rejected (ticker,event) pair -> reject ==")
  _cleanup()
  eid = f'{_EVENT_PREFIX}5'
  ticker = f'{_TICKER_PREFIX}E'
  _inject_event(eid, ticker, affected_tickers=[ticker], primary_ticker=ticker)
  # Pre-record a skipped_misclassified eval for this (ticker, event_id)
  sentry_eval_log.record_eval(
    ticker, 'skipped_misclassified',
    triggered_by='event_score', trigger_event_id=eid,
    skip_reason='manual reject for test',
  )
  qid = sentry_queue.enqueue(ticker, 0.7, triggered_by='event_score',
                             source_event_id=eid)
  conn = get_connection()
  try:
    result = validate_candidate(conn, qid)
  finally:
    conn.close()
  _check("ok = False (repeat reject)", result['ok'] is False)
  _check("suggested_action = drop_repeat_misclassify",
         result['suggested_action'] == 'drop_repeat_misclassify',
         result.get('suggested_action'))


# ===========================================================================
# 3. Per-tool behavior smoke (direct method calls)
# ===========================================================================

def test_tool_get_queue_returns_envelope():
  print("\n== tool: sentry_get_queue returns envelope with candidates ==")
  _cleanup()
  srv = SentryServer()
  eid = f'{_EVENT_PREFIX}q1'
  ticker = f'{_TICKER_PREFIX}Q1'
  _inject_event(eid, ticker, affected_tickers=[ticker], primary_ticker=ticker)
  sentry_queue.enqueue(ticker, 0.8, triggered_by='event_score',
                       source_event_id=eid)
  env = _read_envelope(srv.tool_get_queue({'limit': 10}))
  _check("success", env['success'] is True, str(env.get('metadata', {})))
  candidates = env['data']['candidates']
  _check("at least 1 candidate",
         any(c['ticker'] == ticker for c in candidates),
         f"got tickers: {[c['ticker'] for c in candidates]}")
  for c in candidates:
    if c['ticker'] == ticker:
      _check("event detail folded in",
             c.get('event') is not None
             and c['event']['event_id'] == eid)
      _check("recent_evals folded in (may be empty list)",
             'recent_evals' in c)


def test_tool_get_event_finds_row():
  print("\n== tool: sentry_get_event returns the parsed event row ==")
  _cleanup()
  srv = SentryServer()
  eid = f'{_EVENT_PREFIX}ev1'
  ticker = f'{_TICKER_PREFIX}EV1'
  _inject_event(eid, ticker, affected_tickers=[ticker, 'OTHER'])
  env = _read_envelope(srv.tool_get_event({'event_id': eid}))
  _check("success", env['success'] is True)
  _check("returns event with parsed affected_tickers",
         isinstance(env['data'].get('affected_tickers'), list))
  _check("event_id matches", env['data'].get('event_id') == eid)


def test_tool_get_event_missing_returns_null():
  print("\n== tool: sentry_get_event for unknown id returns data=null ==")
  srv = SentryServer()
  env = _read_envelope(srv.tool_get_event({'event_id': 'doesnotexist'}))
  _check("success and data=null", env['success'] is True
         and env['data'] is None)


def test_tool_validate_candidate_via_server():
  print("\n== tool: sentry_validate_candidate end-to-end ==")
  _cleanup()
  srv = SentryServer()
  eid = f'{_EVENT_PREFIX}val1'
  ticker = f'{_TICKER_PREFIX}V1'
  _inject_event(eid, ticker,
                affected_tickers=['V', 'MA'], primary_ticker=None)
  qid = sentry_queue.enqueue(ticker, 0.7, triggered_by='event_score',
                             source_event_id=eid)
  env = _read_envelope(srv.tool_validate_candidate({'queue_id': qid}))
  _check("envelope success=True", env['success'] is True,
         str(env.get('metadata')))
  _check("ok=False (misclassified)", env['data']['ok'] is False)
  _check("suggested_action=drop_misclassified",
         env['data']['suggested_action'] == 'drop_misclassified')


def test_tool_mark_queue_status_round_trip():
  print("\n== tool: sentry_mark_queue_status flips state ==")
  _cleanup()
  srv = SentryServer()
  ticker = f'{_TICKER_PREFIX}MS'
  qid = sentry_queue.enqueue(ticker, 0.5, triggered_by='event_score')
  env = _read_envelope(srv.tool_mark_queue_status(
    {'queue_id': qid, 'status': 'processing'}))
  _check("success", env['success'] is True)
  _check("status flipped to processing",
         env['data']['updated']['status'] == 'processing')


def test_tool_recent_evals_returns_list():
  print("\n== tool: sentry_recent_evals returns chronological list ==")
  _cleanup()
  srv = SentryServer()
  ticker = f'{_TICKER_PREFIX}RE'
  sentry_eval_log.record_eval(
    ticker, 'researched',
    triggered_by='event_score',
    verdict='watchlist', confidence=0.6, sizing='cautious',
  )
  env = _read_envelope(srv.tool_recent_evals({'ticker': ticker, 'days': 1}))
  _check("success", env['success'] is True)
  _check("at least 1 eval row", env['data']['count'] >= 1,
         str(env['data']))


def test_tool_should_skip_after_cooldown_recorded():
  print("\n== tool: sentry_should_skip honors cooldown ==")
  _cleanup()
  srv = SentryServer()
  ticker = f'{_TICKER_PREFIX}SS'
  # Record an 'acted' eval -> 7-day cooldown
  sentry_eval_log.record_eval(
    ticker, 'acted', triggered_by='event_score',
    verdict='buy', confidence=0.7, sizing='cautious',
  )
  env = _read_envelope(srv.tool_should_skip({'ticker': ticker}))
  _check("success", env['success'] is True)
  _check("skip = True (in cooldown)", env['data']['skip'] is True,
         env['data'].get('reason'))


def test_tool_budget_summary_shape():
  print("\n== tool: sentry_budget_summary returns the daily counts ==")
  srv = SentryServer()
  env = _read_envelope(srv.tool_budget_summary({}))
  _check("success", env['success'] is True)
  expected_keys = {'day', 'research_runs', 'slack_messages',
                   'paper_orders', 'new_positions'}
  _check("envelope has expected keys",
         expected_keys.issubset(set(env['data'].keys())),
         str(env['data'].keys()))


def test_tool_can_act_known_action():
  print("\n== tool: sentry_can_act returns permit + reason ==")
  srv = SentryServer()
  env = _read_envelope(srv.tool_can_act({'action_type': 'research'}))
  _check("success", env['success'] is True)
  _check("permit is bool", isinstance(env['data'].get('permit'), bool))


def test_tool_can_act_unknown_action_errors():
  print("\n== tool: sentry_can_act unknown action returns error envelope ==")
  srv = SentryServer()
  env = _read_envelope(srv.tool_can_act({'action_type': 'nonsense'}))
  _check("success = False", env['success'] is False)


def test_tool_get_watchlist():
  print("\n== tool: sentry_get_watchlist returns ticker list ==")
  srv = SentryServer()
  env = _read_envelope(srv.tool_get_watchlist({}))
  _check("success", env['success'] is True)
  _check("tickers is list", isinstance(env['data'].get('tickers'), list))


def test_tool_get_active_theses():
  print("\n== tool: sentry_get_active_theses returns envelope ==")
  srv = SentryServer()
  env = _read_envelope(srv.tool_get_active_theses({'limit': 5}))
  _check("success", env['success'] is True)
  _check("theses is list", isinstance(env['data'].get('theses'), list))


def test_tool_get_discovery_status():
  print("\n== tool: sentry_get_discovery_status returns dict or null ==")
  srv = SentryServer()
  env = _read_envelope(srv.tool_get_discovery_status({}))
  _check("success", env['success'] is True)
  # data may be None if discovery hasn't run today; either way envelope OK
  _check("data is dict or None",
         env['data'] is None or isinstance(env['data'], dict),
         str(env['data'])[:80])


def test_tool_active_falsifier_alerts_graceful():
  print("\n== tool: sentry_active_falsifier_alerts handles missing table ==")
  srv = SentryServer()
  env = _read_envelope(srv.tool_active_falsifier_alerts({'limit': 5}))
  _check("success (even if table missing)", env['success'] is True)
  _check("alerts is list", isinstance(env['data'].get('alerts'), list))


def test_tool_reset_stuck_queue():
  print("\n== tool: sentry_reset_stuck_queue counts cutoff ==")
  _cleanup()
  srv = SentryServer()
  ticker = f'{_TICKER_PREFIX}STUCK'
  qid = sentry_queue.enqueue(ticker, 0.5, triggered_by='event_score')
  # Move it to processing with an old processed_at
  conn = get_connection()
  try:
    old = (datetime.now(timezone.utc) - timedelta(minutes=60)).isoformat()
    conn.execute(
      "UPDATE sentry_queue SET status='processing', processed_at=? WHERE queue_id=?",
      (old, qid),
    )
    conn.commit()
  finally:
    conn.close()
  env = _read_envelope(srv.tool_reset_stuck_queue({'stuck_minutes': 15}))
  _check("success", env['success'] is True)
  _check("reset_count >= 1",
         env['data']['reset_count'] >= 1, str(env['data']))


# ===========================================================================
# Runner
# ===========================================================================

def main() -> int:
  print("\nnemo_sentry MCP server tests\n")
  init_schema()
  # Schema
  test_lists_19_tools()
  # validate_candidate (4 cases + 1 bonus)
  test_validate_proceed_when_primary_matches()
  test_validate_proceed_when_in_affected()
  test_validate_reject_misclassified()
  test_validate_reject_stale()
  test_validate_reject_repeat_misclassify()
  # Per-tool behavior
  test_tool_get_queue_returns_envelope()
  test_tool_get_event_finds_row()
  test_tool_get_event_missing_returns_null()
  test_tool_validate_candidate_via_server()
  test_tool_mark_queue_status_round_trip()
  test_tool_recent_evals_returns_list()
  test_tool_should_skip_after_cooldown_recorded()
  test_tool_budget_summary_shape()
  test_tool_can_act_known_action()
  test_tool_can_act_unknown_action_errors()
  test_tool_get_watchlist()
  test_tool_get_active_theses()
  test_tool_get_discovery_status()
  test_tool_active_falsifier_alerts_graceful()
  test_tool_reset_stuck_queue()
  _cleanup()

  print(f"\n== Summary ==\n  PASS: {_results['pass']}\n  FAIL: {_results['fail']}")
  for n, h in _results['failures']:
    print(f"  - {n}: {h}")
  return 0 if _results['fail'] == 0 else 1


if __name__ == "__main__":
  sys.exit(main())
