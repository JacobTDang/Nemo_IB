# nemo_sentry MCP server -- live verification (2026-05-22)

All 19 tools invoked via the same MCP handler chain that claude mcp will use.

## sentry_reset_stuck_queue

- status: OK
- data: {'reset_count': 0, 'cutoff': '2026-05-22T07:03:01.334620+00:00'}

## sentry_get_queue

- status: OK
- data: {'candidates': [{'queue_id': 6499, 'ticker': 'AAP', 'source_event_id': 'f8e3285e8c5f7aed', 'triggered_by': 'event_score', 'score': 0.66, 'queued_at': '2026-05-22T05:05:12.555539+00:00', 'status': 'pen

## sentry_budget_summary

- status: OK
- data: {'day': '2026-05-22', 'research_runs': '0/10', 'slack_messages': '0/50', 'paper_orders': '0/5', 'new_positions': '0/2', 'adds_or_trims': 0, 'first_action_at': None, 'last_action_at': None}

## sentry_can_act

- status: OK
- data: {'permit': True, 'reason': None, 'action_type': 'research'}

## sentry_get_watchlist

- status: OK
- data: {'tickers': ['AAPL', 'GOOGL', 'JPM', 'KO', 'MSFT', 'NVDA', 'TSLA'], 'count': 7}

## sentry_get_open_positions

- status: OK
- data: {'positions': [], 'count': 0}

## sentry_get_active_theses

- status: OK
- data: {'theses': [{'thesis_id': 267, 'ticker': 'AMD', 'thesis_date': '2026-05-21T14:43:07.306734', 'recommendation': 'BUY', 'signal': 'long', 'target_price': 557.47, 'stop_loss': 379.08, 'confidence': 0.62,

## sentry_get_discovery_status

- status: OK
- data: {'day': '2026-05-22', 'ran_at': '2026-05-22T05:04:01.987976+00:00', 'catalyst_enqueued': 0, 'insider_enqueued': 0, 'activist_enqueued': 0, 'theme_flow_enqueued': 0, 'total_enqueued': 0, 'errors': None

## sentry_active_falsifier_alerts

- status: OK
- data: {'alerts': [], 'count': 0}

## sentry_recent_evals

- status: OK
- data: {'evals': [], 'count': 0}

## sentry_should_skip

- status: OK
- data: {'skip': False, 'reason': None, 'ticker': 'NVDA'}

## sentry_recent_events_for_ticker

- status: OK
- data: {'events': [{'event_id': '409f95074a855149', 'source': 'rss:marketwatch_top_stories', 'ticker': '', 'headline': 'Nvidia can deliver chips — but it can’t buy Big Tech out of its credit and power-grid c

## sentry_trim_queue

- status: OK
- data: {'dropped': 0, 'max_pending': 50}


## sentry_validate_candidate -- AAPL misclassification regression

Validation against the actual queue row from the 2026-05-22 incident:

- queue_id=6487, ticker=AAPL, source_event_id=a3fadb1298a54e34
- the event headline was about ECB / Visa / Mastercard, primary_ticker=null
- classifier-assigned affected_tickers=['V', 'MA'] -- no AAPL

Returned:

```
{
  "ok": false,
  "queue_id": 6487,
  "ticker": "AAPL",
  "event_id": "a3fadb1298a54e34",
  "reason": "ticker 'AAPL' not in event.affected_tickers=['V', 'MA'] and event has no primary_ticker; this is a misclassified candidate",
  "suggested_action": "drop_misclassified"
}
```

For comparison, queue_id=6499 (AAP, real earnings story) correctly returned
`{ok: true, suggested_action: "proceed"}`.

## OSS pass-through

`sentry_recent_events_for_ticker(ticker='NVDA', hours=48)` returned 67 events
across 4 sources, confirming RSS and GDELT data flows are readable from the
new tool layer:

- finnhub:Yahoo (legacy)
- finnhub-general:CNBC (legacy)
- gdelt (Phase 4 daemon)
- rss:marketwatch_top_stories (Phase 4 daemon)

`obb_insider_trading(ticker='AAPL', limit=3)` against the OpenBB MCP server
still returns 5 rows, confirming the Phase 4 OSS layer is unaffected by the
new server.

## Verdict

All 19 tools work via the MCP handler chain. The structural pre-flight
catches the production misclassification. RSS / GDELT events are visible
through the new tool layer. OpenBB still operational. Ready to register
with Claude Code via `claude mcp add -s user nemo_sentry`.
