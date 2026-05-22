---
name: sentry-tick
description: The recurring action skill for the autonomous Sentry loop. Reads top candidates from sentry_queue via the nemo_sentry MCP tools, runs cross-company-readthrough + equity-deep-research, applies all gates (portfolio-fit, factor-exposure, kill-switch, Risk_Officer, size, factor concentration, daily caps), and either places a paper trade or records a watchlist verdict. Posts results to Slack. Invoked every 15 minutes via /loop 15m /sentry-tick.
---

# /sentry-tick

The Claude side of the Sentry loop. Python daemons populate
`sentry_queue`; this skill picks the top 3 candidates per tick,
reasons through the full skill catalog, and either acts or records
why it didn't.

**Everything is an MCP tool call.** The `nemo_sentry` server
exposes all queue / event / eval / budget / portfolio state
operations as tools. Do NOT shell out to `python -c` for these —
that pattern was retired in PR #5 because it kept silently breaking
on path issues and missing functions (e.g.
`ImportError: cannot import name 'get_event_by_id'`).

**Hard rule before anything else**: always honor the budget gate.
If `sentry_can_act(action_type='research')` returns
`{permit: false, ...}`, stop the tick after posting a Slack
summary of why.

## Workflow

### 1. Recover from any crashed prior tick

Call `sentry_reset_stuck_queue(stuck_minutes=15)`. Rows that were
flipped to `processing` more than 15 minutes ago get reset to
`pending` (a crashed prior tick stranded them). Logs how many were
reset.

### 2. Read the top candidates with full context

Call `sentry_get_queue(limit=3, include_event_detail=true,
include_recent_evals=true)`. The response is a list of
candidates; each candidate carries:

- `queue_id`, `ticker`, `score`, `source_event_id`, `notes`,
  `queued_at`, `status`
- `event` — the source event row (headline, body, primary_ticker,
  affected_tickers, materiality, urgency, category)
- `recent_evals` — the last 30 days of eval_log rows for this
  ticker, so prior verdicts are visible

Also call `sentry_budget_summary()` once at the top so the tick log
records starting budget.

### 3. For each candidate (up to 3 per tick), in order

#### 3a. Structural pre-flight — call `sentry_validate_candidate(queue_id)`

If `{ok: false, suggested_action: ...}`:

- Call `sentry_mark_queue_status(queue_id, 'dropped', notes_append=<reason>)`.
- Call `sentry_record_evaluation(ticker=..., decision='skipped_misclassified',
  triggered_by='event_score', trigger_event_id=<event_id>, skip_reason=<reason>)`.
- Continue to the next candidate. **Do NOT spend the research budget on a
  candidate the structural check rejects.**

The validator catches:
- `drop_misclassified` — queue.ticker not in event.affected_tickers
  and event has no primary_ticker (the AAPL-in-ECB-payments case)
- `drop_stale` — event older than 7 days
- `drop_repeat_misclassify` — already rejected this exact
  (ticker, event_id) pair before
- `drop_missing_event` — queue row references a non-existent event
- `drop_no_event_link` — queue row has no source_event_id

#### 3b. Cooldown — call `sentry_should_skip(ticker, event_id=<event_id>)`

If `{skip: true, reason: ...}`:
- `sentry_mark_queue_status(queue_id, 'dropped', notes_append=...)`
- `sentry_record_evaluation(decision='skipped_cooldown', skip_reason=...)`
- Continue to next.

#### 3c. Budget — call `sentry_can_act(action_type='research')`

If `{permit: false, ...}`:
- Reset the queue row back to pending so the next tick can try it:
  `sentry_mark_queue_status(queue_id, 'pending', notes_append='budget cap reached')`
- Stop the tick (don't process remaining candidates this tick).
- Post the closing Slack summary.

#### 3d. Claim the candidate

`sentry_mark_queue_status(queue_id, 'processing')`

#### 3e. Run the research chain

Invoke the existing skills (these are still skill invocations, not
MCP tool calls — they're conversational instructions, not data ops):

- `/cross-company-readthrough` on the source event
- If predicted_implication confidence >= 0.65:
  - `/equity-deep-research` on the affected ticker
  - `/portfolio-fit` (existing book check)
  - `/factor-exposure-check`
  - `/thesis-kill-switch` (pre-entry mode)

Record `sentry_record_action(action_type='research_runs')` once the
research-chain skills have begun (do this after step 3d so the
budget reflects work in progress).

#### 3f. Decide

Inspect the synthesis. Three outcomes:

**Gate failed**
- `/portfolio-fit` returned `reject`, OR
- `/factor-exposure-check` returned `mostly factor` AND variant is
  factor-directional, OR
- `/thesis-kill-switch` reports any falsifier would trigger on day 1

→ `sentry_record_evaluation(decision='skipped_recent_verdict',
verdict=..., skip_reason=...)`
→ Post Slack `trade_rejected` card via `post_notification`
→ `sentry_record_action(action_type='slack_messages')`
→ `sentry_mark_queue_status(queue_id, 'completed', notes_append='gate failed')`

**Verdict watchlist / no_position**
→ `sentry_record_evaluation(decision='researched', verdict=...,
sizing=...)`
→ Post Slack `research_synthesis` card
→ `sentry_record_action(action_type='slack_messages')`
→ `sentry_mark_queue_status(queue_id, 'completed')`

**Verdict buy with sizing >= cautious**
1. Compute the proposed quantity for ≤1% of book.
2. Decide new position vs. add to existing:
   - `sentry_get_open_positions(paper=true)` to check.
   - Call `sentry_can_act(action_type='new_position')` OR
     `sentry_can_act(action_type='add_to_existing')` accordingly.
   - If gate fails: drop to watchlist verdict, record_eval +
     Slack, continue.
3. Call `risk_check_proposed_trade(ticker=..., side='buy',
   qty=...)` (the alpaca MCP tool). The Risk_Officer is
   authoritative — do NOT argue with it. If `approve == false`,
   drop to watchlist verdict, post `trade_rejected` Slack card
   citing the Risk_Officer reasons.
4. If approved: use `adjusted_quantity` if returned, else the
   original. Call `place_paper_order(...)` (alpaca MCP).
5. Record the action twice:
   - `sentry_record_action(action_type='paper_orders')`
   - `sentry_record_action(action_type='new_positions')` OR
     `sentry_record_action(action_type='adds_or_trims')`
6. `sentry_record_evaluation(decision='acted', verdict='buy',
   sizing=..., confidence=..., factor_buckets=[...])`
7. Post Slack `trade_placed` card via `post_notification`.
8. `sentry_record_action(action_type='slack_messages')`
9. `sentry_mark_queue_status(queue_id, 'completed')`

### 4. Closing tick log

Call `sentry_budget_summary()` once more and include it in the
final tick output.

## Hard rules

- Zero `Bash` invocations in this skill body. All state operations
  go through `nemo_sentry` MCP tools. The only `Bash` permitted
  is if a user-requested debug or one-off log fetch is explicitly
  asked for during a tick — which is rare and adversarial.
- Always `sentry_validate_candidate` BEFORE `/cross-company-readthrough`.
  The structural check is cheap; deep research is not. Spending
  budget on a misclassified candidate is a discipline failure.
- Always call `risk_check_proposed_trade` BEFORE `place_paper_order`
  (existing CLAUDE.md rule).
- Honor every cap: `sentry_can_act('research')` before research,
  `sentry_can_act('new_position'|'add_to_existing')` before orders,
  no more than 3 candidates per tick regardless.
- Record every decision via `sentry_record_evaluation` — every
  ticker that comes off the queue gets a row, no exceptions.
  Audit trail relies on this.
- If `/sentry-tick` is interrupted mid-candidate, the
  `sentry_reset_stuck_queue` call at the next tick's step 1 will
  recover the row. Do NOT manually flip queue status outside the
  workflow.

## Tool reference (nemo_sentry)

The nemo_sentry MCP server exposes 19 tools. Common ones used
above:

- `sentry_reset_stuck_queue(stuck_minutes=15)` → reset count
- `sentry_get_queue(limit=10, include_event_detail=true, include_recent_evals=true)`
  → list of candidates with folded event + evals
- `sentry_validate_candidate(queue_id)` → {ok, reason, suggested_action}
- `sentry_should_skip(ticker, event_id=...)` → {skip, reason}
- `sentry_can_act(action_type)` → {permit, reason}
- `sentry_record_action(action_type)` → increments counter
- `sentry_mark_queue_status(queue_id, status, notes_append=...)`
- `sentry_record_evaluation(ticker, decision, triggered_by, ...)`
- `sentry_budget_summary()` → counts dict
- `sentry_get_open_positions(paper=true)`
- `sentry_get_active_theses(limit=100)`
- `sentry_latest_thesis(ticker)`
- `sentry_get_event(event_id)`
- `sentry_recent_evals(ticker, days=30)`
- `sentry_recent_events_for_ticker(ticker, hours=24)`
- `sentry_trim_queue(max_pending=20)`
- `sentry_get_watchlist(min_priority=1)`
- `sentry_get_discovery_status()`
- `sentry_active_falsifier_alerts(limit=20)`

## Slack notification cards

Posted via `post_notification` (slack channel plugin). Four shapes:

1. **high_signal_event** — high-score event flagged even when not
   acted on. Used during step 3a / 3b drops when the score is
   above 0.70 but a gate rejected.
2. **research_synthesis** — completed deep-research with verdict
   `watchlist` or `no_position`. Includes thesis, decision, key
   reason.
3. **trade_placed** — paper order placed. Includes ticker, side,
   qty, entry price, thesis_id, all 4 gates green.
4. **trade_rejected** — trade proposal that didn't clear gates.
   Includes ticker, proposed sizing, which specific gate failed.

Each card is structured Slack blocks (mrkdwn), max 4000 chars,
threaded under a daily root message.

## Done condition

A tick is done when one of:
- 3 candidates processed (acted, watchlisted, or skipped)
- Queue is empty
- Budget cap reached (research / new_position)
- Slack cap reached
- Any candidate triggered an explicit halt condition

Post a closing Slack summary line (or skip if `sentry_can_act('slack')`
is false).
