---
name: sentry-tick
description: The recurring action skill for the autonomous Sentry loop. Reads top candidates from sentry_queue, runs cross-company-readthrough + equity-deep-research, applies all gates (portfolio-fit, factor-exposure, kill-switch, Risk_Officer, size, factor concentration, daily caps), and either places a paper trade or records a watchlist verdict. Posts results to Slack. Invoked every 15 minutes via /loop 15m /sentry-tick.
---

# /sentry-tick

The Claude side of the Sentry loop. Python daemons populate
`sentry_queue` with high-signal candidates; this skill picks the top 3
per tick, reasons through the full skill catalog, and either acts or
records why it didn't.

**Hard rule before anything else**: always honor the budget gate. If
`can_research()` returns false, stop the tick after posting a Slack
summary of why.

## Workflow

### 1. Read the queue

```bash
.venv/Scripts/python.exe -c "
from state.sentry_queue import dequeue_top
from state.sentry_eval_log import recent_evals_for_ticker
from agent.sentry_budget import summary
import json
candidates = dequeue_top(3)
out = {'candidates': candidates, 'budget_summary': summary()}
for c in candidates:
    c['recent_evals'] = recent_evals_for_ticker(c['ticker'], days=30)
print(json.dumps(out, default=str))
"
```

If `candidates` is empty: post a single-line Slack notification
("Sentry tick: queue empty, no candidates"), record nothing, stop.

If `budget_summary` shows research cap hit: post Slack
("Sentry tick: research cap hit, skipping until tomorrow"), stop.

### 2. For each candidate (up to 3 per tick)

Mark the queue row as `processing`:

```bash
.venv/Scripts/python.exe -c "from state.sentry_queue import mark_status; mark_status({queue_id}, 'processing')"
```

#### 2a. Review prior context

Read the `recent_evals` field from step 1. If there are recent evals
for this ticker, summarize them in 1-2 sentences before continuing —
this gives the analyst (you) context on whether the current candidate
deserves another full research pass or if a recent verdict still
stands. Specifically:

- If the most recent eval was `acted` within 7 days AND no falsifier
  has fired since → skip this candidate (record
  `decision='skipped_recent_verdict', skip_reason='already acted'`)
- If the most recent eval was `watchlist` within 14 days AND the
  triggering event isn't materially different → skip
- Otherwise, proceed.

#### 2b. Run cross-company-readthrough

Invoke `/cross-company-readthrough` with the source event from the
queue row. Read its envelope:

- If `verdict == 'inconclusive'` OR `confidence < 0.50` → skip.
  Record `decision='researched', verdict='no_position',
  skip_reason='read-through inconclusive'`. Post Slack
  `research_synthesis` card.
- If confidence ≥ 0.50 → proceed.

#### 2c. Check research budget

Before running deep research, check `can_research()`. If false, stop
the whole tick (we've hit the daily cap). Record the queue row as
`pending` again (will be retried tomorrow).

#### 2d. Run /equity-deep-research

Invoke the full deep-research skill on the candidate ticker. Record
the budget consumption:

```bash
.venv/Scripts/python.exe -c "from agent.sentry_budget import record_action, ACTION_RESEARCH; record_action(ACTION_RESEARCH)"
```

#### 2e. Run Step 18 companion gates

The equity-deep-research synthesis triggers these — but run them
explicitly to capture their envelopes:

- `/portfolio-fit` — if `verdict == 'reject'`, skip
- `/factor-exposure-check` — if `verdict == 'mostly_factor'` AND the
  variant perception is factor-directional, skip
- `/thesis-kill-switch` (pre-entry mode) — if any proposed falsifier
  would trigger on day 1, skip

For any of these skips: record
`decision='researched', verdict='no_position', skip_reason='<which gate>'`
and post `research_synthesis` Slack card.

#### 2f. Decision

Read the equity-deep-research synthesis. The relevant fields are
`sizing`, `confidence`, and `falsifiers`.

| Synthesis says | Action |
|---|---|
| `sizing == 'no_position'` OR `sizing == 'watchlist'` | Record `decision='researched', verdict='watchlist'` or `'no_position'`. Post `research_synthesis` card. |
| `confidence < 0.65` | Same as above (verdict reflects intent but confidence too low to act). |
| `sizing in ('cautious', 'normal', 'aggressive')` AND `confidence >= 0.65` | Proceed to trade gate (step 2g). |

#### 2g. Trade gate — risk_check_proposed_trade

Compute the proposed quantity from sizing + current paper account
value. Sizing tiers (1% size cap is the hard ceiling):

- `cautious` → 0.5% of book
- `normal` → 1.0% of book (the cap)
- `aggressive` → also 1.0% (Sentry never exceeds 1% per position, even
  when synthesis says aggressive — discipline)

Call `mcp__nemo_alpaca__risk_check_proposed_trade(ticker, side='buy',
quantity=N)`. If NOT approved → record `decision='researched', verdict
matches synthesis, skip_reason='risk_officer rejected: <reason>'`.
Post `trade_rejected` card.

#### 2h. Budget gate — new position vs add/trim

Determine if this is opening a NEW position (no existing position in
this ticker) or ADD/TRIM to an existing one.

```bash
.venv/Scripts/python.exe -c "
from state.positions import open_positions
positions = open_positions(paper=True) or []
existing = [p for p in positions if p['ticker'].upper() == '{TICKER}']
print('add_or_trim' if existing else 'new_position')
"
```

If new position: call `can_open_new_position()`. If add/trim: call
`can_add_to_existing()`. If either returns false → record skip with
`skip_reason='budget cap: <reason>'`. Post `trade_rejected` card.

#### 2i. Place the order

Call `mcp__nemo_alpaca__place_paper_order(ticker, side='buy',
quantity=N, ...)`. Record budget:

```bash
.venv/Scripts/python.exe -c "
from agent.sentry_budget import record_action, ACTION_PAPER_ORDER, ACTION_NEW_POSITION, ACTION_ADD_TRIM
record_action(ACTION_PAPER_ORDER)
record_action(ACTION_NEW_POSITION)  # OR ACTION_ADD_TRIM
"
```

Record eval:

```bash
.venv/Scripts/python.exe -c "
from state.sentry_eval_log import record_eval
record_eval('{TICKER}', decision='acted', triggered_by='{TRIGGERED_BY}',
            verdict='buy', confidence={CONF}, sizing='{SIZING}',
            factor_buckets={FACTORS}, notes='order_id={ORDER_ID}')
"
```

Post `trade_placed` Slack card.

#### 2j. Mark queue row completed

```bash
.venv/Scripts/python.exe -c "from state.sentry_queue import mark_status; mark_status({queue_id}, 'completed')"
```

### 3. Tick summary

After processing 3 candidates (or fewer if queue smaller), post a
single Slack tick summary if any actions occurred:

```
Sentry tick {timestamp}
Processed N candidates: X researched, Y placed, Z skipped (reasons)
Queue: M pending remaining | Daily budget: research M/10, slack S/50, orders O/5
```

Only post the summary card if at least 1 action occurred (skip if
queue was empty).

## Slack card shapes

Four mrkdwn-formatted cards posted via
`mcp__slack__post_notification(text=...)`. Each starts with a
distinctive emoji-free header so the user can skim.

### Card 1: high_signal_event (queue-rejected, but worth flagging)

Used when triage queued something with score ≥ 0.85 but it failed an
early gate (e.g., read-through inconclusive). The user might still
want to see it.

```
[Sentry · high signal event]
*{TICKER}* — {triggered_by}, score {score}

{event headline}

Read-through: {result}
Decision: skipped, no research run
Recent evals: {1-line summary}
```

### Card 2: research_synthesis (researched, no trade)

```
[Sentry · research]
*{TICKER}* — {verdict} (sizing: {sizing}, confidence: {conf})

{1-sentence variant perception}

Bull case (cited):
• {factor 1}
• {factor 2}
• {factor 3}

Bear case (cited):
• {factor 1}
• {factor 2}

Falsifiers:
• {falsifier 1}
• {falsifier 2}

Decision: {watchlist | avoid | no_position} — {1-sentence reason}
Daily budget: research {R}/10, slack {S}/50
```

### Card 3: trade_placed (paper order placed)

```
[Sentry · trade placed]
*BUY {TICKER}* — {quantity} shares @ ~${entry_price}
Sizing: {sizing} ({pct}% of book) | Confidence: {conf}

Thesis: {1-sentence}
Variant perception: {1-sentence}

Gates passed:
✓ Risk_Officer: {approve_reason}
✓ Portfolio fit: {fit_verdict}
✓ Factor exposure: {factor_verdict}
✓ Kill-switch: intact

Falsifiers (will be monitored by falsifier_watcher):
• {falsifier 1}
• {falsifier 2}

Order: {order_id} | Thesis ID: {thesis_id}
Daily budget: orders {O}/5, new positions {N}/2
```

### Card 4: trade_rejected (proposed but gate failed)

```
[Sentry · trade rejected]
*{TICKER}* — proposed sizing: {sizing} ({pct}% of book)

Gate that failed: {gate_name}
Reason: {detailed reason}

Other gate results:
{gate_a}: {result_a}
{gate_b}: {result_b}

Recent evals: {1-line summary}
Daily budget: research {R}/10, orders {O}/5
```

## Hard rules

- Always call `risk_check_proposed_trade` before `place_paper_order`
  (existing CLAUDE.md rule — never bypass).
- Honor every budget cap. If a cap is hit mid-tick, stop processing
  remaining candidates and record them as `pending` (will retry next
  tick).
- Record every decision to `sentry_evaluation_log` — even skips. The
  log is the audit trail.
- Don't reuse a queue row's `processing` state across ticks. If
  something crashes mid-tick, the row is stuck — the next tick should
  detect rows in `processing` older than 30 min and reset them to
  `pending`.
- If any MCP call hangs > 30s, abandon that candidate, mark its row
  `dropped` with `notes='mcp_timeout'`, and continue to the next.
- Never post more than one Slack card per candidate (avoid spam).
  Tick summary is one additional card.
- Sentry never exceeds 1.0% position size, even when synthesis says
  `aggressive`. Discipline cap.

## Stuck-row cleanup (run at start of each tick)

Before reading new candidates, reset any rows stuck in `processing`
for > 30 minutes (likely a previous tick crashed):

```bash
.venv/Scripts/python.exe -c "
from state.schema import get_connection
from datetime import datetime, timezone, timedelta
cutoff = (datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat()
conn = get_connection()
n = conn.execute(
    \"UPDATE sentry_queue SET status = 'pending', notes = COALESCE(notes,'') || ' | reset_from_stuck_processing' WHERE status = 'processing' AND queued_at < ?\",
    (cutoff,)
).rowcount
conn.commit(); conn.close()
print(f'reset {n} stuck rows')
"
```

## When to invoke / skip

Invoke: `/loop 15m /sentry-tick` to start the autonomous loop. The
loop will fire every 15 min while the Claude Code session is open.

Skip: never manually — this skill is designed for the loop. To stop
the loop, run `/loop stop` or close the Claude Code session.

## Output

This skill produces side effects (database updates, Slack posts, paper
orders) — not a textual return value to the user. The Slack cards ARE
the output channel. Standard output to the conversation is brief
status: "Tick complete: N candidates, X placed, Y skipped" or similar.
