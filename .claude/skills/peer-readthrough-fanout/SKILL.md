# /peer-readthrough-fanout

Go layers deeper on the single highest-edge, all-sector pre-earnings signal:
**same-quarter peer readthrough**. Peers, suppliers, and customers that have
ALREADY reported for the current quarter carry direct information about the
target's upcoming print. This skill derives that peer set dynamically, fans out
one sub-agent per reported peer, and aggregates the readthroughs into one signal.

Deterministic rules live in `tools/preearnings/peer_logic.py` (unit-tested);
this skill is the orchestration layer. **No hardcoded peer lists** — the peer
universe is derived at runtime.

## Inputs

- `ticker` (required) — the target reporting soon
- `earnings_date` (required) — the target's upcoming earnings date (YYYY-MM-DD)

## Workflow

### Step 1 — Derive the peer universe (dynamic)

Call both, in parallel, and union the results:
- `get_company_peers(ticker)` — competitors / same-sector peers
- `get_supply_chain(ticker)` — suppliers and customers

Tag each name with its relationship type from the tool that produced it
(`supplier` / `customer` / `competitor` / `peer` / `adjacent`). Never inject a
curated list.

### Step 2 — Establish the current-quarter window

Call `get_earnings_surprises(ticker)` to get the target's most recent report
date. The reporting window is `quarter_window(target_last_earnings, today)` —
from the target's previous print to today (clamped to a sane span). Peers that
reported inside this window are reporting on the same quarter.

### Step 3 — Find peers that already reported this quarter

For each peer, call `get_earnings_surprises(peer)` (batch in parallel) to get its
most recent report date. Keep peers where `reported_this_quarter(report_date,
window)` is true. Rank by `rank_peer_relevance(relationship)` (supplier/customer
> competitor/peer > adjacent), break ties by recency, and cap at the top 6
(`select_peers_for_fanout`) to bound cost.

If zero peers reported this quarter, record a `peer_readthrough` layer with
`direction=na, n=0` and stop — that is the honest result.

### Step 4 — Fan out one sub-agent per reported peer (parallel)

For each selected peer, spawn a sub-agent (in a single message, so they run in
parallel) running `/cross-company-readthrough`. *Cost rule:* when more than 4
same-relationship peers reported (e.g. several hyperscaler competitors with
April prints), the lower-information tail may be GROUPED into one combined
sub-agent — the freshest/highest-relevance peers always get dedicated agents.
Each grouped result still counts each peer's surprise individually in sources.

Per-agent contract:
- Primary company: the peer (the one that reported)
- Read-through target: `ticker`
- Context: "same-quarter peer earnings readthrough to an upcoming reporter"

Each sub-agent must return, for the target:
- `direction`: bullish / bearish / neutral
- `magnitude`: 0.0-1.0
- `sources`: a list of `{claim, tool}` — **every quantitative claim tagged with
  the tool that produced it**. Drop any claim a sub-agent cannot cite.

Per-agent discipline: if an agent errors or times out, mark that peer a data_gap
and continue — never fail the whole fan-out on one peer.

### Step 5 — Aggregate and persist

Combine the per-peer results with `aggregate_readthroughs(items)` (relevance-
weighted net). Persist with `record_layer(ticker, earnings_date, layer=1,
component="peer_readthrough", direction=..., magnitude=..., payload=<detail>,
sources=<flattened cited claims>)`.

## Output

```yaml
---
skill: peer-readthrough-fanout
ticker: {TICKER}
earnings_date: {DATE}
peers_reported: {N}
direction: bullish | bearish | neutral | na
magnitude: 0.XX
---
```

Then a short table: each reported peer, its relationship, its readthrough
direction/magnitude, and the one cited line that drove it. End with the
aggregated net direction.

## Hard rules

- Derive peers from tools only — no hardcoded company/peer lists.
- Every number in the output traces to a tool (citation audit). Uncited → dropped.
- Cap the fan-out (<=6) and time-bound each sub-agent; one failed peer never
  fails the run.
- "Reported this quarter" is defined by the window from the target's own cadence,
  never a hardcoded date.
- If no peer reported this quarter, return `direction: na` — do not fabricate a read.

## When to invoke

- Called by `/preearnings-research` (Layer 1) for any ticker within the deep-
  research window.
- Standalone when the user asks "who already reported that reads through to X".
