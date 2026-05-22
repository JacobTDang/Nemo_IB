---
name: estimate-revision-watch
description: Track whether a signal or new data point is likely to force sell-side analysts to raise or lower estimates. Stocks often move on expectation changes, not on absolute business quality. Returns trajectory (rising/flat/falling), dispersion, and predicted revision direction over the next 30 days. Use whenever a thesis depends on consensus catching up — or whenever you want to know if the sell-side is behind the data you've gathered. Scope: sell-side trajectory only — does NOT pull buyside whisper signals (use /expectations-hurdle-check for that).
---

# /estimate-revision-watch

Stocks move on expectation changes, not absolute business quality. A
great quarter doesn't matter to the price if analysts already
believe it. This skill measures revision trajectory and predicts
whether new data forces analyst numbers up or down.

**Scope (disjoint from /expectations-hurdle-check)**: this skill
focuses on sell-side revisions and beat/miss history. It does NOT
pull options skew, pre-earnings price action, or buyside whisper —
those live in `/expectations-hurdle-check`.

## Inputs

- Primary ticker | Time horizon (default = next 30 days)
- Optional: thesis summary (used to predict revision direction)

## Workflow

### 1. Revision history

`get_analyst_revisions_history(ticker)`. Record upward / downward
revisions (30d, 90d), net revision count, trajectory (rising / flat
/ falling).

### 2. Beat/miss history

`get_earnings_surprises(ticker)`. Record beat rate over last 8 Qs,
average surprise magnitude, most recent print (beat / inline / miss),
and whether the most recent print triggered subsequent revisions.

### 3. Dispersion (from upstream consensus)

Consume the consensus number and dispersion from the upstream
`/equity-deep-research` synthesis (Step 13) OR from a prior
`/expectations-hurdle-check` invocation. Do NOT call
`get_forward_estimates` here — that's the upstream's job to avoid
duplicate work.

If running standalone with no upstream consensus available, mark
`data_gap: no_upstream_consensus` and proceed with revision history
only (still useful — trajectory works without level).

Wide dispersion = unsettled consensus = revisions likely either
direction once new data arrives. Tight dispersion = locked-in
consensus = revisions less likely without major catalyst.

### 4. Predict revision direction

Given the thesis (passed as input or inferred from synthesis):
- Acceleration signals (segment revenue accelerating, margins
  expanding, backlog growing) → predict UPWARD revisions
- Deceleration signals (margin compression, customer concentration
  warnings, capex outpacing revenue) → predict DOWNWARD revisions
- Mixed → predict DISPERSION widening, no clear direction

### 5. Sell-side-behind-the-data signal (highest edge)

If analyst's data shows clear directional signal AND recent revisions
are still flat or going the other way, that's "sell-side behind" —
the highest-edge case.

Examples: accelerating Azure capex + flat revisions → upward
revisions likely next 30d. AMD MI400 supply constraints lifted +
revisions still rising → consensus may be too high.

## Output

```yaml
---
skill: estimate-revision-watch
ticker: <TICKER>
verdict: <up_likely / down_likely / wider_likely / unchanged>
confidence: 0.0-1.0
key_finding: <one sentence on revision dynamic vs thesis>
data_gaps: [<list — note no_upstream_consensus if no consensus input>]
---

**Revision trajectory** (30d / 90d):
Upward: X / Y | Downward: X / Y | Net: X | Trajectory: rising / flat / falling

**Earnings print history** (last 8 Qs):
Beat rate: X% | Avg surprise: +/- X% | Most recent: beat / inline / miss

**Consensus context** (from upstream, if available):
EPS: $X.XX (dispersion: $C, N analysts) | Revenue: $X.XX (dispersion: $C)

**Sell-side-vs-data alignment**:
Analyst data implies revisions: UP / DOWN / WIDEN
Recent revision trajectory: UP / DOWN / FLAT
Alignment: aligned / behind / contradicted

**Predicted revision direction (next 30 days)**:
Direction: up / down / wider / unchanged
Magnitude: small / medium / large
Confidence: 0.0-1.0
Key catalyst forcing the revision: [earnings / guidance / pre-announcement / sector print]

**Implication for thesis** (one paragraph: consensus already up = partly priced in; sell-side behind = more room)

**Data gaps**: [any]
```

## Hard rules

- Don't predict UP with high confidence if recent revisions are
  FALLING and signal is qualitative only. Trajectory is sticky.
- Don't assume dispersion = analyst disagreement. May mean sparse
  coverage. Cross-reference with `analysts covering` count.
- If `get_analyst_revisions_history` returns < 5 data points in last
  90d, mark `data_gap: insufficient_revision_data`. Penny stocks and
  micro-caps have weak sell-side coverage; skill becomes uninformative.
- "Sell-side behind" is high-confidence ONLY when 3+ independent
  data points support the directional read. One data point isn't
  enough.

## When to invoke / skip

Invoke: `/equity-deep-research` Step 18 (always); after major
earnings print; when sizing into a thesis that depends on consensus
catching up; when thesis depends on a re-rate. Skip: pure value
names with no growth narrative; names with < 5 covering analysts;
purely technical / positioning theses. Standalone runs save to
`testing/fixtures/estimate_revisions_<TICKER>_<DATE>.md`.
