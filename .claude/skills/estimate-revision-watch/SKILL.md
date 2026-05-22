---
name: estimate-revision-watch
description: Track whether a signal or new data point is likely to force sell-side analysts to raise or lower estimates. Stocks often move on expectation changes, not on absolute business quality. Returns trajectory (rising/flat/falling), dispersion, and predicted revision direction over the next 30 days. Use whenever a thesis depends on consensus catching up — or whenever you want to know if the sell-side is behind the data you've gathered.
---

# /estimate-revision-watch — Will analysts revise?

The discipline: a great quarter only matters to the stock price if it
forces analysts to raise numbers. A weak signal that consensus already
believes is priced in produces no move. This skill measures the
trajectory of analyst estimates and predicts whether new data will push
revisions up or down.

## Inputs

- Primary ticker
- Optional: time horizon (default = next 30 days)
- Optional: thesis summary (used to predict revision direction)

## Workflow

### 1. Pull revision history

Call `mcp__nemo_finnhub__get_analyst_revisions_history(ticker)`.
Record:
- Number of upward revisions (last 30 days, 90 days)
- Number of downward revisions (same windows)
- Net revision count: upgrades - downgrades
- Trajectory: rising (recent > older) / flat / falling

### 2. Pull current consensus + dispersion

Call `mcp__nemo_finnhub__get_forward_estimates(ticker)`.
Record:
- Consensus EPS / revenue / EBITDA for next 4-6 quarters
- High vs low analyst estimates (dispersion = high - low)
- Number of analysts covering

Wide dispersion = unsettled consensus = high probability of revisions
in either direction once new data arrives. Tight dispersion = locked-in
consensus = revisions less likely without a major catalyst.

### 3. Pull earnings-print history

Call `mcp__nemo_finnhub__get_earnings_surprises(ticker)`.
Record:
- Beat/miss rate over last 8 quarters
- Average surprise magnitude
- Most recent print: beat / inline / miss
- Did the most recent print trigger upward or downward revisions in
  the following 30 days?

### 4. Cross-reference with the current thesis

Given the thesis (passed as input or read from current
`/equity-deep-research` synthesis), predict whether the data the
analyst has gathered would push revisions UP or DOWN:

- Acceleration signals (segment revenue accelerating, margins
  expanding, backlog growing) → predict UPWARD revisions
- Deceleration signals (margin compression, customer concentration
  warnings, capex outpacing revenue) → predict DOWNWARD revisions
- Mixed → predict DISPERSION widening, no clear directional revision

### 5. Sell-side-behind-the-data signal

If the analyst's data shows clear directional signal AND recent
revisions are still flat or going the other way, that's the
"sell-side is behind" signal — the highest-edge case for the thesis.

Examples:
- Thesis data shows accelerating Azure capex → revisions still flat
  → upward revisions likely in next 30 days
- Thesis data shows AMD MI400 supply constraints lifted → revisions
  still rising → consensus may be too high

## Output

```
## /estimate-revision-watch — {TICKER}

**Revision trajectory** (last 30 / 90 days):
- Upward: X / Y
- Downward: X / Y
- Net: X
- Trajectory: rising / flat / falling

**Current consensus** (next 4 quarters):
- EPS: $X.XX (range: $A - $B, dispersion: $C)
- Revenue: $X.XX (range: $A - $B, dispersion: $C)
- Analysts covering: N

**Earnings print history** (last 8 Qs):
- Beat rate: X%
- Average surprise: +/- X%
- Most recent: beat / inline / miss

**Sell-side-vs-data alignment**:
- Analyst data implies revisions: UP / DOWN / WIDEN
- Recent revision trajectory: UP / DOWN / FLAT
- Alignment: aligned / behind / contradicted

**Predicted revision direction (next 30 days)**:
- Direction: up / down / wider / unchanged
- Magnitude: small / medium / large
- Confidence: 0.0-1.0
- Key catalyst that would force the revision: [earnings / guidance /
  pre-announcement / sector data print]

**Implication for thesis**:
[one paragraph: does the revision dynamic support or weaken the
thesis? If consensus is already up, the thesis is partly priced in.
If sell-side is behind, the thesis has more room to play out.]

**Data gaps**:
- [any tool that failed]
```

## Hard rules

- Do not predict UP revisions with high confidence if recent
  revisions are FALLING and the only signal is qualitative. Trajectory
  is sticky.
- Do not assume dispersion = analyst disagreement. It may also mean
  sparse coverage. Cross-reference with `analysts covering` count.
- If `get_analyst_revisions_history` returns < 5 data points in the
  last 90 days, mark `data_gap: insufficient_revision_data`. Penny
  stocks and micro-caps often have weak sell-side coverage; the skill
  becomes uninformative.
- "Sell-side behind the data" is a high-confidence signal ONLY when
  the analyst has 3+ independent data points supporting the
  directional read. One data point is not enough.

## When to invoke

- /equity-deep-research Step 18 (always)
- After a major earnings print to assess revision momentum
- When sizing into a thesis that depends on consensus catching up
- When the thesis depends on a re-rate (the multiple expansion
  requires raised numbers)

## When to skip

- For pure value names with no growth narrative (revisions usually
  irrelevant)
- For names with < 5 covering analysts (insufficient data)
- For purely technical / positioning theses (no fundamental revision
  to track)

## Save output

If invoked standalone:
`testing/fixtures/estimate_revisions_<TICKER>_<DATE>.md`

If invoked from /equity-deep-research, return the predicted revision
direction + implication inline.
