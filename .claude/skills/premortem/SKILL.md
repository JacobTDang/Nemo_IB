---
name: premortem
description: Pre-mortem analysis before committing capital. Forces 3 explicit failure scenarios for a thesis. Use whenever the user is about to size up a position, has just finished a research pass, or asks "what could go wrong with this trade." Skip for completed positions (use /postmortem instead).
---

# /premortem — Pre-commit failure mapping

Before sizing into any position, write 3 plausible scenarios in which the
thesis FAILS within the horizon you've set. The discipline is from
Daniel Kahneman: imagining the failure forward is more cognitively
honest than confirming the bull case.

## Inputs

Pull from the analyst's working memory (or ask the user):
- The thesis (one sentence)
- The time horizon (next quarter / next year / next 3 years)
- The position size relative to portfolio (1% / 5% / 10%+)
- The current falsifiers list (if a thesis row exists)

## The 3 scenarios

For each scenario, write:

1. **What happens** — concrete, specific. Not "the macro turns bad" but
   "10Y yields rise to 5.5% by year-end, forcing multiple compression
   across long-duration assets including the position."

2. **The leading indicator** — what observable would tell you the
   scenario is starting to unfold, BEFORE it shows up in price?

3. **The exit trigger** — what specific data point would force you to
   close the position if this scenario plays out?

## The 3 buckets to cover

To avoid overlap, force one scenario per bucket:

- **Fundamental**: the thesis's central economic claim turns out to be
  wrong (e.g., "Azure ROIC stays above WACC" — but ROIC compresses
  below WACC because hyperscaler capex hits diminishing returns)

- **Macro / regime**: the operating environment shifts in a way that
  invalidates the position regardless of company performance
  (e.g., rates rise enough to compress multiples on growth)

- **Idiosyncratic / company-specific**: something specific to the
  company breaks (CFO leaves, fraud surfaces, customer concentration
  bites, regulatory action)

## Output structure

Write to the conversation in this format:

```
## /premortem — {TICKER}

**Thesis**: <one sentence>
**Horizon**: <period>
**Size assumed**: <X% of portfolio>

### Scenario 1 (Fundamental): <title>
- What happens: ...
- Leading indicator: ...
- Exit trigger: ...

### Scenario 2 (Macro): <title>
- What happens: ...
- Leading indicator: ...
- Exit trigger: ...

### Scenario 3 (Idiosyncratic): <title>
- What happens: ...
- Leading indicator: ...
- Exit trigger: ...

### Falsifiers to add to the thesis row
- <falsifier text, one per line>
```

## Hard rules

- Each scenario must be SPECIFIC. "Things go wrong" is not a scenario.
- Each exit trigger must be OBSERVABLE without prediction. "Stock falls
  20%" is observable; "thesis is wrong" is not.
- All 3 buckets must be filled — no skipping. If you can't construct a
  fundamental scenario, that suggests you haven't actually understood
  the thesis economics.
- The final falsifier list should be added to the thesis row via
  `record_thesis_evolution` or by updating the thesis directly.

## When to invoke

- User says "premortem MSFT" or "what could go wrong with the NVDA
  trade"
- After /deep-research completes, before user commits capital
- When user is about to upsize a position (>2x current weight)

## When to skip

- For closed positions — use /postmortem instead
- For pure data lookups
- When no thesis exists yet — first do /deep-research
