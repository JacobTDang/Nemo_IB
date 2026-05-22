---
name: thesis-kill-switch
description: Make falsifiers actionable. Evaluates all active theses against their declared falsifiers using current data, classifies each as intact / approaching / triggered, and recommends required action (continue / review / reduce / exit). Distinct from /thesis-monitor (broader thesis health) — this skill specifically operationalizes the falsifiers list into a kill-switch decision. Use before any new entry, on a daily/weekly schedule, or whenever a falsifier observable updates.
---

# /thesis-kill-switch — Make falsifiers actionable

A thesis without an actionable kill-switch is a faith claim. This skill
turns the falsifiers list (declared in `/equity-deep-research` Step 18)
into structured decisions: continue, review within 24h, reduce size,
or exit.

The backing engine is `daemons/falsifier_watcher.py`, which already does
the heavy lifting (loads active theses, evaluates falsifiers, records
triggered alerts to `thesis_evolution` with a negative conviction delta).
This skill is the synchronous, on-demand front-end to that daemon.

## Inputs

- Optional: specific thesis_id to evaluate (skips others)
- Optional: ticker (evaluates all theses for that ticker)
- Default: evaluate all active theses

## Workflow

### 1. Trigger a single-pass evaluation

Run the daemon in `--once` mode:

```bash
python -m daemons.falsifier_watcher --once
```

This runs ONE tick: loads active theses, evaluates falsifiers against
recent events + macro snapshot, records any new triggers to
`thesis_evolution`, and exits. Idempotent — does not re-fire triggers
already recorded.

### 2. Pull the post-tick state

Call `mcp__nemo_financial__get_thesis_evolution` for each active thesis
(or the specified thesis_id). Look at the most recent evolution entries
since the tick.

For each thesis, classify its kill-switch status:
- **intact**: no falsifier near trigger; conviction stable or rising
- **approaching**: at least one falsifier within 25% of trigger
  threshold (e.g., revenue $42M vs falsifier "< $40M")
- **one major triggered**: a single load-bearing falsifier fired
  (e.g., CFO replaced, key revenue threshold breached)
- **multiple triggered**: 2+ falsifiers fired across categories
  (fundamental + macro + positioning) — likely time to exit

### 3. For each thesis with approaching/triggered status, gather evidence

Pull the specific data points that fired the falsifier:
- Revenue falsifiers → `mcp__nemo_web__get_segment_financials`
- Short interest falsifiers → `mcp__nemo_financial__get_short_interest`
- Macro falsifiers → `mcp__nemo_fred__get_macro_snapshot`
- Management falsifiers → `mcp__nemo_finnhub__get_company_news`
  (search for executive change keywords)
- Multiple expansion falsifiers → `mcp__nemo_financial__get_market_data`

Include the exact observed value in the output so the analyst can
verify the trigger logic was correct (no false alarms).

### 4. Recommend required action per thesis

Mapping (mirrors handoff doc Section 11.20):
- **intact** → `continue`
- **approaching (1 falsifier)** → `monitor`, re-check in 1 week
- **approaching (2+ falsifiers)** → `review within 24h`,
  consider reducing size
- **one major triggered** → `review within 24h`, mandatory action
  decision: hold / trim / exit
- **multiple triggered** → `exit unless valuation now fully compensates`

### 5. If invoked pre-entry (for a proposed new thesis)

Special mode: the thesis being evaluated isn't yet in the DB. Accept
the proposed falsifier list inline and evaluate each against current
data WITHOUT relying on the daemon. This is the "would we trigger any
falsifier on day 1" sanity check.

## Output

```
## /thesis-kill-switch

**Tick date**: {ISO timestamp}
**Active theses evaluated**: {N}

**Per-thesis status**:

| Thesis ID | Ticker | Status | Falsifiers triggered | Required action |
|-----------|--------|--------|----------------------|-----------------|
| 267       | EOSE   | approaching | 1 of 7 | monitor, re-check in 1 week |
| 122       | MSFT   | intact      | 0 of 5 | continue |
| ...       | ...    | ...         | ...    | ... |

**Triggered falsifiers detail**:

For each thesis with approaching/triggered status:

### {TICKER} (thesis_id={ID})

**Falsifier**: [exact text from thesis row]
**Trigger threshold**: [e.g., "revenue Q2 2026 < $40M"]
**Observed value**: [e.g., "$38M, source: get_segment_financials"]
**Severity**: minor / major
**Recommended action**: continue / monitor / review within 24h /
                       reduce / exit
**Evidence**:
- [tool: data point]
- [tool: data point]

**System-wide alerts**:
- Any thesis where 2+ falsifiers fired in the same tick
- Any thesis where conviction has dropped > 0.30 since inception

**Data gaps**:
- [theses where evaluation was incomplete due to missing data]
```

## Hard rules

- Do not classify a thesis as `triggered` based on a single tool call
  that returned `error` or empty data. Insufficient evidence = mark
  `data_gap`, not `triggered`. False triggers erode trust.
- A falsifier that has fired ONCE in the daemon's history (via the
  `falsifier_alerts` idempotency table) is already recorded — do not
  re-fire. The skill should report it but not produce a duplicate
  thesis_evolution entry.
- Pre-entry mode: if ANY proposed falsifier would already trigger on
  day 1 with current data, do NOT enter the thesis. The thesis is
  born broken.
- Do not recommend `exit` for a single approaching falsifier — exit
  is reserved for major triggered or multiple-triggered states. Use
  `reduce` for intermediate states.

## When to invoke

- /equity-deep-research Step 18 (pre-entry mode, for proposed new
  thesis)
- Daily / weekly cron schedule via `/loop 24h /thesis-kill-switch`
- Whenever a major macro event hits (Fed decision, CPI print, geopolitical)
- After any earnings print for a held position

## When to skip

- For watchlist-only theses (no capital at risk; falsifiers are
  informational only)
- For theses with no falsifier list (the skill has nothing to
  evaluate — the right output is "no falsifiers declared, cannot
  evaluate kill-switch")

## Save output

If invoked standalone:
`testing/fixtures/kill_switch_<DATE>.md` (system-wide) or
`testing/fixtures/kill_switch_<TICKER>_<DATE>.md` (single thesis)

If invoked from /equity-deep-research Step 18 pre-entry mode, return
the per-thesis status inline so the caller can fold it into the
synthesis. If any falsifier would trigger on day 1, the caller MUST
honor the no-entry recommendation.
