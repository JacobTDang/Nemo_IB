# /preearnings-review

Strict, **read-only** pre-print auditor. Runs AFTER `/preearnings-research` and
BEFORE acting on its prediction. It gathers no new research and never mutates
the prediction — it inspects what was collected and rules on whether the
research is fit to act on.

Deterministic checks live in `tools/preearnings/review_logic.py` (unit-tested);
this skill loads the data and feeds them.

## Inputs

- `ticker` (required)
- `earnings_date` (required, YYYY-MM-DD)

## Workflow

### Step 1 — Load the persisted research

- `get_layers(ticker, earnings_date)` — all research-layer components
- `get_eval(ticker, earnings_date)` — the prediction row

If there are no layers AND no eval row, output
`nothing_to_review: run /preearnings-research first` and stop.

### Step 2 — Fetch the second consensus source (the only live calls)

The dispersion check needs two independent reads of the same-quarter consensus:
- `get_forward_estimates(ticker)` -> 0q EPS avg (yfinance-fallback source)
- `get_earnings_calendar(from=today, to=today+45, symbol=ticker)` -> the
  calendar `eps_estimate` (Finnhub source)

These are the reviewer's ONLY network calls — everything else is inspection.

### Step 3 — Run the checks

`run_review(layers, eval_row, dispersion={eps_a, eps_b, label_a, label_b})`
executes, in order:

| Check | fail when | warn when |
|---|---|---|
| completeness | no synthesis / no eval row | missing peer/guidance/KPI/positioning/reaction |
| citations | direction-critical layer uncited | non-critical uncited, malformed entries |
| freshness | component > 72h old | component > 24h old |
| contradictions | — | high-magnitude opposing signals (must be NAMED) |
| stale_flags | — | quotes_stale / sentinel / data_gap markers in payloads |
| hard_rules | in_line sized, conf<0.55 sized, implied>20% sized, low_confidence flag inconsistent | — |
| db_consistency | eval row != latest synthesis | cannot compare |
| estimate_dispersion | — | sources >1% apart on consensus EPS |

Verdict: any fail -> `not_actionable`; any warn -> `sound_with_warnings`;
else `sound`.

### Step 4 — Persist and output

`record_layer(ticker, earnings_date, layer=3, component="review",
direction=None, payload=<full review result>, sources=[{claim: "review of N
checks", tool: "tools.preearnings.review_logic"}])`.

## Output

```yaml
---
skill: preearnings-review
ticker: {TICKER}
earnings_date: {DATE}
verdict: sound | sound_with_warnings | not_actionable
fails: {N}
warns: {N}
checks_run: {N}
---
```

Then: (1) the FAIL list with fixes (if any), (2) the WARN list with what must be
acknowledged in the final verdict, (3) the dispersion read (which consensus is
the bar?), (4) one line: "cleared to act" / "act only after fixes" / "do not act".

## Hard rules

- READ-ONLY: no new research, no prediction mutation, no trading calls. The
  only writes are the `review` layer and nothing else.
- The reviewer's job is disqualification, not cheerleading — when in doubt,
  warn. A warn costs one sentence of acknowledgment; a missed flaw costs money.
- `not_actionable` BLOCKS auto_trade: `/preearnings-research` mode=auto_trade
  must have a review verdict of `sound` or `sound_with_warnings` from within
  24h before placing any order.
- Never resolve a warn by editing the research — re-run the producing step.

## When to invoke

- Immediately after any `/preearnings-research` run, before acting on it
- Always before `mode=auto_trade` places an order (required gate)
- When the user asks "is this research solid?" / "review the ORCL call"
