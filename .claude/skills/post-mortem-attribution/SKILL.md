---
name: post-mortem-attribution
description: Quantitative return decomposition for a closed position. Decomposes realized return into selection vs sector/market beta vs multiple expansion vs earnings growth vs FX/rate effects. Complements the qualitative /postmortem skill — that skill captures lessons; this one captures the math. Use after a position closes to learn whether the thesis was right for the right reason or right for the wrong reason.
---

# /post-mortem-attribution — Was the thesis right for the right reason?

The discipline: a +30% gain on a thesis that called for "AI capex
acceleration" but happened during a 25% market rally is not what it
looks like. The realized return is mostly beta — the alpha is +5%. If
the analyst takes the wrong lesson ("my AI capex thesis worked, do
more of those"), the next bet on the same pattern in a flat market
will disappoint.

This skill decomposes returns into their actual drivers so the analyst
learns the right lesson.

## Inputs

- Thesis ID OR ticker
- Entry date (when position was opened)
- Exit date (when position was closed)
- Position size (optional, for return contribution to portfolio)

## Workflow

### 1. Pull the entry and exit data

For the primary ticker, call `mcp__nemo_financial__get_price_history`
with start=entry_date, end=exit_date.

Record:
- Entry price, exit price
- Total return (price gain + dividends if relevant)
- Holding period in days

### 2. Pull entry-period fundamentals for the multiple decomposition

For both entry date and exit date, pull the relevant multiple:
- For earnings-positive names: P/E (use `mcp__nemo_financial__get_market_data`
  but adjusted for the entry-date EPS — pull historical EPS from
  `mcp__nemo_finnhub__get_financial_statements`)
- For pre-profitability names: EV/Revenue (same approach)

Compute:
- Entry multiple: M_entry
- Exit multiple: M_exit
- Multiple change: (M_exit - M_entry) / M_entry

### 3. Pull SPY and sector ETF returns over the same window

- SPY return: `mcp__nemo_financial__get_price_history(SPY, ...)`
- Sector ETF: identify via `mcp__nemo_finnhub__get_company_profile`
  (gives sector) → pick the matching sector ETF (XLK for tech, XLF
  for financials, XLE for energy, etc.)
- Sector ETF return: same approach

### 4. Pull beta

`mcp__nemo_financial__get_market_data(ticker)` returns the beta. Use
this for the market-attribution piece.

### 5. Decompose the return

Standard equity attribution:

```
Total return = Market beta × SPY return                 (market effect)
             + Sector beta × (Sector ETF - SPY) return  (sector effect)
             + Multiple change × (1 + earnings growth)  (multiple effect)
             + Earnings growth × M_entry / P_entry      (earnings effect)
             + Residual                                  (alpha / selection)
```

Simplified for the skill output:

| Component | Contribution to return | % of total |
|-----------|------------------------|-----------|
| Market (beta × SPY) | X% | A% |
| Sector (sector beta × (sector ETF - SPY)) | X% | B% |
| Multiple expansion / contraction | X% | C% |
| Earnings growth | X% | D% |
| Stock-specific alpha (residual) | X% | E% |
| **Total realized** | **X%** | **100%** |

The largest absolute component is the dominant driver.

### 6. Compare predicted vs actual driver

Cross-reference with the original thesis (read from
`testing/fixtures/research_<TICKER>_<DATE>.md` or
`mcp__nemo_financial__get_thesis_evolution`).

The original thesis claimed a primary driver (e.g., "multiple
expansion as Azure ROIC clears WACC"). The decomposition shows the
actual driver. Classify:

- **correct**: predicted and actual drivers match
- **partially**: thesis identified one of several drivers but missed
  the dominant one
- **wrong**: thesis driver and actual driver are different (right
  for the wrong reason)

### 7. Identify the lesson

For each alignment outcome:
- **correct + positive return**: thesis pattern works, replicate
- **correct + negative return**: thesis was right but timing wrong
  (or position size wrong), tighten entry discipline
- **partially**: thesis was incomplete, expand the model
- **wrong + positive return**: lucky, do NOT replicate the pattern
  on this basis (this is the danger case)
- **wrong + negative return**: clear miss, dissect the analysis

## Output

```
## /post-mortem-attribution — {TICKER}

**Holding period**: {entry_date} to {exit_date} ({N} days)
**Position size**: {if available}

**Return summary**:
- Entry price: $X.XX
- Exit price: $X.XX
- Total realized return: +/- X%
- Annualized: +/- X%

**Benchmark returns over holding period**:
- SPY: +/- X%
- Sector ETF ({TICKER}): +/- X%
- Stock vs SPY: +/- X% (alpha vs market)
- Stock vs sector: +/- X% (alpha vs sector)

**Return decomposition**:

| Component | Contribution | % of total |
|-----------|--------------|-----------|
| Market (beta × SPY) | X% | A% |
| Sector (above market) | X% | B% |
| Multiple change | X% (entry {M_e}x → exit {M_x}x) | C% |
| Earnings growth | X% | D% |
| Stock-specific alpha | X% | E% |
| **Total** | **X%** | **100%** |

**Dominant driver**: [market beta / sector beta / multiple expansion
/ earnings growth / stock alpha]

**Predicted driver** (per original thesis): [...]
**Driver alignment**: correct / partially / wrong

**Lesson**:
[paragraph: was the thesis right for the right reason? What should
change in the analyst's process going forward? Specifically:
- If `wrong + positive`: explicitly call out "right for wrong reason"
  and document what NOT to replicate
- If `correct + positive`: identify the specific data tools / pattern
  that worked and reinforce its use
- If `correct + negative`: identify the timing / sizing failure]

**Add to /knowledge/analogues.md**: yes / no
[If a clean learning emerges that generalizes, suggest adding to the
historical analogue catalog]

**Data gaps**:
- [any tool that failed; e.g., historical EPS may be unavailable]
```

## Hard rules

- Do not skip the multiple-change calculation just because the name
  is unprofitable. Use EV/Revenue for pre-profitability names.
- Do not classify driver alignment as `correct` if the thesis claimed
  alpha but the decomposition shows it was 80%+ beta. Be honest.
- A `wrong + positive` outcome is the most important lesson to
  surface — analysts naturally overweight `correct + positive` and
  ignore `wrong + positive`. Force the discomfort.
- If the holding period is < 30 days, the decomposition is noisier;
  surface a caveat. Short holds attribute mostly to timing rather than
  business fundamentals.
- If the dominant driver is `market beta`, the thesis-specific lesson
  is limited. The position would have returned roughly the same
  amount with any beta-1.0 exposure.

## When to invoke

- After any position closes (paper or otherwise)
- After /postmortem (qualitative) — these are complementary
- When the user asks "was that win/loss real" or "why did X work"

## When to skip

- For positions held < 14 days (signal too noisy)
- For positions where entry / exit dates aren't clearly defined

## Save output

If invoked standalone:
`testing/fixtures/postmortem_attribution_<TICKER>_<EXIT_DATE>.md`

If invoked as a companion to /postmortem, return the decomposition +
lesson inline so the caller can fold it into the qualitative writeup.
