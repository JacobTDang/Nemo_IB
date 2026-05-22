---
name: post-mortem-attribution
description: Quantitative return decomposition for a closed position. Decomposes realized return into selection vs sector/market beta vs multiple expansion vs earnings growth vs FX/rate effects. Complements the qualitative /postmortem skill — that skill captures lessons; this one captures the math. Use after a position closes to learn whether the thesis was right for the right reason or right for the wrong reason.
---

# /post-mortem-attribution

A +30% gain during a 25% market rally is +5% alpha, not +30%. This
skill catches the "right for the wrong reason" failure so the analyst
doesn't replicate luck.

## Inputs

- Thesis ID OR ticker | Entry + exit dates | Position size (optional)

## Workflow

### 1. Price + dividend data

`get_price_history(ticker, entry_date, exit_date)`. Record entry /
exit price, total return (with divs), holding period.

### 2. Multiple at entry vs exit

For earnings-positive: P/E (via `get_market_data` adjusted with
historical EPS from `get_financial_statements`).
For pre-profitability: EV/Revenue (same approach).
Compute: M_entry, M_exit, multiple change %.

### 3. Benchmarks over the same window

- SPY: `get_price_history(SPY, ...)`
- Sector ETF: identify via `get_company_profile` (XLK tech, XLF
  financials, XLE energy, XLV health, XLY discretionary, XLP staples,
  XLI industrials, XLU utilities, XLB materials, XLRE real estate)

### 4. Beta

`get_market_data(ticker)` → beta field.

### 5. Decompose — TWO independent frameworks

These describe the same realized return two different ways. Do NOT
sum them; present side-by-side.

**Framework A — Fundamental (multiplicative price ID):**
```
P_exit / P_entry = (EPS_exit / EPS_entry) × (PE_exit / PE_entry)

Linear approximation (returns < ~25%):
r_total ≈ r_eps + r_pe + r_dividend
```
For pre-profitability, substitute EV/Revenue for PE and revenue growth
for EPS growth. Same identity holds.

**Framework B — Benchmark attribution (cross-sectional):**
```
r_total = β_market × r_spy + β_sector × (r_sector − r_spy) + α_stock_specific
```
α is the residual after subtracting market and sector contributions.

**Worked sanity-check example:**
Stock +30% over horizon. EPS grew 12%, PE 20x→23.4x (+17%), no
dividend. SPY +10%, sector ETF +15%, β = 1.2.
- Framework A: 12% + 17% ≈ **29.4%** ≈ 30% ✓
- Framework B: 1.2×10% + 1.0×(15%−10%) + α = 12% + 5% + α → α = **13%**

Both decompositions sum (within rounding) to realized 30%. Framework
A answers "was the thesis driver correct"; Framework B answers "was
this alpha or beta."

**Math sanity check before publishing:**
- Both framework sums must be within 2% of total return
- If either is off by > 2%, mark `data_gap: attribution_imprecise`
  and surface the discrepancy rather than hide it (most common cause:
  missing EPS at one endpoint — fall back to the clean framework)

### 6. Compare predicted vs actual driver

Read the original thesis driver. Classify:
- `correct`: predicted matches dominant actual
- `partially`: thesis mentioned one of several but missed dominant
- `wrong`: predicted and actual drivers differ (right for wrong reason)

### 7. Identify the lesson

- `correct + positive` → pattern works, replicate
- `correct + negative` → thesis right but timing / sizing wrong
- `partially` → thesis incomplete, expand model
- `wrong + positive` → LUCKY, do NOT replicate (this is the danger)
- `wrong + negative` → clear miss, dissect analysis

## Output

```yaml
---
skill: post-mortem-attribution
ticker: <TICKER>
verdict: <correct / partially / wrong>
confidence: 0.0-1.0
key_finding: <one sentence on dominant driver + alignment>
data_gaps: [<list>]
---

**Holding period**: {entry_date} to {exit_date} ({N} days)
**Position size**: {if available}

**Return summary**: entry $X → exit $Y → total +/- X% (annualized +/- X%)

**Benchmarks**: SPY +/- X% | Sector ETF {ticker} +/- X% | Stock vs SPY +/- X% | Stock vs sector +/- X%

**Decomposition (two independent frameworks)**:

| Framework A — Fundamental | Contribution | % of total |
|---|---|---|
| EPS / revenue growth | X% | A% |
| Multiple change ({M_e}x → {M_x}x) | X% | B% |
| Dividend yield | X% | C% |
| **A+B+C** | **≈ X% (vs realized X%)** | sanity OK / OFF |

| Framework B — Benchmark | Contribution | % of total |
|---|---|---|
| Market beta (β × SPY) | X% | D% |
| Sector beta (above market) | X% | E% |
| Stock-specific alpha (residual) | X% | F% |
| **D+E+F** | **≈ X% (vs realized X%)** | sanity OK / OFF |

**Dominant driver**: [EPS growth / multiple / dividend / market β / sector β / alpha]
**Predicted driver** (original thesis): [...]
**Driver alignment**: correct / partially / wrong

**Lesson** (one paragraph; emphasize "wrong + positive" if applicable —
that's the danger case analysts naturally underweight)

**Add to knowledge/analogues.md**: yes / no
**Data gaps**: [any]
```

## Hard rules

- Don't skip multiple-change for unprofitable names — use EV/Revenue.
- Don't classify alignment as `correct` if thesis claimed alpha but
  decomposition shows 80%+ beta. Be honest.
- `wrong + positive` is the most important lesson — analysts overweight
  `correct + positive` and underweight `wrong + positive`. Force the
  discomfort.
- Holding period < 30 days → decomposition is noisy; surface caveat.
- If dominant driver is `market beta`, the thesis-specific lesson is
  limited (any β=1.0 exposure would've returned similarly).

## When to invoke / skip

Invoke: after any position closes (paper or otherwise); after
`/postmortem` (qualitative — these are complementary); when user asks
"was that win/loss real." Skip: positions held < 14 days (signal too
noisy); positions without clear entry/exit dates. Standalone runs
save to `testing/fixtures/postmortem_attribution_<TICKER>_<EXIT_DATE>.md`.
