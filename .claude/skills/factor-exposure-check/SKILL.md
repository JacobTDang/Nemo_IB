---
name: factor-exposure-check
description: Determine whether a thesis is stock-specific alpha or just disguised factor beta. Lite version — uses beta, sector, and theme-ETF membership instead of a full Barra-style factor model. Classifies primary factor exposures (market beta, momentum, AI/capex theme, rate sensitivity, commodity sensitivity, USD direction) and computes a stock-specific-alpha estimate. Use to prevent the "I'm long alpha, but actually just buying AI beta" mistake.
---

# /factor-exposure-check — Alpha or just factor beta?

The discipline: a +30% return on an "AI capex thesis" during a year
when SOXX returned +28% is not alpha. The analyst was just long
AI/semis beta dressed up as a stock-specific thesis. The next bet on
the same pattern in a flat-beta environment will disappoint.

This is a **lite** factor model. Without Barra / Axioma data, we can't
do a proper PCA-based decomposition. We can do something useful with
beta + sector + theme-ETF membership, and we call it what it is — a
heuristic, not a rigorous model.

## Inputs

- Primary ticker
- Thesis summary (one sentence — the "variant perception" claim)
- Optional: time horizon

## Workflow

### 1. Pull beta and sector

- Beta: `mcp__nemo_financial__get_market_data(ticker)` → beta field
- Sector: `mcp__nemo_finnhub__get_company_profile(ticker)` → sector +
  industry

### 2. Identify theme exposures via ETF membership

Call `mcp__nemo_financial__get_industry_etfs` for the relevant themes:
- AI / data center: AIQ, SOXX, BOTZ, SMH
- Memory cycle: SOXX
- Cloud / SaaS: WCLD, IGV
- Cybersecurity: HACK, CIBR
- Clean energy / battery: LIT, BATT, ICLN, TAN
- Robotics / automation: BOTZ, ROBO
- Genomics / biotech: ARKG, XBI, IBB
- Fintech: FINX, ARKF
- Defense: ITA, XAR, PPA
- Energy / oil: XLE, XOP, OIH
- Banks: KBE, KRE, XLF

For each ETF, check if the primary ticker appears in the top
holdings. Record:
- Themes the ticker is in
- Weight of the ticker in each ETF (proxy for theme purity)

If ticker is heavy in 2+ thematic ETFs → high theme beta.
If ticker is only in broad sector ETFs (XLK, XLF) → lower theme beta,
more idiosyncratic.

### 3. Classify factor exposures (lite taxonomy)

For each factor, classify as `high / medium / low / n/a` based on
heuristics:

**Market beta**:
- high: beta > 1.3
- medium: beta 0.7-1.3
- low: beta < 0.7

**Momentum**:
- high: stock up > 25% in last 90 days
- medium: stock up 10-25%
- low: flat to down

**Theme beta (AI / cloud / battery / etc.)**:
- high: ticker in 2+ thematic ETFs with weight > 2%
- medium: ticker in 1 thematic ETF
- low: only in broad sector ETF

**Rate sensitivity**:
- high: long-duration tech, REITs, utilities, biotech (sector match)
- medium: most equities
- low: short-duration value, banks (negative rate sensitivity)

**Commodity sensitivity**:
- high: energy, materials, agriculture sectors
- medium: industrials, transports
- low: tech, consumer staples

**USD strength sensitivity** (deterministic sector + market-cap
proxy — do NOT parse the company description for international
revenue language; that field is marketing text and rarely contains
the relevant info):
- high: tech mega-caps with mkt cap > $500B in IT sector (typically
  derive > 50% revenue internationally — AAPL, MSFT, GOOG, NVDA);
  consumer staples mega-caps (KO, PG, PEP); multinational pharma
  (PFE, JNJ, MRK, LLY)
- medium: large-cap industrials, large-cap consumer discretionary,
  semis below mega-cap
- low: small/mid-cap US-domestic, financials (banks especially),
  utilities, REITs, US-only services

Inputs: market cap from `get_market_data`, sector from
`get_company_profile`. Classify based on the sector + mkt-cap
combination; do not attempt to read intl-revenue % from any narrative
text field.

### 4. Compute stock-specific alpha estimate

This is the punch line. Given the factor exposures, what fraction of
the thesis is genuinely stock-specific?

Heuristic:
- If ALL factor exposures are low/medium → `mostly alpha`. The
  thesis stands on its own.
- If ONE factor is high (e.g., AI theme) → `partial alpha`. The
  thesis has alpha but also rides a factor.
- If 2+ factors are high (e.g., AI theme + high momentum + high
  beta) → `mostly factor`. The thesis is largely disguised factor
  exposure.

### 5. Cross-reference with the variant perception

Read the variant perception from the thesis. If the variant is
itself a factor bet ("I think AI continues for another year"), call
this out — variant perception should be stock-specific, not factor-
directional.

If the variant is stock-specific ("AMD's MI400 win rate at hyperscaler
X is underappreciated"), AND the factor exposure check shows mostly
alpha → the thesis is in the strongest shape.

### 6. Factor reversal risk

For each high factor exposure, ask: what happens to the thesis if
the factor goes the OTHER way?

Examples:
- High AI theme exposure → what if hyperscaler capex guides DOWN
  next quarter? Position loses 15-30% regardless of company-specific
  performance.
- High rate sensitivity (long-duration tech) → what if 10Y goes to
  5.5%? Multiple compression of 15-25%.
- High momentum → what if momentum factor reverses? Stock can give
  back 90 days of gains in 3 weeks.

This is the **catalyst-independent risk** the analyst is taking.

## Output

```
## /factor-exposure-check — {TICKER}

**Beta**: X.XX (market exposure: high / medium / low)
**Sector**: {sector / industry}
**Themes** (via ETF membership):
- {ETF}: weight X.X%
- {ETF}: weight X.X%

**Factor exposure classification**:
| Factor | Level | Note |
|--------|-------|------|
| Market beta | high/medium/low | beta {X.XX} |
| Momentum | high/medium/low | 90-day return {X%} |
| Theme beta ({theme}) | high/medium/low | in {N} thematic ETFs |
| Rate sensitivity | high/medium/low | {duration / sector reasoning} |
| Commodity sensitivity | high/medium/low | {sector reasoning} |
| USD strength | high/medium/low | {international revenue context} |

**Variant perception type**:
- Stock-specific: [variant perception claim]
- OR
- Factor-directional: [variant perception is itself a factor bet]

**Stock-specific alpha estimate**: mostly alpha / partial alpha / mostly factor

**Reasoning** (one paragraph):
[why this estimate, given the factor exposure profile]

**Factor reversal risks**:
For each high-exposure factor:
- {factor}: what happens if it reverses, and how much does the position
  lose independent of the thesis playing out

**Implication for thesis**:
[paragraph: if the thesis is mostly factor, sizing should be smaller
AND the analyst should explicitly understand they are taking the
factor risk. If mostly alpha, the thesis is in the strongest shape.]

**Data gaps**:
- [any tool that failed; international revenue may not be available]
```

## Hard rules

- Do not claim `mostly alpha` if the thesis is just "I think [theme]
  continues." That's a factor bet, regardless of which ticker you
  pick to express it.
- Do not skip the reversal risk section. The whole point is to
  surface how much of the position is at risk from forces independent
  of the company.
- Be explicit that this is a LITE factor model. No Barra. No PCA.
  No actual factor regression. The analyst should not treat the
  output as institutional-grade — it's a sanity check.
- If ticker doesn't appear in any thematic ETFs, the theme-beta
  classification is `low / unclear` not `mostly alpha`. Absence of
  ETF membership ≠ absence of factor exposure (it just means thematic
  ETFs haven't found the name yet).

## When to invoke

- /equity-deep-research Step 18 (always)
- Before sizing into any name that fits an obvious theme (AI, clean
  energy, China consumer)
- When the user pitches a "stock-specific" thesis that smells like
  factor

## When to skip

- For pure event-driven positions (M&A spread, regulatory binary)
  where factor exposure is dominated by the binary outcome
- For closed positions (use /post-mortem-attribution instead)

## Save output

If invoked standalone:
`testing/fixtures/factor_exposure_<TICKER>_<DATE>.md`

If invoked from /equity-deep-research, return the classification +
reversal risks inline.
