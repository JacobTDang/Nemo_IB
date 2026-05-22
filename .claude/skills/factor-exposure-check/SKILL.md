---
name: factor-exposure-check
description: Determine whether a thesis is stock-specific alpha or just disguised factor beta. Lite version — uses beta, sector, and theme-ETF membership instead of a full Barra-style factor model. Classifies primary factor exposures (market beta, momentum, AI/capex theme, rate sensitivity, commodity sensitivity, USD direction) and computes a stock-specific-alpha estimate. Use to prevent the "I'm long alpha, but actually just buying AI beta" mistake.
---

# /factor-exposure-check

Prevents the "+30% on an AI thesis while SOXX returned +28% — that's
not alpha" mistake. This is a LITE factor model (no Barra / no PCA);
heuristic, not institutional-grade.

## Inputs

- Primary ticker
- Thesis summary (one sentence — the variant perception claim)

## Workflow

### 1. Pull beta, sector, market cap

`get_market_data(ticker)` → beta, market cap.
`get_company_profile(ticker)` → sector / industry.

### 2. Theme exposures via ETF membership

`get_industry_etfs` for relevant themes:
- AI / data center: AIQ, SOXX, BOTZ, SMH
- Memory: SOXX | Cloud / SaaS: WCLD, IGV | Cybersecurity: HACK, CIBR
- Clean energy / battery: LIT, BATT, ICLN, TAN | Robotics: BOTZ, ROBO
- Biotech: ARKG, XBI, IBB | Fintech: FINX, ARKF | Defense: ITA, XAR, PPA
- Energy: XLE, XOP, OIH | Banks: KBE, KRE, XLF

Check if ticker appears in top holdings. 2+ thematic ETFs at >2%
weight = high theme beta. Only in broad sector ETFs (XLK, XLF) =
lower theme beta.

### 3. Classify each factor as high / medium / low

| Factor | high | medium | low |
|---|---|---|---|
| Market beta | β > 1.3 | 0.7-1.3 | < 0.7 |
| Momentum | +25% / 90d | +10-25% | flat or down |
| Theme beta | 2+ thematic ETFs >2% | 1 thematic ETF | broad sector only |
| Rate sensitivity | long-duration tech, REITs, utilities, biotech | most equities | banks, short-duration value |
| Commodity sensitivity | energy, materials, agri | industrials, transports | tech, staples |
| USD strength | tech mega-caps >$500B (AAPL, MSFT, GOOG, NVDA), staples mega-caps (KO, PG, PEP), multi-national pharma (PFE, JNJ, MRK, LLY) | large-cap industrials, large-cap discretionary, sub-mega semis | small/mid-cap US-domestic, banks, utilities, REITs, US-only services |

USD sensitivity is determined by **sector + market cap only** — do
NOT parse the company description for intl-revenue language; that
field is marketing text and rarely has the info.

### 4. Stock-specific alpha estimate

- All factors low/medium → `mostly alpha`
- One factor high → `partial alpha`
- 2+ factors high → `mostly factor`

### 5. Variant perception cross-check

Is the thesis variant perception itself a factor bet ("I think AI
continues another year")? Then variant is factor-directional, not
stock-specific — call it out. The variant should be stock-specific
("AMD's MI400 win rate at hyperscaler X is underappreciated") AND
factor exposure should be mostly alpha for the strongest shape.

### 6. Factor reversal risk

For each high factor exposure: what happens if the factor reverses?
- High AI theme → if hyperscaler capex guides down, position loses
  15-30% regardless of company performance
- High rate sensitivity (long-duration tech) → 10Y to 5.5% means
  15-25% multiple compression
- High momentum → factor reversal can give back 90 days of gains in
  3 weeks

This is the catalyst-independent risk the analyst is taking.

## Output

```yaml
---
skill: factor-exposure-check
ticker: <TICKER>
verdict: <mostly_alpha / partial_alpha / mostly_factor>
confidence: 0.0-1.0
key_finding: <one sentence on dominant factor exposure>
data_gaps: [<list>]
---

**Beta**: X.XX | **Sector**: {sector} | **Mkt cap**: $X

**Themes** (via ETF membership):
- {ETF}: weight X.X%

**Factor exposure**:
| Factor | Level | Note |
|---|---|---|
| Market beta | h/m/l | β {X.XX} |
| Momentum | h/m/l | 90d return {X%} |
| Theme beta ({theme}) | h/m/l | in {N} thematic ETFs |
| Rate sensitivity | h/m/l | {sector reasoning} |
| Commodity sensitivity | h/m/l | {sector reasoning} |
| USD strength | h/m/l | {sector + mkt cap reasoning} |

**Variant perception type**: stock-specific / factor-directional

**Stock-specific alpha estimate**: mostly alpha / partial alpha / mostly factor

**Reasoning** (one paragraph)

**Factor reversal risks** (per high-exposure factor):
- {factor}: {what happens if reverses, % position loss}

**Implication for thesis**: [paragraph]

**Data gaps**: [any]
```

## Hard rules

- Do not claim `mostly alpha` if the thesis is "I think [theme]
  continues." That's a factor bet regardless of ticker pick.
- Do not skip reversal risk — that's the whole point.
- This is a LITE model. No Barra / no PCA / no factor regression.
  Sanity check, not institutional-grade output.
- If ticker doesn't appear in any thematic ETFs, theme-beta is
  `low / unclear`, not `mostly alpha`. Absence of ETF membership
  ≠ absence of factor exposure (thematic ETFs may not have found
  the name yet).

## When to invoke / skip

Invoke: `/equity-deep-research` Step 18 (always); before sizing into
any obvious-theme name; when user pitches "stock-specific" thesis
that smells like factor. Skip: pure event-driven (M&A spread, FDA
binary) where factor exposure is dominated by the binary; closed
positions (use `/post-mortem-attribution`). Standalone runs save to
`testing/fixtures/factor_exposure_<TICKER>_<DATE>.md`.
