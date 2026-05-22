---
name: scenario-builder
description: Convert an investment thesis into bear / base / bull risk-reward math with explicit assumptions and probability weights. Returns a probability-weighted expected return and a risk/reward verdict (attractive / balanced / poor). Use to prevent a good story from hiding bad risk/reward. Invoke from /equity-deep-research Step 16, or standalone when the user asks "what's the upside / downside" or "what does the math say."
---

# /scenario-builder — Risk-reward math

The companion that turns a story into numbers. Three named scenarios,
explicit assumptions for each, probability-weighted expected value.

## Inputs

- Thesis (one sentence)
- Primary ticker
- Time horizon (typically the synthesis's stated horizon)
- Thesis-critical KPI from /equity-deep-research Step 5
- Current segment trajectory from Step 6
- Valuation envelope from /valuation-check (current multiples + peer median)
- Falsifiers list (from any prior /premortem or thesis row)

## Workflow

### 1. Anchor on the current multiple

From `get_market_data(primary_ticker)`, record current price, share
count, EV, and the most relevant multiple (P/E if profitable, EV/EBITDA
otherwise, EV/Revenue for unprofitable growth names).

### 2. Build the three cases

For each of bear / base / bull, write out:

| Field | What to specify |
|---|---|
| Driver narrative | One sentence tying back to the thesis-critical KPI |
| Revenue at horizon | Year-N revenue (in $) — must cite the assumption ("Z3 deliveries reach 5 GWh by 2028") |
| Operating margin at horizon | % — must reference peer median or own history |
| Implied EPS or FCF | computed |
| Exit multiple | Use peer median (from /valuation-check) for base, peer 25th-pct for bear, peer 75th-pct for bull |
| Implied share price | (EPS or FCF × exit multiple) ÷ share count, adjusted for any expected dilution |
| Return vs current price | % |
| Probability | 0.0 to 1.0 |

**Rule**: probabilities across the three cases MUST sum to exactly 1.0.

**Rule**: bear case must explicitly include at least one falsifier from
the thesis's falsifier list (if any falsifier list exists). This forces
the bear case to be grounded in known risk, not a generic "things go
wrong."

**Rule**: bull case revenue assumption may not exceed historical peak
revenue growth × 1.5 unless the analyst can name a specific structural
break (new product, regulatory unlock, etc.). Caps speculative
extrapolation.

### 3. Compute expected value

```
E[return] = P_bear × R_bear + P_base × R_base + P_bull × R_bull
```

### 4. Classify risk/reward

- `attractive`: E[return] > +15% AND P_bear × |R_bear| < 0.5 × P_bull × R_bull
- `balanced`: E[return] in [-5%, +15%]
- `poor`: E[return] < -5% OR P_bear × |R_bear| > P_bull × R_bull

### 5. Surface the load-bearing assumption

Identify the single assumption that, if changed, most moves expected
return. State explicitly: "Expected return is most sensitive to
[assumption]. A 10% change in this variable shifts E[return] by [X%]."

## Output

```yaml
---
skill: scenario-builder
ticker: <TICKER>
verdict: <attractive / balanced / poor>
confidence: 0.0-1.0
key_finding: <one sentence with expected return + dominant assumption>
data_gaps: [<list>]
---

## /scenario-builder — {TICKER}

**Current price**: $X.XX
**Horizon**: {N years}
**Anchor multiple**: {EV/EBITDA / P/E / EV/Revenue}

**Cases**:

| Case | Prob | Driver | Revenue Y{N} | Margin | EPS/FCF | Exit Mult | Price | Return |
|------|------|--------|--------------|--------|---------|-----------|-------|--------|
| Bear | 0.XX | ...    | $X           | Y%     | $Z      | N.x       | $A    | -B%    |
| Base | 0.XX | ...    | ...          | ...    | ...     | ...       | ...   | +/-%   |
| Bull | 0.XX | ...    | ...          | ...    | ...     | ...       | ...   | +Y%    |

**Probabilities sum**: 1.00 (must equal 1.00)

**Expected return**: +/- X%
**Upside / downside skew**: [bull return / bear return ratio]
**Risk / reward verdict**: attractive / balanced / poor

**Load-bearing assumption**:
[the single variable most affecting E[return], with sensitivity quantified]

**Bear-case falsifier integration**:
The bear case incorporates the following thesis falsifier(s):
- [list]

**Data gaps**:
- [any inputs that were missing or estimated]
```

## Hard rules

- Probabilities must sum to 1.00 exactly. Round individual probabilities
  if needed; do not gloss over a mismatch.
- Do not assign P(bull) > 0.50 unless the variant perception in the
  underlying thesis is strongly evidence-backed.
- Do not output `attractive` if the bear case has > 30% probability
  AND the bear return is more than -40%. That's a binary bet, not an
  attractive setup.
- If the load-bearing assumption is something the analyst cannot
  observe (e.g., "competitor TAM share"), surface this — the thesis
  is fragile because the key variable is unobservable.
- Do not fabricate exit multiples. If no peer median is available from
  valuation-check, use the company's own 5-year median multiple and
  note the substitution.

## Save output

If invoked standalone:
`testing/fixtures/scenario_<TICKER>_<DATE>.md`

If invoked from /equity-deep-research, return the structured output
inline so the caller can fold it into Step 16 of the synthesis.
