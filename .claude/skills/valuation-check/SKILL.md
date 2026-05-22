---
name: valuation-check
description: Answer "is this already priced in?" Pulls current multiples, compares against peer set and own history, computes FCF / earnings yield, and back-solves implied growth via reverse DCF. Returns a structured valuation envelope (cheap / fair / expensive / bubble-like / value trap) with the explicit "what must be true for the current price to work" answer. Invoke from /equity-deep-research Step 12, or standalone when the user asks "is X priced in" / "what's X worth" / "what are the comps."
---

# /valuation-check — Is it priced in?

A pure-valuation companion. The core question: at the current price,
what does the market expect, and what would have to happen for that
to be the right expectation?

## Inputs

- Primary ticker
- Time horizon (carries through to the implied-growth calculation)
- Thesis summary (one sentence)
- Peer group if known (otherwise derive via `get_company_peers`)

## Workflow

### 1. Current multiples (primary)

Call `get_market_data(primary_ticker)` and `get_basic_financials(primary_ticker)`.

Record: P/E, P/B, EV/Revenue, EV/EBITDA, EV/EBIT, FCF yield, earnings
yield, ROIC, ROE, gross margin, operating margin, net debt / EBITDA.

If EBITDA or net income is negative, mark P/E and EV/EBITDA as `n/a`
and pivot to EV/Revenue + cash-burn-adjusted analysis (do NOT crash).

### 2. Peer set (4-5 names)

Call `get_company_peers(primary_ticker)`. Take the top 4 (or 5 if any
look noisy and need extra signal).

For each peer in parallel: `get_market_data(peer)` and
`get_basic_financials(peer)`.

If fewer than 3 valid comparables (e.g., peer set is itself mostly
unprofitable), mark `data_gap: insufficient_peers` and fall back to
history-only comparison.

### 3. Build the comparison table

```
Metric            Primary   Peer1   Peer2   Peer3   Peer4   Median   Primary rank
EV/Revenue
EV/EBITDA
P/E
FCF yield
Operating margin
ROIC
Net debt / EBITDA
Revenue growth (YoY)
```

Flag where the primary is the **outlier** (1st or last). The thesis
must explain each outlier ranking.

### 4. History comparison (primary)

Pull historical multiples via `get_basic_financials` (it returns 52-week
high/low for valuation ratios). For deeper history, pull
`get_financial_statements` annually and reconstruct multiples by tying
to historical share count.

Compare current multiple to its own 1-year and 5-year range. Flag if
current is at top decile / bottom decile of own history.

### 5. Reverse DCF — back-solve implied expectations

Call `calculate_dcf(ticker, ...)` with current price as target output
and solve for implied growth (or, if the tool requires growth as
input, run it iteratively with growth values from 0% to 50% and find
the rate that produces an enterprise value matching current EV).

State the implied assumptions clearly:
- Implied revenue growth (5-year CAGR)
- Implied terminal operating margin
- Implied terminal capex intensity
- Implied WACC

Then state: "for the current price to make sense, [primary] must
deliver [implied growth] revenue CAGR through [horizon] while expanding
operating margin to [implied margin]. Compare this to history: company
has delivered [historical CAGR] revenue growth and [historical margin]
operating margin."

### 6. Multiple-compression risk

Assess: if rates rise 100 bps OR sector multiple compresses 20%, what
happens to fair-value calc? Surface the sensitivity explicitly.

## Output

```yaml
---
skill: valuation-check
ticker: <TICKER>
verdict: <cheap / fair / expensive / bubble_like / value_trap>
confidence: 0.0-1.0
key_finding: <one sentence on what the market is pricing in>
data_gaps: [<list>]
---

## /valuation-check — {TICKER}

**Current price**: $X.XX
**Market cap**: $X
**EV**: $X

**Multiples table** (primary vs peer median):
[table from Step 3]

**Historical context**:
- Current EV/Revenue: X.Xx (own 5-yr range: A.Ax - B.Bx, at Nth pct)
- Current EV/EBITDA: X.Xx (own 5-yr range: ...)

**What the market is pricing in**:
- Revenue CAGR: X% through {horizon}
- Operating margin: X%
- Terminal capex intensity: X%

**What must be true for the current price to work**:
[one paragraph, plain English]

**Multiple compression risk**:
- 100 bp rate move: -X% to fair value
- 20% sector multiple compression: -X% to fair value

**Valuation status**: cheap / fair / expensive / bubble-like / value trap

**Verdict**: one sentence on whether the thesis economics are
supported by valuation, contradicted by valuation, or neutral.

**Data gaps**:
- [any tool that failed]
```

## Hard rules

- If both EBITDA and FCF are negative, do not claim "cheap" based on
  EV/Revenue alone. The right answer for pre-profitability names is
  usually `expensive` (high EV/Revenue) or `value trap` (high
  EV/Revenue + decelerating growth) — call it honestly.
- If fewer than 3 valid peers, surface this as a structural risk
  ("no peer pricing backstop"), not just a data_gap footnote.
- Never claim a multiple is "cheap" without anchoring to BOTH peer
  median AND own history. A multiple in the bottom decile of own
  history may still be expensive vs peers.
- Do not output `cheap` for a name with deteriorating fundamentals.
  Mark as `value trap` instead.
- If `calculate_dcf` fails or returns nonsense, mark the reverse-DCF
  step as `data_gap: reverse_dcf_unavailable` rather than fabricating
  implied growth.

## Save output

If invoked standalone:
`testing/fixtures/valuation_check_<TICKER>_<DATE>.md`

If invoked from /equity-deep-research, return the structured output
inline so the caller can fold it into Step 12 of the synthesis.
