---
name: portfolio-fit
description: Check whether a proposed new position fits the current paper book. Detects sector / factor / theme overlap with existing holdings, flags duplicate exposure, recommends maximum size given diversification constraints. Use before sizing into any new position, especially when the proposed thesis sounds similar to existing ones (e.g., another AI capex name when book already has 3).
---

# /portfolio-fit

A bullish NVDA thesis may not justify adding if the book already has
heavy AI capex via TSM/AMD/AVGO/MSFT. Marginal exposure is the
question, not standalone thesis quality.

## Inputs

- Proposed new ticker
- Proposed position size (% of portfolio, e.g., 2.0)
- Thesis tags (from the equity-deep-research synthesis: theme, sector,
  KPI driver, expected factor exposure)

## Workflow

### 1. Snapshot the current book

Call `mcp__nemo_alpaca__get_paper_positions` — returns current paper
positions with ticker, quantity, market value, and current price.

**Empty-book early return:**
If `positions == []` OR total portfolio value == 0:
- Output the envelope with `verdict: strong`, `note: empty_book`,
  `key_finding: "No existing positions; portfolio fit check is
  informational only — no concentration risk possible at zero book."`
- Recommended max size = proposed size (no constraint applies)
- Skip Steps 2-6 entirely; return immediately

Otherwise compute:
- Total portfolio value
- Existing position weights (% of book)
- Existing sector concentrations (group by sector)

### 2. Decompose existing exposures

Call `mcp__nemo_financial__analyze_exposures` — pulls active theses
from the DB and classifies them into factor buckets via the
`exposure_analyzer` taxonomy (ai_capex_long, memory_cycle,
rate_sensitive_long_duration, rate_sensitive_short_duration,
energy_long, china_consumer, etc.).

Record:
- Factor buckets with > 0 weight
- Weight per bucket (sum of position weights tagged to that factor)
- Number of names per bucket

### 3. Classify the proposed addition

For the proposed ticker, identify its factor buckets:
- Direct ticker match in the exposure_analyzer taxonomy?
- Sector match (call `mcp__nemo_finnhub__get_company_profile`)
- Theme match (call `mcp__nemo_financial__get_industry_etfs` and check
  if ticker appears in any returned ETF top holdings)

### 4. Compute the marginal exposure

For each factor bucket the proposed ticker belongs to:
- Current weight in that bucket: X%
- Marginal weight after adding proposed position: X% + proposed_size%
- Concentration warning: if any bucket exceeds 25% → caution; > 35% → reject

### 5. Compute correlation risk

For each existing position in the same factor buckets:
- Flag as a "duplicate exposure" pair if both names share > 1 factor
- Compute a rough correlation proxy: 60-day rolling correlation via
  `mcp__nemo_financial__get_price_history` for both tickers (small
  data window OK — this is a sanity check, not a Barra factor model)

### 6. Determine recommended sizing

- If no factor concentration AND no duplicates → support proposed size
- If concentration approaches 25% AND there's already a similar name
  → recommend half of proposed size
- If concentration would exceed 35% OR > 3 existing positions in the
  same bucket → recommend `reject` or `watchlist only`

## Output

```yaml
---
skill: portfolio-fit
ticker: <TICKER>
verdict: <strong / acceptable / weak / reject>
confidence: 0.0-1.0
key_finding: <one sentence on duplicate exposure or concentration finding>
data_gaps: [<list — may include empty_book>]
---

## /portfolio-fit — {TICKER}

**Proposed**: BUY {TICKER} at {N}% of book

**Current book** ({K} positions, ${TOTAL_VALUE}M):
[sector / factor summary]

**Existing exposure to {TICKER}'s factors**:
| Factor bucket | Current weight | Existing names | After addition |
|--------------|----------------|----------------|----------------|
| ai_capex_long | X%             | NVDA, AMD, AVGO | X+2% |
| ... | ... | ... | ... |

**Duplicate exposure flag**:
[name in book that overlaps most: which factors, current weight,
correlation if computed]

**Recommended position size**: X% (vs proposed Y%)

**Verdict**: strong / acceptable / weak / reject

**Reasoning** (one paragraph):
[why this verdict given the book composition]

**Hidden concentration check**:
- If correlations go to 1.0 in a drawdown, this addition makes the
  book {X}% exposed to factor [Y]. Stress loss in a 20% drawdown of
  factor Y: {Z}% of portfolio value.

**Data gaps**:
- [any tool that failed or returned partial data]
```

## Hard rules

- Do not recommend `strong` if any factor bucket would exceed 25% of
  the book after addition.
- Do not recommend `acceptable` if 3+ existing positions already
  share the proposed ticker's primary factor bucket.
- Do not skip the correlation check just because positions are in
  different sectors — modern correlations in stress often dominate
  sector boundaries.
- Recommend `reject` (no_position) only when concentration + duplicates
  are BOTH severe; otherwise prefer `weak / reduce size` so the
  analyst can scale in over time.

## When to invoke

- /equity-deep-research Step 18 (always, for any thesis with sizing
  >= cautious)
- Before any paper trade is submitted via /place_paper_order
- When user asks "should I add to my book" / "does X fit"

## When to skip

- For watchlist-only theses (not committing capital, just monitoring)
- For closing existing positions (reducing exposure is always safe)
- When the book has fewer than 2 active positions (no concentration
  risk possible)

## Save output

If invoked standalone:
`testing/fixtures/portfolio_fit_<TICKER>_<DATE>.md`

If invoked from /equity-deep-research, return the verdict + recommended
size inline so the caller can fold it into the synthesis.
