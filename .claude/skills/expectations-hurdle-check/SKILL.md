---
name: expectations-hurdle-check
description: Estimate the gap between published sell-side consensus and the unstated buyside expectations (whisper number). Captures the failure mode where a company beats consensus but misses the whisper and still sells off. Returns: consensus hurdle, implied buyside whisper, setup classification (easy / balanced / difficult). Use before earnings, when sizing into a name with run-up momentum, or whenever the thesis depends on "beats expectations."
---

# /expectations-hurdle-check — Beat consensus, miss the whisper

A company can beat consensus EPS by 5% and still sell off if the
buyside whisper number was 12% above consensus. This skill estimates
the gap between published consensus and what the market actually
expects, so the analyst doesn't size into a "beat" that may already
be priced in.

## Inputs

- Primary ticker
- Optional: earnings date (next quarterly print)
- Optional: position context (currently held? proposed entry?)

## Workflow

### 1. Published consensus

Call `mcp__nemo_finnhub__get_forward_estimates(ticker)`.
Record:
- Consensus EPS for next quarter
- Consensus revenue for next quarter
- High / low / mean / dispersion
- Number of analysts

### 2. Recent price action as a hurdle proxy

Call `mcp__nemo_financial__get_price_history(ticker, days=60)`.
Compute:
- Stock return over last 30 days (pre-earnings)
- Stock return over last 60 days
- Stock return vs SPY (relative performance)
- Stock return vs sector ETF (sector-relative)

**Interpretation**:
- Stock up sharply pre-earnings (e.g., +15% in 30 days vs SPY flat) →
  buyside hurdle is ABOVE consensus
- Stock flat or down pre-earnings → buyside hurdle may be AT or BELOW
  consensus
- Stock up modestly tracking sector → buyside hurdle ≈ consensus

### 3. Options skew

Call `mcp__nemo_financial__get_options_metrics(ticker)`.
Record:
- Implied vol (IV) vs realized vol — high IV = market expects big move
- Put/call ratio or skew if available
- 25-delta call IV vs put IV

**Interpretation**:
- Call skew (calls more expensive than puts) = market positioned for
  upside surprise → whisper above consensus
- Put skew = market positioned for downside surprise → whisper below
  consensus
- High IV without directional skew = market expects volatility but
  doesn't know direction → whisper = consensus

### 4. CFO tone trajectory

Call `mcp__nemo_web__extract_call_sentiment(ticker, quarters=2)`.
Record:
- Most recent call: confident / stable / hedging
- Trend over last 2 quarters

**Interpretation**:
- Recent confident tone after a beat → whisper above consensus
  (analysts trust the CFO and have raised privately)
- Recent hedging tone → whisper below consensus
- Stable tone → whisper ≈ consensus

### 5. Synthesize the whisper

Combine the 4 signals into a single whisper estimate:

| Signal | Direction | Magnitude |
|--------|-----------|-----------|
| Price action | UP / DOWN / FLAT | small / medium / large |
| Options skew | UP / DOWN / FLAT | small / medium / large |
| CFO tone | UP / DOWN / FLAT | small / medium / large |
| Revision trajectory* | UP / DOWN / FLAT | small / medium / large |

(* Optionally also call `/estimate-revision-watch` for a 4th signal.)

Aggregate: if 3+ signals point UP, buyside whisper is meaningfully
above consensus. Estimate the magnitude as the implied beat needed
to surprise the buyside:
- All weak signals → whisper = consensus + small (e.g., 2-3%)
- Mixed strong signals → whisper = consensus + medium (5-8%)
- All strong signals → whisper = consensus + large (10%+)

### 6. Classify the setup

- **easy**: stock down pre-earnings, options skew flat, low expectations
  → low bar to beat → upside on any in-line print
- **balanced**: signals mixed → result depends on direction of beat/miss
- **difficult**: stock up sharply, call skew high, CFO confident →
  must beat AND raise AND guide higher to satisfy buyside

## Output

```
## /expectations-hurdle-check — {TICKER}

**Earnings date**: {date if available}

**Published consensus** (next Q):
- EPS: $X.XX (range $A - $B)
- Revenue: $X.XX
- Analysts: N

**Pre-earnings stock action**:
- 30-day return: +/- X%
- 60-day return: +/- X%
- vs SPY: +/- X%
- vs sector: +/- X%
- Signal: hurdle above / at / below consensus

**Options positioning**:
- IV vs realized: X.Xx
- Skew: call-heavy / put-heavy / neutral
- Signal: hurdle above / at / below consensus

**CFO tone**:
- Most recent print: confident / stable / hedging
- Trend: improving / stable / deteriorating
- Signal: hurdle above / at / below consensus

**Implied buyside whisper**:
- EPS: $X.XX (consensus + Y%)
- Revenue: $X.XX (consensus + Z%)
- Confidence in whisper estimate: 0.0-1.0

**What is needed for the stock to go up**:
- Beat consensus EPS by >= X%
- Raise full-year guidance by >= Y%
- KPI [thesis-critical KPI] shows acceleration

**What miss would trigger material downside**:
- EPS miss > X%
- Guidance cut > Y%
- KPI deceleration

**Setup**: easy / balanced / difficult

**Implication for thesis**:
[one paragraph: does the setup favor or disfavor the position
into this print? Should sizing be reduced ahead of the catalyst?]

**Data gaps**:
- [tools that failed; e.g., options data may be unavailable]
```

## Hard rules

- Do not call setup `easy` if stock is up > 10% in 30 days. Hurdle
  is structurally elevated regardless of other signals.
- Do not call setup `difficult` based on options skew alone. Skew can
  reflect hedging, not directional bets — corroborate with at least
  one other signal.
- If options data is unavailable, mark `data_gap` and proceed with
  3-signal aggregate. Do not fabricate a skew read.
- Whisper estimates are inherently uncertain. Do not output magnitude
  more precise than "small / medium / large" — false precision
  misleads.
- If the earnings date is > 4 weeks away, the skill's signals are
  less reliable (price action and options reflect more than just
  earnings expectation). Lower confidence accordingly.

## When to invoke

- /equity-deep-research Step 18 (always for any thesis with held
  position OR proposed entry within 4 weeks of next earnings)
- 1-2 weeks before each quarterly earnings print of held positions
- Whenever the thesis depends on "beats expectations" as a catalyst

## When to skip

- For positions where the next catalyst is > 8 weeks out
  (signals too noisy)
- For very illiquid names where options markets don't exist
- For pure technical / positioning theses with no fundamental
  catalyst

## Save output

If invoked standalone:
`testing/fixtures/expectations_hurdle_<TICKER>_<DATE>.md`

If invoked from /equity-deep-research, return the setup + implication
inline.
