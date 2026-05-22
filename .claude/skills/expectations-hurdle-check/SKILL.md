---
name: expectations-hurdle-check
description: Estimate the gap between published sell-side consensus and the unstated buyside expectations (whisper number). Captures the failure mode where a company beats consensus but misses the whisper and still sells off. Returns: implied buyside whisper, setup classification (easy / balanced / difficult). Use before earnings, when sizing into a name with run-up momentum, or whenever the thesis depends on "beats expectations." Scope: buyside whisper only — does NOT pull sell-side estimates (consume from upstream synthesis or /estimate-revision-watch).
---

# /expectations-hurdle-check

Estimates the buyside whisper gap vs published consensus using price
action + options skew + CFO tone. Disjoint from
`/estimate-revision-watch` (does NOT pull `get_forward_estimates` or
`get_analyst_revisions_history` — consumes consensus from upstream).

## Inputs

- Primary ticker | Earnings date (if known)
- Published consensus EPS and revenue (from upstream — required input)
- Position context: held / proposed entry / watchlist

## Workflow

### 1. Pre-earnings stock action (hurdle proxy)

`get_price_history(ticker, days=60)`. Compute returns: 30-day, 60-day,
vs SPY, vs sector ETF.

- Stock +15% in 30d vs SPY flat → hurdle ABOVE consensus
- Flat or down pre-earnings → hurdle AT or BELOW consensus
- Up modestly tracking sector → hurdle ≈ consensus

### 2. Options skew

`get_options_metrics(ticker)`. Record IV vs realized vol, put/call
skew, 25-delta call IV vs put IV.

- Call skew (calls > puts) → market positioned for upside surprise →
  whisper above consensus
- Put skew → downside surprise positioning → whisper below consensus
- High IV no skew → market expects volatility, no direction →
  whisper ≈ consensus

### 3. CFO tone trajectory

`extract_call_sentiment(ticker, quarters=2)`. Last call: confident /
stable / hedging; trend over 2 quarters.

- Recent confident post-beat → whisper above consensus (analysts
  trust CFO and raised privately)
- Hedging → whisper below consensus
- Stable → whisper ≈ consensus

### 4. Synthesize the whisper

Aggregate 3 signals:

| Signal | Direction | Magnitude |
|---|---|---|
| Price action | UP / DOWN / FLAT | small / medium / large |
| Options skew | UP / DOWN / FLAT | small / medium / large |
| CFO tone | UP / DOWN / FLAT | small / medium / large |

If 2+ signals point UP → whisper meaningfully above consensus.
- All weak signals → consensus + small (2-3%)
- Mixed strong → consensus + medium (5-8%)
- All strong → consensus + large (10%+)

### 5. Classify the setup

- `easy`: down pre-earnings, options skew flat, low expectations → low bar
- `balanced`: signals mixed → depends on direction
- `difficult`: up sharply, call skew high, CFO confident → must
  beat AND raise AND guide higher to satisfy buyside

## Output

```yaml
---
skill: expectations-hurdle-check
ticker: <TICKER>
verdict: <easy / balanced / difficult>
confidence: 0.0-1.0
key_finding: <one sentence on whisper vs consensus gap>
data_gaps: [<list — likely missing options data for illiquid names>]
---

**Earnings date**: {date if available}
**Consensus (from upstream)**: EPS $X.XX | Revenue $X.XX

**Pre-earnings stock action**:
30d +/- X% | 60d +/- X% | vs SPY +/- X% | vs sector +/- X%
Signal: hurdle above / at / below consensus

**Options positioning**:
IV vs realized: X.Xx | Skew: call-heavy / put-heavy / neutral
Signal: hurdle above / at / below

**CFO tone**:
Most recent: confident / stable / hedging | Trend: improving / stable / deteriorating
Signal: hurdle above / at / below

**Implied buyside whisper**:
EPS: $X.XX (consensus + Y%) | Revenue: $X.XX (consensus + Z%)
Confidence in whisper: 0.0-1.0

**What's needed for stock to go up**:
- Beat consensus EPS by >= X%
- Raise full-year guidance by >= Y%
- KPI [thesis-critical KPI] shows acceleration

**Miss that would trigger material downside**:
- EPS miss > X% | Guidance cut > Y% | KPI deceleration

**Setup**: easy / balanced / difficult

**Implication for thesis** (one paragraph: should sizing change ahead of catalyst?)

**Data gaps**: [any]
```

## Hard rules

- Don't call setup `easy` if stock is up > 10% in 30d. Hurdle
  structurally elevated regardless of other signals.
- Don't call setup `difficult` on options skew alone. Skew can be
  hedging, not directional — corroborate with ≥ 1 other signal.
- If options data unavailable, mark `data_gap: options_unavailable`
  and proceed with 2-signal aggregate (price + tone).
- Whisper magnitude is `small / medium / large` only — false
  precision misleads.
- If earnings date > 4 weeks out, signals are noisier (price and
  options reflect more than earnings expectation). Lower confidence.

## When to invoke / skip

Invoke: `/equity-deep-research` Step 18 (for held positions OR
proposed entry within 4 weeks of next earnings); 1-2 weeks before
each quarterly print of held positions; when thesis depends on
"beats expectations" as catalyst. Skip: catalysts > 8 weeks out;
illiquid names without options markets; pure technical theses.
Standalone runs save to
`testing/fixtures/expectations_hurdle_<TICKER>_<DATE>.md`.
