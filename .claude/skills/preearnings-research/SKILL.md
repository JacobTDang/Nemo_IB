# /preearnings-research

Run this 2–7 days before a company's earnings date. Collects alternative data
signals, synthesizes a surprise probability, optionally places a paper bid.

## Inputs

- Primary ticker (required)
- `mode`: `research_only` (default) | `auto_trade` (places bid after synthesis)
- `earnings_date_override`: YYYY-MM-DD (optional; defaults to next confirmed date)

## Workflow

### Step 1 — Confirm earnings date

Call `get_earnings_calendar(from_date=today, to_date=today+45)`.

Find the next entry for the ticker. If none found, output:
```
no_upcoming_earnings: no confirmed earnings date within 45 days for {TICKER}
```
and stop.

If earnings_date < today + 2 days, add to output:
```
WARNING: earnings in < 48 hours — IV crush risk, do not size large
```

Record `days_before_earnings = earnings_date - today` (negative = before).

### Step 2 — Supply chain map + supplier check

Call `get_supply_chain(ticker)`.

For each supplier in the supply chain:
1. Call `get_earnings_surprises(supplier_ticker)` to check if they reported this quarter.
2. If the supplier's most recent report date falls within the current quarter:
   - DELEGATE to `/cross-company-readthrough` with:
     - Primary company: supplier_ticker
     - Read-through target: ticker
     - Context: "supplier earnings impact on downstream company"
   - Record the readthrough direction (bullish / bearish / neutral) and key_findings.
   - Call `sentry_record_action` via `record_supplier_readthrough()` in `state/preearnings.py`.

Signal from supplier readthroughs:
- Any bullish readthrough → +1 signal
- Any bearish readthrough → -1 signal
- Mixed / neutral → 0

### Step 3 — Google Trends demand signal

Build a keyword list: [company brand name, top product name, "buy {product}"].
Examples: AAPL → ["iPhone", "buy iPhone", "Apple Watch"]

Call `get_google_trends(keywords=[...], timeframe="today 12-m", geo="US")`.

The tool handles 429 rate limits internally with retry + 12h SQLite cache.
If still unavailable, mark as `data_gap` and continue.

Interpret `yoy_ratio`:
- > 1.10 → bullish demand (direction=bullish, magnitude=min(yoy_ratio-1, 1.0))
- 0.90–1.10 → neutral
- < 0.90 → bearish (direction=bearish, magnitude=min(1-yoy_ratio, 1.0))

Record signal: `signal_category=demand`, `signal_name=google_trends`.

### Step 4 — Taiwan MOPS supplier revenue (if applicable)

Check if supply chain contains any of: TSMC (2330), Foxconn (2317),
MediaTek (2454), ASE Group (3711).

If yes, call `get_taiwan_monthly_revenue(company_codes=[matching_codes], months=6)`.

For each company returned:
- Compute current-quarter cumulative revenue vs same quarter last year.
- If quarter YoY > +10% → bullish. If < -5% → bearish. Else neutral.

Record signal: `signal_category=supply_chain`, `signal_name=taiwan_mops_revenue`.

If MOPS is unreachable or returns an error, mark as `data_gap` and continue.

### Step 5 — Government contracts

Call `get_government_contracts(ticker=ticker, months=12)`.

The tool queries USASpending.gov (no auth required) and compares trailing
12-month federal contract awards to the prior-year period.

Interpret `signal`:
- `bullish` → YoY contract growth >+10%, or any single award ≥ $1B
- `bearish` → YoY decline < -10%
- `neutral` → stable government revenue
- `not_applicable` → company has < $10M in trailing federal awards; skip from scoring

Note: software companies, consumer brands, and most healthcare names will
return `not_applicable`. Defense, aerospace, and IT services names will
have substantial data.

Record signal: `signal_category=supply_chain`, `signal_name=government_contracts`.

### Step 6 — Hiring signal

Call `get_job_postings_count(company_slug={slug}, ats="greenhouse")`.

The slug is the lowercase company name (e.g., "nvidia", "apple", "stripe").
The tool auto-discovers Greenhouse → Lever → Workday in parallel with no
hardcoded company list. If only a proprietary careers portal is found
(e.g., Oracle Taleo, SAP SuccessFactors), the tool returns a structured
`ats_detected` field with a clean error.

Interpret total postings vs 90-day-ago snapshot from `preearnings_signals`:
- Growth >15% → bullish (company investing ahead of revenue)
- Decline >15% or recent layoff news → bearish
- Otherwise → neutral

Record signal: `signal_category=hiring`, `signal_name=job_postings`.

### Step 7 — Capex announcements

Call `get_capex_announcements(ticker=ticker, lookback_days=180)`.

The tool scans recent news via DuckDuckGo for factory, data center, and
capital investment announcements. It extracts dollar amounts and classifies
each article as bullish (invest, build, expand) or bearish (cancel, delay, cut).

Interpret `signal`:
- `bullish` → major investment ($1B+) or most articles bullish
- `bearish` → cancellation/delay announcements dominate
- `neutral` → balanced or small-scale activity
- `data_gap` → no capex news found in the lookback window

Record signal: `signal_category=supply_chain`, `signal_name=capex_announcements`.

### Step 8 — Options implied move + skew

Call `get_options_implied_move(ticker=ticker, spot_price={current price})`.

The tool automatically fetches the options chain via yfinance when needed
(no obb_options_chain call required). It applies an 8% ATM gap guard:
if the nearest strike in a supplied chain is >8% from spot, yfinance is
used instead to ensure full ATM coverage.

Interpret:
- `implied_move_pct`: the market's own expected magnitude. Use as the
  confidence interval for the prediction (±implied_move_pct around spot).
- `skew_label`:
  - `put_heavy` → bearish lean (institutional hedging)
  - `call_heavy` → bullish lean (speculative positioning)
  - `balanced` → neutral
- `source`: `yfinance` means the chain was auto-fetched (normal);
  `supplied` means the caller passed rows.

Record signal: `signal_category=options`, `signal_name=options_implied_move`.

If `implied_move_pct > 0.20`, add to output:
```
HIGH VOLATILITY WARNING: market implies >20% move — binary event, not a research-edge trade
```

### Step 9 — Analyst estimates + revision trend

Call `get_forward_estimates(ticker)` and `get_analyst_recommendations(ticker)`.
DELEGATE to `/estimate-revision-watch` for the revision trajectory.

Interpret:
- Revision trajectory = rising → bullish
- Revision trajectory = falling → bearish
- Flat → neutral

Record signal: `signal_category=competitor`, `signal_name=estimate_revision`.

### Step 10 — News sentiment (FinBERT)

Call `get_company_news(ticker, from_date=today-30, to_date=today)`.

Extract a list of texts: [headline + ". " + summary] for each article (max 40).

Call `get_finbert_sentiment(texts=[...], ticker=ticker)`.

Interpret `signal`:
- `bullish` → direction=bullish, magnitude=min(net_score, 1.0)
- `bearish` → direction=bearish, magnitude=min(abs(net_score), 1.0)
- `neutral` → direction=neutral, magnitude=0.0

Record signal: `signal_category=sentiment`, `signal_name=finbert_news_sentiment`.

### Step 11 — Policy / legislative signals

Call `get_policy_signals(ticker=ticker, months=6)`.

The tool queries GovTrack (and Congress.gov if CONGRESS_API_KEY is set) for
bills relevant to the company's sector. Each bill is scored for bullish/bearish
language and weighted by its legislative status (enacted > passed > reported >
introduced).

Interpret `signal`:
- `bullish` → net bill score > +0.5 (pro-industry legislation advancing)
- `bearish` → net bill score < -0.5 (regulatory or trade headwinds)
- `neutral` → balanced legislative environment
- `data_gap` → GovTrack unreachable

Record signal: `signal_category=sentiment`, `signal_name=policy_legislative`.

### Step 12 — Consensus vs whisper (expectations hurdle)

DELEGATE to `/expectations-hurdle-check`.

Interpret setup:
- `easy` → market bar is low, beats are easier → bullish for surprise probability
- `difficult` → bar is high, misses are easier → bearish
- `balanced` → neutral

Record signal: `signal_category=sentiment`, `signal_name=expectations_hurdle`.

### Step 13 — Synthesize and output

**Weighted scoring (10 signals, sum = 1.00):**

| Signal | Weight | Direction |
|---|---|---|
| Supplier readthrough | 0.20 | from Step 2 |
| Taiwan MOPS YoY | 0.15 | from Step 4 |
| Google Trends YoY | 0.12 | from Step 3 |
| Options skew | 0.12 | from Step 8 |
| Government contracts | 0.12 | from Step 5 |
| Estimate revision | 0.09 | from Step 9 |
| FinBERT sentiment | 0.08 | from Step 10 |
| Capex announcements | 0.06 | from Step 7 |
| Hiring signal | 0.04 | from Step 6 |
| Policy/legislative | 0.02 | from Step 11 |

- Signals marked `not_applicable` (e.g., government contracts for consumer names)
  are excluded from scoring and their weight redistributed proportionally to the
  remaining signals.
- Signals marked `data_gap` are excluded; if >50% of total weight is missing, add
  `low_confidence: true` to output.

Convert direction to score: bullish=+1, neutral=0, bearish=-1.
Weighted_score = Σ(weight × score).

- `weighted_score > 0.25` → prediction=`likely_beat`, confidence = 0.50 + (weighted_score × 0.50)
- `-0.25 ≤ weighted_score ≤ 0.25` → prediction=`in_line`, confidence = 0.40 + (1 - abs(weighted_score)) × 0.10
- `weighted_score < -0.25` → prediction=`likely_miss`, confidence = 0.50 + (abs(weighted_score) × 0.50)

Cap confidence at 0.85 — no single pre-earnings call should exceed that.

**Store in DB:** call `record_signal()` for each step's signal. Call `record_eval()` with
the final prediction and confidence.

**Save:** `testing/fixtures/preearnings_{TICKER}_{DATE}.md`

## Output

```yaml
---
skill: preearnings-research
ticker: {TICKER}
earnings_date: {DATE}
days_before_earnings: {N}
prediction: likely_beat | in_line | likely_miss
confidence: 0.XX
implied_move_pct: {X.XX}
weighted_score: {X.XX}
low_confidence: true | false
---
```

Then a Markdown body with:

1. **Signal table** — one row per signal, with direction, magnitude, weight, weighted contribution
2. **Top 3 supporting signals** — the highest-magnitude bullish or bearish signals (cited with tool)
3. **Top 2 risk flags** — the highest-magnitude signals pointing opposite to the prediction
4. **Key data gaps** — tools that returned errors or empty data
5. **Implied move context** — what ±X% means for price (support / resistance levels from spot)

## Trading decision (mode=auto_trade only)

After synthesis, if `mode=auto_trade`:

1. Call `sentry_can_act(action_type=new_position)`. If not permitted, stop.
2. Determine side:
   - `likely_beat` + confidence ≥ 0.55 → `buy`
   - `likely_miss` + confidence ≥ 0.55 → `sell`
   - `in_line` or confidence < 0.55 → no trade (insufficient edge)
3. Get current account: `get_paper_account()`. Compute 1% of equity as max position size.
4. Call `risk_check_proposed_trade(ticker, side, quantity, price)` with the 1% size.
5. If approved: `place_paper_order(ticker, side, quantity, price)`.
   Then `sentry_record_action(action_type=paper_orders)` and `sentry_record_action(action_type=new_positions)`.
6. If rejected: report the rejection reasons, do not retry.

## Hard rules

- Never trade if `implied_move_pct > 0.20` (binary event, not research-edge).
- Never trade if `confidence < 0.50`.
- Never trade if `days_before_earnings < 2` (IV crush risk).
- Never fabricate an earnings date — only use what `get_earnings_calendar` returns.
- All signals must cite the tool that produced them.
- If a tool fails, record `data_gap` and continue — do not halt the workflow.
- Always call `risk_check_proposed_trade` before `place_paper_order`.
- Max 1% portfolio weight per earnings trade (binary event discipline).
- If supplier readthrough is available and strongly contradicts Google Trends,
  flag the contradiction explicitly and weight supplier readthrough as authoritative.
- Government contracts signals marked `not_applicable` do not count as data gaps —
  redistribute that weight rather than penalizing confidence.

## When to invoke

- User says "pre-earnings research on X" or "earnings research X"
- Sentry queue contains a candidate with `triggered_by=pre_earnings_5d`
- Run manually 7, 3, and 1 day before known earnings dates on watchlist names
