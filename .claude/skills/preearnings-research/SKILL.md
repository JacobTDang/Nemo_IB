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

### Step 5 — Hiring signal

Call `get_job_postings_count(company_slug={slug}, ats="greenhouse")`.

To derive the slug: use the lowercase company name with hyphens (e.g.,
"nvidia" → "nvidia", "apple" → "apple", "meta" → "meta").
If Greenhouse returns 404, the tool auto-falls back to Lever.

Interpret:
- Total postings this call vs 90-day-ago snapshot (from `preearnings_signals` table if prior run exists):
  - Growth >15% → bullish (company investing ahead of revenue)
  - Decline >15% or recent layoff news (`search` for "{company} layoffs 2026") → bearish
  - Otherwise → neutral

Record signal: `signal_category=hiring`, `signal_name=job_postings`.

### Step 6 — Options implied move + skew

Call `obb_options_chain(ticker=ticker)`.

Pass the `rows` field to `get_options_implied_move`:
```
get_options_implied_move(
  ticker=ticker,
  spot_price={current price from get_market_data},
  options_chain_rows={data.rows from obb_options_chain}
)
```

Interpret:
- `implied_move_pct`: the market's own expected magnitude. Use as the confidence
  interval for the prediction (±implied_move_pct around current price).
- `skew_label`:
  - `put_heavy` → bearish lean (institutional hedging)
  - `call_heavy` → bullish lean (speculative positioning)
  - `balanced` → neutral

Record signal: `signal_category=options`, `signal_name=options_implied_move`.

If `implied_move_pct > 0.20`, add to output:
```
HIGH VOLATILITY WARNING: market implies >20% move — binary event, not a research-edge trade
```

### Step 7 — Analyst estimates + revision trend

Call `get_forward_estimates(ticker)` and `get_analyst_recommendations(ticker)`.
DELEGATE to `/estimate-revision-watch` for the revision trajectory.

Interpret:
- Revision trajectory = rising → bullish
- Revision trajectory = falling → bearish
- Flat → neutral

Record signal: `signal_category=competitor`, `signal_name=estimate_revision`.

### Step 8 — News sentiment (FinBERT)

Call `get_company_news(ticker, from_date=today-30, to_date=today)`.

Extract a list of texts: [headline + ". " + summary] for each article (max 40).

Call `get_finbert_sentiment(texts=[...], ticker=ticker)`.

Interpret `signal`:
- `bullish` → direction=bullish, magnitude=min(net_score, 1.0)
- `bearish` → direction=bearish, magnitude=min(abs(net_score), 1.0)
- `neutral` → direction=neutral, magnitude=0.0

Record signal: `signal_category=sentiment`, `signal_name=finbert_news_sentiment`.

### Step 9 — Consensus vs whisper (expectations hurdle)

DELEGATE to `/expectations-hurdle-check`.

Interpret setup:
- `easy` → market bar is low, beats are easier → bullish for surprise probability
- `difficult` → bar is high, misses are easier → bearish
- `balanced` → neutral

Record signal: `signal_category=sentiment`, `signal_name=expectations_hurdle`.

### Step 10 — Synthesize and output

**Weighted scoring:**

| Signal | Weight | Direction |
|---|---|---|
| Supplier readthrough | 0.25 | from Step 2 |
| Taiwan MOPS YoY | 0.20 | from Step 4 |
| Google Trends YoY | 0.15 | from Step 3 |
| Options skew | 0.15 | from Step 6 |
| Estimate revision | 0.10 | from Step 7 |
| FinBERT sentiment | 0.10 | from Step 8 |
| Hiring signal | 0.05 | from Step 5 |

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

## When to invoke

- User says "pre-earnings research on X" or "earnings research X"
- Sentry queue contains a candidate with `triggered_by=pre_earnings_5d`
- Run manually 7, 3, and 1 day before known earnings dates on watchlist names
