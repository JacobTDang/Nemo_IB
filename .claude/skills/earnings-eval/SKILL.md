# /earnings-eval

Score a prior `/preearnings-research` prediction against the actual earnings
outcome. Builds a track record for improving the signal pipeline over time.

## Inputs

- Ticker (required)
- Earnings date (required — YYYY-MM-DD of the quarter that just reported)

## Workflow

### Step 1 — Load prior prediction

Query `state/preearnings.py:get_eval(ticker, earnings_date)`.

If no prediction row exists, output:
```
no_prediction_found: no preearnings-research was run for {TICKER} on {DATE}
```
and stop.

Also load all signals for this cycle:
`signals_for_ticker(ticker, earnings_date)` — used to identify which signals
were bullish/bearish.

### Step 2 — Actual EPS surprise

Call `get_earnings_surprises(ticker)`.

Find the quarter matching `earnings_date` (±7 days). Extract:
- `actual` EPS
- `estimate` EPS
- `surprise_percent` = (actual - estimate) / |estimate| × 100

Classify outcome:
- `surprise_percent > +3%` → outcome=`beat`
- `surprise_percent < -3%` → outcome=`miss`
- Otherwise → outcome=`in_line`

If no surprise data available (company not in Finnhub coverage), try
`get_financial_statements(ticker)` to get the reported EPS manually.
Mark `actual_eps_surprise = None` if unavailable.

### Step 3 — Revenue surprise

From the same `get_earnings_surprises` result or `get_financial_statements`,
extract the revenue surprise if available:
- `actual_rev_surprise` = (actual_revenue - estimated_revenue) / estimated_revenue

### Step 4 — 1-day price move

Call `get_price_history(ticker, from_date=earnings_date, to_date=earnings_date+2)`.

Extract the closing price on earnings_date and the next trading day.
`actual_price_move_1d` = (next_day_close - earnings_day_close) / earnings_day_close

### Step 5 — Score the prediction

Compare `prediction` (from prior run) to `outcome` (from Step 2):
- If `prediction=likely_beat` and `outcome=beat` → `prediction_correct=1`
- If `prediction=likely_miss` and `outcome=miss` → `prediction_correct=1`
- If `prediction=in_line` and `outcome=in_line` → `prediction_correct=1`
- Otherwise → `prediction_correct=0`

### Step 6 — Signal attribution

For each signal in `signals_for_ticker` (and each direction-bearing research
layer in `get_layers`):
- Was it directionally correct? (signal.direction agrees with actual outcome)
- Record which signals were correct and which were noise.

### Step 6b — Score the ASYMMETRY call separately

The direction model predicts the fundamental outcome; the asymmetry model
predicts the REACTION. Grade them independently:

Load the persisted `positioning` layer (`latest_component(ticker, earnings_date,
"positioning")`). Call `score_reaction(positioning, outcome, price_move_1d_pct,
implied_move_pct, prediction)` (`tools/preearnings/asymmetry_logic.py`):
- `price_direction_match`: did the 1-day move agree with the prediction?
  (a correct EPS call with an opposite price move = right on fundamentals,
  wrong on the trade)
- `asymmetry_correct`: did the crowding thesis hold? (crowded_long + beat ->
  muted reward; crowded_long + miss -> hard punishment; crowded_short
  mirrored). Neutral positioning -> not scored (None) — no call was made.

### Step 7 — Record and output

Call `record_eval(ticker, earnings_date, prediction=..., confidence=...,
actual_eps_surprise=..., actual_rev_surprise=..., actual_price_move_1d=...,
outcome=..., prediction_correct=..., asymmetry_correct=...,
price_direction_match=..., notes=...)`.

## Output

```yaml
---
skill: earnings-eval
ticker: {TICKER}
earnings_date: {DATE}
prior_prediction: likely_beat | in_line | likely_miss
prior_confidence: 0.XX
actual_outcome: beat | in_line | miss
prediction_correct: true | false
eps_surprise_pct: {X.X}%
price_move_1d: {X.X}%
---
```

Then:

**Signal attribution table:**
| Signal | Prior direction | Actual outcome aligns? |
|---|---|---|
| google_trends | bullish | yes |
| finbert_sentiment | neutral | — |
| ... | ... | ... |

**Post-mortem note** (1-2 sentences on the most informative signal and
the biggest miss).

**Aggregate stats** (if ≥ 3 evals exist):
Call `eval_accuracy_summary()` from `state/preearnings.py` and print:
```
Overall accuracy: X/N (XX%)
Avg confidence when correct: 0.XX
Avg confidence when wrong:   0.XX
```

## Hard rules

- Do not fabricate EPS or revenue numbers. If Finnhub returns no surprise
  data, mark `actual_eps_surprise = None` and note the gap.
- Classify outcome using EPS surprise only (primary). Price move is
  supplementary context, not the outcome classifier.
- A prediction of `in_line` that was followed by a 10% stock move is NOT
  wrong if EPS was in-line — note the guidance/multiple expansion component.

## When to invoke

- User says "eval NVDA earnings" or "score my AAPL prediction"
- 1 day after any watchlist company reports earnings
- When the user wants aggregate signal accuracy stats
