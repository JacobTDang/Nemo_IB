---
name: signal-backtest
description: Backtest a quantitative signal (technical indicator, fundamental ratio, multi-condition rule) over historical data to measure hit rate, average return, holding period, and drawdown. Use to validate a thesis pattern before sizing, or to answer "does this signal actually work?" Distinct from /scenario-builder, which projects forward; this looks backward.
---

# /signal-backtest

Validate a thesis pattern historically before sizing. Without this,
the thesis is just narrative.

## Inputs

- Signal definition: either a threshold rule
  (`{"metric": "rsi_14", "op": "<", "value": 30, "hold_days": 30}`)
  or a multi-condition rule (`{"and": [...]}`)
- Ticker universe: list of 5-20 tickers, OR a sector / theme
- Date range: typically 5-10 years for statistical significance

## Workflow

### 1. Define the signal precisely

Translate the thesis pattern into a structured rule. Examples:

**Technical:**
```json
{"metric": "rsi_14", "op": "<", "value": 30, "hold_days": 30}
```

**Fundamental:**
```json
{"and": [
  {"metric": "ttm_revenue_growth", "op": ">", "value": 20},
  {"metric": "fcf_yield", "op": ">", "value": 0.03}
], "hold_days": 252}
```

**Multi-condition with rebalance:**
```json
{"and": [
  {"metric": "operating_margin_yoy_change", "op": ">", "value": 200},
  {"metric": "pe_ratio", "op": "<", "value": 20}
], "hold_days": 126}
```

Available metrics depend on `agent/backtest_engine.compute_indicators()`.
If a needed metric isn't available, mark as `data_gap` and either fall
back to a closest proxy or skip the test.

### 2. Define the universe

Either:
- Explicit ticker list (5-20 names)
- Sector tag (will pull peer set via `get_company_peers` on a seed
  ticker)
- Theme tag (will pull via `get_industry_etfs` then take top holdings)

If running for `/equity-deep-research`, default universe = primary
ticker + 4 peers from Step 2 of the core skill.

### 3. Run the backtest

Call `mcp__nemo_financial__backtest_signal` with:
- The signal definition (dict)
- The ticker universe (list)
- Start date / end date
- Hold period (days)

The MCP tool returns:
- Aggregate stats: n_trades, hit_rate, mean_return, median_return,
  worst_trade, best_trade, max_drawdown, simplified Sharpe
- Per-trade list (if requested)

### 4. Interpret the results

- **hit_rate**: % of trades with positive return. > 55% on a
  multi-decade backtest is meaningful; 50% is noise.
- **mean vs median return**: large gap = fat-tailed distribution.
  The signal may be "wins big sometimes" rather than "wins consistently."
- **max_drawdown**: would the signal have been investable in practice?
  A 60% drawdown is fatal even if mean return is positive.
- **Sharpe**: > 0.5 is meaningful for a single-asset signal; > 1.0 is
  strong; > 2.0 is suspicious (likely overfit).
- **Sector dependence**: re-run on a different universe. If hit_rate
  drops significantly, the signal is sector-specific.

### 5. Output

```yaml
---
skill: signal-backtest
ticker: <null — universe-level>
verdict: <reliable / weak / noisy / overfit_suspect>
confidence: 0.0-1.0
key_finding: <one sentence on hit rate + holding period + caveats>
data_gaps: [<list>]
---

## /signal-backtest

**Signal**:
[rule definition as structured JSON]

**Universe** (N tickers):
[list]

**Date range**: YYYY-MM-DD to YYYY-MM-DD
**Hold period**: N days

**Results**:
- n_trades: X
- hit_rate: X% (target > 55%)
- mean return: X% per trade
- median return: X% per trade
- worst trade: -X%
- best trade: +X%
- max drawdown: -X%
- Sharpe (simplified): X.XX

**Verdict**:
- reliable signal (hit_rate > 55%, Sharpe > 0.5, max DD bearable)
- weak signal (mixed metrics)
- noisy / no edge (hit_rate ~50%, no statistical edge)
- overfit suspect (Sharpe > 2.0 on small N)

**Caveats**:
- Sector dependence: [tested / untested]
- Regime dependence: [bull only / works in drawdowns too]
- Data quality: [any survivorship bias, look-ahead bias, missing data]

**Implication for current thesis**:
[one paragraph: does the historical pattern support taking the trade
in the current setup, or does it suggest the pattern is unreliable?]
```

## Hard rules

- Do not interpret a single-digit n_trades as a useful backtest.
  Need >= 20 triggers for statistical signal.
- If hit_rate is > 70% AND Sharpe is > 2.0 on a backtest spanning <
  5 years, treat as overfit suspect. Real signals are rarely that
  clean.
- Do not extrapolate sector-A backtest results to sector-B without
  re-running. Signals are sector-dependent more often than not.
- If `backtest_signal` returns an error or empty trade list, mark
  `data_gap` and explain in the output why the backtest couldn't run
  (universe too small, date range too short, metric unavailable).

## When to invoke

- /equity-deep-research is testing a pattern thesis (insider cluster
  buying, margin expansion, RSI oversold) — invoke before sizing
- User asks "does this signal actually work" / "has this pattern
  historically worked"
- /scenario-builder is about to assign a probability to a base case
  that depends on a recurring pattern — back the probability with a
  backtest

## When to skip

- For idiosyncratic theses with no quantitative pattern (e.g., "MSFT
  has a moat in cloud") — backtest doesn't apply
- For one-off catalysts (FDA approval, merger close) — no historical
  base rate
- When < 5 tickers in the universe (not enough triggers)

## Save output

If invoked standalone:
`testing/fixtures/signal_backtest_<DESCRIPTION>_<DATE>.md`

If invoked from /equity-deep-research, return the verdict + implication
inline so the caller can fold it into the synthesis.
