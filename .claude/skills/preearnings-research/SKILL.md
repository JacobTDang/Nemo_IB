# /preearnings-research

Run 2-10 days before earnings. An **expectations-centric, layered** research pass:
it scores DIRECTION (will they clear the bar?) and ASYMMETRY (how the stock
reacts given a result) separately, going deep via parallel sub-agents, and
synthesizes a verdict. Optionally places a gated paper bid.

Deterministic math lives in `tools/preearnings/` (unit-tested); this skill is the
orchestration. **Nothing is hardcoded** — peers, KPIs, guidance style, and the
quarter window are all derived at runtime.

## Inputs

- `ticker` (required)
- `mode`: `research_only` (default) | `auto_trade`
- `earnings_date_override`: YYYY-MM-DD (optional)

## Layer 0 — Always (cheap structured signals)

1. **Earnings date.** `get_earnings_calendar(from=today, to=today+45, symbol=ticker)` —
   always pass `symbol` so the single ticker's event is confirmed directly (never
   lost to the summary cap). No entry -> output `no_upcoming_earnings` and stop.
   Record `days_to_earnings` and the `hour` (amc/bmo). If `< 2`, add
   `WARNING: < 48h — IV crush risk`. Never fabricate a date.
2. **Revision velocity.** `get_analyst_revisions_history` + `get_forward_estimates`
   -> rising/flat/falling -> signal `revision_velocity`.
3. **Supplier / MOPS (if applicable).** `get_supply_chain`; if it contains TSMC
   (2330) / Foxconn (2317) / MediaTek (2454) / ASE (3711) call
   `get_taiwan_monthly_revenue`. Else mark signal `supplier_mops = na`.
4. **Thin alt-data (confirmation only).** `get_google_trends`, `get_capex_announcements`,
   `get_government_contracts`, `get_policy_signals`, `get_finbert_sentiment` (on
   `get_company_news`). Combine into one `thin_altdata` signal (majority lean).
5. **Asymmetry inputs.**
   - `get_short_interest` (SI % float, days-to-cover), `get_options_metrics`
     (IV skew AND put/call volume ratio), `get_options_implied_move(ticker, spot)`,
     `get_price_history` (3M momentum + ~500 daily bars), and reuse the Layer-0
     `get_analyst_revisions_history` pct_bullish.
   - **Reaction history:** `get_company_filings_history(ticker, form="8-K")` for
     report dates + `get_earnings_surprises` -> build pairs with
     `pair_surprises_with_reactions(surprises, event_dates, bars)` -> feed
     `reaction_profile`. Realized next-day |moves| also feed `implied_vs_realized`.

## Gate — should we go deep?

Compute `should_deep_research(days_to_earnings, liquid, has_peers, has_options)`
(`tools/preearnings/gating.py`). If `deep == false`, skip Layer 1/2 and synthesize
from Layer 0 only (mark the deep signals `data_gap`). This keeps the Sentry loop
cheap; deep fan-out runs only near earnings on liquid names.

Reuse: for each Layer 1 component, if `is_fresh(ticker, earnings_date, component,
max_age_hours=24)` skip recompute and load via `latest_component`.

## Layer 1 — Deep fan-out (parallel sub-agents)

6. **Peer readthrough.** Invoke `/peer-readthrough-fanout(ticker, earnings_date)`.
   It derives peers dynamically, fans out one `/cross-company-readthrough` per
   reported peer, aggregates, and persists. -> signal `peer_readthrough`.
7. **Guidance archaeology.** Spawn a sub-agent: read `get_earnings_transcripts`
   (last 4Q) + `extract_forward_signals`, extract `{guided_low, guided_high,
   actual}` per quarter, then apply `classify_guide_style` + `bar_position` +
   `guidance_direction` (vs current `get_forward_estimates`). -> signal `guidance`.
8. **KPI drill-down.** Build candidates from `get_segment_financials` (materiality)
   + transcript Q&A mention counts; `rank_kpis` -> top 3. Spawn one sub-agent per
   KPI to get its trajectory vs consensus (`kpi_vs_consensus`). Aggregate the KPI
   directions -> signal `kpi_vs_consensus`.

Every sub-agent returns each number tagged `{claim, tool}`. The referee (this
skill) drops any uncited claim. Persist each component with `record_layer(...)`.

### Sub-agent prompt rules (canonical — do not improvise)

Build every sub-agent prompt from the templates below. `{PLACEHOLDERS}` may be
filled ONLY with runtime tool outputs (tickers, dates, surprise lines,
relationship types as returned by tools). **Never inject company facts from
memory** — no executive names, product names, fiscal calendars, guidance
practices, or relationship characterizations beyond the tool-provided type.
Every template ends with the same contract: STRICT JSON, every number cited
`{claim, tool}` or omitted, agent failure -> that component is data_gap.

**Guidance archaeology template:**
```
Guidance archaeology for {TICKER} ahead of {EARNINGS_DATE}.
Do NOT assume the company's guidance practice, fiscal calendar, executives, or
metric names from memory — discover everything via tools.
1. get_earnings_transcripts({TICKER}): search the last 4 quarters for explicit
   forward guidance (EPS / revenue / segment growth ranges) given by management.
   For each prior quarter with a guide, pair it with the actual
   (get_earnings_surprises) -> {guided_low, guided_high, actual}.
2. get_forward_estimates({TICKER}) for current consensus.
Classify guide_style ONLY from pairs you actually found (0 pairs -> "unknown");
bar_position = consensus vs the upcoming quarter's guide if you found one.
Return STRICT JSON: {"component":"guidance","guide_style":...,"bar_position":...,
"direction":...,"pairs_found":N,"key_finding":...,"sources":[{claim,tool}]}.
```

**KPI drill-down template:**
```
Dynamic KPI drill-down for {TICKER} ahead of {EARNINGS_DATE}.
Derive the 2-3 KPIs the market trades for THIS company. Candidates come ONLY
from (a) get_segment_financials materiality and (b) metrics analysts repeatedly
raise in get_earnings_transcripts Q&A. Do not anchor on any example KPIs and do
not assume KPIs from memory.
For each top KPI: trajectory from tool data vs any consensus/guide anchor you
can cite (get_forward_estimates / transcript guidance). No citable anchor ->
that KPI is neutral with note "no anchor".
Return STRICT JSON: {"component":"kpi","kpis":[{name,direction,evidence}],
"direction":...,"magnitude":...,"key_finding":...,"sources":[{claim,tool}]}.
```

(The peer-readthrough template lives in /peer-readthrough-fanout. Bull/bear
agents receive the Layer 0/1 evidence verbatim and may use ONLY that evidence
plus their own tool calls — never memory facts.)

## Layer 2 — Adversarial (parallel)

9. Spawn two sub-agents:
   - **Bull case:** strongest argument the target beats the bar, citing Layer 0/1.
   - **Bear case:** strongest argument it misses, citing Layer 0/1.
   Neither may introduce uncited numbers. Use their tension to sanity-check the
   direction lean and surface the decisive swing factor.
   *Budget-constrained mode:* the referee may run this adversarial pass itself
   (no extra agents) provided the bull and bear cases are still written out
   separately with citations and the swing factor is named.

## Layer 3 — Synthesize (referee)

10. Assemble the DIRECTION signals as
    `[{name, direction, magnitude}]` for: `guidance`, `peer_readthrough`,
    `kpi_vs_consensus`, `revision_velocity`, `supplier_mops`, `thin_altdata`
    (use `na` where not applicable, `data_gap` where unobserved).

    Compute asymmetry: `classify_positioning(short_interest, days_to_cover, skew,
    put_call_volume_ratio, momentum_3m_pct, analyst_pct_bullish)` — crowded_long
    needs >=2 independent pieces of evidence; `reaction_profile(pairs from
    pair_surprises_with_reactions)`; `implied_vs_realized(implied, realized
    next-day |moves|)`.

    Call `final_verdict(signals, positioning=..., squeeze_risk=..., reaction_pattern=...,
    implied_verdict=..., implied_move_pct=...)` (`tools/preearnings/synthesis.py`).
    It returns `prediction`, `direction_score`, `coverage`, `agreement`,
    `confidence`, `sizing`, `low_confidence`, and `asymmetry_notes`.

11. **Persist:** `record_layer(layer=3, component="synthesis", ...)` and
    `record_eval(ticker, earnings_date, prediction, confidence, implied_move_pct)`.
    **Save:** `testing/fixtures/preearnings_{TICKER}_{DATE}.md`.

## Output

```yaml
---
skill: preearnings-research
ticker: {TICKER}
earnings_date: {DATE}
days_to_earnings: {N}
prediction: likely_beat | in_line | likely_miss
confidence: 0.XX
direction_score: {X.XX}
coverage: {X.XX}
implied_move_pct: {X.XX}
sizing: normal | cautious | no_position
low_confidence: true | false
---
```

Then: (1) DIRECTION signal table (name, direction, magnitude, weight), (2) the
decisive bull vs bear swing factor, (3) ASYMMETRY read (positioning, reaction
pattern, implied vs realized) and what it does to sizing, (4) data gaps,
(5) top cited evidence with the tool that produced each number.

## Trading decision (mode=auto_trade only)

1. `sentry_can_act(action_type=new_position)`; stop if not permitted.
2. Side from `prediction` + `sizing` (`likely_beat`+size!=no_position -> buy;
   `likely_miss`+size!=no_position -> sell; else no trade).
3. `get_paper_account()`; max 1% equity for an earnings trade.
4. `risk_check_proposed_trade(...)` -> only on `approve` -> `place_paper_order(...)`
   with any `adjusted_quantity`. Record sentry actions. If rejected, report and stop.

## Hard rules

- Never fabricate an earnings date — only `get_earnings_calendar` output.
- Every number cites the tool that produced it; uncited sub-agent claims dropped.
- Derive peers/KPIs/guidance from tools — no hardcoded company/peer/KPI lists.
- `coverage < 0.5` or `confidence < 0.5` -> `low_confidence`, no trade.
- Never trade if `implied_move_pct > 0.20`, `confidence < 0.55`, or `days_to_earnings < 2`.
- Always `risk_check_proposed_trade` before `place_paper_order`. Max 1% per earnings trade.
- One failed sub-agent never fails the run — mark that component `data_gap` and continue.

## When to invoke

- "pre-earnings research on X" / "earnings research X"
- Sentry candidate with `triggered_by=pre_earnings`; run 7/3/1 days out on watchlist names.
