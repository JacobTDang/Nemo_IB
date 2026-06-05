# Pre-Earnings Edge Plan — Tier 1 + Tier 2 (expectations-centric, sub-agent depth)

Goal: turn the pre-earnings pipeline from a flat sum of mostly-weak, sector-narrow
free signals into an **expectations-centric, all-sector research engine** that (a)
scores the *bar* not absolute performance, (b) goes layers deeper via parallel
sub-agents, and (c) leaves clean plug-in points for paid data (Tier 2) where the
real proprietary edge lives.

## The reframe (drives every phase)

```
DIRECTION  (will they clear the bar?)        beat / inline / miss lean
REACTION   (how does it move given a result?) asymmetry + sizing
VERDICT    = direction x asymmetry            EV lean + confidence + size
```

Positioning and reaction-history do NOT predict direction; they shape how a
surprise gets punished/rewarded. Keeping them separate is the core correctness fix.

## Non-negotiable design principles

1. **No hardcoding — everything derived at runtime.**
   - Peers/suppliers/customers from `get_company_peers` + `get_supply_chain` (never a curated list).
   - "Current-quarter window" inferred from the target's own earnings cadence
     (`get_earnings_surprises` history), never hardcoded dates.
   - The KPIs that matter derived dynamically from `get_segment_financials` +
     transcript Q&A frequency — never a per-company KPI map.
   - Guidance style derived from the company's *own* transcript history.
   - Sector handling derived from yfinance sector + data availability, not a
     company->bucket table.
   - Every phase has a no-hardcoding audit (grep + a test asserting no embedded
     ticker/company/peer lists).

2. **Depth without data fabrication.** Sub-agents return every numeric claim
   tagged with the tool that produced it (`sources_json`). The Layer-3 referee
   drops any claim it cannot trace to a tool. Preserves the existing
   "every quantitative claim cites a tool" rule across the agent boundary.

3. **Deterministic core, orchestrated edges.** Split each phase into:
   - **Python helpers** (pure, unit-testable: classification, scoring, aggregation, windows)
   - **Skill orchestration** (derives data via MCP tools, spawns sub-agents, persists) —
     verified by live runs, not unit tests.
   This makes "testing" honest: we red-green the math, live-verify the orchestration.

4. **Cost-gated depth.** Layer 0 (cheap structured signals) always runs. Layer 1+
   sub-agent fan-out runs only when a cheap pre-screen passes (near earnings,
   liquid, has peers/options). Sentry escalates, never blanket-fans-out.

5. **Honesty in the output.** Low coverage -> low confidence -> "no trade." The
   system should say "no edge" often and mean it.

## Edge sources (ranked, honest)

- Genuine, all-sector: **same-quarter peer readthrough**, **guidance archaeology
  (sandbag detection)**, **KPI-vs-consensus**, **revision velocity**.
- Genuine, narrow: supplier readthrough / Taiwan MOPS (hardware/semis only).
- Asymmetry (not direction): positioning, historical reaction profile, implied move.
- Marginal/confirmation only: Google Trends, capex, gov contracts, policy, FinBERT.
- Real proprietary edge (Tier 2, budget-gated): card panels, web traffic, app data,
  customs/shipping, foot traffic, expert networks.

Tier 1 = rigor + breadth + correct framing (free). Tier 2 = where new information
actually enters (costs money). Tier 1 is the scaffolding Tier 2 plugs into.

---

## Verification methodology (applies to EVERY phase)

In order, per phase:
1. **Red-green unit tests** for all Python helpers (no network). Confirm they fail
   against the bug/absent-logic, then implement, then pass.
2. **No-hardcoding audit**: a test that greps the new module/skill for embedded
   ticker symbols / company names / peer lists and asserts none; plus manual review.
3. **Citation audit** (orchestration phases): a live run where every number in the
   final writeup is traced to a tool; uncited claims must be dropped by the referee.
4. **Cross-sector live verification** on the matrix below — must include the
   previously-weak sectors (software/financial/healthcare) and prove coverage no
   longer collapses to mostly data_gap.
5. **Coverage assertion**: on a software name, direction-signal coverage >= 0.60 of
   applicable weight (the ORCL gap was 0.45).
6. **Full regression**: `pytest testing/ -m "not slow"` green.
7. **Commit gate**: commit only when 1-6 pass. One phase = one (or few) commits.

### Cross-sector test matrix (reused every phase)

| Sector | Ticker | Why |
|---|---|---|
| Software (prev. weak) | CRM / ORCL | proves expectations reframe fixes coverage |
| Semis/hardware | NVDA / AMD | supplier + peer readthrough both fire |
| Consumer | NKE / MCD | KPI = units/SSS; Trends/cards relevant |
| Financials (prev. weak) | JPM | guidance + peer; banking boundary |
| Healthcare (prev. weak) | PFE | KPI = drug sales; peer readthrough |
| Energy | XOM | commodity KPI |
| Industrials/Defense | LMT | supplier + gov |

Live spot-checks per phase use a 3-4 ticker subset spanning a previously-weak
sector + a hardware name + a consumer name. Full matrix in the final phase.

---

## Phase A — Persistence + freshness

**Python (testable).** New table `preearnings_research_layers`:
```sql
CREATE TABLE IF NOT EXISTS preearnings_research_layers (
  id             INTEGER PRIMARY KEY AUTOINCREMENT,
  ticker         TEXT NOT NULL,
  earnings_date  TEXT NOT NULL,
  layer          INTEGER NOT NULL,        -- 0..3
  component      TEXT NOT NULL,           -- peer_readthrough / guidance / kpi:<name> / positioning / reaction
  direction      TEXT,                    -- bullish/bearish/neutral/na
  magnitude      REAL,
  confidence     REAL,
  payload_json   TEXT NOT NULL,           -- structured findings
  sources_json   TEXT NOT NULL,           -- [{claim, tool}] for citation audit
  created_at     TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
```
State helpers in `state/preearnings.py`: `record_layer(...)`,
`get_layers(ticker, earnings_date)`, `latest_component(ticker, earnings_date, component)`,
`is_fresh(component, max_age_hours)` (enables 7d/3d/1d re-run reuse via upsert on
natural key `ticker+earnings_date+component`).

**Tests:** table creation (PRAGMA), roundtrip, upsert-on-natural-key, freshness/TTL
boundary, sources_json round-trips as list. No-hardcoding audit (schema has nothing
company-specific).

**Commit:** `add preearnings_research_layers persistence with freshness reuse`

---

## Phase B — Peer-readthrough fan-out (highest leverage, all sectors)

**Python helpers (testable)** in new `tools/preearnings/peer_logic.py`:
- `quarter_window(target_last_earnings, today)` -> (start, end) of the current
  reporting window, inferred from cadence — NO hardcoded dates.
- `reported_this_quarter(peer_report_date, window)` -> bool.
- `rank_peer_relevance(relationship)` -> weight (supplier/customer > direct
  competitor > adjacent), driven by the relationship type returned by tools.
- `aggregate_readthroughs(items)` -> net direction + magnitude + per-peer detail,
  relevance-weighted; handles zero-peer case cleanly.

**Skill orchestration (live-verified).** New `/peer-readthrough-fanout`:
1. Derive peer universe: union of `get_company_peers` + `get_supply_chain`. Dynamic.
2. For each peer, `get_earnings_surprises` -> did they report inside `quarter_window`?
3. For the top-N (relevance-ranked, N<=6 for cost) reported peers, **spawn one
   sub-agent each** running `/cross-company-readthrough` (peer -> target), in parallel.
4. Each sub-agent returns `{direction, magnitude, sources:[{claim,tool}]}`.
5. `aggregate_readthroughs` -> persist to layers (component=`peer_readthrough`).

**Sub-agent rules:** per-agent timeout; failed agent -> that peer = data_gap, continue;
citations required; cap N for cost.

**Tests:**
- Unit: window inference, reported-this-quarter classification (incl. edge: peer
  reports day before/after window bounds), relevance ranking, aggregation math,
  zero-reported-peers -> clean "no peer reads".
- No-hardcoding audit: peer_logic.py contains no ticker/company literals.
- Live: a semi/hardware name mid-season (>=1 readthrough with citations); a name
  with no recent peer prints (clean empty); a software name (proves peers derived,
  not hardcoded).
- Citation audit: every readthrough claim cites a tool.

**Commit:** `add peer-readthrough fan-out skill and deterministic peer logic`

---

## Phase C — Guidance archaeology + dynamic KPI drill-down

**Python helpers (testable)** in `tools/preearnings/expectations_logic.py`:
- `classify_guide_style(pairs)` where pairs = [{guided_high, guided_low, actual}]
  -> `sandbag` / `inline` / `aggressive` (e.g., actual repeatedly above guided high
  = sandbag). Derived from the company's own history.
- `bar_position(consensus, typical_guide)` -> consensus sits below/inside/above the
  company's usual guide -> easy/normal/hard bar.
- `rank_kpis(segments, qa_mentions)` -> the KPIs that matter, ranked by segment
  materiality + analyst-question frequency. NO hardcoded KPI map.
- `kpi_vs_consensus(kpi_trend, consensus)` -> direction per KPI.

**Skill orchestration (live-verified).** Extend the pipeline:
- **Guidance agent:** reads `get_earnings_transcripts` (4Q) + `extract_forward_signals`,
  extracts {guided, actual} pairs, calls `classify_guide_style`, compares to current
  `get_forward_estimates`. Persists component=`guidance`.
- **KPI fan-out:** `rank_kpis` from `get_segment_financials` + transcript Q&A; spawn
  one sub-agent per top KPI (<=3) to gather that KPI's trajectory vs consensus +
  alt-data read. Persist component=`kpi:<name>`.

**Tests:**
- Unit: guide-style classifier (synthetic sandbag/aggressive/inline series),
  bar_position thresholds, KPI ranking (materiality vs mention weighting),
  kpi_vs_consensus scoring.
- No-hardcoding audit: no KPI/company literals; KPIs come from inputs only.
- Live: software name -> KPIs surface (e.g., cloud growth, RPO) dynamically;
  consumer name -> different KPIs (units/SSS) surface. Proves dynamic identification.

**Commit:** `add guidance archaeology and dynamic KPI drill-down`

---

## Phase D — Positioning + historical reaction (feeds asymmetry)

**Python helpers (testable)** in `tools/preearnings/asymmetry_logic.py`:
- `reaction_profile(events)` where events = [{surprise_pct, next_day_return}] ->
  pattern: `beats_fade` / `sells_news` / `clean_repricer`, avg |move|, implied-move
  hit rate. Derived from the ticker's own history.
- `classify_positioning(short_interest, options_skew, oi)` -> crowded_long /
  crowded_short / neutral + squeeze_risk.
- `implied_vs_realized(implied_move, realized_history)` -> rich/cheap.

**Skill orchestration:** structured tool calls (`get_short_interest`,
`get_options_metrics`, `get_options_implied_move`, `extract_13f_holdings`,
`get_earnings_surprises` + `get_price_history`). No sub-agents needed. Persist
components `positioning`, `reaction`.

**Tests:** reaction-profile math (synthetic surprise/return series -> correct label),
positioning thresholds, implied-vs-realized. No-hardcoding audit. Live on a known
high-short name + a quiet name.

**Commit:** `add positioning and historical reaction-profile modules`

---

## Phase E — Synthesis rework: direction x asymmetry

**Python helpers (testable)** in `tools/preearnings/synthesis.py`:
- `direction_score(signals)` -> weighted lean over signals WITH data, renormalized
  by coverage; N/A weights redistributed (existing pattern).
- `coverage(signals)` and `agreement(signals)`.
- `asymmetry_adjust(direction, positioning, reaction, implied)` -> trims confidence
  and downsizes when positioning is crowded against the lean; flags
  "miss punishes harder".
- `final_verdict(...)` -> `{prediction, confidence, ev_lean, sizing}` with
  confidence reflecting coverage x agreement x asymmetry; honest floors.

**Weights (defaults; configurable):**
```
DIRECTION (sum 1.0)
  guidance ................... 0.24
  peer_readthrough ........... 0.22
  kpi_vs_consensus ........... 0.20
  revision_velocity .......... 0.14
  supplier/MOPS (if applic.) . 0.10
  thin_altdata (confirm) ..... 0.10
ASYMMETRY (modifies confidence + size)
  positioning, reaction_profile, implied_move
```
Tier 2 panel, when present, REPLACES the matching KPI/proxy weight.

**Tests:** direction scoring (agree/disagree), N/A redistribution, coverage penalty
(software w/o MOPS still >=0.60 coverage), asymmetry trim (crowded-long + beat ->
lower confidence/size), confidence floors and caps, sizing gates (no size into
>20% implied move, confidence<0.50 -> no trade). All deterministic.

**Commit:** `rework synthesis into direction x asymmetry model`

---

## Phase F — Orchestration, cost gating, Sentry, skill rewrite

**Python (testable):** `should_deep_research(days_to_earnings, liquidity, has_peers,
has_options)` gate.

**Skill rewrite:** `/preearnings-research` becomes the Layer-3 referee:
Layer 0 (always) -> gate -> Layer 1 fan-out (B, C) -> Layer 2 bull/bear adversarial
agents (each must cite Layer 0/1) -> Layer 3 synthesis (E) -> persist -> output.
Sentry: Layer 0 every tick; escalate to deep only on gated candidates.

**Tests:** gating unit tests; live end-to-end on software + consumer + hardware
proving depth, coverage, citations; verify far-dated/illiquid skips fan-out;
full regression.

**Commit:** `rewrite preearnings-research as layered direction x asymmetry orchestrator`

---

## Phase G — Tier 2 paid panels (budget-gated, one at a time)

Each is an MCP tool wrapping a vendor API, keyed by env var (like `CONGRESS_API_KEY`),
and SUPERSEDES the free proxy for its KPI when present. Order by edge-per-dollar /
accessibility:
1. **SimilarWeb** (web traffic) — most accessible API; supersedes Google Trends for
   digital names.
2. **Import/customs** (Panjiva/ImportGenius) — goods volumes.
3. **AlphaSense / Tegus** — expert calls / broker research, all sectors.
4. (institutional, later) card panels, app data, foot traffic.

**Per-vendor:** subprocess runner (like trends/finbert/options runners), envelope,
`@network`/key-gated tests, clean error when key absent. No-hardcoding (ticker/domain
derived). Mark requiring budget.

**Commit (per vendor):** `add <vendor> tool to nemo_altdata (tier 2)`

---

## What this does NOT do (honest)

- It does not manufacture data edge. Tier 1 is rigor/breadth/correct framing; the
  information ceiling rises only with Tier 2.
- Sub-agents multiply depth and parallelism, not proprietary information.
- Many runs will still say "no edge / no trade" — by design.

## Commit sequence

A persistence -> B peer fan-out -> C guidance+KPI -> D asymmetry -> E synthesis ->
F orchestration/gating -> G tier-2 (ongoing). PR to main after F; G lands
incrementally as budget allows.
