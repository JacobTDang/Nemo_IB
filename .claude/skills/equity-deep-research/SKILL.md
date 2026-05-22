---
name: equity-deep-research
description: Multi-step equity research workflow for a public company, ticker, or investable theme. Produces a falsifiable investment thesis with valuation, expectations, catalysts, risk/reward, scenario analysis, positioning, and cross-company read-throughs. Use when the user asks for deep investment research, a thesis, a variant perception, a long/short view, or pattern matching across companies/industries. Skip for simple factual lookups.
---

# /equity-deep-research — Investment Thesis Workflow

Drive a full top-down to bottom-up equity research pass. The goal is not
a generic company report — the goal is a decision-ready, falsifiable
investment thesis with valuation, scenario math, red-team review, and
cross-company read-throughs.

Run all 19 steps in order (Step 0 may no-op). Steps 12, 14, 16,
and 17 delegate to companion skills; invoke them inline via the
Skill tool and fold their output into the final synthesis. Skipping
steps defeats the discipline.

## Step 0 — Reconcile prior same-day output

Before Step 1, check whether
`testing/fixtures/research_<TICKER>_<TODAY>.md` already exists. If
it does, you've previously run this skill or a sibling skill on the
same ticker today and produced a file. Read it and extract the
headline numerics that any analyst reading both files would expect
to agree:

- TTM revenue, gross margin, operating margin, EBITDA margin, FCF
- Forward consensus revenue and EPS estimates (FY+1, FY+2)
- Per-segment YoY growth rates (if a segment view is in scope)
- Key valuation multiples (P/E TTM, fwd P/E, EV/EBITDA)

If a number you produce on this pass differs by **> 2% relative**
from the prior-pass figure, the analyst is going to notice. Resolve
the divergence before continuing: identify which source is correct
(usually SEC > Finnhub > yfinance), use it, and document the
reconciliation in a `### Reconciliation with earlier pass` block at
the top of Step 1's output. State the prior figure, the new figure,
and which source you adopted and why.

Do NOT silently use a different number from a same-day prior pass.
If you produce different op margins (60.4% vs 64.0%) without
acknowledging it, the analyst's confidence in the system collapses.

If no prior file exists, this step is a no-op — proceed to Step 1.

## Step 1 — Probe and frame

State the research question in **one sentence**. Examples:
- "Is MSFT's Azure capex cycle producing real ROIC or is it 1999-style
  malinvestment?"
- "Is SK Hynix mispriced for the HBM3e cycle?"
- "Does the EOSE non-lithium chemistry have enough commercial traction
  to survive the cash bridge?"

Then declare:
- **Decision this research supports**: buy / avoid / short / monitor /
  update existing thesis / no_position candidate
- **Time horizon**: next quarter / next year / next 3 years / other
- **Primary ticker** (if applicable)
- **Theme / sector** the thesis sits in
- **Investment type**: long / short / pair trade / watchlist /
  no_position candidate

If the user only asked for a single metric, filing fact, price, or
news summary, **do not run this skill**. Use the narrowest tool
possible instead.

## Step 2 — Theme map and relevant tickers

Tools: `get_industry_etfs(theme)`, `get_company_peers(primary_ticker)`.

Goal: surface adjacent beneficiaries, losers, suppliers, customers,
and second-order exposures — not only the obvious ticker.

Output:
- Theme / sector
- Relevant ETFs and baskets
- Top concentrated holdings
- Unexpected exposures (the "second derivative" names)
- Primary beneficiaries
- Second-order beneficiaries
- Likely losers
- Substitutes
- Adjacent industries affected

## Step 3 — Macro regime

Tools: `get_macro_snapshot`, `get_treasury_yields`,
`get_credit_spreads`. Add `get_fred_series` only if a specific
indicator is thesis-critical.

Output (three to five sentences max):
- Rates / growth / inflation / credit / commodity backdrop
- Macro impact: tailwind / neutral / headwind
- The single most important macro variable for this specific thesis

Keep this compact. Macro informs the thesis but should not dominate
the memo.

## Step 4 — Industry structure and supply chain

Tools: `get_supply_chain(primary_ticker)`, `extract_8k_events` for
recent material disclosures. For each ticker of interest, classify
the named relationships:
- supplier / customer / competitor / substitute / bottleneck /
  distributor / regulator / capital provider

Output:
- Supply chain map
- Top suppliers, top customers, critical bottlenecks, competitors,
  substitutes
- First-order read-throughs
- Second-order read-throughs
- Potential hidden beneficiaries
- Potential hidden losers

Key question: if the main company wins, who else must win? If it
loses, who else is exposed?

## Step 5 — Business model and KPI driver tree

NEW required step. The handoff doc treats this as load-bearing — a
thesis that cannot identify its thesis-critical KPI is not ready.

Identify 3-7 metrics that actually move intrinsic value. By industry:

- **SaaS**: ARR growth, NRR, gross retention, CAC payback, gross
  margin, FCF margin, Rule of 40, SBC dilution
- **Semiconductors**: ASPs, utilization, wafer starts, inventory
  days, gross margin, capex intensity, backlog, advanced packaging
  capacity
- **Banks**: NIM, deposit beta, deposit growth, charge-offs, CET1,
  loan growth, securities losses
- **Retail**: comp sales, traffic, basket size, gross margin,
  inventory turns, shrink, store count
- **Energy / commodities**: realized price, production volume,
  lifting cost, decline rate, reserve replacement, capex intensity
- **Clean-tech / capex-heavy**: revenue ramp vs cash burn, gross
  margin trajectory, factory utilization, customer concentration,
  cash runway, dilution risk

Output:
- Business model classification
- Top 3-7 KPIs ranked by value impact
- Thesis-critical KPI (the single most important)
- KPI that would strengthen the thesis
- KPI that would break the thesis
- Leading indicators vs lagging indicators

## Step 6 — Segment fundamentals

Tools: `get_segment_financials(ticker)`,
`track_segment_growth(ticker)`.

For each reportable segment:
- Revenue growth YoY
- Multi-year CAGR
- Margin trend (expanding / compressing)
- Capex intensity
- Operating leverage signal
- Contribution to total company growth
- Relation to thesis-critical KPI from Step 5

Output:
- Segment table (segment, revenue growth, margin trend, thesis
  relevance, signal)
- Which segment the thesis lives in
- Acceleration vs deceleration signals
- Segment-level concerns

Goal: separate company-wide optics from the segment where the thesis
actually lives.

## Step 7 — Balance sheet and liquidity risk

NEW required step. A good business can still be a bad equity if the
balance sheet eats shareholders.

Tools: `get_market_data` (cash, debt, interest expense),
`get_working_capital`, `get_historical_fcf`, `get_financial_statements`
for maturity schedule details.

Required checks:
- Cash and equivalents
- Net debt
- Net debt / EBITDA
- Interest coverage
- Maturity wall
- Covenant risk
- Refinancing risk
- Cash burn rate
- Liquidity runway (months)
- Dilution risk (convertibles, warrants)
- Pension / lease / off-balance-sheet obligations if relevant

Output:
- Balance sheet status: strong / adequate / stretched / distressed
- Main balance sheet risk
- Refinancing window
- Dilution risk
- Liquidity runway

## Step 8 — Earnings quality and accounting checks

NEW required step. If adjusted numbers are materially better than
GAAP / cash numbers, the analysis must explain why.

Tools: `get_financial_statements` (compute cash conversion = OCF / net
income), `get_basic_financials` (Finnhub ratios),
`get_historical_fcf` (FCF vs reported earnings divergence).

Required checks:
- Cash conversion (OCF / NI)
- Accruals quality
- Revenue growth vs receivables growth
- Inventory growth vs sales growth
- Deferred revenue trend
- Capitalized expenses
- Restructuring addbacks
- Adjusted EBITDA quality
- SBC dilution
- One-time gains/losses
- Working capital drag
- Customer concentration
- Related-party transactions

Output:
- Earnings quality: high / acceptable / questionable / poor
- Main accounting concern
- Cash conversion
- SBC / dilution impact
- Working capital signal
- Adjustment quality

## Step 9 — Management and capital allocation

NEW required step. Does this management team deserve the benefit of
the doubt?

Tools: `get_insider_transactions`, `get_buyback_history`,
`extract_proxy_compensation`, `extract_governance_data`,
`calculate_capital_returns`.

Required checks:
- Guidance track record
- History of meeting / missing targets
- Capital allocation record (reinvestment returns, M&A,
  buybacks, dividends, debt repayment)
- Buybacks: accretive (below intrinsic value) or destructive
  (above)?
- SBC dilution rate
- Insider ownership level
- Insider buying / selling pattern — **MUST** cite
  `current_vs_baseline_ratio` from `get_insider_transactions`.
  Qualitative descriptors ("loud", "heavy", "significant",
  "concerning", "alarming") are permitted ONLY when the ratio
  exceeds 1.5x. Below 1.5x, describe activity as "in line with
  baseline" and treat it as non-signal. If the ratio is `None`
  (< 30 days of prior coverage), state `data_gap` and move on —
  do not improvise a qualitative claim from raw counts.
- CFO / COO / key technical role turnover
- Compensation incentive structure

Output:
- Management quality: excellent / good / mixed / poor
- Capital allocation: value creating / neutral / value destructive
- Guidance credibility
- Insider alignment
- Recent management changes
- Main concern

## Step 10 — Filing and call change detection

Tools: `diff_10k(ticker, item='1A')` and `diff_10k(ticker, item='7')`,
`extract_call_sentiment(ticker, quarters=4)`, `extract_mda`,
`extract_risk_factors`.

Filing diff checks:
- Item 1A risk factor changes (added / removed / modified)
- MD&A wording changes
- Liquidity language changes
- Customer concentration changes
- Legal proceeding changes
- Debt / covenant language
- Accounting policy changes
- Segment disclosure changes

Call sentiment checks:
- CEO tone
- CFO tone
- Analyst Q&A defensiveness
- Guidance confidence
- Capex language
- Demand language
- Pricing language
- Competition language

Output:
- New risks added
- Risks removed
- Management tone: improving / stable / deteriorating
- CFO signal (the single highest-value tonal read)
- Analyst Q&A pressure
- Key wording change

Company filings often reveal management concerns before consensus
catches up. Treat new disclosure changes as important signals, not
boilerplate.

## Step 11 — Positioning and crowding

Tools: `extract_13f_holdings`, `get_schedule_13d_filings`,
`get_short_interest`, `get_insider_transactions` (if not already pulled
in Step 9), `get_options_metrics` for implied vol / put-call.

Output:
- Ownership: insiders %, institutions %, institutions count
- Major holders (top 10)
- Recent accumulation / distribution pattern (QoQ % change)
- Activist setup (13D filer activity, repeat amendments)
- Short interest: % of float, days to cover, trend
- Crowding: crowded long / crowded short / under-owned / neutral
- Positioning risk
- Potential squeeze risk
- Potential forced-selling risk

Hard rule: do not skip positioning. Deep fundamental work can still
lose money if the setup is already crowded.

Hard rule (insider activity): the same baseline-ratio discipline from
Step 9 applies here. If Step 11 cites insider buying / selling, the
narrative MUST reference `current_vs_baseline_ratio`. Soft language
without the ratio is a Step 11 violation.

## Step 12 — Valuation and expectations

**DELEGATE to `/valuation-check`** companion skill via the Skill tool.

Pass the primary ticker and the peer set from Step 2. The companion
returns a structured valuation envelope (cheap / fair / expensive /
bubble-like / value trap) plus "what must be true for the current
price to work" and multiple compression risk.

Fold the result into the synthesis. Do not duplicate the work inline.

## Step 13 — Catalysts and timeline

Tools: `get_earnings_calendar(ticker)`, `extract_forward_signals`
(forward guidance / capex / roadmap language),
`get_analyst_revisions_history` (sell-side estimate trajectory).

Catalyst types to consider:
- earnings / guidance / investor day
- product launch / capacity addition
- regulatory decision / contract renewal
- M&A / activist deadline
- macro event / estimate revision
- index inclusion or removal

Output:
- Upcoming catalysts (top 5, ranked by impact)
- Catalyst type per item
- Timeline: near-term / medium-term / long-term
- Most important catalyst
- What would confirm the thesis
- What would disconfirm the thesis

A thesis without a catalyst can be correct but dead money.

## Step 14 — Cross-company read-through

**DELEGATE to `/cross-company-readthrough`** companion skill via the
Skill tool.

Pass the event / thesis frame and the primary ticker. The companion
returns first-order beneficiaries / losers, second-order
beneficiaries / losers, the market-likely interpretation, and the
underappreciated implication.

This is where investment edge actually lives — the news itself is
rarely the edge.

## Step 15 — Historical analogue and failure modes

Tool: `get_historical_analogue(thesis_description, top_n=3)`. Include
structural tags in the description (capex_peak, valuation_expansion,
supply_constrained, insider_selling, etc.) for better matches.

The tool now surfaces a structured `drawdown_pct`, `duration_months`,
and `direction` (`bear` / `bull` / `setup`) per match. Use these
fields to calibrate the bear case downstream; do not regex-parse
the body excerpt.

Output:
- Top analogues (named) — include each match's `drawdown_pct` and
  `direction` from the tool output verbatim
- Relevant similarities
- Important differences
- Failure modes from prior cycles
- Lessons that apply
- What would make this analogue invalid

**Hard rule (qualitative)**: historical analogues are hypothesis
generators, NOT proof. Do not say "this is just like 1999." Instead
say "this shares the following failure modes with prior capex
cycles: 1, 2, 3."

**Hard rule (calibration)**: if the top match is a `bear` analogue
with `drawdown_pct = D%` (negative), the bear price target produced
by Step 16 / surfaced in Section 6 / Section 8 MUST imply at least
`0.6 × |D|%` downside from current spot. Invoking 1999-Cisco
(D = -86%) with a -39% bear PT is a calibration failure — either
strengthen the bear PT (use ≥ -52%) or drop the analogue from
Section 15 and pick a softer match. The same rule applies upward to
`bull` analogues: a bull PT must imply at least `0.6 × D%` upside
when a bull analogue is invoked. `setup` analogues (no completed
move) have `drawdown_pct = None` and impose no calibration
constraint — but they also cannot be used to justify aggressive
bear or bull cases.

When Step 16 emits its scenario PTs, cross-check them against this
rule before continuing. If they fail, return to Step 16 with a note
that the bear / bull case needs strengthening, or downgrade the
Step 15 analogue selection.

## Step 16 — Scenario analysis

**DELEGATE to `/scenario-builder`** companion skill via the Skill
tool.

Pass the thesis, the KPI driver tree from Step 5, the segment data
from Step 6, and the valuation envelope from Step 12. The companion
returns bear / base / bull cases with probability weights and a
probability-weighted expected return.

The goal is not precision. The goal is to prevent a good story from
hiding bad risk / reward.

## Step 17 — Red-team the thesis

**DELEGATE to `/red-team-thesis`** companion skill via the Skill
tool.

Pass the full draft thesis (all prior steps). The companion returns
the strongest bear argument, what smart shorts would say, what
consensus already understands, where the thesis is overconfident, and
a recommendation: keep / reduce confidence / reduce size / no_position.

Hard rule: if the red team finds a stronger argument than the bull
thesis, downgrade confidence or output `no_position`.

## Step 18 — Portfolio, expectations, and kill-switch layer

For any thesis with sizing >= cautious (i.e., not `no_position`),
DELEGATE the following five companion skills **in parallel** via the
Skill tool. Each returns a structured envelope that gets folded into
the final synthesis.

1. **`/portfolio-fit`** — does this fit the current paper book?
   Returns: `strong / acceptable / weak / reject` + recommended max
   size + duplicate exposure flag.
2. **`/factor-exposure-check`** — is this stock-specific alpha or
   factor beta? Returns: `mostly alpha / partial alpha / mostly
   factor` + factor reversal risks.
3. **`/estimate-revision-watch`** — will sell-side analysts revise
   numbers on this thesis? Returns: predicted revision direction +
   sell-side-vs-data alignment.
4. **`/expectations-hurdle-check`** — what does the buyside whisper
   imply vs published consensus? Returns: setup classification
   (`easy / balanced / difficult`).
5. **`/thesis-kill-switch`** (pre-entry mode) — would any of the
   declared falsifiers trigger on day 1 with current data? Returns:
   per-falsifier status + kill-switch verdict.

Pass each companion the synthesis context built in Steps 1-17.

**Hard rules — these can override the upstream thesis sizing**:

- If `/portfolio-fit` returns `reject` → downgrade sizing to
  `watchlist` regardless of standalone thesis quality.
- If `/factor-exposure-check` returns `mostly factor` AND the variant
  perception is itself factor-directional (rather than stock-
  specific) → downgrade or output `no_position`. The thesis is just
  disguised beta.
- If `/thesis-kill-switch` reports ANY proposed falsifier would
  trigger on day 1 → do NOT enter. The thesis is born broken;
  rework the falsifier list or reframe the thesis.
- If `/expectations-hurdle-check` returns `difficult` AND
  `/estimate-revision-watch` predicts downward revisions → reduce
  size by at least 50%, even if the standalone thesis is strong.

## Step 19 — Final synthesis and thesis recording

Compose the final synthesis. Target ≤ 2 pages of markdown unless the
user explicitly asks for more.

Required sections (in this order):
1. **Thesis** — one sentence
2. **Decision** — buy / watchlist / avoid / short / no_position /
   update existing thesis
3. **Variant perception** — what does this view understand that
   consensus may be missing?
4. **Why now** — catalyst path and timing
5. **Bull case** — 3-5 cited factors. Every direct quote,
   statistic, or specific corporate claim ("the CEO said X", "the
   company guided to Y", "the 10-K discloses Z") must trace to a
   `rag_search` hit on the filing corpus (result's `doc_type`
   starts with `10K_` / `10Q_` / `8K_`, or equals `earnings_release`
   / `mda` / `risk_factors`). If the only supporting hit came from
   `get_company_news` or `search` (web), append the
   `[press-reported]` tag to that bullet so the reader knows it
   isn't filing-grounded. Unlabeled quotes are presumed
   filing-sourced and will fail the Verification (e) check.
6. **Bear case** — 3-5 cited factors. Same provenance rule as
   Section 5.
7. **Valuation / expectations** — from Step 12 companion output.
   If the upstream `/scenario-builder` envelope contains a
   `terminal_sensitivity` field, render it as a 3-row x N-column
   markdown table here (bear / base / bull across the five
   multiples). Any price target cited in Section 2 (Decision) or
   Section 14 (Position sizing) MUST reference a corresponding
   cell. Citing a price target without showing the sensitivity
   table is a Section 7 violation. If the field is absent because
   scenario DCF was run in perpetuity-only mode, state that
   explicitly.
8. **Scenario analysis** — from Step 16 companion output
9. **Positioning** — crowding, ownership, short interest, activist
   risk. Any insider-activity narrative MUST cite
   `current_vs_baseline_ratio` per the Step 9 / Step 11 rules.
10. **Portfolio fit** — from Step 18 `/portfolio-fit` companion output
11. **Factor exposure** — from Step 18 `/factor-exposure-check`.
    MUST contain the literal verdict tag from the companion envelope
    (`mostly_alpha` / `partial_alpha` / `mostly_factor`) on its own
    line, AND render the 6-row factor table verbatim (Market beta,
    Momentum, Theme beta, Rate sensitivity, Commodity sensitivity,
    USD strength). Paraphrased "exposure" or "correlation" paragraphs
    are not a substitute and will fail the Step 19 verification.
    Always include factor reversal risks for any factor classified
    `high`.
12. **Expectations setup** — from Step 18 `/expectations-hurdle-check`
    + `/estimate-revision-watch`; one paragraph synthesizing both
13. **Pre-entry kill-switch** — from Step 18 `/thesis-kill-switch`;
    per-falsifier status table
14. **Position sizing** — aggressive / normal / cautious / no_position
    (must honor any downgrade from Step 18 hard rules)
15. **Confidence** — 0.0 to 1.0
16. **Falsifiers** — REQUIRED. Specific observables that would force
    exit. Without these, the position cannot be stopped out rationally.
17. **Data gaps** — tools / sources that failed or were unavailable
18. **Next monitoring checklist** — what to check next and when

### Verification before save

Before writing the file or calling `record_thesis_evolution`, re-read
the synthesis once and apply each check below. Every check declares
a **PRECONDITION** — the check only fires when the precondition is
met. If the precondition isn't met, the check is *vacuously
satisfied*; do not invent the missing data just to make the check
pass. Heavy checks (a) and (f) further apply only when the verdict
is action-worthy (buy / short); for watchlist / no_position they
are vacuous because the data was never decision-relevant.

(a) **Terminal-sensitivity table in Section 7**
    - PRECONDITION: `calculate_scenario_dcf` ran with
      `terminal_multiple > 0` AND verdict ∈ {buy, short}
    - CHECK: Section 7 renders the `terminal_sensitivity` field as
      a markdown table (3 scenarios × N multiples). Any PT cited in
      Section 2 or 14 references a cell from this table.

(b) **Factor-exposure verdict tag in Section 11**
    - PRECONDITION: `/factor-exposure-check` was invoked
    - CHECK: Section 11 contains the literal verdict (`mostly_alpha`
      / `partial_alpha` / `mostly_factor`) on its own line AND
      renders the 6-row factor table verbatim.

(c) **Insider current_vs_baseline_ratio citation**
    - PRECONDITION: `get_insider_transactions` was called for this
      ticker
    - CHECK: every insider-activity narrative cites the ratio
      explicitly. Soft descriptors (loud, heavy, concerning) are
      only valid when the cited ratio exceeds 1.5x.

(d) **Semantic contradiction sweep**
    - PRECONDITION: always (this is the universal check)
    - CHECK: delegated to the red-team sub-agent in Step 19a. The
      sub-agent flags adjacent claims that argue opposite directions
      on the same KPI within the same horizon; pick one or reconcile
      explicitly.

(e) **Press-vs-filing provenance tagging**
    - PRECONDITION: Sections 5 or 6 contain a direct quote,
      executive statement, or specific corporate claim
    - CHECK: each such item is either (i) traceable to a
      `rag_search` hit with a filing-corpus `doc_type` (`10K_*`,
      `10Q_*`, `8K_*`, `earnings_release`, `mda`, `risk_factors`),
      or (ii) explicitly labeled `[press-reported]`. Unlabeled
      press-only quotes fail this check.

(f) **Historical-analogue calibration**
    - PRECONDITION: `get_historical_analogue` returned a `bear` or
      `bull` analogue with non-null `drawdown_pct` AND verdict ∈
      {buy, short}
    - CHECK: the bear / bull PT in Sections 6 and 8 implies at
      least `0.6 × |drawdown_pct|` move from spot in the analogue's
      direction. `setup` analogues impose no calibration but also
      cannot back aggressive cases.

If any check whose precondition is met fails, the synthesis is
incomplete. Patch in place before saving — silently writing an
under-specified report is a discipline failure, not graceful
degradation. If a check's precondition is unmet, move on.

Save the synthesis to:
`testing/fixtures/research_<TICKER>_<DATE>.md`

If a thesis row already exists for this ticker, call
`record_thesis_evolution(thesis_id, ...)` with the synthesis as the
initial evolution entry.

If no thesis row exists, save the synthesis file as the de-facto
thesis record. (Future: an analyst workflow should insert a thesis
row in `state.theses` and assign an ID; out of scope here.)

## Hard rules

- Do not run this skill for simple factual lookups. Use the narrowest
  tool possible.
- Do not skip positioning.
- Do not skip valuation.
- Do not skip falsifiers.
- Do not treat a great company as a great stock unless valuation
  supports it.
- If the evidence does not support a clear variant perception, output
  `no_position`.
- If exact MCP tools are unavailable, use the closest substitute or
  mark `data_gap`.
- Never fabricate missing tool outputs, filings, multiples, sentiment,
  ownership, insider activity, or short interest.
- Historical analogues are hypothesis generators, not proof.
- Insider selling alone is not bearish unless unusual, clustered, or
  paired with deteriorating fundamentals.
- News alone is not a thesis. Convert news into structured events and
  read-throughs.
- If scenario math shows poor expected value, downgrade the thesis
  even if the story sounds strong.
- If the red-team argument is stronger than the thesis, output
  `no_position` or reduce sizing.
- **Paper-only context**: never propose strategies that only make
  sense with real capital (dividend capture, tax-loss harvesting,
  wash-sale-driven structuring, real-money options writing). If a
  strategy is surfaced that only works with real capital, mark it
  "paper-only thought exercise" and do not size into it.
- **Time budget**: a full run should target 10-20 minutes wall
  clock. If any single tool call has been pending > 60 seconds, log
  the slow tool as `data_gap` and continue without it. Do not block
  synthesis on slow SEC fetches.
- **Step 18 overrides Step 17**: if `/portfolio-fit` returns `reject`
  OR `/factor-exposure-check` returns `mostly factor` AND variant
  perception is factor-directional, the final sizing must be
  downgraded regardless of how strong the standalone red-team-cleared
  thesis looks. Portfolio reality wins.

## Output discipline

- Final synthesis ≤ 2 pages of markdown unless the user asks for more.
- Every numerical claim must cite the tool / source that produced it.
- Use compact inline source tags:
  - `[tool_name: ticker/date]` (e.g., `[get_market_data: EOSE/2026-05-21]`)
  - `[filing: 10-K Item 1A FY2025]`
  - `[call_sentiment: last 4 quarters]`
  - `[valuation: peer_comps/2026-05-22]`
- If a number cannot be tied to a tool or source, remove it.
- Record failed tools or unavailable data under `data_gap`.
- Save synthesis to `testing/fixtures/research_<TICKER>_<DATE>.md`.
- If updating an existing thesis, call `record_thesis_evolution(thesis_id, ...)`.

## When to invoke vs skip

Invoke when the user:
- Names a ticker for analysis ("look at NVDA", "what about $AAPL")
- Asks for a thesis, recommendation, trade idea
- Asks for a pre-market brief, watchlist update, position review
- Asks for a variant perception or "what does this remind us of"

Skip when:
- The user only wants a single metric, filing fact, or news summary
- The query is purely about code or tooling
- The query is conversational
