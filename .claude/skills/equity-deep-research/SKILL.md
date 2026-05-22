---
name: equity-deep-research
description: Multi-step equity research workflow for a public company, ticker, or investable theme. Produces a falsifiable investment thesis with valuation, expectations, catalysts, risk/reward, scenario analysis, positioning, and cross-company read-throughs. Use when the user asks for deep investment research, a thesis, a variant perception, a long/short view, or pattern matching across companies/industries. Skip for simple factual lookups.
---

# /equity-deep-research — Investment Thesis Workflow

Drive a full top-down to bottom-up equity research pass. The goal is not
a generic company report — the goal is a decision-ready, falsifiable
investment thesis with valuation, scenario math, red-team review, and
cross-company read-throughs.

Run all 18 steps in order. Steps 12, 14, 16, and 17 delegate to
companion skills; invoke them inline via the Skill tool and fold their
output into the final synthesis. Skipping steps defeats the discipline.

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
- Insider buying / selling pattern
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

Output:
- Top analogues (named)
- Relevant similarities
- Important differences
- Failure modes from prior cycles
- Lessons that apply
- What would make this analogue invalid

Hard rule: historical analogues are hypothesis generators, NOT proof.
Do not say "this is just like 1999." Instead say "this shares the
following failure modes with prior capex cycles: 1, 2, 3."

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

## Step 18 — Final synthesis and thesis recording

Compose the final synthesis. Target ≤ 2 pages of markdown unless the
user explicitly asks for more.

Required sections (in this order):
1. **Thesis** — one sentence
2. **Decision** — buy / watchlist / avoid / short / no_position /
   update existing thesis
3. **Variant perception** — what does this view understand that
   consensus may be missing?
4. **Why now** — catalyst path and timing
5. **Bull case** — 3-5 cited factors
6. **Bear case** — 3-5 cited factors
7. **Valuation / expectations** — from Step 12 companion output
8. **Scenario analysis** — from Step 16 companion output
9. **Positioning** — crowding, ownership, short interest, activist
   risk
10. **Position sizing** — aggressive / normal / cautious / no_position
11. **Confidence** — 0.0 to 1.0
12. **Falsifiers** — REQUIRED. Specific observables that would force
    exit. Without these, the position cannot be stopped out rationally.
13. **Data gaps** — tools / sources that failed or were unavailable
14. **Next monitoring checklist** — what to check next and when

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
