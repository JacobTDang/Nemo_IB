<!--
External audit, recorded verbatim. Produced by Codex (GPT-5.6) on 2026-08-25
against the state of the repository at that date. Not reviewed or acted on;
kept as a work list rather than as a description of current behaviour.

Read the counts with care: it reports the README claiming 5 servers / 76 tools
against an implementation declaring 96. The README has since been rewritten and
its inventory is now measured against a running instance.
-->

# Codex-Found Issues and Data Gaps

> Audit note: Codex identified the following faults, inconsistencies, and data gaps while reviewing the Nemo_IB repository, with emphasis on `tools/`, the MCP servers, the homelab deployment, and the proposed future financial-automation layer.
>
> This document is an engineering audit, not financial advice. The current intended deployment is a private homelab service used primarily to poll and aggregate financial data. Storage, report history, and portfolio accounting may be implemented as separate services. Trading findings below are future-readiness requirements and do not block a read-only aggregation deployment.

## Executive summary

Nemo is currently best classified as:

- A broad financial-research and data-aggregation service with useful SEC, market, macro, news, alternative-data, and modeling coverage.
- A reasonable homelab MCP data backend, provided its outputs are treated according to their source, timestamp, freshness, and coverage limitations.
- A supervised paper-trading experiment, but not yet a safe unattended trading system.
- Not ready for autonomous real-money portfolio management.

The most important current aggregation gaps are inconsistent response contracts, incomplete freshness and provenance metadata, uneven failure semantics, and insufficient cross-source reconciliation. The most important future trading gaps are missing fill processing, reliance on stale local portfolio state, incomplete risk enforcement, and lack of broker-resident protection.

## Repository inventory discrepancies

The README is out of sync with the implementation.

- The README describes 5 MCP servers and 76 tools.
- The five original research servers currently declare 96 tools:
  - `nemo_web`: 51
  - `nemo_financial`: 20
  - `nemo_finnhub`: 14
  - `nemo_fred`: 5
  - `nemo_alpaca`: 6
- Additional Python MCP servers declare:
  - `nemo_altdata`: 9
  - `nemo_sentry`: 19
  - Excel: 3
- The Slack MCP server declares 2 tools.
- The README describes 15 skills, while `.claude/skills/` contains 20 skill directories.
- The README refers to `CLAUDE.md`, but that file was not present in the reviewed checkout.
- The health-check skill checks the older five-server set and omits altdata and Sentry.
- The remote Docker stack deploys web/SEC, financial, Finnhub, FRED, and altdata; it does not deploy the same five-server set described by the local health skill.

These differences can cause an operator or agent to believe a deployment is healthy while newer services are absent or untested.

## Current homelab data-aggregation findings

### 1. No universal freshness contract

Some tools return `as_of`, timestamps, freshness warnings, or coverage fields, while others return values without enough temporal context. An agent can therefore compare observations from different periods as if they were simultaneous.

Every data response should eventually expose a common minimum envelope such as:

```json
{
  "success": true,
  "data": {},
  "provider": "SEC",
  "source_url": null,
  "retrieved_at": "2026-08-25T14:00:00Z",
  "data_as_of": "2026-06-30",
  "freshness": "current",
  "coverage": "full",
  "cached": false,
  "warnings": [],
  "error": null,
  "request_id": "...",
  "schema_version": "1"
}
```

At minimum, an external agent must be able to distinguish retrieval time from the period represented by the data.

### 2. Response formats and failure semantics are inconsistent

Newer servers use structured envelopes, but older tools return bespoke JSON shapes. The meaning of an empty array, `None`, missing field, error string, and successful zero result is not uniform across the tool catalog.

This creates a major agent-consumption risk: "no result exists" can be confused with "the provider could not be queried" or "this issuer is not covered."

The altdata convention is the strongest current pattern:

- `success`: whether the lookup completed
- `coverage`: `full`, `partial`, or `not_covered`
- `reason`: machine-readable failure cause
- `degraded`: named degradations

That convention should be adopted across the remaining servers.

### 3. Provider provenance is not consistently explicit

Nemo aggregates data from SEC EDGAR, yfinance/Yahoo, Finnhub, FRED, SearxNG, FinMind, Congress.gov, House and Senate disclosures, government-contracting APIs, RSS, and GDELT. Outputs do not always make the original provider obvious.

Every observation should name its upstream provider and, where applicable:

- Filing accession number
- Source URL
- Reporting period
- Exchange or quote feed
- Publication timestamp
- Query parameters used
- Whether the value was derived or directly reported

Tool name alone is not sufficient provenance.

### 4. Cached versus live behavior is not always visible

The project has several cache layers, but callers cannot always determine:

- Whether a response came from cache
- When the cache entry was created
- Its configured TTL
- Whether a live query failed and stale data was returned
- Whether a fallback provider was used

Returning stale cached information can be appropriate for research, but it must be labeled so an agent can refuse time-sensitive decisions.

### 5. Cross-source reconciliation is incomplete

Important values should have optional reconciliation across independent sources, especially:

- Ticker and issuer identity
- Share count
- Market capitalization
- Earnings dates
- Corporate actions
- Forward revenue and EPS consensus
- Insider transactions
- Price used in a valuation

The repository has useful one-off reconciliation tests, but no general service-level reconciliation contract. Where two sources disagree, the result should identify the difference rather than silently select one.

### 6. No consolidated quote snapshot contract

The financial tools are useful for daily research, but Nemo lacks a dedicated quote response consistently containing:

- Latest trade
- Bid and ask
- Spread
- Quote timestamp
- Market session
- Previous close
- Current daily bar
- Provider/feed identifier
- Real-time versus delayed status
- Staleness assessment

This is valuable even before any execution feature is enabled. A report-generating agent must distinguish a current quote from the last daily close or an after-hours fallback.

### 7. Data limitations are documented but not always machine-readable

Known limitations include:

- Short interest is normally 2-3 weeks stale.
- There is no paid/live short-interest alternative such as Ortex.
- Options quotes can be stale after hours.
- Illiquid strikes may produce invalid implied-volatility sentinel values.
- Google Trends coverage is unreliable because of rate limiting.
- The tool named `get_earnings_transcripts` may return earnings releases rather than call transcripts.
- Finnhub earnings-calendar dates can be stale or incorrect.
- Earnings evaluation may attribute a price move to EPS when guidance was the actual driver.
- There is no KPI-level consensus source.
- Web traffic, shipment, card-spending, and similar paid alternative-data sources are absent.
- Debt-maturity extraction has materially incomplete coverage; `not_covered` must never be interpreted as no maturities.
- Litigation extraction may return only a cross-reference to a financial-statement note.
- Some nonstandard 10-K layouts defeat section extraction.
- Forward-signal extraction can surface safe-harbor or risk-factor boilerplate rather than informative guidance.

These conditions should be returned as structured warnings, not left solely in documentation.

### 8. Source quality varies materially

The system mixes primary sources with convenience aggregators:

- SEC and FRED are generally authoritative for their covered fields.
- Finnhub is useful but can have calendar and coverage issues.
- yfinance is convenient for research but is not an execution-grade consolidated market-data feed.
- Search and scraped web results require source-quality scoring and citation preservation.
- Options, short-interest, and consensus data require particular care because freshness and entitlements vary.

Agents should rank primary sources above aggregators and should not silently replace a failed authoritative lookup with an unlabeled lower-quality source.

### 9. Historical persistence should remain a separate responsibility

Keeping polling and storage separate is a sound design, but Nemo responses should include stable identifiers so an external collector can deduplicate and reproduce observations:

- Provider
- Tool name and schema version
- Symbol or entity identifier
- Observation timestamp
- Reporting period
- Filing accession or document identifier
- Request ID
- Cache status
- Hash of raw content where appropriate

The external storage layer should preserve raw immutable responses separately from derived reports.

### 10. Provider-level readiness is weaker than process-level health

`/health` establishes that an HTTP process is alive, not that each upstream provider is usable. A production homelab deployment should distinguish:

- Container alive
- MCP initialization successful
- Credentials present
- Upstream provider reachable
- Rate limit not exhausted
- Cache/database writable where required
- Last successful query time
- Degraded or stale-only mode

The existing Claude health skill is also stale relative to the current server inventory.

### 11. Long-lived MCP reliability has known fragility

The repository documents historical stdio hangs and currently mitigates Alpaca calls with timeouts. Similar protection is not necessarily uniform across every provider call. Remote Streamable HTTP is the appropriate homelab transport, but each tool still needs bounded upstream timeouts and predictable cancellation behavior.

### 12. Security and deployment boundaries

The HTTP wrapper correctly supports bearer authentication and refuses to run silently open. Additional operational recommendations are:

- Bind services only to a private interface, tailnet, or reverse proxy.
- Do not expose ports `8810-8814` directly to the public internet.
- Keep `MCP_AUTH_TOKEN` out of repository files and logs.
- Disable response-body logging when payloads could include sensitive portfolio or credential-adjacent data.
- Use separate credentials per environment where providers permit it.
- Keep read-only research MCP servers separate from any future order-placement service.

## Suggested aggregation architecture

The intended separation should be preserved:

```text
Financial providers
        |
        v
Nemo read-only MCP aggregation services
        |
        +--> External scheduler/collector
        |         |
        |         +--> Raw immutable observations
        |         +--> Normalized database
        |         +--> Reports and report history
        |
        +--> Claude Code, Codex, or other research agents

Future separate boundary:
Decision/risk service --> narrowly scoped paper broker executor
```

Research and reporting agents should never need broker order-placement credentials.

## Second-pass data coverage audit

Codex performed a second-pass review against the broader data lifecycle required for equity research: entity identity, point-in-time history, market data, fundamentals, expectations, events, credit, ownership, positioning, macro, cross-asset, and alternative data. The following gaps are additional to the consistency and provenance findings above.

### 1. Security master and identifier history

Nemo is predominantly ticker-centric and does not contain a complete security master. Missing fields and histories include:

- CUSIP, ISIN, FIGI, and LEI
- Exchange and primary listing venue
- Primary versus secondary listings
- ADR-to-local-share mapping and ADR ratio
- Share-class relationships
- Historical ticker and exchange changes
- Merger, delisting, bankruptcy, and spin-off lineage
- Active/inactive security status
- Trading currency and reporting currency
- Country of incorporation and domicile
- Fiscal year-end
- Security type: common, preferred, ADR, ETF, warrant, unit, and similar instruments

Tickers are not permanent identifiers. Without a security master, long-lived polling and historical analysis can lose delisted companies, join the wrong security after a symbol reuse, confuse ADRs with local shares, or mishandle multi-class issuers.

### 2. Point-in-time historical data

Most current tools retrieve the latest available view rather than the view that was available on a requested historical date. This affects:

- Financial statements and later restatements
- Analyst estimates and recommendations
- Earnings dates that were subsequently changed
- Corporate guidance
- ETF holdings and index membership
- Industry classification
- Shares outstanding and float
- Company profiles

The separate storage service should append every poll rather than overwrite prior values. Each observation should preserve both `retrieved_at` and `data_as_of`, as well as the provider's original publication or filing date.

Without point-in-time history, backtests are exposed to look-ahead bias and survivorship bias.

### 3. Filing amendments and restatement history

Several SEC readers intentionally exclude amendments so a partial amendment is not mistaken for a complete annual report. That is appropriate for current-value extraction, but Nemo lacks a dedicated restatement product covering:

- Original and amended filings
- Original and restated values
- Fields and periods changed
- Restatement reason and materiality
- Item 4.02 non-reliance events
- Auditor changes
- Internal-control weaknesses
- Filing dates and accession numbers

Detecting a restatement event is not equivalent to reconciling its numerical impact.

### 4. Intraday and execution-quality market data

Daily yfinance history is useful for research, but the aggregation layer lacks a complete intraday market dataset:

- Intraday OHLCV bars
- Tick-level trades
- Bid and ask quotes
- NBBO or exact feed identity
- Historical spreads
- Consolidated volume
- Premarket and after-hours sessions
- Auction imbalances
- Trading halts and LULD state
- Exchange timestamps

Even without trading, this data is necessary to establish whether a filing or news event preceded a market move and how quickly the market incorporated it.

### 5. Complete corporate-action lifecycle

Dividends and splits are covered, but the following are absent or incomplete:

- Dividend declaration, ex-date, record date, and payment date
- Special dividends and return-of-capital distributions
- Spin-offs and distribution ratios
- Merger consideration and closing status
- Tender and exchange offers
- Rights offerings
- Reverse splits
- Symbol and exchange changes
- Bankruptcy and liquidation events
- Warrant redemptions
- ADR ratio changes

Adjusted historical prices alone are insufficient because they hide the event that caused the adjustment.

### 6. Canonical normalized quarterly fundamentals

Nemo can retrieve quarterly and annual statements, but it does not yet provide one canonical normalized statement with consistent fiscal-period identity and provenance. Missing or weak areas include:

- Consistent TTM calculations
- Fiscal-quarter and fiscal-year mapping
- Original versus restated figures
- Continuing versus discontinued operations
- Organic versus acquired growth
- Constant-currency versus reported growth
- Acquisition contributions
- Non-GAAP reconciliations
- One-time charges and restructuring costs
- Minority interest and preferred dividends
- Pension obligations
- Deferred tax assets and valuation allowances
- Off-balance-sheet commitments
- Guarantees and contingencies
- Deferred revenue and remaining performance obligations across issuers

The targeted SEC extractors are useful, but they do not replace a canonical point-in-time financial statement layer.

### 7. Bank-specific fundamentals

The code correctly recognizes that EBITDA is not meaningful for banks, but it does not provide an equivalent structured bank dataset. Missing bank metrics include:

- Net interest income and net interest margin
- Deposit balances and deposit beta
- Loan growth and loan mix
- Provisions for credit losses
- Net charge-offs
- Nonperforming assets
- CET1 and other regulatory-capital ratios
- Tangible book value
- Efficiency ratio
- Securities portfolio duration and unrealized losses
- Commercial real-estate exposure
- Uninsured deposits

Bank coverage is therefore structurally weaker than coverage for conventional industrial companies.

### 8. Other sector-specific operating KPIs

Structured historical coverage is also missing for many industry-specific metrics:

- Insurance: combined ratio, premiums, reserve development, catastrophe exposure
- REITs: FFO/AFFO, occupancy, same-store NOI, lease expirations, cap rates
- Biotech: clinical endpoints, enrollment, FDA milestones, approval calendar, cash runway
- Energy: production, realized pricing, reserves, decline rates, hedge books
- Mining: production, grades, reserves, sustaining costs
- SaaS: ARR, NRR, RPO, billings, seats, churn
- Semiconductors: ASPs, wafer starts, utilization, inventory channels
- Retail: same-store sales, traffic, ticket, inventory, store counts
- Airlines: capacity, load factors, yields, fuel cost
- Autos: deliveries, incentives, inventory, pricing

Some of these values appear in filings, but they are not normalized into comparable time series.

### 9. Point-in-time sell-side consensus

Current forward estimates expose averages, highs, lows, and analyst counts, but Nemo lacks a robust historical consensus dataset containing:

- Consensus as it existed on each date
- Individual analyst and broker estimates
- Revision timestamps
- Median, dispersion, and outlier treatment
- Stale-estimate identification
- Correct fiscal-period mapping
- Consensus immediately before earnings
- KPI-level estimates
- Guidance consensus

The recommendation-history tool tracks rating-bucket changes, not the full history of EPS, revenue, EBITDA, or KPI estimate revisions.

### 10. Genuine earnings-call transcripts

The current transcript path can return press releases rather than actual call transcripts. Missing transcript capabilities include:

- Prepared remarks and Q&A separation
- Speaker attribution
- Analyst identity
- Question-topic classification
- Management non-answer or evasion detection
- Quarter-over-quarter wording comparisons
- Original transcript provider and publication timestamp

Q&A is often more informative than the earnings release, so substituting a press release must remain a clearly labeled degradation.

### 11. Unified company and macro event calendar

There is partial support for earnings, IPOs, ex-dividend dates, and macro catalysts, but no single verified calendar covering:

- Confirmed versus estimated earnings dates
- BMO, AMC, or intraday earnings timing
- Earnings-date change history
- Investor days
- Conferences and presentations
- Product launches
- Regulatory decisions and FDA milestones
- Shareholder votes
- Lockup expirations
- Debt maturities and refinancing dates
- Dividend dates
- Index additions, deletions, and rebalances
- Option expiration
- Economic releases
- Central-bank decisions and speeches
- Treasury auctions

Each event should carry its source, confidence, last verification time, and change history.

### 12. Tradable debt and credit-market information

Nemo calculates accounting-based credit ratios and extracts some debt maturities, but it lacks security-level credit data:

- Bond identifiers
- Coupon, maturity, seniority, and collateral
- Bond price and yield
- Yield to maturity and spread
- New issuance
- Credit ratings, outlooks, and rating changes
- CDS spreads
- Revolver availability
- Covenant levels and headroom
- Tender and exchange offers
- Refinancing activity
- Secured versus unsecured capital structure

Credit markets can deteriorate before equity estimates respond, making this a significant research gap for leveraged issuers.

### 13. Dilution and capital-markets events

Share-count and shelf-activity tools exist, but a fuller capital-markets layer would include:

- At-the-market program usage
- Registered direct offerings
- PIPEs
- Convertible debt and conversion prices
- Convertible and warrant dilution
- Employee option overhang
- Earn-outs and contingent consideration
- Preferred securities
- Buyback authorization versus actual execution

### 14. Complete institutional ownership history

Current tools cover selected hedge funds and vendor-aggregated holders, but not a full point-in-time ownership history containing:

- All institutional filers
- Original and amended 13F filings
- Manager-level position changes
- Ownership concentration
- New and exited holders
- Manager style classification
- Shares versus put/call positions
- Cross-manager crowding

Although 13F data is delayed, systematic point-in-time history remains valuable for ownership and crowding analysis.

### 15. ETF and fund flows

Nemo can inspect selected ETF holdings but lacks:

- Daily ETF creations and redemptions
- Mutual-fund and ETF flow history
- Index rebalance schedules
- Upcoming index constituent changes
- Historical holding weights
- Forced-buying and forced-selling estimates

### 16. Short-positioning and securities-lending data

Current short interest is delayed by design. Additional missing information includes:

- Daily short volume
- Securities-lending utilization
- Borrow availability and borrow fee
- Days-to-cover history
- Threshold-security status
- Failures to deliver
- Reg SHO and short-sale restriction status
- Short-interest revision history

This generally requires a paid or specialized feed for timely coverage.

### 17. Options positioning beyond summary metrics

The existing options tool provides useful IV, skew, implied move, volume, and open-interest summaries. It does not provide:

- Historical full-chain snapshots
- Unusual option flow
- Opening versus closing trades
- Dealer gamma exposure
- Delta-adjusted positioning
- Zero-DTE activity
- A complete volatility surface
- IV rank and percentile
- Realized-versus-implied history
- Earnings-event IV-crush history
- Block and sweep classification

### 18. FX, commodities, and cross-asset prices

The research workflow reasons about USD, rates, and commodity sensitivity but lacks dedicated structured tools for:

- FX spot and forward rates
- Trade-weighted dollar indices
- Oil, natural gas, copper, gold, uranium, and lithium prices
- Commodity futures curves
- Crack spreads
- Freight rates
- Regional electricity and power prices
- Carbon credits

An exposure classification is only a hypothesis unless it can be joined to the corresponding underlying price series.

### 19. Macro calendars and vintage data

FRED coverage is valuable, but a complete macro layer would also track:

- Economic-release calendars
- Consensus, actual, and surprise values
- Subsequent revisions
- Historical data vintages
- Surprise indices
- Central-bank speeches and decisions
- Treasury auctions
- Yield curves by observation time
- Inflation breakevens
- Swap curves
- Financial-condition indices

Historical vintages are essential for honest backtesting because many macro series are revised after initial publication.

### 20. Alternative-data history

The repository planning documents already identify several absent alternative datasets:

- Web traffic
- App downloads and rankings
- Credit/debit-card panels
- Shipment and customs data
- Freight rates
- Channel inventories
- Product pricing
- Store traffic
- Satellite imagery
- Employee headcount and attrition
- Customer reviews
- Cloud or web-usage telemetry
- Advertising spend
- A reliable search-trends provider

The current job-postings tool returns a present-time snapshot. Its signal becomes much more valuable when the external collector stores every poll and derives posting acceleration, department mix, posting age, and cancellation trends.

### Second-pass priority recommendation

Before purchasing expensive alternative-data feeds, Codex recommends prioritizing:

1. A durable security master and corporate-action lineage.
2. Append-only point-in-time storage for every existing tool response.
3. A uniform provenance, freshness, coverage, and cache envelope.
4. Intraday quote and market-session snapshots.
5. Canonical quarterly fundamentals with filing dates and restatement tracking.
6. A unified event calendar with date-change history.
7. FX, commodity, and cross-asset time series.
8. Credit ratings and bond-level data.
9. Point-in-time ownership, consensus, and ETF-flow histories.
10. Sector-specific KPI collectors selected according to the investment universe.

Adding more feeds before identifier, timestamp, and point-in-time foundations are reliable risks creating a larger collection of observations that cannot be safely joined, replayed, or backtested.

## Future financial-automation and trading blockers

The following findings are not blockers for a read-only aggregation server. They become mandatory before unattended paper or real-money automation.

### 1. Submitted orders are not followed through the fill lifecycle

The Alpaca path records submitted orders as `pending`, but no production component was found that continuously handles:

- Fills
- Partial fills
- Rejections
- Cancellations
- Expirations
- Replacements
- Trade corrections or busts
- Actual fill price and quantity

The local `positions` module states that a position opens when an order fills, but no production caller of `open_position()` was found. A broker position can therefore exist while the local database remains empty.

Required remediation: consume Alpaca `trade_updates` or implement a durable order reconciler, then make broker execution state authoritative.

### 2. Risk checks use local SQLite rather than authoritative broker state

The Risk Officer receives `portfolio_stats()` and `open_positions()` from the local database. It does not use current broker equity, cash, buying power, broker positions, or pending orders when approving a new order.

A reconciliation tool exists, but `place_paper_order` does not require reconciliation to be successful before placing another order.

Required remediation: fail closed whenever broker/local reconciliation is stale or unsuccessful.

### 3. Daily P&L is not true daily P&L

The current calculation assumes a fixed $100,000 starting value and combines today's realized P&L with total unrealized P&L since position entry. This can mask a current-day loss with an older gain or trigger a false daily halt because of an older loss.

Required remediation: persist start-of-day broker equity and calculate daily performance using broker-authoritative equity adjusted for deposits and withdrawals.

### 4. Position concentration can be exceeded through repeated orders

The 5% cap applies to the proposed order, not the resulting total position. It does not aggregate:

- Existing position in the same ticker
- Pending orders
- Repeated additions
- Current market-value appreciation
- Economically equivalent exposures

Correlation logic also explicitly approves adding to an existing position without assessing the resulting name concentration.

### 5. The sector limit is a placeholder

`MAX_SECTOR_PCT` is declared but not enforced in the deterministic Risk Officer. Sector classification and current sector market-value aggregation are not part of the execution gate.

### 6. Correlation risk fails open

If yfinance or the correlation calculation fails, the proposed trade is allowed. For unattended automation, unavailable risk data should normally reject the trade or reduce its permitted size.

### 7. No authoritative pre-trade quote validation

The agent supplies the `price` used for risk sizing, while the broker submits a market order. There is no mandatory fresh bid/ask snapshot immediately before execution. A stale or hallucinated price can therefore produce materially more exposure than approved.

### 8. No market clock, exchange calendar, or session validation

The execution path does not enforce market-open state, holidays, half days, premarket, after-hours, or daylight-saving-aware session boundaries. A day order submitted outside regular hours may remain pending while the decision context becomes stale.

The Sentry budget also uses a fixed UTC-5 approximation rather than `America/New_York` timezone rules.

### 9. Missing order-management operations

The broker MCP interface does not expose a complete lifecycle for:

- Listing open orders
- Canceling an order
- Canceling all orders
- Replacing an order
- Waiting for fill with a deadline
- Limit orders
- Stop orders
- Bracket orders
- Partial-fill handling
- Safe retry after an ambiguous timeout

The current random client-order ID means retrying after a timeout can create a second economic order if the first submission actually succeeded.

### 10. Stop losses and targets are not broker-resident

The local schema stores stop-loss and target values, but entry orders are plain market orders. No broker-side bracket or protective stop was found. If the agent, homelab, network, or MCP process is unavailable, the broker has no protective instruction.

### 11. Closing a position records an unverified exit price

The local position is marked closed before the close order's actual fill is known, using a local current or entry price as a substitute. This corrupts realized P&L, daily-loss controls, attribution, and later calibration.

The correct lifecycle is `open -> closing -> closed`, with `closed` recorded only after a confirmed broker fill.

### 12. Missing deterministic portfolio risk controls

Mandatory server-side enforcement is absent or incomplete for:

- Gross and net exposure
- Long/short exposure limits
- Cash and buying-power limits
- Sector and industry concentration
- Factor and theme concentration
- Beta-adjusted exposure
- Portfolio drawdown
- Volatility targeting
- Liquidity and average-daily-volume participation
- Maximum bid/ask spread
- Expected slippage
- Earnings and event-risk concentration
- Margin utilization
- Short availability and borrow cost
- Trading halts and symbol status
- Short-sale restrictions

Analytical skills are not a substitute for deterministic execution gates.

### 13. Missing operational safety controls

Before unattended automation, the system should have:

- A global broker-side kill switch
- A dead-man switch tied to recent successful reconciliation
- Maximum daily order count and notional enforced outside the LLM
- Fill/rejection/orphan-position alerts
- Scheduler heartbeat and stale-run detection
- Independent end-of-day reconciliation
- Durable audit records for every decision input
- Risk-policy and skill-version hashes
- Manual override that does not depend on the agent being responsive
- Database backup and recovery procedures
- A substantial supervised paper-trading soak period

## Recommended priorities

### Phase A: homelab aggregation

1. Standardize success, error, coverage, and warning semantics.
2. Add provider, source URL, retrieval time, data-as-of time, and cache status everywhere.
3. Publish a machine-readable manifest of tools, providers, expected freshness, and known coverage.
4. Add provider-level readiness checks and last-success timestamps.
5. Build an external collector that stores raw immutable responses.
6. Add reconciliation for critical identity, price, share-count, calendar, and consensus fields.
7. Add a dedicated quote-snapshot contract.
8. Run recurring coverage and failure-rate sweeps over a representative ticker universe.
9. Update the README and health checks to match the actual services and tool inventory.

### Phase B: report and memory layer

1. Store raw observations separately from derived reports.
2. Version normalized schemas and report templates.
3. Preserve citations and filing/document identifiers.
4. Track revisions rather than overwriting prior observations.
5. Record data gaps explicitly in every report.

### Phase C: supervised paper automation

1. Implement broker order/fill ingestion.
2. Make broker state authoritative and require reconciliation.
3. Add fresh quote, market-clock, spread, and asset-status checks.
4. Implement complete order management and idempotent recovery.
5. Add broker-resident protective orders.
6. Enforce portfolio limits in deterministic code.
7. Run supervised paper trading and measure failures, slippage, and reconciliation accuracy.

### Phase D: any consideration of real-money automation

Real-money automation should not be considered until the paper system has demonstrated reliable order-state accounting, deterministic risk enforcement, fail-closed behavior, operational recovery, and a sufficiently long audited track record. LLM instructions such as "always run the risk check" are not an adequate safety boundary.

## Final Codex assessment

Codex found that Nemo's research breadth is stronger than its consistency layer. For the current homelab purpose, the best investment is not immediately adding more data providers. It is making every existing observation self-describing: where it came from, what date it represents, how fresh it is, whether it was cached, how complete the coverage is, and whether another source agrees.

The future trading system should remain a separate, narrowly permissioned service. Research agents may have broad read access to Nemo, but order placement must sit behind broker-authoritative state, deterministic fail-closed risk controls, and a complete execution lifecycle.
