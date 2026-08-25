# Nemo_IB

MCP servers that aggregate company, market, macro and alternative financial data
from primary sources, served over streamable HTTP for any MCP client.

Five servers ship in the homelab image and declare **96 tools** between them.
Every count on this page is measured against a running instance rather than
maintained by hand.

| Server | Tools | Reads from | Needs |
|---|---:|---|---|
| `sec` | 48 | SEC EDGAR (XBRL, filing text) | `SEC_EMAIL` |
| `financial` | 20 | yfinance, local valuation models | — |
| `finnhub` | 14 | Finnhub | `FINNHUB_API_KEY` |
| `fred` | 5 | FRED | `FRED_API_KEY` |
| `altdata` | 9 | House Clerk, Senate eFD, USAspending, GovTrack, ATS boards, FinMind | `SEC_EMAIL` |

Three further servers exist in the repo and are **deliberately excluded from the
image** — `alpaca` (6 tools, places orders), `excel` (3), `sentry` (19, reads
book state). A data-source host holds no positions and should not be able to
trade, so shipping them would mean "trading tools nobody happens to start".

## What these are for

Answering questions about companies from the filings themselves rather than from
a summary of them. The design rule throughout is that **a tool must never state
something about a company that it did not read**:

- A fetch that fails says so. It never reports an outage as "this filer does not
  disclose it", which is the one answer worse than an error.
- Amounts that the source publishes as ranges stay ranges. Congressional
  disclosures carry `amount_min` and `amount_max` and no midpoint, because the
  filings contain none.
- Coverage travels with the answer. An empty result from a partially-ingested
  store is a gap, not a finding, and says which it is.
- A concept the filer does not tag produces a refusal naming what was looked
  for, not a plausible substitute. `get_ebitda_margin("GS")` explains why EBITDA
  is meaningless for a bank instead of returning a number.

## Running them

```bash
cd deploy
NEMO_MCP_TOKEN=$(openssl rand -hex 32) docker compose up -d
```

Ports publish on `127.0.0.1` unless `NEMO_BIND_ADDR` says otherwise, so a
missing variable fails closed rather than exposing five servers to the LAN. On a
homelab, set it to the machine's Tailscale address. Full deployment notes,
authentication, log retention and the blockers a green test suite does not
catch: [`deploy/README.md`](deploy/README.md).

## Registering with an MCP client

```bash
claude mcp add --transport http nemo-sec       http://127.0.0.1:8810/mcp/ --header "Authorization: Bearer $NEMO_MCP_TOKEN"
claude mcp add --transport http nemo-financial http://127.0.0.1:8811/mcp/ --header "Authorization: Bearer $NEMO_MCP_TOKEN"
claude mcp add --transport http nemo-finnhub   http://127.0.0.1:8812/mcp/ --header "Authorization: Bearer $NEMO_MCP_TOKEN"
claude mcp add --transport http nemo-fred      http://127.0.0.1:8813/mcp/ --header "Authorization: Bearer $NEMO_MCP_TOKEN"
claude mcp add --transport http nemo-altdata   http://127.0.0.1:8814/mcp/ --header "Authorization: Bearer $NEMO_MCP_TOKEN"
```

The trailing slash on `/mcp/` matters: without it the server answers `307` and
some clients drop the `Authorization` header across the redirect.

`stdio` still works for local use — `python -m tools.web_search_server.web_search`
with no argument — but HTTP is what the image serves.

## The tools

### `sec` — 48 tools, SEC EDGAR

Financial statements, filing text, ownership and structure, read from XBRL and
filing documents.

**Statements and metrics.** `get_revenue_base` `get_annual_revenue`
`get_ebitda_margin` `get_margin_breakdown` `get_capex_pct_revenue` `get_tax_rate`
`get_depreciation` `get_historical_fcf` `get_working_capital`
`get_working_capital_trends` `get_accruals_quality` `get_operating_leases`
`get_debt_maturity_schedule` `get_sbc_series` `track_segment_growth`
`get_segment_financials` `get_geographic_revenue` `get_contracted_revenue`

**Filing text.** `extract_mda` `extract_risk_factors` `extract_forward_signals`
`extract_guidance` `extract_litigation` `extract_customer_concentration`
`extract_8k_events` `extract_proxy_compensation` `extract_governance_data`
`extract_disclosure_data` `get_disclosures_names` `diff_10k`
`get_earnings_transcripts` `extract_call_sentiment`

**Ownership and structure.** `get_share_count_series` `get_buyback_history`
`get_shelf_activity` `get_public_float` `get_schedule_13d_filings`
`get_fund_holdings` `compare_fund_holdings` `list_known_funds`

**Company lookup.** `get_latest_filing` `get_company_filings_history`
`get_sic_code` `find_peers_by_sic` `get_foreign_filer_profile`
`get_supply_chain` `get_patent_filings` `get_urls_content`

Foreign private issuers are handled explicitly: asking for a 10-K on a 20-F
filer returns which form it actually files, rather than an empty result. Values
carry their reporting currency — TSM reports in TWD, and a P/E built on a USD
price and TWD earnings is wrong by ~30x.

Two further tools, `rag_search` and `rag_ingest`, are declared but hidden unless
the RAG stack is present, which it is not in the image. That is why the server
declares 50 and serves 48.

### `financial` — 20 tools, market data and valuation

`get_market_data` `get_price_history` `get_options_metrics` `get_short_interest`
`get_trading_metrics` `get_industry_etfs` `get_corporate_actions`
`extract_13f_holdings` `comparable_company_analysis` `calculate_dcf`
`calculate_scenario_dcf` `calculate_wacc` `calculate_lbo`
`calculate_credit_profile` `calculate_capital_returns` `get_historical_analogue`
`backtest_signal` `analyze_exposures` `record_thesis_evolution`
`get_thesis_evolution`

The calculators refuse structures they cannot model rather than returning a
number: an LBO whose debt exceeds the purchase price is unfinanceable, and
saying so beats reporting the 40x MOIC that sizing produces.

`analyze_exposures` and `get_thesis_evolution` read book state, which a
data-source host does not have. They return an empty result rather than failing,
because empty is the truthful answer there.

### `finnhub` — 14 tools, news, estimates, insiders

`get_company_news` `get_market_news` `get_company_profile` `get_company_peers`
`get_basic_financials` `get_financial_statements` `get_earnings_calendar`
`get_earnings_surprises` `get_forward_estimates` `get_analyst_recommendations`
`get_analyst_revisions_history` `get_ipo_calendar` `get_insider_transactions`
`get_insider_sentiment`

### `fred` — 5 tools, macro

`get_macro_snapshot` `get_treasury_yields` `get_credit_spreads` `get_fred_series`
`search_fred`

Spreads reported alongside a curve are struck from that same curve, so the two
always reconcile.

### `altdata` — 9 tools, alternative data

`get_taiwan_monthly_revenue` `get_job_postings_count` `get_government_contracts`
`get_policy_signals` `get_capex_announcements` `get_congress_trades`
`get_congress_holdings` `get_congress_leaderboard` `get_congress_coverage`

**Congressional disclosures** are ingested rather than fetched per call: House
Clerk PTRs (PDF) and Senate eFD filings are parsed once into SQLite, because a
round trip plus a PDF parse per filing bought about twenty filings per call and
made every answer partial.

```bash
docker compose run --rm congress-sync --house 2024 2025 2026 --senate --senate-annual
docker compose run --rm congress-sync --status
```

Safe to re-run and safe to cron; nothing already parsed is fetched twice. The
store is on a named volume rather than the container's RAM-backed cache.

What it cannot tell you: these are **transactions and year-covering snapshots,
not live positions**. Congress publishes no current holdings. Members file up to
45 days after trading, annual reports arrive months after the year they cover,
and roughly a third of holding rows are Excepted Investment Funds whose contents
are legally not itemised. Filings that arrive as scans of paper cannot be parsed
at all and are counted in `coverage` rather than dropped. Holdings are currently
Senate-only.

## Tests

```bash
SKIP_NETWORK_TESTS=1 pytest testing/          # offline: no credentials, no network
STRICT_GATES=1 pytest testing/                # everything gated must actually run
```

The offline run needs nothing and should never fail. The strict run turns a
skipped gate into a failure, because a gate that cannot fail is not a gate and
"skipped" otherwise decays into "deleted".

The deploy gate is `testing/test_http_transport.py`, which completes a real MCP
handshake and one live tool call against each server. A health check only proves
the port is bound; a server whose MCP layer failed still answers it.

## Repository layout

```
tools/
  web_search_server/      SEC/EDGAR reads, filing text extraction
  financial_modeling_engine/  market data, DCF/LBO/WACC, backtests
  news_agregator/         Finnhub and FRED
  altdata_server/         congressional disclosures, job boards, contracts, policy
  alpaca/ excel_server/ sentry_server/   not shipped in the image
  mcp_http.py             shared streamable-HTTP transport, bearer auth
deploy/                   compose stack, deployment and auth notes
testing/                  offline suite plus gated live suites
```
