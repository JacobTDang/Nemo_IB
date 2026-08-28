# Nemo_IB

MCP servers that read company, market, macro and alternative financial data
from primary sources, plus the scheduled jobs that record what was known on
each date.

## Homelab setup

```bash
git clone https://github.com/JacobTDang/Nemo_IB.git
cd Nemo_IB
cp .env.example .env          # SEC_EMAIL, FINNHUB_API_KEY, FRED_API_KEY
docker build -t nemo-data:local .

cd deploy
NEMO_MCP_TOKEN=$(openssl rand -hex 32) docker compose up -d
```

`SEC_EMAIL` is required — SEC fair access wants a real contact address in the
User-Agent, and the servers refuse to call EDGAR without one rather than send a
placeholder.

Ports publish on `127.0.0.1` unless `NEMO_BIND_ADDR` says otherwise, so a
missing variable fails closed instead of exposing five servers to the LAN. On a
homelab, set it to the machine's Tailscale address:

```bash
NEMO_BIND_ADDR=100.x.y.z NEMO_MCP_TOKEN=... docker compose up -d
```

For a different architecture than the build host:

```bash
docker buildx build --platform linux/amd64 -t nemo-data:amd64 --load .
NEMO_IMAGE=nemo-data:amd64 NEMO_TARGET_ARCH=amd64 docker compose up -d
```

Deployment detail — what persists, log retention, auth, and the blockers a
green test suite does not catch: [`deploy/README.md`](deploy/README.md).

### Registering with an MCP client

Streamable HTTP with a bearer token, one entry per server:

```json
{
  "nemo-sec":       { "transport": "http", "url": "http://127.0.0.1:8810/mcp/" },
  "nemo-financial": { "transport": "http", "url": "http://127.0.0.1:8811/mcp/" },
  "nemo-finnhub":   { "transport": "http", "url": "http://127.0.0.1:8812/mcp/" },
  "nemo-fred":      { "transport": "http", "url": "http://127.0.0.1:8813/mcp/" },
  "nemo-altdata":   { "transport": "http", "url": "http://127.0.0.1:8814/mcp/" }
}
```

Each needs `Authorization: Bearer $NEMO_MCP_TOKEN`. The trailing slash on
`/mcp/` matters: without it the server answers `307`, and some clients drop the
`Authorization` header across the redirect.

`stdio` still works locally — `python -m tools.web_search_server.web_search`
with no argument — but HTTP is what the image serves.

## MCP servers

Five ship in the image and serve **96 tools** between them. They declare 99:
`search`, `rag_search` and `rag_ingest` are capability-gated and hidden,
because SearXNG and the RAG stack are not in this image. Every count here is
checked against the servers' own registries by `testing/test_readme_counts.py`,
so it fails rather than drifts.

| Server | Tools | Reads from | Needs |
|---|---:|---|---|
| `sec` | 48 | SEC EDGAR (XBRL, filing text) | `SEC_EMAIL` |
| `financial` | 20 | yfinance, local valuation models | — |
| `finnhub` | 14 | Finnhub | `FINNHUB_API_KEY` |
| `fred` | 5 | FRED | `FRED_API_KEY` |
| `altdata` | 9 | House Clerk, Senate eFD, USAspending, GovTrack, ATS boards, FinMind | `SEC_EMAIL` |

`alpaca` (6 tools, places orders), `excel` (3) and `sentry` (19, reads book
state) exist in the repo and are deliberately excluded from the image. A
data-source host holds no positions and should not be able to trade.

The rule throughout is that a tool never states something about a company it
did not read. A failed fetch says so rather than reporting an outage as "this
filer does not disclose it"; a concept the filer does not tag produces a
refusal naming what was looked for, not a substitute.

### `sec` — 48 tools

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
`get_earnings_releases` `extract_earnings_release_sentiment`

**Ownership and structure.** `get_share_count_series` `get_buyback_history`
`get_shelf_activity` `get_public_float` `get_schedule_13d_filings`
`get_fund_holdings` `compare_fund_holdings` `list_known_funds`

**Company lookup.** `get_latest_filing` `get_company_filings_history`
`get_sic_code` `find_peers_by_sic` `get_foreign_filer_profile`
`get_supply_chain` `get_patent_filings` `get_urls_content`

Foreign private issuers are handled explicitly: asking for a 10-K on a 20-F
filer returns which form it actually files. Values carry their reporting
currency — TSM reports in TWD, and a P/E built on a USD price and TWD earnings
is wrong by ~30x.

This server declares 51 and serves 48. `search` needs SearXNG; `rag_search` and
`rag_ingest` need the RAG stack. `search` is the one worth gating on principle:
it answers a missing SearXNG with an empty result list and no error, so
advertising it would make an absent container look like a query that matched
nothing.

### `financial` — 20 tools

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

### `finnhub` — 14 tools

`get_company_news` `get_market_news` `get_company_profile` `get_company_peers`
`get_basic_financials` `get_financial_statements` `get_earnings_calendar`
`get_earnings_surprises` `get_forward_estimates` `get_analyst_recommendations`
`get_analyst_rating_trend` `get_ipo_calendar` `get_insider_transactions`
`get_insider_sentiment`

### `fred` — 5 tools

`get_macro_snapshot` `get_treasury_yields` `get_credit_spreads` `get_fred_series`
`search_fred`

Spreads reported alongside a curve are struck from that same curve, so the two
always reconcile.

### `altdata` — 9 tools

`get_taiwan_monthly_revenue` `get_job_postings_count` `get_government_contracts`
`get_policy_signals` `get_capex_announcements` `get_congress_trades`
`get_congress_holdings` `get_congress_leaderboard` `get_congress_coverage`

Congressional disclosures are ingested rather than fetched per call — House
Clerk PTRs (PDF) and Senate eFD filings are parsed once into SQLite, because a
round trip plus a PDF parse per filing made every answer partial:

```bash
docker compose run --rm congress-sync --house 2024 2025 2026 --senate --senate-annual
docker compose run --rm congress-sync --status
```

Safe to re-run and safe to cron; nothing already parsed is fetched twice.

These are transactions and year-covering snapshots, **not live positions**.
Members file up to 45 days after trading, annual reports arrive months after
the year they cover, and roughly a third of holding rows are Excepted
Investment Funds whose contents are legally not itemised. Filings that arrive
as scans are counted in `coverage` rather than dropped. Amounts stay as the
ranges the filings publish — `amount_min` and `amount_max`, no midpoint.
Holdings are Senate-only.

## Automation

The same image on a schedule. All are batch jobs under the `sync` profile, so
`docker compose up` never starts them — a `restart: unless-stopped` batch job
is a loop.

```bash
cd deploy
docker compose run --rm research-daily --bootstrap   # first four nights
docker compose run --rm research-daily               # every night after
```

| Job | Does | When |
|---|---|---|
| `research-daily` | one session of prices per name, screens the universe, snapshots consensus | `30 22 * * 1-5` |
| `research-scan` | ranks candidates net of trading cost, files the intended orders | `0 23 * * 1-5` |
| `research-watch` | sweeps EDGAR for Schedule 13D stakes taken in a company | `*/20 13-23 * * 1-5` |
| `research-score` | scores filed orders whose holding period has closed | `0 7 * * 6` |
| `research-announce` | earnings dates and the hour they landed, from Item 2.02 filings | `0 9 * * 6` |
| `research-seed` | reconstructs the quarters of consensus the vendor still serves | `0 8 1 * *` |
| `congress-sync` | ingests congressional disclosures for the `altdata` server | `0 6 * * *` |

Every job exits non-zero when a stage fails, because the exit code is the only
way a scheduler finds out — and exits zero when it finds nothing, because most
nights it will, and paging on that is how the one night that matters gets
ignored.

Cron lines live in [`deploy/README.md`](deploy/README.md) and are checked
against the compose file by `testing/test_readme_counts.py`.

### The record

`research-daily` writes one SQLite file on the `research-data` volume. Some of
what is in it cannot be fetched later at any price: what analysts expected last
Tuesday is unrecoverable by Thursday, a company delisted in March is gone from
the vendor by June, and the vendor serves four quarters of consensus however
many you ask for. Back it up like a database that matters.

```
daily_bar          as-traded OHLCV, never overwritten
bar_revision       where a vendor's later disagreement goes, so the original
                   stands and the disagreement is still on the record
corporate_action   splits and dividends, dated to their own ex-date
universe_snapshot  who was eligible each day, and why the rest were not
consensus_snapshot what the street expected, and what the vendor reported
announcement       fiscal identity, the date the market learned, and whether it
                   landed before the open or after the close
paper_order        decisions, with the session they were for and no fill price
activist_filing    13D events with four timestamps, latency derived at read
run_log            every run, so a day the job did not run is visible
```

The record is append-only and every row carries when it was written as well as
what date it describes. Every reader filters on both, which is what stops a
value learned today from being visible to a question asked about last month.

Prices are stored as the stock printed them. `auto_adjust=False` does not mean
unadjusted — it returns split-adjusted prices, so NVDA's 2024-06-07 close
arrives as 120.89 against a real print of 1208.88. The conversion back happens
on the way in; the adjustment is rebuilt at read time from actions the reader
could have known, so a split announced in June cannot change what a May reader
computed.

## Tests

```bash
SKIP_NETWORK_TESTS=1 pytest testing/          # offline: no credentials, no network
STRICT_GATES=1 pytest testing/                # everything gated must actually run
```

The offline run needs nothing and should never fail. The strict run turns a
skipped gate into a failure, because a gate that cannot fail is not a gate.

The deploy gate is `testing/test_http_transport.py`, which completes a real MCP
handshake and one live tool call against each server. A health check only proves
the port is bound; a server whose MCP layer failed still answers it.
