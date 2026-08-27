# Nemo_IB

Two things that share a machine and a set of upstreams.

**MCP servers** that aggregate company, market, macro and alternative financial
data from primary sources, served over streamable HTTP for any MCP client. Five
ship in the homelab image and serve **96 tools** between them. They declare 99:
`search`, `rag_search` and `rag_ingest` are capability-gated and hidden,
because SearXNG and the RAG stack are not in this image. Every count on this
page is checked against the servers' own registries by `testing/test_readme_counts.py`,
so it fails rather than drifts.

**A point-in-time record** that the same image feeds on a schedule, and the jobs
that read it. Nothing about it is a server: they are batch jobs that finish. See
[Automation](#automation).

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

Streamable HTTP with a bearer token. One entry per server:

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
`/mcp/` matters: without it the server answers `307` and some clients drop the
`Authorization` header across the redirect.

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
`get_earnings_releases` `extract_earnings_release_sentiment`

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

Three further tools are declared and hidden: `search` needs SearXNG, and
`rag_search` and `rag_ingest` need the RAG stack. The image has neither, which
is why this server declares 51 and serves 48. `search` is the one worth gating
on principle rather than on tidiness — it answers a missing SearXNG with an
empty result list and no error, so advertising it would make an absent
container look like a query that matched nothing.

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
data-source host does not have. `analyze_exposures` returns an empty book rather
than failing, because empty is the truthful answer there. `get_thesis_evolution`
asks about one named thesis, so on a book holding no such thesis it refuses --
the same answer on either host. A null row there would be byte-for-byte what a
real thesis awaiting its first check-in returns.

### `finnhub` — 14 tools, news, estimates, insiders

`get_company_news` `get_market_news` `get_company_profile` `get_company_peers`
`get_basic_financials` `get_financial_statements` `get_earnings_calendar`
`get_earnings_surprises` `get_forward_estimates` `get_analyst_recommendations`
`get_analyst_rating_trend` `get_ipo_calendar` `get_insider_transactions`
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

## Automation

A second use of the same image and the same upstreams: a store that records
what was known on each date, and jobs that read it. All of them are batch jobs
under the `sync` profile, so `docker compose up` never starts them — a
`restart: unless-stopped` batch job is a loop.

```bash
cd deploy
docker compose run --rm research-daily --bootstrap   # first four nights
docker compose run --rm research-daily               # every night after
```

| Job | Does | When |
|---|---|---|
| `research-daily` | one session of prices per name, screens the universe, snapshots consensus | `30 22 * * 1-5` |
| `research-scan` | ranks candidates net of trading cost, files the intended orders | `0 23 * * 1-5` |
| `research-watch` | sweeps EDGAR for Schedule 13D stakes taken **in** a company | `*/20 13-23 * * 1-5` |
| `research-score` | scores filed orders whose holding period has closed | `0 7 * * 6` |
| `research-seed` | reconstructs the four quarters of consensus the vendor still serves | `0 8 1 * *` |
| `congress-sync` | ingests congressional disclosures for the `altdata` server | `0 6 * * *` |

Two things are true of every job: it exits non-zero when a stage fails, because
the exit code is the only way a scheduler finds out, and it exits zero when it
finds nothing, because most nights it will and paging on that is how the one
night that matters gets ignored.

### Why a store rather than a query

Some of this data cannot be fetched later at any price. What analysts expected
last Tuesday is unrecoverable by Thursday; a company delisted in March is gone
from the vendor by June; the vendor serves four quarters of consensus history
whether you ask for twelve or thirty. So the record is append-only and every
row carries when it was written as well as what date it describes. Every reader
filters on both, which is what stops a value learned today from being visible
to a question asked about last month.

Prices are stored as the stock printed them. `auto_adjust=False` does not mean
unadjusted — it returns split-adjusted prices, so NVDA's 2024-06-07 close
arrives as 120.89 against a real print of 1208.88 — and the conversion back
happens on the way in, using splits that arrive in the same response. The
adjustment is rebuilt at read time from actions the reader could have known, so
a split announced in June cannot change what a May reader computed.

### The record

`research-daily` writes into one SQLite file on the `research-data` volume.
It is reconstructible from the public record only at the cost of re-reading
every filing, and the consensus in it is not reconstructible at all — back it
up like a database that matters.

```
daily_bar          as-traded OHLCV, never overwritten
bar_revision       where a vendor's later disagreement goes, so the original
                   stands and the disagreement is still on the record
corporate_action   splits and dividends, dated to their own ex-date
universe_snapshot  who was eligible each day, and why the rest were not
consensus_snapshot what the street expected, and what the vendor reported
announcement       fiscal identity and whether the print was before the open
                   or after the close, which decides the reaction session
paper_order        decisions, with the session they were for and no fill price
activist_filing    13D events with four timestamps, latency derived at read
run_log            every run, so a day the job did not run is visible
```

### What answers today, and what does not

The time-series surprise, the cross-sectional one, the spread and cost model,
the universe screen, the 13D watcher and the scanner all run now. Which
surprise the scan ranks on is one constant, and each carries its own
coefficient: a sigma and a rank are different quantities and pricing them with
one number would make them look comparable. The analyst surprise needs eight
quarters of recorded consensus; seeding supplies four and the recorder adds one
a quarter, so it refuses until it has them and says how many it holds.

Nothing here decides how much of the drift a surprise is worth. That
coefficient is declared, reported uncalibrated on every scan, and the point of
recording orders and scoring them later is to replace it with one measured from
this book's own outcomes. A replay over 652 decision dates does not yet support
it: mean +51.3bp against a median of −71.3bp and t = +0.80, which the
calibration gate refuses on both counts.

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

The store and the jobs that read it carry their own suites, written around the
two rules they exist to enforce. One sweep checks that every reader taking an
`as_of` hides a row written after it, and fails if a reader is added and left
out of the sweep. Another checks that every recorder run twice writes nothing
the second time, and that every refusal returns its numbers as `None` rather
than leaving a stale one beside `success: False`. Values that can be worked out
on paper are asserted as numbers rather than as relationships, since a
relationship survives a formula that is wrong by a constant factor everywhere.

## Repository layout

```
tools/
  web_search_server/      SEC/EDGAR reads, filing text extraction
  financial_modeling_engine/  market data, DCF/LBO/WACC, backtests
  news_agregator/         Finnhub and FRED
  altdata_server/         congressional disclosures, job boards, contracts, policy
  alpaca/ excel_server/ sentry_server/   not shipped in the image
  mcp_http.py             shared streamable-HTTP transport, bearer auth
research/
  pit_store.py            the append-only record and its point-in-time reads
  daily_job.py            nightly recorder, universe screen, entry point
  scanner.py              ranks candidates net of cost, files intended orders
  scoring.py              what filed orders did, and what that implies
  spread.py               EDGE effective spread and the round-trip cost model
  sue.py                  quarterly EPS from XBRL, time-series and analyst
  sue_cs.py               the same surprise ranked across names, not time
  seed_consensus.py       reconstructs the consensus history that is still served
  activist_watch.py       subject-side 13D detection and detection latency
  replay.py               the scanner over history, survivorship-biased and says so
deploy/                   compose stack, deployment and auth notes
testing/                  offline suite plus gated live suites
```
