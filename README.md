# Nemo_IB

This project has two parts. Five MCP servers read company, market, macro, and
alternative financial data from primary sources. Scheduled jobs record what was
known on each date.

## Homelab setup

```bash
git clone https://github.com/JacobTDang/Nemo_IB.git
cd Nemo_IB
cp .env.example .env          # SEC_EMAIL, FINNHUB_API_KEY, FRED_API_KEY
docker build -t nemo-data:local .

cd deploy
NEMO_MCP_TOKEN=$(openssl rand -hex 32) docker compose --env-file ../.env up -d
```

`SEC_EMAIL` is necessary. SEC fair access rules require a real contact address
in the User-Agent header. The servers refuse to call EDGAR without one. They do
not send a placeholder.

`--env-file ../.env` is also necessary. Each service declares only the keys that
its own code reads. Compose reads those values from a `.env` file beside the
compose file. That file is `deploy/.env`, and it is not the file at the top of
the repository. Without the flag, every credential becomes empty.

That failure is loud. The servers refuse to start without a token. The `/ready`
endpoint answers `503`. The `finnhub` and `fred` servers start without their
key, but `/ready` reports the missing key, the container stays unhealthy, and
every request fails with the name of the key.

Ports publish on `127.0.0.1` unless `NEMO_BIND_ADDR` gives a different address.
A missing variable therefore fails closed. It does not expose five servers to
the local network. On a homelab, set the variable to the Tailscale address of
the machine:

```bash
NEMO_BIND_ADDR=100.x.y.z NEMO_MCP_TOKEN=... docker compose --env-file ../.env up -d
```

To build for a different architecture than the build host:

```bash
docker buildx build --platform linux/amd64 -t nemo-data:amd64 --load .
NEMO_IMAGE=nemo-data:amd64 NEMO_TARGET_ARCH=amd64 docker compose --env-file ../.env up -d
```

Each server carries a memory limit. The limits total 3.2GB for the five
servers. [`deploy/README.md`](deploy/README.md) gives the measurements behind
those numbers. It also covers what persists, log retention, authentication, and
the problems that a passing test suite does not find.

### Registering with an MCP client

The servers use streamable HTTP with a bearer token. Register one entry for
each server:

```json
{
  "nemo-sec":       { "transport": "http", "url": "http://127.0.0.1:8810/mcp/" },
  "nemo-financial": { "transport": "http", "url": "http://127.0.0.1:8811/mcp/" },
  "nemo-finnhub":   { "transport": "http", "url": "http://127.0.0.1:8812/mcp/" },
  "nemo-fred":      { "transport": "http", "url": "http://127.0.0.1:8813/mcp/" },
  "nemo-altdata":   { "transport": "http", "url": "http://127.0.0.1:8814/mcp/" }
}
```

Each entry needs `Authorization: Bearer $NEMO_MCP_TOKEN`. The trailing slash on
`/mcp/` is necessary. Without the slash, the server answers `307`. Some clients
then drop the `Authorization` header across the redirect.

The `stdio` transport also works locally. Run
`python -m tools.web_search_server.web_search` with no argument. The image
serves HTTP.

## MCP servers

Five servers ship in the image. Together they serve **96 tools**.
They declare 99: the image hides `search`, `rag_search`, and `rag_ingest`. It
contains neither SearXNG nor the RAG stack.

`testing/test_readme_counts.py` checks every count on this page against the
registry of each server. A count that drifts fails the test suite.

| Server | Tools | Reads from | Needs |
|---|---:|---|---|
| `sec` | 48 | SEC EDGAR (XBRL, filing text) | `SEC_EMAIL` |
| `financial` | 20 | yfinance, local valuation models | — |
| `finnhub` | 14 | Finnhub | `FINNHUB_API_KEY` |
| `fred` | 5 | FRED | `FRED_API_KEY` |
| `altdata` | 9 | House Clerk, Senate eFD, USAspending, GovTrack, ATS boards, FinMind | `SEC_EMAIL` |

Three more servers exist in the repository and stay out of the image. They are
`alpaca` with 6 tools that place orders, `excel` with 3 tools, and `sentry` with
19 tools that read book state. A data-source host holds no positions. It must
not be able to trade.

One rule applies to every tool. A tool never states something about a company
that it did not read. A failed fetch reports the failure. It does not report an
outage as "this filer does not disclose it". When a filer tags no value for a
concept, the tool refuses and names the concept it looked for. It supplies no
substitute.

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

This server handles foreign private issuers explicitly. A request for a 10-K
against a 20-F filer returns the form that the company files. Values carry their
reporting currency. TSM reports in TWD. A P/E ratio built on a USD price and TWD
earnings is wrong by approximately 30 times.

This server declares 51 tools and serves 48. The `search` tool needs SearXNG.
The `rag_search` and `rag_ingest` tools need the RAG stack. The `search` tool is
the important one to hide. It answers a missing SearXNG with an empty result
list and no error. To advertise it would make an absent container look like a
query that matched nothing.

### `financial` — 20 tools

`get_market_data` `get_price_history` `get_options_metrics` `get_short_interest`
`get_trading_metrics` `get_industry_etfs` `get_corporate_actions`
`extract_13f_holdings` `comparable_company_analysis` `calculate_dcf`
`calculate_scenario_dcf` `calculate_wacc` `calculate_lbo`
`calculate_credit_profile` `calculate_capital_returns` `get_historical_analogue`
`backtest_signal` `analyze_exposures` `record_thesis_evolution`
`get_thesis_evolution`

The calculators refuse a structure that they cannot model. They return no
number. An LBO with more debt than purchase price cannot be financed. To say so
is better than to report the 40x MOIC that this structure produces.

### `finnhub` — 14 tools

`get_company_news` `get_market_news` `get_company_profile` `get_company_peers`
`get_basic_financials` `get_financial_statements` `get_earnings_calendar`
`get_earnings_surprises` `get_forward_estimates` `get_analyst_recommendations`
`get_analyst_rating_trend` `get_ipo_calendar` `get_insider_transactions`
`get_insider_sentiment`

### `fred` — 5 tools

`get_macro_snapshot` `get_treasury_yields` `get_credit_spreads` `get_fred_series`
`search_fred`

A spread reported beside a curve comes from that same curve. The two values
therefore always agree.

### `altdata` — 9 tools

`get_taiwan_monthly_revenue` `get_job_postings_count` `get_government_contracts`
`get_policy_signals` `get_capex_announcements` `get_congress_trades`
`get_congress_holdings` `get_congress_leaderboard` `get_congress_coverage`

This server reads congressional disclosures from a local store. It does not
fetch them for each call. A separate job parses House Clerk PTR documents and
Senate eFD filings into SQLite once. One network round trip and one PDF parse
for each filing made every answer partial.

```bash
docker compose --env-file ../.env run --rm congress-sync \
  --house 2024 2025 2026 --senate --senate-annual
docker compose --env-file ../.env run --rm congress-sync --status
```

The sync job is safe to run again and safe to schedule. It fetches nothing that
it has already parsed.

These rows are transactions and year-covering snapshots. They are **not live
positions**. A member can file up to 45 days after a trade. An annual report
arrives months after the year that it covers. Approximately one third of the
holding rows are Excepted Investment Funds, and the law does not require a filer
to itemize their contents.

The store counts a filing that arrives as a scan in `coverage`. It drops no such
filing. Amounts stay as the ranges that the filings publish, under `amount_min`
and `amount_max`. The store computes no midpoint. Only the Senate reports
holdings.

## Automation

The jobs run the same image on a schedule. Every job is a batch job under the
`sync` profile. Therefore `docker compose up` starts none of them. A batch job
with `restart: unless-stopped` becomes a loop.

```bash
cd deploy
docker compose --env-file ../.env run --rm research-daily --bootstrap   # first four nights
docker compose --env-file ../.env run --rm research-daily               # every night after
```

| Job | Does | When |
|---|---|---|
| `research-daily` | one session of prices per name, screens the universe, snapshots consensus | `30 22 * * 1-5` |
| `research-scan` | ranks candidates net of trading cost and borrow, files the intended orders | `0 23 * * 1-5` |
| `research-watch` | sweeps EDGAR for Schedule 13D stakes taken in a company | `*/20 13-23 * * 1-5` |
| `research-score` | scores filed orders whose holding period has closed | `0 7 * * 6` |
| `research-announce` | earnings dates and the hour they landed, from Item 2.02 filings | `0 9 * * 6` |
| `research-seed` | reconstructs the quarters of consensus the vendor still serves | `0 8 1 * *` |
| `research-status` | reads the run log and the book, prints one screen, exits 1 when a job needs attention | `0 12 * * 1-5` |
| `congress-sync` | ingests congressional disclosures for the `altdata` server | `0 6 * * *` |

The scan has no default borrow rate. It refuses every short position that it
cannot price. It charges no short position nothing, because a short charged
nothing outranks every name that paid. Load rates with
`python -m research.borrow`, or declare one flat rate with
`research-scan --borrow-rate 0.03`. Until then the book holds long positions
only, and the scan reports that. [`deploy/README.md`](deploy/README.md) gives
the arithmetic.

Every job exits non-zero when a stage fails. The exit code is the only way that
a scheduler learns about the failure. Every job exits zero when it finds
nothing. Most nights it will find nothing. An alert on an empty night teaches
the operator to ignore the alert that matters.

The cron lines are in [`deploy/README.md`](deploy/README.md).
`testing/test_readme_counts.py` checks them against the compose file.

`research-status` shows the last run of every job, its state, and the open
book. It reads the store and writes nothing. It exits 1 when a job crashed,
failed, or is stale. Therefore cron can run it as a check. The `--json` flag
prints the same report for scripts.

```bash
docker compose --env-file ../.env run --rm research-status
```

### The `nemo` command

`nemo` answers "is this thing working" from a terminal. Install it into the
virtual environment once, from the repository root.

```bash
pip install -e .
```

| Command | Does |
|---|---|
| `nemo status` | prints the same screen as `research-status`, and exits 1 when a job needs attention |
| `nemo services` | prints every container, and exits 1 when a server is not running and healthy |
| `nemo logs <service>` | prints the last 200 lines of one service, and follows the log with `-f` |
| `nemo monitor` | prints the status screen and the services table on a loop |

`nemo --monitor` is the same command as `nemo monitor`. It refreshes every 30
seconds until Ctrl-C. The `--every` flag sets a different interval. A refresh
that fails prints the reason in place, and the next refresh runs.

`nemo status` reads the store in this process when `NEMO_PIT_DB` names a
readable file. Otherwise it runs `research-status` in a container, and it says
so on the error stream before it prints. The homelab keeps the store in a
volume that the host user cannot open, so the container is the only reader
there. The `--via local` and `--via docker` flags force one source.

`nemo services` asks compose for every container, and not only for the running
ones. A batch job that exited is on the table. A server with no container at
all is on the table as well, because a missing container is the fact that the
operator needs.

`nemo` finds the compose file from its own location, and not from the working
directory. Therefore it works from any subdirectory.
`python -m research.cli` runs the same command without the install.

The scan ranks on one earnings surprise. The constant SIGNAL_VARIANT in
`research/scanner.py` names it, and the scan ships with `ts`. The environment
does not set it. The `ts` variant
computes the surprise from the XBRL financials. The `ts_release` variant reads
diluted EPS from the 8-K earnings release on the day that it is filed. The `cs`
variant uses the recorded consensus. Every paper order records the variant that
produced it.

A replay over one year compared `ts` and `ts_release` on the same prints. The
release-timed entry earned 61 bps more per trade. The gain comes from prints
where the XBRL filing arrived a week or more after the release.
[`docs/replay_2026-09-03_release_timing.md`](docs/replay_2026-09-03_release_timing.md)
gives the numbers and the caveats.

The universe screen keeps one line per issuer. When two tickers share a CIK,
the more liquid line stays. The other line is excluded, and the record names
the line that stayed. A preferred, warrant, unit, or right is excluded by its
ticker suffix.

### The record

`research-daily` writes one SQLite file on the `research-data` volume. Money
cannot buy back some of what that file holds. What analysts expected last
Tuesday is unrecoverable by Thursday. A company that delisted in March is gone
from the vendor by June. The vendor serves four quarters of consensus however
many quarters you request. Back up this file like a database that matters.

```
daily_bar          as-traded OHLCV, never overwritten
bar_revision       where a vendor's later disagreement goes, so the original
                   stands and the disagreement is still on the record
corporate_action   splits and dividends, dated to their own ex-date
universe_snapshot  who was eligible each day, and why the rest were not
consensus_snapshot what the street expected, and what the vendor reported
announcement       fiscal identity, the date the market learned, and whether it
                   landed before the open or after the close
borrow_rate        what it cost to be short, per name, as somebody quoted it
paper_order        decisions, with the session they were for and no fill price
activist_filing    13D events with four timestamps, latency derived at read
run_log            every run, so a day the job did not run is visible
```

The record is append-only. Every row carries the date that it describes and the
time when the recorder wrote it. Every reader filters on both fields. That
filter is what stops a value learned today from becoming visible to a question
about last month.

The store keeps prices as the stock printed them. `auto_adjust=False` does not
mean unadjusted. It returns split-adjusted prices. The NVDA close for
2024-06-07 arrives as 120.89 against a real print of 1208.88. The recorder
converts the price back on the way in.

A reader rebuilds the adjustment from the corporate actions that the reader
could have known. Therefore a split announced in June cannot change what a May
reader computed.

## Tests

```bash
SKIP_NETWORK_TESTS=1 pytest testing/          # offline: no credentials, no network
NEMO_REQUIRE_SERVICES=1 pytest testing/       # everything gated must actually run
```

The offline run needs no credentials and no network. It must never fail. The
strict run turns a skipped gate into a failure, because a gate that cannot fail
is not a gate.

The deploy gate is `testing/test_http_transport.py`. It completes a real MCP
handshake and one live tool call against each server. A health check proves only
that the port is bound. A server whose MCP layer failed to start still answers
one.
