# Truth-Source Refactor — Design

**Date:** 2026-08-20
**Status:** approved, pending implementation plan
**Scope:** cleanup only. No HTTP transport, no Dockerfiles, no deployment.

## Problem

The MCP tool layer has accumulated four kinds of debt that block treating it as a
clean data layer for Claude Code or any other agent:

1. **A fully redundant server.** All four `nemo_openbb` tools are covered by
   tools that already exist in servers we keep, closer to source. It is also the
   documented indefinite-hang server (`docs/known_issues.md`).
2. **Duplicate options paths.** `get_options_implied_move` (altdata) and
   `get_options_metrics` (financial) both fetch yfinance option chains for the
   same ticker in the same workflow — two network fetches, one of them in a
   subprocess, with four separate options bugs tracked across them.
3. **A judgment tool in a data layer.** `get_finbert_sentiment` returns an
   opinion about text, not a fact about the world. It is the only such tool in
   the inventory, and it is the reason `transformers` is a direct dependency.
4. **Cache and state share one SQLite file.** `Session_Cache` writes
   `db_cache/session.db` — the same file holding theses, positions, and the
   sentry queue. Neither connection sets WAL or `busy_timeout`, and six daemons
   already write concurrently.

## Guiding seam

> Does the tool return a fact about the world, or a fact about my book?

World-facts belong in the tool layer. Book-facts (theses, positions, RAG corpus)
belong with the state database. This refactor moves the tool layer toward that
line; it does not attempt to complete the split.

## Deliverables

### D1 — Merged options tool

`get_options_metrics` becomes the union of both surfaces, from a single yfinance
chain fetch:

| Field group | Source today |
|---|---|
| Term structure (ATM IV ~7/30/60/90d) | financial |
| Put/call skew | both (financial's 0.9/1.1 moneyness method wins) |
| OI + volume put/call ratios | financial |
| ATM straddle implied move, front expiry | altdata |

Spot is fetched internally rather than passed in. The `options_chain_rows`
parameter is dropped — it exists only to accept pre-fetched rows from
`obb_options_chain`, which this refactor deletes.

`/preearnings-research` keeps its current semantics: `event_implied_move` is
derived from the term structure; the raw straddle governs the >20% binary-event
rule. Both now read from one tool call.

### D2 — Delete `nemo_openbb`

Remove `tools/openbb_server/`, `testing/test_openbb_server.py`, and all
`openbb-*` pins from both manifests. Replacement mapping:

| Deleted | Covered by |
|---|---|
| `obb_insider_trading` | finnhub `get_insider_transactions` + `get_insider_sentiment` |
| `obb_analyst_consensus` | finnhub `get_analyst_recommendations` + `get_forward_estimates` + `get_analyst_revisions_history` |
| `obb_news_company` | finnhub `get_company_news` |
| `obb_options_chain` | merged `get_options_metrics` (D1) |

Ordering constraint: D1 must land before D2, because D1 removes the parameter
that couples altdata to `obb_options_chain`.

### D3 — Delete FinBERT and web-traffic

Remove `tools/altdata_server/finbert_runner.py`, `get_web_traffic_signal`,
`tools/preearnings/web_traffic.py`, `testing/test_preearnings_web_traffic.py`,
and the four FinBERT tests in `testing/test_altdata_tools.py`.

The `slow` pytest marker **stays**. Its description in `pyproject.toml` reads
"model-loading tests (FinBERT) excluded from quick runs", but 14 tests in
`testing/test_sec_xbrl_functions.py` also use it. Only the description changes —
removing the marker would break the SEC XBRL suite.

Sentiment moves to the agent. `finnhub_server.py:857-858` already returns
`headline` and `summary` per article, and the news-digest sub-agent in
`/preearnings-research` already reads `get_company_news` in full — FinBERT is a
second pass over text the sub-agent is holding. No capability gap.

**Dependency effect (corrected):** torch is NOT removable. `agent/rag/embedder.py:49`
imports `sentence_transformers`, which requires it. What this buys:

- `transformers`, `accelerate`, `bitsandbytes` lose their only real consumers
  (`agent/huggingface_template.py` is imported by nothing — dead code, also deleted).
  `transformers` remains present transitively via sentence-transformers but stops
  being a direct pin.
- `torch`/`torchvision`/`torchaudio` drop from `+cu121` to CPU wheels. Every
  remaining consumer is CPU-bound (`finbert_runner.py:80` was already `device=-1`;
  MiniLM embedding is trivial on CPU). ~2.5GB → ~200MB, and it unblocks arm64.

### D4 — Split cache from state, harden both

- `Session_Cache` points at `db_cache/tool_cache.db` instead of `session.db`.
  Existing cache rows are abandoned, not migrated — the cache is disposable by
  construction and repopulates on use.
- Both `agent/cache.py` and `state/schema.py:get_connection` set
  `journal_mode=WAL` and a `busy_timeout` (5000ms).
- `DB_PATH` in `state/schema.py:11` becomes absolute (repo-root anchored),
  overridable via `NEMO_DB_PATH`, replacing the current CWD-relative path.
  `Session_Cache` anchors the same way, overridable via `NEMO_CACHE_DB_PATH`.
  Both defaults resolve from the repo root, not the process CWD.

### D5 — Manifest repair

- Add the missing `sentence-transformers==5.5.1` to `pyproject.toml`. It is in
  `requirements.txt` only, so `uv sync` currently produces an environment where
  RAG fails at first `rag_search`.
- Apply every removal from D2/D3 to **both** manifests.

Full manifest reconciliation is explicitly out of scope — the two files have
drifted more broadly (e.g. `aiohttp` 3.13.5 vs 3.13.2) and that is its own task.

### D6 — Skill update

`.claude/skills/preearnings-research/SKILL.md`: the news-digest template drops
the `get_finbert_sentiment` call and the `finbert_net_score` output field,
replaced by an agent-produced sentiment field. The asymmetry-inputs step reads
straddle and term structure from the single merged `get_options_metrics`.

## What survives in `tools/altdata_server/`

Six tools, all direct-source, no heavy dependencies. The directory name stays
accurate — Trends, job postings, government contracts, policy signals, and
Taiwan revenue are textbook alternative data — so every `mcp__nemo_altdata__*`
reference in the skills keeps working.

| Tool | Upstream |
|---|---|
| `get_google_trends` | pytrends |
| `get_job_postings_count` | boards-api.greenhouse.io, api.lever.co |
| `get_government_contracts` | api.usaspending.gov |
| `get_policy_signals` | api.congress.gov, govtrack.us |
| `get_taiwan_monthly_revenue` | api.finmindtrade.com |
| `get_capex_announcements` | DuckDuckGo news (`ddgs`) |

`get_job_postings_count` stays despite having no skill reference — it is tested,
harmless, and wiring it is a separate decision.

## Testing strategy

The suite spans 116 files, only 21 of which mock. Most are live integration
tests marked `network` and skippable via `SKIP_NETWORK_TESTS=1`.
`docs/known_issues.md` cites 314 passing as of the pre-earnings merge; the first
implementation step records a fresh baseline count with
`SKIP_NETWORK_TESTS=1 pytest --collect-only -q`, since every regression gate
below is a delta against that number rather than against 314.
That shapes the approach: correctness for the merge cannot come from asserting
on live numbers, so it comes from differential comparison instead.

**Order is test-first throughout, per project convention.**

1. **Migrate the options tests.** The ~11 tests in `testing/test_altdata_tools.py`
   covering the options path — `test_leg_price_*` (4),
   `test_options_handler_parity_violation_falls_back_to_last_price`,
   `test_find_atm_uses_last_price_after_hours`,
   `test_compute_implied_move_nan_ask_no_nan_output`, `test_skew_classification`,
   `test_find_atm_options_*` (2), `test_options_implied_move_basic_math`,
   `test_options_implied_move_zero_spot` — are rewritten against the merged API
   and must fail before implementation. These encode the four options bugs in
   `known_issues.md` (put-call parity guard, after-hours `last_price` fallback,
   NaN/sentinel IV handling, zero-spot). They are the safety net; none are deleted.

2. **Differential test (temporary).** A `network`-marked test runs old
   `get_options_implied_move` and merged `get_options_metrics` against the same
   tickers in the same run and asserts agreement: identical front expiry,
   same-sign skew, implied move within tolerance. This exists only while both
   implementations coexist, and is deleted in the same commit that removes
   `get_options_implied_move` (implementation step 3). It is the primary
   evidence that the merge is faithful — stronger than unit assertions for
   numerical code, because it compares against a live reference.

3. **Regression gates per deliverable.**
   - D2: full suite minus `test_openbb_server.py`; no import of `openbb` survives
     a repo grep.
   - D3: `pytest -m slow` still collects and passes the 14
     `testing/test_sec_xbrl_functions.py` tests — proving the marker survived the
     FinBERT deletion. A repo grep shows no surviving `finbert` or
     `web_traffic` reference.
   - D4: a test asserting `Session_Cache` writes `tool_cache.db` and never opens
     `session.db`; a concurrent-writer test asserting WAL prevents the
     `OperationalError` that the current default journal mode allows.
   - D5: a clean `uv sync` in a scratch venv imports `agent.rag.embedder`
     successfully — the check that catches the missing sentence-transformers pin.

4. **End-to-end acceptance.** One live `/preearnings-research` run on a single
   ticker with a near-dated earnings print. Pass condition: no `data_gap` for
   options inputs, a sentiment field present and agent-produced, and the
   asymmetry classification reached without falling through to `na`.

## Risks

| Risk | Mitigation |
|---|---|
| Merge silently changes an options number that feeds a frozen prediction | Differential test against the live old implementation before deletion |
| `Session_Cache` split breaks a running daemon mid-flight | Daemons must be stopped before D4 lands and restarted after; `known_issues.md` already documents mixed-version daemon behavior |
| A "FinBERT-only" marker or dependency turns out to have other consumers | Already bit once during spec review — the `slow` marker looked FinBERT-specific but has 14 users in `test_sec_xbrl_functions.py`. Every deletion is preceded by a repo-wide grep for consumers, not by reading the thing's own description |
| Skill edit drifts from tool reality | D6 lands last, after tools are final, and is validated by the end-to-end run |

## Out of scope

HTTP transport, Dockerfiles, deployment, splitting `nemo_web`'s RAG tools from
its SEC extractors, moving `state/` behind a service, and full manifest
reconciliation. Each is its own spec.

### Handed off to the primary-source swap

`2026-08-20-primary-source-swap-design.md` owns the following. They were found
during this refactor's audit but are deliberately NOT fixed here, to keep the two
efforts from touching the same files in the same window:

- `get_financial_statements` and `get_forward_estimates` both 403 on the current
  Finnhub tier and silently fall back to yfinance. Four skills each depend on them.
- Re-backing `get_financial_statements` with the repo's own SEC XBRL extractors.
- The 7 unit-scale test failures on `get_revenue_base` / `get_ebitda_margin`. The
  production code is correct — the tests assert a millions-scale range against raw-dollar
  output. **Do not "fix" the extractor in this refactor.**
- `get_revenue_base`, `get_ebitda_margin`, `get_margin_breakdown` being referenced by
  zero skills.

**File boundary.** This refactor touches `tools/financial_modeling_engine/`,
`tools/altdata_server/`, `tools/openbb_server/`, `agent/cache.py`, `state/schema.py`,
and `.claude/skills/preearnings-research/`. The primary-source swap touches
`tools/news_agregator/finnhub_server.py` and `tools/web_search_server/sec_utils.py`.
The only shared surface is the dependency manifests, and the swap adds no dependencies.

**Ordering.** The swap lands after this refactor's D4 cache split — EDGAR extraction is
cache-heavy and would otherwise write cache rows into the state database.

### Deferred, in neither spec

Found during the audit, tracked in `.superpowers/sdd/progress.md`, to be triaged at the
final review rather than absorbed into either effort:

- `close_paper_position` reports failure after successfully closing
  (`tools/alpaca/server.py:415`, `NameError` swallowed by a broad `except`). Highest
  severity finding; belongs on its own branch.
- README documents Alpaca env var names the code does not read.
- `SEC_EMAIL` silently defaults to `analyst@example.com`.
- Test-infrastructure debt: no `pytest-asyncio` (11 tests cannot run), tests depending on
  a missing `.mcp.json` and a missing root `CLAUDE.md` (17 tests), 5 scratch scripts
  erroring on collection, and `test_all_500_revenue` — a sequential 500-ticker live EDGAR
  loop marked `slow` rather than `network`, which became more hazardous once `SEC_EMAIL`
  was set to a real identity.
