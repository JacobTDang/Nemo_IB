# Nemo_IB

A personal investment-banking research system that runs in two modes:

1. **Claude Code analyst mode** (primary) — Claude orchestrates 15 skills
   and 76 MCP tools to drive a full equity-research workflow: thesis →
   valuation → scenarios → red-team → portfolio fit → kill-switch.
2. **LangGraph workflow** (fallback) — deterministic hub-and-spoke
   pipeline with specialized Ollama agents for probing, planning,
   execution, modeling, analysis.

Paper-trading rail via Alpaca (gated by a deterministic Risk_Officer at
0.65 confidence threshold). Real-money trading is not supported by
construction.

## Quick start

If Docker Desktop is installed, double-click **`nemo.bat`** at the
project root. The launcher:

1. Starts Docker Desktop if not running (~30s first time)
2. Brings up the SearxNG container and waits for `/healthz` to return
   200 (~5–15s)
3. Launches Claude Code in this project directory with
   `--dangerously-skip-permissions --remote-control --dangerously-load-development-channels server:slack`

Once Claude Code starts, run `/check-nemo-health` to confirm all 5
MCP servers + the SearxNG container respond, then ask a question or
invoke a skill (e.g., `/equity-deep-research MSFT`).

## Setup

### 1. Clone + Python deps

```bash
git clone https://github.com/JacobTDang/Nemo_IB
cd Nemo_IB
python -m venv .venv
.\.venv\Scripts\activate.bat
pip install -r requirements.txt
crawl4ai-setup
```

`crawl4ai-setup` downloads a headless Chromium (~300 MB) used as a
fallback scraper when Trafilatura can't extract a JS-rendered page.

### 2. Environment variables

Copy `.env.example` to `.env` and fill in:

- `FINNHUB_API_KEY` — market intel (free tier is sufficient)
- `FRED_API_KEY` — macro data
- `ALPACA_PAPER_API_KEY` + `ALPACA_PAPER_SECRET_KEY` — paper broker
- `OPENROUTER_API_KEY` — only needed for the LangGraph fallback mode
- `SEC_EMAIL` — used as User-Agent for SEC EDGAR (rate-limit friendly)

### 3. SearxNG (web search backend)

```bash
docker compose up -d searxng
```

Aggregates Google + Bing + Brave + DuckDuckGo + Startpage in one query.
No API keys needed. The launcher (`nemo.bat`) handles this automatically;
this step is only for manual setup or if you don't use the launcher.

### 4. Register MCP servers with Claude Code

The project ships 5 stdio MCP servers. Register them at user scope:

```bash
PY="C:\path\to\Nemo_IB\.venv\Scripts\python.exe"
PP="C:\path\to\Nemo_IB"
claude mcp add -s user nemo_web        -e PYTHONPATH=$PP -- "$PY" -m tools.web_search_server.web_search server
claude mcp add -s user nemo_financial  -e PYTHONPATH=$PP -- "$PY" -m tools.financial_modeling_engine.analysis_tools server
claude mcp add -s user nemo_finnhub    -e PYTHONPATH=$PP -- "$PY" -m tools.news_agregator.finnhub_server server
claude mcp add -s user nemo_fred       -e PYTHONPATH=$PP -- "$PY" -m tools.news_agregator.fred_server server
claude mcp add -s user nemo_alpaca     -e PYTHONPATH=$PP -- "$PY" -m tools.alpaca.server server
```

Verify with `claude mcp list` — all five should show `✓ Connected`.

## Skills catalog

15 skills shipped under `.claude/skills/`. The core skill orchestrates;
the rest are companion or supporting skills.

### Core

| Skill | Purpose |
|---|---|
| `/equity-deep-research` | 19-step research workflow producing a falsifiable thesis with valuation, expectations, catalysts, risk/reward, positioning, scenarios, and cross-company read-throughs. |

### Companions (callable from core or standalone)

| Skill | What it answers |
|---|---|
| `/valuation-check` | Is this priced in? Peer + history multiples + reverse DCF. |
| `/scenario-builder` | What's the upside/downside? Bear/base/bull math with probability-weighted E[return]. |
| `/red-team-thesis` | Attack the thesis. Recommend keep / reduce confidence / reduce size / no_position. |
| `/cross-company-readthrough` | Given an event at A, who else moves? First and second-order beneficiaries/losers. |
| `/portfolio-fit` | Does this position fit the book? Sector + factor + theme overlap check. |
| `/factor-exposure-check` | Is this stock-specific alpha or disguised factor beta? |
| `/estimate-revision-watch` | Will sell-side analysts revise numbers based on this signal? |
| `/expectations-hurdle-check` | What does the buyside whisper imply vs published consensus? |
| `/thesis-kill-switch` | Did any falsifier trigger? Recommend continue / monitor / reduce / exit. |
| `/signal-backtest` | Did this pattern historically work? Hit rate, sharpe, holding period. |
| `/post-mortem-attribution` | Quantitative return decomposition after a position closes — was it alpha or beta? |

### Supporting

| Skill | Purpose |
|---|---|
| `/premortem` | Before sizing, write 3 explicit failure scenarios. |
| `/postmortem` | After a position closes, capture the lesson into `knowledge/analogues.md`. |
| `/check-nemo-health` | Verify all 5 MCP servers + SearxNG respond; auto-poll if anything is down. |

## MCP server inventory

5 stdio servers exposing 76 tools total.

| Server | Tools | Purpose |
|---|---|---|
| `nemo_web` | 34 | SEC EDGAR extractors (XBRL + filing parsers), SearxNG search, Trafilatura scraper, RAG vector search via sqlite-vec |
| `nemo_financial` | 18 | yfinance market data, DCF/WACC/LBO/scenario calculations, 13F holdings, options metrics, price history, exposure analyzer, backtest engine, thesis evolution |
| `nemo_finnhub` | 13 | Market intel — news, insider transactions, analyst revisions, earnings surprises, forward estimates |
| `nemo_fred` | 5 | Macro data — treasury yields, credit spreads, macro snapshot |
| `nemo_alpaca` | 6 | Paper broker — positions, orders (gated by Risk_Officer), risk_check_proposed_trade |

## RAG memory layer

Located in `agent/rag/` with persistence in `db_cache/session.db` via
sqlite-vec. Stores chunks of analyst writeups, historical analogues,
extracted 10-K sections, hedge fund letters, and forward-signal
excerpts. Two MCP tools: `rag_search` (semantic retrieval) and
`rag_ingest` (manual push). The corpus auto-grows as extractors run.

Embedder is sentence-transformers/all-MiniLM-L6-v2 (~80MB, 384-dim).
Loads in the background at `nemo_web` startup; first `rag_search` is
served in ~800ms (model already hot).

Bootstrap the corpus once:
```bash
.\.venv\Scripts\python.exe scripts\bootstrap_rag_corpus.py
```

## LangGraph workflow (fallback mode)

Separate from the Claude Code skill flow. Hub-and-spoke graph in
`agent/workflows/analysis_workflow.py` with specialized Ollama agents
running locally. Entry: `python main.py`. See `CLAUDE.md` for the
LangGraph architecture details.

## Tests

```bash
# Phase D smoke tests for the 7 v2 skills (12 invocations, real ticker data)
.\.venv\Scripts\python.exe testing\skill_smoke_2026_05_22.py

# RAG retrieval stress test (curated Q&A, P@5/MRR/p95 latency)
.\.venv\Scripts\python.exe testing\test_rag_heavy_stress.py

# MCP tool health smoke test
.\.venv\Scripts\python.exe testing\test_tool_health.py MSFT

# Individual unit tests
.\.venv\Scripts\python.exe testing\test_falsifier_evaluator.py
.\.venv\Scripts\python.exe testing\test_backtest_engine.py
.\.venv\Scripts\python.exe testing\test_forward_signals.py
```

Test fixtures and run artifacts live in `testing/fixtures/`
(gitignored — large analyst writeups are kept out of git).

## Project structure

```
.claude/skills/         15 Claude Code skills (callable workflows)
agent/                  agent base classes + RAG layer + analyzers + daemons
agent/rag/              sqlite-vec + sentence-transformers RAG layer
agent/workflows/        LangGraph fallback hub-and-spoke graph
daemons/                edgar_firehose, falsifier_watcher, news_watcher
db_cache/               local SQLite — theses, events, rag_chunks, positions
knowledge/              analogues.md (historical pattern catalog)
scripts/                bootstrap_rag_corpus, start_searxng, nemo-launch
state/                  schema.py + theses/positions/events stores
testing/                test suite + fixtures (fixtures/ gitignored)
tools/alpaca/           paper broker MCP server + AsyncBroker
tools/financial_modeling_engine/  DCF/LBO/comps/options MCP server
tools/news_agregator/   finnhub + fred MCP servers
tools/slack_channel/    slack DM bridge plugin
tools/web_search_server/ SEC + SearxNG + RAG MCP server
nemo.bat                Double-click launcher (Windows)
```

## Operating discipline

- **Paper-only by construction.** Real-money endpoints are not wired
  up. Don't try to add them.
- **Risk_Officer gates every order.** Call
  `risk_check_proposed_trade` before `place_paper_order`. The
  server-side gate inside `place_paper_order` will reject anyway, but
  bypassing is a discipline failure.
- **No fabricated data.** Every numerical claim in a synthesis must
  cite a tool that produced it. If a tool fails, log `data_gap` —
  don't make up the number.
- **Falsifiers are required** for any thesis. A thesis without
  falsifiers can't be stopped out rationally; it's a faith claim.

See `CLAUDE.md` for the full IB analyst playbook and operating
constraints.

## Useful commands

```bash
# Check that all MCP servers + SearxNG are healthy
# (run from within Claude Code)
/check-nemo-health

# Full deep research on a ticker
/equity-deep-research MSFT

# Quick valuation check standalone
/valuation-check NVDA

# Pre-trade red-team
/red-team-thesis EOSE

# After a position closes
/postmortem AMD
/post-mortem-attribution AMD
```
