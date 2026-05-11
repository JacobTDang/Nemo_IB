# Nemo_IB

Excel-first MCP-enabled financial analysis agent. Hub-and-spoke LangGraph workflow
with specialized agents for probing, planning, execution, modeling, and analysis.

## Setup

### 1. SearXNG (web search backend)

The agent uses a local SearXNG container for web search. SearXNG aggregates results
from Google, Bing, Brave, DuckDuckGo, Startpage, and Qwant in one query — no API keys
needed.

```bash
docker compose up -d searxng
```

Verify it's running:

```bash
curl "http://localhost:8888/search?q=test&format=json"
```

If you want to change which engines run, edit `searxng/settings.yml` and restart:
`docker compose restart searxng`.

### 2. Python dependencies

```bash
pip install -r requirements.txt
crawl4ai-setup
```

The `crawl4ai-setup` step downloads a headless Chromium (~300 MB) used only as a
fallback when Trafilatura can't extract a JS-rendered page. One-time install.

### 3. Environment variables

Copy `.env.example` to `.env` and fill in API keys for Finnhub, FRED, and OpenRouter.

### 4. Run

```bash
python main.py
```

Type your query. The agent extracts the ticker automatically.

## Architecture

- `agent/Master_Orchestrator.py` — hub orchestrator (Nemotron via OpenRouter)
- `agent/Probing_Agent.py` — identifies data requirements
- `agent/Orchestrator_Agent.py` — plans MCP tool execution
- `agent/Financial_Modeling_Agent.py` — runs scenario DCF, LBO, credit, DDM, sensitivity
- `agent/Financial_Analysis_Agent.py` — synthesizes models and qualitative data
- `tools/web_search_server/` — SearXNG + Trafilatura/Crawl4AI scraper
- `tools/financial_modeling_engine/` — pure-math valuation functions
- `tools/web_search_server/sec_utils.py` — SEC XBRL extractors
- `tools/news_agregator/finnhub_server.py` — Finnhub market intelligence
- `tools/news_agregator/fred_server.py` — FRED macro data

## Tests

```bash
python testing/test_fix_NN_*.py        # unit tests for specific fixes
python testing/test_scraper_NN_*.py    # scraper refactor tests
```
