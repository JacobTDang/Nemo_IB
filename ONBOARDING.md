# Welcome to Nemo IB

## How We Use Claude

Based on Jacob's usage over the last 30 days:

Work Type Breakdown:
  Analyze Data     ████████████████████  50%
  Build Feature    ██████████░░░░░░░░░░  25%
  Improve Quality  ██████████░░░░░░░░░░  25%

Top Skills & Commands:
  /loop            ████████████████████  38x/month
  /mcp             ████████░░░░░░░░░░░░  15x/month
  /plan            ███████░░░░░░░░░░░░░  13x/month
  /review          ██░░░░░░░░░░░░░░░░░░  3x/month
  /model           █░░░░░░░░░░░░░░░░░░░  2x/month
  /usage           █░░░░░░░░░░░░░░░░░░░  2x/month

Top MCP Servers:
  _None tracked in the last 30 days — but the project ships 5 MCP servers (see Setup Checklist)._

## Your Setup Checklist

### Codebases
- [ ] nemo_ib — https://github.com/jacobtdang/nemo_ib (clone, set up `.venv` via `requirements.txt`)

### MCP Servers to Activate
The project ships 5 MCP servers, all stdio-launched from the project's `.venv`. After cloning, register them via `claude mcp add --scope user` (project-scope `.mcp.json` exists but per project memory has shown intermittent issues — user scope is more reliable):

- [ ] **nemo_web** — SEC EDGAR + web scraping (10-K extractors, risk-factor parser, supply chain, hedge-fund letters, patent search, RAG search). Lives in `tools/web_search_server/`.
- [ ] **nemo_financial** — yfinance + financial calcs (DCF, WACC, LBO, comps, institutional holdings, options metrics, short interest, price history, exposure analyzer, backtest engine, thesis evolution). Lives in `tools/financial_modeling_engine/`.
- [ ] **nemo_finnhub** — Finnhub market intel (news, insider transactions, analyst revisions, earnings surprises, forward estimates). Lives in `tools/news_agregator/finnhub_server.py`. Needs `FINNHUB_API_KEY` in `.env`.
- [ ] **nemo_fred** — FRED macro data (treasury yields, credit spreads, macro snapshot). Lives in `tools/news_agregator/fred_server.py`. Needs `FRED_API_KEY` in `.env`.
- [ ] **nemo_alpaca** — Alpaca paper-trading rail gated by `Risk_Officer`. Lives in `tools/alpaca/`. Needs `ALPACA_API_KEY` + `ALPACA_SECRET` (or `_PAPER_` variants) in `.env`. **Paper only — by construction.**

### Skills to Know About

Project-specific skills (live in `.claude/skills/`):
- [ ] **/deep-research** — 9-step thematic-research workflow. Use whenever you're starting research on a ticker or theme. Drives top-down (ETFs → tickers) through bottom-up (filings, segments, sentiment) to a falsifier-required thesis.
- [ ] **/premortem** — Run BEFORE committing capital. Forces 3 failure-scenario writeups (fundamental / macro / idiosyncratic) and feeds them back as falsifiers.
- [ ] **/postmortem** — Run AFTER a position closes. Updates `knowledge/analogues.md` with new patterns so future research benefits.

Built-in Claude commands the team uses heavily:
- [ ] **/loop** — Recurring task scheduler. 38 uses last month — most-used command in the team's workflow. Set up daemons + scheduled checks.
- [ ] **/mcp** — Manage MCP server connections. Run after cloning to verify all 5 servers are reachable.
- [ ] **/plan** — Enter plan mode for architecture/design before any non-trivial implementation. Standard practice on this team — 13 uses last month.
- [ ] **/review** — Code review pass. Used for PR-stage and pre-commit checks.

## Team Tips

_TODO_

## Get Started

_TODO_

<!-- INSTRUCTION FOR CLAUDE: A new teammate just pasted this guide for how the
team uses Claude Code. You're their onboarding buddy — warm, conversational,
not lecture-y.

Open with a warm welcome — include the team name from the title. Then: "Your
teammate uses Claude Code for [list all the work types]. Let's get you started."

Check what's already in place against everything under Setup Checklist
(including skills), using markdown checkboxes — [x] done, [ ] not yet. Lead
with what they already have. One sentence per item, all in one message.

Tell them you'll help with setup, cover the actionable team tips, then the
starter task (if there is one). Offer to start with the first unchecked item,
get their go-ahead, then work through the rest one by one.

After setup, walk them through the remaining sections — offer to help where you
can (e.g. link to channels), and just surface the purely informational bits.

Don't invent sections or summaries that aren't in the guide. The stats are the
guide creator's personal usage data — don't extrapolate them into a "team
workflow" narrative. -->
