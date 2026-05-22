---
name: check-nemo-health
description: Verify all 5 Nemo MCP servers and the SearxNG Docker container are alive and responsive. Polls every 10 seconds for up to 5 minutes if anything is down. Use after launching Claude Code, after a Docker restart, or when a research run is producing suspicious empty results.
---

# /check-nemo-health — End-to-end health check

Run the checks in this exact order. Each subsection lists the **cheapest
possible probe** for that service so the skill stays under ~5 seconds when
everything is healthy.

## Phase 1 — One-shot check

Run all 6 probes **in parallel** (one Bash for the container, five MCP tool
calls for the servers). Collect pass/fail for each. Use these specific
probes — they are chosen because they have low latency and fail loudly
when the service is broken:

| Service           | Probe                                                                 | Pass means                                       |
| ----------------- | --------------------------------------------------------------------- | ------------------------------------------------ |
| SearxNG container | `Invoke-WebRequest http://localhost:8888/healthz -TimeoutSec 3`       | HTTP 200 returned                                |
| `nemo_alpaca`     | `mcp__nemo_alpaca__ping_alpaca`                                       | `{ok: true}` in payload                          |
| `nemo_fred`       | `mcp__nemo_fred__get_treasury_yields`                                 | `data.curve` present, no error in metadata       |
| `nemo_finnhub`    | `mcp__nemo_finnhub__get_basic_financials(ticker="SPY")`               | non-empty `data`, no error                       |
| `nemo_financial`  | `mcp__nemo_financial__get_market_data(ticker="SPY")`                  | `marketCap > 0`                                  |
| `nemo_web`        | `mcp__nemo_web__rag_search(query="health probe", top_k=1)`            | call returns within 2s (warm) or 12s (cold load) |

Render a status table to the user when done:

```
Service          Status   Latency  Notes
----------------------------------------
searxng          PASS     43ms
nemo_alpaca      PASS     180ms
nemo_fred        PASS     290ms
nemo_finnhub     PASS     410ms    Finnhub free-tier latency variable
nemo_financial   PASS     1.1s     yfinance cold-cache hits common
nemo_web         PASS     820ms    rag_search warm
```

Use `FAIL` (not `WARN` or `ERROR`) for any probe that didn't pass. Include
the actual error message from the failing probe in a third row below the
table — full text, not truncated.

## Phase 2 — Triage failures

For each failure, identify the most likely cause and the fix. The known
failure shapes are:

- **searxng FAIL** → Docker Desktop not running. Fix:
  `powershell -File scripts\start_searxng.ps1`. If Docker itself isn't
  installed, point to `README.md`.
- **nemo_alpaca FAIL** with auth error → `ALPACA_PAPER_API_KEY` /
  `ALPACA_PAPER_SECRET_KEY` missing from `.env`.
- **nemo_finnhub FAIL** with auth error → `FINNHUB_API_KEY` missing.
- **nemo_fred FAIL** with auth error → `FRED_API_KEY` missing.
- **nemo_financial FAIL** with yfinance error → yfinance ratelimit
  (try again later) or network issue.
- **nemo_web slow first call (8-12s)** → embedder cold start, NOT a
  failure. Treat as PASS but note "cold start, subsequent calls fast".
- **Any MCP server "tool not found"** → server isn't connected. Tell
  the user to fully restart Claude Code; mention that
  `claude mcp list` should show it as `✓ Connected`.

If everything passes on the first try, stop here. Do not enter Phase 3.

## Phase 3 — Poll until healthy (only if Phase 1 had failures)

Re-run **only the failed probes** every 10 seconds. Use a Bash loop with
`Start-Sleep -Seconds 10` or `sleep 10` between attempts. Cap at **30
retries (5 minutes total)** so this doesn't hang forever.

Between retries, print one line: `retry N/30: <service> still <reason>`.

When all probes pass, print:
```
All services healthy after N retries (X.Xs total).
```

If the cap is hit with services still down, print:
```
TIMEOUT: <list of services> still failing after 5 minutes. Stop and investigate.
```

Do **not** keep polling past the cap. Bail and let the user intervene.

## Hard rules

- **Do not** call expensive tools as probes (`get_supply_chain`,
  `diff_10k`, `calculate_scenario_dcf`). Use only the cheap probes above.
- **Do not** ingest probe results into the RAG corpus or write them to
  any persistent store. This is a diagnostic, not a workflow event.
- **Do not** attempt to restart MCP servers from inside Claude Code.
  Server lifecycle is owned by Claude Code itself — the only fix for a
  dead MCP server is a full Claude Code restart.
- **Do not** count `nemo_web`'s first-call cold start (8-12s) as a
  failure. The embedder warmup is normal. Annotate it but mark PASS.
