---
name: check-nemo-health
description: Verify the five deployed Nemo MCP data servers (sec, financial, finnhub, fred, altdata) are registered, reachable, authenticated, and actually serving tools — and that the congressional store is populated. Separates "not registered" / "nothing listening" / "401" / "serving", which are routinely confused. Use after a Docker restart, after a Claude Code restart, when a research run returns suspicious empty results, or before trusting a deployment as healthy.
---

# /check-nemo-health — Is the deployed stack actually serving?

Five servers, **96 tools**, streamable HTTP on `127.0.0.1:8810-8814`.
The failure this skill exists to prevent is believing a deployment is
healthy while a service is absent or dead — so it checks every server
that ships, and none that does not.

## What is deployed

| Server | Port | Registered as | Tools | Probe |
| --- | --- | --- | ---: | --- |
| sec | 8810 | `nemo-sec` | 48 | `get_sic_code(ticker="AAPL")` |
| financial | 8811 | `nemo-financial` | 20 | `get_market_data(ticker="SPY")` |
| finnhub | 8812 | `nemo-finnhub` | 14 | `get_company_profile(ticker="AAPL")` |
| fred | 8813 | `nemo-fred` | 5 | `get_treasury_yields()` |
| altdata | 8814 | `nemo-altdata` | 9 | `get_congress_coverage()` |

The MCP namespace is **hyphenated**: `mcp__nemo-sec__get_sic_code`,
`mcp__nemo-altdata__get_congress_coverage`. Underscore forms
(`mcp__nemo_web__`, `mcp__nemo_alpaca__`) are from an older layout and
resolve to nothing.

**Absent on purpose — never probe these:**

- `alpaca` (6 tools, places orders). A data-source host holds no
  positions and must not be able to trade, so the image excludes it.
  A check that expects alpaca reports a *correct* deployment as broken.
- `excel` (3 tools), `sentry` (19, reads book state).
- SearXNG and the RAG stack. The sec server declares 50 tools and
  serves 48: `search`, `rag_search` and `rag_ingest` are hidden because
  advertising a `search` that silently returns nothing would make an
  absent container look like a query that matched nothing. Their
  absence is the design.

## The four states, and why they get confused

A server is in exactly one of these. Naming the wrong one sends the
next reader debugging the wrong thing.

| State | Signature | Fix |
| --- | --- | --- |
| **Not registered** | absent from `claude mcp list` | `claude mcp add`, then restart Claude Code |
| **Nothing listening** | `ConnectionRefused` — `/health` refuses the TCP connection | container down or port not published; `docker compose up -d` |
| **Rejecting auth** | `/health` answers 200, `/mcp/` answers **401** | registered bearer token ≠ server's `NEMO_MCP_TOKEN` |
| **Serving** | handshake + a real tool call both succeed | — |

**The trap.** A stale registration pointing at a dead port has been
reported as `Auth: ✘ not authenticated` / `SDK auth failed`. Nothing
was listening; the auth line was a symptom. Chasing the token there
wastes the whole session. Settle it with the ports before touching a
token:

```bash
# The server's own token, so this probes the port and not the client config.
# It is supplied to compose from the shell, so it is not always in .env.
NEMO_MCP_TOKEN=$(docker exec nemo-sec printenv MCP_AUTH_TOKEN)
for p in 8810 8811 8812 8813 8814; do
  printf '%s ' "$p"
  curl -s -m 3 -o /dev/null -w 'health=%{http_code} ' "http://127.0.0.1:$p/health"
  curl -s -m 5 -o /dev/null -w 'mcp=%{http_code}\n' -X POST "http://127.0.0.1:$p/mcp/" \
    -H "Authorization: Bearer $NEMO_MCP_TOKEN" \
    -H 'Content-Type: application/json' \
    -H 'Accept: application/json, text/event-stream' \
    -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"health","version":"1"}}}'
done
```

If the `docker exec` itself fails with `cannot exec in a stopped
container`, stop there: the stack is down, and no token question
arises. Run the `/health` half of the loop alone to see how far it goes.

- `health=000` (curl exit 7) → **nothing listening**. Not auth.
- `health=200 mcp=401` → **the token is wrong**, and only this is auth.
  `/health` is exempt from auth by design, which is what makes the pair
  diagnostic.
- `health=200 mcp=200` → the port and the token are fine; go to step 2.

`claude mcp list` does distinguish the two — `ConnectionRefused: Unable
to connect` versus `Server rejected the configured Authorization header
(HTTP 401)`. Read the whole line, not the `✘`.

On a real 401, compare the two ends — they must be identical:

```bash
docker exec nemo-sec printenv MCP_AUTH_TOKEN                      # server
claude mcp get nemo-sec | sed -n 's/.*Authorization: Bearer //p'  # client
```

The client's copy lives in `~/.claude.json` in plaintext; re-add the
server with the correct header rather than editing it by hand.

## Workflow

### 1. Confirm registration

```bash
claude mcp list 2>&1 | grep nemo
docker ps --filter name=nemo- --format '{{.Names}}\t{{.Status}}\t{{.Ports}}'
```

Expect exactly five `nemo-*` entries, all `✔ Connected`, and five
containers `Up … (healthy)`.

- **Four entries, no `nemo-altdata`** → the most likely gap. A stack
  registered before altdata shipped is missing 9 tools including the
  entire congressional pipeline, and every congress question will
  answer "tool not found" rather than "no data". Re-add per
  `README.md` → *Registering with an MCP client*.
- **`nemo_alpaca` / `nemo_web` present** → stale registrations from the
  old layout. Remove them.
- A newly added server does not appear until Claude Code restarts.

### 2. Probe each server with a real tool call

Run the five probes **in parallel**. `/health` is not a probe: it
reports only that uvicorn bound the port and the app started, and a
server whose MCP layer failed still answers it 200. Only a tool call
proves the thing serves.

| Probe | Pass means | Latency |
| --- | --- | --- |
| `get_sic_code("AAPL")` | `sic == "3571"` | ~0.1s |
| `get_market_data("SPY")` | `currentPrice > 0` | ~0.2s, ~1s cold |
| `get_company_profile("AAPL")` | `data.name` present | ~0.1s |
| `get_treasury_yields()` | `data.curve["10Y"]` present | **~7s**, 0.4s cached |
| `get_congress_coverage()` | `data.total` is an integer | ~0.01s |

Every payload carries `success`. Treat `success: false` as FAIL and
quote `metadata.errors` verbatim.

Two ways to false-fail a healthy server:

- **Timing out fred.** `get_treasury_yields` makes 10 FRED calls and
  takes 7-10s uncached. The tool cache lives in a tmpfs, so every
  container restart throws it away — a slow fred right after
  `docker compose up` is expected. Allow 20s before calling it a
  failure; only altdata's local read is genuinely instant.
- **Asserting `marketCap > 0` on SPY.** An ETF has no market cap and
  yfinance returns `null`. Check `currentPrice` instead.

### 3. Check the congressional store separately

altdata answering is not the same as the store holding anything.
`get_congress_coverage` reports what it holds. Three states:

- **Populated** — `data.total > 0`, split `parsed` / `unparsed`.
  `unparsed` are filings submitted on paper and carry no extractable
  text; they never become readable on a retry, so `complete: false` is
  permanent and normal. Not a fault.
- **Empty, schema present** — `success: true`, `data.total: 0`, and
  `data.note` opens *"The congressional disclosure store is empty, so
  this is not a statement about what was traded"* and names the sync
  command. `get_congress_trades`, `get_congress_holdings` and
  `get_congress_leaderboard` additionally set `data.store_empty: true`.
  **This is expected on a fresh deployment.** Report it as *unsynced*,
  never as broken, and give the command:
  ```bash
  NEMO_MCP_TOKEN=$(docker exec nemo-sec printenv MCP_AUTH_TOKEN) \
    docker compose -f deploy/docker-compose.yml run --rm congress-sync \
      --house 2024 2025 2026 --senate --senate-annual
  ```
  Roughly 40 minutes of throttled requests. Safe to re-run and to cron.
  `congress-sync` inherits the shared service anchor, so it **refuses to
  start without `NEMO_MCP_TOKEN`** — `required variable NEMO_MCP_TOKEN
  is missing a value` — even though it binds no port and authenticates
  nothing. On a host where the token was only ever passed at
  `compose up` time it is not in the environment, and the bare command
  in the READMEs fails. That is a missing variable, not a broken sync.
- **No store at all** — `success: false` with
  `metadata.errors: ["OperationalError: no such table: filings"]`. This
  **is** a fault: `NEMO_CONGRESS_DB` points where the schema was never
  created, or the `congress-data` volume is not mounted. Syncing will
  not fix a path problem.

`get_congress_coverage` never sets `store_empty` — it signals empty
through `data.total: 0` plus the note. Keying only on `store_empty`
misses the coverage tool entirely.

The shell equivalent, when altdata itself is the thing in doubt:

```bash
NEMO_MCP_TOKEN=$(docker exec nemo-sec printenv MCP_AUTH_TOKEN) \
  docker compose -f deploy/docker-compose.yml run --rm congress-sync --status
```

### 4. Poll only if a failure can change

Re-run **only** the failed probes every 10 seconds, capped at 30
retries (5 minutes). One line per retry:
`retry N/30: <server> still <reason>`. Bail at the cap with
`TIMEOUT: <servers> still failing after 5 minutes.`

Poll only states that time can fix — a container still reporting
`health: starting`, or a cold first call. A missing registration, a
wrong token and a missing congress schema read identically on retry 30;
polling them just delays the report.

## Output

```
## /check-nemo-health

server     registered       listening  auth  tools  probe  latency
sec        ✔ nemo-sec       ✔ :8810    ✔     48     PASS   0.09s
financial  ✔ nemo-financial ✔ :8811    ✔     20     PASS   0.20s
finnhub    ✔ nemo-finnhub   ✔ :8812    ✔     14     PASS   0.07s
fred       ✔ nemo-fred      ✔ :8813    ✔     5      PASS   6.68s
altdata    ✔ nemo-altdata   ✔ :8814    ✔     9      PASS   0.01s

96 tools across 5 servers.
congress store: 2143 filings, 1894 parsed, 249 paper scans — populated

Not deployed (expected): alpaca, excel, sentry, SearXNG/RAG.

Failures:
<server>: <state from the four-state table> — <verbatim error>
<the one command that fixes it>
```

Report the tool count per server. A server that connects but serves
fewer tools than the table above is a partially-started server, which
looks healthy on every other line.

## Hard rules

- **Never probe alpaca.** Excluded from the image on purpose; expecting
  it turns a correct deployment into a red report.
- **Never omit altdata.** A skipped server reads as a healthy stack
  while a real service is absent — the failure this skill was rewritten
  to fix.
- **`/health` 200 is never proof.** It means the port is bound. Nothing
  more.
- **Say "refused" or "401", never a bare "auth failure".** Run the curl
  pair first.
- **An unsynced congressional store is not a failure.** Report it as
  unsynced with the sync command.
- Cheap probes only. Never `get_supply_chain`, `diff_10k`,
  `calculate_scenario_dcf`, or a wide-window `get_congress_trades`.
- Do not restart servers from inside a skill run. Lifecycle belongs to
  `docker compose`; a registration change needs a Claude Code restart.
- Do not ingest probe results into any store. This is a diagnostic, not
  a workflow event.

## When to invoke

- After `docker compose up -d` / `restart`, or after a Claude Code restart
- After adding or re-pointing a registration
- When a research run returns suspicious empties — especially congress
  tools, where an unsynced store and a dead server look alike
- Before asserting anywhere that the deployment is healthy

## When to skip

- Mid-research while tools are already returning data
- To diagnose one tool's *wrong answer* — that is a data-source
  question, not a health question
