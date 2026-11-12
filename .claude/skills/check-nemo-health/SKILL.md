---
name: check-nemo-health
description: Verify the deployed Nemo stack on the homelab VM — six MCP data servers (sec, financial, finnhub, fred, altdata, quiver) behind one gateway, 111 tools, reached over Tailscale through the single `nemo` Claude Code entry — is reachable, registered, authenticated, and actually serving tools, and that the congressional store is populated. Separates "Tailscale down" / "not registered" / "nothing listening" / "401" / "serving", which are routinely confused. Use after a deploy or restart on the VM, after a Claude Code restart, when a research run returns suspicious empty results, or before trusting a deployment as healthy.
---

# /check-nemo-health — Is the deployed stack actually serving?

Six servers and a gateway, **111 tools**, streamable HTTP on the Proxmox VM
`nemo-finn-server`, reached over Tailscale: `ssh jacob@100.118.253.108`. Code
at `/srv/nemo`; compose runs from `/srv/nemo/deploy` as
`docker compose --env-file ../.env ...`. Ports publish on the tailnet IP
**only** — `127.0.0.1:8810-8816` refuses on the Mac and on the VM, so a check
aimed at loopback reports a healthy stack as dead.

The failure this skill exists to prevent is believing a deployment is
healthy while a service is absent or dead — so it checks every server
that ships, and none that does not.

## What is deployed

| Service | Container | Port | Tools | Probe |
| --- | --- | --- | ---: | --- |
| sec | `nemo-sec` | 8810 | 48 | `get_sic_code(ticker="AAPL")` |
| financial | `nemo-financial` | 8811 | 20 | `get_market_data(ticker="SPY")` |
| finnhub | `nemo-finnhub` | 8812 | 14 | `get_company_profile(ticker="AAPL")` |
| fred | `nemo-fred` | 8813 | 5 | `get_treasury_yields()` |
| altdata | `nemo-altdata` | 8814 | 9 | `get_congress_coverage()` |
| quiver | `nemo-quiver` | 8816 | 15 | `get_quiver_congress_trades(limit=1)` |
| nemo (gateway) | `nemo-gateway` | 8815 | 111 | — |

Claude Code on the Mac has **one** user-scope entry, `nemo` →
`http://nemo-finn-server:8815/mcp/`, with a bearer header. Tools are
`mcp__nemo__<tool>` (`mcp__nemo__get_quiver_congress_trades`). The per-server
entries (`nemo-sec`, ...), their `mcp__nemo-sec__` names and the underscore
forms (`mcp__nemo_web__`) are from older layouts and resolve to nothing.

**The gateway hides a dead server.** It reads every server's tool list once
and keeps it, so a server that dies later leaves the list at 111 and the
entry `✔ Connected`; only calls to its tools fail, with `the <name> server
did not answer <tool>`. (A gateway that starts while a server is down lists
nothing and names it.) Step 2 checks each server behind it.

**Absent on purpose — never probe these:**

- `alpaca` (6 tools, places orders). A data-source host holds no
  positions and must not be able to trade, so the image excludes it.
  A check that expects alpaca reports a *correct* deployment as broken.
- `excel` (3 tools), `sentry` (19, reads book state).
- SearXNG and the RAG stack. The sec server declares 51 tools and
  serves 48: `search`, `rag_search` and `rag_ingest` are hidden because
  advertising a `search` that silently returns nothing would make an
  absent container look like a query that matched nothing.

## The five states, and why they get confused

Naming the wrong one sends the next reader debugging the wrong thing.

| State | Signature | Fix |
| --- | --- | --- |
| **Tailscale down on the Mac** | `tailscale status` prints `Tailscale is stopped.` or has no `nemo-finn-server` line; `ssh` and every port fail | Start Tailscale on the Mac. The box is fine. |
| **Not registered** | no `nemo` line in `claude mcp list` | Re-add the `nemo` entry, restart Claude Code |
| **Nothing listening** | Tailscale up, `/health` refused (`000`, curl exit 7) | Container down; `nemo services` on the VM |
| **Rejecting auth** | `/health` 200, `/mcp/` **401** | The Mac's token ≠ the server's |
| **Serving** | a tool list and a real tool call both succeed | — |

**The trap.** A stale registration pointing at a dead port has been
reported as `Auth: ✘ not authenticated` / `SDK auth failed`; nothing was
listening. A stopped Tailscale on the Mac — found more than once — looks the
same from Claude Code. Settle reachability before touching a token. (If
Tailscale shows `nemo-finn-server` as `offline`, the VM itself is down.)

## Workflow

### 1. The Mac: Tailscale, the entry, the path

```bash
tailscale status | grep nemo-finn-server
claude mcp list 2>&1 | grep nemo
for p in 8810 8811 8812 8813 8814 8816 8815; do
  printf '%s ' "$p"; curl -s -m 5 -o /dev/null -w 'health=%{http_code}' "http://nemo-finn-server:$p/health"; echo " curl=$?"
done
# The registered token, read inside the command and never printed.
curl -s -m 10 -o /dev/null -w 'mcp=%{http_code}\n' -X POST http://nemo-finn-server:8815/mcp/ \
  -H "Authorization: Bearer $(claude mcp get nemo | sed -n 's/.*Authorization: Bearer //p')" \
  -H 'Content-Type: application/json' -H 'Accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'
```

Expect a `nemo-finn-server` line that is not `offline`; one entry,
`nemo: http://nemo-finn-server:8815/mcp/ (HTTP) - ✔ Connected`; seven
`health=200 curl=0`; `mcp=200`. Trust curl's exit code only once Tailscale
shows the box. `health=000 curl=7` is **nothing listening**, not auth.
`health=200` with `mcp=401` is **the token is wrong**, and only that is auth:
`/health` is exempt from auth, which makes the pair diagnostic.

**No `nemo` entry** → add one user-scope HTTP entry `nemo` for that URL with
`Authorization: Bearer <NEMO_MCP_TOKEN>` (`deploy/README.md` → *Registering
with an MCP client*); the token is in `/srv/nemo/.env` on the VM, and the
Mac's `.env` has none. Remove any `nemo-*` or `nemo_*` entries. Restart
Claude Code after either change.

On a 401, compare the two ends without printing either token:

```bash
srv=$(ssh jacob@100.118.253.108 'docker exec nemo-gateway printenv MCP_AUTH_TOKEN' | shasum -a 256)
cli=$(claude mcp get nemo | sed -n 's/.*Authorization: Bearer //p' | shasum -a 256)
[ "$srv" = "$cli" ] && echo "tokens match" || echo "tokens differ"
```

`differ` on an entry that connects is an agent token from
`NEMO_MCP_AGENT_TOKENS`, not a fault. Fix a wrong token by re-adding the entry.

### 2. The box: containers and each server

```bash
ssh jacob@100.118.253.108 nemo services
```

It reads the servers from the compose file and exits non-zero unless all six
and the gateway (`nemo`) are running and healthy; a server with no container
gets a `missing` row. `nemo-bot` (the Discord bot) shows health `-`: normal.
Healthy means `/ready` passed. For an unhealthy row, `blocking` names why:

```bash
ssh jacob@100.118.253.108 "docker inspect --format '{{range .State.Health.Log}}{{.ExitCode}} {{.Output}}{{println}}{{end}}' nemo-sec | grep . | tail -1"
```

Then list every server's tools from inside the compose network (the image has
python and httpx, not curl). The script reads `nemo-sec`'s own
`MCP_AUTH_TOKEN`, so no token crosses a command line and the result describes
the box, not the Mac. If `nemo-sec` is down, use `nemo-gateway`.

```bash
ssh jacob@100.118.253.108 'docker exec -i nemo-sec python -' <<'EOF'
import json, os, httpx
H = {"Authorization": "Bearer " + os.environ["MCP_AUTH_TOKEN"],
     "Accept": "application/json, text/event-stream"}
for svc in ["sec", "financial", "finnhub", "fred", "altdata", "quiver", "nemo"]:
    try:
        health = httpx.get(f"http://{svc}:8080/health", timeout=3).status_code
        r = httpx.post(f"http://{svc}:8080/mcp/", headers=H, timeout=60,
                       json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    except httpx.HTTPError as exc:
        print(f"{svc:10} {type(exc).__name__}: {exc}")
        continue
    data = [l[5:] for l in r.text.splitlines() if l.startswith("data:")]
    tools = len(json.loads(data[0])["result"]["tools"]) if r.status_code == 200 else "-"
    print(f"{svc:10} health={health} mcp={r.status_code} tools={tools}")
EOF
```

Expect `health=200 mcp=200` and the table's counts. A name that does not
resolve has no container.

### 3. Probe each server with a real tool call

Call the six probes as `mcp__nemo__*` tools, **in parallel**: this proves
the whole path research runs use. `/health` is not a probe: it reports only
that uvicorn bound the port, and a server whose MCP layer failed answers 200.

| Probe | Pass means | Latency |
| --- | --- | --- |
| `get_sic_code("AAPL")` | `sic == "3571"` | ~0.3s |
| `get_market_data("SPY")` | `currentPrice > 0` | ~0.3s, ~1s cold |
| `get_company_profile("AAPL")` | `data.name` present | ~0.2s |
| `get_treasury_yields()` | `data.curve["10Y"]` present | **~7s**, <1s cached |
| `get_congress_coverage()` | `data.total` is an integer | ~0.1s |
| `get_quiver_congress_trades(limit=1)` | `data.trades` non-empty | ~0.1s cached (15 min) |

Every payload carries `success`. Treat `success: false` as FAIL and quote
the error verbatim (`error`, `metadata.errors` or `data.reason`, by server).

Three ways to false-fail a healthy server:

- **Timing out fred.** 10 FRED calls, 7-10s uncached, and the cache is a
  tmpfs that every restart empties. Allow 20s.
- **Asserting `marketCap > 0` on SPY.** An ETF has none; yfinance returns
  `null`. Check `currentPrice`.
- **Blaming the box for Quiver.** Quiver is quiverquant.com's web pages,
  scraped, behind Cloudflare. A probe can fail with `data.reason`
  `rate_limited` (Cloudflare answered 403 or 429) or `provider_unavailable`
  (any other upstream failure) while `nemo-quiver` is healthy. Report Quiver
  unavailable upstream, not the box broken. `parse_failed` means Quiver
  changed a page: a parser fault, still not the box.

### 4. Check the congressional store separately

altdata answering is not the store holding anything. `get_congress_coverage`:

- **Populated** — `data.total > 0`, split `parsed` / `unparsed`, with a
  `last_successful_sync` per chamber. `unparsed` are paper filings with no
  extractable text, so `complete: false` is permanent and normal. Cron syncs
  at 06:00 UTC daily; a sync more than a day old means that run failed, and
  past 3 days the congress tools add a `stale_chamber_sync` warning.
- **Empty** — `data.total: 0`, and `data.note` opens *"The congressional
  disclosure store is empty, so this is not a statement about what was
  traded"*. The trades, holdings and leaderboard tools also set
  `data.store_empty: true`; coverage does not. On a fresh deployment this is
  *unsynced*, not broken (backfill in `deploy/README.md`); on this box it is a
  fault. The store creates its schema on first connect, so an unmounted
  volume reads as empty too, not as `no such table`.

The same store read through the `congress-data` volume: totals, each chamber
with its `last_successful_sync`, and member, transaction and holding counts.
Filings here while altdata reports 0 means altdata is not reading the volume.
Sync only through `congress-sync`: `nemo-altdata` has no `SEC_EMAIL`.

```bash
ssh jacob@100.118.253.108 'cd /srv/nemo/deploy && docker compose --env-file ../.env run --rm -T congress-sync --status'
```

### 5. Poll only if a failure can change

Re-run **only** the failed probes every 10 seconds, at most 30 times:
`retry N/30: <server> still <reason>`, then `TIMEOUT: <servers> still failing
after 5 minutes.` Poll only what time can fix — `health: starting`, a cold
call, a Quiver rate limit — never a stopped Tailscale, a missing entry or a
wrong token.

## Output

```
## /check-nemo-health
tailscale ✔ nemo-finn-server    entry ✔ nemo → :8815 Connected

server     listening  auth  tools  probe  latency
sec        ✔ :8810    ✔     48     PASS   0.30s
financial  ✔ :8811    ✔     20     PASS   0.27s
finnhub    ✔ :8812    ✔     14     PASS   0.17s
fred       ✔ :8813    ✔     5      PASS   0.46s
altdata    ✔ :8814    ✔     9      PASS   0.08s
quiver     ✔ :8816    ✔     15     PASS   0.10s
gateway    ✔ :8815    ✔     111    —      —

congress store: 1594 filings, 1402 parsed, 192 paper scans, synced 06:00 UTC — populated
Not deployed (expected): alpaca, excel, sentry, SearXNG/RAG.

Failures:
<server>: <state from the five-state table> — <verbatim error>
<the one command that fixes it>
```

Report the tool count per server. A server that serves fewer tools than the
table is partially started, and looks healthy on every other line.

## Hard rules

- **Never probe alpaca.** Expecting it turns a correct deployment red.
- **Never omit a server.** A skipped server reads as a healthy stack, and
  the gateway's 111 does not cover for a dead one.
- **`/health` 200 is never proof**, and `127.0.0.1` is never the address.
- **Say "Tailscale down", "refused" or "401", never "auth failure".**
- **Never print a token or an `.env` value.** Read it inside the command.
- Cheap probes only. Never `get_supply_chain`, `diff_10k`,
  `calculate_scenario_dcf`, or a wide-window `get_congress_trades`.
- Read-only, and no probe result goes into any store. Never restart, rebuild
  or recreate. Never run research jobs, `research-status` or `nemo status` —
  they post to Discord.

## When to invoke

- After a deploy or restart on the VM, a Claude Code restart, or a change to
  the `nemo` entry
- When a research run returns suspicious empties — especially congress
  tools, where an unsynced store and a dead server look alike
- Before asserting anywhere that the deployment is healthy

Skip it mid-research while tools are already returning data, and for one
tool's *wrong answer* — that is a data-source question, not a health one.
