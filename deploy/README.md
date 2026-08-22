# Running the data-source servers

Stateless by design. A request comes in, a response goes out, and nothing
survives the container.

## Why nothing persists

Two paths are written at runtime:

| Path | What writes it | Growth |
|---|---|---|
| `/root/.edgar` | edgartools' filing cache | **2MB after a single filing**, unbounded across tickers |
| `/app/db_cache` | tool cache and session schema | slow, but unbounded |

Both are mounted as size-capped tmpfs, so they live in RAM, cannot grow past
their cap, and are freed when the container exits. Measured with the mounts in
place: the image layer stays untouched apart from the `/etc/resolv.conf` and
`/etc/hosts` that Docker manages itself.

The trade is that a cold container re-fetches filings from SEC. The in-process
throttle keeps that inside SEC's fair-access ceiling, and the cache still works
normally for the life of a session — it just does not outlive it.

## Registering with Claude Code

The servers speak streamable HTTP, so Claude Code connects to them rather than
spawning them. Bring the stack up first:

```bash
docker compose -f deploy/docker-compose.yml up -d
```

Then register each one:

```bash
claude mcp add --transport http nemo-sec       http://<host>:8810/mcp/
claude mcp add --transport http nemo-financial http://<host>:8811/mcp/
claude mcp add --transport http nemo-finnhub   http://<host>:8812/mcp/
claude mcp add --transport http nemo-fred      http://<host>:8813/mcp/
claude mcp add --transport http nemo-altdata   http://<host>:8814/mcp/
```

| Server | Port | Module | Tools |
|---|---|---|---|
| SEC EDGAR | 8810 | `tools.web_search_server.web_search` | 42 of 45 |
| Market data and modelling | 8811 | `tools.financial_modeling_engine.analysis_tools` | 19 |
| Finnhub | 8812 | `tools.news_agregator.finnhub_server` | 14 |
| FRED macro | 8813 | `tools.news_agregator.fred_server` | 5 |
| Alt data | 8814 | `tools.altdata_server.server` | 5 |

**Register the URL with its trailing slash.** `Mount` only matches `/mcp/`, so
posting to `/mcp` answers `307` and every single call pays an extra round trip.

The SEC server advertises 42 of its 45 tools: `search` and the two rag tools
are hidden because SearXNG and the RAG stack are not in this image. That is
deliberate. `search` returns an empty result list rather than an error when
SearXNG is missing, so advertising it would make an absent container
indistinguishable from a query that matched nothing.

stdio still works for local use -- swap `http` for `server` in the command.

### Health checks are not a deploy gate

`/health` reports that uvicorn bound the port and the app started. A server
whose MCP layer failed still answers it. Compose uses it for restart decisions;
the actual gate is `testing/test_http_transport.py`, which completes a real
handshake and one live tool call against each server.

## Building
## Building

```bash
docker build -t nemo-data:local .                                  # native, local testing
docker buildx build --platform linux/amd64 -t nemo-data:amd64 .    # for Proxmox
```

The image copies only the servers it serves, which means **a new top-level
module under `tools/` must be added to the `COPY` line explicitly** or it is
silently absent at runtime. `tools/mcp_http.py` was missed exactly this way on
its first run. The build-time import check now covers it, so the build fails
rather than the container.

Proxmox VE is x86-64 only, so a homelab image must be built for `linux/amd64`.
Building on Apple Silicon without `--platform` produces an arm64 image that will
not run there.

## Secrets

`FINNHUB_API_KEY`, `FRED_API_KEY`, `SEC_EMAIL`, `NAME`. No LLM credentials:
every server here imports and runs with `GROQ_API_KEY` and `OPENROUTER_API_KEY`
unset, verified.

`CONGRESS_API_KEY` and `FINMIND_TOKEN` are optional and currently unset, so
`get_policy_signals`, `get_government_contracts`, and
`get_taiwan_monthly_revenue` will degrade.

## Known gaps in this image

- `analyze_exposures` and `get_thesis_evolution` read book state. The entrypoint
  creates the schema so they return an empty result rather than failing, but a
  data-source host holds no positions, so empty is the truthful answer.
- Employee count is not available from XBRL: no filer examined tags it, so it
  would need cover-page text extraction.

## Authentication

Two layers. The network one does most of the work.

### Private network

Put the host on a Tailscale or WireGuard overlay and bind the published ports
to its tailnet address rather than every interface:

```yaml
ports: ["100.x.y.z:8810:8080"]
```

Nothing is then reachable from the internet or from the untrusted LAN — only
from devices you have enrolled. Tailscale is per machine, not per project: one
install gives the box a tailnet IP and every service on it becomes reachable by
port. If other projects live on the same host they need no separate setup, and
`--advertise-routes` turns one node into a subnet router for the whole LAN.

### Bearer token

Defence in depth, and the only mechanism every MCP client can use — a header
needs no client-side implementation, where OAuth needs both an interactive flow
and code in the client.

```bash
export NEMO_MCP_TOKEN=$(openssl rand -hex 32)
docker compose -f deploy/docker-compose.yml up -d

claude mcp add --transport http nemo-sec http://<host>:8810/mcp/ \
  --header "Authorization: Bearer $NEMO_MCP_TOKEN"
```

**A server with no token refuses to start.** Defaulting to open means one
forgotten variable silently publishes every tool while the running server looks
perfectly healthy. If a deployment is genuinely reachable only through a tunnel
you control, set `MCP_ALLOW_UNAUTHENTICATED=1` and it will start with a warning
— the point is that running open has to be stated rather than defaulted into.

Tokens shorter than 24 characters are refused: a guessable token is worse than
none because it looks like security.

`/health` is exempt so the container healthcheck works without credentials. It
reports liveness only and never returns data.

### Why not OAuth

The MCP specification defines an OAuth 2.1 flow, and it is right for a
multi-user service. For one person it means running an auth server and debugging
redirect URIs to authenticate exactly one identity.

Blast radius argues the same way. These five servers are read-only — Alpaca is
deliberately excluded — so a leaked token exposes market-data queries against
free-tier keys, not money and not trades. The one genuine harm is SEC identity
misuse getting `SEC_EMAIL` rate-limited. A rotatable token behind a private
network is proportionate to that; an OAuth server is not.

## Logs

Container logs are capped at 30MB per service — `max-size: 10m`, `max-file: 3`
— and rotated by Docker itself. Size-based rather than time-based on purpose: a
daily reset does not bound the worst case, because a runaway loop fills the disk
long before midnight.

This matters more than it looks. Docker writes container logs to the **host**
disk under `/var/lib/docker/containers/`, outside the tmpfs mounts that bound
everything else, and the default json-file driver has no limit at all.

By default a request logs as `POST /mcp/ 200 OK` and nothing about the payload.
To see what a tool actually returned:

```bash
MCP_LOG_RESPONSES=1 docker compose -f deploy/docker-compose.yml up -d
docker logs nemo-fred
```

```
[mcp_http] 200 /mcp/ -> event: message
data: {"jsonrpc":"2.0","id":1,"result":{"content":[{"type":"text","text":"{\"curve\": {\"10Y\": 4.69 ...
```

Bodies are truncated at `MCP_LOG_RESPONSE_CHARS` (default 4000) because an MD&A
extract runs to 80KB. Off by default for the same reason the cap exists.
`/health` is never logged — it fires every thirty seconds forever and its body
carries nothing.
