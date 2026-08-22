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
claude mcp add --transport http nemo-sec       http://<host>:8810/mcp
claude mcp add --transport http nemo-financial http://<host>:8811/mcp
claude mcp add --transport http nemo-finnhub   http://<host>:8812/mcp
claude mcp add --transport http nemo-fred      http://<host>:8813/mcp
claude mcp add --transport http nemo-altdata   http://<host>:8814/mcp
```

| Server | Port | Module | Tools |
|---|---|---|---|
| SEC EDGAR | 8810 | `tools.web_search_server.web_search` | 39 of 42 |
| Market data and modelling | 8811 | `tools.financial_modeling_engine.analysis_tools` | 19 |
| Finnhub | 8812 | `tools.news_agregator.finnhub_server` | 14 |
| FRED macro | 8813 | `tools.news_agregator.fred_server` | 5 |
| Alt data | 8814 | `tools.altdata_server.server` | 5 |

The SEC server advertises 39 of its 42 tools: `search` and the two rag tools
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
- **Authentication is not implemented.** Anything that can reach these ports
  can call every tool using the host's Finnhub and FRED keys. Bind to a LAN
  address and do not forward the ports at the router until this is addressed.
