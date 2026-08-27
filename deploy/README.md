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
their cap, and are freed when the container exits.

One path is deliberately **not** disposable. The congressional disclosure store
at `/app/data/congress.db` sits on the named volume `congress-data`, because
rebuilding it means re-fetching and re-parsing thousands of PDFs from the House
Clerk -- slow, and rude to a public service. Putting it in the tmpfs above would
make the pipeline unusable, since every restart would start from nothing. Measured with the mounts in
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
| SEC EDGAR | 8810 | `tools.web_search_server.web_search` | 48 of 51 |
| Market data and modelling | 8811 | `tools.financial_modeling_engine.analysis_tools` | 20 |
| Finnhub | 8812 | `tools.news_agregator.finnhub_server` | 14 |
| FRED macro | 8813 | `tools.news_agregator.fred_server` | 5 |
| Alt data | 8814 | `tools.altdata_server.server` | 9 |

96 tools served. The SEC server declares 51 and serves 48: `search`,
`rag_search` and `rag_ingest` are capability-gated and hidden, because this
image installs neither SearXNG nor the RAG stack -- the Dockerfile copies three
`agent/` modules and `agent.rag` is not among them.

Counts here are checked, not maintained: `testing/test_readme_counts.py` reads
each server's own registry and fails when a page disagrees with it. Run
`python -m tools.manifest` inside a container to see them for yourself, and
note it reports what *that* environment can serve. On a host that happens to
have SearXNG running, the SEC server serves 49.

**Register the URL with its trailing slash.** `Mount` only matches `/mcp/`, so
posting to `/mcp` answers `307` and every single call pays an extra round trip.

Gating `search` is the deliberate part. It returns an empty result list rather
than an error when SearXNG is missing, so advertising it would make an absent
container indistinguishable from a query that matched nothing. `rag_search` at
least raises.

stdio still works for local use -- swap `http` for `server` in the command.

### Health checks are not a deploy gate

`/health` reports that uvicorn bound the port and the app started. A server
whose MCP layer failed still answers it. Compose uses it for restart decisions;
the actual gate is `testing/test_http_transport.py`, which completes a real
handshake and one live tool call against each server.

## Building

```bash
docker build -t nemo-data:local .                                  # native, local testing
docker buildx build --platform linux/amd64 -t nemo-data:amd64 --load .   # for Proxmox
NEMO_IMAGE=nemo-data:amd64 NEMO_TARGET_ARCH=amd64 docker compose up -d
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

`FINNHUB_API_KEY`, `FRED_API_KEY`, `SEC_EMAIL`, `NAME`. No LLM credentials of
any kind: every server here imports and runs with none set, verified.

`CONGRESS_API_KEY` and `FINMIND_TOKEN` are optional and currently unset, so
`get_policy_signals`, `get_government_contracts`, and
`get_taiwan_monthly_revenue` will degrade.

## Getting secrets onto the host

Nothing secret is in the image, and that is verifiable rather than assumed:
`env_file` is read by compose on the host at **run** time, the Dockerfile has no
`ARG` and never references `.env`, and `.dockerignore` excludes it so a careless
`COPY . .` cannot bake one into a layer. To check a specific key:

```bash
docker run --rm -e SEEK="$(grep '^FINNHUB_API_KEY=' .env | cut -d= -f2- | head -c 16)" \
  --entrypoint sh nemo-data:local -c 'grep -rlF "$SEEK" /app /etc /root /usr/local || echo CLEAN'
```

Search `/app /etc /root /usr/local`, never `/`: grepping `/` matches the
process's own `/proc/self/cmdline` and reports a leak that is not there. Check
the needle is non-empty first, too -- `grep -rl ""` matches every file in the
image.

So the secrets have to reach the host separately. Over the tailnet:

```bash
scp .env you@100.x.y.z:/srv/nemo/.env          # never over the public internet
ssh you@100.x.y.z 'chmod 600 /srv/nemo/.env'
```

**The bearer token is generated, not chosen.** 32 random bytes, once, on the
host that will run the stack:

```bash
openssl rand -hex 32
```

Where it lives is a real trade-off:

- **In the shell** (`NEMO_MCP_TOKEN=... docker compose up -d`) keeps it out of
  every file on disk. Interpolation happens at `up`, so containers keep their
  copy across a reboot -- but any later `docker compose up` or `restart` needs
  the variable present again, and forgetting it stops the stack rather than
  publishing it unauthenticated.
- **In `.env`** alongside the API keys survives unattended restarts and is one
  fewer thing to remember. It is also one more secret in a file, though that
  file already holds every API key, so the marginal exposure is small.

For a homelab that should come back up on its own, `.env` with `chmod 600` is
the pragmatic choice. The client end is not free either: `claude mcp add
--header "Authorization: Bearer ..."` writes the token to `~/.claude.json` in
plaintext, so that file is a secret on every machine you register from.

## Blockers hit while building this, and how they present

Each of these failed *after* a green test suite, so none of them is caught by
running the tests.

**`uv.lock` drifting from `pyproject.toml`.** The Dockerfile installs with
`uv sync --frozen`, which refuses a lockfile that disagrees with the project
file, so the build dies at the dependency step. It happens whenever a package
moves between dependency groups -- `pdfplumber` joining the `server` group for
the House PTR parser did it. `uv lock --check` says so in one line; run it
before building.

**A lazily-imported package missing from the `server` group.** The build-time
import check imports each server, so it only sees module-level imports. A
package imported inside a function -- `pdfplumber` in `fetch_house_ptr`,
`bs4` in `parse_senate_ptr` -- is invisible to it, the image builds clean, and
the ImportError arrives on the first real tool call in production.
`testing/test_server_dependencies_are_declared.py` walks every shipped file and
requires each third-party import to be declared, which is what closes this.

**A new top-level module under `tools/` not added to the `COPY` line.** Covered
in Building above; it is silently absent at runtime rather than a build error.

**The congressional store ships empty.** The volume persists but the image
contains no data, and the tools answer "no trades" for everything until it is
filled. They say so explicitly -- `store_empty: true` and the command to run --
rather than presenting an empty store as an empty record. After first boot:

```bash
docker compose run --rm congress-sync --house 2024 2025 2026 --senate --senate-annual
```

Roughly 40 minutes of throttled requests for a full backfill. Safe to re-run and
safe to cron; nothing already parsed is fetched twice.

## Known gaps in this image

- Congressional **holdings are Senate-only**. House annual reports are PDFs whose
  columns must be read geometrically from the header rectangles rather than from
  the extracted text, and that parser is not built. The tool description states
  the limit, so the gap is visible to a caller rather than silent.
- `analyze_exposures` and `get_thesis_evolution` read book state. The entrypoint
  creates the schema so the queries run rather than failing on a missing table.
  A data-source host holds no positions, so `analyze_exposures` answering with
  an empty book is the truthful answer. `get_thesis_evolution` names one thesis,
  and a book that does not hold it refuses rather than returning a null row.
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
