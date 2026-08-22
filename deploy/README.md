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

Each server is a separate MCP entry. Claude Code spawns the container per
session, so `--rm` plus tmpfs means cleanup is automatic rather than a chore.

```json
{
  "mcpServers": {
    "nemo-sec": {
      "command": "docker",
      "args": [
        "run", "-i", "--rm",
        "--env-file", "/path/to/.env",
        "--tmpfs", "/root/.edgar:rw,size=512m",
        "--tmpfs", "/app/db_cache:rw,size=128m",
        "nemo-data:local",
        "python", "-m", "tools.web_search_server.web_search", "server"
      ]
    }
  }
}
```

The five data-source servers and their modules:

| Server | Module |
|---|---|
| SEC EDGAR | `tools.web_search_server.web_search` |
| Market data and modelling | `tools.financial_modeling_engine.analysis_tools` |
| Finnhub | `tools.news_agregator.finnhub_server` |
| FRED macro | `tools.news_agregator.fred_server` |
| Alt data | `tools.altdata_server.server` |

## Building

```bash
docker build -t nemo-data:local .                        # native, for local testing
docker buildx build --platform linux/amd64 -t nemo-data:amd64 .   # for Proxmox
```

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
- Transport is stdio. Claude Code spawns the process, so these servers are not
  reachable over a network yet. Serving them from the homelab needs the HTTP
  transport work.
