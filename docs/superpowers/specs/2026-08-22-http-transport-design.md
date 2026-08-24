# HTTP Transport for the Data-Source Servers — Design

**Date:** 2026-08-22
**Status:** approved, in progress
**Depends on:** the container image (PR #13)
**Scope:** transport and packaging. Authentication is deliberately deferred to the end.

## Problem

All five data-source servers call `stdio_server()`. stdio means the *client*
spawns the server as a child process and talks over its stdin and stdout, so a
stdio server on a remote host is unreachable by definition. Containerising one
produces a subprocess in a box, not a server.

Everything built so far — 42 SEC and market tools, the dilution work, the
research-gap closures — is unreachable from anywhere but the machine that
spawns it. This is the only work that changes that.

## What the spike established

Proven end to end on 2026-08-22 before this spec was written, so none of the
following is assumption:

- `mcp.server.streamable_http_manager.StreamableHTTPSessionManager` works, and
  **no new dependency is required**: starlette, uvicorn and sse_starlette all
  arrive transitively with `mcp` and survived the dependency prune.
- A container serving `fred_server` over HTTP answers a full MCP handshake from
  the host: `initialize` → `fred v1.25.0`, `tools/list` → 5 tools, `tools/call`
  → a live treasury curve.
- **Claude Code connects**: `claude mcp add --transport http` produced
  `nemo-fred-spike: http://localhost:8807/mcp (HTTP) - ✔ Connected`.
- **Stateless mode accumulates nothing.** Across 20 requests to a long-lived
  container: `db_cache` 204K→204K, `/tmp` 4.0K→4.0K, `/root` 16K→16K, no files
  left behind, processes 5→5, threads 2→2, open descriptors 17→17.

That last point matters more than it looks. Until now the no-accumulation
guarantee came from `--rm` destroying the container per session. A long-lived
server does not get that for free, and stateless transport is what replaces it.

## Guiding principle

> The server holds no memory of who asked.

Every tool here answers a self-contained question about the world. Nothing
needs per-client state, so nothing should keep any.

## Deliverables

### D1 — `http` subcommand on the remaining four servers

`tools/mcp_http.py` already exists and is server-agnostic: each server
constructs an `mcp.server.Server` and differs only in which instance it hands
over. `finnhub_server`, `web_search`, `analysis_tools` and `altdata_server` each
gain the same subcommand.

stdio stays the default. This is additive; local development is unaffected.

### D2 — One port per server

Five services, five ports, five Claude Code entries. The alternative — merging
all five tool registries behind one endpoint — is a real refactor that would
collapse five independent restart domains into one, and buys only a shorter
config block.

| Server | Port |
|---|---|
| SEC EDGAR (`web_search`) | 8810 |
| Market data and modelling (`analysis_tools`) | 8811 |
| Finnhub | 8812 |
| FRED | 8813 |
| Alt-data | 8814 |

### D3 — Compose file

One service per server off the shared image, each with its own command and
port. Per-service restart policy and healthcheck against `/health`, which the
spike added precisely so a runtime can distinguish "process alive" from "port
bound but the MCP layer never started".

tmpfs mounts carried over from the stdio invocation: `/root/.edgar` capped at
512m, `/app/db_cache` at 128m.

### D4 — Guaranteed cleanup for anything that writes a file

Stateless transport covers today's tools, none of which write files — verified,
no server in the image accepts a file path. That is a property of the current
tool set, not a guarantee about future ones.

Provide a `request_scratch()` context manager that yields a temp directory and
unlinks it in a `finally`, so cleanup happens on the error path too. Any future
tool that handles a file uses it. Establishing this now is cheap; retrofitting
it after the first leak is not.

### D5 — Deployment documentation

Extend `deploy/README.md`: the Claude Code registration block for five HTTP
servers, the compose invocation, `docker buildx --platform linux/amd64` for
Proxmox, and the healthcheck semantics.

Record the Dockerfile gotcha the spike hit: the selective `COPY` means any new
top-level module under `tools/` must be added explicitly, or it is silently
absent at runtime.

## Testing strategy

1. **Handshake per server.** Each of the five answers `initialize`,
   `tools/list` and one real `tools/call` over HTTP. Not a health check —
   `/health` returns fine from a server whose MCP layer never started.
2. **No accumulation under load.** After N requests to a long-lived container,
   disk, process, thread and descriptor counts are unchanged. This is the
   regression guard on the property that replaces `--rm`.
3. **Capability gating survives the transport.** The SEC server must advertise
   39 of 42 tools over HTTP exactly as it does over stdio; `search` and the
   rag pair stay hidden without their dependencies.
4. **stdio still works.** The subcommand is additive, and a regression here
   breaks local development silently.
5. **amd64.** The handshake test runs against the `linux/amd64` image, since
   that is what Proxmox will run and it has never been the default build here.

## Risks

| Risk | Mitigation |
|---|---|
| A new top-level module is silently missing from the image | build-time import check already fails the build; extend it to import `tools.mcp_http` |
| Stateless mode drops something a client needs | the spike exercised initialize, list and call; anything server-initiated would need session state, and no tool here pushes |
| Five ports is five things to misconfigure | compose owns the mapping; the README carries the exact Claude Code block |
| A long-lived container accumulates where `--rm` used to save us | test 2 is the standing guard, and D4 covers file-handling tools before they exist |
| Port bound but MCP dead | `/health` reports the transport, and the handshake test is what actually gates a deploy |

## Explicitly deferred

**Authentication.** Anything that reaches the port can call every tool using
the host's Finnhub and FRED keys. On a LAN that is a real exposure decision,
and it is deliberately the last thing addressed rather than the first —
transport correctness is a prerequisite for deciding how to guard it.

Also out of scope: TLS, multi-host deployment, and the aggregated-endpoint
alternative to D2.
