# Running the data-source servers

Stateless by design. A request comes in, a response goes out, and nothing
survives the container.

## What persists and what does not

The servers keep nothing. The scheduled jobs keep everything. Two paths are
written at runtime by the servers:

| Path | What writes it | Growth |
|---|---|---|
| `/root/.edgar` | edgartools' filing cache | **2MB after a single filing**, unbounded across tickers |
| `/app/db_cache` | tool cache and session schema | slow, but unbounded |

Both are mounted as size-capped tmpfs, so they live in RAM, cannot grow past
their cap, and are freed when the container exits.

Two paths are deliberately **not** disposable, and both belong to the scheduled
jobs rather than to the servers.

`/app/data/congress.db` on the named volume `congress-data`, because rebuilding
it means re-fetching and re-parsing thousands of PDFs from the House Clerk --
slow, and rude to a public service.

`/app/data/pit.db` on `research-data`, which is the one thing here that cannot
be rebuilt at any price. Filings can be re-read from EDGAR at the cost of time;
what analysts expected on a past Tuesday cannot be recovered by Thursday, and
the vendor serves four quarters of consensus history however many are asked
for. See [The scheduled jobs](#the-scheduled-jobs).

Putting either in the tmpfs above would make its pipeline unusable, since every
restart would begin from nothing. Measured with the mounts in place: the image
layer stays untouched apart from the `/etc/resolv.conf` and `/etc/hosts` that
Docker manages itself.

The trade is that a cold container re-fetches filings from SEC. The in-process
throttle keeps that inside SEC's fair-access ceiling, and the cache still works
normally for the life of a session — it just does not outlive it.

## Registering with an MCP client

The servers speak streamable HTTP, so a client connects to them rather than
spawning them. Bring the stack up first, from `deploy/` — every command on this
page runs from there:

```bash
docker compose --env-file ../.env up -d
```

**`--env-file ../.env` is not optional.** Compose interpolates `${VAR}` from the
shell and from the `.env` beside the compose file — never from an `env_file:`
entry, and never from `../.env` on its own. Every service now names the keys it
gets instead of taking the whole file (see [Secrets](#secrets)), so without the
flag those names resolve to empty and the servers come up with no credentials
and no token. They say so — `/ready` answers 503 and `resolve_auth_token`
refuses to start — but the flag is what stops it happening.

Then point the client at each one, with `Authorization: Bearer $NEMO_MCP_TOKEN`
on every request:

```
nemo-sec        http://<host>:8810/mcp/
nemo-financial  http://<host>:8811/mcp/
nemo-finnhub    http://<host>:8812/mcp/
nemo-fred       http://<host>:8813/mcp/
nemo-altdata    http://<host>:8814/mcp/
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

`/health` reports that uvicorn bound the port and the app started, and nothing
else. A server whose MCP session manager never came up answers it. So does one
with no `SEC_EMAIL`, serving all 48 SEC tools, every one of which fails on its
first EDGAR call — nothing builds the SEC identity at startup, so that failure
does not arrive until a caller asks.

The compose healthcheck therefore reads `/ready`, which checks the MCP layer and
the credentials each server declares it cannot work without, and answers **503**
when either fails. The probe prints the body, so `docker inspect` names the
failing check rather than only reporting that one failed.

It deliberately does *not* answer 503 for a server that has simply not been
asked anything yet. That is true of every container for its first seconds and
of any server nobody queried today; gating the code on it would mark all five
unhealthy every morning, which is a signal nobody reads by the time it means
something. `last_success` stays in the body under `degraded`.

The deploy gate is still `testing/test_http_transport.py`, which completes a
real handshake and one live tool call against each server. A healthcheck can
only prove a server is not obviously broken.

## Building

```bash
docker build -t nemo-data:local .                                  # native, local testing
docker buildx build --platform linux/amd64 -t nemo-data:amd64 --load .   # for Proxmox
NEMO_IMAGE=nemo-data:amd64 NEMO_TARGET_ARCH=amd64 docker compose --env-file ../.env up -d
```

The image copies only the servers it serves, which means **a new top-level
module under `tools/` must be added to the `COPY` line explicitly** or it is
silently absent at runtime. `tools/mcp_http.py` was missed exactly this way on
its first run. The build-time import check now covers it, so the build fails
rather than the container.

A new top-level *package* is the same hazard twice over, because the build has
two stages and each names what it copies. `common/` — two files, no imports,
holding the `Secret` credential wrapper that `finnhub`, `fred` and `altdata`
all read a key through — has a `COPY` line in each.
`testing/test_server_dependencies_are_declared.py` now walks the shipped
servers' imports against both lists, so a package added to one and not the
other fails a test rather than a container.

Proxmox VE is x86-64 only, so a homelab image must be built for `linux/amd64`.
Building on Apple Silicon without `--platform` produces an arm64 image that will
not run there.

## Secrets

`FINNHUB_API_KEY`, `FRED_API_KEY`, `SEC_EMAIL`, `NAME`. Each service is handed
the keys the code it runs actually reads, and nothing else:

| Service | What it is handed |
|---|---|
| sec | `SEC_EMAIL`, `NAME` |
| financial | nothing — Yahoo takes no key and the calculators take arguments |
| finnhub | `FINNHUB_API_KEY` |
| fred | `FRED_API_KEY` |
| altdata | `CONGRESS_API_KEY`, `FINMIND_TOKEN` |
| congress-sync | `SEC_EMAIL` |
| research-daily, research-seed, research-announce | `SEC_EMAIL`, `FINNHUB_API_KEY` |
| research-scan | `SEC_EMAIL` |
| research-watch | `SEC_EMAIL`, `NAME`, `FINNHUB_API_KEY` |
| research-score | nothing — it opens `pit.db` and calls no upstream |
| research-backup | nothing |

`CONGRESS_API_KEY` and `FINMIND_TOKEN` are optional and currently unset, so
`get_policy_signals`, `get_government_contracts`, and
`get_taiwan_monthly_revenue` will degrade. They are still named in the compose
file: without those lines, filling them in in `.env` would change nothing, which
is a worse failure than an absent key.

The table is checked, not maintained. `testing/test_compose_env_minimality.py`
walks the import graph from the module each service runs and fails on any key
compose declares that no module in that graph reads.

### The claim this section used to make, and why it was false

It said *"No model credentials of any kind: every server here imports and runs
with none set, verified."* That was true of the **image** and false of the
**deployment**. `env_file: ../.env` was set on all twelve services, and
`env_file` cannot be filtered — it hands the container every line in the file.
Measured on `research-score`, a weekly job that opens `pit.db` and calls
nothing: eighteen keys arrived, including `ALPACA_LIVE_KEY` and
`ALPACA_LIVE_SECRET`.

The verification recipe below is the reason it went unnoticed for so long: it
greps the **image filesystem**, which is a genuinely different question, and it
passed throughout while the keys sat in `/proc/<pid>/environ` and in
`docker inspect` for all twelve containers.

To ask the other question — what a running container actually holds — ask the
container. Names only; never print the values:

```bash
docker inspect nemo-sec --format '{{range .Config.Env}}{{println .}}{{end}}' | cut -d= -f1 | sort
docker compose --env-file ../.env run --rm --entrypoint env research-score | cut -d= -f1 | sort
```

`OPENROUTER_API_KEY`, `GROQ_API_KEY` and the four `ALPACA_*` keys are named by no
service in the compose file, so they now appear in neither.

## Getting secrets onto the host

Nothing secret is in the image, and that is verifiable rather than assumed:
compose reads `.env` on the host at **run** time, the Dockerfile has no `ARG` and
never references `.env`, and `.dockerignore` excludes it so a careless
`COPY . .` cannot bake one into a layer. To check a specific key:

```bash
docker run --rm -e SEEK="$(grep '^FINNHUB_API_KEY=' .env | cut -d= -f2- | head -c 16)" \
  --entrypoint sh nemo-data:local -c 'grep -rlF "$SEEK" /app /etc /root /usr/local || echo CLEAN'
```

Search `/app /etc /root /usr/local`, never `/`: grepping `/` matches the
process's own `/proc/self/cmdline` and reports a leak that is not there. Check
the needle is non-empty first, too -- `grep -rl ""` matches every file in the
image. And note what this proves and what it does not: it answers "is the key in
a layer", never "does a container hold it". The two commands above are for that.

`.env` stays at the repo root rather than moving next to the compose file, which
is why every command here carries `--env-file ../.env`. `.dockerignore` excludes
`.env` at the context root and would not exclude `deploy/.env`; moving the file
in here to save a flag would put it back inside the build context.

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

- **In the shell** (`NEMO_MCP_TOKEN=... docker compose --env-file ../.env up -d`)
  keeps it out of every file on disk. Interpolation happens at `up`, so
  containers keep their copy across a reboot -- but any later `up` or `restart`
  needs the variable present again, and forgetting it stops the
  stack rather than publishing it unauthenticated.
- **In `.env`** alongside the API keys survives unattended restarts and is one
  fewer thing to remember. It is also one more secret in a file, though that
  file already holds every API key, so the marginal exposure is small.

This second option only started working with `--env-file ../.env`. Under
`env_file:` the token had to be exported into the shell whatever the page said:
`env_file` delivers variables *into the container* and is never a source for
`${...}` interpolation, so `NEMO_MCP_TOKEN` in `.env` resolved to empty and the
server refused to start.

For a homelab that should come back up on its own, `.env` with `chmod 600` is
the pragmatic choice. The client end is not free either: a client configured
with `Authorization: Bearer ...` stores that header in its own config file in
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
docker compose --env-file ../.env run --rm congress-sync --house 2024 2025 2026 --senate --senate-annual
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

## The scheduled jobs

Eight services in the compose file are not servers. They bind no port, serve
nothing, and finish — so they sit under the `sync` profile and `docker compose
up` never starts them. A `restart: unless-stopped` batch job is a loop.

```bash
docker compose --env-file ../.env run --rm research-daily --bootstrap   # first four nights
docker compose --env-file ../.env run --rm research-daily               # every night after
docker compose --env-file ../.env run --rm research-scan
```

Cron lines, in the order they should run:

```cron
CRON_TZ=UTC
30 22 * * 1-5 cd /srv/nemo/deploy && flock -n /var/lock/nemo-research-daily.lock docker compose --env-file ../.env run --rm research-daily && flock -n /var/lock/nemo-research-scan.lock docker compose --env-file ../.env run --rm research-scan
*/20 13-23 * * 1-5 cd /srv/nemo/deploy && flock -n /var/lock/nemo-research-watch.lock docker compose --env-file ../.env run --rm research-watch
 0  7 * * 6 cd /srv/nemo/deploy && flock -n /var/lock/nemo-research-score.lock docker compose --env-file ../.env run --rm research-score
 0  9 * * 6 cd /srv/nemo/deploy && flock -n /var/lock/nemo-research-announce.lock docker compose --env-file ../.env run --rm research-announce
 0  8 1 * * cd /srv/nemo/deploy && flock -n /var/lock/nemo-research-seed.lock docker compose --env-file ../.env run --rm research-seed
 0  6 * * * cd /srv/nemo/deploy && flock -n /var/lock/nemo-congress-sync.lock docker compose --env-file ../.env run --rm congress-sync
15  3 * * * cd /srv/nemo/deploy && flock -n /var/lock/nemo-research-backup.lock docker compose --env-file ../.env run --rm research-backup
 0  4 * * * cd /srv/nemo/deploy && docker container prune -f --filter until=24h --filter label=com.docker.compose.project=deploy
```

### What a failed stage costs, and why the chain still uses `&&`

The recorder exits non-zero if **any** stage failed, and the cron line joins it
to the scan with `&&`, so a bad night costs that night's decisions as well. That
coupling is deliberate — a scan against a half-written store files a
permanently wrong book and logs it `ok` — but it means the blast radius of one
vendor hiccup is larger than it first looks.

The hiccup that matters is consensus, because that series accrues forward only:
Finnhub serves four quarters however many you ask for, so a night missed is a
night gone. A cold bootstrap here failed exactly that way on its first run — one
transient `HTTP 503` from Finnhub's edge, gone within twenty seconds, after
1.12M bars and 10,391 registrants had already been written successfully.

The fix is in the retry policy, which used to be backwards: `_fetch_bars`
retried three times with backoff, and the calendar fetch did not retry at all.
Both now do, and `FinnhubClient` retries any 5xx as well as a 429 — a 4xx is
still returned as it came, since a malformed request does not improve by being
repeated.

What is left is a genuine trade rather than a bug: a **sustained** vendor outage
still costs that night's scan, even though the scan does not need consensus to
run. It narrows candidates with it and falls back to sweeping the whole eligible
universe without it, which is slower and works — verified on a real run that
produced twenty candidates with the consensus stage failed and zero rows in the
table. Prising the two apart would mean the exit code no longer answering "is
this record fit to decide on", which is the question `&&` is asking. If you would
rather have the scan than the alert, run the two as separate cron lines and
accept that a half-written store gets scanned.

### The scan is long-only until somebody prices the borrow

`research-scan` refuses every short it cannot price a borrow on, and reports the
count as `borrow_unpriced`. That is deliberate. Being short is a position held
open and the stock loan bills for every calendar day of it — 23bp at a 3% rate
over a twenty-session hold, against a drift of a few tens of basis points — and
a short charged nothing outranks every name that paid. There is no default rate
because there is no free source for one.

Two ways to price it, and the first wins where both exist:

```bash
# a quote per name, from a broker file, dated like everything else in the store
python -m research.borrow --as-of 2026-08-31 --from-csv rates.csv \
  --units fraction --source ibkr

# one blanket assumption, made out loud and recorded on every row it priced
docker compose --env-file ../.env run --rm research-scan --borrow-rate 0.03
```

`--units` has no default on purpose: `3` and `0.03` are both plausible readings
of the same column and they differ by a hundred.

A broker publishes its short list for a date and you collect it the next
morning, so `--as-of` is usually yesterday. Add `--backfill` for that: it stamps
the rows at the end of the day they describe instead of now, and marks them
`backfilled` so a study can tell them from rates captured live. Without it the
rows are stamped now and a scan dated yesterday will not see them —
`date(recorded_at) <= as_of` is doing its job, and the loader warns rather than
reporting a silent success.

```bash
python -m research.borrow --as-of 2026-08-31 --from-csv rates.csv \
  --units percent --source ibkr --backfill
```

### The watcher sweeps a slice, not the universe

`research-watch` runs `--max-seconds 600` (set in the compose file, override it
on the command line for a one-off). A full sweep does not fit the interval it
runs on: 1,565 eligible names took over 40 minutes uncapped, against a `*/20`
timer, so two firings in three were refused by `flock` and exited non-zero —
which is how an operator learns to read a failing job as normal.

**The bound is a clock, not a ticker count.** Three measurements on one machine
in one afternoon put a ticker at **0.90s, then 3.0s, then 6.0s** — the rate got
worse as the sample got *smaller*, so the variable is EDGAR's throttle state and
not the work. Any fixed ticker cap sized against one of those numbers is wrong
on the other two.

A cursor (`job_cursor` in the store) records where each pass stopped, so the
next one resumes there instead of restarting at the head — otherwise a bounded
pass is the same blind spot with extra steps, permanently never reaching
whoever sorts last. A pass that does not finish reports `partial`, says what
stopped it and where the next one resumes, so rotating coverage is
distinguishable from stuck coverage in the run log.

On a slow day the cycle is simply longer, against a 13D that must be filed
within five days. That is the trade: coverage *rate* degrades, coverage
*completeness* does not.

Bounding is management, not a cure. The real answer is EDGAR's market-wide
current-filings feed — one request for the whole market rather than one per name
— which `research/activist_watch.py` names in its own docstring and does not
implement. The feed is live and returns Atom; see issue #82.

The cron line above passes neither, so the nightly book is long-only until you
decide otherwise. Commission and the long leg's financing are still not modelled
at all; both are small next to borrow over this horizon, and neither is zero.

### Why every line carries a `flock`, and why the last line exists

`docker compose run` takes no lock of its own. A pass that overruns its interval
— a `research-watch` sweep of the whole eligible universe against a `*/20` timer
— starts a second container against the same database. `flock -n` makes the
second exit non-zero immediately rather than queue behind the first, because a
run that is skipped silently is indistinguishable from one that never fired.

`--rm` is not a guarantee either. The removal happens in the compose **CLI**
after the container exits, so a host reboot mid-job kills the CLI and leaves the
container in `Exited` forever. These services set no `container_name`, so each
run creates a uniquely-named container and the orphans accumulate rather than
colliding. Each one keeps up to 30MB of json-file logs under
`/var/lib/docker/containers/` — the log cap is **per container**, so 30MB × N
orphans is unbounded in aggregate, which is exactly the growth the cap was
reasoned to prevent. Hence the daily prune.

The `--filter label=` is what keeps that prune off every other project's exited
containers on the same host. Its value is the compose project name, which
compose takes from this directory's name — which is why that line, alone among
the ones that do not need it, still starts with the `cd`.

A reboot mid-job does **not** block the next run, and that part is well built:
`start_run` in `research/pit_store.py` only INSERTs, and `missing_days` counts a
started-but-unfinished row as a gap. There is no lock file or run marker to go
stale.

### Both clocks are UTC, and the block above says so

Every service in the compose file sets `TZ=UTC`, written out rather than
interpolated. `research/daily_job._today()` is UTC unconditionally and
`python:3.12-slim` sets no `TZ` at all, so a container's local time used to be
UTC by accident; `${TZ:-UTC}` would have left the operator's environment able to
move it, which is the same mismatch by another route.

That pin stops inside the container. The cron lines run on the **host**, in
host-local time, and nothing in this repo can reach the host's crontab — which
is why the block opens with `CRON_TZ=UTC`. Vixie-derived crons (cronie on
RHEL-alikes, Debian's `cron`) read it and interpret every line below it as UTC.
If yours does not — `busybox` crond ignores it, and a systemd timer never sees
it — set the host itself to UTC with `timedatectl set-timezone UTC` and read the
times below as UTC either way.

The cost of getting it wrong is issue #28, and it is not a small one. On an
`America/New_York` host, 22:30 is 02:30 UTC the next day, so `as_of` is
tomorrow, `_fetch_bars` returns SPY through today, and the recorder concludes
the exchange was shut. Zero bars, `status="closed"`, exit 0, and `missing_days`
counting the night as covered — every night, indefinitely. The recorder now
refuses a session that has not happened yet rather than filing it as a holiday,
so this fails loudly now; pinning both clocks is what stops it arising.

`research-backup` gets the pin for a smaller reason of the same shape: it names
its file `pit-<date>.db` from `date.today()`, which is local, so a container an
hour the wrong side of midnight writes a name for a day nobody backed up and
the fourteen-file window then keeps the wrong ones.

### Why the scan is chained to the recorder rather than timed after it

The first line runs two jobs, joined by `&&`. That used to be two lines thirty
minutes apart — `30 22` and `0 23` — which is a guess, not a dependency.

The recorder does 15+ yfinance batches with up to three retries at 5s and 10s
backoff, then screens all 10,388 registrants with one `sqlite3.connect()` per
name, then consensus. On the four `--bootstrap` nights it also pulls 730 days
for 3,000 of them. Overrunning half an hour there is close to certain, and
nothing about the second line noticed.

What a scan that starts early does is the part worth stating, because every
piece of it is behaving correctly. `universe_as_of` returns the last membership
row on or before the date, so a store without tonight's row answers with
yesterday's — which is exactly what a point-in-time read is for. `record_scan`
is append-only, so the book filed off that stale universe cannot be corrected
later; a re-run only reports `superseded`. Two right decisions, and between them
a permanently wrong book logged `status="ok"`.

The scanner now consults the run log and raises `RecorderNotRunning` rather than
filing that book, which turns a wrong record into a failed run. That is better
and it is not enough: a failed run every bootstrap night is still an outage.
`&&` is what makes "after" mean after — the scan cannot start until the recorder
has exited, and does not start at all if it exited non-zero.

Both halves keep their own `flock`. The chain holds the recorder's lock for the
whole evening, and the scan's lock is what stops a hand-started re-run against a
past date from opening a second container on the same database.

`research-announce` reads Item 2.02 filings for the announcement date and the
hour it landed, which is what decides whether the reaction is that session or
the next. It must run before `research-seed`, which dates each reconstructed
actual by the release where one is on record.

`research-watch` reports the detection latency accumulated so far on every
pass, not just what it recorded this time -- catching a filing quickly is the
reason for the twenty-minute timer, and a job that never says how quickly
cannot be judged.

Each job exits non-zero when a stage fails and zero when it merely finds
nothing. That distinction is the whole contract with cron — most nights the
watcher finds no 13D and the scanner finds no candidate, and a job that pages
on those is one nobody reads by the time it matters.

### The filing cache the jobs share with the servers

`research-watch` and `research-announce` drive edgartools, which caches every
document it fetches under `/root/.edgar` — the same 512MB tmpfs that reached
100% on the servers and made every SEC read fail with `[Errno 28] No space left
on device`. A `research-watch` pass over the whole eligible universe, thousands
of names with no default `--max-tickers`, is the shape of run that gets there.

`NEMO_FILING_CACHE_CAP_MB` and `NEMO_FILING_CACHE_INTERVAL_S` are now set on
these jobs as well as on the servers. **They do not take effect yet**, and that
is worth saying rather than leaving to be discovered: the interval pruner starts
in the HTTP app's lifespan (`tools/mcp_http.py`) and nowhere else, and the batch
jobs never import that module. Their only bound today is the tmpfs cap itself,
which fails the write rather than evicting — the job exits non-zero, which cron
reports, so it is loud, not silent. The half still missing is a prune call from
the jobs. How large a real pass actually gets is unmeasured; these jobs have
never run on the audit host.

### Backing up `research-data`

`research-daily` writes to `research-data`, a named volume holding one SQLite
file. It survives `docker compose down` and container replacement, and it is
the only thing here that cannot be rebuilt: the filings can be re-read from
EDGAR at the cost of time, but what analysts expected on a past Tuesday is
gone, and the vendor serves four quarters of consensus history however many you
ask for.

This page said that and the compose file said that, and for a long time nothing
acted on it. There was no backup anywhere in the repo — one `docker compose
down -v` or `docker system prune --volumes` and the only copy was gone.

`research-backup` is that copy. It runs at 03:15 daily, in the cron block above.

**It uses SQLite's online backup API, not `cp`.** A plain copy of a live SQLite
file reads pages while they are being written and restores torn; there is no
obviously quiet window to copy in either, with six jobs on a schedule and the
13D watcher running every twenty minutes through the session. The API — reached
here from Python, and the same one the `sqlite3` CLI's `.backup` dot-command
drives — takes a consistent snapshot of a database that is being written to.

Backups land on a **host path**, `/srv/nemo/backups` by default and
`NEMO_BACKUP_DIR` if you want another. Deliberately not a named volume: a named
volume is destroyed by the same `docker compose down -v` this exists to survive.
The job keeps the last fourteen and deletes the rest, because host disk is the
one place here nothing else bounds.

It refuses rather than writes an empty file if `/app/data/pit.db` is not there.
A mistyped mount gives an empty `/app/data`, and a backup job whose failure mode
is a valid-looking zero-row database is worse than no backup at all.

**Copy it off the host.** A backup on the machine that dies is not a backup:

```bash
rsync -a --delete /srv/nemo/backups/ you@100.x.y.z:/srv/backups/nemo/   # over the tailnet
```

#### Restoring

The volume is only readable through a container, so the restore goes through one
too. Stop everything that writes first — a restore into a live database is the
torn copy problem again, from the other end:

```bash
cd /srv/nemo/deploy
docker compose --env-file ../.env down          # servers; the jobs are already --rm

# Check the copy before trusting it. A backup nobody has read is a hypothesis.
docker run --rm -v /srv/nemo/backups:/backups --entrypoint python nemo-data:local \
  -c "import sqlite3; db=sqlite3.connect('/backups/pit-2026-08-26.db'); \
print(db.execute('PRAGMA integrity_check').fetchone()[0]); \
print(db.execute('select count(*) from daily_bar').fetchone()[0], 'bars')"

# Then put it back, at the path NEMO_PIT_DB points at inside the volume.
docker run --rm -v deploy_research-data:/app/data -v /srv/nemo/backups:/backups \
  --entrypoint cp nemo-data:local /backups/pit-2026-08-26.db /app/data/pit.db

docker compose --env-file ../.env up -d
```

**`deploy_research-data`, not `research-data`.** Compose prefixes volume names
with the project name, which it takes from this directory. `docker run` does no
such prefixing, and `-v research-data:` would silently create a new empty volume
and restore into it — a restore that reports success and changes nothing. Run
`docker volume ls` and use the name it prints.

`--entrypoint` on both, because the image's entrypoint creates the session
schema before running the command and neither of these wants that.

`cp` is correct **here** and wrong in the backup direction: nothing is writing
the file during a restore, which is the entire difference.

`congress-sync` writes to `congress-data`, which is reconstructible from the
public record — expensively, but reconstructible. It is not backed up for that
reason.

### A cold store needs four bootstrap nights

The nightly ask is capped at 3,000 names, because asking for all 10,388 SEC
registrants at once earns a rate limit; the rest arrive on a rotating slice
that covers the list in about four runs. Without `--bootstrap` each of those
names gets one session and eligibility needs sixty, so the universe would take
a couple of hundred days to fill instead of four.

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
docker compose --env-file ../.env up -d
```

Then register `http://<host>:8810/mcp/` in the client, with the header
`Authorization: Bearer $NEMO_MCP_TOKEN`.

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

Container logs are capped at 30MB per container — `max-size: 10m`,
`max-file: 3` — and rotated by Docker itself. Size-based rather than time-based
on purpose: a daily reset does not bound the worst case, because a runaway loop
fills the disk long before midnight.

This matters more than it looks. Docker writes container logs to the **host**
disk under `/var/lib/docker/containers/`, outside the tmpfs mounts that bound
everything else, and the default json-file driver has no limit at all.

Per **container**, not per service, and that distinction is the whole reason the
cron block ends in a prune. The five servers are one long-lived container each,
so 30MB each is the end of it. Every `docker compose run --rm` makes a new one,
and an orphan left behind by a reboot keeps its 30MB — so the aggregate is
bounded only by how many orphans accumulate. See
[the cron block](#why-every-line-carries-a-flock-and-why-the-last-line-exists).

By default a request logs as `POST /mcp/ 200 OK` and nothing about the payload.
To see what a tool actually returned:

```bash
MCP_LOG_RESPONSES=1 docker compose --env-file ../.env up -d
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
