# Data-source MCP servers for the homelab.
#
# Carries only the servers that answer questions about the world: FRED macro,
# Finnhub, SEC EDGAR, market data and the modelling calculators, and alt-data.
# Deliberately excludes alpaca (positions and orders are the book, not a data
# source), sentry (queue and thesis state), and excel (reads local files, which
# means nothing on a remote host).
#
# No ML stack. torch is only ever reached transitively through
# sentence-transformers for RAG embedding, and the RAG tools are not part of
# this image -- Firecrawl covers search on the client side. Excluding them drops
# roughly 300MB before counting the CUDA runtime that the linux wheel would
# otherwise pull in.
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

# curl is needed to fetch uv; git because a few packages inspect repo metadata.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Dependency layer first so source edits do not invalidate it. The cache mount
# keeps uv's download cache out of the image -- left in a layer it costs nearly
# a gigabyte, which is larger than the venv it builds.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --only-group server

# Copy only the servers this image serves. `COPY tools/` would ship alpaca,
# sentry, excel and slack as well -- excluded from running but present on disk,
# which on a LAN-reachable host is the difference between "no trading tools"
# and "trading tools nobody happens to start".
COPY tools/__init__.py tools/mcp_http.py tools/response_meta.py tools/manifest.py tools/filing_cache.py tools/ticker.py tools/gateway.py ./tools/
COPY tools/news_agregator/ ./tools/news_agregator/
COPY tools/web_search_server/ ./tools/web_search_server/
COPY tools/financial_modeling_engine/ ./tools/financial_modeling_engine/
COPY tools/altdata_server/ ./tools/altdata_server/
COPY tools/preearnings/ ./tools/preearnings/
# The point-in-time recorder and the signals over it. No server, no port and no
# bearer token: it is a batch job with a __main__, scheduled the same way
# congress-sync is. It ships here because it needs the same upstreams and the
# same pinned versions, and running it anywhere else means maintaining a second
# environment that has to stay in step with this one.
COPY research/ ./research/
# Only the three agent modules the servers reach. Retiring the
# LangGraph/OpenRouter layer (issue #63) removed workflows/ and twelve
# *_Agent.py files, so this list is no longer holding back LangGraph and
# LangChain -- those are gone from the dependency set entirely. It is still
# load-bearing for what remains: copying agent/ wholesale brings groq_template
# (and with it the OpenAI client and ollama), Execution_Agent (the broker),
# MCP_manager, and rag/ (sentence-transformers, and behind it torch) -- none of
# which this image runs. Verified by importing all five servers and recording
# what loads.
COPY agent/__init__.py ./agent/
COPY agent/cache.py ./agent/
COPY agent/exposure_analyzer.py ./agent/
COPY agent/backtest_engine.py ./agent/
COPY state/ ./state/
COPY knowledge/ ./knowledge/
# Two files, no imports, and every server that reads a credential needs one of
# them: `Secret` used to be copied into each module for the same reason this
# image copies only three files out of agent/ -- reaching the original would
# have dragged openai and ollama in behind it. Small enough to copy wholesale,
# and it must be here or finnhub, fred and altdata all fail to import.
COPY common/ ./common/

# The tool cache lives here; mount a volume so it survives container replacement.
RUN mkdir -p /app/db_cache
ENV NEMO_CACHE_DB_PATH=/app/db_cache/tool_cache.db \
    NEMO_DB_PATH=/app/db_cache/session.db \
    PYTHONPATH=/app

# Fails the build if any server cannot be imported, so a broken image never
# reaches the host.
RUN python -c "\
import importlib;\
mods=['tools.mcp_http','tools.news_agregator.fred_server','tools.news_agregator.finnhub_server',\
'tools.web_search_server.web_search','tools.financial_modeling_engine.analysis_tools',\
'tools.altdata_server.server','tools.response_meta','tools.manifest',\
'tools.filing_cache','tools.ticker'];\
[importlib.import_module(m) for m in mods];\
print('all 5 data-source servers and the HTTP transport import')"

# Default to the simplest server; compose overrides this per service.
CMD ["python", "-m", "tools.news_agregator.fred_server", "server"]


# --------------------------------------------------------------------------
# The paper-fill job (issue #116), a separate target:
#
#   docker build --target fills -t nemo-fills:local .
#
# It is the one thing in this repo that can place an order, so it is its own
# image: the data image's layers plus the broker client and the job, and the
# Alpaca keys only in its compose service. It sits above `runtime` on purpose.
# A plain `docker build` builds the last stage, and the last stage must stay
# the data image, which cannot trade (testing/test_paper_fills.py pins it).
# --------------------------------------------------------------------------
FROM base AS fills-src
# The nemo bot's Discord client, beside the server group. Only this image
# carries it.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --only-group server --only-group bot
COPY tools/alpaca/__init__.py tools/alpaca/async_broker.py tools/alpaca/fills.py tools/alpaca/portfolio.py tools/alpaca/bot.py tools/alpaca/ask.py ./tools/alpaca/
RUN python -c "import tools.alpaca.fills, tools.alpaca.portfolio, tools.alpaca.bot, tools.alpaca.ask; print('the paper-fill job, the portfolio card and the bot import')"

# The unmodified Claude Code program, which /ask runs signed in as the bot's
# owner (tools/alpaca/ask.py). One release, from Anthropic's download site,
# checked against the checksum its manifest gives for that release, so a
# rebuild cannot pick up a different binary. To move to another release,
# take both checksums from
#   https://downloads.claude.ai/claude-code-releases/<version>/manifest.json
FROM base AS claude-code
ARG TARGETARCH
ARG CLAUDE_CODE_VERSION=2.1.322
ARG CLAUDE_CODE_SHA256_AMD64=bfb3703ad5380242328459bc62ada4bde8ae7ba57c6e1eeadb9c00e08874c478
ARG CLAUDE_CODE_SHA256_ARM64=f220ca50ea4f3222043ff83cc224a1262e0196e9897acc0871efa94235b0c1fd
RUN python -c "\
import hashlib, os, sys, urllib.request; \
arch = os.environ['TARGETARCH']; \
platform, want = {'amd64': ('linux-x64', os.environ['CLAUDE_CODE_SHA256_AMD64']), \
                  'arm64': ('linux-arm64', os.environ['CLAUDE_CODE_SHA256_ARM64'])}[arch]; \
url = 'https://downloads.claude.ai/claude-code-releases/%s/%s/claude' % (os.environ['CLAUDE_CODE_VERSION'], platform); \
urllib.request.urlretrieve(url, '/usr/local/bin/claude'); \
got = hashlib.sha256(open('/usr/local/bin/claude', 'rb').read()).hexdigest(); \
sys.exit(0 if got == want else 'checksum mismatch for %s: %s' % (url, got))" \
 && chmod 0755 /usr/local/bin/claude \
 && DISABLE_UPDATES=1 HOME=/tmp /usr/local/bin/claude --version

FROM python:3.12-slim AS fills
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PATH="/app/.venv/bin:$PATH"
WORKDIR /app
COPY --from=fills-src /app/.venv /app/.venv
COPY --from=fills-src /app/tools /app/tools
COPY --from=fills-src /app/research /app/research
COPY --from=fills-src /app/common /app/common
COPY --from=claude-code /usr/local/bin/claude /usr/local/bin/claude
ENTRYPOINT ["python", "-m", "tools.alpaca.fills"]


# --------------------------------------------------------------------------
# Runtime stage: carries the venv and the source, not uv or curl or the
# apt lists used to install them.
# --------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PATH="/app/.venv/bin:$PATH" \
    NEMO_CACHE_DB_PATH=/app/db_cache/tool_cache.db \
    NEMO_DB_PATH=/app/db_cache/session.db

WORKDIR /app
COPY --from=base /app/.venv /app/.venv
COPY --from=base /app/tools /app/tools
COPY --from=base /app/research /app/research
COPY --from=base /app/agent /app/agent
COPY --from=base /app/common /app/common
COPY --from=base /app/state /app/state
COPY --from=base /app/knowledge /app/knowledge
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN mkdir -p /app/db_cache && chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["python", "-m", "tools.news_agregator.fred_server", "server"]
