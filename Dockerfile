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
    uv sync --frozen --no-install-project \
      --no-install-package torch \
      --no-install-package sentence-transformers

COPY tools/ ./tools/
COPY agent/ ./agent/
COPY state/ ./state/
COPY knowledge/ ./knowledge/

# The tool cache lives here; mount a volume so it survives container replacement.
RUN mkdir -p /app/db_cache
ENV NEMO_CACHE_DB_PATH=/app/db_cache/tool_cache.db \
    NEMO_DB_PATH=/app/db_cache/session.db \
    PYTHONPATH=/app

# Fails the build if any server cannot be imported, so a broken image never
# reaches the host.
RUN python -c "\
import importlib;\
mods=['tools.news_agregator.fred_server','tools.news_agregator.finnhub_server',\
'tools.web_search_server.web_search','tools.financial_modeling_engine.analysis_tools',\
'tools.altdata_server.server'];\
[importlib.import_module(m) for m in mods];\
print('all 5 data-source servers import')"

# Default to the simplest server; compose overrides this per service.
CMD ["python", "-m", "tools.news_agregator.fred_server", "server"]


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
COPY --from=base /app/agent /app/agent
COPY --from=base /app/state /app/state
COPY --from=base /app/knowledge /app/knowledge
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN mkdir -p /app/db_cache && chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["python", "-m", "tools.news_agregator.fred_server", "server"]
