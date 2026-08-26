"""A machine-readable description of what this tool catalogue covers.

`tools/list` gives a caller names, descriptions and input schemas. It does not
say which upstream answers a tool, how fresh that answer can be, or what the
tool is known to miss -- and those are the facts an agent needs before it
decides whether an empty result means "nothing there" or "we cannot see it".
Until now they existed only as prose in `README.md` and in
`docs/audits/2026-08-25-codex-audit.md`.

Two things are built rather than written down here:

* **Names and counts** come from instantiating each server and reading its
  registered `list_tools` handler. A literal list would drift, which is how the
  README came to advertise 76 tools while the servers served 96. It also makes
  the manifest honest about capability gating: `search` and the `rag_*` pair
  are hidden when SearXNG or the RAG stack is absent, so the count is a
  property of this deployment rather than of the source tree.

* **Providers, and the caveats responses already carry**, come from the
  `annotating(...)` wrapper each server applies to its `call_tool` dispatcher.
  That wrapper is what stamps `provider` and `warnings` on the response a
  caller actually holds; reading it here means the manifest cannot disagree
  with what the responses report, and cannot quietly omit a caveat a caller
  would see at call time.

`CURATED_LIMITS` below adds the two things the dispatcher cannot express: the
freshness a tool's data can be expected to have, and documented limitations
that are not attached as response warnings. The rule for it is strict --
nothing goes in that is not already written down in this repository. A guessed
caveat is worse than a missing one because it will be believed.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
from datetime import datetime, timezone
from typing import Any, Iterable, NamedTuple

SCHEMA_VERSION = "1"


class ServerSpec(NamedTuple):
    """Where a server lives. The only hand-maintained list in this module --
    you cannot introspect a server you have not imported. Everything about the
    tools themselves is read from the instance."""
    name: str
    module: str
    cls: str


# The five servers that ship in the homelab image. alpaca, excel and sentry are
# deliberately excluded from the image (they place orders and read book state),
# so a data-source manifest that advertised them would describe a surface this
# deployment does not expose.
SERVERS: tuple[ServerSpec, ...] = (
    ServerSpec("sec", "tools.web_search_server.web_search", "WebSearchServer"),
    ServerSpec("financial", "tools.financial_modeling_engine.analysis_tools",
               "Financial_Analysis"),
    ServerSpec("finnhub", "tools.news_agregator.finnhub_server", "FinnhubServer"),
    ServerSpec("fred", "tools.news_agregator.fred_server", "FredServer"),
    ServerSpec("altdata", "tools.altdata_server.server", "AltDataServer"),
)


# ---------------------------------------------------------------------------
# CURATED. Everything below this line is written by hand, and every entry has a
# source in this repository -- `README.md` or
# `docs/audits/2026-08-25-codex-audit.md` (section 7, "Data limitations are
# documented but not always machine-readable"). The citation is in the comment
# above each group.
#
# This dict is deliberately NOT a second copy of the caveats the servers attach
# through `annotating(..., warnings_per_tool=...)`. Those already ride on the
# response and are merged in by `_describe_tool`, so restating one here would
# create the two-lists-that-drift problem the extraction exists to avoid. What
# belongs here is what the dispatcher cannot say: an expected freshness, and a
# documented limitation that is not attached as a response warning.
#
# Rules for adding to this dict:
#   1. If it is not documented in the repository, it does not go in. A tool
#      with nothing documented anywhere gets an empty `known_limits` and a null
#      `expected_freshness`.
#   2. `expected_freshness` is null unless a source states a lag. "Probably
#      daily" is a guess, and a caller will treat it as a fact.
#   3. Every key must name a tool some server actually serves.
#      testing/test_tool_manifest.py fails on a stale one, because a caveat
#      attached to a deleted tool reads as a limitation that is handled.
#
# Note what is deliberately absent: the audit's "Google Trends coverage is
# unreliable because of rate limiting" names no tool in this catalogue, so
# there is nothing here to attach it to.
# ---------------------------------------------------------------------------

CURATED_LIMITS: dict[str, dict[str, Any]] = {

    # --- financial ---------------------------------------------------------
    # Audit s7: "Short interest is normally 2-3 weeks stale." The staleness and
    # the absence of a live alternative both ride on the response already; only
    # the freshness field is new here.
    "get_short_interest": {
        "expected_freshness": "normally 2-3 weeks stale",
        "known_limits": [],
    },
    # Audit s7: "Options quotes can be stale after hours." The response already
    # carries that and the illiquid-strike sentinel; only the freshness is new.
    "get_options_metrics": {
        "expected_freshness": "quotes can be stale outside regular trading hours",
        "known_limits": [],
    },
    # README, `financial` section: these "read book state, which a data-source
    # host does not have. They return an empty result rather than failing."
    "analyze_exposures": {
        "expected_freshness": None,
        "known_limits": [
            "Reads book state, which the shipped data-source image does not "
            "have; there it returns an empty result rather than failing.",
        ],
    },
    "get_thesis_evolution": {
        "expected_freshness": None,
        "known_limits": [
            "Reads book state, which the shipped data-source image does not "
            "have; there it returns an empty result rather than failing.",
        ],
    },

    # --- sec ---------------------------------------------------------------
    # Audit s7: "Some nonstandard 10-K layouts defeat section extraction." The
    # response carries this on extract_mda; extract_risk_factors is the other
    # named-section extractor and the same sentence covers it.
    "extract_risk_factors": {
        "expected_freshness": None,
        "known_limits": [
            "Some nonstandard 10-K layouts defeat section extraction.",
        ],
    },

    # --- finnhub -----------------------------------------------------------
    # Audit s7: "There is no KPI-level consensus source."
    "get_forward_estimates": {
        "expected_freshness": None,
        "known_limits": [
            "No KPI-level consensus source is available; only the headline "
            "estimates Finnhub publishes.",
        ],
    },

    # --- altdata -----------------------------------------------------------
    # README, `altdata` section: "these are transactions and year-covering
    # snapshots, not live positions. Congress publishes no current holdings.
    # Members file up to 45 days after trading ... Filings that arrive as scans
    # of paper cannot be parsed at all and are counted in `coverage` rather
    # than dropped. Holdings are currently Senate-only."
    "get_congress_trades": {
        "expected_freshness": "up to 45 days after the transaction",
        "known_limits": [
            "Transactions, not positions: Congress publishes no current "
            "holdings.",
            "Filings that arrive as scans of paper cannot be parsed and are "
            "counted in `coverage` rather than dropped.",
        ],
    },
    "get_congress_holdings": {
        "expected_freshness": "annual; reports arrive months after the year "
                              "they cover",
        "known_limits": [
            "Roughly a third of holding rows are Excepted Investment Funds "
            "whose contents are legally not itemised.",
            "Filings that arrive as scans of paper cannot be parsed and are "
            "counted in `coverage` rather than dropped.",
        ],
    },
    "get_congress_leaderboard": {
        "expected_freshness": "up to 45 days after the transaction",
        "known_limits": [
            "Ranks disclosed transactions, not positions: Congress publishes "
            "no current holdings.",
        ],
    },
}


# ---------------------------------------------------------------------------
# Introspection
# ---------------------------------------------------------------------------

def _handler_for(instance, request: str):
    """The server's registered handler for `request` ("ListTools"/"CallTool").

    Matched on the string form of the key because the MCP layer keys
    `request_handlers` by request type, and importing those types here would
    couple the manifest to a private corner of the library's module layout.
    """
    for key, handler in instance.server.request_handlers.items():
        if request in str(key):
            return handler
    return None


def _functions_reachable_from(fn, depth: int = 4) -> Iterable[Any]:
    """`fn` plus the functions it closes over, breadth-first, depth-bounded.

    The MCP `call_tool()` decorator registers a closure of its own, and our
    annotating wrapper is a cell inside it. Walking closures rather than
    hard-coding "the cell named func" keeps this working if the library adds
    another layer.
    """
    seen: set[int] = set()
    queue = [(fn, depth)]
    while queue:
        current, remaining = queue.pop(0)
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        if remaining <= 0:
            continue
        for cell in getattr(current, "__closure__", None) or ():
            try:
                value = cell.cell_contents
            except ValueError:      # an unfilled cell, mid-definition
                continue
            if inspect.isfunction(value):
                queue.append((value, remaining - 1))
        wrapped = getattr(current, "__wrapped__", None)
        if inspect.isfunction(wrapped):
            queue.append((wrapped, remaining - 1))


def _annotation_config(instance) -> dict:
    """What the server's dispatcher was wrapped with: providers and caveats.

    Preferred path is the `annotation_config` attribute `annotating()` exposes.
    The fallback reads the wrapper's closure directly, so the manifest still
    reports true providers on a build where that attribute is absent -- which
    matters because the alternative is restating the providers here, and the
    two copies would drift.

    Raises rather than returning a default: a server whose provider config
    cannot be found would otherwise produce a manifest naming the wrong
    upstream, and a wrong provenance is worse than no manifest.
    """
    dispatcher = _handler_for(instance, "CallTool")
    if dispatcher is None:
        raise LookupError("server registered no CallTool handler")

    def shaped(provider, per_tool, warnings_per_tool) -> dict:
        return {
            "provider": provider,
            "per_tool": dict(per_tool or {}),
            "warnings_per_tool": {
                name: list(entries)
                for name, entries in (warnings_per_tool or {}).items()
            },
        }

    for fn in _functions_reachable_from(dispatcher):
        config = getattr(fn, "annotation_config", None)
        if config is not None:
            return shaped(config["provider"], config.get("per_tool"),
                          config.get("warnings_per_tool"))

    for fn in _functions_reachable_from(dispatcher):
        free = getattr(getattr(fn, "__code__", None), "co_freevars", ())
        if "provider" in free and "per_tool" in free:
            cells = fn.__closure__

            def cell(name):
                return cells[free.index(name)].cell_contents if name in free else None

            return shaped(cell("provider"), cell("per_tool"),
                          cell("warnings_per_tool"))

    raise LookupError(
        "the CallTool dispatcher carries no annotating() configuration; the "
        "manifest cannot report a provider it did not read")


def _list_tools(instance) -> list:
    """The tools this server would answer `tools/list` with, right now."""
    handler = _handler_for(instance, "ListTools")
    if handler is None:
        raise LookupError("server registered no ListTools handler")
    return list(asyncio.run(handler(None)).root.tools)


def _known_limits(tool_name: str, config: dict) -> list[str]:
    """Caveats the response carries, then those only the manifest records.

    The dispatcher's warnings come first because those are what a caller sees
    at call time; reading them here rather than restating them is what keeps
    the manifest from omitting a caveat the tool itself reports. Deduplicated
    on the exact string, so a limitation curated in both places is stated once.
    """
    limits: list[str] = []
    for entry in config["warnings_per_tool"].get(tool_name, []):
        message = entry.get("message") if isinstance(entry, dict) else None
        if message and message not in limits:
            limits.append(message)
    for message in CURATED_LIMITS.get(tool_name, {}).get("known_limits", []):
        if message not in limits:
            limits.append(message)
    return limits


def _describe_tool(tool, config: dict) -> dict:
    curated = CURATED_LIMITS.get(tool.name, {})
    return {
        "name": tool.name,
        "provider": config["per_tool"].get(tool.name, config["provider"]),
        "description": tool.description,
        # Null, not a guess. See the rules above CURATED_LIMITS.
        "expected_freshness": curated.get("expected_freshness"),
        "known_limits": _known_limits(tool.name, config),
    }


def _describe_server(spec: ServerSpec) -> dict:
    entry = {
        "name": spec.name,
        "module": spec.module,
        "tool_count": 0,
        "tools": [],
        "error": None,
    }
    try:
        module = importlib.import_module(spec.module)
        instance = getattr(module, spec.cls)()
        config = _annotation_config(instance)
        entry["tools"] = [_describe_tool(t, config) for t in _list_tools(instance)]
        entry["tool_count"] = len(entry["tools"])
    except Exception as exc:                       # noqa: BLE001
        # Recorded, never swallowed and never dropped. A server missing from a
        # manifest reads as "this server does not exist", which is a stronger
        # claim than "it could not be constructed here" and sends a caller
        # looking for the capability somewhere else.
        entry["error"] = f"{type(exc).__name__}: {exc}"
    return entry


def build_manifest(specs: Iterable[ServerSpec] = SERVERS) -> dict:
    """Describe the tool catalogue as this deployment can actually serve it.

    `specs` exists for tests; production callers want the default. Nothing is
    cached: the catalogue is capability-gated, so a manifest built before
    SearXNG came up would keep claiming `search` is unavailable.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc)
            .replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "servers": [_describe_server(spec) for spec in specs],
    }


if __name__ == "__main__":
    import json
    print(json.dumps(build_manifest(), indent=2))
