"""Documented caveats, attached to the tools they apply to.

A limitation that lives only in prose cannot be acted on. "Short interest is
2-3 weeks stale" is in the README and in an audit; an agent calling
`get_short_interest` and comparing the answer against today's price has no way
to learn it. Moving these onto the response is the whole point of the
`warnings` field.

The risk in curating them is drift in the other direction: a caveat naming a
tool that has been renamed describes something nobody can call, and reads as
reassurance that a limit is handled. So every curated entry is checked against
the live catalogue, and a stale one fails here rather than misleading a reader.
"""
import asyncio

import pytest

SERVERS = [
    ("sec", "tools.web_search_server.web_search", "WebSearchServer"),
    ("finnhub", "tools.news_agregator.finnhub_server", "FinnhubServer"),
    ("fred", "tools.news_agregator.fred_server", "FredServer"),
    ("altdata", "tools.altdata_server.server", "AltDataServer"),
]


# Declared but hidden unless their capability is present, so they are absent
# from a `tools/list` in this environment without being absent from the
# catalogue. Naming a provider for one is correct: it applies whenever the
# tool is actually served.
CAPABILITY_GATED = {"rag_search", "rag_ingest", "search"}


def _server(module_name, class_name):
    import importlib
    module = importlib.import_module(module_name)
    return getattr(module, class_name)()


def _tool_names(instance):
    async def go():
        for key, handler in instance.server.request_handlers.items():
            if "ListTools" in str(key):
                return {t.name for t in (await handler(None)).root.tools}
        return set()
    return asyncio.run(go())


def _annotation_config(instance):
    """The provider and warning config the dispatcher was wrapped with.

    The MCP SDK wraps the registered function in its own handler and does not
    use functools.wraps, so there is no `__wrapped__` to follow. Ours is held
    in the wrapping handler's closure.
    """
    for key, handler in instance.server.request_handlers.items():
        if "CallTool" not in str(key):
            continue
        config = getattr(handler, "annotation_config", None)
        if config is not None:
            return config
        for cell in (handler.__closure__ or ()):
            config = getattr(cell.cell_contents, "annotation_config", None)
            if config is not None:
                return config
    return None


@pytest.mark.parametrize("label,module,cls", SERVERS,
                         ids=[s[0] for s in SERVERS])
def test_the_dispatcher_exposes_its_annotation_config(label, module, cls):
    """Without this the manifest would have to restate what responses report,
    and the two would drift apart."""
    config = _annotation_config(_server(module, cls))
    assert config is not None, f"{label}: no annotation_config on the dispatcher"
    assert config["provider"], f"{label}: no default provider"


@pytest.mark.parametrize("label,module,cls", SERVERS,
                         ids=[s[0] for s in SERVERS])
def test_every_curated_caveat_names_a_tool_that_exists(label, module, cls):
    instance = _server(module, cls)
    config = _annotation_config(instance)
    served = _tool_names(instance)

    for name in (config.get("warnings_per_tool") or {}):
        assert name in served, (
            f"{label}: a caveat is attached to {name!r}, which this server "
            f"does not serve. A limitation describing a tool nobody can call "
            f"reads as a limit that is handled.")


@pytest.mark.parametrize("label,module,cls", SERVERS,
                         ids=[s[0] for s in SERVERS])
def test_every_per_tool_provider_names_a_tool_that_exists(label, module, cls):
    instance = _server(module, cls)
    config = _annotation_config(instance)
    served = _tool_names(instance)

    for name in (config.get("per_tool") or {}):
        assert name in served or name in CAPABILITY_GATED, (
            f"{label}: a provider override names {name!r}, which is neither "
            f"served nor a known capability-gated tool")


@pytest.mark.parametrize("label,module,cls", SERVERS,
                         ids=[s[0] for s in SERVERS])
def test_curated_caveats_are_structured(label, module, cls):
    """A free-text caveat cannot be branched on, which is why they stayed in
    documentation in the first place."""
    config = _annotation_config(_server(module, cls))
    for name, entries in (config.get("warnings_per_tool") or {}).items():
        assert isinstance(entries, (list, tuple)), f"{name}: not a list"
        for entry in entries:
            assert isinstance(entry, dict), f"{name}: {entry!r} is not a dict"
            assert entry.get("code"), f"{name}: a caveat with no code"
            assert entry.get("message"), f"{name}: a caveat with no message"


def test_the_caveats_that_prompted_this_are_actually_attached():
    """Named explicitly so removing one is a deliberate act, not an omission.

    get_debt_maturity_schedule is deliberately NOT in this list. Its caveat is
    about coverage, which is a property of each answer rather than of the
    tool: attached statically it fired on responses whose six buckets were
    verified against SEC and whose own `coverage` field said "full". The
    tool-level version of that fact lives in tools/manifest.py, and the
    per-response one is raised by the tool only when the answer really is
    incomplete. `testing/test_conditional_warnings.py` holds both halves.
    """
    expected = {
        "sec": {"get_earnings_transcripts"},
        "altdata": {"get_congress_holdings"},
    }
    for label, module, cls in SERVERS:
        if label not in expected:
            continue
        config = _annotation_config(_server(module, cls))
        attached = set(config.get("warnings_per_tool") or {})
        missing = expected[label] - attached
        assert not missing, f"{label}: no caveat attached to {sorted(missing)}"
