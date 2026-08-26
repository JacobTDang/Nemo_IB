"""What the catalogue covers, in a form an agent can read.

`tools/list` tells a caller a tool's name, description and input schema. It
says nothing about which upstream answers it, how fresh that answer can be, or
what the tool is known to miss. That lives in README prose and in an audit
document, so an agent planning a query cannot reason about it and a caller
comparing a three-week-old short-interest figure against today's price has no
way to learn that it is three weeks old.

The manifest exists to move that into machine-readable form. Two failure modes
matter more than the feature itself:

1. **Drift.** A hand-maintained inventory decays. The README claimed 76 tools
   while the servers served 96 -- nothing checked, so nothing complained. Every
   name and count in the manifest therefore comes from instantiating the
   servers and reading their registered `list_tools` handler, and these tests
   compare the result against an independent introspection so a tool added next
   month cannot go missing quietly.

2. **Invention.** A caveat that was guessed rather than sourced is worse than a
   missing one, because it will be believed. So a tool with nothing documented
   carries an empty `known_limits` and a null `expected_freshness`, and a
   curated entry naming a tool nobody can call fails here rather than shipping
   as reassurance that a limit is handled.

The caveats themselves live in two places -- the warnings each server attaches
to its responses, and the freshness and extra limits curated in
`tools/manifest.py` -- so two tests below pin them together: every caveat the
manifest states must trace to one of those sources, and every caveat a response
carries must appear in the manifest.
"""
import asyncio
import datetime as dt
import importlib
import json

import pytest

from tools.manifest import CURATED_LIMITS, SERVERS, build_manifest


# --------------------------------------------------------------------------
# Independent introspection.
#
# Deliberately not imported from tools.manifest. If the manifest and the check
# shared a helper, a bug in that helper would report itself as agreement --
# both sides would be wrong in the same direction and the test would pass.
# --------------------------------------------------------------------------

def _live_catalogue():
    """{server name: {tool name: description}} read straight from the servers."""
    async def go():
        catalogue = {}
        for spec in SERVERS:
            module = importlib.import_module(spec.module)
            instance = getattr(module, spec.cls)()
            for key, handler in instance.server.request_handlers.items():
                if "ListTools" in str(key):
                    result = await handler(None)
                    catalogue[spec.name] = {
                        tool.name: tool.description for tool in result.root.tools
                    }
        return catalogue
    return asyncio.run(go())


def _live_annotation_configs():
    """{server name: the config its `annotating(...)` dispatcher was built with}.

    Reached by hand through the MCP layer's own closure rather than through the
    manifest's helper, for the same reason as above.
    """
    configs = {}
    for spec in SERVERS:
        module = importlib.import_module(spec.module)
        instance = getattr(module, spec.cls)()
        for key, handler in instance.server.request_handlers.items():
            if "CallTool" not in str(key):
                continue
            code = handler.__code__
            if "func" in code.co_freevars:
                inner = handler.__closure__[
                    code.co_freevars.index("func")].cell_contents
                configs[spec.name] = getattr(inner, "annotation_config", None)
    return configs


@pytest.fixture(scope="module")
def manifest():
    return build_manifest()


@pytest.fixture(scope="module")
def live():
    return _live_catalogue()


@pytest.fixture(scope="module")
def configs():
    return _live_annotation_configs()


# --------------------------------------------------------------------------
# Shape
# --------------------------------------------------------------------------

def test_the_manifest_declares_its_schema_version(manifest):
    """A consumer that cannot tell which shape it received cannot upgrade
    safely; the version is what lets the shape change without breaking it."""
    assert manifest["schema_version"] == "1"


def test_generated_at_is_utc_iso8601_ending_in_z(manifest):
    """A timestamp with no zone is ambiguous by up to a day. `Z` is the one
    suffix every ISO-8601 parser agrees means UTC, and a manifest read by a
    machine in another timezone must not be off by the offset."""
    stamp = manifest["generated_at"]
    assert stamp.endswith("Z"), f"generated_at is not UTC-marked: {stamp!r}"

    parsed = dt.datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    assert parsed.tzinfo is not None
    assert parsed.utcoffset() == dt.timedelta(0)


def test_the_manifest_survives_a_json_round_trip(manifest):
    """It is a machine-readable artifact or it is nothing. A datetime or a set
    left in the structure serialises here rather than at whatever boundary
    ships it."""
    assert json.loads(json.dumps(manifest)) == manifest


# --------------------------------------------------------------------------
# Servers
# --------------------------------------------------------------------------

def test_every_shipped_server_appears(manifest):
    """A server absent from the manifest reads as "this server does not
    exist", which is a stronger and more wrong claim than "it failed to
    load"."""
    named = [server["name"] for server in manifest["servers"]]
    assert named == [spec.name for spec in SERVERS]


def test_a_server_that_cannot_be_constructed_is_recorded_with_an_error():
    """Construction can fail in an environment without credentials or without
    an optional dependency. Dropping the server would silently shrink the
    catalogue; recording the failure keeps "we could not load it" and "it is
    not there" distinguishable."""
    from tools.manifest import ServerSpec

    broken = ServerSpec("ghost", "tools.no_such_module_exists", "Nope")
    result = build_manifest(specs=[broken])

    entry = result["servers"][0]
    assert entry["name"] == "ghost"
    assert entry["error"], "a server that failed to construct recorded no error"
    assert entry["tool_count"] == 0
    assert entry["tools"] == []


def test_a_server_that_loaded_records_no_error(manifest):
    """The inverse of the above: `error` is present on every entry so a reader
    checks one key rather than guessing from a missing one, and it must be
    null when nothing went wrong."""
    for server in manifest["servers"]:
        assert server["error"] is None, (
            f"{server['name']} failed to construct: {server['error']}")


# --------------------------------------------------------------------------
# Tools come from the servers, not from a list
# --------------------------------------------------------------------------

def test_every_tool_the_servers_serve_appears_in_the_manifest(manifest, live):
    """The drift guard in the direction that matters most. A tool added to a
    server and missing from the manifest is a capability an agent will never
    know it has -- exactly how the README came to advertise 76 of 96 tools."""
    for server in manifest["servers"]:
        listed = {tool["name"] for tool in server["tools"]}
        assert listed == set(live[server["name"]]), (
            f"{server['name']}: manifest and live catalogue disagree; "
            f"missing {set(live[server['name']]) - listed}, "
            f"extra {listed - set(live[server['name']])}")


def test_tool_count_equals_the_number_of_tools_listed(manifest):
    """A count maintained separately from the thing it counts is the original
    bug in miniature."""
    for server in manifest["servers"]:
        assert server["tool_count"] == len(server["tools"])


def test_the_combined_count_matches_live_introspection(manifest, live):
    """No number in the manifest is asserted against a constant here on
    purpose: the catalogue is capability-gated, so `search` and the `rag_*`
    pair come and go with SearXNG and the RAG stack. A hard-coded total would
    fail on a machine that is merely configured differently, and would teach
    the next reader to update the constant rather than trust the servers."""
    total = sum(server["tool_count"] for server in manifest["servers"])
    assert total == sum(len(tools) for tools in live.values())
    assert total > 0, "introspection produced an empty catalogue"


def test_descriptions_are_the_servers_own(manifest, live):
    """A restated description drifts from the schema the caller is validated
    against. Copying it from `list_tools` is the only way the two agree."""
    for server in manifest["servers"]:
        for tool in server["tools"]:
            assert tool["description"] == live[server["name"]][tool["name"]]


# --------------------------------------------------------------------------
# Providers come from the annotating() seam
# --------------------------------------------------------------------------

def test_every_tool_names_a_provider(manifest):
    """"Which upstream said this" is the question the manifest exists to
    answer. A blank provider is the answer that cannot be acted on."""
    for server in manifest["servers"]:
        for tool in server["tools"]:
            assert tool["provider"], (
                f"{server['name']}.{tool['name']} names no provider")


def test_providers_match_what_responses_actually_report(manifest, configs):
    """Extracted from the `annotating(...)` wrapper rather than restated. Two
    lists of providers drift apart, and the manifest would then describe a
    source different from the one stamped on the response a caller holds."""
    by_name = {server["name"]: server for server in manifest["servers"]}
    for spec in SERVERS:
        config = configs.get(spec.name)
        assert config is not None, f"{spec.name}: no annotation config found"
        for tool in by_name[spec.name]["tools"]:
            expected = config["per_tool"].get(tool["name"], config["provider"])
            assert tool["provider"] == expected, (
                f"{spec.name}.{tool['name']}: manifest says "
                f"{tool['provider']!r}, responses say {expected!r}")


def test_a_multi_upstream_server_does_not_name_the_aggregator(manifest):
    """altdata reads six upstreams. Reporting all nine of its tools as
    "altdata" would name the aggregator instead of the source, which is the
    reason `per_tool` exists."""
    altdata = next(s for s in manifest["servers"] if s["name"] == "altdata")
    providers = {tool["provider"] for tool in altdata["tools"]}
    assert "altdata" not in providers, (
        "an altdata tool still reports the aggregator as its provider")
    assert len(providers) > 1


# --------------------------------------------------------------------------
# Curation: documented, or absent
# --------------------------------------------------------------------------

def test_every_curated_entry_names_a_tool_that_exists(manifest):
    """The drift guard in the other direction. A caveat attached to a renamed
    or deleted tool describes something nobody can call, and a reader scanning
    for "is this limitation handled?" sees the entry and concludes yes."""
    served = {
        tool["name"]
        for server in manifest["servers"]
        for tool in server["tools"]
    }
    stale = sorted(set(CURATED_LIMITS) - served)
    assert not stale, (
        f"curated entries name tools no server serves: {stale}. Rename or "
        f"remove them; a limitation describing a phantom tool reads as a "
        f"limitation that is handled.")


def test_every_caveat_in_the_manifest_traces_to_a_source(manifest, configs):
    """The rule that keeps the manifest trustworthy. An invented limitation is
    worse than a missing one because a caller will act on it, so every string
    in `known_limits` must come either from the warnings the server attaches to
    its responses or from `CURATED_LIMITS` -- and a tool documented nowhere
    carries an empty list and a null freshness rather than a plausible guess."""
    for server in manifest["servers"]:
        attached = configs[server["name"]]["warnings_per_tool"]
        for tool in server["tools"]:
            sourced = {
                entry["message"] for entry in attached.get(tool["name"], [])
            } | set(CURATED_LIMITS.get(tool["name"], {}).get("known_limits", []))
            unsourced = [x for x in tool["known_limits"] if x not in sourced]
            assert not unsourced, (
                f"{tool['name']} carries limitations with no source: {unsourced}")

            if tool["name"] not in CURATED_LIMITS:
                assert tool["expected_freshness"] is None, (
                    f"{tool['name']} claims a freshness with no curated source")


def test_every_caveat_a_response_carries_appears_in_the_manifest(
        manifest, configs):
    """The manifest must not under-report against the tool itself.

    The caveats are curated twice over: `annotating(..., warnings_per_tool=)`
    puts them on the response, and this module adds the freshness and the
    limits that are not expressible as a warning. Reading the first rather than
    copying it is what stops the two from disagreeing -- a caller who planned
    from the manifest and then met a warning the manifest never mentioned would
    have planned around a limit it did not know about."""
    for server in manifest["servers"]:
        attached = configs[server["name"]]["warnings_per_tool"]
        for tool in server["tools"]:
            for entry in attached.get(tool["name"], []):
                assert entry["message"] in tool["known_limits"], (
                    f"{tool['name']} reports a warning at call time that the "
                    f"manifest omits: {entry['message']!r}")


def test_curated_fields_have_the_declared_types(manifest):
    """`known_limits` is always a list so a consumer can iterate it without a
    None check, and an empty one means "nothing documented" rather than
    "nothing to worry about"."""
    for server in manifest["servers"]:
        for tool in server["tools"]:
            assert isinstance(tool["known_limits"], list)
            assert all(isinstance(item, str) for item in tool["known_limits"])
            assert tool["expected_freshness"] is None or isinstance(
                tool["expected_freshness"], str)


@pytest.mark.parametrize("tool_name,fragment", [
    # README and docs/audits/2026-08-25-codex-audit.md, section 7.
    ("get_short_interest", "2-3 weeks"),
    ("get_earnings_transcripts", "release"),
    ("get_debt_maturity_schedule", "not_covered"),
    ("get_congress_holdings", "Senate"),
])
def test_the_documented_limitations_reach_the_tools_they_describe(
        manifest, tool_name, fragment):
    """Spot-checks, not an inventory. These four are the ones the audit and
    README call out by name, and each is a case where the plain reading of an
    answer is wrong: an empty debt-maturity result is missing coverage rather
    than a debt-free filer, and congressional holdings are one chamber's
    year-old snapshot rather than anybody's current book."""
    tools = {
        tool["name"]: tool
        for server in manifest["servers"]
        for tool in server["tools"]
    }
    assert tool_name in tools, f"{tool_name} is no longer served"

    text = " ".join(tools[tool_name]["known_limits"])
    assert fragment in text, (
        f"{tool_name} no longer carries its documented limitation "
        f"(looked for {fragment!r} in {text!r})")
