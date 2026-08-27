"""The README says its counts are measured. This is the measurement.

They were written by hand, and a count kept in step by remembering falls out of
step. This reads the registry each server declares, reads the numbers off the
page, and fails when they disagree.

What it measures is what the **image** serves, which is not the same as what
the machine running the tests serves. Three SEC tools are capability-gated --
`search` needs SearXNG, `rag_search` and `rag_ingest` need `agent.rag` -- and
the image installs neither, so it serves 48 of the 51 declared. A development
box with the RAG package present serves 49 or 50, and a test that simply
counted `list_tools` here would pin the laptop's number to a page describing
the image. That is the mistake this file exists to make impossible, and it was
made once already: the page's original 96 was right, a local measurement of 98
looked like a correction, and it was not.

So the gates are evaluated as the image evaluates them -- closed -- rather than
as this host happens to.
"""
import asyncio
import importlib
import inspect
import pathlib
import re

import mcp.types as types
import pytest

# The five that ship in the image, and how to reach the server object each
# module exposes -- they do not agree on a convention.
SERVERS = {
    "sec": ("tools.web_search_server.web_search", None),
    "financial": ("tools.financial_modeling_engine.analysis_tools",
                  "Financial_Analysis"),
    "finnhub": ("tools.news_agregator.finnhub_server", "FinnhubServer"),
    "fred": ("tools.news_agregator.fred_server", None),
    "altdata": ("tools.altdata_server.server", None),
}

README = pathlib.Path(__file__).resolve().parent.parent / "README.md"


def _server(modname, cls):
    mod = importlib.import_module(modname)
    if cls:
        return getattr(mod, cls)().server
    for obj in vars(mod).values():
        if obj.__class__.__name__ == "Server":
            return obj
    for attr, obj in vars(mod).items():
        if inspect.isclass(obj) and attr.endswith("Server") \
                and obj.__module__ == modname:
            return obj().server
    raise AssertionError(f"no MCP server object found in {modname}")


def _declared(name):
    """What the image serves for this server.

    Every tool the module declares, minus the ones capability-gating hides in
    the image. Read from the registry rather than from `list_tools`, because
    `list_tools` answers for the host it runs on and the README describes the
    image.
    """
    mod = importlib.import_module(SERVERS[name][0])
    tools = getattr(mod, "_ALL_TOOLS", None)
    if tools is None:
        srv = _server(*SERVERS[name])
        handler = srv.request_handlers[types.ListToolsRequest]
        result = asyncio.run(handler(types.ListToolsRequest(method="tools/list")))
        return [t.name for t in result.root.tools]
    gated = set(getattr(mod, "_GATED_TOOLS", {}))
    return [t.name for t in tools if t.name not in gated]


def _all_declared(name):
    """Including the gated ones, which the page states separately."""
    mod = importlib.import_module(SERVERS[name][0])
    tools = getattr(mod, "_ALL_TOOLS", None)
    if tools is not None:
        return [t.name for t in tools]
    return _declared(name)


@pytest.fixture(scope="module")
def readme():
    return README.read_text()


@pytest.mark.parametrize("name", sorted(SERVERS))
def test_the_table_row_matches_what_the_server_serves(name, readme):
    row = re.search(rf"^\|\s*`{name}`\s*\|\s*(\d+)\s*\|", readme, re.M)
    assert row, f"no table row for `{name}` in the README"
    assert int(row.group(1)) == len(_declared(name)), (
        f"README says {row.group(1)} tools for `{name}`; it serves "
        f"{len(_declared(name))}")


@pytest.mark.parametrize("name", sorted(SERVERS))
def test_the_section_heading_matches_too(name, readme):
    """The heading and the table are written separately and drift apart
    separately."""
    heading = re.search(rf"^### `{name}` — (\d+) tools", readme, re.M)
    assert heading, f"no `### {name}` heading in the README"
    assert int(heading.group(1)) == len(_declared(name)), (
        f"the `{name}` heading says {heading.group(1)}; it serves "
        f"{len(_declared(name))}")


def test_the_total_is_what_the_image_serves(readme):
    stated = re.search(r"serve \*\*(\d+) tools\*\*", readme)
    assert stated, "the README no longer states what the image serves"
    served = sum(len(_declared(n)) for n in SERVERS)
    assert int(stated.group(1)) == served, (
        f"README says the image serves {stated.group(1)}; it serves {served}")


def test_the_declared_total_is_stated_too(readme):
    """The gap between declared and served is the whole capability-gating
    story, and a page giving one number without the other invites exactly the
    correction that was made wrongly once."""
    stated = re.search(r"They declare (\d+):", readme)
    assert stated, "the README no longer states the declared total"
    declared = sum(len(_all_declared(n)) for n in SERVERS)
    assert int(stated.group(1)) == declared, (
        f"README says {stated.group(1)} declared; the modules declare {declared}")


def test_the_table_rows_add_up_to_the_stated_total(readme):
    """Independently of the servers: the page has to agree with itself."""
    rows = [int(n) for n in
            re.findall(r"^\|\s*`\w+`\s*\|\s*(\d+)\s*\|", readme, re.M)]
    stated = int(re.search(r"serve \*\*(\d+) tools\*\*", readme).group(1))
    assert sum(rows) == stated, f"rows sum to {sum(rows)}, page says {stated}"


def test_every_tool_named_in_the_readme_actually_exists(readme):
    """The prose names individual tools. One renamed and not updated here reads
    as a tool the reader cannot find."""
    served = {t for name in SERVERS for t in _declared(name)}
    # Backticked identifiers that look like tool calls, e.g. `get_public_float`.
    mentioned = set(re.findall(r"`(get_[a-z0-9_]+|extract_[a-z0-9_]+|"
                               r"calculate_[a-z0-9_]+|compare_[a-z0-9_]+|"
                               r"analyze_[a-z0-9_]+|find_[a-z0-9_]+|"
                               r"track_[a-z0-9_]+|diff_[a-z0-9_]+|"
                               r"list_[a-z0-9_]+|backtest_[a-z0-9_]+|"
                               r"record_[a-z0-9_]+)\(?", readme))
    # Names belonging to the research package rather than to a server.
    research = {"record_daily_bars", "record_consensus_snapshots",
                "record_universe", "record_scan", "record_prints",
                "record_paper_orders", "record_bars", "record_consensus",
                "list_known_funds"}
    unknown = {m for m in mentioned if m not in served} - research
    assert not unknown, f"README names tools no server serves: {sorted(unknown)}"


def _section(readme, name):
    m = re.search(rf"### `{name}` — \d+ tools.*?(?=\n### |\n## )", readme, re.S)
    assert m, f"no `### {name}` section in the README"
    return m.group(0)


@pytest.mark.parametrize("name", sorted(SERVERS))
def test_the_section_names_every_tool_the_server_serves(name, readme):
    """Stronger than the count, and the property a reader actually relies on:
    a tool that exists and is not on the page is one nobody finds."""
    served = set(_declared(name))
    listed = set(re.findall(r"`([a-z][a-z0-9_]+)`", _section(readme, name)))
    missing = served - listed
    assert not missing, (
        f"`{name}` serves these and the README does not name them: "
        f"{sorted(missing)}")


@pytest.mark.parametrize("name", sorted(SERVERS))
def test_the_section_names_no_tool_the_server_does_not_serve(name, readme):
    """The other direction: a renamed tool still listed under its old name
    sends the reader looking for something that is not there."""
    # A gated tool is a real tool that this image hides, and the page
    # documents it as such -- that is accurate, not stale.
    real = set(_all_declared(name))
    # Backticked lowercase identifiers with an underscore read as tool names;
    # bare words like the server's own name do not.
    listed = {t for t in re.findall(r"`([a-z][a-z0-9_]+)`",
                                    _section(readme, name)) if "_" in t}
    # Names the prose legitimately mentions that belong elsewhere.
    elsewhere = {"amount_min", "amount_max", "amount_max_total", "as_of",
                 "recorded_at", "fiscal_period", "eps_actual", "eps_estimate",
                 "auto_adjust", "adj_close"}
    stale = listed - real - elsewhere
    assert not stale, (
        f"the `{name}` section names tools it does not serve: {sorted(stale)}")


@pytest.mark.parametrize("name", sorted(SERVERS))
def test_a_gated_tool_is_never_counted_as_served(name, readme):
    """The row and the heading state what the image serves, so a tool it hides
    must not be in that number. Counting them is the arithmetic that turned 96
    into a wrong 98."""
    gated = set(_all_declared(name)) - set(_declared(name))
    row = int(re.search(rf"^\|\s*`{name}`\s*\|\s*(\d+)\s*\|",
                        readme, re.M).group(1))
    assert row == len(_all_declared(name)) - len(gated)


def test_the_sec_gating_is_described_accurately(readme):
    """The page explains why declared and served differ. It said two tools and
    50 declared; it is three and 51, and the same sentence was wrong in
    deploy/README.md with a third set of numbers again."""
    from tools.web_search_server import web_search

    gated = sorted(web_search._GATED_TOOLS)
    section = _section(readme, "sec")
    for tool in gated:
        assert f"`{tool}`" in section, f"{tool} is gated and not explained"
    declared = len(web_search._ALL_TOOLS)
    assert f"declares {declared}" in section
    assert f"serves {declared - len(gated)}" in section
