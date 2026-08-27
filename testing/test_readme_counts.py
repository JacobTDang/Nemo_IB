"""The README says its counts are measured. This is the measurement.

They were not measured. They were written by hand, which is why `sec` drifted
to 50 tools while the page still said 48 and the total still said 96 -- a
number nobody could have caught by reading, because the page asserting it is
the same page that is wrong.

A count that has to be kept in step by remembering will fall out of step. This
reads the tool registry each server actually serves, reads the numbers off the
README, and fails when they disagree. Adding a tool now means the page changes
in the same commit or the suite goes red.
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
    """What this server actually serves, from its own list_tools handler."""
    srv = _server(*SERVERS[name])
    handler = srv.request_handlers[types.ListToolsRequest]
    result = asyncio.run(handler(types.ListToolsRequest(method="tools/list")))
    return [t.name for t in result.root.tools]


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


def test_the_total_is_the_sum_of_the_rows(readme):
    stated = re.search(r"declare \*\*(\d+) tools\*\*", readme)
    assert stated, "the README no longer states a total"
    served = sum(len(_declared(n)) for n in SERVERS)
    assert int(stated.group(1)) == served, (
        f"README totals {stated.group(1)}; the five servers serve {served}")


def test_the_table_rows_add_up_to_the_stated_total(readme):
    """Independently of the servers: the page has to agree with itself."""
    rows = [int(n) for n in
            re.findall(r"^\|\s*`\w+`\s*\|\s*(\d+)\s*\|", readme, re.M)]
    stated = int(re.search(r"declare \*\*(\d+) tools\*\*", readme).group(1))
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
    served = set(_declared(name))
    # Backticked lowercase identifiers with an underscore read as tool names;
    # bare words like the server's own name do not.
    listed = {t for t in re.findall(r"`([a-z][a-z0-9_]+)`",
                                    _section(readme, name)) if "_" in t}
    # Names the prose legitimately mentions that belong elsewhere.
    elsewhere = {"amount_min", "amount_max", "amount_max_total", "as_of",
                 "recorded_at", "fiscal_period", "eps_actual", "eps_estimate",
                 "auto_adjust", "adj_close"}
    stale = listed - served - elsewhere
    assert not stale, (
        f"the `{name}` section names tools it does not serve: {sorted(stale)}")
