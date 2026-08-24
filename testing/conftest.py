"""Per-module database isolation.

Tests write positions, theses, orders, and alert rows into one SQLite file.
Later modules then observe earlier modules' rows. That invents failures -- a
reconcile test sees another module's positions and reports a discrepancy --
and it masks them, because a schema test can pass on tables a previous module
happened to create. Measured 2026-08-22: the three position/risk modules
contribute 4 failures against the shared db_cache/session.db and pass 29/29
against a fresh one.

Module scope rather than function scope: many of these tests build state
across several test functions in the same file and would break under
per-function isolation.
"""
import os

import pytest


@pytest.fixture(autouse=True, scope="module")
def isolated_databases(tmp_path_factory, request):
    module_name = request.module.__name__.rsplit(".", 1)[-1]
    directory = tmp_path_factory.mktemp(f"db_{module_name}")

    previous = {}
    for variable, filename in (
        ("NEMO_DB_PATH", "session.db"),
        ("NEMO_CACHE_DB_PATH", "tool_cache.db"),
    ):
        previous[variable] = os.environ.get(variable)
        os.environ[variable] = str(directory / filename)

    # Empty file, not empty schema. Every daemon and MCP entrypoint calls
    # init_schema() on boot, so "the tables exist" is a real invariant rather
    # than a leftover -- but plenty of test modules relied on the shared
    # db_cache/session.db to supply it and never created their own. Give them
    # the boot invariant on a private file. Anything genuinely missing from
    # CREATE_SCHEMA still fails here, which is the point.
    from state.schema import init_schema
    init_schema()

    yield

    for variable, value in previous.items():
        if value is None:
            os.environ.pop(variable, None)
        else:
            os.environ[variable] = value


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "fail_missing_service(name, reason): gated test that must fail, not "
        "skip, when NEMO_REQUIRE_SERVICES=1 and the service is unavailable",
    )


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem):
    """Under NEMO_REQUIRE_SERVICES=1, a gated test fails instead of skipping.

    This fires in the call phase deliberately. The obvious alternative --
    pytest.fail() inside pytest_runtest_setup -- reports the test as an ERROR
    rather than a FAILURE, which muddies the "0 failed, 0 errors" target.
    Verified: setup gives `1 error`, this gives `1 failed`.

    tryfirst so it runs ahead of any other pytest_pyfunc_call implementation
    (pytest-asyncio ships one) and therefore gates async tests too.
    """
    marker = pyfuncitem.get_closest_marker("fail_missing_service")
    if marker is not None:
        pytest.fail(
            f"NEMO_REQUIRE_SERVICES=1 but {marker.kwargs['name']} is "
            f"unavailable: {marker.kwargs['reason']}",
            pytrace=False,
        )
