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


# --- the offline suite has to actually be offline ---------------------------
#
# `SKIP_NETWORK_TESTS=1 pytest testing/` is documented as "offline: no
# credentials, no network", and the gates are what make it so. An audit of one
# run found 13 tests reaching api.stlouisfed.org, data.sec.gov, finnhub.io,
# openrouter.ai and api.usaspending.gov anyway -- tests that pass while the
# upstream is healthy and fail, slowly, when it is not. They were also the
# reason a long session earned real SEC 429s.
#
# Gating them one by one fixes the thirteen that exist today. Refusing the
# connection fixes the class: a test that reaches out under SKIP_NETWORK_TESTS
# now fails immediately, naming itself and the host, instead of quietly
# depending on the weather.
#
# Set NEMO_ALLOW_OFFLINE_NETWORK=1 to audit rather than enforce.

_OFFLINE_LOCAL = {"localhost", "127.0.0.1", "::1", "0.0.0.0", ""}


class OfflineNetworkAttempt(RuntimeError):
    """A test reached the network during an offline run."""


def pytest_collection_modifyitems(session, config, items):
    if os.environ.get("SKIP_NETWORK_TESTS", "0") != "1":
        return

    # pyproject.toml declares the marker as "live-network tests skippable via
    # SKIP_NETWORK_TESTS=1". Nothing implemented the skip, so all 39 tests
    # carrying it ran anyway -- which is how the offline suite came to depend
    # on five upstreams being healthy. Declaring a marker registers its name;
    # it does not give it behaviour.
    skip_network = pytest.mark.skip(
        reason="SKIP_NETWORK_TESTS=1: marked @pytest.mark.network")
    for item in items:
        if item.get_closest_marker("network") is not None:
            item.add_marker(skip_network)

    if os.environ.get("NEMO_ALLOW_OFFLINE_NETWORK", "0") == "1":
        return

    import socket

    real_getaddrinfo = socket.getaddrinfo

    def guarded(host, port, *args, **kwargs):
        if host not in _OFFLINE_LOCAL and host is not None:
            raise OfflineNetworkAttempt(
                f"this test tried to resolve {host!r} while "
                f"SKIP_NETWORK_TESTS=1. The offline suite must not reach the "
                f"network: gate it with one of the decorators in "
                f"testing/_gates.py, or stub the fetch. Set "
                f"NEMO_ALLOW_OFFLINE_NETWORK=1 to audit instead of enforce.")
        return real_getaddrinfo(host, port, *args, **kwargs)

    socket.getaddrinfo = guarded

    # curl_cffi resolves inside libcurl, in C, and never calls the Python
    # socket module -- so the guard above does not see it. yfinance moved to
    # that transport, which is how five tests kept reaching Yahoo under
    # SKIP_NETWORK_TESTS=1 and only surfaced when a rate limit made them fail
    # instead of pass. Patching Session.request covers the module-level
    # helpers too, since those build a Session internally.
    try:
        from curl_cffi import requests as curl_requests
    except ImportError:
        return

    real_request = curl_requests.Session.request

    def guarded_request(self, method, url, *args, **kwargs):
        from urllib.parse import urlparse

        host = urlparse(str(url)).hostname or ""
        if host not in _OFFLINE_LOCAL:
            raise OfflineNetworkAttempt(
                f"this test tried to reach {host!r} over curl_cffi while "
                f"SKIP_NETWORK_TESTS=1. The offline suite must not reach the "
                f"network: gate it with one of the decorators in "
                f"testing/_gates.py, or stub the fetch. Set "
                f"NEMO_ALLOW_OFFLINE_NETWORK=1 to audit instead of enforce.")
        return real_request(self, method, url, *args, **kwargs)

    curl_requests.Session.request = guarded_request
