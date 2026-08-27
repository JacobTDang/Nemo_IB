"""The offline suite must actually be offline.

conftest guards `socket.getaddrinfo`, which covers anything built on Python
sockets -- requests, aiohttp, urllib. yfinance is not: it moved to curl_cffi,
which resolves inside libcurl in C and never touches the Python socket module.
So five tests kept reaching Yahoo under SKIP_NETWORK_TESTS=1, passed while the
upstream was healthy, and failed with YFRateLimitError when it was not. That is
the exact failure the guard was written to make impossible, arriving through a
door it did not cover.

Gating those five would fix the five. Guarding the transport fixes the class,
including the next library that swaps its HTTP stack.
"""
import os

import pytest

OFFLINE = os.environ.get("SKIP_NETWORK_TESTS", "0") == "1"
AUDIT_ONLY = os.environ.get("NEMO_ALLOW_OFFLINE_NETWORK", "0") == "1"

pytestmark = pytest.mark.skipif(
    not OFFLINE or AUDIT_ONLY,
    reason="only meaningful while the offline guard is enforcing")


# conftest is imported under the rootdir package name, so `from
# testing.conftest import ...` builds a second module object with a second,
# unequal class. Matching the base class and the message is identity-proof.

def test_curl_cffi_is_refused_like_every_other_transport():
    curl_cffi = pytest.importorskip("curl_cffi")

    with pytest.raises(RuntimeError, match="finance.yahoo.com"):
        curl_cffi.requests.get("https://finance.yahoo.com/quote/AAPL")


def test_a_session_object_is_refused_too():
    """yfinance holds a long-lived Session rather than calling the module."""
    curl_cffi = pytest.importorskip("curl_cffi")

    with pytest.raises(RuntimeError, match="query2.finance.yahoo.com"):
        curl_cffi.requests.Session().get("https://query2.finance.yahoo.com/v1/t")


def test_the_socket_guard_still_holds():
    """The original door, still shut."""
    import socket

    with pytest.raises(RuntimeError, match="data.sec.gov"):
        socket.getaddrinfo("data.sec.gov", 443)
