"""Whether a printed coverage rate means anything.

`test_research_coverage_sweep` prints a rate per tool, and a rate-limited
request comes back the same shape as a filer that does not tag the concept.
Counted together they reported "share_count covers 62.9% of filers" when the
truth was "SEC blocked us on ten of them" -- the identical basket measured
91.4% run alone minutes later, with no code between the two.

`_coverage` lives in a module gated behind the network marker, so these pin
its classification here instead: the function is pure, and a rule about what
counts as a measurement should not need a throttled SEC to demonstrate.
"""
from testing.test_research_coverage_sweep import _coverage


def test_a_throttled_request_is_not_counted_as_a_coverage_miss():
    results = {
        "AAPL": {"success": True},
        "MSFT": {"success": False, "error": "HTTPError: 429 Too Many Requests"},
        "GOOGL": {"success": False, "error": "filer does not tag this concept"},
    }
    ok, measured, throttled = _coverage(results)
    assert (ok, measured, throttled) == (1, 2, 1), (
        "the 429 must leave the denominator, and the genuine miss must stay in it")


def test_a_genuine_miss_is_still_a_miss():
    """The narrow marker set must not swallow real coverage gaps.

    Recent IPOs are in the basket, and "no filing" is the honest answer for
    one. Reading that as throttling would report full coverage of a tool that
    covers less.
    """
    results = {
        "RDDT": {"success": False, "error": "No filing found for RDDT"},
        "CART": {"success": False, "error": "No XBRL data available"},
    }
    assert _coverage(results) == (0, 2, 0)


def test_an_unresolved_symbol_is_a_miss_not_a_transport_failure():
    results = {"NKLA": {"success": False, "error": "'NKLA' did not resolve",
                        "splits": [], "dividends": []}}
    assert _coverage(results) == (0, 1, 0)


def test_a_fully_throttled_tool_has_no_measurable_rate():
    """Zero measured filers must not divide by zero or read as 0% coverage."""
    results = {"AAPL": {"success": False, "error": "429 Too Many Requests"}}
    ok, measured, throttled = _coverage(results)
    assert (ok, measured, throttled) == (0, 0, 1)
