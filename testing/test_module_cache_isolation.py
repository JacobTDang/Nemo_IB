"""A module-level cache must not carry one test module's state into the next.

`sec_utils` keeps an LRU of SEC filings keyed by (ticker, form_type), and
`get_latest_filing` returns a hit before it constructs `Company`. Any test that
patches `sec_utils.Company` to observe a failure path therefore never reaches
its own mock if an earlier module already fetched that ticker -- the mock's
side effect does not fire, and the test reads the cached filing instead.

That is what happened to TestErrorHandling::test_network_error_handling and
test_api_timeout_handling: 9/9 green when the class runs alone, two failures
in a full run, with nothing between them but collection order.

conftest already isolates databases per module for this exact reason. This
pins the same property for the in-process cache: state, not just storage.

Two generated modules in a subprocess, because a module-scoped fixture cannot
be observed from inside the module it scopes.
"""
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

_FILLS_CACHE = '''
from tools.web_search_server import sec_utils


def test_fills_the_cache():
    """No network: put a sentinel straight into the LRU."""
    sec_utils._filing_cache_lru[("AAPL", "10-K")] = {"sentinel": True}
    assert ("AAPL", "10-K") in sec_utils._filing_cache_lru
'''

_EXPECTS_EMPTY = '''
from tools.web_search_server import sec_utils


def test_does_not_see_the_previous_modules_cache():
    assert ("AAPL", "10-K") not in sec_utils._filing_cache_lru, (
        "the previous test module's cached filing is still here, so a test "
        "patching sec_utils.Company would return this instead of reaching "
        "its own mock")
'''


def test_filing_cache_does_not_leak_between_modules(tmp_path):
    first = ROOT / "testing" / "test_zz_cache_a_generated.py"
    second = ROOT / "testing" / "test_zz_cache_b_generated.py"
    first.write_text(_FILLS_CACHE)
    second.write_text(_EXPECTS_EMPTY)
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(first), str(second),
             "-q", "-p", "no:randomly"],
            cwd=ROOT, capture_output=True, text=True, timeout=300)
    finally:
        first.unlink(missing_ok=True)
        second.unlink(missing_ok=True)

    assert result.returncode == 0, (
        "sec_utils' filing cache survived into the next test module:\n"
        + result.stdout[-2000:])
