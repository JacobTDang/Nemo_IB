"""An amendment must not displace the filing that carries the data.

`fetch_concept_series` already passes `amendments=False`, after a 10-K/A
carrying only Part III proxy information made Tesla read as tagging no
revenue, no assets and no leases at all.

The same fault remained on every path that reaches EDGAR through
`get_latest_filing` -- roughly ten tools. Altria's most recent "10-K" is a
10-K/A from 2026-05-27 with no financial statements, so `get_revenue_base`
returned "No revenue concept found": a claim that Altria does not report
revenue, which is not a thing that happens.

The symptom is silence, not an error, which is what makes it dangerous.
"""
import os
from collections import OrderedDict

import pytest

SKIP_NETWORK = os.environ.get("SKIP_NETWORK_TESTS") == "1"
network = pytest.mark.skipif(SKIP_NETWORK, reason="live EDGAR test")


def test_the_filing_walk_excludes_amendments(monkeypatch):
    """The unit-level guard: whatever else changes, the flag must be passed."""
    seen = {}

    class _Filings(list):
        def head(self, n):
            return self[:n]

    def get_filings(self, form=None, amendments=True, **kwargs):
        seen["amendments"] = amendments
        return _Filings()

    import tools.web_search_server.sec_utils as su
    # get_latest_filing caches failures as well as successes, and monkeypatch
    # cannot un-poison a cache. Without a fresh one, the None this stub
    # produces for MO is what the live Altria test below reads back -- the
    # test that proves the fix would fail because of the test that unit-checks
    # it, and only in the orders where this runs first.
    monkeypatch.setattr(su, "_filing_cache_lru", OrderedDict())
    monkeypatch.setattr(su, "Company",
                        lambda ticker: type("C", (), {"get_filings": get_filings})())
    su.get_latest_filing("MO", "10-K")
    assert seen.get("amendments") is False, (
        "get_latest_filing still accepts amendments; a 10-K/A carrying no "
        "statements will displace the 10-K that does")


@pytest.fixture(scope="module", autouse=True)
def _load_env():
    from dotenv import load_dotenv
    load_dotenv()


@network
def test_altria_reports_revenue():
    """The concrete regression. MO's latest 10-K on EDGAR is an amendment."""
    from tools.web_search_server.sec_utils import get_revenue_base
    result = get_revenue_base("MO")
    assert result["success"] is True, result.get("error")
    assert result["revenue_base"] > 15e9, (
        f"MO revenue {result['revenue_base']:,.0f} is implausibly low")


@network
def test_an_unamended_filer_is_unaffected():
    from tools.web_search_server.sec_utils import get_revenue_base
    result = get_revenue_base("MSFT")
    assert result["success"] is True
    assert result["revenue_base"] == pytest.approx(331_839_000_000.0, rel=0.001)
