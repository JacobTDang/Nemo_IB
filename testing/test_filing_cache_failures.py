"""A transient SEC failure must not become a permanent wrong answer.

`get_latest_filing` caches what it returns, deliberately, so that N tools
asking for the same filing cause one download. It also cached failures --
and it derived those failures from a bare `except Exception`, so a network
blip, a rate limit or an SEC 503 was stored under the ticker and returned
to every later caller for the lifetime of the process.

That matters here more than it would in a script. This runs as a
long-lived MCP server: one blip while answering a question about Altria
leaves `get_revenue_base("MO")` answering "no revenue concept found" until
someone restarts the container, with nothing logged to say why.

A company that genuinely files no 10-K is a different thing -- a stable
fact, worth caching. The two must not share a code path.
"""
from collections import OrderedDict

import pytest

import tools.web_search_server.sec_utils as su


class _Filing:
    filing_date = "2026-02-25"
    accession_number = "0000764180-26-000018"
    url = "https://example.invalid/mo.htm"

    def xbrl(self):
        return {"ok": True}


@pytest.fixture(autouse=True)
def _isolated_cache(monkeypatch):
    """Never let one test's stub answer another test's question."""
    monkeypatch.setattr(su, "_filing_cache_lru", OrderedDict())
    monkeypatch.setattr(su, "_require_identity", lambda: None)


def _company_returning(*responses):
    """A Company whose get_filings walks `responses`, raising any exception."""
    calls = {"n": 0}

    def get_filings(self, form=None, amendments=True, **kwargs):
        i = min(calls["n"], len(responses) - 1)
        calls["n"] += 1
        response = responses[i]
        if isinstance(response, Exception):
            raise response
        return response

    return calls, lambda ticker: type("C", (), {"get_filings": get_filings})()


def test_a_transient_failure_is_not_cached(monkeypatch):
    """The blip, then the recovery. The second call must reach SEC again."""
    calls, company = _company_returning(
        ConnectionError("SEC returned 503"), [_Filing()])
    monkeypatch.setattr(su, "Company", company)

    first = su.get_latest_filing("MO", "10-K")
    assert first is None, "a failed fetch still reports no filing"

    second = su.get_latest_filing("MO", "10-K")
    assert second is not None, (
        "the 503 was cached: every later caller is told Altria files no "
        "10-K, for the life of the process, with no way to recover")
    assert second["accession_number"] == _Filing.accession_number
    assert calls["n"] == 2, "the retry never reached SEC"


def test_a_genuine_absence_is_still_cached(monkeypatch):
    """The retry-storm protection this cache exists for must survive."""
    calls, company = _company_returning([])
    monkeypatch.setattr(su, "Company", company)

    assert su.get_latest_filing("NOFILINGS", "10-K") is None
    assert su.get_latest_filing("NOFILINGS", "10-K") is None
    assert calls["n"] == 1, (
        "a company that genuinely files nothing is a stable fact; asking "
        "SEC again on every call is the retry storm the cache prevents")


def test_a_success_is_still_cached(monkeypatch):
    calls, company = _company_returning([_Filing()])
    monkeypatch.setattr(su, "Company", company)

    assert su.get_latest_filing("MO", "10-K") is not None
    assert su.get_latest_filing("MO", "10-K") is not None
    assert calls["n"] == 1, "the single-flight cache stopped working"


def test_the_failure_is_logged(monkeypatch, caplog):
    """Swallowing it silently is what made this invisible for so long."""
    _, company = _company_returning(ConnectionError("SEC returned 503"))
    monkeypatch.setattr(su, "Company", company)

    with caplog.at_level("WARNING", logger=su.__name__):
        su.get_latest_filing("MO", "10-K")

    assert any("MO" in r.message and "503" in r.message for r in caplog.records), (
        f"the reason the filing is missing was swallowed: {caplog.records}")


class _FilingWithBrokenXBRL(_Filing):
    """A filing whose XBRL fetch fails the first time and then recovers."""

    def __init__(self):
        self.attempts = 0

    def xbrl(self):
        self.attempts += 1
        if self.attempts == 1:
            raise ConnectionError("SEC returned 503")
        return {"ok": True}


def test_a_transient_xbrl_failure_is_not_cached(monkeypatch):
    """Worse than a failed fetch: the filing looks present but tags nothing."""
    filing = _FilingWithBrokenXBRL()
    _, company = _company_returning([filing])
    monkeypatch.setattr(su, "Company", company)

    first = su.get_latest_filing("MO", "10-K")
    assert first is not None and first["xbrl_data"] is None

    second = su.get_latest_filing("MO", "10-K")
    assert second["xbrl_data"] == {"ok": True}, (
        "the filing was cached with xbrl_data=None, so every tool reading "
        "it concludes Altria tags no financial data at all")
    assert filing.attempts == 2
