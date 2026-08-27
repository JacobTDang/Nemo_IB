"""A failed EDGAR lookup must not become "this filer has no annual report".

`_fetch_annual_filing_index` asked EDGAR for each of 10-K, 20-F and 40-F and
swallowed every exception with the comment "a form this filer never used". But
`get_filings` returns an EMPTY LIST for a form a filer never used -- it does
not raise. An exception therefore always meant something broke, and the one
thing it was documented as meaning was the one thing it could not mean.

Under an SEC rate limit all three lookups raised, the index came back empty,
and `get_annual_revenue("NVDA")` answered:

    NVDA: no 10-K, 20-F or 40-F filing found on EDGAR, so there is no annual
    report to read revenue from.

NVDA files a 10-K every year. The response is a confident statement about the
company, sourced entirely from our own outage.

The index is also memoised, so one throttled call poisoned every later call
for that ticker in the same process. A wrong answer that is cached stops
looking like a transient failure and starts looking like a fact.
"""
import pytest

from tools.web_search_server import foreign_issuer as fi


@pytest.fixture(autouse=True)
def _clear_index_cache(monkeypatch):
    monkeypatch.setenv("SEC_EMAIL", "test@example.invalid")
    with fi._index_lock:
        fi._index_cache.clear()
    yield
    with fi._index_lock:
        fi._index_cache.clear()


class _Throttled:
    def __init__(self, cik=None):
        pass

    def get_filings(self, form=None, amendments=True):
        raise RuntimeError("Too Many Requests")


def _install(monkeypatch, company):
    import edgar
    monkeypatch.setattr(edgar, "Company", company)


def test_a_throttled_index_is_not_reported_as_no_filings(monkeypatch):
    _install(monkeypatch, _Throttled)
    result = fi.get_annual_revenue("NVDA")

    assert result["success"] is False
    message = (result.get("error") or "").lower()
    assert "no 10-k" not in message, (
        f"an outage was reported as NVDA having no annual report: {message}")
    assert "too many requests" in message or "failed" in message, (
        f"the response does not say what actually went wrong: {message}")


def test_a_failed_index_is_not_cached(monkeypatch):
    """A cached outage stops looking transient and starts looking like a fact."""
    _install(monkeypatch, _Throttled)
    with pytest.raises(Exception):
        fi._annual_filing_index("NVDA")

    assert "NVDA" not in fi._index_cache, (
        "a failed lookup was memoised and will be replayed as an answer")


def test_a_filer_that_genuinely_uses_one_form_still_resolves(monkeypatch):
    """The fix must not turn an ordinary absence into an error."""
    class _Filing:
        filing_date = "2025-02-26"

    class _OnlyTenK:
        def __init__(self, cik=None):
            pass

        def get_filings(self, form=None, amendments=True):
            class _L(list):
                def head(self, n):
                    return _L(self[:n])
            return _L([_Filing()] if form == "10-K" else [])

    _install(monkeypatch, _OnlyTenK)
    form, index = fi._latest_annual_form("NVDA")
    assert form == "10-K"
    assert set(index) == {"10-K"}
