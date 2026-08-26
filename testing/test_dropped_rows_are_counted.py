"""A row that failed to parse must not vanish from the total.

`get_fund_holdings` builds a 13F position list row by row and `continue`s past
any row that raises. It then reports `total_holdings: len(holdings_list)` and
`total_value_usd: sum(...)` as though the list were the filing. A 13F where a
tenth of the rows fail to parse reports a fund that is a tenth smaller, and
nothing in the response says a row was dropped.

This is the same defect as counting a page and naming it for the set: the
number is real, it just answers a question nobody asked. A caller comparing
two quarters would read the parse failure as the fund selling.
"""
import pytest
import pandas as pd


class _FakeDataObject:
    has_infotable = True

    def __init__(self, df):
        self.holdings = df


class _FakeFiling:
    filing_date = "2025-06-30"
    accession_number = "0001234567-25-000001"

    def __init__(self, df):
        self._df = df

    def data_object(self):
        return _FakeDataObject(self._df)


def _filing_with_one_unparseable_row():
    return _FakeFiling(pd.DataFrame([
        {"Issuer": "GOOD CO", "Ticker": "GC", "Cusip": "111111111",
         "Class": "COM", "SharesPrnAmount": 100, "Value": 5000, "PutCall": ""},
        {"Issuer": "BAD CO", "Ticker": "BC", "Cusip": "222222222",
         "Class": "COM", "SharesPrnAmount": "not-a-number", "Value": 9999,
         "PutCall": ""},
    ]))


def _install_filings(monkeypatch, hf, filings):
    """Stand in for the EDGAR company lookup at its real seam."""
    class _Filings(list):
        def head(self, n):
            return _Filings(self[:n])

    class _Company:
        def __init__(self, cik):
            pass

        def get_filings(self, form=None):
            return _Filings(filings)

    monkeypatch.setattr(hf, "Company", _Company)


@pytest.fixture
def holdings(monkeypatch):
    import tools.web_search_server.hf_letters as hf
    monkeypatch.setattr(hf, "_require_identity", lambda: "test@example.com",
                        raising=False)
    monkeypatch.setattr(hf, "_resolve_fund",
                        lambda x: {"name": "T", "cik": "1"}, raising=False)
    _install_filings(monkeypatch, hf, [_filing_with_one_unparseable_row()])
    return hf


def test_a_dropped_row_is_reported(holdings):
    result = holdings.get_fund_holdings("TEST")
    filing = result["filings"][0]

    assert filing.get("rows_failed") == 1, (
        "one 13F row failed to parse and the response does not say so")
    assert filing.get("rows_in_filing") == 2
    assert filing["total_holdings"] == 1


def test_a_clean_filing_is_not_flagged(holdings, monkeypatch):
    clean = _FakeFiling(pd.DataFrame([
        {"Issuer": "GOOD CO", "Ticker": "GC", "Cusip": "111111111",
         "Class": "COM", "SharesPrnAmount": 100, "Value": 5000, "PutCall": ""},
    ]))
    _install_filings(monkeypatch, holdings, [clean])

    filing = holdings.get_fund_holdings("TEST")["filings"][0]
    assert filing.get("rows_failed") == 0
    assert filing.get("complete") is True


def test_the_total_value_says_it_excludes_dropped_rows(holdings):
    """A value total that silently omits rows reads as a smaller fund."""
    filing = holdings.get_fund_holdings("TEST")["filings"][0]
    assert filing.get("complete") is False, (
        "total_value_usd omits a dropped row and claims to be the filing total")
