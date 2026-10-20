"""The SEC server reads a registrant by its CIK, and refuses a ticker it cannot map.

edgartools resolves a ticker from a parquet file bundled with the package. A
ticker missing from that file maps to nothing, and `Company(ticker)` does not
raise for it: it returns a company whose CIK is -999999999 and whose filing
list is empty. On 2026-10-19 that was 103 of the 1,810 eligible names (#136).
Over MCP, `get_schedule_13d_filings("DMC")` answered `success: true, count: 0`
while DMC's folder held 14 SC 13D filings -- an answer no caller could tell
from a quiet company.

Every one of those names is in SEC's own `company_tickers.json`. So the server
looks the ticker up there and asks EDGAR by the CIK, and a ticker that file
does not map is refused by name rather than read as a company that files
nothing. `research/registrant.py` is the jobs' half of the same fix (#130).
"""
import ast
import asyncio
import json
import pathlib
from collections import OrderedDict

import edgar
import pytest

from tools.web_search_server import foreign_issuer, registrant, sec_series
from tools.web_search_server import sec_utils as su

SENTINEL = -999999999
DMC = 1047340
AADX = 2118195

# SEC's company_tickers.json, cut to the rows these tests ask about.
SEC_TICKERS = {"DMC": DMC, "AADX": AADX, "BRK-B": 1067983, "GONE": 1234567}


class _Filings(list):
    def head(self, n):
        return _Filings(self[:n])


class _Party:
    def __init__(self, name, cik):
        self.company_information = type("CI", (), {"name": name, "cik": cik})()


class _Schedule13:
    """A Schedule 13D in DMC's folder, filed by a holder about DMC."""

    def __init__(self, accession, filed):
        self.accession_number = accession
        self.filing_date = filed
        self.file_number = "005-56187"
        self.filing_url = f"https://www.sec.gov/{accession}.htm"
        self.header = type("H", (), {
            "subject_companies": [_Party("DMC Global Inc.", "0001047340")],
            "filers": [_Party("Cove Street Capital", "0001537834")],
        })()

    def text(self):
        return "Percent of class represented by amount in row (11): 6.1%"


class _Quarterly:
    accession_number = "0002118195-26-000012"
    filing_date = "2026-08-14"
    url = "https://www.sec.gov/aadx-10q.htm"

    def xbrl(self):
        return {"tagged": True}


FOLDERS = {
    DMC: {"SC 13D": [_Schedule13("0000921895-26-000101", "2026-03-02"),
                     _Schedule13("0000921895-25-000877", "2025-06-18")]},
    AADX: {"10-Q": [_Quarterly()]},
}


@pytest.fixture
def edgar_as_observed(monkeypatch):
    """edgartools 5.8.0 as observed live on 2026-10-20, and every key it was
    asked for.

    The bundled ticker map holds none of these names, so a ticker gets the
    sentinel CIK and an empty folder, without raising. The CIK gets the real
    folder. A CIK SEC holds no submissions for comes back `not_found`, which is
    edgartools' answer to a 404 and does not raise either.
    """
    asked = []

    class _Company:
        def __init__(self, key):
            asked.append(key)
            if isinstance(key, str) and not key.isdigit():
                self.cik = SENTINEL
            else:
                self.cik = int(key)
            self.not_found = self.cik not in FOLDERS

        def get_filings(self, form=None, amendments=True, **kwargs):
            return _Filings(FOLDERS.get(self.cik, {}).get(form, []))

    monkeypatch.setattr(edgar, "Company", _Company)
    monkeypatch.setattr(registrant, "_fetch_cik_map", lambda: dict(SEC_TICKERS))
    monkeypatch.setattr(registrant, "_cik_map_cache", None)
    # An address for the identity check, without handing edgartools a test
    # identity that would outlive this test.
    monkeypatch.setenv("SEC_EMAIL", "unit-test@example.invalid")
    monkeypatch.setattr(sec_series, "_identity_set", True)
    monkeypatch.setattr(su, "_filing_cache_lru", OrderedDict())
    with foreign_issuer._index_lock:
        foreign_issuer._index_cache.clear()
    yield asked
    with foreign_issuer._index_lock:
        foreign_issuer._index_cache.clear()


# ------------------------------------------------- the names #136 measured

def test_a_name_the_bundled_map_lacks_reads_its_13d_filings(edgar_as_observed):
    """DMC is the issue's own case: `success: true, count: 0` against a folder
    that holds Schedule 13D filings."""
    result = su.get_schedule_13d_filings("DMC")

    assert result["success"] is True
    assert result["count"] == 2, (
        f"DMC's 13D filings were not read: {result.get('note')}")
    assert {row["subject_cik"] for row in result["filings"]} == {str(DMC)}
    assert edgar_as_observed == [DMC], "EDGAR was asked by ticker, not by CIK"


def test_a_name_the_bundled_map_lacks_reads_its_latest_filing(edgar_as_observed):
    """`get_latest_filing` feeds most of the server's XBRL tools, so a name
    missing here is missing from all of them."""
    found = su.get_latest_filing("AADX", "10-Q")

    assert found is not None, "AADX's 10-Q was read as absent"
    assert found["accession_number"] == _Quarterly.accession_number
    assert edgar_as_observed == [AADX]


def _parser():
    import importlib
    return importlib.import_module(
        "tools.web_search_server.8K_and_DEF14A_utils").SECFilingParser()


def _tool(name):
    from tools.web_search_server import (
        dilution, earnings_quality, peers,
    )
    return {
        "get_latest_filing (via get_revenue_base)": su.get_revenue_base,
        "get_schedule_13d_filings": su.get_schedule_13d_filings,
        "diff_10k": su.diff_10k,
        "get_company_filings_history": su.get_company_filings_history,
        "get_earnings_releases": su.get_earnings_releases,
        "extract_8k_events": lambda t: _parser().extract_8k_events(t),
        "extract_proxy_compensation":
            lambda t: _parser().extract_proxy_compensation(t),
        "extract_governance_data":
            lambda t: _parser().extract_governance_data(t),
        "get_foreign_filer_profile": foreign_issuer.get_foreign_filer_profile,
        "fetch_concept_series (via get_share_count_series)":
            dilution.get_share_count_series,
        "shared filing walk (via get_working_capital_trends)":
            earnings_quality.get_working_capital_trends,
        "get_sic_code": peers.get_sic_code,
        "get_shelf_activity": dilution.get_shelf_activity,
    }[name]


@pytest.mark.parametrize("name", [
    "get_latest_filing (via get_revenue_base)",
    "get_schedule_13d_filings",
    "diff_10k",
    "get_company_filings_history",
    "get_earnings_releases",
    "extract_8k_events",
    "extract_proxy_compensation",
    "extract_governance_data",
    "get_foreign_filer_profile",
    "fetch_concept_series (via get_share_count_series)",
    "shared filing walk (via get_working_capital_trends)",
    "get_sic_code",
    "get_shelf_activity",
])
def test_an_unmapped_ticker_is_refused_by_name(edgar_as_observed, name):
    """One entry point per place the server builds a `Company`. Each must
    answer with the refusal, naming the ticker -- not with an empty success,
    and not with a "nothing found" that reads as a fact about a filer."""
    result = _tool(name)("ZZZZQ")

    assert result["success"] is False, f"{name} answered an unmapped ticker"
    error = str(result.get("error"))
    assert "ZZZZQ" in error and "company_tickers.json" in error, (
        f"{name} did not say why it has no answer: {error}")
    assert edgar_as_observed == [], "EDGAR was asked about a ticker SEC does not map"


def test_the_latest_filing_tool_names_an_unmapped_ticker(edgar_as_observed):
    """The MCP wrapper answered every miss with "No filing found", which for
    an unmapped ticker is the absence claim #136 is about."""
    from tools.web_search_server.web_search import WebSearchServer

    content = asyncio.run(WebSearchServer.get_latest_filing(None, "ZZZZQ", "10-K"))
    result = json.loads(content[0].text)

    assert result["success"] is False
    assert "ZZZZQ" in result["error"] and "company_tickers.json" in result["error"]


# ------------------------------------------------------------ the resolver

def test_the_ticker_file_is_read_once_per_process(monkeypatch):
    fetches = []

    def fetch():
        fetches.append(1)
        return dict(SEC_TICKERS)

    monkeypatch.setattr(registrant, "_fetch_cik_map", fetch)
    monkeypatch.setattr(registrant, "_cik_map_cache", None)

    assert registrant.cik_for("DMC") == DMC
    assert registrant.cik_for("AADX") == AADX
    assert len(fetches) == 1, "SEC's ticker file was fetched on every lookup"


def test_a_failed_fetch_is_a_failure_not_an_unmapped_ticker(monkeypatch):
    """A 503 says nothing about the ticker. Reported as one, it would refuse
    every name as unmapped; cached, it would do so until a restart."""
    responses = [ConnectionError("SEC returned 503"), dict(SEC_TICKERS)]

    def fetch():
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(registrant, "_fetch_cik_map", fetch)
    monkeypatch.setattr(registrant, "_cik_map_cache", None)

    with pytest.raises(ConnectionError):
        registrant.cik_for("DMC")
    assert registrant.cik_for("DMC") == DMC, "the failed fetch was cached"


def test_a_share_class_resolves_in_either_spelling(monkeypatch):
    """SEC's file writes BRK-B. edgartools accepted BRK.B too, so callers send
    it, and refusing it would turn a spelling into a missing registrant."""
    monkeypatch.setattr(registrant, "_cik_map", lambda: dict(SEC_TICKERS))

    assert registrant.cik_for("BRK.B") == 1067983
    assert registrant.cik_for("brk-b") == 1067983


def test_a_cik_is_read_as_a_cik(edgar_as_observed, monkeypatch):
    """edgartools read an all-digit key as a CIK, so a caller that already had
    one could pass it. It still goes straight to EDGAR."""
    def no_map():
        raise AssertionError("the ticker file was consulted for a CIK")

    monkeypatch.setattr(registrant, "_cik_map", no_map)

    assert registrant.company("0001047340").cik == DMC
    assert edgar_as_observed == [DMC]


def test_a_cik_sec_has_no_submissions_for_is_refused(edgar_as_observed):
    """The CIK route has its own empty answer: edgartools turns SEC's 404 into
    a company flagged `not_found` with no filings."""
    with pytest.raises(LookupError, match="GONE"):
        registrant.company("GONE")


def test_no_tool_asks_edgartools_for_a_ticker():
    """Every `Company(...)` in the SEC server is handed a CIK.

    A `Company(ticker)` left anywhere in the package is a tool that reads the
    103 names as having filed nothing. The scan reads the source rather than
    trusting the tests above to reach every call site.
    """
    package = pathlib.Path(registrant.__file__).parent
    offenders, seen = [], []
    for path in sorted(package.glob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
            if name != "Company":
                continue
            argument = ast.unparse(node.args[0]) if node.args else ""
            seen.append(path.name)
            if "cik" not in argument.lower():
                offenders.append(f"{path.name}:{node.lineno} Company({argument})")

    assert "registrant.py" in seen, "the scan found no Company call at all"
    assert not offenders, (
        "edgartools is asked for a ticker here, and answers an unmapped one "
        "with an empty folder; use registrant.company(ticker): "
        + "; ".join(offenders))
