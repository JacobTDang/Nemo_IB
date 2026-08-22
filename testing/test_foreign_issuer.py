"""Foreign private issuers.

The SEC layer had zero coverage of 20-F, so every ADR -- TSM, ASML, BABA, SAP,
NVO -- came back empty. Empty reads as a clean bill of health: measured before
this module existed, `get_debt_maturity_schedule("TSM")` answered "TSM does not
tag long-term debt maturities in its 10-K" and `get_geographic_revenue("TSM")`
answered "TSM does not disaggregate revenue by geography in its 10-K". TSMC has
no 10-K at all, and its 20-F disaggregates revenue by geography across four
regions. Both answers were about a filing that does not exist.

Three things were probed live against EDGAR before any of this was written,
and two of them contradict the obvious assumption:

1. **Taxonomy is a property of the filer, not the form.** 20-F filers split
   both ways: TSM, SAP and NVO tag `ifrs-full`, while ASML and BABA tag
   `us-gaap` in the same form. Choosing concepts by form would miss two of
   five.
2. **6-K carries no XBRL.** Not sparse -- absent. edgartools reports "No XBRL
   attachments found" for every recent 6-K from TSM, ASML and BABA. There is
   no interim tagged-data path for a foreign issuer at all, so a quarterly
   tool must say that rather than return nothing.
3. Currency is never USD by default. TSM reports TWD, SAP and ASML EUR, NVO
   DKK, BABA CNY.
"""
import os

import pytest

from tools.web_search_server import foreign_issuer as fi
from tools.web_search_server.sec_series import ConceptFact, FilingPoint, NotCovered

SKIP_NETWORK = os.environ.get("SKIP_NETWORK_TESTS") == "1"


def network(func):
    func = pytest.mark.network(func)
    return pytest.mark.skipif(SKIP_NETWORK, reason="live EDGAR test")(func)


@pytest.fixture(autouse=True)
def _clear_caches():
    fi._reset_caches()
    yield
    fi._reset_caches()


def _index(monkeypatch, mapping):
    """Stub the cheap forms lookup: {form: latest filing date}."""
    monkeypatch.setattr(fi, "_annual_filing_index", lambda ticker: dict(mapping))


def _facts(monkeypatch, **by_ticker):
    monkeypatch.setattr(fi, "_annual_facts",
                        lambda ticker, form: by_ticker.get(ticker.upper()))


def _frame(rows):
    import pandas as pd
    return pd.DataFrame(rows)


TSM_FRAME = _frame([
    {"concept": "ifrs-full:RevenueFromContractsWithCustomers",
     "value": "3809054300000", "unit_ref": "twd"},
    {"concept": "ifrs-full:Revenue", "value": "352271200000", "unit_ref": "twd"},
    {"concept": "ifrs-full:Assets", "value": "1", "unit_ref": "twd"},
    {"concept": "ifrs-full:RevenueFromContractsWithCustomers",
     "value": "121423500000", "unit_ref": "usd"},
    {"concept": "tsm:SomethingCompanySpecific", "value": "1", "unit_ref": "twd"},
    {"concept": "dei:DocumentAccountingStandard",
     "value": "International Financial Reporting Standards", "unit_ref": None},
    {"concept": "dei:EntityIncorporationStateCountryCode", "value": "F5",
     "unit_ref": None},
    {"concept": "dei:EntityAddressCountry", "value": "TW", "unit_ref": None},
    {"concept": "dei:EntityRegistrantName",
     "value": "Taiwan Semiconductor Manufacturing Company Limited", "unit_ref": None},
])

BABA_FRAME = _frame([
    {"concept": "us-gaap:Revenues", "value": "1023670000000", "unit_ref": "U_CNY"},
    {"concept": "us-gaap:Assets", "value": "1", "unit_ref": "U_CNY"},
    {"concept": "us-gaap:Revenues", "value": "148401000000", "unit_ref": "U_USD"},
    {"concept": "dei:DocumentAccountingStandard", "value": "U.S. GAAP",
     "unit_ref": None},
    {"concept": "dei:EntityAddressCountry", "value": "HK", "unit_ref": None},
])

AAPL_FRAME = _frame([
    {"concept": "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
     "value": "416161000000", "unit_ref": "usd"},
    {"concept": "us-gaap:Assets", "value": "1", "unit_ref": "usd"},
    {"concept": "dei:EntityAddressCountry", "value": "US", "unit_ref": None},
])


# ------------------------------------------------------------------- profile

def test_a_20f_filer_is_identified_as_a_foreign_private_issuer(monkeypatch):
    _index(monkeypatch, {"20-F": "2026-04-16"})
    _facts(monkeypatch, TSM=("2026-04-16", "0001628280-26-025362", TSM_FRAME))
    profile = fi.get_foreign_filer_profile("TSM")
    assert profile["success"] is True
    assert profile["is_foreign_private_issuer"] is True
    assert profile["annual_form"] == "20-F"
    assert profile["annual_filing_date"] == "2026-04-16"


def test_a_10k_filer_is_not_a_foreign_private_issuer(monkeypatch):
    """The negative answer has to be as clear as the positive one, or a caller
    cannot use this to route."""
    _index(monkeypatch, {"10-K": "2026-10-31"})
    _facts(monkeypatch, AAPL=("2026-10-31", "acc", AAPL_FRAME))
    profile = fi.get_foreign_filer_profile("AAPL")
    assert profile["success"] is True
    assert profile["is_foreign_private_issuer"] is False
    assert profile["annual_form"] == "10-K"
    assert profile["interim_form"] == "10-Q"
    assert profile["interim_xbrl"] is True


def test_a_40f_filer_is_a_foreign_private_issuer(monkeypatch):
    """Canadian MJDS filers are foreign issuers too, on a different form."""
    _index(monkeypatch, {"40-F": "2026-03-06"})
    _facts(monkeypatch, BCE=("2026-03-06", "acc", _frame([
        {"concept": "ifrs-full:Revenue", "value": "1", "unit_ref": "cad"},
        {"concept": "ifrs-full:Assets", "value": "1", "unit_ref": "cad"},
    ])))
    profile = fi.get_foreign_filer_profile("BCE")
    assert profile["is_foreign_private_issuer"] is True
    assert profile["annual_form"] == "40-F"
    assert profile["reporting_currency"] == "CAD"


def test_the_most_recent_annual_form_decides_not_the_history(monkeypatch):
    """SHOP filed 20-F in 2016 and 40-F through 2024, then graduated to 10-K.
    Reading "has ever filed 40-F" as "is a foreign issuer" gets it wrong, and
    so does reading "has a 10-K" as "always had one"."""
    _index(monkeypatch, {"10-K": "2026-02-11", "40-F": "2024-02-13",
                         "20-F": "2016-02-17"})
    _facts(monkeypatch, SHOP=("2026-02-11", "acc", AAPL_FRAME))
    profile = fi.get_foreign_filer_profile("SHOP")
    assert profile["is_foreign_private_issuer"] is False
    assert profile["annual_form"] == "10-K"
    assert profile["former_annual_forms"] == {"40-F": "2024-02-13",
                                              "20-F": "2016-02-17"}
    assert "40-F" in profile["note"]


def test_ifrs_taxonomy_is_detected(monkeypatch):
    _index(monkeypatch, {"20-F": "2026-04-16"})
    _facts(monkeypatch, TSM=("2026-04-16", "acc", TSM_FRAME))
    profile = fi.get_foreign_filer_profile("TSM")
    assert profile["taxonomy"] == "ifrs-full"
    assert profile["accounting_standard"] == "International Financial Reporting Standards"
    assert profile["accounting_standard_source"] == "dei:DocumentAccountingStandard"


def test_a_20f_filer_reporting_under_us_gaap_is_reported_as_such(monkeypatch):
    """The finding that breaks the obvious design. ASML and BABA file 20-F and
    tag us-gaap; picking concepts from the form alone would miss them."""
    _index(monkeypatch, {"20-F": "2026-05-20"})
    _facts(monkeypatch, BABA=("2026-05-20", "acc", BABA_FRAME))
    profile = fi.get_foreign_filer_profile("BABA")
    assert profile["is_foreign_private_issuer"] is True
    assert profile["taxonomy"] == "us-gaap"
    assert profile["accounting_standard"] == "U.S. GAAP"


def test_taxonomy_falls_back_to_counting_concepts_when_untagged(monkeypatch):
    """Domestic 10-Ks never tag dei:DocumentAccountingStandard, and neither do
    40-F filers. The prefix histogram still answers."""
    _index(monkeypatch, {"10-K": "2026-10-31"})
    _facts(monkeypatch, AAPL=("2026-10-31", "acc", AAPL_FRAME))
    profile = fi.get_foreign_filer_profile("AAPL")
    assert profile["taxonomy"] == "us-gaap"
    assert profile["accounting_standard_source"] == "concept prefixes"


def test_reporting_currency_and_the_convenience_translation(monkeypatch):
    _index(monkeypatch, {"20-F": "2026-04-16"})
    _facts(monkeypatch, TSM=("2026-04-16", "acc", TSM_FRAME))
    profile = fi.get_foreign_filer_profile("TSM")
    assert profile["reporting_currency"] == "TWD"
    assert profile["usd_convenience_translation"] is True
    assert profile["currencies_present"]["USD"] == 1


def test_a_us_filer_has_no_convenience_translation(monkeypatch):
    _index(monkeypatch, {"10-K": "2026-10-31"})
    _facts(monkeypatch, AAPL=("2026-10-31", "acc", AAPL_FRAME))
    profile = fi.get_foreign_filer_profile("AAPL")
    assert profile["reporting_currency"] == "USD"
    assert profile["usd_convenience_translation"] is False


def test_a_foreign_issuer_is_told_its_interim_reports_are_untagged(monkeypatch):
    """The most useful single fact about an FPI for a tool caller: there is no
    quarterly XBRL to read, at all."""
    _index(monkeypatch, {"20-F": "2026-04-16"})
    _facts(monkeypatch, TSM=("2026-04-16", "acc", TSM_FRAME))
    profile = fi.get_foreign_filer_profile("TSM")
    assert profile["interim_form"] == "6-K"
    assert profile["interim_xbrl"] is False


def test_a_ticker_with_no_annual_filing_at_all_fails_loudly(monkeypatch):
    _index(monkeypatch, {})
    _facts(monkeypatch)
    profile = fi.get_foreign_filer_profile("NOTATICKER")
    assert profile["success"] is False
    assert profile["is_foreign_private_issuer"] is None
    assert "no 10-K, 20-F or 40-F" in profile["error"]


def test_an_annual_filing_with_no_parseable_xbrl_still_reports_the_form(monkeypatch):
    """CNI's 40-F carries 41 cover-page facts and no financial statements.
    The form and the foreign-issuer status are still true and still useful."""
    _index(monkeypatch, {"40-F": "2026-02-04"})
    _facts(monkeypatch, CNI=None)
    profile = fi.get_foreign_filer_profile("CNI")
    assert profile["success"] is True
    assert profile["is_foreign_private_issuer"] is True
    assert profile["annual_form"] == "40-F"
    assert profile["taxonomy"] is None
    assert profile["reporting_currency"] is None
    assert "XBRL" in profile["note"]


# ------------------------------------------------------------- mismatch guard

def test_asking_a_10k_tool_about_a_20f_filer_says_so(monkeypatch):
    _index(monkeypatch, {"20-F": "2026-04-16"})
    note = fi.form_mismatch_note("TSM", "10-K")
    assert note is not None
    assert "20-F" in note and "10-K" in note
    assert "foreign private issuer" in note.lower()


def test_asking_a_10q_tool_about_a_foreign_issuer_names_the_6k_gap(monkeypatch):
    """6-K exhibits carry no XBRL, so this is a permanent absence rather than
    a wrong-form problem the caller can fix by passing another argument."""
    _index(monkeypatch, {"20-F": "2026-04-16"})
    note = fi.form_mismatch_note("TSM", "10-Q")
    assert note is not None
    assert "6-K" in note
    assert "no XBRL" in note or "not tagged" in note


def test_no_note_when_the_form_matches(monkeypatch):
    """The guard must stay silent for the 99% case or it becomes noise."""
    _index(monkeypatch, {"10-K": "2026-10-31"})
    assert fi.form_mismatch_note("AAPL", "10-K") is None
    assert fi.form_mismatch_note("AAPL", "10-Q") is None


def test_no_note_when_a_foreign_issuer_is_asked_about_its_own_form(monkeypatch):
    _index(monkeypatch, {"20-F": "2026-04-16"})
    assert fi.form_mismatch_note("TSM", "20-F") is None


def test_asking_for_20f_from_a_domestic_filer_says_so(monkeypatch):
    _index(monkeypatch, {"10-K": "2026-10-31"})
    note = fi.form_mismatch_note("AAPL", "10-K")
    assert note is None
    note = fi.form_mismatch_note("AAPL", "20-F")
    assert note is not None and "10-K" in note


def test_the_guard_never_raises_when_edgar_is_unreachable(monkeypatch):
    """A guard that throws turns a partial answer into no answer. It is an
    annotation on an error path, so failing to annotate must be survivable."""
    def boom(ticker):
        raise RuntimeError("EDGAR timeout")
    monkeypatch.setattr(fi, "_annual_filing_index", boom)
    assert fi.form_mismatch_note("TSM", "10-K") is None


def test_the_forms_index_is_fetched_once_per_ticker(monkeypatch):
    """form_mismatch_note is called from every failure path in the SEC layer.
    One EDGAR round trip per tool call would be a real cost."""
    calls = []
    monkeypatch.setattr(fi, "_fetch_annual_filing_index",
                        lambda ticker: calls.append(ticker) or {"20-F": "2026-04-16"})
    fi.form_mismatch_note("TSM", "10-K")
    fi.form_mismatch_note("TSM", "10-Q")
    fi.form_mismatch_note("TSM", "20-F")
    assert calls == ["TSM"]


# ------------------------------------------------------------------- revenue

def _point(value, currency, period="duration_2025-01-01_2025-12-31",
           filing_date="2026-04-16", form="20-F", extra=()):
    facts = [ConceptFact(value, period, {}, "c-1", "x", currency)]
    facts.extend(extra)
    return FilingPoint(filing_date, form, "acc", facts=facts)


def test_revenue_reports_the_currency_it_is_denominated_in(monkeypatch):
    """3,809,054,300,000 without "TWD" attached reads as $3.8 trillion."""
    _index(monkeypatch, {"20-F": "2026-04-16"})
    monkeypatch.setattr(fi, "fetch_concept_series", lambda t, c, **k: (
        [_point(3_809_054_300_000.0, "twd")]
        if c == "ifrs-full:RevenueFromContractsWithCustomers"
        else (_ for _ in ()).throw(NotCovered(c))))
    result = fi.get_annual_revenue("TSM")
    assert result["success"] is True
    assert result["currency"] == "TWD"
    assert result["latest_revenue"] == 3_809_054_300_000.0
    assert result["concept_used"] == "ifrs-full:RevenueFromContractsWithCustomers"
    assert result["form"] == "20-F"


def test_a_concept_tagged_only_with_dimensions_falls_through(monkeypatch):
    """TSM tags ifrs-full:Revenue, but every one of its six facts is
    dimensioned -- it is a geography breakdown, not the total. Treating "the
    concept exists" as "the concept answers" reports NT$352bn of revenue
    against a real NT$3,809bn."""
    _index(monkeypatch, {"20-F": "2026-04-16"})

    def fake(ticker, concept, **kwargs):
        if concept == "ifrs-full:Revenue":
            return [FilingPoint("2026-04-16", "20-F", "acc", facts=[
                ConceptFact(352_271_200_000.0, "duration_2025-01-01_2025-12-31",
                            {"ifrs-full:GeographicalAreasAxis": "country:TW"},
                            "c-1139", "ifrs-full:Revenue", "twd")])]
        if concept == "ifrs-full:RevenueFromContractsWithCustomers":
            return [_point(3_809_054_300_000.0, "twd")]
        raise NotCovered(concept)

    monkeypatch.setattr(fi, "fetch_concept_series", fake)
    result = fi.get_annual_revenue("TSM")
    assert result["latest_revenue"] == 3_809_054_300_000.0


def test_a_concept_the_newest_filing_dropped_does_not_answer(monkeypatch):
    """The regression this cost a live run to find.

    TSM tagged ifrs-full:Revenue undimensioned in its 2024 and 2025 20-Fs and
    stopped in the 2026 one, where all six facts are geographies. The chain
    saw rows from the two older filings, accepted the concept, and reported
    FY2024's NT$2,894bn as the latest revenue -- one year stale, 24% low,
    right currency, entirely plausible. A chain member only answers if it
    answers in the most recent annual filing.
    """
    _index(monkeypatch, {"20-F": "2026-04-16"})

    def fake(ticker, concept, **kwargs):
        if concept == "ifrs-full:Revenue":
            return [
                FilingPoint("2026-04-16", "20-F", "acc", facts=[
                    ConceptFact(352_271_200_000.0, "duration_2025-01-01_2025-12-31",
                                {"ifrs-full:GeographicalAreasAxis": "country:TW"},
                                "c-1139", "ifrs-full:Revenue", "twd")]),
                _point(2_894_307_700_000.0, "twd",
                       period="duration_2024-01-01_2024-12-31",
                       filing_date="2025-04-17"),
            ]
        if concept == "ifrs-full:RevenueFromContractsWithCustomers":
            return [_point(3_809_054_300_000.0, "twd")]
        raise NotCovered(concept)

    monkeypatch.setattr(fi, "fetch_concept_series", fake)
    result = fi.get_annual_revenue("TSM")
    assert result["latest_revenue"] == 3_809_054_300_000.0
    assert result["concept_used"] == "ifrs-full:RevenueFromContractsWithCustomers"


def test_the_series_is_ordered_newest_period_first(monkeypatch):
    _index(monkeypatch, {"20-F": "2026-04-16"})
    monkeypatch.setattr(fi, "fetch_concept_series", lambda t, c, **k: (
        [_point(2_894_307_700_000.0, "twd",
                period="duration_2024-01-01_2024-12-31", filing_date="2026-04-16"),
         _point(3_809_054_300_000.0, "twd", filing_date="2026-04-16")]
        if c == "ifrs-full:Revenue" else (_ for _ in ()).throw(NotCovered(c))))
    result = fi.get_annual_revenue("TSM")
    assert [row["value"] for row in result["series"]] == [
        3_809_054_300_000.0, 2_894_307_700_000.0]
    assert result["latest_revenue"] == 3_809_054_300_000.0


def test_the_usd_convenience_translation_is_reported_separately(monkeypatch):
    """Useful and dangerous in equal measure: it is the filer's own rate on
    their own date, not a live conversion."""
    _index(monkeypatch, {"20-F": "2026-04-16"})
    monkeypatch.setattr(fi, "fetch_concept_series", lambda t, c, **k: (
        [_point(3_809_054_300_000.0, "twd", extra=[
            ConceptFact(121_423_500_000.0, "duration_2025-01-01_2025-12-31",
                        {}, "c-1", "x", "usd")])]
        if c == "ifrs-full:RevenueFromContractsWithCustomers"
        else (_ for _ in ()).throw(NotCovered(c))))
    result = fi.get_annual_revenue("TSM")
    assert result["latest_revenue"] == 3_809_054_300_000.0
    assert result["currency"] == "TWD"
    assert result["latest_revenue_usd"] == 121_423_500_000.0
    assert result["usd_is_filer_translation"] is True


def test_a_us_gaap_20f_filer_is_served_by_the_same_call(monkeypatch):
    """ASML and BABA file 20-F under us-gaap. Both chains get tried."""
    _index(monkeypatch, {"20-F": "2026-02-25"})
    monkeypatch.setattr(fi, "fetch_concept_series", lambda t, c, **k: (
        [_point(32_667_300_000.0, "eur", filing_date="2026-02-25")]
        if c == "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax"
        else (_ for _ in ()).throw(NotCovered(c))))
    result = fi.get_annual_revenue("ASML")
    assert result["success"] is True
    assert result["currency"] == "EUR"
    assert result["taxonomy_used"] == "us-gaap"


def test_a_domestic_filer_still_works_on_the_10k_path(monkeypatch):
    _index(monkeypatch, {"10-K": "2026-10-31"})
    monkeypatch.setattr(fi, "fetch_concept_series", lambda t, c, **k: (
        [_point(416_161_000_000.0, "usd", filing_date="2026-10-31", form="10-K")]
        if c == "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax"
        else (_ for _ in ()).throw(NotCovered(c))))
    result = fi.get_annual_revenue("AAPL")
    assert result["success"] is True
    assert result["form"] == "10-K"
    assert result["currency"] == "USD"
    assert result["latest_revenue_usd"] == 416_161_000_000.0
    assert result["usd_is_filer_translation"] is False


def test_no_revenue_concept_at_all_is_an_explicit_failure(monkeypatch):
    _index(monkeypatch, {"20-F": "2026-04-16"})
    monkeypatch.setattr(fi, "fetch_concept_series",
                        lambda t, c, **k: (_ for _ in ()).throw(NotCovered(c)))
    result = fi.get_annual_revenue("WEIRD")
    assert result["success"] is False
    assert result["latest_revenue"] is None
    assert "20-F" in result["error"]
    assert result["concepts_tried"]


def test_revenue_for_a_ticker_with_no_annual_filing_says_that(monkeypatch):
    _index(monkeypatch, {})
    result = fi.get_annual_revenue("NOTATICKER")
    assert result["success"] is False
    assert "no 10-K, 20-F or 40-F" in result["error"]


# ------------------------------------------------------------- live golden set

@pytest.fixture(scope="module", autouse=True)
def _load_env():
    from dotenv import load_dotenv
    load_dotenv()


@network
def test_tsm_is_an_ifrs_20f_filer_reporting_in_twd():
    profile = fi.get_foreign_filer_profile("TSM")
    assert profile["is_foreign_private_issuer"] is True
    assert profile["annual_form"] == "20-F"
    assert profile["taxonomy"] == "ifrs-full"
    assert profile["reporting_currency"] == "TWD"
    assert profile["interim_xbrl"] is False


@network
def test_asml_files_20f_under_us_gaap():
    """The empirical contradiction of the obvious rule, pinned live."""
    profile = fi.get_foreign_filer_profile("ASML")
    assert profile["is_foreign_private_issuer"] is True
    assert profile["annual_form"] == "20-F"
    assert profile["taxonomy"] == "us-gaap"
    assert profile["accounting_standard"] == "U.S. GAAP"
    assert profile["reporting_currency"] == "EUR"


@network
def test_baba_files_20f_under_us_gaap_in_renminbi():
    profile = fi.get_foreign_filer_profile("BABA")
    assert profile["taxonomy"] == "us-gaap"
    assert profile["reporting_currency"] == "CNY"
    assert profile["usd_convenience_translation"] is True


@network
def test_aapl_is_not_a_foreign_issuer_and_nothing_changed_for_it():
    profile = fi.get_foreign_filer_profile("AAPL")
    assert profile["is_foreign_private_issuer"] is False
    assert profile["annual_form"] == "10-K"
    assert profile["taxonomy"] == "us-gaap"
    assert profile["reporting_currency"] == "USD"
    assert profile["interim_form"] == "10-Q"


@network
def test_tsm_fy2025_revenue_matches_the_20f():
    """NT$3,809,054,300,000 for FY2025, against NT$2,894,307,700,000 in FY2024.
    Read live from the 2026-04-16 20-F."""
    result = fi.get_annual_revenue("TSM")
    assert result["success"] is True
    assert result["currency"] == "TWD"
    assert result["latest_revenue"] == pytest.approx(3.8090543e12, rel=1e-6)
    assert result["latest_revenue_usd"] == pytest.approx(1.214235e11, rel=1e-4)


@network
def test_asml_fy2025_revenue_matches_the_20f():
    """EUR 32,667,300,000 for FY2025 against EUR 28,262,900,000 in FY2024."""
    result = fi.get_annual_revenue("ASML")
    assert result["success"] is True
    assert result["currency"] == "EUR"
    assert result["latest_revenue"] == pytest.approx(3.26673e10, rel=1e-4)


@network
def test_sap_and_nvo_report_in_their_own_currencies():
    sap = fi.get_annual_revenue("SAP")
    assert sap["currency"] == "EUR"
    assert sap["latest_revenue"] == pytest.approx(3.68e10, rel=1e-4)
    nvo = fi.get_annual_revenue("NVO")
    assert nvo["currency"] == "DKK"
    assert nvo["latest_revenue"] == pytest.approx(3.09064e11, rel=1e-4)


@network
def test_a_us_filer_is_unaffected_by_the_new_path():
    """Proof the 10-K path did not move. MSFT FY2026 revenue is 281.7bn."""
    result = fi.get_annual_revenue("MSFT")
    assert result["success"] is True
    assert result["form"] == "10-K"
    assert result["currency"] == "USD"
    assert result["latest_revenue"] > 2.0e11


@network
def test_the_mismatch_note_fires_for_a_real_adr():
    note = fi.form_mismatch_note("TSM", "10-K")
    assert note and "20-F" in note
    assert fi.form_mismatch_note("MSFT", "10-K") is None
