"""A ticker can map to a holding company that has not filed the form asked for.

SEC's ticker file maps XOM to ExxonMobil Holdings Corp (CIK 2115436). That
registrant has no 10-K. The 10-Ks stay with EXXON MOBIL CORP (CIK 34088). The
two are linked in SEC's own index: the 2026-08-03 10-Q, accession
0000034088-26-000093, is listed under both CIKs, and the filing names both as
filers. The accession begins with the predecessor's CIK.
"""
from collections import OrderedDict

import edgar

from tools.web_search_server import sec_utils as su


SUCCESSOR = 2115436
PREDECESSOR = 34088
AGENT_PREFIX = 1193125


class _Filing:
    def __init__(self, accession, form="10-K", filing_date="2026-02-18"):
        self.accession_number = accession
        self.form = form
        self.filing_date = filing_date
        self.filing_url = "https://example.invalid/" + accession

    def xbrl(self):
        return object()

    def text(self):
        return (
            " " * 35000
            + "ITEM 1A. RISK FACTORS\n\n"
            + f"Sample risk factor paragraph for {self.accession_number} that is long enough to count.\n\n"
            + "ITEM 1B"
        )


class _FilingsList(list):
    def head(self, n):
        return _FilingsList(self[:n])


class _Company:
    def __init__(self, cik, name, by_form, not_found=False):
        self.cik = cik
        self.name = name
        self.not_found = not_found
        self._by_form = by_form

    def get_filings(self, form=None, amendments=True, **kwargs):
        if self.not_found:
            raise RuntimeError(f"CIK {self.cik} is not a registrant")
        if form is None:
            rows = []
            for group in self._by_form.values():
                rows.extend(group)
            return _FilingsList(rows)
        return _FilingsList(self._by_form.get(form, []))


def _install(monkeypatch, companies, ticker_company):
    monkeypatch.setattr(su, "_filing_cache_lru", OrderedDict())
    monkeypatch.setattr(su, "_require_identity", lambda: None)
    monkeypatch.setattr(su.registrant, "company", lambda ticker: ticker_company)

    def company_for_cik(cik):
        found = companies.get(int(cik))
        if found is None:
            return _Company(int(cik), "", {}, not_found=True)
        return found

    monkeypatch.setattr(edgar, "Company", company_for_cik)


def test_a_registrant_with_no_requested_form_is_named_with_its_cik(monkeypatch):
    """No 10-K, and the only other accession belongs to a filing agent that is
    not a CIK. The refusal names the registrant, its CIK and the form."""
    holdco = _Company(
        SUCCESSOR, "ExxonMobil Holdings Corp",
        {"8-K12B": [_Filing("0001193125-26-291990", form="8-K12B")]})
    _install(monkeypatch, {}, holdco)

    found = su.get_latest_filing("XOM", "10-K")
    message = found["error"]
    assert "ExxonMobil Holdings Corp" in message
    assert str(SUCCESSOR) in message
    assert "10-K" in message

    revenue = su.get_revenue_base("XOM")
    assert revenue["success"] is False
    assert "No XBRL data available" not in revenue["error"]
    assert "ExxonMobil Holdings Corp" in revenue["error"]
    assert str(SUCCESSOR) in revenue["error"]
    assert "10-K" in revenue["error"]


def test_a_co_registrant_accession_is_read_when_the_successor_has_no_form(
        monkeypatch):
    """The successor lists a filing whose accession begins with another
    registrant's CIK, and that registrant has the 10-K."""
    ten_k = _Filing("0000034088-26-000045")
    holdco = _Company(
        SUCCESSOR, "ExxonMobil Holdings Corp",
        {"10-Q": [_Filing("0000034088-26-000093", form="10-Q")]})
    predecessor = _Company(
        PREDECESSOR, "EXXON MOBIL CORP",
        {"10-K": [ten_k],
         "10-Q": [_Filing("0000034088-26-000093", form="10-Q")]})
    _install(monkeypatch, {PREDECESSOR: predecessor, AGENT_PREFIX: None}, holdco)
    # AGENT_PREFIX is absent from the map, so it is not a registrant. The
    # 10-Q accession is the only link.

    found = su.get_latest_filing("XOM", "10-K")
    assert found["accession_number"] == ten_k.accession_number


def test_a_predecessor_filing_names_the_registrant_it_was_read_from(monkeypatch):
    """The 10-K is the predecessor's. The result says whose filing was read."""
    ten_k = _Filing("0000034088-26-000045")
    holdco = _Company(
        SUCCESSOR, "ExxonMobil Holdings Corp",
        {"10-Q": [_Filing("0000034088-26-000093", form="10-Q")]})
    predecessor = _Company(
        PREDECESSOR, "EXXON MOBIL CORP",
        {"10-K": [ten_k],
         "10-Q": [_Filing("0000034088-26-000093", form="10-Q")]})
    _install(monkeypatch, {PREDECESSOR: predecessor}, holdco)

    found = su.get_latest_filing("XOM", "10-K")
    assert found["read_from"]["name"] == "EXXON MOBIL CORP"
    assert found["read_from"]["cik"] == PREDECESSOR

    revenue = su.get_revenue_base("XOM")
    text = f"{revenue.get('error', '')} {revenue.get('note', '')}"
    assert "EXXON MOBIL CORP" in text
    assert str(PREDECESSOR) in text


HOLDER = 1364742


def test_a_holder_report_about_the_company_is_not_a_predecessor(monkeypatch):
    """A SCHEDULE 13G about a company sits in that company's list, and its
    accession carries the holder's CIK. BlackRock files 10-Ks. Read as a
    co-registrant, a recent listing or a foreign filer with no 10-K would be
    answered with BlackRock's."""
    holdco = _Company(
        SUCCESSOR, "ExxonMobil Holdings Corp",
        {"SCHEDULE 13G": [_Filing("0001364742-26-000101",
                                  form="SCHEDULE 13G")]})
    holder = _Company(
        HOLDER, "BlackRock, Inc.",
        {"10-K": [_Filing("0001364742-26-000012")],
         "SCHEDULE 13G": [_Filing("0001364742-26-000101",
                                  form="SCHEDULE 13G")]})
    _install(monkeypatch, {HOLDER: holder}, holdco)

    found = su.get_latest_filing("XOM", "10-K")
    assert not found.get("accession_number"), "read the holder's 10-K"
    assert "ExxonMobil Holdings Corp" in found["error"]


def test_a_shared_accession_must_be_listed_under_both_registrants(monkeypatch):
    """The prefix alone is who submitted the filing. Only a filing that SEC
    lists under both CIKs makes the other one a co-registrant."""
    holdco = _Company(
        SUCCESSOR, "ExxonMobil Holdings Corp",
        {"10-Q": [_Filing("0000034088-26-000093", form="10-Q")]})
    unrelated = _Company(
        PREDECESSOR, "EXXON MOBIL CORP",
        {"10-K": [_Filing("0000034088-26-000045")]})
    _install(monkeypatch, {PREDECESSOR: unrelated}, holdco)

    found = su.get_latest_filing("XOM", "10-K")
    assert not found.get("accession_number")
    assert "ExxonMobil Holdings Corp" in found["error"]


def test_diff_10k_reads_predecessors_two_10ks_when_successor_has_no_10k(
        monkeypatch):
    """The successor has no 10-K, but lists a 10-Q under the predecessor's CIK.
    The predecessor has two 10-Ks. diff_10k reads the predecessor's two 10-Ks
    and names it in read_from."""
    ten_k_cur = _Filing("0000034088-26-000045", filing_date="2026-02-18")
    ten_k_pri = _Filing("0000034088-25-000030", filing_date="2025-02-19")
    holdco = _Company(
        SUCCESSOR, "ExxonMobil Holdings Corp",
        {"10-Q": [_Filing("0000034088-26-000093", form="10-Q")]})
    predecessor = _Company(
        PREDECESSOR, "EXXON MOBIL CORP",
        {"10-K": [ten_k_cur, ten_k_pri],
         "10-Q": [_Filing("0000034088-26-000093", form="10-Q")]})
    _install(monkeypatch, {PREDECESSOR: predecessor}, holdco)

    result = su.diff_10k("XOM")
    assert result["success"] is True
    assert result["read_from"]["name"] == "EXXON MOBIL CORP"
    assert result["read_from"]["cik"] == PREDECESSOR
    assert "EXXON MOBIL CORP" in result.get("note", "")
    assert str(PREDECESSOR) in result.get("note", "")
    assert result["current_filing_date"] == "2026-02-18"
    assert result["prior_filing_date"] == "2025-02-19"


def test_diff_10k_uses_successors_own_filings_when_it_has_two_10ks(monkeypatch):
    """A successor with its own two 10-Ks is diffed from its own filings, not
    from any co-registrant."""
    holdco_cur = _Filing("0002115436-26-000010", filing_date="2026-02-20")
    holdco_pri = _Filing("0002115436-25-000005", filing_date="2025-02-21")
    holdco = _Company(
        SUCCESSOR, "ExxonMobil Holdings Corp",
        {"10-K": [holdco_cur, holdco_pri],
         "10-Q": [_Filing("0000034088-26-000093", form="10-Q")]})
    pred_cur = _Filing("0000034088-26-000045", filing_date="2026-02-18")
    pred_pri = _Filing("0000034088-25-000030", filing_date="2025-02-19")
    predecessor = _Company(
        PREDECESSOR, "EXXON MOBIL CORP",
        {"10-K": [pred_cur, pred_pri],
         "10-Q": [_Filing("0000034088-26-000093", form="10-Q")]})
    _install(monkeypatch, {PREDECESSOR: predecessor}, holdco)

    result = su.diff_10k("XOM")
    assert result["success"] is True
    assert result.get("read_from") is None
    assert "read_from" not in result
    assert result["current_filing_date"] == "2026-02-20"
    assert result["prior_filing_date"] == "2025-02-21"


def test_diff_10k_diffs_a_first_own_10k_against_the_predecessors_last(
        monkeypatch):
    """Once the successor files its first 10-K it has one of its own. That is
    the current year; the prior year is the predecessor's latest. Switching to
    the predecessor's two would diff two years that are both out of date."""
    holdco_first = _Filing("0002115436-27-000010", filing_date="2027-02-19")
    holdco = _Company(
        SUCCESSOR, "ExxonMobil Holdings Corp",
        {"10-K": [holdco_first],
         "10-Q": [_Filing("0000034088-26-000093", form="10-Q")]})
    pred_cur = _Filing("0000034088-26-000045", filing_date="2026-02-18")
    pred_pri = _Filing("0000034088-25-000030", filing_date="2025-02-19")
    predecessor = _Company(
        PREDECESSOR, "EXXON MOBIL CORP",
        {"10-K": [pred_cur, pred_pri],
         "10-Q": [_Filing("0000034088-26-000093", form="10-Q")]})
    _install(monkeypatch, {PREDECESSOR: predecessor}, holdco)

    result = su.diff_10k("XOM")
    assert result["success"] is True
    assert result["current_filing_date"] == "2027-02-19"
    assert result["prior_filing_date"] == "2026-02-18"
    assert result["read_from"]["cik"] == PREDECESSOR
