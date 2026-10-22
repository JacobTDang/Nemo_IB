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


import pytest

SUCCESSOR = 2115436
PREDECESSOR = 34088
AGENT_PREFIX = 1193125


@pytest.fixture(autouse=True)
def _clear_cache():
    su.registrant._PREDECESSOR_CACHE.clear()
    yield
    su.registrant._PREDECESSOR_CACHE.clear()


class _Filing:
    def __init__(self, accession, form="10-K", filing_date="2026-02-18",
                 items=None, concept_values=None):
        self.accession_number = accession
        self.accession_no = accession
        self.form = form
        self.filing_date = filing_date
        self.filing_url = "https://example.invalid/" + accession
        self.items = items if items is not None else ("2.02" if form == "8-K" else "")
        self.acceptance_datetime = None
        self._concept_values = dict(concept_values or {})

    def xbrl(self):
        import pandas as pd

        class _Query:
            def __init__(self, rows):
                self._rows = rows

            def by_concept(self, concept):
                prefix = str(concept).replace("_", ":")
                keep = [r for r in self._rows
                        if str(r.get("concept", "")).replace("_", ":").startswith(prefix)]
                return _Query(keep)

            def to_dataframe(self):
                return pd.DataFrame(self._rows)

        class _Facts:
            def __init__(self, rows):
                self._rows = rows

            def query(self):
                return _Query(self._rows)

        rows = []
        for c, v in self._concept_values.items():
            rows.append({
                "concept": c,
                "period_instant": str(self.filing_date),
                "period_key": f"duration_{self.filing_date}",
                "numeric_value": float(v),
                "value": float(v),
                "unit_ref": "USD",
                "decimals": 0,
            })
        return type("XBRL", (), {"facts": _Facts(rows), "contexts": {}})()

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
    monkeypatch.setattr(su.registrant, "company", lambda ticker, cik=None: ticker_company)

    def company_for_cik(cik):
        try:
            val = int(cik)
            found = companies.get(val)
        except (ValueError, TypeError):
            return ticker_company
        if found is None:
            return _Company(int(cik) if str(cik).isdigit() else 0, "", {}, not_found=True)
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


def test_get_company_filings_history_reads_co_registrant_when_own_history_is_short(
        monkeypatch):
    """The successor has no 10-K, but lists a 10-Q under the predecessor's CIK.
    get_company_filings_history reads the predecessor's 10-Ks and names it in read_from."""
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

    result = su.get_company_filings_history("XOM", form_type="10-K", n=5)
    assert result["success"] is True
    assert result["filings_returned"] == 2
    assert result["read_from"]["name"] == "EXXON MOBIL CORP"
    assert result["read_from"]["cik"] == PREDECESSOR
    assert "EXXON MOBIL CORP" in result.get("note", "")
    assert str(PREDECESSOR) in result.get("note", "")
    assert result["filings"][0]["accession_number"] == ten_k_cur.accession_number
    assert result["filings"][1]["accession_number"] == ten_k_pri.accession_number


def test_get_company_filings_history_uses_own_when_sufficient(monkeypatch):
    """When the successor has enough filings of its own for n, it does not
    query the co-registrant."""
    holdco_filings = [_Filing(f"0002115436-26-0000{i:02d}", filing_date=f"2026-02-{20-i:02d}")
                      for i in range(5)]
    holdco = _Company(
        SUCCESSOR, "ExxonMobil Holdings Corp",
        {"10-K": holdco_filings,
         "10-Q": [_Filing("0000034088-26-000093", form="10-Q")]})
    pred_cur = _Filing("0000034088-26-000045", filing_date="2026-02-18")
    predecessor = _Company(
        PREDECESSOR, "EXXON MOBIL CORP",
        {"10-K": [pred_cur],
         "10-Q": [_Filing("0000034088-26-000093", form="10-Q")]})
    _install(monkeypatch, {PREDECESSOR: predecessor}, holdco)

    result = su.get_company_filings_history("XOM", form_type="10-K", n=5)
    assert result["success"] is True
    assert result["filings_returned"] == 5
    assert result.get("read_from") is None


def test_fetch_concept_series_reads_co_registrant_when_own_history_is_short(
        monkeypatch):
    """The successor has only one 10-Q with the concept; fetch_concept_series
    reaches into the co-registrant to satisfy limit and names the co-registrant."""
    from tools.web_search_server import sec_series

    q_holdco = _Filing("0000034088-26-000093", form="10-Q", filing_date="2026-08-03",
                       concept_values={"Revenues": 100.0})
    q_pred1 = _Filing("0000034088-26-000060", form="10-Q", filing_date="2026-05-02",
                      concept_values={"Revenues": 90.0})
    q_pred2 = _Filing("0000034088-26-000020", form="10-Q", filing_date="2026-02-15",
                      concept_values={"Revenues": 80.0})

    holdco = _Company(
        SUCCESSOR, "ExxonMobil Holdings Corp",
        {"10-Q": [q_holdco]})
    predecessor = _Company(
        PREDECESSOR, "EXXON MOBIL CORP",
        {"10-Q": [q_holdco, q_pred1, q_pred2]})
    _install(monkeypatch, {PREDECESSOR: predecessor}, holdco)
    monkeypatch.setattr(sec_series, "_require_identity", lambda: "test user@test.com")
    monkeypatch.setattr(sec_series, "_throttle", lambda: None)

    points = sec_series.fetch_concept_series("XOM", "Revenues", form="10-Q", limit=3)
    assert len(points) == 3
    assert points[0].read_from is None
    assert points[1].read_from["name"] == "EXXON MOBIL CORP"
    assert points[1].read_from["cik"] == PREDECESSOR
    assert points[2].read_from["name"] == "EXXON MOBIL CORP"
    assert points[2].read_from["cik"] == PREDECESSOR


def test_ordinary_company_with_long_history_never_calls_predecessor_with_form(
        monkeypatch):
    """An ordinary company whose own history reaches back to `since` (or meets
    `min_count`) never queries predecessor_with_form."""
    from tools.web_search_server import registrant

    # Filings going back to 2020
    jpm_filings = [
        _Filing("0000019617-26-000010", form="8-K", filing_date="2026-10-15"),
        _Filing("0000019617-25-000020", form="8-K", filing_date="2025-05-10"),
        _Filing("0000019617-24-000030", form="8-K", filing_date="2024-01-15"),
        _Filing("0000019617-20-000040", form="8-K", filing_date="2020-03-01"),
    ]
    comp = _Company(19617, "JPMorgan Chase & Co", {"8-K": jpm_filings})

    called = []
    def _fail_if_called(c, form):
        called.append((c, form))
        raise AssertionError("predecessor_with_form was called for an ordinary company with long history")

    monkeypatch.setattr(registrant, "predecessor_with_form", _fail_if_called)

    filings, read_from = registrant.filings_history(
        comp, form="8-K", since="2024-01-01")
    assert len(filings) == 4
    assert read_from is None
    assert not called


def test_predecessor_with_form_caches_result_per_cik_and_form(monkeypatch):
    """predecessor_with_form caches results so repeated calls for (cik, form)
    do not re-parse filings."""
    from tools.web_search_server import registrant

    holdco = _Company(
        SUCCESSOR, "ExxonMobil Holdings Corp",
        {"10-Q": [_Filing("0000034088-26-000093", form="10-Q")]})
    predecessor = _Company(
        PREDECESSOR, "EXXON MOBIL CORP",
        {"10-K": [_Filing("0000034088-26-000045", form="10-K")],
         "10-Q": [_Filing("0000034088-26-000093", form="10-Q")]})
    _install(monkeypatch, {PREDECESSOR: predecessor}, holdco)
    registrant._PREDECESSOR_CACHE.clear()

    res1 = registrant.predecessor_with_form(holdco, "10-K")
    assert res1 is predecessor
    assert (SUCCESSOR, "10-K") in registrant._PREDECESSOR_CACHE

    # Corrupt or make get_filings fail on holdco; cached lookup still returns predecessor
    holdco.get_filings = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("must not be called"))
    res2 = registrant.predecessor_with_form(holdco, "10-K")
    assert res2 is predecessor


def test_filings_history_propagates_multiple_registrants_refusal(monkeypatch):
    """When predecessor_with_form raises a refusal (such as more than one
    co-registrant with the form), filings_history lets it propagate."""
    import pytest
    from tools.web_search_server import registrant

    holdco = _Company(
        SUCCESSOR, "ExxonMobil Holdings Corp",
        {"10-K": [],
         "10-Q": [_Filing("0000034088-26-000093", form="10-Q")]})

    def _raise_multiple(c, form):
        raise RuntimeError("CIK 2115436 lists filings submitted by more than one registrant: refusal")

    monkeypatch.setattr(registrant, "predecessor_with_form", _raise_multiple)

    with pytest.raises(RuntimeError, match="more than one registrant"):
        registrant.filings_history(holdco, form="10-K")


def test_coregistrant_filing_after_as_of_is_not_used(monkeypatch):
    """Point-in-time: a co-registrant 8-K filed after as_of is never included
    in earnings_releases."""
    from research import announcements

    r_pred_before = _Filing("0000034088-26-000040", form="8-K", filing_date="2026-05-01")
    r_pred_after = _Filing("0000034088-26-000070", form="8-K", filing_date="2026-06-15")

    holdco = _Company(
        SUCCESSOR, "ExxonMobil Holdings Corp",
        {"8-K": [],
         "10-Q": [_Filing("0000034088-26-000093", form="10-Q")]})
    predecessor = _Company(
        PREDECESSOR, "EXXON MOBIL CORP",
        {"8-K": [r_pred_after, r_pred_before],
         "10-Q": [_Filing("0000034088-26-000093", form="10-Q")]})
    _install(monkeypatch, {PREDECESSOR: predecessor}, holdco)
    monkeypatch.setenv("SEC_EMAIL", "test@example.com")
    monkeypatch.setattr("edgar.set_identity", lambda id: None)

    releases = announcements.earnings_releases("XOM", as_of="2026-06-01")
    assert len(releases) == 1
    assert releases[0]["accession"] == "0000034088-26-000040"
    assert releases[0]["announced_date"] == "2026-05-01"
    assert releases[0]["read_from"]["name"] == "EXXON MOBIL CORP"


def test_dilution_get_share_count_series_includes_coregistrant_read_from(
        monkeypatch):
    """dilution.get_share_count_series reports read_from and note when points
    are sourced from a co-registrant."""
    from tools.web_search_server import dilution, sec_series

    shares_concept = "us-gaap:CommonStockSharesOutstanding"
    pt_holdco = sec_series.FilingPoint(
        filing_date="2026-08-03", form="10-Q", accession="0000034088-26-000093",
        facts=[sec_series.ConceptFact(concept=shares_concept, value=1000.0, period="2026-06-30", context_ref="c1")]
    )
    pt_pred = sec_series.FilingPoint(
        filing_date="2026-05-02", form="10-Q", accession="0000034088-26-000060",
        facts=[sec_series.ConceptFact(concept=shares_concept, value=990.0, period="2026-03-31", context_ref="c2")],
        read_from={"name": "EXXON MOBIL CORP", "cik": PREDECESSOR}
    )

    monkeypatch.setattr(
        dilution, "fetch_concept_series",
        lambda ticker, concept, form="10-Q", limit=8: [pt_holdco, pt_pred]
    )
    monkeypatch.setattr(dilution, "_consult_split_calendar", lambda t, d: ([], None))

    result = dilution.get_share_count_series("XOM")
    assert result["success"] is True
    assert result["read_from"]["name"] == "EXXON MOBIL CORP"
    assert result["read_from"]["cik"] == PREDECESSOR
    assert "EXXON MOBIL CORP" in result.get("note", "")


def test_sue_ts_release_includes_coregistrant_read_from(monkeypatch):
    """research.release_eps.sue_ts_release attaches read_from when the release
    is read from the co-registrant."""
    from research import announcements, release_eps, sue

    series = {
        "ticker": "XOM", "cik": str(SUCCESSOR), "concept": "EPS", "concepts_tried": ["EPS"],
        "basis_changes": [], "basis_unresolved": [], "success": True, "error": None,
        "quarters": [{
            "fiscal_period": "2026Q2", "fiscal_year": 2026, "fiscal_quarter": 2,
            "eps": 2.0, "eps_as_filed": 2.0, "basis_factor": 1.0, "source": "xbrl",
            "concept": "EPS", "known_at": "2026-08-05", "accession": "acc-1",
            "form": "10-Q", "derivation": None, "period_end": "2026-06-30"
        }]
    }
    monkeypatch.setattr(sue, "eps_series", lambda t, as_of=None: series)
    monkeypatch.setattr(
        announcements, "earnings_releases",
        lambda t, as_of=None, cik=None, since=None: [{
            "ticker": "XOM", "announced_date": "2026-05-01", "accession": "0000034088-26-000040",
            "timing": "bmo", "read_from": {"name": "EXXON MOBIL CORP", "cik": PREDECESSOR}
        }]
    )
    monkeypatch.setattr(
        announcements, "for_quarters",
        lambda t, as_of=None, quarters=None, cik=None, since=None: {
            "2026Q2": {
                "ticker": "XOM", "announced_date": "2026-05-01", "accession": "0000034088-26-000040",
                "fiscal_period": "2026Q2", "read_from": {"name": "EXXON MOBIL CORP", "cik": PREDECESSOR}
            }
        }
    )
    monkeypatch.setattr(
        release_eps, "read_release",
        lambda ticker, accession, period_end=None, filings=None: {"eps": 2.10, "reason": None}
    )
    holdco = _Company(
        SUCCESSOR, "ExxonMobil Holdings Corp",
        {"8-K": [],
         "10-Q": [_Filing("0000034088-26-000093", form="10-Q")]})
    predecessor = _Company(
        PREDECESSOR, "EXXON MOBIL CORP",
        {"8-K": [_Filing("0000034088-26-000040", form="8-K", filing_date="2026-05-01")],
         "10-Q": [_Filing("0000034088-26-000093", form="10-Q")]})
    _install(monkeypatch, {PREDECESSOR: predecessor}, holdco)
    monkeypatch.setattr(release_eps, "_company", lambda t, c=None: holdco)

    sig = release_eps.sue_ts_release("XOM")
    assert sig.get("read_from") == {"name": "EXXON MOBIL CORP", "cik": PREDECESSOR}
    assert "EXXON MOBIL CORP" in sig.get("note", "")


def test_announcements_and_release_eps_read_co_registrant_8ks(monkeypatch):
    """8-K earnings releases for the successor include the predecessor's releases
    when the successor's 8-K list is short, and each record carries read_from."""
    from research import announcements

    r_holdco = _Filing("0002115436-26-000088", form="8-K", filing_date="2026-07-28")
    r_pred = _Filing("0000034088-26-000040", form="8-K", filing_date="2026-04-26")

    holdco = _Company(
        SUCCESSOR, "ExxonMobil Holdings Corp",
        {"8-K": [r_holdco],
         "10-Q": [_Filing("0000034088-26-000093", form="10-Q")]})
    predecessor = _Company(
        PREDECESSOR, "EXXON MOBIL CORP",
        {"8-K": [r_pred],
         "10-Q": [_Filing("0000034088-26-000093", form="10-Q")]})
    _install(monkeypatch, {PREDECESSOR: predecessor}, holdco)
    monkeypatch.setenv("SEC_EMAIL", "test@example.com")
    monkeypatch.setattr("edgar.set_identity", lambda id: None)

    releases = announcements.earnings_releases("XOM")
    assert len(releases) == 2
    assert releases[0]["accession"] == "0002115436-26-000088"
    assert releases[0].get("read_from") is None
    assert releases[1]["accession"] == "0000034088-26-000040"
    assert releases[1]["read_from"]["name"] == "EXXON MOBIL CORP"
    assert releases[1]["read_from"]["cik"] == PREDECESSOR


def test_release_eps_exhibits_finds_co_registrant_8k(monkeypatch):
    """release_eps._exhibits resolves 8-K filings from predecessor when holdco
    has no 8-Ks of its own."""
    from research import release_eps

    r_pred = _Filing("0000034088-26-000040", form="8-K", filing_date="2026-04-26")
    holdco = _Company(
        SUCCESSOR, "ExxonMobil Holdings Corp",
        {"8-K": [],
         "10-Q": [_Filing("0000034088-26-000093", form="10-Q")]})
    predecessor = _Company(
        PREDECESSOR, "EXXON MOBIL CORP",
        {"8-K": [r_pred],
         "10-Q": [_Filing("0000034088-26-000093", form="10-Q")]})
    _install(monkeypatch, {PREDECESSOR: predecessor}, holdco)
    monkeypatch.setenv("SEC_EMAIL", "test@example.com")
    monkeypatch.setattr("edgar.set_identity", lambda id: None)
    monkeypatch.setattr(release_eps, "_company", lambda t, c=None: holdco)

    mock_doc = type("Doc", (), {"document": "ex99-1.htm", "description": "EX-99.1"})()
    r_pred.homepage = type("HP", (), {"documents": [mock_doc]})()
    exhibits = release_eps._exhibits("XOM", "0000034088-26-000040")
    assert exhibits == [mock_doc]


