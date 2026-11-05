"""A 13D/G in a company's EDGAR folder is not necessarily *about* that company.

`get_schedule_13d_filings` promises "filings naming the target ticker as
subject company". It reads `Company(ticker).get_filings(form='SC 13D')`, and a
CIK's folder holds both sides of the relationship: filings where the company is
the SUBJECT (someone took a stake in it) and filings where it is the FILER
(it took a stake in someone else).

Measured live on INTC: 71 of 100 returned rows were Intel filing on other
issuers -- MariaDB, Mobileye, Joby, Vuzix, Clearwire. `activist_count: 124` was
not 124 activist positions in Intel.

An earlier fix made this count stable across page sizes, which it now is. It
made a contaminated number consistently wrong rather than inconsistently wrong,
and the stability was mistaken for correctness because the number was non-zero.
That is the failure this test exists to prevent recurring.

The two sides are distinguishable for free. EDGAR assigns an `005-` file number
belonging to the SUBJECT of a Schedule 13 filing. In the subject's own folder
that number is present and constant; on filings the company made about others
it is blank. Verified against header ground truth on INTC: 28 of 28 filings
agreed, and every subject-side row carried Intel's own `005-19567`.

The submissions index carries the file number, so the whole set can be
classified without fetching anything. The page we return is verified against
the filing header, which is the authority, and a disagreement between the two
is reported rather than resolved silently.
"""
import pytest


class _CI:
    def __init__(self, name, cik):
        self.name = name
        self.cik = cik


class _Party:
    def __init__(self, name, cik):
        self.company_information = _CI(name, cik)


class _Header:
    def __init__(self, subject, filer):
        self.subject_companies = [_Party(*subject)] if subject else []
        self.filers = [_Party(*filer)] if filer else []


class _Filing:
    def __init__(self, accession, date, file_number, subject, filer,
                 form=None):
        self.form = form
        self.accession_number = accession
        self.filing_date = date
        self.file_number = file_number
        self.header = _Header(subject, filer)
        self.filing_url = f"https://sec.gov/{accession}.htm"

    def text(self):
        return "Percent of class represented by amount in row (11): 7.5%"


class _Filings(list):
    def head(self, n):
        return _Filings(self[:n])


INTC = ("INTEL CORP", "0000050863")


def _company_with(filings):
    class _Company:
        cik = 50863

        def __init__(self, ticker):
            pass

        def get_filings(self, form=None, amendments=True):
            return _Filings([f for f in filings if f[1] == form][0][0]
                            if False else
                            [f for (f, frm) in filings if frm == form])
    return _Company


@pytest.fixture
def patched(monkeypatch):
    import tools.web_search_server.sec_utils as su
    monkeypatch.setattr(su, "_require_identity", lambda: "t@example.invalid")
    return su


def _install(monkeypatch, su, by_form):
    """`by_form` maps each query to what edgartools answers it with. A filing
    with no form of its own takes the form of the query it is listed under."""
    for form, filings in by_form.items():
        for f in filings:
            if f.form is None:
                f.form = form

    class _Company:
        cik = 50863

        def __init__(self, ticker):
            pass

        def get_filings(self, form=None, amendments=True):
            return _Filings(by_form.get(form, []))

    monkeypatch.setattr(su.registrant, "company", _Company)


def test_a_filing_the_company_made_about_someone_else_is_not_counted(
        patched, monkeypatch):
    su = patched
    _install(monkeypatch, su, {
        "SC 13D": [
            # Intel taking a stake in Vuzix -- not an activist in Intel.
            _Filing("a1", "2021-01-29", "", ("Vuzix Corp", "0001463972"), INTC),
        ],
        "SC 13D/A": [],
        "SC 13G": [
            # Vanguard taking a stake in Intel -- this one counts.
            _Filing("a2", "2024-02-13", "005-19567", INTC,
                    ("VANGUARD GROUP INC", "0000102909")),
        ],
        "SC 13G/A": [],
    })

    result = su.get_schedule_13d_filings("INTC")

    assert result["activist_count"] == 0, (
        "a stake Intel took in Vuzix was counted as an activist in Intel")
    assert result["passive_count"] == 1
    assert result["count"] == 1


def test_the_companys_own_filings_are_reported_separately_not_hidden(
        patched, monkeypatch):
    """Intel's stakes in other issuers are real information -- just a different
    question from the one this tool answers."""
    su = patched
    _install(monkeypatch, su, {
        "SC 13D": [_Filing("a1", "2021-01-29", "",
                           ("Vuzix Corp", "0001463972"), INTC)],
        "SC 13D/A": [], "SC 13G": [], "SC 13G/A": [],
    })

    result = su.get_schedule_13d_filings("INTC")
    assert result.get("filed_by_this_company_count") == 1, (
        "the filer-side rows vanished without trace")


def test_each_returned_row_names_the_subject(patched, monkeypatch):
    su = patched
    _install(monkeypatch, su, {
        "SC 13D": [], "SC 13D/A": [],
        "SC 13G": [_Filing("a2", "2024-02-13", "005-19567", INTC,
                           ("VANGUARD GROUP INC", "0000102909"))],
        "SC 13G/A": [],
    })

    row = su.get_schedule_13d_filings("INTC")["filings"][0]
    assert row["filer_name"] == "VANGUARD GROUP INC"
    assert row.get("subject_name") == "INTEL CORP"
    assert row.get("is_subject") is True


def test_a_subject_we_cannot_confirm_is_flagged_not_dropped(patched,
                                                            monkeypatch):
    """A row whose header will not parse must not be silently discarded --
    that would be an absence created by our own failure."""
    su = patched
    unreadable = _Filing("a3", "2024-03-01", "005-19567", None, None)
    _install(monkeypatch, su, {
        "SC 13D": [], "SC 13D/A": [],
        "SC 13G": [unreadable], "SC 13G/A": [],
    })

    result = su.get_schedule_13d_filings("INTC")
    assert result["count"] == 1, "an unverifiable row was dropped"
    assert result["filings"][0].get("subject_verified") is False


# --- structured Schedule 13D/A & 13G/A form names (issue #180) -------------

def test_get_schedule_13d_filings_pulls_new_schedule_names(patched,
                                                           monkeypatch):
    """get_schedule_13d_filings must pull SCHEDULE 13D, 13D/A, 13G, 13G/A."""
    su = patched
    d_initial = _Filing("n1", "2025-07-17", "005-19567", INTC,
                        ("ACTIVIST LP", "0001234567"))
    d_amend = _Filing("n2", "2025-07-31", "005-19567", INTC,
                      ("ACTIVIST LP", "0001234567"))
    g_initial = _Filing("n3", "2025-08-15", "005-19567", INTC,
                        ("PASSIVE FUND", "0007654321"))
    g_amend = _Filing("n4", "2025-08-30", "005-19567", INTC,
                      ("PASSIVE FUND", "0007654321"))

    _install(monkeypatch, su, {
        "SC 13D": [], "SC 13D/A": [], "SC 13G": [], "SC 13G/A": [],
        "SCHEDULE 13D": [d_initial],
        "SCHEDULE 13D/A": [d_amend],
        "SCHEDULE 13G": [g_initial],
        "SCHEDULE 13G/A": [g_amend],
    })

    result = su.get_schedule_13d_filings("INTC")
    assert result["count"] == 4
    assert result["activist_count"] == 2
    assert result["passive_count"] == 2


def test_get_schedule_13d_filings_classifies_schedule_13g_as_passive(
        patched, monkeypatch):
    """SCHEDULE 13D/A is activist (is_activist=True), SCHEDULE 13G is passive
    (is_activist=False)."""
    su = patched
    d_amend = _Filing("n2", "2025-07-31", "005-19567", INTC,
                      ("ACTIVIST LP", "0001234567"))
    g_row = _Filing("n3", "2025-08-15", "005-19567", INTC,
                    ("PASSIVE FUND", "0007654321"))

    _install(monkeypatch, su, {
        "SC 13D": [], "SC 13D/A": [], "SC 13G": [], "SC 13G/A": [],
        "SCHEDULE 13D": [],
        "SCHEDULE 13D/A": [d_amend],
        "SCHEDULE 13G": [g_row],
        "SCHEDULE 13G/A": [],
    })

    result = su.get_schedule_13d_filings("INTC")
    rows = {r["accession"]: r for r in result["filings"]}
    assert rows["n2"]["is_activist"] is True
    assert rows["n2"]["is_amendment"] is True
    assert rows["n3"]["is_activist"] is False
    assert rows["n3"]["is_amendment"] is False
    assert result["activist_count"] == 1
    assert result["passive_count"] == 1


def test_get_schedule_13d_filings_preserves_legacy_sc_13d_filings(
        patched, monkeypatch):
    """Old SC 13D and SC 13D/A filings continue to be pulled and classified as
    activist."""
    su = patched
    sc_d = _Filing("old1", "2024-06-01", "005-19567", INTC,
                   ("OLD ACTIVIST", "0001111111"))
    _install(monkeypatch, su, {
        "SC 13D": [sc_d], "SC 13D/A": [], "SC 13G": [], "SC 13G/A": [],
        "SCHEDULE 13D": [], "SCHEDULE 13D/A": [],
        "SCHEDULE 13G": [], "SCHEDULE 13G/A": [],
    })
    result = su.get_schedule_13d_filings("INTC")
    assert result["count"] == 1
    assert result["activist_count"] == 1
    assert result["filings"][0]["is_activist"] is True


def test_an_amendment_returned_for_the_initial_form_keeps_its_own_form(
        patched, monkeypatch):
    """edgartools answers a query for `SCHEDULE 13D` with the amendments as
    well. Labelled by the query, GNK's 14 amendments read as 14 new stakes."""
    su = patched
    initial = _Filing("i1", "2025-07-17", "005-19567", INTC,
                      ("ACTIVIST LP", "0001234567"), form="SCHEDULE 13D")
    amendment = _Filing("a1", "2025-07-31", "005-19567", INTC,
                        ("ACTIVIST LP", "0001234567"), form="SCHEDULE 13D/A")
    _install(monkeypatch, su, {
        "SCHEDULE 13D": [initial, amendment],
        "SCHEDULE 13D/A": [amendment],
    })

    result = su.get_schedule_13d_filings("INTC", include_passive=False)
    rows = {r["accession"]: r for r in result["filings"]}

    assert result["rows_returned"] == 2 and result["count"] == 2
    assert rows["a1"]["form"] == "SCHEDULE 13D/A"
    assert rows["a1"]["is_amendment"] is True
    assert rows["i1"]["form"] == "SCHEDULE 13D"
    assert rows["i1"]["is_amendment"] is False
