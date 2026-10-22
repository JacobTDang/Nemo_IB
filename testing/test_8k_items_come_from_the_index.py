"""An 8-K's items come from the filing index, and a failed read is a failure.

Both 8-K readers parsed the document to learn which items a filing reports:
`get_earnings_releases` through `data_object().items`, `extract_8k_events`
through `obj().items`. Both swallowed the failure. In October 2026 SEC began
to answer edgartools 5.8.0's filing index URL with 403 (#138), and every parse
failed with `'NoneType' object has no attribute 'download'`:

    get_earnings_releases("WINA")   "No 8-K Item 2.02 filings found in last
                                     30 8-Ks" -- WINA reports Item 2.02 on 17
                                     of them
    extract_8k_events("WINA")       the 2026-10-14 results 8-K with
                                     sec_items [] and confidence 0.0

The second also lost its text: `safe_get_text` turned the same exception into
an empty string, so the regex fallback had nothing to read either.

EDGAR's submissions index lists the items next to each accession number
("2.02,7.01,8.01,9.01"), and it comes with the filing list, so reading it
costs no request. An outage that reaches the document must say which filing
it could not read. It must never read as a company that files no earnings
releases, or as an 8-K about nothing.
"""
import importlib

import pytest

import tools.web_search_server.sec_utils as su

_utils = importlib.import_module("tools.web_search_server.8K_and_DEF14A_utils")

# What edgartools 5.8.0 raised for every inline-XBRL document once SEC refused
# the index page.
_INDEX_403 = AttributeError("'NoneType' object has no attribute 'download'")

_RELEASE = ("WINMARK CORPORATION ANNOUNCES THIRD QUARTER RESULTS. Net income "
            "for the quarter was $11.2 million, or $3.01 per diluted share.")
_EIGHT_K = ("FORM 8-K\nItem 2.02 Results of Operations and Financial Condition.\n"
            "On October 14, 2026 the Company issued a press release announcing "
            "its financial results for the quarter ended September 27, 2026.\n"
            "Item 9.01 Financial Statements and Exhibits.\n")


class _Attachment:
    def __init__(self, document, description, text=None, error=None):
        self.document = document
        self.description = description
        self._text = text
        self._error = error

    def text(self):
        if self._error is not None:
            raise self._error
        return self._text


class _Filing:
    """An 8-K as `get_filings` lists it. Every read of the document fails the
    way it failed under the 403, unless the test supplies the text."""
    form = "8-K"

    def __init__(self, accession, filing_date, items, attachments=(),
                 attachments_error=None, text=None, text_error=_INDEX_403):
        self.accession_no = accession
        self.accession_number = accession
        self.filing_date = filing_date
        self.items = items
        self.filing_url = f"https://www.sec.gov/Archives/edgar/data/908315/{accession}.txt"
        self._attachments = list(attachments)
        self._attachments_error = attachments_error
        self._text = text
        self._text_error = text_error

    @property
    def attachments(self):
        if self._attachments_error is not None:
            raise self._attachments_error
        return self._attachments

    def text(self):
        if self._text is not None:
            return self._text
        raise self._text_error

    def html(self):
        raise _INDEX_403

    def data_object(self):
        raise _INDEX_403

    def obj(self):
        raise _INDEX_403


class _Filings(list):
    def head(self, n):
        return _Filings(self[:n])


def _results_8k(accession="0000908315-26-000038", filing_date="2026-10-14",
                **kwargs):
    kwargs.setdefault("attachments", [
        _Attachment("wina-20261013x8k.htm", "8-K"),
        _Attachment("wina-ex99d1.htm", "EX-99.1", text=_RELEASE)])
    return _Filing(accession, filing_date, "2.02,7.01,8.01,9.01", **kwargs)


def _install(monkeypatch, module, filings):
    """Serve `filings` where `module` asks for the registrant: both readers go
    through `registrant.company` (#136)."""
    class _Company:
        def __init__(self, ticker):
            pass

        def get_filings(self, form=None):
            return _Filings(filings)

    monkeypatch.setattr(module.registrant, "company", _Company)
    monkeypatch.setenv("SEC_EMAIL", "test@example.invalid")


# ---------------------------------------------------------------------------
# get_earnings_releases
# ---------------------------------------------------------------------------

def test_earnings_releases_read_the_items_from_the_index(monkeypatch):
    """The reproduction. Every document read fails as it did under the 403,
    and the index still says which 8-Ks are results releases."""
    _install(monkeypatch, su, [
        _Filing("0000908315-26-000040", "2026-10-16", "5.02,9.01"),
        _results_8k()])

    out = su.get_earnings_releases("WINA", max_quarters=1)

    assert out["success"] is True, out.get("error")
    [release] = out["releases"]
    assert release["accession_number"] == "0000908315-26-000038"
    assert release["items"] == ["2.02", "7.01", "8.01", "9.01"]
    assert release["text"] == _RELEASE


def test_earnings_releases_match_item_2_02_whole(monkeypatch):
    """A substring test reads "12.02" as a results release. The index lists
    codes, so the match is on the code."""
    _install(monkeypatch, su, [
        _Filing("0000908315-26-000041", "2026-10-17", "12.02,9.01")])

    out = su.get_earnings_releases("WINA", max_quarters=1)

    assert out["success"] is False
    assert "No 8-K Item 2.02" in out["error"]


def test_an_index_entry_with_no_items_is_a_failure_naming_the_filing(monkeypatch):
    """Every 8-K reports at least one item, so an empty entry is a gap in what
    EDGAR gave us. Read as "no Item 2.02", it becomes a claim about the
    company."""
    _install(monkeypatch, su, [
        _Filing("0000908315-26-000040", "2026-10-16", ""),
        _results_8k()])

    out = su.get_earnings_releases("WINA", max_quarters=1)

    assert out["success"] is False
    assert "0000908315-26-000040" in out["error"], out["error"]
    assert "No 8-K Item 2.02" not in out["error"], (
        f"an unreadable filing was reported as an absence: {out['error']}")


def test_a_filing_with_no_items_attribute_is_a_failure_naming_it(monkeypatch):
    filing = _results_8k()
    del filing.items
    _install(monkeypatch, su, [filing])

    out = su.get_earnings_releases("WINA", max_quarters=1)

    assert out["success"] is False
    assert "0000908315-26-000038" in out["error"], out["error"]
    assert "No 8-K Item 2.02" not in out["error"], out["error"]


def test_a_release_whose_text_cannot_be_read_is_a_failure_naming_the_filing(
        monkeypatch):
    """A release with `text: None` reads as a filer that attached nothing."""
    _install(monkeypatch, su, [_results_8k(attachments=[
        _Attachment("wina-ex99d1.htm", "EX-99.1", error=_INDEX_403)])])

    out = su.get_earnings_releases("WINA", max_quarters=1)

    assert out["success"] is False
    assert "0000908315-26-000038" in out["error"], out["error"]
    assert "download" in out["error"], (
        f"the response does not say what went wrong: {out['error']}")


def test_a_release_whose_attachments_cannot_be_listed_is_a_failure_naming_it(
        monkeypatch):
    _install(monkeypatch, su, [_results_8k(attachments_error=_INDEX_403)])

    out = su.get_earnings_releases("WINA", max_quarters=1)

    assert out["success"] is False
    assert "0000908315-26-000038" in out["error"], out["error"]
    assert "download" in out["error"], out["error"]


def test_a_filer_with_no_results_8ks_is_still_reported_as_having_none(
        monkeypatch):
    """The fix must not turn an ordinary absence into an error."""
    _install(monkeypatch, su, [
        _Filing("0000908315-26-000040", "2026-10-16", "5.02,9.01"),
        _Filing("0000908315-26-000021", "2026-05-06", "5.07")])

    out = su.get_earnings_releases("WINA", max_quarters=1)

    assert out["success"] is False
    assert "No 8-K Item 2.02 filings found" in out["error"]


# ---------------------------------------------------------------------------
# extract_8k_events
# ---------------------------------------------------------------------------

def test_8k_events_read_the_items_from_the_index(monkeypatch):
    """The second reproduction. `obj()` fails as it did under the 403, and
    the event still carries its items and a classification."""
    _install(monkeypatch, _utils, [_results_8k(text=_EIGHT_K)])

    out = _utils.SECFilingParser().extract_8k_events("WINA", 1)

    assert out["success"] is True, out.get("error")
    [event] = out["events_by_date"].values()
    assert event["sec_items"] == ["2.02", "7.01", "8.01", "9.01"]
    assert event["items_source"] == "filing_index"
    assert event["event_type"] != "unknown"
    # The index is the filer's own declaration of its items: the structured
    # source the 0.6 floor was for, not a regex guess over the text.
    assert event["confidence"] >= 0.6, event["confidence"]


def test_8k_event_text_that_cannot_be_read_is_a_failure_naming_the_filing(
        monkeypatch):
    """An empty body is evidence of nothing. Kept as "", it reached the
    classifier and the response as an 8-K with no text."""
    _install(monkeypatch, _utils, [_results_8k()])

    out = _utils.SECFilingParser().extract_8k_events("WINA", 1)

    assert out["success"] is False
    assert "0000908315-26-000038" in out["error"], out["error"]
    assert "download" in out["error"], out["error"]


def test_an_8k_event_with_no_index_items_is_a_failure_naming_the_filing(
        monkeypatch):
    _install(monkeypatch, _utils, [
        _Filing("0000908315-26-000040", "2026-10-16", None, text=_EIGHT_K)])

    out = _utils.SECFilingParser().extract_8k_events("WINA", 1)

    assert out["success"] is False
    assert "0000908315-26-000040" in out["error"], out["error"]


# ---------------------------------------------------------------------------
# The proxy readers share the text and HTML helpers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("read", ["filing_text", "filing_html"])
def test_a_document_read_that_fails_names_the_filing(read):
    """`extract_proxy_compensation` and `extract_governance_data` read the
    same way. An empty string from a failed read became "no executives" and
    "no board members"."""
    filing = _Filing("0001045810-26-000021", "2026-05-08", None)
    filing.form = "DEF 14A"

    with pytest.raises(Exception) as caught:
        getattr(_utils, read)(filing)

    message = str(caught.value)
    assert "0001045810-26-000021" in message, message
    assert "download" in message, message


# ---------------------------------------------------------------------------
# The upgrade
# ---------------------------------------------------------------------------

def test_edgartools_asks_for_the_index_page_inside_the_accession_folder():
    """The cause. edgartools 5.8.0 asked for
    /data/1045810/0001045810-26-000021-index.html, which SEC answers with 403;
    the same page inside the accession folder answers 200. Every 10-K whose
    primary document is inline XBRL is read through that page, so a pin back
    below the fix (edgartools #1421) breaks the text tools again -- and only
    on a cold cache, which is the state a fresh container starts in."""
    from edgar import Filing

    filing = Filing(cik=1045810, company="NVIDIA CORP", form="10-K",
                    filing_date="2026-02-25",
                    accession_no="0001045810-26-000021")

    assert filing.homepage_url.startswith(
        "https://www.sec.gov/Archives/edgar/data/1045810/000104581026000021/"), (
        filing.homepage_url)
