"""An 8-K event must fit in a tool result, and must say what it cut.

`extract_8k_events` carried the entire filing body per event under `text`,
commented "Full 8-K text (no truncation)". Measured live 2026-08-26:

    extract_8k_events("NVDA", limit=8)   73,716 chars, 50,507 of them `text`
    extract_8k_events("AMAT", limit=8)   47,938 chars, 34,264 of them `text`

Roughly 70% of every response is raw filing text, and at limit=8 the result
exceeds the tool-result budget outright -- a caller asking for eight events
receives nothing at all. The `total_events` / `events_returned` / `truncated`
contract added earlier bounds the *count*; nothing bounded the per-event
payload, so the count contract could be satisfied by a response too big to
deliver.

Head-truncating is not the fix on its own. An 8-K opens with its SEC cover
page -- registrant name, address, Rule 425 checkboxes, the 12(b) securities
table -- and on the AMAT 2026-03-13 filing the first `Item 5.07` heading sits
at character 2,313. A naive `text[:1500]` therefore keeps only boilerplate
and drops every item section, which is the one part of the filing that makes
the classification checkable. So the excerpt is anchored at the first item
heading, and the response says where it starts and how much it left behind.

The rule, the same one the window tools follow: a payload that cannot be
delivered whole must be bounded and the shortfall declared, never silently
cut.
"""
import importlib
import json

import pytest

from testing._gates import requires_sec

_utils = importlib.import_module("tools.web_search_server.8K_and_DEF14A_utils")


# A stand-in 8-K: the SEC cover page runs long before the first item heading,
# exactly as the real ones do, so a head-truncating bound fails these tests
# the way it fails a real filing.
_COVER = ("UNITED STATES\nSECURITIES AND EXCHANGE COMMISSION\nWashington, D. C. 20549\n"
          "FORM 8-K\nCURRENT REPORT\nPursuant to Section 13 or 15(d)\n"
          "Check the appropriate box below if the Form 8-K filing is intended to "
          "simultaneously satisfy the filing obligation of the registrant\n" * 40)
_BODY = ("Item 5.02 Departure of Directors or Certain Officers.\n"
         "On March 12, 2026 the Board appointed JANE ROE as Chief Financial Officer.\n" * 60)
_FILING_TEXT = _COVER + _BODY


def test_the_excerpt_is_bounded():
    """The whole point. An unbounded per-event field makes the response size a
    function of how verbose the filer was, which no caller can predict or
    plan a limit around."""
    out = _utils.bound_filing_text(_FILING_TEXT)
    assert len(out["text"]) <= _utils.EVENT_TEXT_CAP, (
        f"{len(out['text'])} chars returned against a cap of {_utils.EVENT_TEXT_CAP}")


def test_the_excerpt_keeps_the_item_section_not_the_cover_page():
    """`text[:cap]` on a real 8-K returns the registrant's mailing address and
    the Rule 425 checkboxes, and stops before the first item. That is the
    worst possible 1,500 characters to keep: it is identical across every
    filing a company ever makes, and it is not evidence of anything."""
    out = _utils.bound_filing_text(_FILING_TEXT)
    assert "Item 5.02" in out["text"], (
        "the excerpt dropped every item section and kept the cover page")
    assert out["text_excerpt_from_char"] == _FILING_TEXT.index("Item 5.02"), (
        f"excerpt should start at the first item heading, started at "
        f"{out['text_excerpt_from_char']}")


def test_truncation_is_declared_with_the_size_it_cut():
    """"Say so per event rather than silently cutting." A flag alone leaves
    the caller unable to judge whether the remainder was a paragraph or forty
    pages, so the full length travels with it."""
    out = _utils.bound_filing_text(_FILING_TEXT)
    assert out["text_truncated"] is True
    assert out["text_length_chars"] == len(_FILING_TEXT)
    assert out["text_chars_returned"] == len(out["text"])
    assert out["text_chars_returned"] < out["text_length_chars"]


def test_a_short_filing_is_not_marked_truncated():
    """The declaration must distinguish, or every event carries a warning
    that means nothing."""
    short = "Item 8.01 Other Events.\nThe company issued a press release."
    out = _utils.bound_filing_text(short)
    assert out["text"] == short
    assert out["text_truncated"] is False
    assert out["text_length_chars"] == out["text_chars_returned"] == len(short)
    assert out["text_excerpt_from_char"] == 0


def test_an_empty_body_reports_zero_rather_than_a_truncation():
    """Some filings come back with no extractable text at all -- AMAT's
    2025-10-23 8-K did, live. That is a fetch that returned nothing, not a
    filing we shortened, and `text_length_chars: 0` is what makes it visible
    instead of reading as a very brief 8-K."""
    out = _utils.bound_filing_text("")
    assert out["text_length_chars"] == 0
    assert out["text_truncated"] is False


# ---------------------------------------------------------------------------
# The tool itself
# ---------------------------------------------------------------------------

@requires_sec
@pytest.mark.parametrize("ticker", ["NVDA"])
def test_live_eight_events_fit_in_a_tool_result(ticker):
    """The reproduction. limit=8 must be answerable -- the count contract is
    worthless if the response it describes cannot be returned."""
    result = _utils.SECFilingParser().extract_8k_events(ticker, 8)
    if not result.get("success"):
        pytest.skip(f"EDGAR did not answer for {ticker}: {result.get('error')}")

    payload = json.dumps(result, default=str)
    assert len(payload) < 30_000, (
        f"{len(payload)} chars for limit=8, against 73,716 before the bound; "
        f"the per-event text field is unbounded again")

    events = result["events_by_date"]
    assert events, "no events parsed, so this proves nothing about the bound"
    for date_key, event in events.items():
        assert "text_truncated" in event, f"{date_key} does not declare truncation"
        assert len(event["text"]) <= _utils.EVENT_TEXT_CAP, date_key
        if event["text_truncated"]:
            assert event["text_length_chars"] > event["text_chars_returned"], date_key
            assert event["filing_url"], (
                f"{date_key} was truncated with no pointer to the rest of it")


@requires_sec
def test_live_the_count_contract_still_holds():
    """Regression: bounding the payload must not disturb the earlier fix that
    made `total_events` describe the company rather than the page."""
    result = _utils.SECFilingParser().extract_8k_events("NVDA", 3)
    if not result.get("success"):
        pytest.skip(f"EDGAR did not answer: {result.get('error')}")
    assert result["events_returned"] == len(result["events_by_date"]) == 3
    assert result["total_events"] > 3
    assert result["truncated"] is True
