"""Litigation (Item 3) and customer concentration.

Both had zero coverage. Litigation is a numbered item and reuses the existing
item-extraction approach. Customer concentration is not an item at all -- it
appears in Item 1, Item 1A, or the concentration-of-credit-risk footnote -- so
it is found by disclosure language rather than by heading.

The parsing logic is unit-tested against synthetic filing text so the assertions
are deterministic. Live tests confirm the patterns survive contact with real
filings.
"""
import os

import pytest

from tools.web_search_server import sec_utils

SKIP_NETWORK = os.environ.get("SKIP_NETWORK_TESTS") == "1"


def network(func):
  """Apply the real `network` marker plus the offline skip.

  This name used to be bound to a bare pytest.mark.skipif. A skipif is not
  a registered marker, so `-m network` and `-m "not network"` collected
  nothing here -- the tests were selectable only by file path.
  """
  func = pytest.mark.network(func)
  return pytest.mark.skipif(SKIP_NETWORK, reason="live EDGAR test")(func)


# ------------------------------------------------------------ item boundaries

FILING = (
    "TABLE OF CONTENTS\n"
    "ITEM 3. LEGAL PROCEEDINGS .......... 24\n"
    + ("filler " * 6000) +
    "ITEM 3. LEGAL PROCEEDINGS\n"
    "We are party to various legal proceedings. In March 2025 a class action "
    "was filed in the Northern District of California alleging patent "
    "infringement. We believe the claims are without merit.\n"
    "ITEM 4. MINE SAFETY DISCLOSURES\n"
    "Not applicable.\n"
)


def test_item_section_skips_the_table_of_contents_entry():
    """The heading appears twice: once in the TOC and once in the body. Taking
    the first match returns a page number instead of the disclosure."""
    section = sec_utils._locate_item_section(
        FILING, r"ITEM\s+3\.?\s+LEGAL\s+PROCEEDINGS", r"ITEM\s+4\b")
    assert "class action" in section
    assert "..........." not in section


def test_item_section_stops_at_the_next_item():
    section = sec_utils._locate_item_section(
        FILING, r"ITEM\s+3\.?\s+LEGAL\s+PROCEEDINGS", r"ITEM\s+4\b")
    assert "MINE SAFETY" not in section


def test_item_section_returns_none_when_absent():
    section = sec_utils._locate_item_section(
        "nothing here at all", r"ITEM\s+3\b", r"ITEM\s+4\b")
    assert section is None


# ------------------------------------------------- customer concentration text

def test_named_customer_with_percentage_is_captured():
    text = ("Concentration of Credit Risk. One customer, Acme Corporation, "
            "accounted for approximately 19% of total revenue in fiscal 2026.")
    found = sec_utils._scan_customer_concentration(text)
    assert found["has_concentration"] is True
    assert any(c["pct_of_revenue"] == 19.0 for c in found["named_customers"])


def test_unnamed_customer_percentage_is_still_captured():
    """Most filers say 'one customer' without naming it. A null name must not
    discard the percentage, which is the part that matters."""
    text = ("No single customer accounted for more than 10% of revenue, "
            "except one customer which represented 12% of net revenue.")
    found = sec_utils._scan_customer_concentration(text)
    assert found["has_concentration"] is True
    assert any(c["pct_of_revenue"] == 12.0 for c in found["named_customers"])


def test_multiple_customers_are_all_captured():
    text = ("Customer A accounted for 22% of revenue and Customer B accounted "
            "for 15% of revenue during the year.")
    found = sec_utils._scan_customer_concentration(text)
    pcts = sorted(c["pct_of_revenue"] for c in found["named_customers"])
    assert pcts == [15.0, 22.0]


def test_explicit_absence_of_concentration_is_reported_as_such():
    """'No customer exceeded 10%' is a real disclosure meaning low
    concentration. It must not be reported as 'no data'."""
    text = ("No single customer accounted for 10% or more of our total "
            "revenue in any period presented.")
    found = sec_utils._scan_customer_concentration(text)
    assert found["has_concentration"] is False
    assert found["explicitly_none"] is True


def test_unrelated_percentages_are_not_mistaken_for_concentration():
    """Gross margin percentages must not be scraped as customer shares."""
    text = "Gross margin was 45% compared with 43% in the prior year."
    found = sec_utils._scan_customer_concentration(text)
    assert found["named_customers"] == []
    assert found["has_concentration"] is False


# ------------------------------------------------------------- live golden set

@pytest.fixture(scope="module", autouse=True)
def _load_env():
    from dotenv import load_dotenv
    load_dotenv()


@network
def test_litigation_extracts_from_a_real_filing():
    result = sec_utils.extract_litigation("MSFT", "10-K")
    assert result["success"] is True, result.get("error")
    assert len(result["text"]) > 200
    lowered = result["text"].lower()
    assert "legal" in lowered or "proceeding" in lowered


@network
def test_customer_concentration_on_a_concentrated_filer():
    """NVDA discloses direct-customer concentration. Whether it finds a figure
    or an explicit denial, it must not fail."""
    result = sec_utils.extract_customer_concentration("NVDA", "10-K")
    assert result["success"] is True, result.get("error")
    assert "has_concentration" in result


# --------------------------------------------------------------------------
# Regressions found by running against real filings. MSFT was reported as
# having 10% customer concentration when its 10-K says the opposite.
# --------------------------------------------------------------------------

def test_no_sales_to_an_individual_customer_is_a_denial():
    """MSFT's actual phrasing. The denial regex originally required
    'no single customer', so this read as a positive 10% disclosure -- the
    exact inversion of what the filing says."""
    text = ("No sales to an individual customer or country other than the "
            "United States accounted for more than 10% of revenue.")
    found = sec_utils._scan_customer_concentration(text)
    assert found["has_concentration"] is False
    assert found["explicitly_none"] is True


def test_no_customer_represented_more_than_is_a_denial():
    text = "No customer represented more than 10% of net revenue in fiscal 2026."
    found = sec_utils._scan_customer_concentration(text)
    assert found["has_concentration"] is False


def test_sentence_fragments_are_not_captured_as_customer_names():
    """'or country other than the United States' was being returned as a
    customer name. A name whose words are not all capitalised is a fragment."""
    text = ("One customer or country other than domestic accounted for 15% of "
            "revenue.")
    found = sec_utils._scan_customer_concentration(text)
    for row in found["named_customers"]:
        assert row["name"] is None, f"captured fragment as a name: {row['name']!r}"


def test_a_real_proper_noun_is_still_captured_as_a_name():
    text = ("One customer, Acme Holdings Inc, accounted for 19% of revenue.")
    found = sec_utils._scan_customer_concentration(text)
    assert any(r["name"] == "Acme Holdings Inc" for r in found["named_customers"])


def test_nvda_style_disclosure_still_captured_after_the_fix():
    """The true positive must survive tightening the denial patterns."""
    text = ("For fiscal year 2026, sales to one direct customer represented 22% "
            "of total revenue and sales to another direct customer represented "
            "14% of total revenue.")
    found = sec_utils._scan_customer_concentration(text)
    pcts = sorted(r["pct_of_revenue"] for r in found["named_customers"])
    assert pcts == [14.0, 22.0]
    assert found["has_concentration"] is True


def test_customer_concentration_failure_keeps_the_same_shape(monkeypatch):
    """ARM's filing could not be fetched and the error path returned a dict
    without has_concentration, so a caller reading the documented field got a
    KeyError instead of an answer. Failure must not change the contract."""
    monkeypatch.setattr(sec_utils, "get_latest_filing", lambda *a, **k: None)
    result = sec_utils.extract_customer_concentration("ARM", "10-K")
    assert result["success"] is False
    for key in ("has_concentration", "explicitly_none", "named_customers"):
        assert key in result, f"failure path dropped {key!r}"
    assert result["has_concentration"] is False
    assert result["named_customers"] == []
