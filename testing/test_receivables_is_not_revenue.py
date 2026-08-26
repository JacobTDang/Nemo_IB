"""A share of receivables is not a share of revenue.

AVGO's 10-K discloses both, one paragraph apart:

    "Direct sales to one semiconductor solutions customer ... accounted for
     32% and 28% of our net revenue for fiscal years 2025 and 2024"

    "One customer accounted for 44% and 18% of our net accounts receivable
     balance as of November 2, 2025 and November 3, 2024"

Both were extracted into `pct_of_revenue`, so the tool reported a customer at
44% of revenue when the filing says 32%. The larger, wronger number sorts
first and is the one an analyst reads.

They are different measures of different things. A receivables concentration
is a real disclosure worth keeping -- it is a credit-risk signal, and a gap
between 32% of revenue and 44% of receivables says that customer is paying
slower than the rest. So the row stays and gets labelled, rather than being
dropped or silently counted as revenue.

`total_pct` is a revenue check -- it warns when disclosed shares exceed 100%
of revenue -- so a receivables row must not be summed into it.
"""
import pytest

from tools.web_search_server import sec_utils as su


def _rows(text):
    return su._extract_customer_rows(text) if hasattr(su, "_extract_customer_rows") \
        else None


AVGO_REVENUE = ("Direct sales to one semiconductor solutions customer, which "
                "is a distributor, accounted for 32% and 28% of our net "
                "revenue for fiscal years 2025 and 2024, respectively.")
AVGO_RECEIVABLES = ("One customer accounted for 44% and 18% of our net "
                    "accounts receivable balance as of November 2, 2025 and "
                    "November 3, 2024, respectively.")


def _extract(text, monkeypatch):
    """Drive extract_customer_concentration over supplied filing text."""
    class _Filing:
        filing_date = "2025-12-18"

        def text(self):
            return text

    monkeypatch.setattr(su, "_require_identity", lambda: "t@example.invalid",
                        raising=False)
    monkeypatch.setattr(su, "get_latest_filing",
                        lambda *a, **k: {"filing_object": _Filing(),
                                         "filing_date": "2025-12-18"})
    return su.extract_customer_concentration("AVGO")


def test_a_receivables_share_is_not_reported_as_revenue(monkeypatch):
    result = _extract(AVGO_RECEIVABLES, monkeypatch)
    rows = result.get("customer_disclosures") or []
    assert rows, "the receivables disclosure was dropped entirely"
    for row in rows:
        assert row.get("pct_of_revenue") != 44.0, (
            "44% of accounts receivable was reported as 44% of revenue")


def test_the_receivables_row_is_kept_and_labelled(monkeypatch):
    result = _extract(AVGO_RECEIVABLES, monkeypatch)
    rows = result.get("customer_disclosures") or []
    bases = {r.get("basis") for r in rows}
    assert "accounts_receivable" in bases, (
        f"the row carries no basis saying what it measures: {bases}")


def test_a_revenue_share_is_still_revenue(monkeypatch):
    result = _extract(AVGO_REVENUE, monkeypatch)
    rows = result.get("customer_disclosures") or []
    assert any(r.get("pct_of_revenue") == 32.0 and r.get("basis") == "revenue"
               for r in rows), f"the revenue disclosure was lost: {rows}"


def test_receivables_do_not_inflate_the_revenue_total(monkeypatch):
    """total_pct exists to warn when shares exceed 100% of revenue. A
    receivables row summed into it makes that warning fire on a filing that
    never over-disclosed."""
    result = _extract(AVGO_REVENUE + " " + AVGO_RECEIVABLES, monkeypatch)
    for period in result.get("periods") or []:
        assert period["total_pct"] <= 32.0 + 0.01, (
            f"a receivables share was counted toward revenue: {period}")


# --- "top five customers" is not one customer -------------------------------
#
# AVGO's fiscal 2025 rows summed to 144% of revenue, which fired the
# impossible-total warning. The cause is visible in the disclosures: an
# aggregate and a single customer sitting in the same list.
#
#   "aggregate sales to our top five end customers ... approximately 40% of
#    our net revenue"
#   "Direct sales to one semiconductor solutions customer ... 32% ... of our
#    net revenue"
#
# Both are true. Adding them is not. A caller iterating the rows reads the 40%
# as one buyer, which is the same defect as the receivables row above: a
# number in a field that names something it is not.

AVGO_AGGREGATE = ("We believe aggregate sales to our top five end customers, "
                  "through all channels, accounted for approximately 40% of "
                  "our net revenue for fiscal year 2025.")


def test_an_aggregate_of_several_customers_is_labelled(monkeypatch):
    result = _extract(AVGO_AGGREGATE, monkeypatch)
    rows = result.get("customer_disclosures") or []
    assert rows, "the aggregate disclosure was dropped"
    assert all(r.get("scope") == "aggregate" for r in rows), (
        f"a five-customer total is presented as one customer: {rows}")


def test_a_single_customer_keeps_its_scope(monkeypatch):
    result = _extract(AVGO_REVENUE, monkeypatch)
    rows = result.get("customer_disclosures") or []
    assert any(r.get("scope") == "single_customer" for r in rows)


def test_an_aggregate_is_not_summed_with_the_customers_inside_it(monkeypatch):
    """40% across five customers plus 32% for one of them is not 72%."""
    result = _extract(AVGO_AGGREGATE + " " + AVGO_REVENUE, monkeypatch)
    for period in result.get("periods") or []:
        if period["fiscal_year"] == 2025:
            assert period["total_pct"] <= 32.0 + 0.01, (
                f"an aggregate was added to a customer inside it: {period}")


# --- an ambiguity the tool must not resolve on its own ----------------------
#
# AVGO discloses the same customer in two places, worded differently:
#
#   "Direct sales to one semiconductor solutions customer, which is a
#    distributor, accounted for 32% and 28% ... fiscal years 2025 and 2024"
#   "During fiscal years 2025, 2024 and 2023, one customer accounted for 32%,
#    28% and 21% ..."
#
# Same share, same year, neither named. Almost certainly one customer stated
# twice -- but NVDA's 10-K genuinely reports two different direct customers at
# the same percentage, so merging on an equal share would erase a real one.
#
# Excluding aggregates from the total dropped AVGO from 144% to 64% and with
# it the impossible-total warning, which had been the only thing telling the
# caller the rows were double-counted. 64% is still wrong -- the filing says
# 32% -- and now says so quietly. The tool cannot tell which case it is
# looking at, so it says that rather than picking one.


def test_two_unnamed_rows_with_the_same_share_and_year_are_flagged(monkeypatch):
    both = (AVGO_REVENUE + " During fiscal years 2025, 2024 and 2023, one "
            "customer accounted for 32%, 28% and 21% of our net revenue, "
            "respectively.")
    result = _extract(both, monkeypatch)

    codes = [w.get("code") for w in (result.get("warnings") or [])]
    assert "possible_duplicate_disclosure" in codes, (
        f"the same share twice in one year passed without comment: {codes}")


def test_distinct_shares_are_not_flagged(monkeypatch):
    """NVDA's 22% and 14% are two customers and must stay silent."""
    text = ("For fiscal year 2026, sales to one direct customer represented "
            "22% of total revenue and sales to another direct customer "
            "represented 14% of total revenue.")
    result = _extract(text, monkeypatch)
    codes = [w.get("code") for w in (result.get("warnings") or [])]
    assert "possible_duplicate_disclosure" not in codes
