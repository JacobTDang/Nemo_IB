"""A multi-year disclosure means the latest year, not the first one printed.

Foreign filers state concentration as one sentence covering three years. TSMC's
20-F:

    "Our largest customer in 2023, 2024 and 2025 accounted for 25%, 22% and
     19% of our net revenue in the respective year."
    "Our second largest customer in 2023, 2024 and 2025 accounted for 11%, 12%,
     and 17% of our net revenue in the respective year."

The extractor takes the first percentage it sees, so it reported 25% and 11%
-- both 2023 figures -- with `fiscal_year: null`.

For the largest customer that is stale. For the second it is worse than stale:
the real series runs 11 -> 12 -> 17, and reporting 11 inverts the trend. That
customer's rise is precisely the AI read-through signal someone would come to
this tool for, and the tool says it is the smallest it has ever been.

`fiscal_year: null` means nothing downstream can catch it. And because each row
carries a different year's number, `periods.total_pct` was summing 2023's
largest to 2023's second to a third value entirely -- 46.0 for a filer whose
2025 top two are 19 + 17 = 36.

The third value was not a customer at all. It came from a table header:

    "Major customers representing at least 10% of net revenue"

Ten percent there is the disclosure THRESHOLD -- the rule for which customers
must be named. It is not anybody's share, and it must not be a row.
"""
import pytest

import tools.web_search_server.sec_utils as su


def _extract(text, monkeypatch, form_type="20-F"):
    class _Filing:
        filing_date = "2026-04-16"

        def text(self):
            return text

    monkeypatch.setattr(su, "_require_identity", lambda: "t@example.invalid")
    monkeypatch.setattr(su, "get_latest_filing",
                        lambda *a, **k: {"filing_object": _Filing(),
                                         "filing_date": "2026-04-16"})
    return su.extract_customer_concentration("TSM", form_type=form_type)


LARGEST = ("Our largest customer in 2023, 2024 and 2025 accounted for 25%, "
           "22% and 19% of our net revenue in the respective year.")
SECOND = ("Our second largest customer in 2023, 2024 and 2025 accounted for "
          "11%, 12%, and 17% of our net revenue in the respective year.")
THRESHOLD = ("2) Major customers representing at least 10% of net revenue "
             "Years Ended December 31 2023 2024 2025")


def _rows(result):
    return result.get("customer_disclosures") or []


def test_the_latest_year_is_reported_not_the_first(monkeypatch):
    rows = _rows(_extract(LARGEST, monkeypatch))
    assert rows, "the disclosure was dropped"
    assert rows[0]["pct_of_revenue"] == 19.0, (
        f"reported {rows[0]['pct_of_revenue']}% -- the 2023 figure -- for a "
        f"customer that is 19% in 2025")
    assert rows[0]["fiscal_year"] == 2025


def test_a_rising_customer_is_not_reported_as_its_smallest(monkeypatch):
    """11 -> 12 -> 17 must not read as 11."""
    rows = _rows(_extract(SECOND, monkeypatch))
    assert rows[0]["pct_of_revenue"] == 17.0, (
        f"a customer rising 11->17 was reported at "
        f"{rows[0]['pct_of_revenue']}%, inverting the trend")


def test_the_earlier_years_are_kept_not_discarded(monkeypatch):
    """The trend is the point. Losing it is why the stale figure was wrong."""
    rows = _rows(_extract(SECOND, monkeypatch))
    history = rows[0].get("by_year") or {}
    assert history.get(2023) == 11.0
    assert history.get(2024) == 12.0
    assert history.get(2025) == 17.0


def test_a_disclosure_threshold_is_not_a_customer(monkeypatch):
    rows = _rows(_extract(THRESHOLD, monkeypatch))
    assert not rows, (
        f"the 10% naming threshold was recorded as a customer's share: {rows}")


def test_the_period_total_uses_one_year(monkeypatch):
    """46.0 came from adding three different years together."""
    result = _extract(LARGEST + " " + SECOND + " " + THRESHOLD, monkeypatch)
    totals = {p["fiscal_year"]: p["total_pct"] for p in result["periods"]}
    assert totals.get(2025) == pytest.approx(36.0), (
        f"2025's top two are 19 + 17 = 36; got {totals}")


def test_a_single_year_disclosure_is_unaffected(monkeypatch):
    """NVDA's shape must keep working."""
    text = ("For fiscal year 2026, sales to one direct customer represented "
            "22% of total revenue.")
    rows = _rows(_extract(text, monkeypatch, form_type="10-K"))
    assert rows[0]["pct_of_revenue"] == 22.0
    assert rows[0]["fiscal_year"] == 2026
