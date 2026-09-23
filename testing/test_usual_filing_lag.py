"""How long an issuer usually takes to file its 10-Q after the earnings release.

The release-timed replay's gain over XBRL timing sat almost entirely in prints
whose 10-Q arrived a week or more after the release (issue #115). The current
quarter's lag is unknown when its release lands. The issuer's lag in earlier
quarters is known, so it can be a point-in-time screen -- provided nothing
from after the decision date leaks into it.

These tests pin the measurement only. The screen built on it, and its
threshold, are fixed in issue #115 before any held-out data is examined.
"""
import pytest

from research import release_eps

# Five quarters of one filer: release date and 10-Q filing date per period.
RELEASES = {
    "2025Q1": "2025-04-24", "2025Q2": "2025-07-24", "2025Q3": "2025-10-23",
    "2025Q4": "2026-02-05", "2026Q1": "2026-04-23",
}
FILINGS = {
    "2025Q1": "2025-05-08",  # 14 days
    "2025Q2": "2025-07-25",  # 1
    "2025Q3": "2025-11-06",  # 14
    "2025Q4": "2026-02-26",  # 21
    "2026Q1": "2026-05-07",  # 14
}


def test_the_lag_is_calendar_days_from_release_to_filing():
    rows = release_eps.filing_lags(RELEASES, FILINGS)

    assert [r["lag_days"] for r in rows] == [14, 1, 14, 21, 14]
    assert [r["fiscal_period"] for r in rows] == [
        "2025Q1", "2025Q2", "2025Q3", "2025Q4", "2026Q1"]


def test_the_usual_lag_is_the_median_of_the_last_four_filed():
    out = release_eps.usual_filing_lag(RELEASES, FILINGS, as_of="2026-06-30")

    assert out["lag_days"] == pytest.approx(14.0)
    assert [q["fiscal_period"] for q in out["quarters"]] == [
        "2025Q2", "2025Q3", "2025Q4", "2026Q1"]


def test_a_quarter_filed_after_the_decision_date_is_not_seen():
    """On 2026-05-01 the 2026Q1 release is public and its 10-Q is not. Its lag
    does not exist yet, whatever the store holds today."""
    out = release_eps.usual_filing_lag(RELEASES, FILINGS, as_of="2026-05-01")

    assert [q["fiscal_period"] for q in out["quarters"]] == [
        "2025Q1", "2025Q2", "2025Q3", "2025Q4"]
    assert out["lag_days"] == pytest.approx(14.0)


def test_too_few_quarters_is_a_refusal_not_a_guess():
    out = release_eps.usual_filing_lag(RELEASES, FILINGS, as_of="2025-12-31")

    assert out["lag_days"] is None
    assert "3" in out["reason"] and "4" in out["reason"]


def test_a_quarter_without_a_release_is_left_out_not_counted_as_zero():
    releases = {p: d for p, d in RELEASES.items() if p != "2025Q3"}

    rows = release_eps.filing_lags(releases, FILINGS)

    assert "2025Q3" not in [r["fiscal_period"] for r in rows]
    assert 0 not in [r["lag_days"] for r in rows]


def test_the_live_helper_asks_both_sources_as_of_the_decision_date(monkeypatch):
    asked = {}

    def eps_series(ticker, as_of=None):
        asked["series"] = as_of
        return {"success": True, "error": None, "quarters": [
            {"fiscal_period": p, "period_end": None, "known_at": d}
            for p, d in FILINGS.items()]}

    def for_quarters(ticker, as_of=None, quarters=None):
        asked["releases"] = as_of
        return {p: {"announced_date": d, "accession": f"acc-{p}"}
                for p, d in RELEASES.items()}

    from research import announcements
    monkeypatch.setattr(release_eps.sue, "eps_series", eps_series)
    monkeypatch.setattr(announcements, "for_quarters", for_quarters)

    out = release_eps.usual_filing_lag_for("ACME", as_of="2026-06-30")

    assert asked == {"series": "2026-06-30", "releases": "2026-06-30"}
    assert out["lag_days"] == pytest.approx(14.0)
    assert out["ticker"] == "ACME"


def test_a_series_that_failed_is_a_refusal_with_its_cause(monkeypatch):
    monkeypatch.setattr(release_eps.sue, "eps_series", lambda t, as_of=None: {
        "success": False, "error": "SEC_EMAIL is not set", "quarters": []})

    out = release_eps.usual_filing_lag_for("ACME", as_of="2026-06-30")

    assert out["lag_days"] is None
    assert "SEC_EMAIL" in out["reason"]
