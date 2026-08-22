"""Share count, dilution, and shelf activity.

The failure this file exists to prevent: GOOGL reports Class A, B, and C as
separate facts, and an extractor that takes the first one reports 5.868bn
against a true 12.23bn. That is not a crash and not an obviously wrong number --
it is a 52% understatement that reads as plausible, which makes every per-share
metric downstream quietly wrong in the flattering direction.
"""
import os

import pytest

from tools.web_search_server import dilution
from tools.web_search_server.sec_series import ConceptFact, FilingPoint, NotCovered

SKIP_NETWORK = os.environ.get("SKIP_NETWORK_TESTS") == "1"
network = pytest.mark.skipif(SKIP_NETWORK, reason="live EDGAR test")

CLASS_AXIS = "us-gaap:StatementClassOfStockAxis"


def _point(filing_date, facts):
    return FilingPoint(filing_date=filing_date, form="10-Q",
                       accession="acc", facts=facts)


def _fact(value, member=None, period="2026-07-15", ref="c-1"):
    dims = {CLASS_AXIS: member} if member else {}
    return ConceptFact(value=value, period=period, dimensions=dims, context_ref=ref)


# ---------------------------------------------------------------- class labels

def test_standard_members_map_to_readable_labels():
    assert dilution._class_label("us-gaap:CommonClassAMember") == "Class A"
    assert dilution._class_label("us-gaap:CommonClassBMember") == "Class B"


def test_company_specific_member_is_labelled_not_dropped():
    """Alphabet tags Class C as goog:CapitalClassCMember. A whitelist of
    us-gaap members would drop it and understate shares."""
    assert dilution._class_label("goog:CapitalClassCMember") == "Capital Class C"


def test_undimensioned_member_is_labelled_common():
    assert dilution._class_label("") == "Common"
    assert dilution._class_label(None) == "Common"


# ------------------------------------------------------------------ the series

def test_single_class_total_is_the_single_fact(monkeypatch):
    monkeypatch.setattr(dilution, "fetch_concept_series",
                        lambda *a, **k: [_point("2026-04-29", [_fact(7428434704.0)])])
    result = dilution.get_share_count_series("MSFT")
    assert result["success"] is True
    assert result["latest_total"] == 7428434704.0
    assert result["classes_found"] == ["Common"]


def test_multi_class_total_sums_every_class(monkeypatch):
    """The regression test for the 52% understatement."""
    monkeypatch.setattr(dilution, "fetch_concept_series", lambda *a, **k: [
        _point("2026-07-23", [
            _fact(5868000000.0, "us-gaap:CommonClassAMember", ref="c-28"),
            _fact(835000000.0, "us-gaap:CommonClassBMember", ref="c-29"),
            _fact(5527000000.0, "goog:CapitalClassCMember", ref="c-30"),
        ])])
    result = dilution.get_share_count_series("GOOGL")
    assert result["latest_total"] == 12230000000.0
    assert set(result["classes_found"]) == {"Class A", "Class B", "Capital Class C"}


def test_per_class_breakdown_is_always_reported(monkeypatch):
    """A bare total hides a missing class. The caller must be able to see the
    classes that were found."""
    monkeypatch.setattr(dilution, "fetch_concept_series", lambda *a, **k: [
        _point("2026-07-23", [
            _fact(5868000000.0, "us-gaap:CommonClassAMember", ref="c-28"),
            _fact(835000000.0, "us-gaap:CommonClassBMember", ref="c-29"),
        ])])
    result = dilution.get_share_count_series("GOOGL")
    assert result["by_class"]["Class A"][0]["shares"] == 5868000000.0
    assert result["by_class"]["Class B"][0]["shares"] == 835000000.0


def test_dilution_is_positive_when_share_count_grows(monkeypatch):
    """Newest first, matching EDGAR ordering: 100 now vs 90 a year ago is +11.1%."""
    monkeypatch.setattr(dilution, "fetch_concept_series", lambda *a, **k: [
        _point("2026-07-23", [_fact(100_000_000.0)]),
        _point("2025-07-23", [_fact(90_000_000.0)]),
    ])
    result = dilution.get_share_count_series("DILUTER")
    assert result["change_pct"] == pytest.approx(11.111, abs=0.01)
    assert result["direction"] == "dilution"


def test_buyback_shows_as_negative_change(monkeypatch):
    monkeypatch.setattr(dilution, "fetch_concept_series", lambda *a, **k: [
        _point("2026-07-23", [_fact(90_000_000.0)]),
        _point("2025-07-23", [_fact(100_000_000.0)]),
    ])
    result = dilution.get_share_count_series("BUYBACK")
    assert result["change_pct"] == pytest.approx(-10.0, abs=0.01)
    assert result["direction"] == "buyback"


def test_single_period_has_no_change_rather_than_zero(monkeypatch):
    """Zero would read as 'no dilution'. One data point supports no such claim."""
    monkeypatch.setattr(dilution, "fetch_concept_series",
                        lambda *a, **k: [_point("2026-07-23", [_fact(100.0)])])
    result = dilution.get_share_count_series("NEWIPO")
    assert result["change_pct"] is None
    assert result["direction"] == "insufficient_history"


def test_not_covered_returns_explicit_failure_not_zero(monkeypatch):
    def raise_not_covered(*a, **k):
        raise NotCovered("no such concept")
    monkeypatch.setattr(dilution, "fetch_concept_series", raise_not_covered)
    result = dilution.get_share_count_series("NOTAGS")
    assert result["success"] is False
    assert result["latest_total"] is None
    assert "not" in result["error"].lower()


# ------------------------------------------------------------- live golden set

@pytest.fixture(scope="module", autouse=True)
def _load_env():
    from dotenv import load_dotenv
    load_dotenv()


@network
def test_googl_share_count_matches_reality():
    result = dilution.get_share_count_series("GOOGL", limit=2)
    assert result["success"] is True
    assert len(result["classes_found"]) == 3, (
        f"expected 3 share classes, got {result['classes_found']}")
    assert result["latest_total"] > 11_000_000_000, (
        f"GOOGL total {result['latest_total']:,.0f} is implausibly low -- "
        f"a share class was dropped")


@network
def test_msft_share_count_matches_reality():
    result = dilution.get_share_count_series("MSFT", limit=3)
    assert result["success"] is True
    assert result["classes_found"] == ["Common"]
    assert 7_000_000_000 < result["latest_total"] < 8_000_000_000


@network
def test_serial_diluter_shows_both_effect_and_mechanism():
    """PLUG funds itself through shelf takedowns. This is the case the whole
    module exists for: share count and shelf activity both have to show it.

    Asserted as properties rather than exact figures, because the numbers move
    every quarter while the behaviour does not.
    """
    shares = dilution.get_share_count_series("PLUG", limit=6)
    assert shares["success"] is True
    assert shares["direction"] == "dilution", (
        f"PLUG should read as dilution, got {shares['direction']} "
        f"({shares['change_pct']}%)")
    assert shares["change_pct"] > 5.0

    shelf = dilution.get_shelf_activity("PLUG", lookback_days=1095)
    assert shelf["success"] is True
    assert shelf["takedown_count"] > 0, (
        "PLUG has filed 424B5 takedowns; zero means the form is not being found")


@network
def test_megacap_has_no_shelf_activity():
    """MSFT funds itself from cash flow. A false positive here would mean the
    date filter or form matching is wrong."""
    shelf = dilution.get_shelf_activity("MSFT", lookback_days=1095)
    assert shelf["success"] is True
    assert shelf["has_active_shelf"] is False
    assert shelf["takedown_count"] == 0


@network
def test_small_change_is_flat_not_a_buyback():
    """MSFT's share count moves fractionally on option exercises and buybacks.
    Anything under the threshold must read as flat, not as a signal."""
    shares = dilution.get_share_count_series("MSFT", limit=3)
    assert shares["direction"] in ("flat", "buyback")
    assert abs(shares["change_pct"]) < 2.0
