"""Debt maturity schedule.

`calculate_credit_profile` could tell you leverage was 3x but not when the debt
came due, and those are different companies. A wall next year is a refinancing
risk; a wall in 2031 is someone else's problem.

Coverage is genuinely partial. Measured live: MSFT, T, and AAPL tag all six
buckets; Ford and PLUG tag none, despite Ford being one of the largest debt
issuers in the market. This tool therefore reports what it found rather than
implying that silence means no maturities.
"""
import os

import pytest

from tools.web_search_server import debt_maturity as dm
from tools.web_search_server.sec_series import ConceptFact, FilingPoint, NotCovered

SKIP_NETWORK = os.environ.get("SKIP_NETWORK_TESTS") == "1"


def network(func):
  """Apply the real `network` marker plus the offline skip.

  This name used to be bound to a bare pytest.mark.skipif. A skipif is not
  a registered marker, so `-m network` and `-m "not network"` collected
  nothing here -- the tests were selectable only by file path.
  """
  func = pytest.mark.network(func)
  return pytest.mark.skipif(SKIP_NETWORK, reason="live EDGAR test")(func)


def _series(value):
    return [FilingPoint("2026-07-29", "10-K", "acc",
                        facts=[ConceptFact(value, "2026-06-30", {}, "c-1")])]


def _covering(mapping):
    """Fake fetch that returns a value only for the named concepts."""
    def fake(ticker, concept, **kwargs):
        if concept in mapping:
            return _series(mapping[concept])
        raise NotCovered(concept)
    return fake


def test_all_six_buckets_reports_full_coverage(monkeypatch):
    mapping = {c: 1.0e9 for bucket in dm.MATURITY_CONCEPTS.values() for c in bucket[:1]}
    monkeypatch.setattr(dm, "fetch_concept_series", _covering(mapping))
    result = dm.get_debt_maturity_schedule("MSFT")
    assert result["coverage"] == "full"
    assert result["buckets_found"] == 6
    assert result["total"] == pytest.approx(6.0e9)


def test_no_buckets_reports_not_covered_not_an_empty_schedule(monkeypatch):
    """Ford tags none of these concepts. An empty schedule would read as
    'no maturities', which for Ford would be spectacularly wrong."""
    monkeypatch.setattr(dm, "fetch_concept_series", _covering({}))
    result = dm.get_debt_maturity_schedule("F")
    assert result["coverage"] == "not_covered"
    assert result["success"] is False
    assert result["total"] is None
    assert result["concepts_tried"]


def test_some_buckets_reports_partial(monkeypatch):
    first = {bucket: options[0] for bucket, options in dm.MATURITY_CONCEPTS.items()}
    monkeypatch.setattr(dm, "fetch_concept_series",
                        _covering({first["year_1"]: 5.0e9, first["year_2"]: 3.0e9}))
    result = dm.get_debt_maturity_schedule("PARTIAL")
    assert result["coverage"] == "partial"
    assert result["buckets_found"] == 2
    assert result["by_year"]["year_3"] is None


def test_a_tagged_zero_is_kept_not_treated_as_missing(monkeypatch):
    """MSFT genuinely reports 0 maturing in year two. Zero is falsy in Python,
    so a truthiness check would misreport a real disclosure as untagged."""
    first = {bucket: options[0] for bucket, options in dm.MATURITY_CONCEPTS.items()}
    monkeypatch.setattr(dm, "fetch_concept_series",
                        _covering({first["year_1"]: 9.25e9, first["year_2"]: 0.0}))
    result = dm.get_debt_maturity_schedule("MSFT")
    assert result["by_year"]["year_2"] == 0.0
    assert result["by_year"]["year_2"] is not None
    assert result["buckets_found"] == 2


def test_total_sums_only_the_buckets_that_were_found(monkeypatch):
    first = {bucket: options[0] for bucket, options in dm.MATURITY_CONCEPTS.items()}
    monkeypatch.setattr(dm, "fetch_concept_series",
                        _covering({first["year_1"]: 5.0e9, first["after_year_5"]: 20.0e9}))
    result = dm.get_debt_maturity_schedule("SPARSE")
    assert result["total"] == pytest.approx(25.0e9)


def test_fallback_concept_is_tried_when_the_primary_is_absent(monkeypatch):
    """Filers use either the fixed-year or the rolling-year concept family."""
    rolling = dm.MATURITY_CONCEPTS["year_1"][1]
    monkeypatch.setattr(dm, "fetch_concept_series", _covering({rolling: 7.0e9}))
    result = dm.get_debt_maturity_schedule("ROLLING")
    assert result["by_year"]["year_1"] == 7.0e9


def test_near_term_wall_is_flagged(monkeypatch):
    """The point of the tool: what share comes due inside twelve months."""
    first = {bucket: options[0] for bucket, options in dm.MATURITY_CONCEPTS.items()}
    monkeypatch.setattr(dm, "fetch_concept_series", _covering({
        first["year_1"]: 60.0e9, first["after_year_5"]: 40.0e9}))
    result = dm.get_debt_maturity_schedule("WALL")
    assert result["pct_due_within_one_year"] == pytest.approx(60.0)


# ------------------------------------------------------------- live golden set

@pytest.fixture(scope="module", autouse=True)
def _load_env():
    from dotenv import load_dotenv
    load_dotenv()


@network
def test_msft_schedule_is_fully_covered_and_back_loaded():
    result = dm.get_debt_maturity_schedule("MSFT")
    assert result["coverage"] == "full"
    assert result["by_year"]["after_year_5"] > result["by_year"]["year_1"], (
        "MSFT's debt is long-dated; a front-loaded result means the buckets "
        "were mapped wrong")


@network
def test_att_schedule_is_fully_covered():
    result = dm.get_debt_maturity_schedule("T")
    assert result["coverage"] == "full"
    assert result["total"] > 100e9


@network
def test_ford_is_reported_as_uncovered_rather_than_debt_free():
    """Ford carries enormous debt but does not tag these concepts. The tool
    must say so instead of returning an empty, reassuring schedule."""
    result = dm.get_debt_maturity_schedule("F")
    assert result["coverage"] == "not_covered"
    assert result["total"] is None
