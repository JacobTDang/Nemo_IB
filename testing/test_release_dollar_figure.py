"""A diluted phrase whose dollar figure the adjacency rule does not see.

Long-lag filers (usual release-to-10-Q lag of 7 days or more, measured as of
2026-09-01) refused 7 of 60 recent releases with "no dollar figure sits beside
it". Two wordings, from the releases themselves:

* Southern Missouri Bancorp (0001104659-26-085879): "$1.83 per fully diluted
  common share", and the same quarter as "Earnings per common share (diluted)
  were $1.83". The diluted phrase the reader did match was the full-year
  "$6.43", with "for fiscal year 2026" between the phrase and the verb.
* Greene County Bancorp (0001140361-26-016085): "$0.62 per basic and diluted
  share" for the quarter and "$1.74 per basic and diluted share" for the nine
  months. One figure is both bases. A basic figure alone is still not diluted.
"""
from pathlib import Path

import pytest

from research import release_eps

_FIXTURES = Path(__file__).parent / "release_excerpts"


def _fixture(name: str) -> str:
    return (_FIXTURES / name).read_text()


def test_per_fully_diluted_share_is_the_quarter_not_the_year():
    """0001104659-26-085879. "per fully diluted common share" is the quarter.
    The full-year diluted phrase in the same release is $6.43."""
    out = release_eps.extract_diluted_eps(
        _fixture("0001104659-26-085879_fully_diluted.txt"))

    assert out["eps"] == pytest.approx(1.83)
    assert out["eps"] != pytest.approx(6.43)
    adjusted = ("Adjusted net income for the fourth quarter was $2.10 per "
                "fully diluted common share. " + _fixture(
                    "0001104659-26-085879_fully_diluted.txt"))
    assert release_eps.extract_diluted_eps(adjusted)["eps"] == pytest.approx(1.83)


def test_earnings_per_common_share_diluted_in_parentheses_is_read():
    """0001104659-26-085879. The quarter figure follows "(diluted)"."""
    out = release_eps.extract_diluted_eps(
        _fixture("0001104659-26-085879_share_diluted.txt"))

    assert out["eps"] == pytest.approx(1.83)
    adjusted = ("Adjusted earnings per common share (diluted) were $2.10. "
                + _fixture("0001104659-26-085879_share_diluted.txt"))
    assert release_eps.extract_diluted_eps(adjusted)["eps"] == pytest.approx(1.83)


def test_one_figure_per_basic_and_diluted_share_is_the_quarter():
    """0001140361-26-016085. Basic and diluted are the same number. The
    quarter is $0.62; the nine months in the same sentence are $1.74. A
    basic figure that is not also diluted is still refused."""
    out = release_eps.extract_diluted_eps(
        _fixture("0001140361-26-016085_basic_and_diluted.txt"))

    assert out["eps"] == pytest.approx(0.62)
    assert out["eps"] != pytest.approx(1.74)
    basic_only = "Net income was $10.5 million, or $0.62 per basic share."
    assert release_eps.extract_diluted_eps(basic_only)["eps"] is None
    adjusted = ("Adjusted net income for the third quarter was $9.0 million, "
                "or $0.99 per basic and diluted share. " + _fixture(
                    "0001140361-26-016085_basic_and_diluted.txt"))
    assert release_eps.extract_diluted_eps(adjusted)["eps"] == pytest.approx(0.62)
