"""Unit tests for 8-K earnings release misread and refusal fixes.

Each test is pinned to a real release excerpt fixture in testing/release_excerpts/
and guards exactly one extraction or refusal rule.
"""
from pathlib import Path

import pytest

from research import release_eps

_FIXTURES = Path(__file__).parent / "release_excerpts"


def _fixture(name: str) -> str:
    return (_FIXTURES / name).read_text().strip()


def test_core_eps_is_refused_as_non_gaap():
    """0001628280-22-001183. "Core EPS was $1.04" is non-GAAP.

    With only non-GAAP Core EPS figures, the release must be refused, not
    read as GAAP."""
    text = _fixture("0001628280-22-001183_core_eps.txt")
    out = release_eps.extract_diluted_eps(text)

    assert out["eps"] is None
    assert "non-GAAP or adjusted" in out["reason"]


def test_diluted_earnings_per_share_with_parenthetical_symbol_is_read():
    """0000811589-25-000072. "Diluted earnings per share ("D-EPS") was $0.49".

    A parenthetical shorthand like ("D-EPS") adjacent to the diluted phrase
    does not break the adjacency rule."""
    text = _fixture("0000811589-25-000072_d_eps_symbol.txt")
    out = release_eps.extract_diluted_eps(text)

    assert out["eps"] == pytest.approx(0.49)


def test_merger_related_expenses_component_loses_to_clean_diluted_eps():
    """0000905729-20-000091. Merger-related expenses of $0.04 per diluted share
    is a component of earnings and must lose to clean GAAP diluted EPS of $0.45."""
    text = _fixture("0000905729-20-000091_merger_expenses_component.txt")
    out = release_eps.extract_diluted_eps(text)

    assert out["eps"] == pytest.approx(0.45)
    assert out["eps"] != pytest.approx(0.04)


def test_release_with_only_component_charges_is_refused():
    """0001558370-25-000359. "net charge to earnings of ... $0.08 per diluted share"
    is a component, not net income. A release with only component figures must
    be refused rather than crowning the charge as earnings."""
    text = _fixture("0001558370-25-000359_charge_to_earnings_component.txt")
    out = release_eps.extract_diluted_eps(text)

    assert out["eps"] is None
    assert "earnings component" in out["reason"]


def test_diluted_loss_in_earnings_per_share_is_read_negative():
    """0000950170-25-060598. "Diluted loss in earnings per share was $1.29".

    The phrase carries "loss in earnings" and must be read as negative -1.29."""
    text = _fixture("0000950170-25-060598_diluted_loss_in_earnings.txt")
    out = release_eps.extract_diluted_eps(text)

    assert out["eps"] == pytest.approx(-1.29)


def test_annual_sentence_is_never_read_as_quarterly_eps():
    """0000950170-23-007983. "Net loss for the year ended ... was $2.44 per share".

    An annual/full-year sentence must never be crowned as quarterly EPS when
    reading a quarterly release."""
    text = _fixture("0000950170-23-007983_annual_figure_only.txt")
    out = release_eps.extract_diluted_eps(text)

    assert out["eps"] is None
    assert "annual or multi-period" in out["reason"]


def test_flattened_statement_of_income_is_skipped_by_prose():
    """0001140361-20-016628. Consolidated Statements of Income with truncated
    headers "Dol… in tho… exc…" is a flattened statement and must be skipped
    by prose rather than reading a full-year cell."""
    text = _fixture("0001140361-20-016628_flattened_statement_dol_in_tho.txt")
    out = release_eps.extract_diluted_eps(text)

    assert out["eps"] is None
    assert out["reason"] == "no diluted per-share figure named in the release"


def test_prior_year_earnings_is_treated_as_comparative():
    """0000736772-23-000033. "The Corporation’s prior year earnings ... were $0.84"
    is a comparative sentence. When unaccompanied by current figures, it is
    refused."""
    text = _fixture("0000736772-23-000033_prior_year_earnings.txt")
    out = release_eps.extract_diluted_eps(text)

    assert out["eps"] is None
    assert "comparative" in out["reason"]


def test_table_row_with_basic_then_diluted_reads_diluted():
    """0001326801-24-000044. A table with "Basic $ 4.86 $ 2.21 Diluted $ 4.71 $ 2.20"
    must read the diluted row figures, not the preceding basic figures."""
    text = _fixture("0001326801-24-000044_basic_before_diluted_table.txt")
    out = release_eps.extract_diluted_eps(text, period_end="2024-03-31")

    assert out["basis"] == "table"
    assert out["eps"] == pytest.approx(4.71)
    assert out["eps"] != pytest.approx(4.86)


def test_basic_and_diluted_respectively_reads_diluted_figure():
    """0000950170-23-034763. "Basic earnings per share and diluted earnings per
    share were $0.57 ... and $0.53 ..., respectively": the second figure is
    diluted."""
    text = _fixture("0000950170-23-034763_basic_and_diluted_respectively.txt")
    out = release_eps.extract_diluted_eps(text)

    assert out["eps"] == pytest.approx(0.53)
    assert out["eps"] != pytest.approx(0.57)


def test_sentence_boundary_before_four_digit_year():
    """0001347178-20-000022. "... second quarter of 2019. 2020 Financial Guidance ..."
    splits at the sentence boundary before the year, so guidance text does not
    contaminate the reported earnings sentence."""
    text = _fixture("0001347178-20-000022_sentence_boundary_before_year.txt")
    out = release_eps.extract_diluted_eps(text)

    assert out["eps"] == pytest.approx(0.16)
