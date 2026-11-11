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


def test_include_marks_a_component_of_earnings():
    """0001046025-24-000159. "The second quarter results include a loss on
    sale of securities of $1.9 million, or $0.04 per diluted share" names the
    quarter, so it outranked the headline. It is a component; the headline's
    $0.41 is the figure."""
    text = _fixture("0001046025-24-000159_include_loss_on_sale_component.txt")
    out = release_eps.extract_diluted_eps(text)

    assert out["eps"] == pytest.approx(0.41)


def test_six_month_period_is_not_the_quarter():
    """0001564590-21-036632. The quote with "the six-month period ended June
    30, 2021 ... $1.03" comes first; the three-month period's $0.48 is the
    quarter."""
    text = _fixture("0001564590-21-036632_six_month_period.txt")
    out = release_eps.extract_diluted_eps(text)

    assert out["eps"] == pytest.approx(0.48)


def test_three_and_twelve_months_reads_the_first_figure():
    """0000950170-24-007227. "were $0.70 and $2.82 in the three and twelve
    months ended": the quarter comes first."""
    text = _fixture("0000950170-24-007227_three_and_twelve_months.txt")
    out = release_eps.extract_diluted_eps(text)

    assert out["eps"] == pytest.approx(0.70)


def test_quarter_and_fiscal_year_reads_the_first_figure():
    """0001140361-21-025390. "Net income for the quarter and fiscal year ...
    was $7.6 million, or $0.89 per basic and diluted share, and $23.9 million,
    or $2.81 ..., respectively": the quarter comes first."""
    text = _fixture("0001140361-21-025390_quarter_and_fiscal_year.txt")
    out = release_eps.extract_diluted_eps(text)

    assert out["eps"] == pytest.approx(0.89)


def test_including_in_the_company_description_is_not_a_component():
    """0000719220-22-000024. "with operations in five markets including
    Western Pennsylvania, ... announced net income of $29.1 million, or $0.74
    per diluted share". The net income phrase comes after "including", so the
    figure is net income, not a component."""
    text = _fixture("0000719220-22-000024_including_in_company_description.txt")
    out = release_eps.extract_diluted_eps(text)

    assert out["eps"] == pytest.approx(0.74)


def test_diluted_cells_that_do_not_fill_the_columns_are_refused():
    """0001326801-24-000044, cut after the first diluted cell. The row opens
    with the basic figures; when the diluted figures after it do not fill the
    columns, the basic $4.86 is not read in their place."""
    text = _fixture("0001326801-24-000044_diluted_cells_cut_short.txt")
    out = release_eps.extract_diluted_eps(text, period_end="2024-03-31")

    assert out["eps"] is None
    assert "against 2 columns" in out["reason"]


def test_basic_and_split_around_the_cells_reads_them_as_diluted():
    """0001506293-22-000039. Pinterest's row is "Net loss per share, basic and
    $ -0.01 $ -0.03 diluted": the label is split around cells that are both
    basic and diluted, so they are read in place."""
    text = _fixture("0001506293-22-000039_basic_and_split_around_cells.txt")
    out = release_eps.extract_diluted_eps(text, period_end="2022-03-31")

    assert out["basis"] == "table"
    assert out["eps"] == pytest.approx(-0.01)


def test_quarter_and_year_after_the_figure_is_not_a_quarter_word():
    """0001437749-21-001576. A bullet splits "merger-related expenses of
    approximately ● $547,000 ... ($0.07 per diluted share ...), for the quarter
    and year ended", and the fragment keeps no component word. After the
    figure, "quarter and year" does not name the quarter, so the headline's
    $0.52 is read."""
    text = _fixture("0001437749-21-001576_quarter_and_year_after_split_figure.txt")
    out = release_eps.extract_diluted_eps(text)

    assert out["eps"] == pytest.approx(0.52)
