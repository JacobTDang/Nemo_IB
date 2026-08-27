"""A ratio across two currencies is not a ratio.

`get_market_data` computed every multiple as a price-derived numerator over a
financials-derived denominator -- marketCap / netIncomeToCommon,
enterpriseValue / revenue, and three more. For a US filer both sides are USD
and it works. For an ADR the numerator is the quote currency and the
denominator is the filer's, so every multiple is wrong by the exchange rate:

    TSM  reported P/E 0.977   true ~30.7   (TWD, /31.4)
    SONY reported P/E 0.126   true ~18.9   (JPY, /150)
    BABA reported P/E 3.90    true ~27.9   (CNY, /7.15)
    SAP  reported P/E 32.12   true ~29.7   (EUR, /1.08)

SAP is why this needs a rule rather than a spot check. An 8% error looks
entirely plausible, carries no smell, and would be acted on.

The provider supplies both currencies and its own correctly-computed P/E, so
the wrong number was being manufactured next to the right one.

This file is the class guard. It is not about `get_market_data`: any tool
dividing a price-derived figure by a filer-reported one has to answer the same
question first.
"""
import pytest

from tools.financial_modeling_engine.utils import (
    ratio_is_comparable, cross_currency_note)


class TestTheRule:
    def test_same_currency_is_comparable(self):
        assert ratio_is_comparable("USD", "USD") is True

    @pytest.mark.parametrize("quote,filing", [
        ("USD", "TWD"), ("USD", "JPY"), ("USD", "CNY"), ("USD", "EUR"),
        ("EUR", "USD"),
    ])
    def test_different_currencies_are_not(self, quote, filing):
        assert ratio_is_comparable(quote, filing) is False

    def test_case_and_whitespace_do_not_create_a_mismatch(self):
        assert ratio_is_comparable("usd", " USD ") is True

    @pytest.mark.parametrize("quote,filing", [(None, "USD"), ("USD", None),
                                              (None, None), ("", "USD")])
    def test_an_unknown_currency_is_not_assumed_to_match(self, quote, filing):
        """The failure this exists to prevent is assuming they agree.

        Defaulting an unknown to "probably the same" reproduces the bug for
        exactly the securities whose metadata is thinnest.
        """
        assert ratio_is_comparable(quote, filing) is False

    def test_the_note_names_both_currencies(self):
        note = cross_currency_note("USD", "TWD")
        assert "USD" in note and "TWD" in note
        assert note, "a suppressed ratio must say why"


@pytest.mark.network
class TestAgainstRealIssuers:
    """The rule applied end to end. Needs yfinance."""

    @pytest.mark.parametrize("ticker,expected_filing_currency", [
        ("TSM", "TWD"), ("SAP", "EUR"), ("SONY", "JPY"),
    ])
    def test_a_foreign_issuer_reports_both_currencies(self, ticker,
                                                      expected_filing_currency):
        from tools.financial_modeling_engine.utils import get_data
        data = get_data(ticker)
        assert data.get("currency"), f"{ticker}: no quote currency reported"
        assert data.get("financial_currency") == expected_filing_currency

    @pytest.mark.parametrize("ticker", ["TSM", "SAP", "SONY", "BABA"])
    def test_no_cross_currency_multiple_is_emitted(self, ticker):
        from tools.financial_modeling_engine.utils import get_data
        data = get_data(ticker)
        for key in ("pe_ratio", "pb_ratio", "ev_revenue", "ev_ebitda", "ev_ebit"):
            assert data.get(key) is None, (
                f"{ticker}: {key} = {data.get(key)} was computed across "
                f"{data.get('currency')} and {data.get('financial_currency')}")

    @pytest.mark.parametrize("ticker", ["TSM", "SAP", "SONY"])
    def test_the_suppression_is_explained(self, ticker):
        from tools.financial_modeling_engine.utils import get_data
        data = get_data(ticker)
        assert data.get("multiples_suppressed_reason"), (
            f"{ticker}: multiples are absent with no reason given, which reads "
            f"as data we could not obtain rather than a ratio we refuse to "
            f"invent")

    @pytest.mark.parametrize("ticker", ["AAPL", "MSFT", "NVDA"])
    def test_a_domestic_filer_still_gets_its_multiples(self, ticker):
        """The guard must not cost US names their ratios."""
        from tools.financial_modeling_engine.utils import get_data
        data = get_data(ticker)
        assert data.get("pe_ratio") is not None
        assert data.get("multiples_suppressed_reason") is None
        assert 0 < data["pe_ratio"] < 500, f"{ticker}: implausible {data['pe_ratio']}"

    @pytest.mark.parametrize("ticker", ["TSM", "SAP", "SONY", "AAPL"])
    def test_the_providers_own_pe_is_passed_through(self, ticker):
        """It is computed correctly upstream; refusing ours should not lose it."""
        from tools.financial_modeling_engine.utils import get_data
        data = get_data(ticker)
        pe = data.get("provider_trailing_pe")
        assert pe is not None, f"{ticker}: provider P/E dropped"
        assert 0 < pe < 500, f"{ticker}: implausible provider P/E {pe}"
