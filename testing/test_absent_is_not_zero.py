"""An absent value must be null, never zero.

Zero is a claim. "This company holds no cash" and "we could not find out how
much cash it holds" are different statements, and only one of them is ever
true of a ticker that does not exist.

`get_market_data("ZZZQQQ")` returned `success: true` with `cash: 0` and
`totalDebt: 0` from a single line each:

    data['cash'] = info.get('totalCash', 0)

yfinance returns an info dict with one key for an unknown ticker, so both
defaulted. Those two fields feed `calculate_wacc` and
`calculate_credit_profile`, which would then value a company that does not
exist as debt-free.

This is the same shape as the Amazon defect fixed earlier -- `ocf - (capex or
0)` turning a missing capex into a free cash flow 18x too high. That was
fixed in one tool. This file is the class.

A count over an empty result set is a different thing and stays zero: nobody
traded it, we looked. What must never be zero is a quantity nobody read.
"""
import pytest


class TestMarketData:
    """Needs yfinance; the fake ticker costs one cheap lookup."""

    @pytest.mark.network
    @pytest.mark.parametrize("ticker", ["ZZZQQQ", "NOTAREALTICKER99"])
    def test_a_company_that_does_not_exist_has_no_cash(self, ticker):
        from tools.financial_modeling_engine.utils import get_data
        data = get_data(ticker)

        assert data.get("cash") is None, (
            f"{ticker}: cash={data.get('cash')!r}. Zero says the company holds "
            f"no cash; the truth is that there is no company.")
        assert data.get("totalDebt") is None, (
            f"{ticker}: totalDebt={data.get('totalDebt')!r}, which reads as "
            f"debt-free and would value it that way")

    @pytest.mark.network
    def test_a_real_company_keeps_its_balance_sheet(self):
        """The guard must not cost a real filer its figures."""
        from tools.financial_modeling_engine.utils import get_data
        data = get_data("AAPL")
        assert data.get("cash") and data["cash"] > 0
        assert data.get("totalDebt") and data["totalDebt"] > 0

    @pytest.mark.network
    def test_a_genuine_zero_still_survives(self):
        """A filer that really holds no debt must still be able to say so.

        Distinguished by the key being present upstream rather than absent, so
        this asserts the mechanism rather than a particular company's balance
        sheet.
        """
        from tools.financial_modeling_engine.utils import _balance_or_none
        assert _balance_or_none({"totalDebt": 0}, "totalDebt") == 0
        assert _balance_or_none({}, "totalDebt") is None
        assert _balance_or_none({"totalDebt": None}, "totalDebt") is None


class TestThirteenF:
    def test_a_holding_with_no_reported_value_is_not_worth_nothing(self):
        """13F rows missing a Value column were coerced to $0.

        A holding recorded as worth zero drops out of every ranking and sum
        silently, which understates a fund's book with no sign that anything
        was lost.
        """
        from tools.web_search_server.hf_letters import _holding_value
        assert _holding_value({"Value": 1500}) == 1500
        assert _holding_value({"Value": 0}) == 0
        assert _holding_value({}) is None
        assert _holding_value({"Value": None}) is None


class TestAltmanInputs:
    def test_missing_total_assets_does_not_become_one_dollar(self):
        """`financials.get('total_assets', 0) or 1` guarded a division by
        substituting a $1 balance sheet, which makes every ratio enormous and
        the score meaningless rather than absent."""
        from tools.financial_modeling_engine.analysis_tools import _altman_inputs_ok
        assert _altman_inputs_ok({"total_assets": 1_000_000}) is True
        assert _altman_inputs_ok({"total_assets": 0}) is False
        assert _altman_inputs_ok({}) is False
        assert _altman_inputs_ok({"total_assets": None}) is False
