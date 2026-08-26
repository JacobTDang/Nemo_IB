"""Two defects that make a correct-looking response wrong or unusable.

**NaN is not JSON.** `get_market_data("AAPL")` emitted
`"interestExpense": NaN` inside its payload. RFC 8259 has no NaN literal:
Python accepts it by default, which is why it survived, but any JavaScript,
Go or Rust client loses the ENTIRE response, not one field. A tool that
silently fails for every non-Python caller is worse than one that returns a
null.

**A fiscal-year figure sitting unlabelled among TTM ones.** `EBIT` is read
from the annual income statement while `revenue_ttm`, `ebitda_ttm` and
`net_income_ttm` come from the trailing-twelve-month feed. Nothing says so,
and the consequences compound:

    NVDA  EBITDA - EBIT implies D&A of $35.1bn, against an actual $2.84bn
    NVDA  ev_ebit 39.23 divides a current EV by a year-old EBIT (true ~31.5)

Both figures are individually right. Only their pairing is wrong, which is
the hardest kind of error to notice.
"""
import json
import math

import pytest


class TestJsonValidity:
    def test_a_nan_never_reaches_the_payload(self):
        from tools.financial_modeling_engine.utils import _json_safe
        cleaned = _json_safe({"a": float("nan"), "b": 1.5,
                              "c": {"d": float("inf")},
                              "e": [float("nan"), 2]})
        assert cleaned["a"] is None
        assert cleaned["b"] == 1.5
        assert cleaned["c"]["d"] is None, "infinity is not JSON either"
        assert cleaned["e"] == [None, 2]

    def test_the_result_survives_a_strict_parser(self):
        """The failure was total, not partial: one NaN loses the whole body."""
        from tools.financial_modeling_engine.utils import _json_safe
        payload = _json_safe({"interestExpense": float("nan"), "revenue": 1})
        text = json.dumps(payload)
        assert "NaN" not in text
        json.loads(text)          # a strict parser would reject NaN here

    def test_a_genuine_zero_and_a_genuine_none_are_untouched(self):
        from tools.financial_modeling_engine.utils import _json_safe
        assert _json_safe({"a": 0, "b": None, "c": 0.0}) == {"a": 0, "b": None, "c": 0.0}

    @pytest.mark.network
    def test_a_real_response_carries_no_nan(self):
        from tools.financial_modeling_engine.utils import get_data
        for ticker in ("AAPL", "NVDA"):
            text = json.dumps(get_data(ticker))
            assert "NaN" not in text and "Infinity" not in text, (
                f"{ticker}: response is not valid JSON for a strict parser")


class TestReportingBasis:
    @pytest.mark.network
    def test_ebit_says_which_period_it_covers(self):
        """Unlabelled, it reads as TTM like everything around it."""
        from tools.financial_modeling_engine.utils import get_data
        data = get_data("NVDA")
        if data.get("EBIT") is None:
            pytest.skip("no EBIT available for this filer")
        assert data.get("ebit_basis"), "EBIT carries no basis"
        assert data.get("ebit_period_end"), "EBIT carries no period end"

    @pytest.mark.network
    def test_a_mixed_basis_is_declared_rather_than_left_to_be_noticed(self):
        from tools.financial_modeling_engine.utils import get_data
        data = get_data("NVDA")
        if data.get("EBIT") is None or data.get("ebitda_ttm") is None:
            pytest.skip("not enough data to compare bases")
        assert data.get("basis_warning"), (
            "EBIT is fiscal-year and the ebitda/revenue/net income beside it "
            "are TTM, and nothing in the response says so. Subtracting one "
            "from the other gives NVDA an implied D&A of $35bn against an "
            "actual $2.8bn.")

    @pytest.mark.network
    def test_ev_ebit_declares_the_mismatch_it_is_built_on(self):
        """A current EV over a year-old EBIT is not a current multiple."""
        from tools.financial_modeling_engine.utils import get_data
        data = get_data("NVDA")
        if data.get("ev_ebit") is None:
            pytest.skip("no ev_ebit for this filer")
        assert data.get("ev_ebit_basis"), "ev_ebit does not say what it divides"
