"""A negative enterprise value does not produce a valuation multiple.

For BRK-B the market-data provider reports:

    marketCap        1,079,602,774,016
    totalDebt          128,598,999,040
    cash               365,513,998,336
    enterpriseValue   -233,928,261,632     <- the provider's own figure

Market cap plus debt less cash is +842.7bn. The provider's enterprise value
disagrees with its own components by more than a trillion dollars -- Yahoo
treats Berkshire's insurance investments as cash -- and we passed it straight
through into:

    ev_revenue  -0.61
    ev_ebitda   -1.79

Those are not cheap valuations. They are not valuations at all: a multiple
built on a negative numerator has no ordering, so "-1.79x EBITDA" sorts below
zero and reads as the cheapest name in any comp table it lands in.

The module already suppresses multiples it cannot justify, with a reason, when
the quote and the filing are in different currencies. This is the same
situation -- a numerator we cannot stand behind -- and gets the same treatment.
The raw enterpriseValue is kept, because it is what the provider said.
"""
import pytest

from tools.financial_modeling_engine import utils


class _Handle:
    history_metadata = {"symbol": "BRK-B", "currency": "USD"}

    def __init__(self, info):
        self.info = info

    def __getattr__(self, name):
        return lambda *a, **k: None


def _market_data(monkeypatch, info):
    monkeypatch.setattr(utils.yf, "Ticker", lambda t: _Handle(info))
    return utils.get_data("BRK-B")


BRK = {
    "marketCap": 1_079_602_774_016, "enterpriseValue": -233_928_261_632,
    "totalCash": 365_513_998_336, "totalDebt": 128_598_999_040,
    "totalRevenue": 384_687_013_888, "ebitda": 130_683_002_880,
    "currency": "USD", "financialCurrency": "USD", "symbol": "BRK-B",
}


def test_a_negative_enterprise_value_suppresses_the_multiples(monkeypatch):
    data = _market_data(monkeypatch, BRK)
    for field in ("ev_revenue", "ev_ebitda", "ev_ebit"):
        assert data.get(field) is None, (
            f"{field} was computed on a negative enterprise value: "
            f"{data.get(field)}")


def test_the_suppression_says_why(monkeypatch):
    data = _market_data(monkeypatch, BRK)
    reason = str(data.get("multiples_suppressed_reason") or "")
    assert reason, "the multiples vanished with no explanation"
    assert "enterprise value" in reason.lower()


def test_the_providers_own_figure_is_still_reported(monkeypatch):
    """We disagree with it; we do not hide it."""
    data = _market_data(monkeypatch, BRK)
    assert data.get("enterpriseValue") == -233_928_261_632


def test_an_ordinary_company_keeps_its_multiples(monkeypatch):
    data = _market_data(monkeypatch, {
        **BRK, "enterpriseValue": 900_000_000_000, "symbol": "NVDA"})
    assert data.get("ev_revenue") == pytest.approx(2.339, rel=1e-2)
    assert data.get("multiples_suppressed_reason") is None


def test_a_zero_enterprise_value_is_also_refused(monkeypatch):
    """Zero divides into nothing and multiplies out to nothing."""
    data = _market_data(monkeypatch, {**BRK, "enterpriseValue": 0})
    assert data.get("ev_revenue") is None
