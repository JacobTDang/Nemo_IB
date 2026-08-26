"""A per-share figure must divide by every share, not one class of them.

`get_market_data` reports `marketCap` on an all-class basis and, from the
provider, `sharesOutstanding` for a single class. For GOOGL those differ by
2.0845x and for BRK-B by 1.5204x. That is now reconciled inside the response:
`shares_outstanding_all_classes` reproduces the market cap to 0.0000%, and a
`share_count_basis_mismatch` warning names the multiple.

The reconciliation is worth nothing if the models keep reading the old field.
`calculate_dcf` divides equity value by a share count to produce
`price_per_share`, and the execution engine resolved that from
`sharesOutstanding` -- so a fair value for GOOGL came out 2.08x too high, and
the fix one layer up would have looked complete while the number a caller acts
on stayed wrong.

Single-class filers are unaffected: for them the two fields are the same
number, which is why this went unnoticed on NVDA, AAPL and MSFT.
"""
import pytest


def test_the_engine_resolves_the_all_class_count():
    from agent.workflows import execution_engine

    mapping = execution_engine.VARIABLE_SOURCE_KEYS \
        if hasattr(execution_engine, "VARIABLE_SOURCE_KEYS") else None
    if mapping is None:
        import re
        src = open(execution_engine.__file__).read()
        assert "'shares_outstanding': 'shares_outstanding_all_classes'" in src, (
            "the engine still maps shares_outstanding to the single-class "
            "provider field, so a multi-class DCF stays 2.08x too high")
        return
    assert mapping.get("shares_outstanding") == "shares_outstanding_all_classes"


def test_the_modelling_agent_asks_for_the_all_class_count():
    import agent.Financial_Modeling_Agent as fma

    src = open(fma.__file__).read()
    assert "shares_outstanding_all_classes" in src, (
        "the modelling agent still resolves sharesOutstanding, which is one "
        "class on GOOGL and BRK-B")


@pytest.mark.network
def test_a_multi_class_price_per_share_is_not_inflated():
    """End to end: the divisor must reproduce the market cap."""
    from tools.financial_modeling_engine.utils import get_data

    data = get_data("GOOGL")
    price = data["currentPrice"]
    all_classes = data["shares_outstanding_all_classes"]
    assert abs(price * all_classes - data["marketCap"]) / data["marketCap"] < 1e-4

    single = data["sharesOutstanding"]
    assert single < all_classes, (
        "GOOGL's provider share count should be the narrower Class A figure")
