"""A per-share figure must divide by every share, not one class of them.

`get_market_data` reports `marketCap` on an all-class basis and, from the
provider, `sharesOutstanding` for a single class. For GOOGL those differ by
2.0845x and for BRK-B by 1.5204x. That is now reconciled inside the response:
`shares_outstanding_all_classes` reproduces the market cap to 0.0000%, and a
`share_count_basis_mismatch` warning names the multiple.

The reconciliation is worth nothing if the number itself is wrong.
`calculate_dcf` divides equity value by a share count to produce
`price_per_share`, and a caller resolving that from `sharesOutstanding` gets a
fair value for GOOGL 2.08x too high -- so the test below checks the divisor
against the market cap rather than against any one caller.

Single-class filers are unaffected: for them the two fields are the same
number, which is why this went unnoticed on NVDA, AAPL and MSFT.
"""
import pytest


# The two consumers this file pinned -- `agent/workflows/execution_engine.py`,
# which resolved the divisor for `calculate_dcf`, and
# `agent/Financial_Modeling_Agent.py`, which asked for it -- were deleted with
# the LangGraph/OpenRouter layer (issue #63). Nothing in the tree reads
# `sharesOutstanding` as a share count any more, so what is left to pin is the
# reconciliation itself: the field the response publishes has to reproduce the
# market cap, which is the property every future consumer will depend on.


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
