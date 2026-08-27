"""One security, two spellings, and a join that silently returns nothing.

Congressional filings write class shares with a dot -- `BRK.B`, `BF.B`,
`HEI.A` -- because that is what the filer typed. Market-data providers use a
dash: `BRK-B`. 108 transactions in the store use the dotted form.

Measured before this fix:

    ticker_activity("BRK.B") -> 98 transactions,  get_data("BRK.B") -> nulls
    ticker_activity("BRK-B") ->  0 transactions,  get_data("BRK-B") -> $1.08tn

Neither string works for both. A workflow reading congressional flow into
market data gets rows from one side and nothing from the other, whichever
spelling it picks, and nothing says why -- the empty side looks like a company
nobody traded.

The stored text is left exactly as filed. A filing said `BRK.B` and rewriting
it would put words in the filer's mouth. What changes is that a lookup matches
either spelling, and the response says which form it resolved to, so a caller
can see the two sides were joined on the same security.
"""
import pytest

from tools.ticker import normalize_ticker, ticker_variants


def test_the_two_spellings_normalize_together():
    assert normalize_ticker("BRK.B") == normalize_ticker("BRK-B")
    assert normalize_ticker("brk.b") == normalize_ticker("BRK-B")


def test_an_ordinary_ticker_is_unchanged():
    assert normalize_ticker("NVDA") == "NVDA"
    assert normalize_ticker(" nvda ") == "NVDA"


def test_nothing_is_invented_from_nothing():
    assert normalize_ticker("") is None
    assert normalize_ticker(None) is None


def test_both_spellings_are_offered_for_lookup():
    """A provider that only knows one form should still be reachable."""
    assert set(ticker_variants("BRK.B")) == {"BRK-B", "BRK.B"}
    assert set(ticker_variants("NVDA")) == {"NVDA"}


@pytest.mark.network
def test_the_congress_store_matches_either_spelling():
    from tools.altdata_server import congress_queries as q

    dotted = q.ticker_activity("BRK.B")
    dashed = q.ticker_activity("BRK-B")

    assert dotted["transaction_count"] > 0, "the dotted form found nothing"
    assert dashed["transaction_count"] == dotted["transaction_count"], (
        f"BRK-B found {dashed['transaction_count']} and BRK.B found "
        f"{dotted['transaction_count']} for the same security")


@pytest.mark.network
def test_market_data_resolves_either_spelling():
    from tools.financial_modeling_engine.utils import get_data

    dotted = get_data("BRK.B")
    dashed = get_data("BRK-B")

    assert dotted.get("marketCap"), "BRK.B returned no market cap"
    assert dotted["marketCap"] == dashed["marketCap"]


@pytest.mark.network
def test_the_response_says_which_form_it_used():
    """So a caller can tell the two sides were joined on one security."""
    from tools.financial_modeling_engine.utils import get_data

    data = get_data("BRK.B")
    assert data.get("ticker_resolved") == "BRK-B"
    assert data.get("ticker") == "BRK.B", "the requested form should be echoed"
