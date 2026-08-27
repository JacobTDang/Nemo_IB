"""A disclosure amount is a statutory bracket or it is nothing.

The House and Senate report trade sizes as fixed bands from the Ethics in
Government Act -- $1,001-$15,000, $15,001-$50,000, and so on. They are not
free figures, which is why this pipeline stores `amount_min` and `amount_max`
and no midpoint.

24 rows in the ingested store hold `amount_min > amount_max`, every one of
them with `amount_max = 200`:

    CSCO   sale          min=50,001   max=200
    AAPL   sale_partial  min=15,001   max=200
    BRK.B  purchase      min=15,001   max=200

`$200` is not a bracket bound. It comes from the PTR's own column header,
"Cap. Gains > $200?", which bled into the row when the entry spanned a page
break -- the same rows carry "ID Owner Asset Transaction Date Notification
Amount Cap. Type Date Gains > ?" inside `asset_name`. The lower bound is
correct in every case; the upper bound is a header artefact.

An inverted range is not a smaller range. Summing `amount_max` over these rows
under-counts by orders of magnitude, and a caller reading "$50,001 to $200"
has been handed a fact no filing contains.

So a pair that is not a statutory bracket does not get stored as one. Where
the floor is a real bracket bound we keep it and refuse the ceiling, because
"at least $50,001" is true and "at most $200" is not.
"""
import pytest

from tools.altdata_server.congress_trades import parse_amount_range


def test_a_real_bracket_parses_unchanged():
    assert parse_amount_range("$1,001 - $15,000") == (1001, 15000)
    assert parse_amount_range("$250,001 - $500,000") == (250001, 500000)


def test_an_open_ended_top_bracket_keeps_no_ceiling():
    assert parse_amount_range("$50,000,000 +") == (50000000, None)


def test_the_capital_gains_header_is_not_an_upper_bound():
    """The exact shape found in the store, 24 times over.

    Originally this asserted the ceiling became None, to avoid inventing a
    figure. That was reversed once the cost showed up: `_open_ended` treats any
    floor-without-ceiling as an unbounded ">$50,000,000" disclosure, so two
    such rows erased `amount_max_total` across all 200 AAPL rows. The floor
    identifies which statutory box the filer ticked and the Act defines that
    box's ceiling, so the ceiling is definitional rather than inferred. See
    test_header_bleed.py.
    """
    assert parse_amount_range("$50,001 $200") == (50001, 100000)
    assert parse_amount_range("$15,001 Cap. Gains > $200?") == (15001, 50000)


def test_an_inverted_range_never_survives():
    low, high = parse_amount_range("$100,001 - $200")
    assert high is None or high >= low, (
        f"stored an impossible range: {low} to {high}")


def test_a_floor_that_is_a_real_bracket_bound_is_kept():
    """"At least $50,001" is true even when the ceiling was lost."""
    assert parse_amount_range("$50,001 $200")[0] == 50001


def test_a_pair_that_is_not_a_bracket_at_all_is_refused():
    assert parse_amount_range("$37 - $94") == (None, None)


def test_nothing_parses_to_zero():
    """A zero bracket reads as a costless trade, which no filing claims."""
    assert parse_amount_range("") == (None, None)
    assert parse_amount_range("None") == (None, None)


@pytest.mark.parametrize("low,high", [
    (1001, 15000), (15001, 50000), (50001, 100000), (100001, 250000),
    (250001, 500000), (500001, 1000000), (1000001, 5000000),
    (5000001, 25000000), (25000001, 50000000),
])
def test_every_statutory_bracket_round_trips(low, high):
    assert parse_amount_range(f"${low:,} - ${high:,}") == (low, high)
