"""The PTR's own table header is not part of the trade.

When a House PTR entry spans a page break, the repeated table header is
absorbed into the row. 411 of 16,518 stored transactions carry the phrase
"ID Owner Asset Transaction Date Notification Amount Cap. Type Date Gains > ?"
inside `asset_name`:

    "BlackRock, Inc. Common Stock ID Owner Asset Transaction Date ..."
    "Berkshire Hathaway Inc. New ID Owner Asset ... Gains > ? Common Stock"

The second shows the header landing MID-NAME, so the security is split across
it.

In 24 of the 411 the header also ate the amount ceiling. Those floors are
50001, 15001, 250001 and 100001 -- every one an exact statutory bracket bound,
so the filer ticked a real box and the box's ceiling is defined by the Ethics
in Government Act. It is not inferred data; it is what that checkbox means.

An earlier fix nulled those ceilings rather than filling them, to avoid
inventing a figure. That was too cautious and had a cost: `_open_ended` treats
any floor-without-ceiling as an unbounded ">$50,000,000" disclosure, and once
one such row enters a sum the whole total loses its ceiling. Two bad rows
suppressed `amount_max_total` across all 200 AAPL rows. A null that erases a
correct total for 200 rows is worse than the statutory ceiling for 2.

Only the genuine top bracket is open-ended. Its floor is 50,000,000, and no
row in the store has one.
"""
import pytest

from tools.altdata_server.congress_trades import parse_amount_range
from tools.altdata_server.congress_queries import _open_ended


def test_a_ceiling_eaten_by_the_header_falls_back_to_the_statutory_one():
    """The floor identifies the checkbox; the checkbox defines the ceiling."""
    assert parse_amount_range("$15,001 Cap. Gains > $200?") == (15001, 50000)
    assert parse_amount_range("$50,001 $200") == (50001, 100000)
    assert parse_amount_range("$250,001 $200") == (250001, 500000)


def test_the_top_bracket_still_has_no_ceiling():
    assert parse_amount_range("$50,000,000 +") == (50000000, None)


def test_an_ordinary_bracket_is_unchanged():
    assert parse_amount_range("$1,001 - $15,000") == (1001, 15000)
    assert parse_amount_range("$250,001 - $500,000") == (250001, 500000)


def test_a_floor_that_is_not_a_bracket_is_still_refused():
    assert parse_amount_range("$37 - $94") == (None, None)


def test_only_the_top_bracket_makes_a_total_open_ended():
    """A null ceiling on a mid-bracket floor is a parse failure, not an
    unbounded disclosure, and must not erase the total for every other row."""
    genuine = [{"amount_min": 50000000, "amount_max": None}]
    assert _open_ended(genuine, "amount_min", "amount_max") is True

    parse_failure = [{"amount_min": 15001, "amount_max": None}]
    assert _open_ended(parse_failure, "amount_min", "amount_max") is False


def test_a_bounded_set_is_not_open_ended():
    rows = [{"amount_min": 1001, "amount_max": 15000},
            {"amount_min": 15001, "amount_max": 50000}]
    assert _open_ended(rows, "amount_min", "amount_max") is False


# --- the header does not belong in the security's name ---------------------

from tools.altdata_server.congress_trades import strip_table_header


def test_the_header_is_stripped_from_a_name():
    assert strip_table_header(
        "BlackRock, Inc. Common Stock ID Owner Asset Transaction Date "
        "Notification Amount Cap. Type Date Gains > ?"
    ) == "BlackRock, Inc. Common Stock"


def test_a_name_split_across_the_header_is_rejoined():
    assert strip_table_header(
        "Berkshire Hathaway Inc. New ID Owner Asset Transaction Date "
        "Notification Amount Cap. Type Date Gains > ? Common Stock"
    ) == "Berkshire Hathaway Inc. New Common Stock"


def test_an_ordinary_name_is_untouched():
    assert strip_table_header("Apple Inc. - Common Stock") == \
        "Apple Inc. - Common Stock"
