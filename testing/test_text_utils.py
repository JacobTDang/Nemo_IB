"""Pure unit tests for the shared altdata text utilities (no network).

These pin the exact bugs the /review pass found: substring traps in keyword
matching, missing bn/mn dollar units, and timezone-naive date comparison.
"""
from __future__ import annotations

from datetime import datetime, timezone

from tools.altdata_server.text_utils import (
    text_contains,
    count_matches,
    extract_dollar_amounts,
    parse_news_date,
)


# ---------------------------------------------------------------------------
# text_contains — word boundary vs phrase
# ---------------------------------------------------------------------------

def test_cut_not_in_circuit():
    assert text_contains("the integrated circuit board", "cut") is False
    assert text_contains("executive prosecuted the case", "cut") is False
    assert text_contains("we will cut capex next year", "cut") is True


def test_ban_not_in_banking():
    assert text_contains("banking regulation reform act", "ban") is False
    assert text_contains("a ban on chip exports", "ban") is True


def test_invest_not_in_investigation():
    assert text_contains("investigation into pricing", "invest") is False
    assert text_contains("plans to invest in new fabs", "invest") is True


def test_fund_not_in_refund():
    assert text_contains("customer refund policy", "fund") is False
    assert text_contains("federal fund for research", "fund") is True


def test_cap_not_in_capacity():
    assert text_contains("adding capacity at the plant", "cap") is False
    assert text_contains("a price cap on drugs", "cap") is True


def test_phrase_substring_match():
    assert text_contains("building a new data center", "data center") is True
    assert text_contains("a generous tax credit program", "tax credit") is True
    assert text_contains("no relevant content", "data center") is False


def test_count_matches():
    text = "the company will invest and expand, then cancel one site"
    assert count_matches(text, ["invest", "expand", "cancel"]) == 3
    assert count_matches(text, ["invest", "merge", "spinoff"]) == 1
    # substring traps must not inflate the count
    assert count_matches("integrated circuit design", ["cut"]) == 0


# ---------------------------------------------------------------------------
# extract_dollar_amounts
# ---------------------------------------------------------------------------

def test_dollars_long_units():
    assert extract_dollar_amounts("invest $2.5 billion in a plant") == [2.5e9]
    assert extract_dollar_amounts("$1.2 trillion package") == [1.2e12]
    assert extract_dollar_amounts("$300 million facility") == [3e8]


def test_dollars_abbreviations():
    assert extract_dollar_amounts("a $5bn data center") == [5e9]
    assert extract_dollar_amounts("$300mn upgrade") == [3e8]
    assert extract_dollar_amounts("$1.2trn over a decade") == [1.2e12]


def test_dollars_single_letters():
    vals = extract_dollar_amounts("$500M chips and $3B funding")
    assert 5e8 in vals and 3e9 in vals


def test_dollars_comma_grouped():
    assert extract_dollar_amounts("$1,200 million expansion") == [1.2e9]


def test_dollars_none_present():
    assert extract_dollar_amounts("announced new products, no figures given") == []
    assert extract_dollar_amounts("the stock fell $10 today") == []  # no unit


# ---------------------------------------------------------------------------
# parse_news_date — always tz-aware UTC or None
# ---------------------------------------------------------------------------

def test_parse_bare_iso_date_is_aware():
    dt = parse_news_date("2026-06-01")
    assert dt is not None
    assert dt.tzinfo is not None
    assert dt.year == 2026 and dt.month == 6 and dt.day == 1


def test_parse_iso_z_suffix():
    dt = parse_news_date("2026-06-01T12:00:00Z")
    assert dt is not None and dt.tzinfo is not None
    assert dt.hour == 12


def test_parse_rfc2822():
    dt = parse_news_date("Mon, 01 Jun 2026 12:00:00 GMT")
    assert dt is not None and dt.tzinfo is not None
    assert dt.year == 2026 and dt.month == 6


def test_parse_garbage_returns_none():
    assert parse_news_date("not a date") is None
    assert parse_news_date("") is None
    assert parse_news_date(None) is None  # type: ignore[arg-type]


def test_parsed_date_comparable_to_aware_now():
    """The exact crash the review found: comparing parsed date to aware now()."""
    cutoff = datetime.now(timezone.utc)
    dt = parse_news_date("2020-01-01")   # bare date -> was naive -> TypeError
    assert dt is not None
    # This comparison must not raise
    assert (dt >= cutoff) in (True, False)
    assert dt < cutoff
