'''Parsing Senate annual reports, which are the closest thing to positions.

Every fixture row is verbatim from Boozman's CY2025 annual report.

The distinction that matters most here is the container row. An account, IRA
or trust appears as its own row with a value of "--" and the positions inside
it appear beneath it as 8.1, 8.2 and so on. Counting the container as a
holding invents a position; dropping it loses the account the leaves belong
to. Verified against a live report: every "--" row has children and no other
row does, so the rule is exact rather than a heuristic.

Three sentinels are not brackets and must not be coerced into one:
"Unascertainable" (state pensions, family trusts), "Over $50,000,000" and
"Over $1,000,000 and held independently by spouse or dependent child" -- the
EIGA cap, beyond which nothing is ever disclosed. Each has a lower bound and
no upper bound, which is not the same as an upper bound of zero.

And Schedule A covers assets held *at any point during* the year, not a
year-end snapshot. Rows carrying "No longer held" are exactly the ones a
reader would otherwise take for current positions.
'''
import pytest

from tools.altdata_server import senate_annual as sa

REPORT = '''
<html><body>
<h1>Annual Report for CY 2025 (Amendment 2)</h1>
<h2 class="filedReport">The Honorable John Boozman (Boozman, John)</h2>
<p class="muted">Filed 08/24/2026 @ 11:21 AM</p>
<section><h3>Part 3. Assets</h3>
<table id="grid_items" class="table dataTable">
<thead><tr><th></th><th>Asset</th><th>Asset Type</th><th>Owner</th>
<th>Value</th><th>Income Type</th><th>Income</th></tr></thead>
<tbody>
<tr class="nowrap"><td>1</td><td class="span4"><strong class="marginit-right">Arvest (Bentonville, Arkansas)</strong><div class="muted"><em>Type:</em> Checking</div></td><td>Bank Deposit</td><td>Joint</td><td>$15,001 - $50,000</td><td>Interest</td><td>None (or less than $201)</td></tr>
<tr class="nowrap"><td>4</td><td class="span4"><strong class="marginit-right">Merrill Lynch Account</strong><div class="addUnderlyingAssets noWrap"></div><div class="muted"></div></td><td>Brokerage/Managed Account</td><td>Joint</td><td>--</td><td></td><td></td></tr>
<tr class="nowrap"><td>4.1</td><td class="span4"><strong class="marginit-right">U.S. Treasury Bills</strong><div class="muted"><em>Rate/Coupon:</em> None <em>Matures:</em> None</div></td><td>Government Securities<div class="muted">US Treasury/Agency Security</div></td><td>Joint</td><td>$50,001 - $100,000</td><td>Interest</td><td>$201 - $1,000</td></tr>
<tr class="nowrap"><td>8</td><td class="span4"><strong class="marginit-right">John Boozman IRA</strong><div class="addUnderlyingAssets noWrap"></div><div class="muted"></div></td><td>Retirement Plans<div class="muted">IRA</div></td><td>Joint</td><td>--</td><td></td><td></td></tr>
<tr class="nowrap"><td>8.1</td><td class="span4"><strong class="marginit-right"><a href="http://finance.yahoo.com/q?s=TBLL" target="_blank">TBLL</a> - Invesco Short Term Treasury ETF</strong><div class="muted"></div></td><td>Mutual Funds<div class="muted">Exchange Traded Fund/Note</div></td><td>Joint</td><td>$50,001 - $100,000</td><td>Interest</td><td>$201 - $1,000</td></tr>
<tr class="nowrap"><td>8.2</td><td class="span4"><strong class="marginit-right"><a href="http://finance.yahoo.com/q?s=EBAY" target="_blank">EBAY</a> - Ebay Inc</strong><div class="muted"><em>Filer comment:</em> No longer held</div></td><td>Corporate Securities<div class="muted">Stock</div></td><td>Joint</td><td>None (or less than $1,001)</td><td>None</td><td>None (or less than $201)</td></tr>
<tr class="nowrap"><td>9</td><td class="span4"><strong class="marginit-right">SBUX - Starbucks Corporation - Common Stock</strong><div class="muted"></div></td><td>Corporate Securities<div class="muted">Stock</div></td><td>Spouse</td><td>Over $50,000,000</td><td>Dividends</td><td>Over $5,000,000</td></tr>
<tr class="nowrap"><td>10</td><td class="span4"><strong class="marginit-right">PERA - Public Employees Retirement Association</strong><div class="muted"></div></td><td>Retirement Plans<div class="muted">Pension</div></td><td>Self</td><td>Unascertainable</td><td>None</td><td>None (or less than $201)</td></tr>
<tr class="nowrap"><td>11</td><td class="span4"><strong class="marginit-right">Vanguard Balanced Fund</strong><div class="muted"></div></td><td>Mutual Funds<div class="muted">Fund</div></td><td>Spouse</td><td>Over $1,000,000 and held independently by spouse or dependent child</td><td>Excepted Investment Fund</td><td>None (or less than $201)</td></tr>
</tbody></table></section></body></html>
'''


@pytest.fixture(scope="module")
def parsed():
    return sa.parse_senate_annual(REPORT)


def test_the_report_header_is_read(parsed):
    assert parsed["member"] == "John Boozman"
    assert parsed["calendar_year"] == 2025
    assert parsed["amendment"] == 2
    assert parsed["filed_date"] == "2026-08-24"


def test_container_rows_are_marked_and_excluded_from_holdings(parsed):
    """An account is not a position; the things inside it are."""
    containers = [r for r in parsed["rows"] if r["is_container"]]
    assert {c["row_number"] for c in containers} == {"4", "8"}
    assert all(c["value_min"] is None for c in containers)
    assert {h["row_number"] for h in parsed["holdings"]}.isdisjoint({"4", "8"}), (
        "a brokerage account was counted as a holding alongside its contents")


def test_a_leaf_knows_the_account_it_sits_in(parsed):
    tbll = next(r for r in parsed["holdings"] if r["ticker"] == "TBLL")
    assert tbll["parent_row"] == "8"
    assert tbll["depth"] == 2


def test_a_linked_ticker_is_taken_from_the_anchor(parsed):
    tbll = next(r for r in parsed["holdings"] if r["row_number"] == "8.1")
    assert tbll["ticker"] == "TBLL"
    assert tbll["asset_name"] == "Invesco Short Term Treasury ETF"
    assert tbll["asset_type"] == "Mutual Funds"
    assert tbll["asset_subtype"] == "Exchange Traded Fund/Note"


def test_an_unlinked_ticker_is_recovered_only_for_security_types(parsed):
    """`SBUX - Starbucks` has no anchor; `PERA - Public Employees` is not a ticker."""
    sbux = next(r for r in parsed["holdings"] if r["row_number"] == "9")
    assert sbux["ticker"] == "SBUX"

    pera = next(r for r in parsed["holdings"] if r["row_number"] == "10")
    assert pera["ticker"] is None, (
        "'PERA' is a retirement association, not a symbol; resolving it sends "
        "a caller to an unrelated security")


def test_value_brackets_parse_to_their_bounds(parsed):
    treasury = next(r for r in parsed["holdings"] if r["row_number"] == "4.1")
    assert treasury["value_min"] == 50_001
    assert treasury["value_max"] == 100_000
    assert treasury["income_min"] == 201
    assert treasury["income_max"] == 1_000


def test_the_lowest_bracket_is_not_confused_with_zero(parsed):
    ebay = next(r for r in parsed["holdings"] if r["row_number"] == "8.2")
    assert ebay["value_min"] == 0
    assert ebay["value_max"] == 1_000
    assert ebay["value_text"].startswith("None (or less than")


def test_an_open_topped_bracket_has_no_upper_bound(parsed):
    sbux = next(r for r in parsed["holdings"] if r["row_number"] == "9")
    assert sbux["value_min"] == 50_000_000
    assert sbux["value_max"] is None, (
        "'Over $50,000,000' has no disclosed ceiling; inventing one "
        "understates the largest positions in the record")


def test_the_spouse_cap_is_a_floor_with_no_ceiling(parsed):
    """Beyond $1m a spouse's asset is never quantified further."""
    fund = next(r for r in parsed["holdings"] if r["row_number"] == "11")
    assert fund["value_min"] == 1_000_000
    assert fund["value_max"] is None
    assert fund["spouse_capped"] is True


def test_unascertainable_is_not_zero(parsed):
    pera = next(r for r in parsed["holdings"] if r["row_number"] == "10")
    assert pera["value_min"] is None and pera["value_max"] is None
    assert pera["value_unascertainable"] is True, (
        "an unascertainable value recorded as zero reads as an asset worth "
        "nothing rather than one the filer could not price")


def test_no_longer_held_is_carried_through(parsed):
    """Schedule A covers the whole year, not its final day."""
    ebay = next(r for r in parsed["holdings"] if r["row_number"] == "8.2")
    assert ebay["no_longer_held"] is True

    tbll = next(r for r in parsed["holdings"] if r["row_number"] == "8.1")
    assert tbll["no_longer_held"] is False


def test_an_excepted_investment_fund_is_flagged(parsed):
    """EIF means the underlying holdings are legally not itemised."""
    fund = next(r for r in parsed["holdings"] if r["row_number"] == "11")
    assert fund["excepted_investment_fund"] is True


def test_owner_is_normalised(parsed):
    owners = {r["row_number"]: r["owner"] for r in parsed["holdings"]}
    assert owners["4.1"] == "joint"
    assert owners["9"] == "spouse"
    assert owners["10"] == "self"


def test_a_report_without_the_assets_table_is_empty_not_an_error():
    """Some senators answer 'No' to Part 3; that is a real answer."""
    result = sa.parse_senate_annual(
        "<html><body><h1>Annual Report for CY 2025</h1>"
        "<h2 class='filedReport'>The Honorable A B (B, A)</h2></body></html>")
    assert result["holdings"] == []
    assert result["has_assets_table"] is False
