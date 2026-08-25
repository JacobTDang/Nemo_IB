"""Parsing congressional STOCK Act disclosures.

The text fixtures below are verbatim from the real filings -- House PTR
20032062 (Aderholt) and 20026537 (Allen), and Senate PTR
51455bcd-4966-4e77-b481-09897ada81ae (Boozman) -- because every hard case
here is a real formatting quirk rather than an imagined one:

* House PTRs are PDFs whose rows wrap. The amount bracket splits across two
  lines, and the ticker lands before the transaction on one row and after it
  on the next, depending on where the line broke.
* Treasuries carry a CUSIP in the parentheses where a ticker would sit. A
  CUSIP is not a ticker and must never be returned as one.
* The owner code (SP spouse, JT joint, DC dependent child) prefixes the
  asset name when present and is absent when the filer holds it directly.
* The PDF renders field labels in a decorative font whose interior letters
  extract as NUL bytes, so "Filing Status" arrives as
  `F\x00\x00\x00\x00\x00 S\x00\x00\x00\x00\x00: New`. These fixtures carry
  the real bytes. An earlier version of this file was transcribed from printed
  output, where the terminal displayed those NULs as spaces -- the tests passed
  against text that no filing produces, while the parser silently absorbed
  every label into the asset name beside it.

The amounts are brackets, not numbers. Every assertion here keeps them as
brackets: the disclosure regime does not publish what a member actually
spent, and a tool that reports a midpoint invents a precision Congress
never disclosed.
"""
import pytest

from tools.altdata_server import congress_trades as ct

HOUSE_SINGLE = """Filing ID #20032062
Clerk of the House of Representatives
Name: Hon. Robert B. Aderholt
Status: Member
State/District: AL04
ID Owner Asset Transaction Date Notification Amount Cap.
Type Date Gains >
$200?
GSK plc American Depositary Shares S 07/28/2025 08/11/2025 $1,001 - $15,000
(GSK) [ST]
F\x00\x00\x00\x00\x00 S\x00\x00\x00\x00\x00: New
* For the complete list of asset type abbreviations, please visit https://fd.house.gov/
Digitally Signed: Hon. Robert B. Aderholt , 09/10/2025
"""

HOUSE_WRAPPED = """Filing ID #20026537
Name: Hon. Richard W. Allen
Status: Member
State/District: GA12
ID Owner Asset Transaction Date Notification Amount Cap.
Type Date Gains >
$200?
SP Rollins, Inc. Common Stock (ROL) P 12/12/2024 01/08/2025 $15,001 -
[ST] $50,000
F\x00\x00\x00\x00\x00 S\x00\x00\x00\x00\x00: New
S\x00\x00\x00\x00\x00\x00\x00\x00\x00 O\x00\x00: R.W. Allen & Associates, Inc. > RWA&A - Securities
SP US TREASU NOTE 4.375% DUE P 12/03/2024 01/08/2025 $100,001 -
12/15/26 (91282CJP7) [GS] $250,000
F\x00\x00\x00\x00\x00 S\x00\x00\x00\x00\x00: New
S\x00\x00\x00\x00\x00\x00\x00\x00\x00 O\x00\x00: R.W. Allen & Associates, Inc. > RWA&A - Securities
US TREASURY BILL DUE 03/20/25 P 12/03/2024 01/08/2024 $15,001 -
(912797KJ5) [GS] $50,000
F\x00\x00\x00\x00\x00 S\x00\x00\x00\x00\x00: New
S\x00\x00\x00\x00\x00\x00\x00\x00\x00 O\x00\x00: SCH1
* For the complete list of asset type abbreviations, please visit https://fd.house.gov/
Digitally Signed: Hon. Richard W. Allen , 01/16/2025
"""

SENATE_HTML = """
<html><body><h1>Periodic Transaction Report</h1>
<table><thead><tr>
  <th>#</th><th>Transaction Date</th><th>Owner</th><th>Ticker</th>
  <th>Asset Name</th><th>Asset Type</th><th>Type</th><th>Amount</th>
  <th>Comment</th></tr></thead>
<tbody>
  <tr><td>14</td><td>11/21/2025</td><td>Joint</td><td>VEA</td>
      <td>Vanguard FTSE Developed Markets ETF</td><td>Stock</td>
      <td>Sale (Partial)</td><td>$1,001 - $15,000</td><td>--</td></tr>
  <tr><td>11</td><td>11/13/2025</td><td>Joint</td><td>CRM</td>
      <td>Salesforce Inc</td><td>Stock</td>
      <td>Sale (Full)</td><td>$1,001 - $15,000</td><td>--</td></tr>
  <tr><td>3</td><td>10/02/2025</td><td>Self</td><td>--</td>
      <td>US Treasury Note</td><td>Corporate Bond</td>
      <td>Purchase</td><td>$50,001 - $100,000</td><td>--</td></tr>
</tbody></table></body></html>
"""


def test_a_nul_padded_label_still_closes_a_record():
    """`\\s` does not match NUL, so the terminator has to see it first."""
    filing = ct.parse_house_ptr(HOUSE_SINGLE)
    name = filing["transactions"][0]["asset_name"]

    assert "\x00" not in name
    assert "Filing" not in name and ": New" not in name, (
        f"the Filing Status label was absorbed into the asset name: {name!r}")
    assert name.strip() == "GSK plc American Depositary Shares"


# ------------------------------------------------------------------ House

def test_a_single_house_transaction_is_read_whole():
    filing = ct.parse_house_ptr(HOUSE_SINGLE)

    assert filing["member"] == "Hon. Robert B. Aderholt"
    assert filing["state_district"] == "AL04"
    assert filing["doc_id"] == "20032062"
    assert len(filing["transactions"]) == 1

    txn = filing["transactions"][0]
    assert txn["ticker"] == "GSK"
    assert txn["transaction_type"] == "sale"
    assert txn["transaction_date"] == "2025-07-28"
    assert txn["notification_date"] == "2025-08-11"
    assert txn["amount_min"] == 1001
    assert txn["amount_max"] == 15000
    assert txn["owner"] == "self"
    assert "GSK plc" in txn["asset_name"]


def test_a_wrapped_amount_bracket_is_rejoined():
    """`$15,001 -` on one line and `$50,000` on the next is one bracket."""
    filing = ct.parse_house_ptr(HOUSE_WRAPPED)
    rollins = next(t for t in filing["transactions"] if t["ticker"] == "ROL")

    assert rollins["amount_min"] == 15001
    assert rollins["amount_max"] == 50000, (
        "the upper bound wrapped to the next line and was lost, which turns a "
        "$50,000 bracket into an open-ended one")
    assert rollins["transaction_type"] == "purchase"
    assert rollins["owner"] == "spouse"


def test_a_cusip_is_never_reported_as_a_ticker():
    """Treasuries put a CUSIP where a ticker would sit."""
    filing = ct.parse_house_ptr(HOUSE_WRAPPED)
    treasuries = [t for t in filing["transactions"]
                  if "TREASU" in t["asset_name"].upper()]

    assert len(treasuries) == 2, f"found {len(treasuries)} treasury rows"
    for txn in treasuries:
        assert txn["ticker"] is None, (
            f"{txn['ticker']!r} is a CUSIP, not a ticker; a caller looking it "
            f"up gets a different security or nothing")


def test_every_house_transaction_in_the_filing_is_found():
    filing = ct.parse_house_ptr(HOUSE_WRAPPED)
    assert len(filing["transactions"]) == 3, (
        f"found {len(filing['transactions'])}: a dropped row is a trade that "
        f"silently did not happen")


def test_house_owner_defaults_to_self_when_unmarked():
    filing = ct.parse_house_ptr(HOUSE_WRAPPED)
    bill = next(t for t in filing["transactions"]
                if "TREASURY BILL" in t["asset_name"].upper())
    assert bill["owner"] == "self"


# ----------------------------------------------------------------- Senate

def test_senate_rows_are_read_from_the_table():
    txns = ct.parse_senate_ptr(SENATE_HTML)
    assert len(txns) == 3

    vea = txns[0]
    assert vea["ticker"] == "VEA"
    assert vea["transaction_date"] == "2025-11-21"
    assert vea["owner"] == "joint"
    assert vea["transaction_type"] == "sale_partial"
    assert vea["amount_min"] == 1001 and vea["amount_max"] == 15000


def test_senate_full_and_partial_sales_stay_distinct():
    txns = ct.parse_senate_ptr(SENATE_HTML)
    kinds = {t["ticker"]: t["transaction_type"] for t in txns if t["ticker"]}
    assert kinds["VEA"] == "sale_partial"
    assert kinds["CRM"] == "sale_full", (
        "a full exit and a trim are different signals and must not collapse")


def test_a_senate_placeholder_ticker_becomes_none():
    """The table prints `--` when there is no ticker."""
    txns = ct.parse_senate_ptr(SENATE_HTML)
    treasury = next(t for t in txns if "Treasury" in t["asset_name"])
    assert treasury["ticker"] is None, "'--' is not a ticker"
    assert treasury["transaction_type"] == "purchase"


def test_a_wrapped_asset_name_rejoins_without_the_amount_dash():
    """`$500,001 -` wraps; removing the figure must take its dash with it."""
    filing = ct.parse_house_ptr(HOUSE_WRAPPED)
    note = next(t for t in filing["transactions"]
                if t["asset_name"].startswith("US TREASU NOTE"))
    assert note["asset_name"] == "US TREASU NOTE 4.375% DUE 12/15/26", (
        f"leftover punctuation from the bracket: {note['asset_name']!r}")
    assert note["amount_min"] == 100_001 and note["amount_max"] == 250_000, (
        "the wrapped date was read as the upper bound of the bracket")
    assert note["cusip"] == "91282CJP7", (
        "the CUSIP identifies a bond that has no ticker; it must be kept, "
        "just never as the ticker")


# ----------------------------------------------------------- amount ranges

@pytest.mark.parametrize("text,low,high", [
    ("$1,001 - $15,000", 1001, 15000),
    ("$15,001 - $50,000", 15001, 50000),
    ("$1,000,001 - $5,000,000", 1_000_001, 5_000_000),
    ("$50,000,000 +", 50_000_000, None),
])
def test_amount_brackets_parse_to_their_bounds(text, low, high):
    assert ct.parse_amount_range(text) == (low, high)


def test_an_unreadable_amount_is_none_not_zero():
    """Zero would read as a free trade rather than an unparsed one."""
    assert ct.parse_amount_range("") == (None, None)
    assert ct.parse_amount_range("Undetermined") == (None, None)


def test_no_midpoint_is_ever_reported():
    """The regime publishes brackets. A midpoint invents a precision.

    This is the one that matters: every downstream number must carry both
    bounds so a caller cannot mistake an estimate for a disclosure.
    """
    filing = ct.parse_house_ptr(HOUSE_SINGLE)
    txn = filing["transactions"][0]
    assert "amount" not in txn, "a single `amount` field invites a point estimate"
    assert {"amount_min", "amount_max"} <= set(txn)


# --------------------------------------------------------------- coverage

@pytest.fixture
def _house_only(monkeypatch):
    """A House index of `n` PTRs, each yielding one readable filing."""
    def install(n, unreadable=0):
        today = __import__("datetime").datetime.now().strftime("%m/%d/%Y")
        index = [{"last": f"Member{i}", "first": "A", "filing_type": "P",
                  "state_district": "XX01", "filing_date": today,
                  "doc_id": str(i), "year": "2026"} for i in range(n)]
        monkeypatch.setattr(ct, "_house_index_cached", lambda year: index)

        def fetch(doc_id, year, session=None):
            if int(doc_id) < unreadable:
                raise ct.DisclosureUnavailable(
                    f"House PTR {doc_id} carries no extractable text; it is "
                    f"most likely a scan of a paper filing.")
            return {"doc_id": doc_id, "member": f"Hon. Member{doc_id}",
                    "state_district": "XX01", "chamber": "house",
                    "source_url": "https://example.invalid",
                    "transactions": [{"chamber": "house", "ticker": "AAPL",
                                      "asset_name": "Apple Inc",
                                      "transaction_type": "purchase",
                                      "transaction_date": "2026-01-05",
                                      "amount_min": 1001, "amount_max": 15000}]}
        monkeypatch.setattr(ct, "fetch_house_ptr", fetch)
    return install


def test_capped_filings_are_counted_not_dropped(_house_only):
    _house_only(30)
    result = ct.get_congress_trades(chamber="house", max_filings=10)

    house = result["coverage"]["house"]
    assert house["filings_available"] == 30
    assert house["filings_read"] == 10
    assert house["capped_unread"] == 20, (
        "twenty filings went unread and the result said nothing about them")
    assert result["coverage_complete"] is False


def test_incomplete_coverage_says_so_in_the_note(_house_only):
    """An absent ticker must never read as a ticker that was not traded."""
    _house_only(30)
    result = ct.get_congress_trades(ticker="NVDA", chamber="house", max_filings=5)

    assert result["transactions"] == []
    assert result["coverage_complete"] is False
    assert "does not mean it was not traded" in result["note"], (
        "an empty result with partial coverage reads as a finding about NVDA")


def test_complete_coverage_is_stated_plainly(_house_only):
    _house_only(4)
    result = ct.get_congress_trades(chamber="house", max_filings=10)

    assert result["coverage_complete"] is True
    assert result["coverage"]["house"]["capped_unread"] == 0
    assert "Coverage is incomplete" not in result["note"]


def test_a_scanned_filing_is_counted_rather_than_ignored(_house_only):
    """A paper filing is a gap in coverage, not a member who did not trade."""
    _house_only(6, unreadable=2)
    result = ct.get_congress_trades(chamber="house", max_filings=10)

    house = result["coverage"]["house"]
    assert house["unreadable_scans"] == 2
    assert house["filings_read"] == 4
    assert result["coverage_complete"] is False


def test_amounts_survive_as_brackets_through_the_tool(_house_only):
    _house_only(2)
    result = ct.get_congress_trades(chamber="house", max_filings=10)

    for txn in result["transactions"]:
        assert txn["amount_min"] == 1001 and txn["amount_max"] == 15000
        assert "amount" not in txn
    assert "brackets rather than figures" in result["note"]
