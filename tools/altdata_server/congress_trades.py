"""Congressional stock trades from the official STOCK Act disclosures.

Two chambers, two very different formats, one normalised answer.

* **House** publishes an annual ZIP per year containing an XML index of every
  filing. `FilingType` "P" is a Periodic Transaction Report -- the trades.
  The report itself is a PDF whose rows wrap: the amount bracket splits over
  two lines and the ticker lands before or after the transaction depending on
  where the line broke.
* **Senate** requires accepting a prohibition agreement before searching, then
  serves electronic reports as HTML tables with a dedicated ticker column.
  Paper filings are scanned PDFs and are reported as unreadable rather than
  guessed at.

What this cannot tell you
-------------------------
Positions. The STOCK Act discloses *transactions*, in brackets, up to 45 days
late. A member's current holdings are not published, and the annual reports
that come closest give year-end ranges. Every amount here is therefore a pair
of bounds, never a number: reporting a midpoint would invent a precision
Congress never disclosed, and that estimate would then travel as if it were
the filing.
"""
from __future__ import annotations

import io
import re
import zipfile
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from xml.etree import ElementTree as ET

import requests

HOUSE_INDEX_URL = "https://disclosures-clerk.house.gov/public_disc/financial-pdfs/{year}FD.zip"
HOUSE_PTR_URL = "https://disclosures-clerk.house.gov/public_disc/ptr-pdfs/{year}/{doc_id}.pdf"
SENATE_HOME = "https://efdsearch.senate.gov/search/home/"
SENATE_SEARCH = "https://efdsearch.senate.gov/search/report/data/"
SENATE_VIEW = "https://efdsearch.senate.gov/search/view/ptr/{uuid}/"

# The Senate search form's report-type code for a Periodic Transaction Report.
SENATE_PTR_REPORT_TYPE = 11

HTTP_TIMEOUT = 30

# The search endpoint caps a page at 100 rows no matter what `length` asks
# for: length=400 returns 100 and reports recordsTotal=234. A single request
# therefore sees part of the record and says nothing about the rest, so the
# walk follows every page rather than trusting one.
SENATE_PAGE_SIZE = 100


# House PTRs mark who holds the asset; an unmarked row is the filer's own.
_HOUSE_OWNER = {"SP": "spouse", "JT": "joint", "DC": "dependent_child"}
_SENATE_OWNER = {"self": "self", "spouse": "spouse", "joint": "joint",
                 "child": "dependent_child", "dependent child": "dependent_child"}

# A transaction row is anchored by its type followed by two dates. Longest
# alternative first, so "S (partial)" is not read as a bare "S".
_TXN_ANCHOR = re.compile(
    r"\b(?P<type>P|S \(partial\)|S|E)\s+"
    r"(?P<txn_date>\d{2}/\d{2}/\d{4})\s+"
    r"(?P<notified>\d{2}/\d{2}/\d{4})\s+"
    r"(?P<amount>\$[\d,]+(?:\s*-\s*\$[\d,]+)?\s*\+?)")

# The PDF renders field labels in a decorative font whose interior letters
# extract as NUL bytes: "Filing Status" arrives as
# 'F\x00\x00\x00\x00\x00 S\x00\x00\x00\x00\x00: New'. Every such label --
# Filing Status, Description, Subholding Of -- closes a transaction record, and
# they are recognised by that shape rather than by name, so a label this code
# has not seen still terminates cleanly. NULs are normalised to spaces first,
# because `\s` does not match NUL and the terminator silently never fired.
_RECORD_END = re.compile(r"^\s*[A-Z]\s{2,}")
_TABLE_START = re.compile(r"^\s*ID\s+Owner\s+Asset\s+Transaction")
_TABLE_END = re.compile(r"^\s*\*\s*For the complete list of asset type")

# 1-6 letters, optionally dotted. A CUSIP is nine alphanumerics and cannot
# match, which is the point: BRK.B is a ticker, 91282CJP7 is not.
_TICKER = re.compile(r"\(([A-Z][A-Z.]{0,5})\)")
_ASSET_TYPE = re.compile(r"\[([A-Z]{2})\]")
# Nine alphanumerics in parentheses: a CUSIP, which identifies bonds and
# treasuries that have no ticker at all.
_CUSIP = re.compile(r"\(([0-9A-Z]{9})\)")

_TYPE_NAMES = {"P": "purchase", "S": "sale", "S (partial)": "sale_partial",
               "E": "exchange"}


class DisclosureUnavailable(RuntimeError):
    """A filing could not be read. Never a statement about the member."""


def _user_agent(contact: Optional[str] = None) -> str:
    """Identify the caller, as the House and Senate both ask callers to do."""
    import os
    contact = contact or os.environ.get("SEC_EMAIL") or ""
    if not contact:
        raise DisclosureUnavailable(
            "SEC_EMAIL is not set. The House and Senate disclosure sites ask "
            "for a real contact address in the User-Agent; set SEC_EMAIL in "
            "your .env file.")
    return f"Nemo-IB research ({contact})"


# ------------------------------------------------------------------ amounts

# The bands the Ethics in Government Act defines. Trade sizes are reported as
# these and only these, so a pair that is not one of them was not read off the
# filing correctly -- it was assembled from something else on the page.
AMOUNT_BRACKETS: Tuple[Tuple[int, Optional[int]], ...] = (
    (1, 1000),
    (1001, 15000),
    (15001, 50000),
    (50001, 100000),
    (100001, 250000),
    (250001, 500000),
    (500001, 1000000),
    (1000001, 5000000),
    (5000001, 25000000),
    (25000001, 50000000),
    (50000000, None),          # "$50,000,000 +"
)
_BRACKET_FLOORS = {low: high for low, high in AMOUNT_BRACKETS}


# The PTR's table header, repeated at the top of each page. When an entry
# spans a page break the header is absorbed into the row -- 411 of 16,518
# stored transactions carry it inside `asset_name`, sometimes mid-name, so the
# security is split across it.
_TABLE_HEADER_RE = re.compile(
    r"\s*ID\s+Owner\s+Asset\s+Transaction\s+Date\s+Notification\s+Amount\s+"
    r"Cap\.\s+Type\s+Date\s+Gains\s*>\s*\??\s*",
    re.IGNORECASE)


def strip_table_header(name: str) -> str:
    """Remove the page header from a security name, rejoining what it split."""
    if not name:
        return name
    cleaned = _TABLE_HEADER_RE.sub(" ", name)
    return re.sub(r"\s{2,}", " ", cleaned).strip()


def parse_amount_range(text: str) -> Tuple[Optional[int], Optional[int]]:
    """`"$1,001 - $15,000"` to `(1001, 15000)`.

    Returns `(None, None)` rather than zeros when nothing parses: a zero
    bracket reads as a costless trade, which is a claim the filing never made.
    An open-ended top bracket ("$50,000,000 +") has no upper bound, so the
    upper bound is None rather than a repeat of the lower one.

    The pair is checked against the statutory brackets. 24 rows in the store
    held `amount_min > amount_max` with `amount_max = 200` -- "$200" is the
    PTR's own "Cap. Gains > $200?" column header, which bleeds into the row
    when an entry spans a page break and the amount cell loses its ceiling.
    An inverted range is not a smaller range: summing it under-counts by
    orders of magnitude, and "$50,001 to $200" is a fact no filing contains.

    Where the floor is a real bracket bound it is kept and the ceiling
    refused, because "at least $50,001" is true and "at most $200" is not.
    """
    if not text:
        return (None, None)
    numbers = [int(n.replace(",", "")) for n in re.findall(r"\$([\d,]+)", text)]
    if not numbers:
        return (None, None)

    low = numbers[0]
    if low not in _BRACKET_FLOORS:
        # Not a disclosure band at all. Reporting it as one would dress up a
        # misread as a filing figure.
        return (None, None)

    official_high = _BRACKET_FLOORS[low]
    if len(numbers) == 1:
        # "$50,000,000 +" is open-ended. A lone floor without a "+" lost its
        # ceiling on the page; the box it belongs to still defines one.
        return (low, None if "+" in text else official_high)

    high = numbers[1]
    if high == official_high:
        return (low, high)
    # The ceiling disagrees with the band the floor belongs to, so it came from
    # somewhere else on the page -- usually the "Cap. Gains > $200?" header.
    # The floor identifies which box the filer ticked and the Act defines that
    # box's ceiling, so the statutory ceiling is what the filing means, not an
    # inference. Returning None instead made these rows look like unbounded
    # ">$50,000,000" disclosures, and one such row erases the ceiling of every
    # total it enters: two suppressed `amount_max_total` across 200 AAPL rows.
    return (low, official_high)


def _iso(date_text: str) -> Optional[str]:
    for fmt in ("%m/%d/%Y", "%m/%d/%y"):
        try:
            return datetime.strptime(date_text.strip(), fmt).date().isoformat()
        except ValueError:
            continue
    return None


# -------------------------------------------------------------------- House

def _normalise(text: str) -> str:
    """PDF label glyphs extract as NUL bytes; make them ordinary spaces."""
    return text.replace("\x00", " ")


def _house_metadata(text: str) -> Dict[str, Optional[str]]:
    def grab(pattern: str) -> Optional[str]:
        match = re.search(pattern, text)
        return match.group(1).strip() if match else None

    return {
        "doc_id": grab(r"Filing ID\s*#?\s*(\d+)"),
        "member": grab(r"Name:\s*(.+)"),
        "status": grab(r"Status:\s*(.+)"),
        "state_district": grab(r"State/District:\s*(\S+)"),
    }


def _house_records(text: str) -> List[str]:
    """The transaction block, split into one joined string per row.

    A row starts at the line carrying the anchor and runs until the Filing
    Status marker beneath it, absorbing the continuation lines that hold the
    wrapped asset name, the ticker and the upper bound of the bracket.
    """
    lines = _normalise(text).split("\n")
    start = next((i + 1 for i, l in enumerate(lines) if _TABLE_START.match(l)), None)
    if start is None:
        return []
    end = next((i for i, l in enumerate(lines) if _TABLE_END.match(l)), len(lines))

    records: List[str] = []
    current: List[str] = []
    for line in lines[start:end]:
        if _RECORD_END.match(line):
            if current:
                records.append(" ".join(current))
                current = []
            continue
        if _TXN_ANCHOR.search(line) and current:
            records.append(" ".join(current))
            current = []
        current.append(line.strip())
    if current:
        records.append(" ".join(current))

    return [r for r in records if _TXN_ANCHOR.search(r)]


def _house_transaction(record: str) -> Optional[Dict[str, Any]]:
    anchor = _TXN_ANCHOR.search(record)
    if anchor is None:
        return None

    head = record[:anchor.start()].strip()
    tail = record[anchor.end():].strip()

    owner = "self"
    owner_match = re.match(r"^(SP|JT|DC)\b", head)
    if owner_match:
        owner = _HOUSE_OWNER[owner_match.group(1)]
        head = head[owner_match.end():].strip()

    # The bracket's upper bound wraps onto the continuation line, so read the
    # amount from the anchor and the tail together.
    amount_min, amount_max = parse_amount_range(
        f"{anchor.group('amount')} {tail}")

    # The anchor stops at the lower bound, so a wrapped bracket leaves its
    # separating dash at the head of the tail. It belongs to the amount, not
    # to the asset name continuing on the next line.
    combined = f"{head} {re.sub(r'^\s*-\s*', '', tail)}"
    ticker_match = _TICKER.search(combined)
    asset_type_match = _ASSET_TYPE.search(combined)
    cusip_match = _CUSIP.search(combined)

    asset_name = combined
    asset_name = _TICKER.sub("", asset_name)
    asset_name = _CUSIP.sub("", asset_name)
    asset_name = _ASSET_TYPE.sub("", asset_name)
    # Take the whole bracket including its separating dash. Removing only the
    # figures leaves "Tax - Ref 5.00%" where the filing said "Tax Ref 5.00%".
    asset_name = re.sub(r"\$[\d,]+\s*-\s*\$?[\d,]*\s*\+?", " ", asset_name)
    asset_name = re.sub(r"\$[\d,]+", " ", asset_name)
    asset_name = re.sub(r"\s*-\s*$", "", asset_name.strip())
    asset_name = re.sub(r"\s{2,}", " ", asset_name).strip(" -")

    return {
        "chamber": "house",
        "owner": owner,
        "ticker": ticker_match.group(1) if ticker_match else None,
        "cusip": cusip_match.group(1) if cusip_match else None,
        "asset_name": asset_name,
        "asset_type_code": asset_type_match.group(1) if asset_type_match else None,
        "transaction_type": _TYPE_NAMES.get(anchor.group("type")),
        "transaction_date": _iso(anchor.group("txn_date")),
        "notification_date": _iso(anchor.group("notified")),
        "amount_min": amount_min,
        "amount_max": amount_max,
    }


def parse_house_ptr(text: str) -> Dict[str, Any]:
    """One House Periodic Transaction Report, as extracted PDF text."""
    text = _normalise(text)
    filing = _house_metadata(text)
    transactions = [t for t in (_house_transaction(r) for r in _house_records(text))
                    if t is not None]
    for txn in transactions:
        txn["doc_id"] = filing["doc_id"]
        txn["member"] = filing["member"]
        txn["state_district"] = filing["state_district"]
    filing["chamber"] = "house"
    filing["transactions"] = transactions
    return filing


# ------------------------------------------------------------------- Senate

def _senate_type(text: str) -> Optional[str]:
    lowered = text.strip().lower()
    if lowered.startswith("purchase"):
        return "purchase"
    if lowered.startswith("exchange"):
        return "exchange"
    if lowered.startswith("sale"):
        # A full exit and a trim are different signals; keep them apart.
        if "partial" in lowered:
            return "sale_partial"
        if "full" in lowered:
            return "sale_full"
        return "sale"
    return None


def parse_senate_ptr(html: str) -> List[Dict[str, Any]]:
    """The transaction table from one electronic Senate PTR."""
    from bs4 import BeautifulSoup

    # html.parser is stdlib. The lxml backend would parse this table just as
    # well, but it is not in the container's dependency group and the import
    # here is lazy, so the failure would surface on the first call in
    # production rather than at build time.
    soup = BeautifulSoup(html, "html.parser")
    transactions: List[Dict[str, Any]] = []

    for row in soup.select("table tbody tr"):
        cells = [c.get_text(" ", strip=True) for c in row.find_all(["td", "th"])]
        if len(cells) < 8:
            continue
        _, txn_date, owner, ticker, asset_name, asset_type, kind, amount = cells[:8]

        # The table prints "--" where a security has no ticker.
        ticker = ticker.strip()
        if ticker in {"--", "", "-"}:
            ticker = None

        amount_min, amount_max = parse_amount_range(amount)
        transactions.append({
            "chamber": "senate",
            "owner": _SENATE_OWNER.get(owner.strip().lower(), owner.strip().lower()),
            "ticker": ticker,
            "asset_name": asset_name,
            "asset_type_code": asset_type,
            "transaction_type": _senate_type(kind),
            "transaction_date": _iso(txn_date),
            "notification_date": None,
            "amount_min": amount_min,
            "amount_max": amount_max,
        })

    return transactions


# ------------------------------------------------------------------ fetching

def fetch_house_index(year: int, session: Optional[requests.Session] = None
                      ) -> List[Dict[str, str]]:
    """Every filing the House published for `year`, from the annual ZIP."""
    getter = session or requests
    url = HOUSE_INDEX_URL.format(year=year)
    try:
        response = getter.get(url, timeout=HTTP_TIMEOUT,
                              headers={"User-Agent": _user_agent()})
        response.raise_for_status()
        archive = zipfile.ZipFile(io.BytesIO(response.content))
        name = next(n for n in archive.namelist() if n.lower().endswith(".xml"))
        root = ET.fromstring(archive.read(name))
    except DisclosureUnavailable:
        raise
    except Exception as exc:  # noqa: BLE001 - surfaced, never masked
        raise DisclosureUnavailable(
            f"House index for {year} could not be read: {exc}") from exc

    filings = []
    for member in root.findall("Member"):
        filings.append({
            "last": (member.findtext("Last") or "").strip(),
            "first": (member.findtext("First") or "").strip(),
            "filing_type": (member.findtext("FilingType") or "").strip(),
            "state_district": (member.findtext("StateDst") or "").strip(),
            "filing_date": (member.findtext("FilingDate") or "").strip(),
            "doc_id": (member.findtext("DocID") or "").strip(),
            "year": (member.findtext("Year") or "").strip(),
        })
    return filings


def fetch_house_ptr(doc_id: str, year: int,
                    session: Optional[requests.Session] = None) -> Dict[str, Any]:
    """Download and parse one House PTR."""
    import pdfplumber

    getter = session or requests
    url = HOUSE_PTR_URL.format(year=year, doc_id=doc_id)
    try:
        response = getter.get(url, timeout=HTTP_TIMEOUT,
                              headers={"User-Agent": _user_agent()})
        response.raise_for_status()
        with pdfplumber.open(io.BytesIO(response.content)) as pdf:
            text = "\n".join((page.extract_text() or "") for page in pdf.pages)
    except DisclosureUnavailable:
        raise
    except Exception as exc:  # noqa: BLE001 - surfaced, never masked
        raise DisclosureUnavailable(
            f"House PTR {doc_id} ({year}) could not be read: {exc}") from exc

    if not text.strip():
        # Older filings are scanned paper. Saying so beats returning nothing
        # and letting a caller read it as a member who traded nothing.
        raise DisclosureUnavailable(
            f"House PTR {doc_id} ({year}) carries no extractable text; it is "
            f"most likely a scan of a paper filing. Source: {url}")

    filing = parse_house_ptr(text)
    filing["source_url"] = url
    return filing


def senate_session() -> requests.Session:
    """A session that has accepted the Senate's prohibition agreement."""
    session = requests.Session()
    session.headers["User-Agent"] = _user_agent()
    try:
        home = session.get(SENATE_HOME, timeout=HTTP_TIMEOUT)
        home.raise_for_status()
        token = re.search(r'name="csrfmiddlewaretoken" value="([^"]+)"', home.text)
        if token is None:
            raise DisclosureUnavailable(
                "the Senate search page did not carry a CSRF token; the form "
                "has probably changed")
        session.post(SENATE_HOME, timeout=HTTP_TIMEOUT,
                     headers={"Referer": SENATE_HOME},
                     data={"csrfmiddlewaretoken": token.group(1),
                           "prohibition_agreement": "1"}).raise_for_status()
    except DisclosureUnavailable:
        raise
    except Exception as exc:  # noqa: BLE001 - surfaced, never masked
        raise DisclosureUnavailable(
            f"could not open a Senate disclosure session: {exc}") from exc
    return session


def senate_search_pages(session: requests.Session, since: str,
                        report_types: str, filer_types: str = "[]",
                        limit: Optional[int] = None) -> List[List[Any]]:
    """Every result row for a search, following the server's 100-row pages.

    `limit` caps the walk for a caller that genuinely wants only the newest
    few; left unset, the walk continues to `recordsTotal` so a partial read
    can never be mistaken for the whole record.
    """
    rows: List[List[Any]] = []
    start = 0
    total = None
    while True:
        page_size = SENATE_PAGE_SIZE if limit is None else min(
            SENATE_PAGE_SIZE, limit - len(rows))
        if page_size <= 0:
            break
        try:
            response = session.post(
                SENATE_SEARCH, timeout=HTTP_TIMEOUT,
                headers={"Referer": "https://efdsearch.senate.gov/search/",
                         "X-CSRFToken": session.cookies.get("csrftoken", ""),
                         "X-Requested-With": "XMLHttpRequest"},
                data={"start": start, "length": page_size,
                      "report_types": report_types,
                      "filer_types": filer_types,
                      "submitted_start_date": f"{since} 00:00:00",
                      "submitted_end_date": "",
                      "candidate_state": "", "senator_state": "",
                      "office_id": "", "first_name": "", "last_name": ""})
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:  # noqa: BLE001 - surfaced, never masked
            raise DisclosureUnavailable(
                f"the Senate search failed at row {start}: {exc}") from exc

        page = payload.get("data") or []
        if total is None:
            total = payload.get("recordsTotal")
        rows.extend(page)
        # An empty page ends the walk. Without this a server that stops
        # returning rows while still reporting a larger total spins forever.
        if not page:
            break
        start += len(page)
        if total is not None and start >= total:
            break
        if limit is not None and len(rows) >= limit:
            break
    return rows if limit is None else rows[:limit]


def search_senate_ptrs(session: requests.Session, since: str,
                       limit: Optional[int] = None) -> List[Dict[str, str]]:
    """Senate PTRs submitted on or after `since` (MM/DD/YYYY)."""
    rows = senate_search_pages(session, since,
                               report_types=f"[{SENATE_PTR_REPORT_TYPE}]",
                               filer_types="[]", limit=limit)

    filings = []
    for row in rows:
        if len(row) < 5:
            continue
        first, last, office, link, filed = row[:5]
        uuid = re.search(r"/search/view/(\w+)/([0-9a-f-]+)/", link or "")
        filings.append({
            "first": first, "last": last, "office": office,
            "filed_date": filed,
            "kind": uuid.group(1) if uuid else None,
            "uuid": uuid.group(2) if uuid else None,
        })
    return filings


def fetch_senate_ptr(session: requests.Session, uuid: str) -> List[Dict[str, Any]]:
    url = SENATE_VIEW.format(uuid=uuid)
    try:
        response = session.get(url, timeout=HTTP_TIMEOUT)
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001 - surfaced, never masked
        raise DisclosureUnavailable(
            f"Senate PTR {uuid} could not be read: {exc}") from exc

    transactions = parse_senate_ptr(response.text)
    for txn in transactions:
        txn["source_url"] = url
        txn["doc_id"] = uuid
    return transactions


# ---------------------------------------------------------------- the tool

# Successful index fetches only. A failed one is a fact about the network,
# not about the year, and caching it would answer every later caller with an
# outage that has since passed.
_INDEX_CACHE: Dict[int, List[Dict[str, str]]] = {}


def _house_index_cached(year: int) -> List[Dict[str, str]]:
    if year not in _INDEX_CACHE:
        _INDEX_CACHE[year] = fetch_house_index(year)
    return _INDEX_CACHE[year]


def _within(date_text: str, since: datetime) -> bool:
    for fmt in ("%m/%d/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(date_text.strip(), fmt) >= since
        except ValueError:
            continue
    return False


def get_congress_trades(ticker: Optional[str] = None,
                        member: Optional[str] = None,
                        chamber: str = "both",
                        days: int = 45,
                        max_filings: int = 30) -> Dict[str, Any]:
    """Congressional stock trades disclosed under the STOCK Act.

    These are transactions, not positions, and the amounts are brackets. A
    member discloses within 45 days, so the default window is 45 days of
    *filings* -- the trades inside them are older still, and each row carries
    both its transaction date and the date it was disclosed.

    `coverage` reports how many filings were available against how many were
    read. That distinction is the whole point: an empty `transactions` list
    means nothing at all unless you know whether every filing was opened.
    Filings capped away by `max_filings`, and older ones filed on paper and
    scanned, are both counted rather than passed over silently.
    """
    since = datetime.now() - __import__("datetime").timedelta(days=days)
    wanted_ticker = ticker.upper().strip() if ticker else None
    wanted_member = member.lower().strip() if member else None

    transactions: List[Dict[str, Any]] = []
    coverage: Dict[str, Any] = {}
    errors: List[str] = []

    if chamber in ("both", "house"):
        house = {"filings_available": 0, "filings_read": 0,
                 "unreadable_scans": 0, "capped_unread": 0}
        try:
            years = {since.year, datetime.now().year}
            candidates = []
            for year in sorted(years):
                for filing in _house_index_cached(year):
                    if filing["filing_type"] != "P":
                        continue
                    if not _within(filing["filing_date"], since):
                        continue
                    if wanted_member and wanted_member not in (
                            f"{filing['first']} {filing['last']}".lower()):
                        continue
                    candidates.append((year, filing))

            house["filings_available"] = len(candidates)
            for year, filing in candidates[:max_filings]:
                try:
                    parsed = fetch_house_ptr(filing["doc_id"], year)
                except DisclosureUnavailable as exc:
                    if "no extractable text" in str(exc):
                        house["unreadable_scans"] += 1
                    else:
                        errors.append(str(exc))
                    continue
                house["filings_read"] += 1
                for txn in parsed["transactions"]:
                    txn["filed_date"] = filing["filing_date"]
                    txn["source_url"] = parsed.get("source_url")
                    transactions.append(txn)
            house["capped_unread"] = max(0, len(candidates) - max_filings)
        except DisclosureUnavailable as exc:
            errors.append(str(exc))
            house["error"] = str(exc)
        coverage["house"] = house

    if chamber in ("both", "senate"):
        senate = {"filings_available": 0, "filings_read": 0,
                  "unreadable_scans": 0, "capped_unread": 0}
        try:
            session = senate_session()
            found = search_senate_ptrs(
                session, since.strftime("%m/%d/%Y"), limit=max(max_filings, 100))
            if wanted_member:
                found = [f for f in found
                         if wanted_member in f"{f['first']} {f['last']}".lower()]
            senate["filings_available"] = len(found)
            for filing in found[:max_filings]:
                if filing["kind"] != "ptr":
                    # A paper filing is a scan behind a different route.
                    senate["unreadable_scans"] += 1
                    continue
                try:
                    rows = fetch_senate_ptr(session, filing["uuid"])
                except DisclosureUnavailable as exc:
                    errors.append(str(exc))
                    continue
                senate["filings_read"] += 1
                for txn in rows:
                    txn["member"] = f"{filing['first']} {filing['last']}".strip()
                    txn["office"] = filing["office"]
                    txn["filed_date"] = filing["filed_date"]
                    transactions.append(txn)
            senate["capped_unread"] = max(0, len(found) - max_filings)
        except DisclosureUnavailable as exc:
            errors.append(str(exc))
            senate["error"] = str(exc)
        coverage["senate"] = senate

    if wanted_ticker:
        transactions = [t for t in transactions if t.get("ticker") == wanted_ticker]

    transactions.sort(key=lambda t: t.get("transaction_date") or "", reverse=True)

    read = sum(c.get("filings_read", 0) for c in coverage.values())
    available = sum(c.get("filings_available", 0) for c in coverage.values())
    complete = read == available and not errors

    return {
        "success": read > 0 or available == 0,
        "query": {"ticker": wanted_ticker, "member": member,
                  "chamber": chamber, "days": days},
        "coverage": coverage,
        "coverage_complete": complete,
        "transactions": transactions,
        "transaction_count": len(transactions),
        "errors": errors,
        "note": (
            "STOCK Act disclosures are transactions, not holdings, and the "
            "amounts are brackets rather than figures: amount_min and "
            "amount_max are the bounds the filer disclosed and there is no "
            "midpoint to report. Members file up to 45 days after trading, so "
            "transaction_date and filed_date differ. "
            + ("" if complete else
               f"Coverage is incomplete: {read} of {available} filings were "
               f"read, so an absent ticker does not mean it was not traded.")),
    }
