"""Senate annual financial disclosures -- the closest thing to positions.

A Periodic Transaction Report says what changed. The annual report says what
was held, which is the question people actually ask, and it is the only place
either chamber publishes anything like a portfolio.

It comes with four caveats that the parser refuses to let a caller lose.

**Containers are not positions.** An account, IRA or trust is its own row with
a value of "--", and the things inside it appear beneath as 8.1, 8.2 and so
on. Counting the container inflates the book; dropping it loses the account
the leaves belong to. Verified on live reports: every "--" row has children
and no other row does, so `is_container` is exact rather than a guess.

**Three values are not brackets.** "Unascertainable" (state pensions, family
trusts), "Over $50,000,000", and "Over $1,000,000 and held independently by
spouse or dependent child" -- the EIGA cap, past which nothing is ever
disclosed. Each is a floor with no ceiling, which is not an upper bound of
zero.

**Schedule A covers the whole year, not its last day.** Rows tagged "No longer
held" are precisely the ones that would otherwise read as current positions.

**An Excepted Investment Fund does not itemise.** Roughly a third of leaf rows
are EIFs or blind trusts whose underlying holdings are legally undisclosed, so
absence of a ticker inside one is not absence of exposure.

Amendments are full restatements rather than deltas: take the highest
amendment number per (senator, calendar year) and never merge them.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

ASSETS_HEADER = ["", "Asset", "Asset Type", "Owner", "Value",
                 "Income Type", "Income"]

_OWNER = {"self": "self", "spouse": "spouse", "joint": "joint",
          "child": "dependent_child"}

# A container row's value cell. Verified exact against live reports.
CONTAINER_VALUE = "--"

UNASCERTAINABLE = "unascertainable"
_SPOUSE_CAP = "held independently by spouse or dependent child"

# `SBUX - Starbucks Corporation` is a ticker the filer typed without using the
# picker. `PERA - Public Employees Retirement Association` is not a ticker at
# all, and neither are UBS, TIAA or a dozen other plan administrators. Gating
# on the asset type is what separates them: only these two types are
# individually listed securities.
_TICKER_TYPES = {"Mutual Funds", "Corporate Securities"}
_LEADING_SYMBOL = re.compile(r"^([A-Z][A-Z0-9.\-]{0,5})\s+-\s+(.*)$")
_YAHOO = re.compile(r"finance\.yahoo\.com/q\?s=", re.I)


def _money(text: str) -> Optional[int]:
    digits = re.sub(r"[^\d]", "", text or "")
    return int(digits) if digits else None


def parse_value_bracket(text: str) -> Dict[str, Any]:
    """A Senate value cell, as bounds plus the reason when it has none.

    Never returns zero for something it could not read. A holding the filer
    could not price and a holding worth nothing are different facts, and only
    one of them is ever disclosed.
    """
    raw = (text or "").strip()
    out: Dict[str, Any] = {
        "value_text": raw, "value_min": None, "value_max": None,
        "value_unascertainable": False, "spouse_capped": False,
    }
    if not raw or raw == CONTAINER_VALUE:
        return out
    if UNASCERTAINABLE in raw.lower():
        out["value_unascertainable"] = True
        return out
    if _SPOUSE_CAP in raw.lower():
        # "Over $1,000,000 and held independently by spouse or dependent
        # child" is the end of the disclosure; there is no bracket above it.
        out["value_min"] = 1_000_000
        out["spouse_capped"] = True
        return out

    amounts = [_money(a) for a in re.findall(r"\$[\d,]+", raw)]
    if raw.lower().startswith("over") and amounts:
        out["value_min"] = amounts[0]
        return out
    if raw.lower().startswith("none") and amounts:
        # "None (or less than $1,001)" is a real floor of zero, not a missing
        # value: the filer is saying the holding is below the first bracket.
        out["value_min"] = 0
        out["value_max"] = amounts[0] - 1 if amounts[0] else None
        return out
    if len(amounts) >= 2:
        out["value_min"], out["value_max"] = amounts[0], amounts[1]
    elif amounts:
        out["value_min"] = out["value_max"] = amounts[0]
    return out


def parse_income_bracket(text: str) -> Tuple[Optional[int], Optional[int]]:
    """The income cell, which is not a clean enum.

    When the income type is "Other" the cell carries a literal figure and can
    hold both a bracket and an amount, e.g.
    `None (or less than $201) Other $100,001.00`.
    """
    raw = (text or "").strip()
    if not raw:
        return (None, None)
    amounts = [_money(a) for a in re.findall(r"\$[\d,]+(?:\.\d+)?", raw)]
    if not amounts:
        return (None, None)
    if raw.lower().startswith("over"):
        return (amounts[0], None)
    if raw.lower().startswith("none"):
        return (0, amounts[0] - 1 if amounts[0] else None)
    if len(amounts) >= 2:
        return (amounts[0], amounts[1])
    return (amounts[0], amounts[0])


def _cell_type(cell) -> Tuple[Optional[str], Optional[str]]:
    """`Mutual Funds` plus the `div.muted` subtype beneath it."""
    muted = cell.find("div", class_="muted")
    subtype = muted.get_text(" ", strip=True) if muted else None
    if muted:
        muted.extract()
    return (cell.get_text(" ", strip=True) or None, subtype or None)


def _asset_cell(cell) -> Dict[str, Any]:
    detail_div = cell.find("div", class_="muted")
    detail = detail_div.get_text(" ", strip=True) if detail_div else ""

    anchor = cell.find("a", href=_YAHOO)
    ticker = anchor.get_text(strip=True).upper() if anchor else None

    strong = cell.find("strong")
    raw_name = strong.get_text(" ", strip=True) if strong else \
        cell.get_text(" ", strip=True)
    name = raw_name
    if ticker and raw_name.upper().startswith(ticker):
        name = re.sub(rf"^{re.escape(ticker)}\s*-\s*", "", raw_name).strip()

    return {"raw_asset": raw_name, "asset_name": name, "ticker": ticker,
            "detail": detail}


def parse_senate_annual(html: str) -> Dict[str, Any]:
    """One Senate annual report: header, container rows, and the leaf holdings."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")

    heading = soup.find("h1")
    heading_text = heading.get_text(" ", strip=True) if heading else ""
    year = re.search(r"\bCY\s*(\d{4})", heading_text)
    amendment = re.search(r"Amendment\s+(\d+)", heading_text, re.I)

    filer = soup.find(class_="filedReport")
    member = None
    if filer:
        text = filer.get_text(" ", strip=True)
        text = re.sub(r"^The Honorable\s+", "", text)
        member = re.sub(r"\s*\(.*\)\s*$", "", text).strip()

    filed = None
    for tag in soup.find_all(class_="muted"):
        match = re.search(r"Filed\s+(\d{2}/\d{2}/\d{4})", tag.get_text(" ", strip=True))
        if match:
            filed = datetime.strptime(match.group(1), "%m/%d/%Y").date().isoformat()
            break

    result: Dict[str, Any] = {
        "member": member,
        "calendar_year": int(year.group(1)) if year else None,
        "amendment": int(amendment.group(1)) if amendment else 0,
        "filed_date": filed,
        "has_assets_table": False,
        "rows": [],
        "holdings": [],
    }

    # Locate by the header signature, not by table index: the Parts a senator
    # answers "No" to are omitted entirely, so the assets table is not at a
    # fixed position.
    grid = None
    for table in soup.find_all("table"):
        header = [th.get_text(" ", strip=True) for th in table.select("thead th")]
        if header == ASSETS_HEADER:
            grid = table
            break
    if grid is None:
        return result

    result["has_assets_table"] = True
    raw_rows = []
    for tr in grid.select("tbody tr"):
        cells = tr.find_all("td")
        if len(cells) < 7:
            continue
        raw_rows.append(cells)

    numbers = [c[0].get_text(" ", strip=True) for c in raw_rows]

    def has_children(number: str) -> bool:
        return any(o != number and o.startswith(f"{number}.") for o in numbers)

    for cells in raw_rows:
        number = cells[0].get_text(" ", strip=True)
        asset = _asset_cell(cells[1])
        asset_type, asset_subtype = _cell_type(cells[2])
        owner_raw = cells[3].get_text(" ", strip=True)
        value = parse_value_bracket(cells[4].get_text(" ", strip=True))
        income_type = cells[5].get_text(" ", strip=True) or None
        income_min, income_max = parse_income_bracket(
            cells[6].get_text(" ", strip=True))

        ticker = asset["ticker"]
        if ticker is None and asset_type in _TICKER_TYPES:
            match = _LEADING_SYMBOL.match(asset["raw_asset"])
            if match:
                ticker = match.group(1).upper()
                asset["asset_name"] = match.group(2).strip()

        parent = number.rsplit(".", 1)[0] if "." in number else None
        row = {
            "row_number": number,
            "depth": number.count(".") + 1,
            "parent_row": parent,
            "is_container": has_children(number),
            "ticker": ticker,
            "asset_name": asset["asset_name"],
            "raw_asset": asset["raw_asset"],
            "asset_detail": asset["detail"] or None,
            "asset_type": asset_type,
            "asset_subtype": asset_subtype,
            "owner": _OWNER.get(owner_raw.strip().lower(), owner_raw.strip().lower()),
            "income_type": income_type,
            "income_min": income_min,
            "income_max": income_max,
            # Schedule A is a during-the-year disclosure. These rows are the
            # ones a reader would otherwise take for current positions.
            "no_longer_held": "no longer held" in (asset["detail"] or "").lower(),
            # An EIF's underlying holdings are legally not itemised, so no
            # amount of parsing will see inside one.
            "excepted_investment_fund": "excepted investment fund" in
                                        (income_type or "").lower(),
            **value,
        }
        result["rows"].append(row)

    result["holdings"] = [r for r in result["rows"] if not r["is_container"]]
    return result


# ------------------------------------------------------------------ fetching

SENATE_ANNUAL_VIEW = "https://efdsearch.senate.gov/search/view/annual/{uuid}/"

# Verified by set arithmetic against the live search API: [7] returns 2,462
# records and [7,5,6,8,12] returns the same 2,462, so 7 is the umbrella that
# already contains Candidate, New Filer, Termination and paper amendments.
# Requesting the leaves separately adds nothing and risks missing a code.
SENATE_ANNUAL_REPORT_TYPE = 7
SENATE_SENATOR_FILER_TYPE = 1


def search_senate_annuals(session, since: str, limit: int = 200) -> List[Dict[str, Any]]:
    """Senate annual reports submitted on or after `since` (MM/DD/YYYY).

    Empty date bounds make the search endpoint answer 503 rather than an
    error, so `since` is required rather than defaulted away.
    """
    from .congress_trades import SENATE_SEARCH, DisclosureUnavailable

    if not since:
        raise DisclosureUnavailable(
            "a start date is required: the Senate search answers 503 to an "
            "unbounded query rather than returning an error")
    try:
        response = session.post(
            SENATE_SEARCH, timeout=30,
            headers={"Referer": "https://efdsearch.senate.gov/search/",
                     "X-CSRFToken": session.cookies.get("csrftoken", ""),
                     "X-Requested-With": "XMLHttpRequest"},
            data={"start": 0, "length": limit,
                  "report_types": f"[{SENATE_ANNUAL_REPORT_TYPE}]",
                  "filer_types": f"[{SENATE_SENATOR_FILER_TYPE}]",
                  "submitted_start_date": f"{since} 00:00:00",
                  "submitted_end_date": "", "candidate_state": "",
                  "senator_state": "", "office_id": "", "first_name": "",
                  "last_name": ""})
        response.raise_for_status()
        rows = response.json().get("data", [])
    except Exception as exc:  # noqa: BLE001 - surfaced, never masked
        raise DisclosureUnavailable(
            f"the Senate annual-report search failed: {exc}") from exc

    filings = []
    for row in rows:
        if len(row) < 5:
            continue
        first, last, office, link, filed = row[:5]
        match = re.search(r"/search/view/(\w+)/([0-9a-f-]+)/", link or "")
        label = re.sub(r"<[^>]+>", "", link or "").strip()
        year = re.search(r"\bCY\s*(\d{4})", label)
        amendment = re.search(r"Amendment\s+(\d+)", label, re.I)
        filings.append({
            "first": first, "last": last, "office": office,
            "filed_date": filed, "label": label,
            # "paper" means scanned GIF page images, which no parser reads.
            "kind": match.group(1) if match else None,
            "uuid": match.group(2) if match else None,
            "calendar_year": int(year.group(1)) if year else None,
            "amendment": int(amendment.group(1)) if amendment else 0,
        })
    return filings


def fetch_senate_annual(session, uuid: str) -> Dict[str, Any]:
    from .congress_trades import DisclosureUnavailable

    url = SENATE_ANNUAL_VIEW.format(uuid=uuid)
    try:
        response = session.get(url, timeout=30)
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001 - surfaced, never masked
        raise DisclosureUnavailable(
            f"Senate annual report {uuid} could not be read: {exc}") from exc

    parsed = parse_senate_annual(response.text)
    parsed["source_url"] = url
    parsed["uuid"] = uuid
    return parsed


def latest_amendments(filings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One filing per (senator, calendar year): the highest amendment.

    Amendments are full restatements rather than deltas. Merging them would
    double every unchanged holding, and taking the base would silently keep a
    figure the filer has since corrected.
    """
    best: Dict[tuple, Dict[str, Any]] = {}
    for filing in filings:
        key = (filing["first"], filing["last"], filing["calendar_year"])
        if key not in best or filing["amendment"] > best[key]["amendment"]:
            best[key] = filing
    return list(best.values())
