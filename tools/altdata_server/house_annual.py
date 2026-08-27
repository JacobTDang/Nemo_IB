"""House annual financial disclosures -- the House half of "what was held".

The Senate publishes annual reports as HTML tables with a cell per column.
The House publishes a PDF, and `extract_text()` on it is a trap: the asset
name wraps, the value bracket wraps, the income bracket wraps *into the next
visual line*, a nested holding puts the parent account and the child asset on
two separate lines, and the location/description strips run the full width of
the table so their words land in whichever column they happen to sit under.
Every one of those breaks a line-based regex, and each break is silent.

So this reads the table the way the renderer drew it.

**Columns come from the header rectangles.** The header row is six filled
grey rectangles (0.9333) whose x-edges are the column boundaries. They differ
between documents -- a Candidate report's Asset column is 25pt narrower than
an Annual report's -- so they are measured per document, never assumed.

**Rows come from the alternating shading.** There are no rules between rows.
Every other entry is painted with a background fill (0.9608) that covers that
entry exactly, description strip included, and the unshaded entries are the
gaps between them. That shading is the only thing in the file that says where
one holding stops and the next begins.

**Entries split across page breaks.** The shading alternates continuously
through the document, so a page that begins with the same shade its
predecessor ended on is showing the *rest* of that entry, not a new one.
Measured against real filings this is the single largest error source if
skipped: a wrapped asset name arrives as a row with no value, and the row it
belonged to arrives with no asset.

**Small-cap glyphs extract as NUL bytes.** `Schedule A: Assets and "Unearned"
Income` arrives as `S\\x00\\x00\\x00\\x00\\x00\\x00\\x00 A: A\\x00...`, and
`\\s` does not match NUL, so anything matched before normalising silently
never fires. That exact bug already bit the PTR parser once.

Four semantics this refuses to let a caller lose
------------------------------------------------

**`None` in the value column is a sale, not a zero.** It is what a filer
selects for an asset that was disposed of during the year but still threw off
more than $200 of income. Reading it as "a holding worth nothing" keeps an
exited position on the books. Its year-end value really is zero, so the
bounds are zero -- and `no_longer_held` is what says why.

**`Undetermined` has no bounds at all.** Not zero. A holding nobody could
price and a holding worth nothing are different disclosures.

**`Over $50,000,000` and `Spouse/DC Over $1,000,000` are floors with no
ceiling.** `value_max` is None rather than a repeat of the floor. Treating
the top bracket as zero-width once produced a Senate portfolio whose minimum
exceeded its maximum.

**A parent account is not a position.** Nested holdings are written
`Account ⇒ Asset`; when the account also files a row of its own, that row is
a container and counting it alongside its contents double-counts the book.

New Filer and Candidate reports drop the `Tx. > $1,000?` column and carry two
income columns instead. So does an Amendment to one of those, while an
Amendment to an annual report keeps the annual layout -- which is why the
layout is read off the header text rather than off the filing type.

The `Tx. > $1,000?` column itself is drawn as a vector glyph rather than a
character and has never extracted as text, so `tx_over_1000` is left None
rather than being reported as a False nobody read.
"""
from __future__ import annotations

import io
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .congress_trades import (DisclosureUnavailable, HTTP_TIMEOUT, _user_agent,
                              fetch_house_index)

HOUSE_ANNUAL_URL = ("https://disclosures-clerk.house.gov/public_disc/"
                    "financial-pdfs/{year}/{doc_id}.pdf")

# Every filing type that carries a Schedule A. "P" (Periodic Transaction
# Report) is the one that does not, and it lives under ptr-pdfs/ anyway.
ANNUAL_FILING_TYPES = {
    "O": "annual",          # FD Original
    "A": "amendment",
    "H": "new_filer",
    "T": "termination",
    "C": "candidate",
}

_FILING_TYPE_TEXT = {
    "annual report": "annual",
    "amendment report": "amendment",
    "new filer report": "new_filer",
    "terminated filer report": "termination",
    "termination report": "termination",
    "candidate report": "candidate",
}

# The renderer's two greys. The header band and the every-other-row fill.
HEADER_GREY = 0.9333
ROW_GREY = 0.9608
_GREY_TOLERANCE = 0.004

# The official Schedule A value tiers, in filing order. `Over $50,000,000`
# and `Spouse/DC Over $1,000,000` end with no ceiling because the disclosure
# ends there; nothing above them is ever reported.
VALUE_BRACKETS: Dict[str, Tuple[Optional[int], Optional[int]]] = {
    "None": (0, 0),
    "$1 - $1,000": (1, 1_000),
    "$1,001 - $15,000": (1_001, 15_000),
    "$15,001 - $50,000": (15_001, 50_000),
    "$50,001 - $100,000": (50_001, 100_000),
    "$100,001 - $250,000": (100_001, 250_000),
    "$250,001 - $500,000": (250_001, 500_000),
    "$500,001 - $1,000,000": (500_001, 1_000_000),
    "$1,000,001 - $5,000,000": (1_000_001, 5_000_000),
    "$5,000,001 - $25,000,000": (5_000_001, 25_000_000),
    "$25,000,001 - $50,000,000": (25_000_001, 50_000_000),
    "Over $50,000,000": (50_000_000, None),
    "Spouse/DC Over $1,000,000": (1_000_000, None),
}

INCOME_BRACKETS: Dict[str, Tuple[Optional[int], Optional[int]]] = {
    "None": (0, 0),
    "$1 - $200": (1, 200),
    "$201 - $1,000": (201, 1_000),
    "$1,001 - $2,500": (1_001, 2_500),
    "$2,501 - $5,000": (2_501, 5_000),
    "$5,001 - $15,000": (5_001, 15_000),
    "$15,001 - $50,000": (15_001, 50_000),
    "$50,001 - $100,000": (50_001, 100_000),
    "$100,001 - $1,000,000": (100_001, 1_000_000),
    "$1,000,001 - $5,000,000": (1_000_001, 5_000_000),
    "Over $5,000,000": (5_000_000, None),
}

UNDETERMINED = "undetermined"
_SPOUSE_CAP = "spouse/dc over"

_OWNER = {"SP": "spouse", "JT": "joint", "DC": "dependent_child"}

# The House's own asset type codes, as printed in `[ST]` after the name.
ASSET_TYPES = {
    "5C": "529 college savings plan (cash)",
    "5F": "529 college savings plan (fund)",
    "5P": "529 prepaid tuition plan",
    "AB": "asset-backed security",
    "BA": "bank deposit",
    "BK": "brokerage account",
    "CO": "collectible",
    "CS": "corporate security (bond or note)",
    "CT": "cryptocurrency",
    "DB": "defined benefit pension",
    "DO": "debt obligation",
    "DS": "stock appreciation right",
    "EF": "exchange traded fund",
    "EQ": "excepted/qualified blind trust",
    "ET": "exchange traded note",
    "FA": "farm",
    "FE": "foreign exchange position",
    "FN": "fixed annuity",
    "FU": "futures",
    "GS": "government security",
    "HE": "hedge fund",
    "HN": "hedge fund (non-public)",
    "IC": "investment club",
    "IH": "cash/money market/other holding",
    "IP": "intellectual property/royalties",
    "MA": "managed account",
    "MF": "mutual fund",
    "MO": "mineral/oil/gas rights",
    "OI": "ownership interest (holding investments)",
    "OL": "ownership interest (engaged in a trade or business)",
    "OP": "options",
    "OT": "other",
    "PE": "pension",
    "PM": "precious metals",
    "PS": "stock (privately held)",
    "RE": "real estate invest. trust (REIT)",
    "RP": "real property",
    "RS": "restricted stock unit",
    "SA": "stock appreciation right",
    "ST": "stock (publicly traded)",
    "TR": "trust",
    "VA": "variable annuity",
    "VI": "variable insurance",
    "WU": "whole/universal insurance",
}

# `Walmart Inc. - Common Stock (WMT) [ST]`. One to six upper-case letters,
# so a nine-character CUSIP cannot match and `(1)` -- the House's own
# disambiguator on repeated account names -- cannot either. `MET$E` is a real
# House ticker for a depositary share, hence the `$`.
_TICKER = re.compile(r"\(([A-Z][A-Z.$]{0,5})\)")
_ASSET_CODE = re.compile(r"\[([A-Z0-9]{2})\]\s*$")
# Things that sit in parentheses and are emphatically not tickers.
_NOT_TICKERS = {"LLC", "LP", "LLP", "INC", "IRA", "ETF", "US", "USA", "PLC",
                "SP", "JT", "DC", "HSA", "TSP", "REIT", "CD", "UK", "NA",
                "N.A.", "LTD", "CO.", "II", "III", "IV", "SEP", "UTMA",
                "UGMA", "ESOP", "TOD", "JTWROS", "RSU", "IRS"}

# A location/description/comment strip: a small-caps label whose interior
# glyphs extract as NUL. `L\x00\x00\x00\x00\x00\x00\x00:` is LOCATION.
_STRIP_LABEL = re.compile(r"^([A-Za-z])\x00*:$")
_STRIP_KINDS = {"L": "location", "D": "description", "C": "comment"}

_SCHEDULE_A_HEADER = "Asset Owner Value of Asset"
_VEHICLE_HEADING = re.compile(r"^S\s+A\s+B\s+I\s+V\s+D\b")
_FOOTNOTE = re.compile(r"complete list of asset type", re.I)


# --------------------------------------------------------------- primitives

def normalise_glyphs(text: str) -> str:
    """Small-cap glyphs arrive as NUL. Make them ordinary spaces.

    `\\s` does not match NUL, so every pattern in this module runs after
    this. Skipping it is how a terminator silently never fires.
    """
    return (text or "").replace("\x00", " ")


def _flat(text: str) -> str:
    return re.sub(r"\s+", " ", normalise_glyphs(text)).strip()


def _money(text: str) -> Optional[int]:
    """`$131,432.00` is 131432, not 13143200."""
    match = re.search(r"[\d,]+(?:\.\d+)?", text or "")
    if not match:
        return None
    return int(float(match.group(0).replace(",", "")))


def _amounts(text: str) -> List[int]:
    return [_money(a) for a in re.findall(r"\$\s?[\d,]+(?:\.\d+)?", text or "")]


def parse_value_bracket(text: str) -> Dict[str, Any]:
    """A Schedule A value cell, as bounds plus the reason when it has none.

    Never invents a zero for something it could not read, and never gives the
    open top brackets a ceiling.
    """
    raw = _flat(text)
    out: Dict[str, Any] = {
        "value_text": raw or None,
        "value_min": None,
        "value_max": None,
        "value_unascertainable": False,
        "spouse_capped": False,
        # Schedule A's `None` is not "nothing here". It is the option a filer
        # picks for an asset sold during the year that still paid out more
        # than $200 -- an exited position, which is a different disclosure
        # from a holding worth nothing.
        "no_longer_held": False,
        "value_canonical": False,
    }
    if not raw:
        return out
    if raw in VALUE_BRACKETS:
        out["value_min"], out["value_max"] = VALUE_BRACKETS[raw]
        out["value_canonical"] = True
        out["no_longer_held"] = raw == "None"
        out["spouse_capped"] = raw.lower().startswith(_SPOUSE_CAP)
        return out
    if UNDETERMINED in raw.lower():
        # The filer could not price it. Both bounds stay None; a zero here
        # would read as a worthless holding, which is not what was filed.
        out["value_unascertainable"] = True
        return out

    # Not one of the thirteen tiers. Read what bounds are legible rather than
    # dropping the row, and leave `value_canonical` False so a caller can
    # count how often that happens.
    amounts = [a for a in _amounts(raw) if a is not None]
    if raw.lower().startswith("spouse/dc over") and amounts:
        out["value_min"] = amounts[0]
        out["spouse_capped"] = True
    elif raw.lower().startswith("over") and amounts:
        out["value_min"] = amounts[0]
    elif len(amounts) >= 2:
        out["value_min"], out["value_max"] = amounts[0], amounts[1]
    elif amounts:
        out["value_min"] = out["value_max"] = amounts[0]
    return out


def parse_income_bracket(text: str) -> Tuple[Optional[int], Optional[int]]:
    """The income cell, which is a bracket unless the type is `Other`.

    `Other` income is filed as a literal figure -- `$131,432.00` -- so this
    has to cope with both.
    """
    raw = _flat(text)
    if not raw:
        return (None, None)
    if raw in INCOME_BRACKETS:
        return INCOME_BRACKETS[raw]
    lowered = raw.lower()
    if lowered in ("n/a", "not applicable", "tax-deferred"):
        # Tax-deferred income is not disclosed at all, and "not applicable"
        # is a refusal to state one. Zero would be a claim neither made.
        return (None, None)
    amounts = [a for a in _amounts(raw) if a is not None]
    if not amounts:
        return (None, None)
    if lowered.startswith("over"):
        return (amounts[0], None)
    if len(amounts) >= 2:
        return (amounts[0], amounts[1])
    return (amounts[0], amounts[0])


def split_asset_cell(text: str) -> Dict[str, Any]:
    """`Account ⇒ Walmart Inc. - Common Stock (WMT) [ST]` pulled apart.

    Roughly four rows in five are nested, so the parent account has to come
    off before anything else is read: the ticker and the type code belong to
    the child, and the name that matters for matching is the child's.
    """
    raw = _flat(text)
    # Nesting can go more than one level deep: `Our Hidden Lake LLC ⇒ UBS
    # Brokerage 2 ⇒ Allstate Corporation Preferred` is an account inside an
    # LLC. Splitting on the first arrow leaves the middle account welded to
    # the front of the asset name, so the whole chain comes off.
    chain = [part.strip() for part in raw.split("⇒")]
    leaf = chain.pop().strip()
    parent_chain = [part for part in chain if part]
    parent = parent_chain[-1] if parent_chain else None

    code_match = _ASSET_CODE.search(leaf)
    code = code_match.group(1) if code_match else None
    name = leaf[:code_match.start()].strip() if code_match else leaf

    ticker = None
    for candidate in _TICKER.findall(name):
        if candidate.strip(".") in _NOT_TICKERS:
            continue
        ticker = candidate  # the last plausible one: `Kroger Company (KR)`
    if ticker:
        name = re.sub(rf"\s*\({re.escape(ticker)}\)\s*", " ", name).strip()
    name = re.sub(r"\s*[-–]\s*$", "", re.sub(r"\s+", " ", name)).strip()

    return {
        "raw_asset": raw,
        "parent_account": parent,
        "parent_chain": parent_chain,
        "asset_name": name or None,
        "ticker": ticker,
        "asset_type_code": code,
        "asset_type": ASSET_TYPES.get(code) if code else None,
    }


def is_scanned_doc_id(doc_id: str) -> bool:
    """A seven-digit DocID is a scan of a paper filing; eight is electronic.

    Measured across 1,330 filings: every seven-digit id carried no
    extractable text and every eight-digit one parsed. It is the *length*
    that discriminates, not the leading digit.
    """
    return len(str(doc_id).strip()) == 7


# ------------------------------------------------------------------ geometry

def _grey(rect: Dict[str, Any]) -> Optional[float]:
    """The fill level of a rectangle, or None if it is not a flat grey."""
    colour = rect.get("non_stroking_color")
    if colour is None:
        return None
    if isinstance(colour, (int, float)):
        return round(float(colour), 4)
    try:
        values = list(colour)
    except TypeError:
        return None
    if len(values) == 1:
        return round(float(values[0]), 4)
    if len(values) == 3 and max(values) - min(values) < 1e-6:
        return round(float(values[0]), 4)
    return None


def page_geometry(page) -> Dict[str, Any]:
    """The only place a pdfplumber page is touched.

    Everything downstream works on plain dicts, so the table reader can be
    exercised against captured geometry without a PDF in the loop.
    """
    return {
        "width": float(page.width),
        "height": float(page.height),
        "text": page.extract_text() or "",
        "rects": [{"x0": float(r["x0"]), "x1": float(r["x1"]),
                   "top": float(r["top"]), "bottom": float(r["bottom"]),
                   "grey": _grey(r)} for r in page.rects],
        "words": [{"text": w["text"], "x0": float(w["x0"]),
                   "x1": float(w["x1"]), "top": float(w["top"])}
                  for w in page.extract_words()],
    }


def _is_grey(rect: Dict[str, Any], level: float) -> bool:
    grey = rect.get("grey")
    return grey is not None and abs(grey - level) < _GREY_TOLERANCE


def _lines(words: Sequence[Dict[str, Any]], tolerance: float = 3.0
           ) -> List[List[Dict[str, Any]]]:
    """Group words into visual lines.

    A wrapped income bracket sits a fraction of a point above the asset text
    it wraps beside -- `$2,500` at 190.97 against `Kroger` at 191.72 -- so
    rounding the top to a key splits one line in two. Cluster instead.
    """
    out: List[List[Dict[str, Any]]] = []
    for word in sorted(words, key=lambda w: (w["top"], w["x0"])):
        if out and word["top"] - out[-1][0]["top"] <= tolerance:
            out[-1].append(word)
        else:
            out.append([word])
    return [sorted(line, key=lambda w: w["x0"]) for line in out]


def _header_bands(page: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Every drawn header band on the page, with its column edges and text."""
    bands: Dict[float, List[Dict[str, Any]]] = {}
    for rect in page["rects"]:
        if not _is_grey(rect, HEADER_GREY):
            continue
        if rect["bottom"] - rect["top"] < 4:
            continue  # a hairline border, not the band
        bands.setdefault(round(rect["top"], 1), []).append(rect)

    out = []
    for top, rects in sorted(bands.items()):
        bottom = max(r["bottom"] for r in rects)
        edges = sorted({round(r["x0"], 2) for r in rects} |
                       {round(r["x1"], 2) for r in rects})
        text = _flat(" ".join(
            w["text"] for w in sorted(
                (w for w in page["words"] if top - 1 <= w["top"] < bottom),
                key=lambda w: (w["top"], w["x0"]))))
        out.append({"top": top, "bottom": bottom, "edges": edges, "text": text})
    return out


def _row_spans(page: Dict[str, Any], top: float, bottom: float
               ) -> List[Tuple[float, float, bool]]:
    """Where each entry starts and stops, from the alternating shading.

    The shaded fills give the shaded entries exactly, description strip
    included. The unshaded entries are precisely the gaps between them, since
    the fills run edge to edge with no padding between rows.
    """
    shaded: List[List[float]] = []
    for rect in page["rects"]:
        if not _is_grey(rect, ROW_GREY):
            continue
        # An entry continued from the previous page has its fill painted from
        # the top of the page's content area, with the repeated header band
        # drawn over it. Dropping a rectangle for overhanging the region
        # therefore loses exactly the continuation rows, which are the ones
        # the stitching needs to see.
        start = max(rect["top"], top)
        end = min(rect["bottom"], bottom)
        if end - start > 0.5:
            shaded.append([start, end])
    shaded.sort()

    merged: List[List[float]] = []
    for start, end in shaded:
        if merged and start <= merged[-1][1] + 0.05:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    spans: List[Tuple[float, float, bool]] = []
    cursor = top
    for start, end in merged:
        if start - cursor > 1:
            spans.append((cursor, start, False))
        spans.append((start, end, True))
        cursor = end
    if bottom - cursor > 1:
        spans.append((cursor, bottom, False))
    return spans


def _schedule_a_blocks(pages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One block per drawn entry, in document order, page breaks unresolved."""
    blocks: List[Dict[str, Any]] = []
    for index, page in enumerate(pages):
        bands = _header_bands(page)
        header = next((b for b in bands
                       if b["text"].startswith(_SCHEDULE_A_HEADER)), None)
        if header is None:
            continue

        # Clip at whatever comes next: the asset-code footnote, or the next
        # schedule's header band when two schedules share a page. Without the
        # clip the footnote's words bleed into the last row's cells.
        stop = page["height"]
        for band in bands:
            if header["bottom"] < band["top"] < stop:
                stop = band["top"]
        for line in _lines([w for w in page["words"]
                            if w["top"] > header["bottom"]]):
            text = _flat(" ".join(w["text"] for w in line))
            if text.startswith("*") or _FOOTNOTE.search(text):
                stop = min(stop, line[0]["top"] - 1)
                break

        page_blocks = []
        for start, end, shaded in _row_spans(page, header["bottom"], stop):
            words = [w for w in page["words"] if start - 0.6 <= w["top"] < end - 0.5]
            # A wordless block is kept unless it is the last on the page. An
            # entry whose final line fell exactly on the break leaves nothing
            # but its padding at the top of the next page, and that empty
            # block still carries the shade the alternation is counted in --
            # drop it and the parity flips, welding two unrelated entries
            # together. The block below the last entry, on the other hand, is
            # only the page's bottom margin and never an entry at all.
            page_blocks.append({"page": index, "shaded": shaded,
                                "edges": header["edges"],
                                "header": header["text"],
                                "lines": _lines(words)})
        while page_blocks and not page_blocks[-1]["lines"]:
            page_blocks.pop()
        blocks.extend(page_blocks)
    return blocks


def _stitch(blocks: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rejoin entries the page break cut in half.

    The shading alternates entry by entry and keeps alternating across pages,
    so a page whose first entry is drawn in the same shade its predecessor's
    last entry was drawn in is not a new entry -- it is the remainder of that
    one. Without this an asset name that wrapped over the break arrives as a
    row with no value, and the row it belongs to arrives with no asset.
    """
    entries: List[Dict[str, Any]] = []
    for block in blocks:
        previous = entries[-1] if entries else None
        if (previous is not None
                and block["page"] != previous["blocks"][-1]["page"]
                and block["shaded"] == previous["blocks"][-1]["shaded"]):
            previous["blocks"].append(block)
            previous["lines"].extend(block["lines"])
            continue
        entries.append({"edges": block["edges"], "header": block["header"],
                        "page": block["page"], "shaded": block["shaded"],
                        "blocks": [block], "lines": list(block["lines"])})
    return [e for e in entries if e["lines"]]


def _column_of(word: Dict[str, Any], edges: Sequence[float]) -> int:
    centre = (word["x0"] + word["x1"]) / 2
    for index in range(len(edges) - 1):
        if edges[index] - 0.5 <= centre < edges[index + 1]:
            return index
    return 0 if centre < edges[0] else len(edges) - 2


def _entry_cells(entry: Dict[str, Any]) -> Tuple[List[str], List[Tuple[str, str]]]:
    """The entry's six cells, plus its location/description/comment strips.

    The strips have to come off first. They are drawn across the full width
    of the table, so if they are left in, `Shares were forfeited when spouse
    terminated her employment` scatters `spouse` into the Owner column and
    `84.51` into Value.
    """
    edges = entry["edges"]
    columns: List[List[Tuple[int, float, str]]] = [[] for _ in range(len(edges) - 1)]
    strips: List[Tuple[str, str]] = []
    in_strip = False
    kind = "description"

    for order, line in enumerate(entry["lines"]):
        first = line[0]
        label = _STRIP_LABEL.match(first["text"])
        if label and abs(first["x0"] - edges[0]) < 8:
            in_strip = True
            kind = _STRIP_KINDS.get(label.group(1).upper(), "description")
            strips.append((kind, _flat(" ".join(w["text"] for w in line[1:]))))
            continue
        if in_strip:
            # A long description wraps with no label on the following line.
            body = _flat(" ".join(w["text"] for w in line))
            if strips:
                strips[-1] = (strips[-1][0], f"{strips[-1][1]} {body}".strip())
            else:
                strips.append((kind, body))
            continue
        for word in line:
            columns[_column_of(word, edges)].append((order, word["x0"], word["text"]))

    cells = []
    for column in columns:
        column.sort()
        cells.append(_flat(" ".join(text for _, _, text in column)))
    return cells, strips


# ------------------------------------------------------------------ metadata

def _metadata(pages: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    text = normalise_glyphs(pages[0]["text"]) if pages else ""

    def field(name: str) -> Optional[str]:
        match = re.search(rf"^{name}:\s*(.+)$", text, re.M)
        return match.group(1).strip() if match else None

    member = field("Name")
    if member:
        member = re.sub(r"^(Hon\.|Mr\.|Mrs\.|Ms\.|Dr\.)\s+", "", member).strip()

    filing_type = field("Filing Type") or ""
    kind = _FILING_TYPE_TEXT.get(filing_type.strip().lower())

    year = field("Filing Year")
    filed = field("Filing Date")
    period = field("Period Covered")

    covered_to = None
    if period:
        dates = re.findall(r"\d{2}/\d{2}/\d{4}", period)
        if dates:
            covered_to = _iso(dates[-1])

    return {
        "member": member,
        "status": field("Status"),
        "state_district": field("State/District"),
        "report_kind": kind,
        "filing_type_text": filing_type or None,
        "calendar_year": int(year) if year and year.isdigit() else None,
        "filed_date": _iso(filed) if filed else None,
        "period_covered": period,
        "period_covered_to": covered_to,
    }


def _iso(date_text: str) -> Optional[str]:
    match = re.search(r"(\d{2})/(\d{2})/(\d{4})", date_text or "")
    if not match:
        return None
    month, day, year = match.groups()
    return f"{year}-{month}-{day}"


def _investment_vehicles(pages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The trailing `Schedule A and B Investment Vehicle Details` section.

    It is where the parent account on the left of every `⇒` is defined, and
    the only place an account's owner is stated when the entries that sit
    inside it leave the Owner column blank.
    """
    vehicles: Dict[str, Dict[str, Any]] = {}
    collecting = False
    for page in pages:
        for raw_line in page["text"].split("\n"):
            raw_line = raw_line.strip()
            line = _flat(raw_line)
            if not line:
                continue
            # Every section heading in the document is set in the small-caps
            # face, so it is exactly the lines carrying NUL that open and
            # close this one. A vehicle's name never does.
            if "\x00" in raw_line:
                if _VEHICLE_HEADING.match(line):
                    collecting = True
                elif collecting and not _STRIP_LABEL.match(raw_line.split(" ")[0]):
                    collecting = False
                continue
            if not collecting:
                continue
            owner = re.search(r"\(Owner:\s*([A-Z]{2})\)", line)
            interest = re.search(r"\((\d+(?:\.\d+)?)%\s*Interest\)", line)
            name = re.sub(r"\s*\((?:Owner:[^)]*|[\d.]+%\s*Interest)\)",
                          "", line).strip()
            if not name:
                continue
            # The same account is listed twice when it holds both Schedule A
            # and Schedule B entries, once with its owner and once with the
            # filer's percentage interest. It is one account.
            entry = vehicles.setdefault(
                name, {"name": name, "owner": None, "interest_pct": None})
            if owner and entry["owner"] is None:
                entry["owner"] = _OWNER.get(owner.group(1), "self")
            if interest and entry["interest_pct"] is None:
                entry["interest_pct"] = float(interest.group(1))
    return list(vehicles.values())


# -------------------------------------------------------------------- parse

def parse_house_annual(pages: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """One House annual-style report: header, every Schedule A row, the leaves.

    `pages` is a list of `page_geometry()` dicts.
    """
    meta = _metadata(pages)
    kind = meta["report_kind"] or "annual"
    year = meta["calendar_year"]

    if kind == "annual" and year:
        as_of = f"{year}-12-31"
    else:
        # A Candidate, New Filer or Termination report covers a stub period
        # that ends when it ends. Stamping it to a year end would invent a
        # period the filing never covered.
        as_of = meta["period_covered_to"] or meta["filed_date"] or (
            f"{year}-12-31" if year else None)

    result: Dict[str, Any] = {
        "member": meta["member"],
        "status": meta["status"],
        "state_district": meta["state_district"],
        "report_kind": kind,
        "filing_type_text": meta["filing_type_text"],
        "calendar_year": year,
        "as_of": as_of,
        "filed_date": meta["filed_date"],
        "period_covered": meta["period_covered"],
        "layout": None,
        "has_assets_table": False,
        "investment_vehicles": _investment_vehicles(pages),
        "rows": [],
        "holdings": [],
    }

    entries = _stitch(_schedule_a_blocks(pages))
    if not entries:
        return result

    header = entries[0]["header"]
    # New Filer, Amendment and Candidate reports drop `Tx. > $1,000?` and
    # carry two income columns instead. Branch on what was drawn.
    dual_income = "Preceding" in header
    result["layout"] = "dual_income" if dual_income else "annual"
    result["has_assets_table"] = True

    parsed: List[Dict[str, Any]] = []
    for number, entry in enumerate(entries, start=1):
        cells, strips = _entry_cells(entry)
        cells = (cells + [""] * 6)[:6]
        asset = split_asset_cell(cells[0])
        value = parse_value_bracket(cells[2])
        income_type = cells[3] or None
        income_min, income_max = parse_income_bracket(cells[4])

        row: Dict[str, Any] = {
            "row_number": str(number),
            "page": entry["page"],
            "spans_page_break": sum(1 for b in entry["blocks"] if b["lines"]) > 1,
            "depth": len(asset["parent_chain"]) + 1,
            "parent_account": asset["parent_account"],
            "parent_chain": asset["parent_chain"],
            "parent_row": None,
            "is_container": False,
            "ticker": asset["ticker"],
            "asset_name": asset["asset_name"],
            "raw_asset": asset["raw_asset"],
            "asset_type_code": asset["asset_type_code"],
            "asset_type": asset["asset_type"],
            "asset_subtype": None,
            "owner": _OWNER.get(cells[1].strip(), "self" if not cells[1].strip()
                                else cells[1].strip().lower()),
            "location": next((t for k, t in strips if k == "location"), None),
            "asset_detail": " ".join(t for k, t in strips
                                     if k in ("description", "comment")) or None,
            "income_type": income_type,
            "income_min": income_min,
            "income_max": income_max,
            "excepted_investment_fund": False,
            **value,
        }
        if dual_income:
            row["income_preceding_min"], row["income_preceding_max"] = \
                parse_income_bracket(cells[5])
            row["income_preceding_text"] = cells[5] or None
            row["tx_over_1000"] = None
        else:
            # The `Tx. > $1,000?` box is drawn as a vector glyph, not a
            # character: across 6,996 measured rows the cell has never
            # yielded a single word, and the drawn box is byte-identical
            # whether or not it is ticked. So an unmarked cell is `None`,
            # meaning unread -- never `False`, which would assert that
            # thousands of members reported no qualifying transaction when
            # nobody has actually read the answer.
            row["tx_over_1000"] = True if cells[5].strip() else None
            row["income_preceding_min"] = row["income_preceding_max"] = None
            row["income_preceding_text"] = None
        parsed.append(row)

    # A parent account that also files a row of its own is a container: its
    # value already covers everything nested beneath it, so counting both
    # double-counts the book.
    # Having children is what makes a container, exactly as in the Senate
    # parser -- not merely appearing in the Investment Vehicle Details list.
    # An account listed there whose contents are all Schedule B transactions
    # has no Schedule A children, and dropping its row would delete a real
    # holding rather than a double count.
    parents = {name for r in parsed for name in r["parent_chain"]}
    by_name: Dict[str, str] = {}
    for row in parsed:
        if row["depth"] == 1 and row["asset_name"] in parents:
            row["is_container"] = True
            by_name[row["asset_name"]] = row["row_number"]
    for row in parsed:
        if row["parent_account"] in by_name:
            row["parent_row"] = by_name[row["parent_account"]]

    result["rows"] = parsed
    result["holdings"] = [r for r in parsed if not r["is_container"]]
    return result


def parse_house_annual_pdf(data: bytes) -> Dict[str, Any]:
    import pdfplumber

    with pdfplumber.open(io.BytesIO(data)) as pdf:
        pages = [page_geometry(page) for page in pdf.pages]
    if not any(page["text"].strip() for page in pages):
        raise DisclosureUnavailable(
            "the filing carries no extractable text; it is a scan of a paper "
            "filing, not an electronic one")
    return parse_house_annual(pages)


# ------------------------------------------------------------------ fetching

def list_house_annuals(year: int, session=None,
                       filing_types: Optional[Iterable[str]] = None
                       ) -> List[Dict[str, Any]]:
    """Every Schedule A-bearing House filing for a coverage year.

    `year` is the year the report covers, which is also the directory the PDF
    lives in -- not the year it was filed. A CY2025 annual report is filed in
    2026 and still sits under `financial-pdfs/2025/`.
    """
    wanted = set(filing_types) if filing_types else set(ANNUAL_FILING_TYPES)
    out = []
    for filing in fetch_house_index(year, session=session):
        if filing["filing_type"] not in wanted:
            continue
        doc_id = filing["doc_id"]
        out.append({
            **filing,
            "report_kind": ANNUAL_FILING_TYPES[filing["filing_type"]],
            "coverage_year": int(filing["year"]) if filing["year"].isdigit() else year,
            "is_scan": is_scanned_doc_id(doc_id),
            "source_url": HOUSE_ANNUAL_URL.format(
                year=filing["year"] or year, doc_id=doc_id),
        })
    return out


def fetch_house_annual(doc_id: str, year: int, session=None) -> Dict[str, Any]:
    """Download and parse one House annual-style report."""
    import requests

    getter = session or requests
    url = HOUSE_ANNUAL_URL.format(year=year, doc_id=doc_id)
    if is_scanned_doc_id(doc_id):
        raise DisclosureUnavailable(
            f"House filing {doc_id} ({year}) has a seven-digit document id, "
            f"which is a scan of a paper filing and carries no extractable "
            f"text. Source: {url}")
    try:
        response = getter.get(url, timeout=HTTP_TIMEOUT,
                              headers={"User-Agent": _user_agent()})
        response.raise_for_status()
    except DisclosureUnavailable:
        raise
    except Exception as exc:  # noqa: BLE001 - surfaced, never masked
        raise DisclosureUnavailable(
            f"House annual report {doc_id} ({year}) could not be read: {exc}"
        ) from exc

    try:
        parsed = parse_house_annual_pdf(response.content)
    except DisclosureUnavailable as exc:
        raise DisclosureUnavailable(
            f"House annual report {doc_id} ({year}): {exc}. Source: {url}"
        ) from exc
    parsed["doc_id"] = doc_id
    parsed["source_url"] = url
    return parsed
