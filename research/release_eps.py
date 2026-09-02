"""The time-series surprise, dated from the earnings release.

`sue.sue_ts` computes the surprise from XBRL, and XBRL does not exist until the
10-Q is filed -- a median of eight days after the earnings 8-K, mean 12, range
0 to 45. The published drift is dated from the announcement and most of it
happens in the first days. A full replay made the cost of that lag concrete:
over the same 230 dates, the XBRL-timed arm returned a median trade of -36bp
with a coefficient whose interval contained zero, while the release-timed
cross-sectional arm returned +122bp mean against -20bp.

Only one number has to move earlier. The standardisation needs eight quarters
of history, and those are months old and safely XBRL. The quarter being
announced is the one whose EPS must come from the 8-K, with the 8-K's date as
`known_at`. Everything else in the signal -- the year-on-year deltas, their
sigma, the basis handling -- is `sue._signal_from_series`, unchanged.

The whole risk is reading the right number out of a release, and four names
measured against the 10-Q's XBRL showed the shapes of it:

  Apple      6 of 6 right. One clean sentence.
  Photronics one wrong. Q4 releases lead with the fiscal year, so the first
             figure was the year's $2.28 and the quarter's $1.07 came later.
  JPMorgan   0 of 12 read. Banks say "per share" and never "diluted" -- and
             the exhibit is named `exhibit991narrative`, which a substring
             check on `ex99` missed in favour of `ex992supplement`, tables only.
  Rivian     0 of 19 read. The loss is only in a table whose columns run
             prior-year first, so which column is "this quarter" cannot be
             known from the row at all.

Each is a rule, not a heuristic tuned to a name: a sentence naming the quarter
outranks one that does not; an unqualified "per share" is accepted only beside
an earnings word and never beside a dividend or book value; a table is read by
matching its header date to the quarter's period end, and refused when it
cannot be. And every extraction is checked against the 10-Q's XBRL once that
exists. Agreement is the extractor's measured accuracy, on the record.
Disagreement is reported, not hidden.
"""
from __future__ import annotations

import copy
import os
import re
import time
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

from research import sue

VARIANT = "ts_release"

# How far a release figure may sit from the later XBRL figure and still be the
# same number. A cent covers rounding; anything more is a different basis, a
# restatement, or the wrong figure.
AGREEMENT_TOLERANCE = 0.015

# One retry on a read timeout. EDGAR drops long SGML pulls under load and the
# second attempt usually lands; a third rarely does, and hammering a vendor
# that is refusing is how the whole pass gets throttled.
FETCH_RETRIES = 2
FETCH_RETRY_BACKOFF = 3.0

# Phrases that name the diluted per-share figure. Each requires "diluted", so
# a basic-EPS sentence never matches.
_DILUTED = re.compile(
    r"(?:per\s+diluted\s+(?:common\s+)?share"
    r"|diluted\s+(?:net\s+)?(?:earnings|income|loss|eps)(?:\s+per\s+(?:common\s+)?share)?"
    r"|(?:earnings|income|loss)\s+per\s+diluted\s+(?:common\s+)?share"
    r"|diluted\s+eps)", re.I)

# The unqualified form banks use: "$7.70 per share", "EPS of $7.70". Accepted
# only in a sentence that is about earnings and not about a distribution.
_PER_SHARE = re.compile(r"(?:per\s+(?:common\s+)?share|\bEPS\b)", re.I)
# Net income, net earnings or net loss -- "earnings" alone let too much
# through: 57% right against 78% for the diluted phrase on a hundred names.
_EARNINGS_WORDS = re.compile(r"net\s+(?:income|loss|earnings)|\bEPS\b", re.I)
_NOT_EARNINGS = re.compile(
    r"dividend|book\s+value|repurchas|buy\s*back|offering|price\s+of|"
    r"average\s+(?:price|cost)|exercise|conversion", re.I)

# Qualifiers that make a figure something other than the GAAP one.
_NON_GAAP = re.compile(
    r"non[-\s]?gaap|adjusted|pro\s*forma|normali[sz]ed|core\s+(?:eps|earnings)"
    r"|excluding|ex[-\s]items|significant\s+items|\bcomparable\b"
    # A REIT's or an insurer's headline metric, which is non-GAAP by another
    # name: AGNC's "net spread and dollar roll income per common share".
    r"|net\s+spread|dollar\s+roll|distributable|funds\s+from\s+operations"
    r"|\bA?FFO\b|comprehensive\s+income|operating\s+earnings|economic\s+return",
    re.I)

# A component of earnings is not the earnings. Aflac's quarter-naming sentence
# was "included pretax net realized investment losses of $322 million, or
# $0.42 per diluted share", and it outranked the headline. Demoted, not
# skipped: some headlines say "included" too.
_COMPONENT = re.compile(
    r"\b(?:included|includes|including|impact\s+of|benefit\s+of|charges?\s+of"
    r"|gains?\s+of|losses\s+of|expense\s+of|related\s+to"
    # AGNC: "a net loss of $(433) million in other gain (loss), net, or
    # $(0.39) per common share" -- the amount is a line item.
    r"|\bin\s+other\b|\bin\s+[a-z\s()]{3,40},\s*net\b)", re.I)

# A sentence that exists to compare. Advanced Energy's "This compares with
# $1.94 per diluted share in the fourth quarter of 2025" names a quarter --
# the prior one -- and beat the headline that named none.
_COMPARATIVE = re.compile(
    r"\b(?:this|that|which)\s+compares\b|^\s*compared\s+(?:with|to)"
    r"|^\s*(?:versus|vs\.?)\s", re.I)

# XBRL's diluted EPS is the total. Continuing operations is a different basis
# when there were discontinued ones -- demoted, and accepted when it is all
# there is.
_CONTINUING = re.compile(r"continuing\s+operations", re.I)

# Whether the clause is about a loss, whatever the phrase says. Alcoa: "Net
# loss attributable to Alcoa Corporation was $746 million, or $4.17 per share"
# read as +4.17 because the phrase was "per share". A parenthetical "(loss)"
# in a row label -- "Net income (loss) per share" -- is not a loss.
_NET_LOSS = re.compile(r"\bnet\s+loss\b|\bloss\s+per\b|\bloss\s+of\b", re.I)
_NET_INCOME = re.compile(r"\bnet\s+(?:income|earnings)\b", re.I)

# Whether a sentence is about the quarter or the year. Q4 releases carry both,
# and the year comes first.
_QUARTER = re.compile(
    r"(?:first|second|third|fourth)[-\s]quarter|\bQ[1-4]\b|\d[qQ]\d{2}\b"
    r"|three\s+months|\bquarterly\b", re.I)
_ANNUAL = re.compile(
    r"fiscal\s+(?:year|20\d\d)|full[-\s]year|twelve\s+months|year\s+ended"
    r"|\bfor\s+(?:the\s+year|20\d\d)\b|\bin\s+20\d\d\b"
    # Year-to-date is not the quarter either.
    r"|(?:six|nine)\s+months|year[-\s]to[-\s]date|first\s+half", re.I)

# A flattened financial statement is not prose. Its header interleaves column
# labels -- "(In millions except per Three June share Months 27, June 28, ...
# Ended 2025 2024)" -- and a per-share phrase inside it sits beside whichever
# column's figure comes next. Only the table reader may read it, by column.
#
# Only the markers that never occur in prose. "Three months ended June 27,
# 2025" is how a headline paragraph names its own quarter, and a guard on it
# refused all 23 of Coca-Cola's releases.
_STATEMENT = re.compile(
    r"\(in\s+(?:millions|thousands|billions)|except\s+per[-\s]+share", re.I)

# A dollar figure, with the three ways a loss is written: $(0.12), ($0.12),
# -$0.12, and a table's bare "$ -0.33".
_MONEY_BEFORE = re.compile(
    r"(?:(?:-\s?\$|\(\s?\$|\$\s?\(|\$)\s?-?(\d*\.\d{2})\)?"
    r"|(\d{1,3})\s+cents)\s*(?:,|or)?\s*$", re.I)
# ...or through a growth verb and a percentage: "EPS grew 18% to $0.91",
# "increased 12% to $2.02", "rose to $1.10". Coca-Cola writes every headline
# that way, and an adjacency rule without it refused all 23 of its releases.
_MONEY_AFTER = re.compile(
    r"^\s*(?::\s*EPS)?\s*"
    r"(?:was|were|of|totaled|totalled|came\s+in\s+at|reached|at|:|,|\(|"
    r"(?:grew|increased|rose|climbed|improved|expanded|declined|decreased|fell|"
    r"dropped|was\s+up|was\s+down)(?:\s+\d+(?:\.\d+)?\s*%)?\s+(?:to|at))?"
    r"\s*(?:(?:-\s?\$|\(\s?\$|\$\s?\(|\$)\s?-?(\d*\.\d{2})\)?"
    r"|(\d{1,3})\s+cents)", re.I)

# A bullet is a sentence boundary too: older releases are flattened bullet
# lists with no full stops, and to a splitter on punctuation alone the whole
# page is one sentence whose first figure is the year's.
# ...but only where a new item starts. Photronics also wraps lines with a
# bullet mid-sentence ("$28.9 • million"), and splitting there strands the
# non-GAAP sentence's figure in a fragment carrying no qualifier.
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z“\"(])|\s*[•·▪■]\s*(?=[A-Z“\"])")

_MONTHS = {m: i for i, m in enumerate(
    ("january", "february", "march", "april", "may", "june", "july",
     "august", "september", "october", "november", "december"), start=1)}


def _sign(sentence: str, start: int, end: int, clause_is_loss: bool) -> int:
    """Negative for a loss, written three ways: "-$0.12", "$(0.12)", "($0.12)".

    The paren forms close on the digits. "( $7.70 PER SHARE)" is a parenthetical
    around a whole phrase and closes after the words, so an opening paren alone
    is not a loss -- JPMorgan's headline came back as -7.70 while it was.
    """
    if clause_is_loss or sentence[end:end + 1] == ")":
        return -1
    return -1 if "-" in sentence[max(0, start - 3):start] else 1


def _clause_is_loss(sentence: str, upto: int) -> bool:
    """Whether the words before the figure describe a loss.

    "Net loss ... was $746 million, or $4.17 per share" is a loss whatever
    the phrase says. "Net income (loss) per share" is a row label carrying
    both words and is not."""
    clause = sentence[:upto]
    if _NET_INCOME.search(clause) and not re.search(r"\bnet\s+loss\b", clause, re.I):
        return False
    return bool(_NET_LOSS.search(clause))


def _value_of(match) -> float:
    dollars, cents = match.group(1), match.group(2)
    return float(dollars) if dollars else float(cents) / 100.0


def _figure_position(sentence: str, match: "re.Match") -> int:
    """Where the figure adjacent to the phrase starts, or the phrase start."""
    before = sentence[:match.start()]
    tail = _MONEY_BEFORE.search(before)
    if tail:
        return tail.start()
    after = sentence[match.end():match.end() + 60]
    lead = _MONEY_AFTER.search(after)
    return match.end() + lead.start() if lead else match.start()


def _figure_beside(sentence: str, match: "re.Match"):
    """The first figure adjacent to a phrase: immediately before it ("$0.49
    per diluted share", "32 cents per share") or within a few words after it
    ("was $2.02")."""
    before = sentence[:match.start()]
    tail = _MONEY_BEFORE.search(before)
    if tail:
        g = 1 if tail.group(1) else 2
        loss = _clause_is_loss(sentence, tail.start(g)) or \
            "loss" in match.group(0).lower()
        return _sign(sentence, tail.start(g), tail.end(g), loss) * _value_of(tail)
    after = sentence[match.end():match.end() + 60]
    lead = _MONEY_AFTER.search(after)
    if lead:
        g = 1 if lead.group(1) else 2
        start, end = match.end() + lead.start(g), match.end() + lead.end(g)
        loss = _clause_is_loss(sentence, start) or "loss" in match.group(0).lower()
        return _sign(sentence, start, end, loss) * _value_of(lead)
    return None


def _from_prose(text: str) -> Dict[str, Any]:
    """Sentence by sentence, in document order, ranked.

    Rank 0: names the quarter. Rank 1: names neither. Rank 2: names the year.
    Within a rank the first sentence wins, because the headline comes first
    and the comparatives come after the current figure within a sentence.
    The diluted phrase is preferred; the unqualified "per share" is accepted
    only in an earnings sentence that is not about a distribution.
    """
    candidates: List[tuple] = []
    saw_phrase = saw_gaap = False
    for index, sentence in enumerate(_SENTENCE.split(text)):
        if _STATEMENT.search(sentence):
            continue
        match = _DILUTED.search(sentence)
        basis = "gaap"
        if not match:
            match = _PER_SHARE.search(sentence)
            if not match or not _EARNINGS_WORDS.search(sentence) \
                    or _NOT_EARNINGS.search(sentence) \
                    or re.search(r"\bbasic\b", sentence, re.I):
                continue
            basis = "per_share"
        saw_phrase = True
        if _NON_GAAP.search(sentence[:match.start()]):
            continue
        saw_gaap = True
        value = _figure_beside(sentence, match)
        if value is None:
            continue
        # A quarter word counts only before the figure. After it, it names
        # the period being compared against: "... $1.94 per diluted share in
        # the fourth quarter of 2025".
        figure_at = _figure_position(sentence, match)
        head = sentence[:figure_at]
        rank = 0 if _QUARTER.search(head) else (
            2 if _ANNUAL.search(sentence) else 1)
        # The wrong number beats the wrong period. A component of earnings
        # ("included ... losses of $322 million, or $0.42 per diluted share")
        # and a continuing-operations basis are wrong figures whatever quarter
        # they name, so those demerits come before the period rank. Then the
        # rank, then a non-GAAP qualifier anywhere -- "$1.64 ... when
        # excluding the one-time charge" comes after the figure, so it is not
        # a reason to skip, only to lose to a clean sentence -- then a
        # diluted phrase over an unqualified one, then document order.
        key = (bool(_COMPONENT.search(sentence[:match.start()]))
               or bool(_COMPARATIVE.search(sentence)),
               bool(_CONTINUING.search(sentence)),
               rank,
               bool(_NON_GAAP.search(sentence)),
               basis != "gaap",
               index)
        candidates.append((key, value, basis, sentence.strip()))

    if candidates:
        _, value, basis, evidence = min(candidates)
        return {"eps": value, "basis": basis, "evidence": evidence,
                "reason": None}
    if not saw_phrase:
        reason = "no diluted per-share figure named in the release"
    elif not saw_gaap:
        reason = ("every diluted per-share figure is qualified as non-GAAP or "
                  "adjusted; there is no GAAP figure to read")
    else:
        reason = ("a diluted per-share phrase was found but no dollar figure "
                  "sits beside it")
    return {"eps": None, "basis": None, "evidence": None, "reason": reason}


_HEADER = re.compile(
    r"three\s+months\s+ended\s+([A-Za-z]+)\s+(\d{1,2}),?\s+((?:\d{4}\s*){2,})",
    re.I)
_ROW = re.compile(
    r"((?:net\s+)?(?:income|earnings|loss)\s+per\s+(?:common\s+)?share[^$]{0,120}?)"
    r"((?:\$\s?-?\(?\d+\.\d{2}\)?\s*){2,})"
    r"([^$]{0,80}?diluted|)", re.I)
_CELL = re.compile(r"\$\s?(-?)(\(?)(\d+\.\d{2})\)?")


def _from_table(text: str, period_end: Optional[str]) -> Dict[str, Any]:
    """A statement table, read by matching a column to the period end.

    The row alone cannot say which column is this quarter -- Rivian's run
    prior-year first. The header can: "Three Months Ended March 31, 2025 2026"
    gives one date per column, and the quarter's period end picks the column.
    Anything short of that exact match is a refusal, because guessing a column
    is guessing the sign and the size of the surprise at once.
    """
    out: Dict[str, Any] = {"eps": None, "basis": None, "evidence": None,
                           "reason": None}
    header = _HEADER.search(text)
    row = _ROW.search(text)
    # The diluted word may sit before the cells ("Net loss per share, basic
    # and diluted $ (0.90) $ (1.77)") or after them (Rivian's later layout).
    if row and not row.group(3) and not re.search(
            r"diluted", row.group(1), re.I):
        row = None
    if not header or not row:
        out["reason"] = "no diluted per-share table row with a period header"
        return out
    if not period_end:
        out["reason"] = ("the figure is in a table and the column for this "
                         "quarter cannot be chosen without its period end")
        return out
    month = _MONTHS.get(header.group(1).lower())
    if month is None:
        out["reason"] = f"unreadable table header month {header.group(1)!r}"
        return out
    day = int(header.group(2))
    columns = [f"{int(y)}-{month:02d}-{day:02d}" for y in header.group(3).split()]
    cells = _CELL.findall(row.group(2))
    if period_end not in columns:
        out["reason"] = (f"the table's columns are {columns} and none is the "
                         f"quarter's period end {period_end}")
        return out
    if len(cells) != len(columns):
        out["reason"] = (f"the table row has {len(cells)} figures against "
                         f"{len(columns)} columns; the mapping is ambiguous")
        return out
    minus, paren, digits = cells[columns.index(period_end)]
    out.update({"eps": (-1 if (minus or paren) else 1) * float(digits),
                "basis": "table",
                "evidence": " ".join(row.group(0).split())[:200]})
    return out


def extract_diluted_eps(text: Optional[str],
                        period_end: Optional[str] = None) -> Dict[str, Any]:
    """The GAAP diluted EPS a release reports, or a refusal that says why.

    Prose first: it is what a headline is for, and it is unambiguous about
    which period it describes. A table only when the prose has nothing, and
    only with a period end to pick the column by.
    """
    if not text or not text.strip():
        return {"eps": None, "basis": None, "evidence": None,
                "reason": "no release text to read"}
    flat = " ".join(text.split())
    prose = _from_prose(flat)
    if prose["eps"] is not None:
        return prose
    table = _from_table(flat, period_end)
    if table["eps"] is not None:
        return table
    # The more specific refusal: a table that exists but cannot be read says
    # more than "no figure in the prose".
    return table if _ROW.search(flat) else prose


def signal_from_release(series: Dict[str, Any], fiscal_period: str,
                        release_eps: float, announced_date: str,
                        as_of: Optional[str] = None,
                        accession: Optional[str] = None) -> Dict[str, Any]:
    """`sue._signal_from_series` with one quarter's EPS taken from the 8-K.

    The announced quarter is replaced or, when the 10-Q has not been filed
    yet, inserted. The eight quarters the standardisation reads are untouched
    -- substituting the release figure into the history too would change the
    sigma with hindsight.

    The release figure is on the share basis of its own day, exactly as the
    XBRL fact was as filed, so it takes the quarter's `basis_factor` -- the
    later split adjustment that `eps_series` applied to the XBRL value.
    """
    key = sue._parse_period(fiscal_period)
    if key is None:
        return {**sue._signal_shell(series["ticker"], as_of or announced_date),
                "error": f"{fiscal_period!r} is not a fiscal period"}

    quarters: List[Dict[str, Any]] = copy.deepcopy(series["quarters"])
    existing = next((q for q in quarters if q["fiscal_period"] == fiscal_period),
                    None)

    if existing is not None and announced_date > existing["known_at"]:
        return {**sue._signal_shell(series["ticker"], as_of or announced_date),
                "fiscal_period": fiscal_period,
                "error": (f"the release dated {announced_date} is after the "
                          f"filing dated {existing['known_at']}, so it is not "
                          f"the announcement of {fiscal_period}")}

    if existing is not None:
        factor = existing["basis_factor"]
        xbrl_eps = existing.get("eps_as_filed")
        entry = existing
    else:
        # Live: the 8-K is out, the 10-Q is not. The basis is the most recent
        # quarter's, which is the only one that can be known.
        prior = quarters[-1] if quarters else None
        factor = prior["basis_factor"] if prior else 1.0
        xbrl_eps = None
        entry = {"fiscal_period": fiscal_period, "fiscal_year": key[0],
                 "fiscal_quarter": key[1], "period_start": None,
                 "period_end": None, "concept": series.get("concept"),
                 "derivation": None}
        quarters.append(entry)

    entry.update({
        "eps": release_eps * factor, "eps_as_filed": release_eps,
        "basis_factor": factor, "source": "release", "form": "8-K",
        "known_at": announced_date, "accession": accession,
    })
    quarters.sort(key=lambda q: (q["fiscal_year"], q["fiscal_quarter"]))

    patched = {**series, "quarters": quarters}
    signal = sue._signal_from_series(patched, key, as_of or announced_date)
    signal.update({
        "variant": VARIANT, "source": "release", "release_eps": release_eps,
        "xbrl_eps": xbrl_eps,
        "agrees_with_xbrl": (None if xbrl_eps is None
                             else abs(xbrl_eps - release_eps)
                             <= AGREEMENT_TOLERANCE),
    })
    return signal


# ----------------------------------------------------------- reading EDGAR

def _company(ticker: str):
    """`edgar.Company`, with the SEC identity set here rather than assumed.

    Every other EDGAR reader in the package sets it on its own path; relying
    on one of them having run first in this process is how a lone call from a
    driver script raised IdentityNotSetException.
    """
    from edgar import Company, set_identity

    email = os.environ.get("SEC_EMAIL")
    if not email:
        raise RuntimeError("SEC_EMAIL is not set; EDGAR requires a contact "
                           "address in the User-Agent and refuses without one")
    set_identity(f"{os.environ.get('NAME') or 'Investment Analyst'} {email}")
    return Company(ticker)


def _exhibit_rank(attachment) -> Optional[int]:
    """0 for EX-99.1, 1 for any other EX-99.x, None for not an exhibit 99.

    Matched on the document name and the description together, because
    JPMorgan's are `exhibit991narrative` and `ex992supplement`: a substring
    check on `ex99` alone matched the supplement -- tables only -- and never
    read the narrative press release.
    """
    doc = (getattr(attachment, "document", "") or "").lower()
    descr = (getattr(attachment, "description", "") or "").lower()
    if not re.search(r"ex(?:hibit)?[-_ ]?99", doc + " " + descr):
        return None
    # Not a word boundary after the 1: `exhibit991narrative` runs straight
    # into a letter. What must not follow is another digit, or 99.1 reads as
    # 99.10.
    if re.search(r"99[-_. ]?1(?!\d)", doc) or re.search(r"99[-_. ]?1(?!\d)", descr):
        return 0
    return 1


def _filing_by_accession(accession: str):
    """The slow path: resolve one accession through the quarterly index."""
    import edgar

    return edgar.get_by_accession_number(accession)


def _exhibits(ticker: str, accession: str, filings=None) -> List[Any]:
    """The EX-99 attachments of one 8-K, EX-99.1 first.

    Found through the company's own filings list, which `announcements`
    fetched already and edgartools caches, rather than through
    `get_by_accession_number`, which downloads a whole quarterly index to
    resolve one accession.
    """
    if filings is None:
        filings = _company(ticker).get_filings(form="8-K")
    filing = next((f for f in filings
                   if getattr(f, "accession_no", None) == accession), None)
    if filing is None:
        filing = _filing_by_accession(accession)
    if filing is None:
        raise LookupError(f"{accession} is not in {ticker}'s 8-K list and "
                          f"could not be resolved from the index")
    # The filing's index page, not `filing.attachments`: the latter parses
    # the whole SGML submission to list the exhibits -- for Rivian, 28
    # embedded images -- at 17 seconds a release, and the read timeouts came
    # from there. The index is a small page listing the same documents.
    #
    # An empty index is a transient, not a fact: 113 releases in a hundred
    # names came back "no EX-99 exhibit" from filings that plainly carry one.
    # Asked again, then read the slow way before believing it.
    documents = list(filing.homepage.documents or [])
    if not documents:
        time.sleep(FETCH_RETRY_BACKOFF)
        documents = list(filing.homepage.documents or [])
    if not documents:
        documents = list(filing.attachments or [])
    ranked = [(rank, i, a) for i, a in enumerate(documents)
              for rank in (_exhibit_rank(a),) if rank is not None]
    return [a for _, _, a in sorted(ranked, key=lambda t: (t[0], t[1]))]


def _text_of(attachment) -> Optional[str]:
    """One attachment's text, with one retry on a read timeout."""
    last: Optional[BaseException] = None
    for attempt in range(FETCH_RETRIES):
        try:
            text = getattr(attachment, "text", None)
            return text() if callable(text) else None
        except Exception as exc:  # noqa: BLE001 - retried once, then raised
            last = exc
            if attempt + 1 < FETCH_RETRIES:
                time.sleep(FETCH_RETRY_BACKOFF * (attempt + 1))
    assert last is not None
    raise last


def _release_text(ticker: str, accession: str) -> Optional[str]:
    """The EX-99.1 press release attached to an 8-K, or None."""
    exhibits = _exhibits(ticker, accession)
    return _text_of(exhibits[0]) if exhibits else None


def read_release(ticker: str, accession: str,
                 period_end: Optional[str] = None,
                 filings=None) -> Dict[str, Any]:
    """Every EX-99 exhibit of the 8-K, in order, until one yields a figure.

    The first refusal is kept and reported if none does, so a release whose
    narrative has no figure and whose supplement has an unreadable table
    still says which of those it was.
    """
    try:
        exhibits = _exhibits(ticker, accession, filings=filings)
    except LookupError as exc:
        return {"eps": None, "basis": None, "evidence": None,
                "reason": f"filing not found: {exc}"}
    if not exhibits:
        return {"eps": None, "basis": None, "evidence": None,
                "reason": "the 8-K lists documents and none is an EX-99 exhibit"}
    first_refusal = None
    for attachment in exhibits:
        read = extract_diluted_eps(_text_of(attachment), period_end=period_end)
        if read["eps"] is not None:
            read["exhibit"] = getattr(attachment, "document", None)
            return read
        first_refusal = first_refusal or read
    return first_refusal


def _period_end_for(series: Dict[str, Any], fiscal_period: str) -> Optional[str]:
    """The quarter's period end from XBRL, or the quarter after the last one
    XBRL holds -- three months on -- when the 10-Q is not filed yet."""
    for q in series["quarters"]:
        if q["fiscal_period"] == fiscal_period and q.get("period_end"):
            return q["period_end"]
    last = series["quarters"][-1] if series["quarters"] else None
    if not last or not last.get("period_end"):
        return None
    year, month, _ = (int(x) for x in last["period_end"].split("-"))
    month += 3
    if month > 12:
        year, month = year + 1, month - 12
    first_of_next = (date(year + 1, 1, 1) if month == 12
                     else date(year, month + 1, 1))
    return (first_of_next - timedelta(days=1)).isoformat()


def release_history(ticker: str) -> Dict[str, Any]:
    """Every quarter's release-timed signal, plus the extraction record.

    One `eps_series` fetch for the history, one `for_quarters` pass to map
    each quarter to its 8-K, then one exhibit read per quarter. The rows are
    what `replay.load_signals` wants; `extractions` is the audit trail -- what
    was read, from which exhibit, what XBRL later said, and every refusal
    with its reason.
    """
    from research import announcements

    series = sue.eps_series(ticker)
    if not series["success"]:
        return {"ticker": ticker, "signals": [], "extractions": [],
                "error": series["error"]}

    by_period = {q["fiscal_period"]: {"period_end": q.get("period_end"),
                                      "known_at": q["known_at"]}
                 for q in series["quarters"]}
    releases = announcements.for_quarters(ticker, quarters=by_period)
    # One filings list for the name. Per accession it was 14 seconds a
    # release on JPMorgan, and every release lives in the same list.
    filings = _company(ticker).get_filings(form="8-K") if releases else []

    signals, extractions = [], []
    for period, release in sorted(releases.items()):
        record = {"fiscal_period": period, "accession": release.get("accession"),
                  "announced_date": release["announced_date"],
                  "release_eps": None, "xbrl_eps": None, "basis": None,
                  "exhibit": None, "agrees_with_xbrl": None, "reason": None}
        if not release.get("accession"):
            record["reason"] = "the release carries no accession to fetch"
            extractions.append(record)
            continue
        try:
            read = read_release(ticker, release["accession"],
                                period_end=_period_end_for(series, period),
                                filings=filings)
        except Exception as exc:  # noqa: BLE001 - recorded, not masked
            # One EDGAR timeout used to take the whole name with it, and in a
            # replay driver the whole run. The quarter is refused with the
            # reason on the record; the others are still read.
            record["reason"] = f"fetch failed: {type(exc).__name__}: {exc}"[:160]
            extractions.append(record)
            continue
        record.update({"basis": read.get("basis"),
                       "exhibit": read.get("exhibit")})
        if read["eps"] is None:
            record["reason"] = read["reason"]
            extractions.append(record)
            continue
        signal = signal_from_release(series, period, read["eps"],
                                     release["announced_date"],
                                     accession=release.get("accession"))
        record.update({"release_eps": read["eps"],
                       "xbrl_eps": signal.get("xbrl_eps"),
                       "agrees_with_xbrl": signal.get("agrees_with_xbrl"),
                       "reason": signal.get("error")})
        extractions.append(record)
        if signal.get("success") and signal.get("sue") is not None:
            signals.append(signal)
    return {"ticker": ticker, "signals": signals, "extractions": extractions,
            "error": None}


def sue_ts_release(ticker: str, as_of: Optional[str] = None) -> Dict[str, Any]:
    """The release-timed surprise for the most recent announcement on `as_of`.

    What the nightly scan asks when `SIGNAL_VARIANT` is `ts_release`. Falls
    back to nothing: if the latest release cannot be read, that is a refusal
    for this name tonight, not a reason to use the XBRL-timed signal instead
    and quietly mix the two variants in one book.
    """
    from research import announcements

    as_of = as_of or sue._today()
    series = sue.eps_series(ticker, as_of=as_of)
    if not series["success"]:
        return {**sue._signal_shell(ticker, as_of), "error": series["error"],
                "variant": VARIANT}

    releases = announcements.earnings_releases(ticker, as_of=as_of)
    if not releases:
        return {**sue._signal_shell(ticker, as_of), "variant": VARIANT,
                "error": f"no Item 2.02 release on or before {as_of} for {ticker}"}
    latest = releases[0]

    by_period = {q["fiscal_period"]: {"period_end": q.get("period_end"),
                                      "known_at": q["known_at"]}
                 for q in series["quarters"]}
    mapped = announcements.for_quarters(ticker, as_of=as_of, quarters=by_period)
    period = next((p for p, r in mapped.items()
                   if r.get("accession") == latest.get("accession")), None)
    if period is None:
        last = series["quarters"][-1] if series["quarters"] else None
        if last is None:
            return {**sue._signal_shell(ticker, as_of), "variant": VARIANT,
                    "error": f"{ticker} has no EPS history to standardise against"}
        fy, fq = sue._shift(last["fiscal_year"], last["fiscal_quarter"], back=-1)
        period = sue._period_key(fy, fq)

    read = read_release(ticker, latest["accession"],
                        period_end=_period_end_for(series, period))
    if read["eps"] is None:
        return {**sue._signal_shell(ticker, as_of), "variant": VARIANT,
                "fiscal_period": period,
                "error": f"{ticker} {period} release: {read['reason']}"}
    return signal_from_release(series, period, read["eps"],
                               latest["announced_date"], as_of=as_of,
                               accession=latest.get("accession"))
