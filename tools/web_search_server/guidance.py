"""Company-issued forward guidance, as filed.

Management credibility starts with what management actually said it would do.
Nothing in this repo captured that: `extract_forward_signals` matches
forward-looking *language* with `[^.]{20,300}`, which stops at the first
literal period, so "$1.5 billion" truncates to "$1". No tool anywhere
produced a guided number.

WHAT THIS TOOL DOES NOT DO, AND WHY
-----------------------------------
It does not say whether guidance was met. That was the original brief and the
filings do not support it. Three findings from probing 24 large caps:

1. Structured guidance lives in HTML tables, and the text extraction this repo
   uses flattens them column-interleaved. Salesforce's EPS guidance renders as
   "GAAP diluted net income per share range(1)(2) Fiscal 2027 $1.74 - $7.93 -
   Plus Q2 $1.76 FY27 $7.99". The real table is two columns, Q2 at
   $1.74-$1.76 and FY27 at $7.93-$7.99. "$1.74 - $7.93" is a range nobody
   guided, and nothing in the flattened text marks it as wrong. Intel's
   "Gross margin 41.0% 42.0%" is the GAAP and non-GAAP columns, not a range.
   Coca-Cola's guidance table has Current and Prior columns, so the
   superseded number reads exactly like the live one.

2. Grading a guide needs the actual on the same basis for the same fiscal
   period. Salesforce's GAAP and non-GAAP EPS differ roughly fourfold, and the
   only actuals source here labels fiscal quarters with calendar-quarter ends
   -- documented in the earnings-eval skill as off by up to 60 days for
   fiscal-offset filers. That is a fuzzy join stacked on a fuzzy extraction.

3. Plenty of guidance is never filed at all. Microsoft and Apple give theirs
   on the call, and `get_earnings_transcripts` returns 8-K press releases
   rather than transcripts (known issue). Absence from this corpus is not
   absence of guidance.

So this returns the sentence as filed, with the filing it came from, and
lets the caller read it. A verdict the data cannot support is worse than no
verdict.

WHAT IT REFUSES
---------------
Anything bearing a table signature: a box-drawing rule, a truncated-cell
ellipsis, a Current/Prior or GAAP/Non-GAAP column pair, or two amounts with
nothing but whitespace between them. Prose always puts a word between two
numbers -- "74.9% and 75.0%" -- so the absence of one is the tell. Also
quotations, safe-harbour boilerplate, backward references to guidance already
given, and reported results that happen to sit near a guidance cue.

`guidance_may_be_table_only` is set when the refused regions mention
guidance, which is the difference between "does not guide" and "guides where
this tool will not look".

MEASURED, at the tool's own defaults (four releases per company, each
truncated to 50k characters by get_earnings_releases)
------------------------------------------------------------------------
24 large caps. 3 had no usable source: Caterpillar filed no 8-K Item 2.02 in
the window, Delta's and Procter & Gamble's exhibit text would not extract at
all. Of the 21 readable, 17 yielded prose guidance -- 145 statements -- and 4
came back empty: Apple, Costco and Microsoft genuinely do not guide in the
release, and Walmart does but only in tables, where it is flagged.

Auditing all 145 by hand: 1 is not guidance (a General Motors results
headline whose amounts belong to a dividend and a buyback) and 2 carry a
wrong period label. The rest are real, correctly attributed guidance.

How many quarters you scan changes what you find, and not gently. Coca-Cola's
most recent release is truncated before its outlook section, so quarters=1
returns nothing for a company that guides every quarter; quarters=4 returns
25 statements. Read a small `quarters` as a weaker search, never as evidence
of silence.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from tools.web_search_server.sec_utils import get_earnings_releases

# edgartools renders table rules with box-drawing characters and abbreviates
# overflowing cells with a horizontal ellipsis. Either one means the text
# around it came out of a grid.
# Box rules and truncated-cell ellipses come from edgartools' table renderer.
# U+200B is Deere's: its industry outlook grid separates cells with zero-width
# spaces and carries no visible table signature at all.
TABLE_CHARS = re.compile('[\u2500-\u257f\u2026\u200b]')

AMOUNT_RE = (r'(?:\$\s?\d[\d,]*(?:\.\d+)?(?:\s*(?:billion|million|trillion|B|M))?'
             r'|\d+(?:\.\d+)?\s*%)')
AMOUNT = re.compile(AMOUNT_RE)
# Two amounts separated by nothing but whitespace: a row, not a range.
ADJACENT_AMOUNTS = re.compile(AMOUNT_RE + r'\s+' + AMOUNT_RE)
# "Non- GAAP" with a stray space appears 29 times in the measured corpus, so
# the hyphen may be followed by whitespace. Numbers themselves are never
# broken -- no split decimals, separators or units in 1.4M characters -- so
# only the hyphenated word needs the allowance.
COLUMN_HEADERS = re.compile(
    r'(?i)\bGAAP\s+Non-?\s*GAAP\b|\bNon-?\s*GAAP\s+GAAP\b'
    r'|\bCurrent\s+Prior\b'
    r'|\bPrior\s+Current\b|\bLow\s+High\b|\bHigh\s+Low\b'
    r'|\bQ[1-4]\s*\d{4}\s+GAAP\b|\bGuidance\s+Current\b')
# A table lead-in: "as follows:", "based on the following FY26 figures:".
AS_FOLLOWS = re.compile(
    r'(?i)(?:as follows|based on the following|the following[^.:]{0,40})\s*:')
# ".39" with no leading zero is a table cell. Prose does not write decimals
# that way, and Lilly's EPS reconciliation is nothing but them.
BARE_DECIMAL = re.compile(r'(?<![\d.])\.\d{1,2}\b')
_MAX_BARE_DECIMALS = 1
# Captions that only ever head a financial table. Target's non-GAAP EPS
# reconciliation reached the caller through every other guard.
TABLE_CAPTION = re.compile(
    r'(?i)\(unaudited\)|\breconciliation of\b'
    r'|\(in\s+(?:millions|thousands|billions)\b')
# "$7.93 - Plus": a range that opens and never closes, because the cell it
# closed in sat on the next row of the table.
DANGLING_RANGE = re.compile(AMOUNT_RE + r'\s*[-–—]\s+(?![$\d])')
# One statement guides one period. Salesforce's flattened EPS table carries
# "Fiscal 2027", "Q2" and "FY27" at once, which is the grid showing through.
_MAX_PERIOD_PHRASES = 2

BOILERPLATE = re.compile(
    r'(?i)forward[- ]looking statements?|section 27a|section 21e'
    r'|private securities litigation|risk factors|safe harbor|safe harbour')
BACKWARD_REFERENCE = re.compile(
    r'(?i)(?:prior|previous|original|initial)\s+guidance'
    r'|guidance\s+(?:provided|issued|given)\s+(?:on|in)'
    r'|(?:compared|relative)\s+to\s+.{0,20}guidance'
    r'|versus\s+.{0,15}guidance'
    r'|\b(?:exceed(?:s|ed|ing)?|beat|beats|surpass\w*|ahead\s+of'
    r'|above|below)\s+(?:\w+\s+){0,2}guidance\b')
QUOTATION = re.compile(r'[“”]|"\s*,?\s*said|said\s+[A-Z][a-z]+\s+[A-Z]')

# Unambiguously forward-looking on its own.
STRONG_CUE = re.compile(
    r'(?i)\b(?:expects?|expecting|is expected to|are expected to|anticipates?'
    r'|anticipating|projects?|projecting|forecasts?|forecasting|outlook'
    r'|guidance|guiding|sees|plans? to|targets|targeting|targeted)\b')
# Forward-looking only next to a guidance noun. "lower revenue" is a report of
# what happened; "lowers its guidance" is a statement about what will.
DIRECTIONAL_CUE = re.compile(
    r'(?i)\b(?:raise[sd]?|raising|lower[sed]*|lowering|cut(?:s|ting)?'
    r'|reaffirms?|reaffirming|reiterat\w+|initiat\w+|updates?|updating'
    r'|narrow\w*|maintains?|increases?|reduces?|improved)\b[^.]{0,70}?'
    r'\b(?:guidance|outlook|forecast|estimates?|target)\b'
    r'|\b(?:guidance|outlook|forecast)\b[^.]{0,70}?'
    r'\b(?:raised|lowered|cut|reaffirmed|reiterated|initiated|updated'
    r'|narrowed|improved)\b')
PAST_REPORT = re.compile(
    r'(?i)\b(?:was|were|grew|rose|fell|declined|totaled|totalled|reported'
    r'|delivered|posted|paid|repurchased|came in)\b')

PERIOD = re.compile(
    r'(?i)('
    r'(?:the\s+)?(?:first|second|third|fourth)[-\s]quarter'
    r'(?:\s+of)?(?:\s+fiscal)?(?:\s+year)?(?:\s+(?:\d{4}|FY\s?\'?\d{2,4}))?'
    r'|full[-\s]year(?:\s+fiscal)?(?:\s+year)?(?:\s+(?:\d{4}|FY\s?\'?\d{2,4}))?'
    r'|fiscal(?:\s+year)?\s+\d{4}'
    r'|Q[1-4]\s*(?:FY)?\s*\'?\d{2,4}'
    r'|FY\s?\'?\d{2,4}'
    r'|calendar\s+year\s+\d{4}'
    r')')
# Only a lead-in like "our outlook for the second quarter of fiscal 2027 is as
# follows" sets the period for the bullets under it. "reported revenue for the
# first quarter" must not.
SECTION_LEAD_IN = re.compile(
    r'(?i)\b(?:outlook|guidance)\s+for\b'
    r'|\bfor\s+(?:the\s+)?(?:first|second|third|fourth)[-\s]quarter'
    r'|\bfor\s+the\s+full[-\s]year|\bfor\s+fiscal\b')

_BULLETS = re.compile('[•▪●·]')
_BREAK = ''
# A press-release dateline, and the two other places a release runs several
# paragraphs together with no sentence punctuation between them.
_DATELINE = re.compile(
    r'\s(?:January|February|March|April|May|June|July|August|September'
    r'|October|November|December)\s+\d{1,2},?\s*\d{4}\s*[-–—]{1,3}\s*')
_OPENER = re.compile(r'(?i)\b(?=today\s+(?:reported|announced|released|issued))')
_PAGE_FURNITURE = re.compile(
    r'(?i)[-–—]\s*more\s*[-–—]|Page\s+\d+\s+of\s+\d+'
    r'|Exhibit\s+99\.\d+')

# What a refused table has to mention before its refusal counts as evidence
# that the company guides somewhere this tool cannot read.
_GUIDANCE_WORD = re.compile(r'(?i)\bguidance\b|\boutlook\b|\btargets?\b')

_MAX_STATEMENT_CHARS = 420
# A section heading is short. "Second Quarter 2026 Considerations" is one at
# 34 characters; a sentence that merely opens with a period phrase is not.
_MAX_HEADING_CHARS = 60
_MAX_NUMERIC_DENSITY = 0.28
# How far after a forward cue an amount can sit and still belong to it.
_CUE_TO_AMOUNT_CHARS = 200

LIMITATIONS = (
    "This tool reports what management said, not whether it was achieved. It "
    "does not compute a beat or a miss: grading a guide requires the actual "
    "on the same basis (GAAP vs non-GAAP EPS can differ severalfold) for the "
    "same fiscal period, and the available actuals are labelled by calendar "
    "quarter-end. Guidance rendered only in tables is refused rather than "
    "parsed, because the flattening interleaves columns. Guidance given "
    "verbally on the earnings call is not in this corpus at all."
)


def _split_statements(text: str) -> List[str]:
  """Break a release into candidate statements.

  The blank line comes first and matters most. Coca-Cola guides in prose
  paragraphs closed by a status annotation -- "the company expects to deliver
  organic revenue (non-GAAP) growth of 4% to 5%. -- No Update" -- and there is
  no sentence punctuation between one paragraph and the next. Collapsing the
  document to a single line welds that guide onto the currency paragraph
  after it and reports two claims as one.

  A single newline is the opposite case: these releases hard-wrap mid
  sentence, so splitting on one would cut "4% to" away from "5%".

  Sentence punctuation alone is not enough either. A release opens with its
  headline, its dateline and its first result sentence in one unbroken run,
  so "Reaffirms Fiscal 2026 Guidance" ends up joined to "reported sales of
  $47.9 billion" and the actual becomes the guide.
  """
  paragraphs = re.split(r'\n[ \t]*\n', text or '')
  parts: List[str] = []
  for paragraph in paragraphs:
    paragraph = re.sub(r'\s+', ' ', paragraph)
    paragraph = _BULLETS.sub(_BREAK, paragraph)
    paragraph = _DATELINE.sub(_BREAK, paragraph)
    paragraph = _OPENER.sub(_BREAK, paragraph)
    paragraph = _PAGE_FURNITURE.sub(_BREAK, paragraph)
    for block in paragraph.split(_BREAK):
      parts.extend(re.split(r'(?<=[.!?])\s+(?=[A-Z(“"])', block))
  return [part.strip() for part in parts if part.strip()]


def _numeric_density(statement: str) -> float:
  tokens = statement.split()
  if not tokens:
    return 1.0
  return sum(1 for token in tokens if re.search(r'\d', token)) / len(tokens)


def _looks_tabular(statement: str) -> bool:
  return bool(
      TABLE_CHARS.search(statement)
      or ADJACENT_AMOUNTS.search(statement)
      or COLUMN_HEADERS.search(statement)
      or TABLE_CAPTION.search(statement)
      or DANGLING_RANGE.search(statement)
      or len(PERIOD.findall(statement)) > _MAX_PERIOD_PHRASES
      or _numeric_density(statement) > _MAX_NUMERIC_DENSITY
      or len(BARE_DECIMAL.findall(statement)) > _MAX_BARE_DECIMALS
      or (AS_FOLLOWS.search(statement) and len(AMOUNT.findall(statement)) >= 2))


def _amount_belongs_to_a_cue(statement: str) -> bool:
  """An amount must follow a forward cue closely enough to be its object.

  "we delivered 48% revenue growth and raised our full-year guidance" has both
  a cue and an amount, and the amount is a reported result sitting in front of
  the cue rather than after it.
  """
  cue_positions = [match.end() for match in STRONG_CUE.finditer(statement)]
  cue_positions += [match.end() for match in DIRECTIONAL_CUE.finditer(statement)]
  if not cue_positions:
    return False
  for amount in AMOUNT.finditer(statement):
    for position in cue_positions:
      if 0 <= amount.start() - position <= _CUE_TO_AMOUNT_CHARS:
        return True
  return False


def _classify(statement: str) -> str:
  """Why a candidate was kept or dropped. 'accept' is the only keeper."""
  if _looks_tabular(statement):
    return "table_layout"
  if len(statement) > _MAX_STATEMENT_CHARS:
    return "too_long"
  if BOILERPLATE.search(statement):
    return "boilerplate"
  if BACKWARD_REFERENCE.search(statement):
    return "backward_reference"
  if QUOTATION.search(statement):
    return "quotation"
  if not (STRONG_CUE.search(statement) or DIRECTIONAL_CUE.search(statement)):
    return "no_cue"
  if not AMOUNT.search(statement):
    return "no_amount"
  tail = statement.split()[-1].strip('.:,;')
  if re.fullmatch(r'(?i)guidance|outlook', tail):
    return "heading_fragment"
  if not _amount_belongs_to_a_cue(statement):
    return "past_tense_report"
  return "accept"


def _longest_period(statement: str) -> Optional[re.Match]:
  """The most specific period phrase present.

  "the second quarter of fiscal 2027" and "fiscal 2027" both match; taking the
  shorter one turns a quarterly guide into an annual one.
  """
  best: Optional[re.Match] = None
  for match in PERIOD.finditer(statement):
    if best is None or len(match.group(0)) > len(best.group(0)):
      best = match
  return best


def _period_for(statement: str,
                inherited: Optional[str]) -> Tuple[Optional[str], Optional[str], List[str]]:
  caveats: List[str] = []
  own = _longest_period(statement)
  if own is not None:
    first_amount = AMOUNT.search(statement)
    if first_amount is not None and own.start() > first_amount.start():
      # Target's "guidance range of $9.90 to $10.90, which includes second
      # quarter tariff refund benefits" is full-year guidance; the quarter
      # belongs to the benefit, not to the guide.
      caveats.append("period_may_not_be_the_guided_period")
    return own.group(0).strip(), "in_statement", caveats
  if inherited:
    caveats.append("period_inherited_from_section_lead_in")
    return inherited, "section_lead_in", caveats
  return None, None, caveats


def _is_period_heading(candidate: str) -> bool:
  """A short line that is only a period phrase, e.g. "Full Year 2026".

  Once paragraphs are split properly these headings stop being glued to the
  sentences beneath them, and without this every Coca-Cola guidance line
  reports no period at all.
  """
  if len(candidate) > _MAX_HEADING_CHARS:
    return False
  if AMOUNT.search(candidate):
    return False
  return PERIOD.search(candidate) is not None


def _scan_guidance(text: str) -> Dict[str, Any]:
  """Pure text scan. No I/O, so it can be tested against filing strings."""
  statements: List[Dict[str, Any]] = []
  rejected: Dict[str, int] = {}
  refused_text_mentions_guidance = False
  inherited: Optional[str] = None

  for candidate in _split_statements(text):
    verdict = _classify(candidate)

    sets_context = _is_period_heading(candidate) or (
        SECTION_LEAD_IN.search(candidate) is not None
        and (STRONG_CUE.search(candidate) or DIRECTIONAL_CUE.search(candidate)))
    if sets_context:
      lead_in = _longest_period(candidate)
      if lead_in is not None:
        inherited = lead_in.group(0).strip()

    if verdict != "accept":
      rejected[verdict] = rejected.get(verdict, 0) + 1
      if verdict == "table_layout" and _GUIDANCE_WORD.search(candidate):
        refused_text_mentions_guidance = True
      continue

    period_label, period_source, caveats = _period_for(candidate, inherited)
    if period_label is None:
      caveats.append("no_period_identified")
    if PAST_REPORT.search(candidate):
      caveats.append("contains_past_tense_reporting")
    statements.append({
        "text": candidate,
        "period_label": period_label,
        "period_source": period_source,
        "caveats": caveats,
    })

  return {
      "statements": statements,
      "rejected": rejected,
      "refused_text_mentions_guidance": refused_text_mentions_guidance,
  }


def _empty(ticker: str, quarters: int, error: Optional[str],
           reason: Optional[str], sources: Optional[List[Dict[str, Any]]] = None,
           success: bool = True) -> Dict[str, Any]:
  """Every return path carries the same keys.

  extract_customer_concentration once dropped a documented field on its error
  path, so a caller reading it got a KeyError rather than an answer.
  """
  return {
      "ticker": ticker.upper(),
      "success": success,
      "error": error,
      "quarters_requested": quarters,
      "sources": sources or [],
      "guidance_found": False,
      "statement_count": 0,
      "statements": [],
      "no_guidance_reason": reason,
      "guidance_may_be_table_only": False,
      "limitations": LIMITATIONS,
  }


def extract_guidance(ticker: str, quarters: int = 4) -> Dict[str, Any]:
  """Forward guidance statements from 8-K Item 2.02 earnings releases.

  Returns the sentence as filed with the filing it came from. Makes no claim
  about whether the guidance was met -- see LIMITATIONS.
  """
  try:
    releases = get_earnings_releases(ticker, max_quarters=quarters)
  except Exception as exc:  # noqa: BLE001 - reported, not swallowed
    return _empty(ticker, quarters, f"{type(exc).__name__}: {exc}",
                  "no_earnings_releases_found", success=False)

  rows = releases.get("releases") or []
  if not rows:
    return _empty(ticker, quarters,
                  releases.get("error") or "no earnings releases returned",
                  "no_earnings_releases_found",
                  success=bool(releases.get("success")))

  sources: List[Dict[str, Any]] = []
  statements: List[Dict[str, Any]] = []
  any_text = False
  table_only_signal = False

  for row in rows:
    text = row.get("text")
    source = {
        "filing_date": row.get("filing_date"),
        "accession": row.get("accession_number"),
        "url": row.get("filing_url"),
        "attachment": row.get("attachment_doc"),
        "text_available": bool(text),
        "statements_found": 0,
        "table_regions_refused": 0,
    }
    if not text:
      sources.append(source)
      continue

    any_text = True
    found = _scan_guidance(text)
    source["statements_found"] = len(found["statements"])
    source["table_regions_refused"] = found["rejected"].get("table_layout", 0)
    table_only_signal = table_only_signal or found["refused_text_mentions_guidance"]
    for statement in found["statements"]:
      statements.append({
          **statement,
          "filing_date": row.get("filing_date"),
          "accession": row.get("accession_number"),
          "source_url": row.get("filing_url"),
      })
    sources.append(source)

  if not any_text:
    result = _empty(ticker, quarters, None, "release_text_unavailable", sources)
    return result

  if not statements:
    result = _empty(ticker, quarters, None, "no_guidance_language_found", sources)
    result["guidance_may_be_table_only"] = table_only_signal
    return result

  return {
      "ticker": ticker.upper(),
      "success": True,
      "error": None,
      "quarters_requested": quarters,
      "sources": sources,
      "guidance_found": True,
      "statement_count": len(statements),
      "statements": statements,
      "no_guidance_reason": None,
      "guidance_may_be_table_only": False,
      "limitations": LIMITATIONS,
  }
