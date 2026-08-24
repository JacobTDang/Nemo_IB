"""Multi-filing XBRL access.

Everything else in this package reads a single filing. `get_segment_financials`
gets five years of history from one 10-K because a filing's XBRL carries
comparative periods, and `get_historical_fcf` reads only the latest filing
despite its name. That is fine for income-statement items and useless for
anything reported as an instant on the cover page.

Share count is the motivating case. `dei:EntityCommonStockSharesOutstanding` is
one instant per filing, so a dilution series requires walking filings over time.

Two hard-won details about dimension handling, both verified against live EDGAR
before this module was written:

1. `facts.query().to_dataframe()` carries **no dimension column**. Multi-class
   filers return several rows with identical labels, distinguishable only by
   `context_ref`. Class identity comes from `xbrl.contexts[ref].dimensions`.
2. `facts.query().by_dimension(axis, member)` returns **empty** for cover-page
   concepts even when the facts plainly exist. The dimension-query path that
   `get_segment_financials` uses does not work here.

Getting this wrong is not a crash. GOOGL reports Class A, B, and C separately;
taking the first row yields 5.868bn against a true 12.23bn — a 52% understatement
that looks entirely plausible.
"""
from __future__ import annotations

import os
import re
from collections import Counter
from datetime import date
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from edgar import Company, set_identity

# SEC fair access allows roughly 10 requests/second. Filings are fetched
# sequentially and each one costs several requests, so this is a floor on the
# gap between filings rather than a precise budget.
_MIN_REQUEST_GAP_S = 0.15
_last_request_at = 0.0
_throttle_lock = threading.Lock()

_identity_set = False


class NotCovered(Exception):
    """A concept appears in none of the filings examined.

    Distinct from an error. It means this filer does not tag the concept, which
    is information the caller needs — not a reason to return zero.
    """


def _require_identity() -> str:
    """Resolve the SEC contact identity, or refuse to make the request.

    Resolved on use rather than at import so this module can be imported without
    credentials. SEC fair access asks for a real contact address; defaulting to a
    placeholder misrepresents the caller, so an unset value stops us here.
    """
    global _identity_set
    email = os.getenv("SEC_EMAIL", "").strip()
    if not email:
        raise ValueError(
            "SEC_EMAIL is not set. SEC fair access requires a real contact "
            "address in the User-Agent header. Set SEC_EMAIL in your .env file.")
    name = os.getenv("NAME", "").strip() or "Investment Analyst"
    identity = f"{name} {email}"
    if not _identity_set:
        set_identity(identity)
        _identity_set = True
    return identity


def _throttle() -> None:
    global _last_request_at
    with _throttle_lock:
        elapsed = time.monotonic() - _last_request_at
        if elapsed < _MIN_REQUEST_GAP_S:
            time.sleep(_MIN_REQUEST_GAP_S - elapsed)
        _last_request_at = time.monotonic()


def _clean_number(value: Any) -> Optional[float]:
    """Coerce a cell to a float, treating NaN and blanks as absent.

    pandas yields float('nan') for a missing numeric cell, and nan is truthy.
    That makes the obvious `row.get(a) or row.get(b)` fallback silently keep the
    nan instead of falling through, and float(nan) succeeds, so a bad value
    propagates all the way into the series looking like real data.
    """
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    return number


def _clean_decimals(value: Any) -> Optional[int]:
    """XBRL `decimals`, or None when absent or non-numeric.

    "INF" marks an exactly-stated value and is treated as maximum precision.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    if text.upper() == "INF":
        return 99
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def _clean_period(row: Any) -> str:
    """Best available period label for a fact.

    Instant concepts (share count) populate period_instant. Duration concepts
    (SBC, revenue) leave it NaN and carry period_key instead.
    """
    def _get(key: str) -> Optional[str]:
        try:
            value = row.get(key)
        except AttributeError:
            return None
        if value is None:
            return None
        text = str(value)
        if not text or text.lower() == "nan":
            return None
        return text

    return _get("period_instant") or _get("period_key") or ""


def _period_rank(period: str) -> tuple:
    """Sort key for a period label: (end date, span in days).

    Handles "duration_START_END", "instant_DATE", and a bare date. Anything
    unparseable sorts lowest so it never displaces a well-formed period.
    """
    if not period:
        return ("", -1)
    text = period.strip()
    for prefix in ("duration_", "instant_"):
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    parts = [p for p in text.split("_") if p]
    dates: List[date] = []
    for part in parts:
        try:
            dates.append(date.fromisoformat(part))
        except ValueError:
            continue
    if not dates:
        return ("", -1)
    if len(dates) == 1:
        return (dates[0].isoformat(), 0)
    start, end = min(dates), max(dates)
    return (end.isoformat(), (end - start).days)


def _in_span(days: int, low: Optional[int], high: Optional[int]) -> bool:
    """Whether a period length falls inside an inclusive window.

    `_period_rank` returns -1 for a period it could not parse. That is outside
    every window deliberately: a fact whose period is unknown must not be
    handed back as the annual figure because no lower bound happened to be set.
    """
    if days < 0:
        return False
    if low is not None and days < low:
        return False
    if high is not None and days > high:
        return False
    return True


def _concept_matches(row_concept: Any, requested: str) -> bool:
    """Whether a fact row is the requested concept rather than a longer name.

    `facts.query().by_concept(name)` matches by **prefix**, which is not what
    any caller here wants. Querying `us-gaap:Assets` against MSFT's FY2026 10-K
    returns `us-gaap:AssetsCurrent` alongside it, and the current-assets fact
    shares the balance-sheet context, so it survives every dimension filter and
    can win `latest_undimensioned()`. That returned 207.7bn as MSFT's total
    assets against a real 758.4bn — plausible enough to go unnoticed, and it is
    the denominator of the accrual ratio.

    Prefixes are separated with either ':' or '_' depending on the source, so
    both spellings of the same element compare equal.
    """
    if row_concept is None:
        return False
    left = str(row_concept).strip().replace("_", ":")
    right = str(requested).strip().replace("_", ":")
    return left == right


# Wrapper tokens filers put around the measure in a unit reference. SAP emits
# "Unit_Standard_EUR_<opaque>", Alibaba "U_CNY", TSMC a bare "twd".
_UNIT_WRAPPER_TOKENS = {"unit", "u", "standard", "measure", "iso4217"}


def currency_of(unit_ref: Any) -> Optional[str]:
    """ISO-style currency code for an XBRL unit reference, or None.

    Every fact in a domestic 10-K is denominated in usd, so nothing in this
    package needed the unit until foreign private issuers arrived. TSM reports
    in TWD, SAP and ASML in EUR, NVO in DKK, BABA in CNY -- and a bare
    3,809,054,300,000 reads as $3.8 trillion.

    None means "not a plain amount of money", which covers share counts,
    percentages, headcounts, per-share amounts and FX rates. Guessing a
    currency for `usdPerShare` would put a price where a total belongs, and
    Enbridge's 2017 40-F names its units `Unit12`, so "no currency" has to
    stay expressible rather than being forced to a default.

    The three-letter token is taken as tagged rather than validated against
    ISO 4217: a whitelist would silently drop whichever currency it had not
    seen, and that is the failure this package keeps relearning.
    """
    text = str(unit_ref or "").strip()
    if not text:
        return None
    lowered = text.lower()
    # "usdPerShare", "dkkPerUSD", "Unit_Divide_EUR_shares_..." are ratios. The
    # numerator is a currency but the fact is not an amount of money.
    if "per" in lowered or "divide" in lowered:
        return None
    for part in re.split(r"[_:]", text):
        if not part or part.lower() in _UNIT_WRAPPER_TOKENS:
            continue
        return part.upper() if re.fullmatch(r"[A-Za-z]{3}", part) else None
    return None


def resolve_dimensions(xbrl: Any, context_ref: str) -> Dict[str, str]:
    """Map a fact's context reference to its dimension members.

    Returns an empty dict for undimensioned facts, for a context the filing does
    not define, and for a context object that carries no dimensions attribute.
    A malformed filing should degrade to "no dimensions", not crash a series.
    """
    contexts = getattr(xbrl, "contexts", None)
    if not contexts:
        return {}
    context = contexts.get(context_ref) if hasattr(contexts, "get") else None
    if context is None:
        return {}
    dimensions = getattr(context, "dimensions", None)
    if not isinstance(dimensions, dict):
        return {}
    return dict(dimensions)


@dataclass
class ConceptFact:
    """One tagged value from one filing, with its dimensional context."""

    value: float
    period: str
    dimensions: Dict[str, str] = field(default_factory=dict)
    context_ref: str = ""
    concept: str = ""
    unit: str = ""
    # XBRL precision. Higher means accurate to a smaller unit: -6 is millions,
    # -8 only hundred millions. None when the filer did not state it, which is
    # unknown precision rather than poor precision.
    decimals: Optional[int] = None

    def dimension_member(self, axis: str) -> Optional[str]:
        return self.dimensions.get(axis)

    @property
    def currency(self) -> Optional[str]:
        """Currency this value is denominated in, or None if it is not money."""
        return currency_of(self.unit)


@dataclass
class FilingPoint:
    """Every fact for one concept from one filing."""

    filing_date: str
    form: str
    accession: str
    facts: List[ConceptFact] = field(default_factory=list)

    def deduplicated(self) -> List[ConceptFact]:
        """Facts with exact duplicates removed.

        Some filers emit the same fact twice. Biogen reports its share count
        with identical value, period, and context_ref on two rows; since an
        XBRL context plus a concept defines exactly one fact, those are the
        same fact rather than two share classes. Summing them doubled Biogen's
        share count.

        Keyed on the full triple rather than context alone, so two genuinely
        different values are never collapsed.
        """
        seen = set()
        out: List[ConceptFact] = []
        for fact in self.facts:
            key = (fact.context_ref, fact.period, fact.value)
            if key in seen:
                continue
            seen.add(key)
            out.append(fact)
        return out

    def currencies(self) -> Dict[str, int]:
        """How many distinct facts here are tagged in each currency.

        Empty when the concept is not monetary (share counts, percentages) or
        when the filer names its units opaquely, as Enbridge's 40-F does.
        """
        counts = Counter(f.currency for f in self.deduplicated() if f.currency)
        return dict(counts)

    def reporting_currency(self) -> Optional[str]:
        """The currency the statements themselves are presented in.

        SEC rules let a foreign private issuer add a US-dollar convenience
        translation, but only of the most recent period. The reporting
        currency is therefore the one carrying every comparative year, so the
        commonest currency wins.

        On a tie -- one period's worth of facts in two currencies, which is
        what `limit=1` on a single-period concept produces -- USD loses. A
        filer presenting in dollars has no second currency to be confused
        with, so a USD fact sitting beside another currency is the
        translation rather than the statement.
        """
        counts = self.currencies()
        if not counts:
            return None
        return max(counts.items(), key=lambda kv: (kv[1], kv[0] != "USD"))[0]

    def _in_reporting_currency(self, facts: List[ConceptFact],
                               currency: Optional[str]) -> List[ConceptFact]:
        """Narrow to one currency so translations cannot mix with statements.

        TSM and BABA tag their USD convenience translation with the same
        concept, the same period and the *same context* as the reporting
        figure, and it carries no dimensions -- so it survives every filter
        this class had. `total()` was adding TWD to USD, and
        `latest_undimensioned()` had two maximal candidates and returned
        whichever pandas yielded first.

        Facts with no currency at all (share counts, opaque units) are never
        dropped: for those the whole notion does not apply.
        """
        if currency is not None:
            wanted = currency.upper()
            return [f for f in facts if (f.currency or "").upper() == wanted]
        counts = Counter(f.currency for f in facts if f.currency)
        if len(counts) < 2:
            return facts
        dominant = max(counts.items(), key=lambda kv: (kv[1], kv[0] != "USD"))[0]
        return [f for f in facts if f.currency in (None, dominant)]

    def total(self, currency: Optional[str] = None) -> Optional[float]:
        """Sum across distinct facts, or None when there are none.

        None rather than 0.0 deliberately: for a share count, zero is a
        meaningful and alarming value, so absence must stay distinguishable
        from it.

        Facts in a second currency are excluded rather than added -- see
        `_in_reporting_currency`. Share classes, which carry no currency at
        all, still sum as before: NVO's A and B counts are both `shares`.
        """
        facts = self._in_reporting_currency(self.deduplicated(), currency)
        if not facts:
            return None
        return float(sum(f.value for f in facts))

    def undimensioned(self) -> List[ConceptFact]:
        """Facts carrying no dimensions -- the consolidated figures.

        Dimensioned facts are usually a breakdown of the total, not additions
        to it. NVDA reports 59+ SBC facts in one 10-K, nearly all split by
        award type; summing them would report several times the real expense.
        Share count is the exception where dimensions are additive, which is
        why `total()` exists separately.
        """
        return [f for f in self.deduplicated() if not f.dimensions]

    def latest_undimensioned(
            self, currency: Optional[str] = None,
            span_days: Optional[tuple] = None) -> Optional[ConceptFact]:
        """The consolidated fact for the most recent, longest period here.

        A 10-K's XBRL carries three comparative years for a duration concept,
        and often quarterly durations alongside them. Sorting on the raw period
        string is wrong: "duration_2025-10-27_2026-01-25" (Q4) sorts above
        "duration_2025-01-27_2026-01-25" (FY) because "10" > "01" at the month
        position, so a quarter's figure would be returned as the year's -- a
        roughly 4x understatement wearing a plausible face.

        Ranked by period end first, then by span, so the annual fact wins over
        a quarter ending the same day.

        Period ranking alone is not enough for a foreign private issuer. A USD
        convenience translation shares the period *and* the context with the
        reporting-currency fact, so both rank identically and the winner was
        whichever row pandas produced first. Candidates are narrowed to one
        currency before ranking; pass `currency` to ask for the translation
        deliberately, and get None rather than a substitute if it is absent.

        Nor is ranking enough when the caller wants a period of a particular
        length. A 10-Q carries the year-to-date duration alongside the quarter
        and both end on the same day, so the longest-span rule returns nine
        months where three were asked for. `span_days` is an inclusive
        (minimum, maximum) window on the period length, either bound optional
        as None; a period whose length cannot be determined is outside every
        window rather than inside the first one.
        """
        candidates = self._in_reporting_currency(self.undimensioned(), currency)
        if span_days is not None:
            low, high = span_days
            candidates = [f for f in candidates
                          if _in_span(_period_rank(f.period)[1], low, high)]
        if not candidates:
            return None
        # Period first -- freshness beats precision. Precision settles only a
        # tie, where a filer tagged one concept twice in one context at two
        # different roundings. Amazon does exactly that for income tax.
        return max(candidates,
                   key=lambda f: (_period_rank(f.period),
                                  f.decimals if f.decimals is not None else -999))

    def by_axis(self, axis: str) -> Dict[str, float]:
        """Facts keyed by their member on one axis.

        Undimensioned facts land under the empty-string key, which keeps
        single-class filers representable without a separate code path.
        """
        out: Dict[str, float] = {}
        for fact in self.deduplicated():
            out[fact.dimension_member(axis) or ""] = fact.value
        return out


def concept_point(xbrl: Any, concept: str, filing_date: str, form: str,
                  accession: str = "") -> Optional[FilingPoint]:
    """Every fact for one concept in one already-parsed filing, or None.

    Split out of `fetch_concept_series` so a caller holding an XBRL object can
    read many concepts from it without re-fetching the filing once per concept.
    Reading thirty concepts for one filer costs thirty filing downloads through
    the series walk and one through this, which is the difference between a
    sweep that completes and a sweep that earns an SEC 429.

    None means the concept is not in this filing, which is distinct from a
    filing that could not be parsed -- that is the caller's exception to catch.
    """
    frame = xbrl.facts.query().by_concept(concept).to_dataframe()
    if frame is None or len(frame) == 0:
        return None

    # by_concept is a prefix match. Keep only the concept actually asked
    # for. When the frame carries no concept column at all there is nothing
    # to filter on, so the rows pass through as before rather than every
    # filer being reported uncovered.
    if "concept" in getattr(frame, "columns", []):
        frame = frame[frame["concept"].map(
            lambda c: _concept_matches(c, concept))]
        if len(frame) == 0:
            return None

    facts: List[ConceptFact] = []
    for _, row in frame.iterrows():
        value = _clean_number(row.get("numeric_value"))
        if value is None:
            continue
        context_ref = str(row.get("context_ref") or "")
        facts.append(ConceptFact(
            value=value,
            period=_clean_period(row),
            dimensions=resolve_dimensions(xbrl, context_ref),
            context_ref=context_ref,
            concept=str(row.get("concept") or concept),
            # Blank rather than absent when the frame carries no unit
            # column, so a currency-unaware edgartools degrades to the
            # old behaviour instead of raising.
            unit=str(row.get("unit_ref") or ""),
            decimals=_clean_decimals(row.get("decimals")),
        ))

    if not facts:
        return None
    return FilingPoint(filing_date=filing_date, form=form,
                       accession=accession, facts=facts)


def fetch_concept_series(ticker: str, concept: str, form: str = "10-Q",
                         limit: int = 8) -> List[FilingPoint]:
    """Walk the most recent `limit` filings and pull `concept` from each.

    Returns newest first, matching EDGAR's own ordering. Filings that fail to
    parse are skipped rather than aborting the series — one bad filing should not
    cost the caller the other seven.

    Raises NotCovered when the concept appears in no filing examined.
    """
    _require_identity()

    company = Company(ticker)
    # Amendments are excluded. TSLA's most recent "10-K" is a 10-K/A carrying
    # the Part III proxy information and 37 fact rows -- no financial
    # statements. Left in, it takes a slot in the walk and every concept in the
    # real 10-K behind it reads as untagged.
    filings = company.get_filings(form=form, amendments=False).head(limit)

    points: List[FilingPoint] = []
    for filing in filings:
        _throttle()
        try:
            xbrl = filing.xbrl()
            if xbrl is None:
                continue
            point = concept_point(
                xbrl, concept,
                filing_date=str(filing.filing_date),
                form=str(filing.form),
                accession=str(getattr(filing, "accession_no", "")))
        except Exception:
            # A single unparseable filing must not sink the whole series.
            continue

        if point is not None:
            points.append(point)

    if not points:
        raise NotCovered(
            f"{ticker}: concept {concept!r} found in none of the last "
            f"{limit} {form} filings")
    return points
