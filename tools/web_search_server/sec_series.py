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

    def dimension_member(self, axis: str) -> Optional[str]:
        return self.dimensions.get(axis)


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

    def total(self) -> Optional[float]:
        """Sum across distinct facts, or None when there are none.

        None rather than 0.0 deliberately: for a share count, zero is a
        meaningful and alarming value, so absence must stay distinguishable
        from it.
        """
        facts = self.deduplicated()
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

    def latest_undimensioned(self) -> Optional[ConceptFact]:
        """The consolidated fact for the most recent, longest period here.

        A 10-K's XBRL carries three comparative years for a duration concept,
        and often quarterly durations alongside them. Sorting on the raw period
        string is wrong: "duration_2025-10-27_2026-01-25" (Q4) sorts above
        "duration_2025-01-27_2026-01-25" (FY) because "10" > "01" at the month
        position, so a quarter's figure would be returned as the year's -- a
        roughly 4x understatement wearing a plausible face.

        Ranked by period end first, then by span, so the annual fact wins over
        a quarter ending the same day.
        """
        candidates = self.undimensioned()
        if not candidates:
            return None
        return max(candidates, key=lambda f: _period_rank(f.period))

    def by_axis(self, axis: str) -> Dict[str, float]:
        """Facts keyed by their member on one axis.

        Undimensioned facts land under the empty-string key, which keeps
        single-class filers representable without a separate code path.
        """
        out: Dict[str, float] = {}
        for fact in self.deduplicated():
            out[fact.dimension_member(axis) or ""] = fact.value
        return out


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
    filings = company.get_filings(form=form).head(limit)

    points: List[FilingPoint] = []
    for filing in filings:
        _throttle()
        try:
            xbrl = filing.xbrl()
            if xbrl is None:
                continue
            frame = xbrl.facts.query().by_concept(concept).to_dataframe()
        except Exception:
            # A single unparseable filing must not sink the whole series.
            continue

        if frame is None or len(frame) == 0:
            continue

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
            ))

        if facts:
            points.append(FilingPoint(
                filing_date=str(filing.filing_date),
                form=str(filing.form),
                accession=str(getattr(filing, "accession_no", "")),
                facts=facts,
            ))

    if not points:
        raise NotCovered(
            f"{ticker}: concept {concept!r} found in none of the last "
            f"{limit} {form} filings")
    return points
