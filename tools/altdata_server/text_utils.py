"""Shared text-analysis utilities for altdata classifiers.

These functions are deliberately small, pure, and sector-agnostic. They exist
because the capex and policy classifiers previously used naive substring matching
(`"cut"` inside "circuit", `"ban"` inside "banking", `"invest"` inside
"investigation") and timezone-naive date parsing (which crashed and was silently
swallowed). Keeping the logic here makes it unit-testable without any network.

Design rule for all-sectors robustness: classifiers should key on universal
direction verbs and dollar magnitude, never on sector-specific nouns.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import List, Optional

__all__ = ["text_contains", "count_matches", "extract_dollar_amounts", "parse_news_date"]


# ---------------------------------------------------------------------------
# Keyword matching
# ---------------------------------------------------------------------------

def text_contains(text: str, term: str) -> bool:
    """True if `term` appears in `text`.

    Single words use word-boundary matching (`\\bterm\\b`) so "cut" does not
    match "circuit" and "ban" does not match "banking". Multi-word or hyphenated
    phrases ("data center", "tax credit") use plain substring matching, since a
    phrase is already specific enough to be safe.
    """
    if not text or not term:
        return False
    text_l = text.lower()
    term_l = term.lower().strip()
    if not term_l:
        return False
    if " " in term_l or "-" in term_l:
        return term_l in text_l
    return re.search(rf"\b{re.escape(term_l)}\b", text_l) is not None


def count_matches(text: str, terms) -> int:
    """Number of distinct terms from `terms` present in `text`."""
    return sum(1 for t in terms if text_contains(text, t))


# ---------------------------------------------------------------------------
# Dollar amount extraction
# ---------------------------------------------------------------------------

_UNIT_MULTIPLIERS = {
    "trillion": 1e12, "trn": 1e12, "t": 1e12,
    "billion": 1e9, "bn": 1e9, "b": 1e9,
    "million": 1e6, "mn": 1e6, "m": 1e6,
}

# $ + (comma-grouped number) + optional space + unit word/abbrev.
_AMOUNT_PATTERN = re.compile(
    r"\$\s*([\d,]+(?:\.\d+)?)\s*"
    r"(trillion|billion|million|trn|bn|mn|[tbm])\b",
    re.IGNORECASE,
)


def extract_dollar_amounts(text: str) -> List[float]:
    """Extract USD amounts like '$2.5 billion', '$300mn', '$1,200 million'.

    Returns a list of floats in dollars. Requires an explicit unit so bare
    numbers ('$10') are ignored (news capex figures are always unit-qualified).
    """
    if not text:
        return []
    out: List[float] = []
    for m in _AMOUNT_PATTERN.finditer(text):
        raw = m.group(1).replace(",", "")
        try:
            val = float(raw)
        except ValueError:
            continue
        mult = _UNIT_MULTIPLIERS.get(m.group(2).lower())
        if mult:
            out.append(val * mult)
    return out


# ---------------------------------------------------------------------------
# Date parsing — ALWAYS returns timezone-aware UTC or None (never naive)
# ---------------------------------------------------------------------------

def parse_news_date(s: str) -> Optional[datetime]:
    """Parse a news/article date string into a tz-aware UTC datetime, or None.

    Handles ISO 8601 (incl. trailing 'Z' and bare 'YYYY-MM-DD') and RFC 2822
    ('Mon, 01 Jun 2026 12:00:00 GMT'). A naive result is forced to UTC so callers
    can always compare against `datetime.now(timezone.utc)` without crashing.
    """
    if not s or not isinstance(s, str):
        return None
    s = s.strip()
    if not s:
        return None

    dt: Optional[datetime] = None
    # ISO 8601 (fromisoformat handles bare dates and offsets; map 'Z' first)
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        # RFC 2822 (email-style)
        try:
            from email.utils import parsedate_to_datetime
            dt = parsedate_to_datetime(s)
        except (TypeError, ValueError, IndexError):
            return None

    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)
