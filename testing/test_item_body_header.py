"""Item 1A and Item 7 are named in the table of contents and again as the body.

The contents line comes first. A fixed offset of 30,000 characters misses a
body that starts earlier, which is where Apple's Item 1A header sits. The
body is the header whose own end marker — Item 1B, or Item 7A then Item 8 —
is furthest away.
"""
from pathlib import Path

import pytest

from tools.web_search_server import sec_utils as su

_EXCERPTS = Path(__file__).parent / "item_excerpts"


def _text(name: str) -> str:
    return (_EXCERPTS / name).read_text()


class _Filing:
    def __init__(self, text):
        self._text = text

    def text(self):
        return self._text


def _use(monkeypatch, text):
    monkeypatch.setattr(su, "get_latest_filing", lambda *args, **kwargs: {
        "filing_object": _Filing(text),
        "filing_date": "2026-02-26",
        "url": None,
    })


def test_item_1a_starts_at_the_body_header_not_the_contents(monkeypatch):
    """NVDA and AAPL excerpts. The contents line is padded out to a page
    number; the body header is not, and the sentence between them is Item 1."""
    for name, absent, heading in (
        ("nvda_item_1a.txt", "http://www.sec.gov",
         "Risks Related to Our Industry and Markets"),
        ("aapl_item_1a.txt", "inactive textual references",
         "Macroeconomic and Industry Risks"),
    ):
        text = _text(name)
        _use(monkeypatch, text)
        out = su.extract_risk_factors("X")
        assert out["success"], out.get("error")
        assert out["text"].startswith("Item 1A. Risk Factors\n"), out["text"][:80]
        assert absent not in out["text"]
        assert heading in [h["heading"] for h in out["section_headings"]]
        section = su._extract_section_from_filing_obj(_Filing(text), "1A")
        assert section.startswith("Item 1A. Risk Factors\n")
        assert absent not in section

    # Letter-spaced "RIS K" is still a header. The body one is the section.
    spaced = (
        "Item 1A. RIS K FACTORS  4\n"
        "Item 1B. Unresolved Staff Comments  5\n"
        "This sentence is the business section, not the risk factors.\n"
        "Item 1A. RIS K FACTORS\n"
        "The following risks are the body.\n"
        "Item 1B. Unresolved Staff Comments\n"
    )
    _use(monkeypatch, spaced)
    spaced_out = su.extract_risk_factors("X")
    assert spaced_out["text"].startswith("Item 1A. RIS K FACTORS\nThe following")


def test_item_7_starts_at_the_body_header_not_the_contents(monkeypatch):
    """The contents Item 7 is padded; the body header is the one Item 7A
    sits far after. A nearby "Item 8" citation does not end the section."""
    for name, absent in (
        ("nvda_item_7.txt", "Item 6. [Reserved]"),
        ("aapl_item_7.txt", "2025 Form 10-K"),
    ):
        text = _text(name)
        _use(monkeypatch, text)
        out = su.extract_mda("X")
        assert out["success"], out.get("error")
        assert out["text"].startswith("Item 7. Management"), out["text"][:90]
        assert "Item 7.       Management" not in out["text"][:40]
        assert "Item 7.         Management" not in out["text"][:40]
        assert absent not in out["text"]
        section = su._extract_section_from_filing_obj(_Filing(text), "7")
        assert section.startswith("Item 7. Management")
        assert absent not in section


def test_a_filing_text_failure_propagates():
    """A failed read of the filing is not an Item that could not be found."""
    class _Broken:
        def text(self):
            raise RuntimeError("SEC read failed for the 10-K")

    with pytest.raises(RuntimeError, match="SEC read failed"):
        su._extract_section_from_filing_obj(_Broken(), "1A")
