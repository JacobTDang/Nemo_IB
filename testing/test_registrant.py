"""Which registrant a job reads, and the empty answer that is not one.

edgartools resolves a ticker from a parquet file bundled with the package
before it asks SEC. A ticker missing from that file maps to nothing, and
`Company(ticker)` does not raise for it: it returns a company whose CIK is
-999999999 and whose filing list is empty. Measured against the 2026-10-19
universe, 103 of 1,810 eligible names resolved that way (#130), and every one
of them resolves by the CIK the universe already holds.

So a caller with a CIK asks by it, and a caller with only a ticker gets a
refusal rather than an empty folder -- empty is what a quiet company returns,
and the two must not be told apart by luck.
"""
import edgar
import pytest

from research import registrant

SENTINEL = -999999999


@pytest.fixture
def asked(monkeypatch):
    """edgartools as it stood on 2026-10-20, and every key it was asked for.

    The bundled ticker map holds INTC and not DMC. SEC's submissions index
    holds both, and a CIK it does not hold comes back `not_found` with an
    empty folder -- edgartools' answer to a 404, which does not raise either.
    """
    keys = []
    bundled = {"INTC": 50863}
    folders = {50863: ["intc-8k"], 1047340: ["dmc-8k"]}

    class _Company:
        def __init__(self, key):
            keys.append(key)
            self.cik = bundled.get(key, SENTINEL) if isinstance(key, str) \
                else key
            self.not_found = self.cik not in folders

        def get_filings(self, form=None):
            return list(folders.get(self.cik, []))

    monkeypatch.setattr(edgar, "Company", _Company)
    return keys


def test_a_name_with_a_cik_is_asked_for_by_its_cik(asked):
    """The universe records the CIK zero-padded, as SEC's file gives it.
    Asking by it never touches the bundled ticker map, which is the map that
    does not know DMC."""
    found = registrant.company("DMC", "0001047340")

    assert found.cik == 1047340
    assert found.get_filings(form="8-K") == ["dmc-8k"]
    assert asked == [1047340], "the ticker was asked for despite a CIK"


def test_a_ticker_the_bundled_map_lacks_is_refused_not_read_as_empty(asked):
    """Refused on the sentinel itself, before its submissions are read. The
    `not_found` check would catch it too, but only after asking SEC for CIK
    -999999999, and with a message that sends whoever reads it looking for a
    wrong CIK rather than an unmapped ticker."""
    with pytest.raises(LookupError, match="does not map DMC"):
        registrant.company("DMC")


def test_a_ticker_the_bundled_map_holds_still_resolves(asked):
    """A caller with only a ticker keeps working for the names edgartools
    knows. The refusal is for the sentinel, not for asking by ticker."""
    assert registrant.company("INTC").cik == 50863
    assert asked == ["INTC"]


def test_a_cik_sec_has_no_submissions_for_is_refused(asked):
    """The CIK route has its own empty answer. edgartools turns SEC's 404
    into a company flagged `not_found` with no filings, the same silence the
    sentinel is, arriving by a different door."""
    with pytest.raises(LookupError, match="GONE"):
        registrant.company("GONE", "1234567")
