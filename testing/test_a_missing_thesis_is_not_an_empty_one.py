"""A thesis that does not exist is not a thesis with nothing in it.

`get_thesis_evolution(1)` answered, against a book holding zero theses:

    {"success": true, "thesis_id": 1, "ticker": null,
     "current_conviction": null, "falsifiers": null,
     "variant_perception": null, "evolution_count": 0, "evolution": []}

That is byte-for-byte what a real, freshly-recorded thesis returns before its
first check-in. A caller reading `evolution_count: 0` on a successful response
concludes the thesis is intact and has simply not moved -- the reflexivity
trace is empty because nothing has happened yet. There is no thesis. The
`success: true` is the whole problem: it certifies an answer about an object
the book has never held.

The information was already there. `record_thesis_evolution` on the same id
raises `ValueError: thesis 999999 not found` from the state layer, so the
read path had everything it needed to refuse and chose to report null instead.

Two rules are held here:

1. A missing thesis is refused. An existing thesis with no evolution rows is
   NOT refused -- it is a real, answerable empty, and it is distinguishable
   because its ticker and conviction come back populated.

2. The refusal is written for a reader. `"ValueError: thesis 999999 not
   found"` leaks the exception class of the process that failed, which tells
   a caller about our internals rather than about their request, and reads as
   a crash rather than a decision. Every other refusal in this codebase is
   prose -- `unresolved_symbol_error` says "did not resolve to a listed
   security ... This is a failed lookup, not a company without financials".

What must NOT change: `analyze_exposures` and `get_thesis_evolution` read book
state, and the shipped data-source image has none. Returning an empty book
there is the truthful answer and stays successful -- see README, `financial`
section. Refusing one *named* thesis the book does not hold is a different
question with a different honest answer, and it is the same answer on a book
host as on a data-source host.
"""
import asyncio
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture()
def book(tmp_path, monkeypatch):
    """An isolated, schema'd, empty book -- exactly a data-source host."""
    monkeypatch.setenv("NEMO_DB_PATH", str(tmp_path / "session.db"))
    from state.schema import init_schema
    init_schema()
    from tools.financial_modeling_engine.analysis_tools import Financial_Analysis
    return Financial_Analysis()


def _call(coro):
    payload = asyncio.run(coro)
    return json.loads(payload[0].text)


# An id no book in this repo's fixtures reaches.
MISSING_ID = 999999


def _seed(ticker="TEST_MISSING", confidence=0.6):
    from state.theses import insert_thesis
    return insert_thesis(
        ticker=ticker, recommendation="BUY", signal="bullish",
        target_price=100.0, stop_loss=80.0, confidence=confidence,
        analysis_summary="seeded for the missing-thesis tests",
        key_assumptions=["a"], data_gaps=[], full_report_md="r",
        falsifiers=["revenue growth turns negative"],
        variant_perception="consensus underrates the mix shift",
    )


def test_reading_a_thesis_the_book_does_not_hold_is_refused(book):
    out = _call(book.get_thesis_evolution(MISSING_ID))
    assert out.get("success") is False, (
        f"a thesis the book has never held was answered successfully: {out}")


def test_the_refusal_does_not_report_a_count_over_nothing(book):
    """`evolution_count: 0` is a measurement, and there was nothing to measure."""
    out = _call(book.get_thesis_evolution(MISSING_ID))
    assert out.get("evolution_count") is None, (
        "the refusal still counts the evolution rows of a thesis that does "
        f"not exist: evolution_count={out.get('evolution_count')!r}")
    assert not out.get("evolution"), (
        "the refusal still carries an evolution log")


def test_the_refusal_names_the_thesis_it_could_not_find(book):
    out = _call(book.get_thesis_evolution(MISSING_ID))
    error = str(out.get("error") or "")
    assert str(MISSING_ID) in error, f"the refusal does not say which id: {error!r}"
    assert "thesis" in error.lower(), f"the refusal does not say what was missing: {error!r}"


@pytest.mark.parametrize("call_name", ["get_thesis_evolution", "record_thesis_evolution"])
def test_the_refusal_is_prose_not_a_python_exception(book, call_name):
    """A leaked exception class describes our process, not the caller's request."""
    if call_name == "get_thesis_evolution":
        out = _call(book.get_thesis_evolution(MISSING_ID))
    else:
        out = _call(book.record_thesis_evolution(
            MISSING_ID, "an observation", -0.05, None))
    assert out.get("success") is False
    error = str(out.get("error") or "")
    for leak in ("ValueError", "KeyError", "TypeError", "Traceback",
                 "sqlite3", "OperationalError"):
        assert leak not in error, (
            f"{call_name} leaked the exception class into the error: {error!r}")


def test_an_existing_thesis_with_no_events_is_a_real_empty(book):
    """The case the refusal must not swallow: nothing has happened *yet*."""
    tid = _seed()
    out = _call(book.get_thesis_evolution(tid))
    assert out.get("success") is True, (
        f"a real thesis awaiting its first check-in was refused: {out}")
    assert out.get("evolution_count") == 0
    assert out.get("evolution") == []
    # This is what separates it from the missing case for any caller.
    assert out.get("ticker") == "TEST_MISSING"
    assert out.get("current_conviction") is not None
    assert out.get("falsifiers") == ["revenue growth turns negative"]


def test_a_thesis_with_events_still_reads_back(book):
    tid = _seed(ticker="TEST_MOVED")
    rec = _call(book.record_thesis_evolution(tid, "beat and raised", 0.05, "earnings"))
    assert rec.get("success") is True, rec
    out = _call(book.get_thesis_evolution(tid))
    assert out.get("success") is True
    assert out.get("evolution_count") == 1
    assert out["evolution"][0]["observation"] == "beat and raised"


def test_an_empty_book_is_still_an_empty_book_not_a_failure(book):
    """The data-source-host invariant this change must not break.

    README, `financial` section: analyze_exposures reads book state a
    data-source host does not have, and returns an empty result rather than
    failing, because empty is the truthful answer there.
    """
    out = _call(book.analyze_exposures_tool())
    assert isinstance(out, dict)
    assert not out.get("error"), (
        f"an empty book was reported as a failure: {out.get('error')!r}")
