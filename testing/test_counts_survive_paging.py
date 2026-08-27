"""A count of the whole set must not change when the page size does.

`get_schedule_13d_filings("INTC")` answers "are there activist investors in
Intel?" with `activist_count: 0` at the default limit and `31` at limit=100.
The rows were truncated first and counted afterwards, so every count described
the page while being named for the set. Zero activists is not a smaller
version of thirty-one; it is the opposite answer.

A sweep of every tool taking a page-size argument found three:

    get_schedule_13d_filings  activist_count 0 -> 31, passive_count 3 -> 69
    extract_8k_events         total_events   3 -> 100
    find_peers_by_sic         peer_count     1 -> 24, unresolved_count 9 -> 76

The congressional tools were repaired earlier and hold steady, which is how
this sweep was built: the same question, asked of everything that pages.

The rule: count before truncating. `rows_returned` describes the page, the
count describes the set, and `truncated` says when they differ.
"""
import pytest

from testing._gates import requires_sec


@requires_sec
@pytest.mark.parametrize("module,func,ticker,count_fields", [
    ("tools.web_search_server.sec_utils", "get_schedule_13d_filings", "INTC",
     ("count", "activist_count", "passive_count")),
    ("tools.web_search_server.peers", "find_peers_by_sic", "INTC",
     ("peer_count", "unresolved_count")),
])
def test_counts_do_not_move_with_the_page_size(module, func, ticker, count_fields):
    import importlib
    fn = getattr(importlib.import_module(module), func)

    small = fn(ticker, limit=3)
    large = fn(ticker, limit=100)

    for field in count_fields:
        if field not in small or field not in large:
            continue
        assert small[field] == large[field], (
            f"{func}.{field} is {small[field]} at limit=3 and {large[field]} "
            f"at limit=100. It is named for the whole set and describes the "
            f"page.")


@requires_sec
def test_the_page_is_reported_separately_from_the_set():
    from tools.web_search_server.sec_utils import get_schedule_13d_filings
    result = get_schedule_13d_filings("INTC", limit=3)

    assert result.get("rows_returned") == len(result["filings"]) <= 3
    assert result["count"] >= result["rows_returned"]
    if result["count"] > result["rows_returned"]:
        assert result.get("truncated") is True, (
            "more filings matched than were returned and nothing says so")


@requires_sec
def test_the_answer_does_not_depend_on_the_page_size():
    """The question this tool exists to answer must not depend on paging.

    This test used to assert `activist_count > 0` for INTC, on the premise
    that Intel had 31 activist 13D filings. It does not. Those rows were
    Intel FILING on MariaDB, Mobileye, Joby and Vuzix -- see
    test_13d_subject_not_filer.py. The true figure is zero activists and 40
    passive holders, which is unremarkable for a mega-cap.

    Asserting a non-zero count made a wrong number look like proof the paging
    fix worked. The invariant that actually matters is that the number does
    not move, so that is what is checked -- whatever the number is.
    """
    from tools.web_search_server.sec_utils import get_schedule_13d_filings
    small = get_schedule_13d_filings("INTC", limit=3)
    large = get_schedule_13d_filings("INTC", limit=100)

    assert small["activist_count"] == large["activist_count"]
    assert small["passive_count"] == large["passive_count"]
    assert small["count"] == large["count"]
    assert small["passive_count"] > 0, (
        "Vanguard and BlackRock file 13Gs on Intel; zero would mean the "
        "subject-side filter is now too aggressive")


@requires_sec
def test_an_untruncated_result_is_not_flagged():
    from tools.web_search_server.sec_utils import get_schedule_13d_filings
    result = get_schedule_13d_filings("INTC", limit=500)
    assert result.get("truncated") is False
    assert result["count"] == result.get("rows_returned")


def test_a_throttled_form_query_is_not_reported_as_zero_activists(monkeypatch):
    """A 429 from SEC once became `activist_count: 0` -- "Intel has no activist
    investors". The form most likely to be missing is the one whose absence
    inverts the answer, so a failed listing must refuse, not under-count."""
    import tools.web_search_server.sec_utils as su

    class _Throttled:
        cik = 50863

        def get_filings(self, form=None):
            if form == "SC 13D":
                raise RuntimeError("Too Many Requests")
            return []

    monkeypatch.setattr(su, "_require_identity", lambda: "test@example.com")
    monkeypatch.setattr(su, "Company", lambda t: _Throttled())

    result = su.get_schedule_13d_filings("INTC")

    assert result["success"] is False
    assert "SC 13D" in result["error"]
    assert result.get("activist_count") is None, (
        "a throttled query must not report an activist count at all")
