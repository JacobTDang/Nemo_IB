"""A count must match its rows, and a sum must be a sum.

    get_capex_announcements(ticker="NVDA", days=90)
        -> announcement_count 15, 8 rows returned,
           total_announced_usd 1,738,000,000,000

Two separate defects in one response.

The count describes the set and the rows are a page of it, with no
`rows_returned` and no `truncated` flag to say which is which -- the same
shape `get_schedule_13d_filings` was repaired for in
test_counts_survive_paging.py. A caller adding up the rows gets a different
number from the one printed above them and has no way to know why.

The $1.738T is worse, because it is not a sum of anything real. The live run
carried NVIDIA's $105B Ohio data centre as two articles and the $10B NAVER
deal as two more, and every article's largest figure was added in. It is a
sum over news mentions presented as a sum over capital projects, and a news
corpus restating one announcement four times moves it by hundreds of
billions. Nothing in the payload said the aggregate was of headlines rather
than of projects.

A news search cannot reliably tell two projects apart, so the honest
aggregate is over distinct dollar figures, with the collapse visible: how
many articles carried each figure, and how many figures there were. A
reader can then see that six figures arrived across fourteen articles
instead of reading a total that quietly counted eight of them twice.

The total was later scoped further -- to figures the text attributes to the
company SPENDING, rather than every figure in the corpus -- because summing
distinct figures still summed customer deals and debt raises alongside plant.
That is test_capex_is_not_revenue.py. The rule this file holds is orthogonal
and still binding: however the figures are chosen, each one is counted once
however many outlets restated it.
"""
import pytest

import tools.altdata_server.server as alt


class _FakeDDGS:
    """The ddgs context manager, minus the network."""

    def __init__(self, results):
        self._results = results
        self.queries = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def news(self, query, max_results=10, timelimit=None):
        self.queries.append(query)
        # Every query returns the same corpus; the caller dedupes by title.
        return list(self._results)


def _install(monkeypatch, results):
    import sys
    import types
    module = types.ModuleType("ddgs")
    fake = _FakeDDGS(results)
    module.DDGS = lambda *a, **k: fake
    monkeypatch.setitem(sys.modules, "ddgs", module)
    return fake


# The live NVDA/90d corpus, trimmed to the figures that matter. The Ohio data
# centre and the NAVER deal each arrive twice under different headlines, which
# is exactly what a news search does and exactly what the old total added up.
_NVDA_CORPUS = [
    {"title": "Nvidia's $750 billion AI bet deepens fears of a circular tech bubble",
     "body": "Nvidia will invest.", "date": "", "url": "a1"},
    {"title": "Who wins most in NVIDIA's $500 billion private capital deal?",
     "body": "Analysis.", "date": "", "url": "a2"},
    {"title": "Nvidia considers $250 billion financing for OpenAI's Ohio data center",
     "body": "Nvidia will expand capacity.", "date": "", "url": "a3"},
    {"title": "NVIDIA Just Bet $105 Billion That OpenAI's Data Center Hunger Won't Slow",
     "body": "Nvidia will build the site.", "date": "", "url": "a4"},
    {"title": "Tech giant Nvidia to finance up to $105 billion for new data center in Ohio",
     "body": "Nvidia will construct the plant.", "date": "", "url": "a5"},
    {"title": "Naver and NVIDIA triple Korea's AI factory in $10 billion deal",
     "body": "Nvidia will expand the factory.", "date": "", "url": "a6"},
    {"title": "NAVER and NVIDIA Triple Korea's AI Factory in $10 Billion Deal",
     "body": "Nvidia will expand the factory.", "date": "", "url": "a7"},
    {"title": "NVIDIA's $3 billion bet on the power behind AI",
     "body": "Nvidia invests.", "date": "", "url": "a8"},
    {"title": "Nvidia commits $2 billion to a new campus",
     "body": "Nvidia will build.", "date": "", "url": "a9"},
    {"title": "Nvidia adds $1 billion to its supply agreement",
     "body": "Nvidia will expand.", "date": "", "url": "a10"},
]


@pytest.fixture
def nvda(monkeypatch):
    _install(monkeypatch, _NVDA_CORPUS)
    return alt._fetch_capex_announcements("NVDA", "NVIDIA Corporation", 90)


# ---------------------------------------------------------------------------
# The count and the page
# ---------------------------------------------------------------------------

def test_the_page_is_reported_separately_from_the_set(nvda):
    """`announcement_count: 15` above 8 rows is a count named for the set and
    printed beside the page, with nothing to tell them apart."""
    assert nvda["announcement_count"] == 10
    assert nvda["rows_returned"] == len(nvda["announcements"])
    assert nvda["rows_returned"] < nvda["announcement_count"]
    assert nvda["truncated"] is True


def test_an_untruncated_result_is_not_flagged(monkeypatch):
    _install(monkeypatch, _NVDA_CORPUS[:3])
    out = alt._fetch_capex_announcements("NVDA", "NVIDIA Corporation", 90)
    assert out["announcement_count"] == out["rows_returned"] == 3
    assert out["truncated"] is False


# ---------------------------------------------------------------------------
# The total
# ---------------------------------------------------------------------------

def test_a_figure_reported_twice_is_counted_once(nvda):
    """$105B Ohio appeared as two articles and $10B NAVER as two more. Summing
    per article turned six announcements into eight and every total into a
    number no filing supports. The figures are now sorted into categories and
    only some of them are summed, so the rule is checked across all of them:
    no category may count a figure twice."""
    per_article = 750 + 500 + 250 + 105 + 105 + 10 + 10 + 3 + 2 + 1
    distinct = 750 + 500 + 250 + 105 + 10 + 3 + 2 + 1

    everything = sum(slot["total_usd"]
                     for slot in nvda["amounts_by_category"].values())
    assert everything == distinct * 1e9
    assert everything != per_article * 1e9
    assert len(nvda["figures"]) == len({f["amount_usd"] for f in nvda["figures"]})


def test_the_collapse_is_visible(nvda):
    """A reader has to be able to see that the total is over figures, not
    articles -- otherwise the corrected number is just as unexplained as the
    wrong one."""
    assert nvda["figure_count"] == 8
    assert nvda["announcement_count"] == 10
    assert nvda["capex_total_basis"], (
        "the total is over distinct figures and the payload does not say so")
    assert "distinct" in nvda["capex_total_basis"]


def test_each_row_says_how_many_articles_carried_its_figure(nvda):
    """The duplication is a property of the corpus, so it belongs on the rows
    the corpus produced."""
    by_amount = {a["largest_figure_usd"]: a for a in nvda["announcements"]}
    assert by_amount[105e9]["mentions"] == 2
    assert by_amount[10e9]["mentions"] == 2
    assert by_amount[750e9]["mentions"] == 1


def test_the_largest_figure_is_reported_without_being_summed(nvda):
    """The biggest number in the corpus is still shown -- it is the headline a
    reader is going to ask about -- but it is shown as what it is. $750B leads
    the figure list and contributes nothing to the capex total, which rests on
    the single $2B campus."""
    assert nvda["figures"][0]["amount_usd"] == 750e9
    assert nvda["figures"][0]["category"] != "capital_expenditure"
    assert nvda["largest_capex_usd"] == 2e9


@pytest.fixture
def unpriced(monkeypatch):
    _install(monkeypatch, [
        {"title": "Nvidia opens a new campus", "body": "Nvidia will build.",
         "date": "", "url": "b1"},
        {"title": "Nvidia expands in Ohio", "body": "Nvidia will expand.",
         "date": "", "url": "b2"},
        {"title": "Nvidia commits $4 billion", "body": "Nvidia will invest.",
         "date": "", "url": "b3"},
    ])
    return alt._fetch_capex_announcements("NVDA", "NVIDIA Corporation", 90)


def test_articles_with_no_dollar_figure_do_not_inflate_the_count(unpriced):
    """A zero is not a distinct figure; counting it would make the basis line
    disagree with the arithmetic."""
    assert unpriced["announcement_count"] == 3
    assert unpriced["figure_count"] == 1
    assert unpriced["figures"][0]["amount_usd"] == 4e9


def test_an_article_with_no_figure_is_not_reported_as_a_restatement(unpriced):
    """A live INTC run put `mentions: 7` on every row carrying no dollar
    amount -- grouping them all under 0 and reading as "seven outlets restated
    this figure". There is no figure, so there is nothing to have restated,
    and a count invented for the occasion is the defect this file is about."""
    unpriced_rows = [a for a in unpriced["announcements"]
                     if a["largest_figure_usd"] is None]
    assert len(unpriced_rows) == 2
    for row in unpriced_rows:
        assert row["mentions"] is None, (
            "an article with no dollar figure was given a restatement count")


def test_a_lookup_that_found_nothing_still_answers_the_new_fields(monkeypatch):
    """The no-results branch is a separate return; it must not start raising
    KeyError on a caller that reads the fields the success branch promises."""
    _install(monkeypatch, [])
    out = alt._fetch_capex_announcements("KO", "Coca-Cola Company", 90)

    assert out["success"] is False
    assert out["reason"] == "no_results"
    assert out["rows_returned"] == 0
    assert out["truncated"] is False
