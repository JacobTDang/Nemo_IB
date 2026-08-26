"""A tool's name and its field names are claims. They have to be true.

Every response in this file was internally consistent. The arithmetic was
right, the coverage labels were right, nothing was swallowed. What was wrong
was the LABEL -- the tool name, or a field name -- so a caller reading it
correctly still ended up with an answer to a different question than the one
they asked, and no way to notice.

Four found live on 2026-08-26:

    get_analyst_revisions_history
        Returns strongBuy/buy/hold/sell counts from Finnhub
        /stock/recommendation. Rating buckets. Not EPS revisions, not revenue
        revisions, not a price target. A pre-earnings workflow asking "are
        analysts taking estimates up into the print?" got a recommendation
        distribution and could not tell. It read `upgrading` on AMAT, which
        then gapped -6.57% and drifted another -5.31%.

        Verified against the live key the same day: /stock/eps-estimate,
        /stock/revenue-estimate, /stock/upgrade-downgrade and
        /stock/price-target all answer 403 "You don't have access to this
        resource." There is no estimate-revision feed on this plan, so the
        name cannot be made true and has to change instead.

    extract_call_sentiment / get_earnings_transcripts
        Neither reads a call. Both read the 8-K Item 2.02 press release
        (EX-99.1), which is prepared prose with no analyst Q&A -- and both
        reported `provider: "Finnhub"` while serving SEC EDGAR documents from
        the sec server.

    get_policy_signals
        `keywords_searched` listed four keywords for LMT; three were queried.
        The dropped one was "NDAA" -- the single most important bill for a
        defense prime. LMT also auto-detected as sector "Industrials", because
        that is what yfinance reports; the "Defense" keyword set the module
        already contains is unreachable from any GICS sector.
        And `bill_count: 21` shipped beside ten bills with no truncation flag,
        on the same server where get_congress_trades says "The row list is
        truncated".

    get_supply_chain
        `related_companies` put TSM (a foundry NVDA buys wafers from) and AMD
        (a competitor) in one flat list with identical schema and no direction
        field. The distinction existed only in `sample_context` prose, so a
        read-through workflow chaining this tool had nothing machine-readable
        to chain. It also matched `facebook -> META` out of the social-media
        link footer at the end of Item 1.

The rule: a name that cannot be made true is changed, not caveated. A field
that measures something narrower than its name is renamed to what it measures
or given the span/basis alongside it, and a set that was truncated says so.

Run:
  SKIP_NETWORK_TESTS=1 .venv/bin/python -m pytest \
      testing/test_the_label_names_what_it_measures.py -q
"""
from __future__ import annotations

import asyncio
import json

import pytest
import requests

from testing._gates import requires_sec


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _tool_map(instance) -> dict:
    """{name: Tool} the server would answer `tools/list` with right now."""
    async def go():
        for key, handler in instance.server.request_handlers.items():
            if "ListTools" in str(key):
                return {t.name: t for t in (await handler(None)).root.tools}
        return {}
    return asyncio.run(go())


def _annotation_config(instance) -> dict:
    """The provider/warning config the dispatcher was wrapped with.

    Read through tools/manifest.py rather than reimplemented: that is the
    reader the shipped manifest uses, so a provider this test approves is the
    provider a caller is actually told about.
    """
    from tools.manifest import _annotation_config as read

    return read(instance)


# ===========================================================================
# 1. The Finnhub tool is named for estimate revisions and measures ratings
# ===========================================================================

def _finnhub_server(payload):
    from tools.news_agregator.finnhub_server import FinnhubServer

    server = FinnhubServer.__new__(FinnhubServer)

    class _Client:
        async def get(self, endpoint, params=None):
            return payload

    server.client = _Client()
    return server


def _month(period, sb, b, h, s=0, ss=0):
    return {"period": period, "strongBuy": sb, "buy": b, "hold": h,
            "sell": s, "strongSell": ss}


# AMAT as Finnhub served it on 2026-08-26: four monthly snapshots, coverage
# growing from 43 firms to 45. Two of the three net-bullish points came from
# new initiations, not from anybody upgrading.
_AMAT_LIVE = [
    _month("2026-08-01", 12, 27, 6),
    _month("2026-07-01", 12, 27, 6),
    _month("2026-06-01", 12, 27, 7),
    _month("2026-05-01", 11, 25, 7),
]

# The case that made "downgrading" out of noise: 65 net bullish out of 68
# analysts falling to 63 out of 68. Two firms out of sixty-eight.
_WIDE_COVERAGE_DRIFT = [
    _month("2026-08-01", 30, 33, 5),   # net 63 of 68
    _month("2026-07-01", 31, 33, 4),
    _month("2026-06-01", 31, 34, 3),
    _month("2026-05-01", 32, 33, 3),   # net 65 of 68
]

# A real re-rating: nine of thirty analysts move off buy onto hold.
_REAL_DOWNGRADE = [
    _month("2026-08-01", 5, 10, 15),   # net 15 of 30
    _month("2026-07-01", 7, 12, 11),
    _month("2026-06-01", 9, 14, 7),
    _month("2026-05-01", 10, 14, 6),   # net 24 of 30
]


def _rating_trend(payload, **kwargs):
    server = _finnhub_server(payload)
    contents = asyncio.run(server.get_analyst_rating_trend("TEST", **kwargs))
    return json.loads(contents[0].text)


def test_the_finnhub_tool_is_not_named_for_a_feed_this_plan_cannot_see():
    """`get_analyst_revisions_history` is the name of a tool that reads an
    estimate-revision feed. Finnhub answers 403 for every one of those
    endpoints on this key, so the tool cannot ever become that tool. The
    honest move is to name it for the rating buckets it does return."""
    from tools.news_agregator.finnhub_server import FinnhubServer

    names = set(_tool_map(FinnhubServer()))
    assert "get_analyst_rating_trend" in names, sorted(names)
    assert "get_analyst_revisions_history" not in names, (
        "the tool is still advertised under a name that promises estimate "
        "revisions it does not carry")


def test_the_description_turns_away_the_question_it_cannot_answer():
    """Renaming the tool is not enough on its own: an agent shopping the tool
    list for revision data has to be told, in the description, that this is
    not it -- otherwise it picks the nearest-looking thing and proceeds."""
    from tools.news_agregator.finnhub_server import FinnhubServer

    tool = _tool_map(FinnhubServer())["get_analyst_rating_trend"]
    text = tool.description.lower()
    assert "rating" in text
    assert "estimate revision" not in text, (
        f"the description still offers estimate revisions: {tool.description}")
    assert "no eps or revenue" in text, (
        f"the description does not say what is absent: {tool.description}")


def test_the_payload_says_what_it_measures():
    """A caller holding the JSON, with no tool name in hand, still has to be
    able to tell recommendation counts from estimate revisions."""
    data = _rating_trend(_AMAT_LIVE)["data"]
    assert data["measures"] == "analyst_rating_buckets", data.get("measures")


def test_a_twelve_month_request_answered_with_four_months_says_so():
    """lookback_months=12 returned periods_returned=4 and nothing else. The
    request and the coverage were both in the response and a reader had to
    notice the gap themselves -- which is exactly what nobody does."""
    envelope = _rating_trend(_AMAT_LIVE, lookback_months=12)
    data = envelope["data"]

    assert data["lookback_months_requested"] == 12
    assert data["periods_returned"] == 4
    codes = {w["code"] for w in (envelope.get("warnings") or [])}
    assert "history_shorter_than_requested" in codes, (
        f"twelve months asked, four returned, no warning: {envelope.get('warnings')}")


def test_a_delta_the_description_promises_but_cannot_compute_is_named():
    """The description advertised 1mo/3mo/6mo deltas. With four monthly
    snapshots the 6mo delta cannot exist, and it was simply absent from
    `momentum` -- indistinguishable from a delta of zero to anything reading
    the dict with .get()."""
    data = _rating_trend(_AMAT_LIVE)["data"]

    assert "6mo_net_bullish_delta" not in data["momentum"]
    unavailable = data["momentum_unavailable"]
    assert "6mo_net_bullish_delta" in unavailable
    assert "4" in unavailable["6mo_net_bullish_delta"], (
        "the reason must name how much history there actually was: "
        f"{unavailable['6mo_net_bullish_delta']!r}")


def test_two_analysts_out_of_sixty_eight_is_not_a_downgrade_trend():
    """The classifier read the raw count delta, so the same two-analyst move
    was "downgrading" whether the firm was covered by 6 analysts or 68. A
    signal has to be measured against the coverage it moved within."""
    data = _rating_trend(_WIDE_COVERAGE_DRIFT)["data"]

    assert data["momentum"]["3mo_net_bullish_delta"] == -2
    assert data["signal"] == "neutral", (
        f"-2 net bullish out of 68 analysts classified {data['signal']!r}")


def test_new_coverage_is_not_read_as_an_upgrade():
    """AMAT's +3 net bullish over three months came with coverage rising from
    43 firms to 45: two of the three points are initiations, not upgrades.
    Reported as "upgrading" the day before a -6.57% gap."""
    data = _rating_trend(_AMAT_LIVE)["data"]

    assert data["momentum"]["3mo_total_analysts_delta"] == 2, (
        "the coverage change has to be visible next to the net-bullish change")
    assert data["signal"] == "neutral", (
        f"+3 net bullish on +2 new analysts classified {data['signal']!r}")


def test_a_real_re_rating_still_registers():
    """Coverage-normalising must not flatten everything to neutral: nine of
    thirty analysts moving off buy is a re-rating and has to read as one."""
    data = _rating_trend(_REAL_DOWNGRADE)["data"]
    assert data["signal"] in ("downgrading", "downgrading_strong"), data["signal"]


def test_the_signal_states_the_basis_it_was_computed_from():
    data = _rating_trend(_REAL_DOWNGRADE)["data"]
    assert data["signal_basis"], "a classifier with no stated basis"
    assert "share" in data["signal_basis"].lower()


# ===========================================================================
# 2. The sentiment tool is named for calls and reads press releases
# ===========================================================================

def _release(filing_date, text):
    return {"filing_date": filing_date, "accession_number": "x",
            "items": ["2.02"], "attachment_doc": "ex99-1.htm",
            "text": text, "text_length_chars": len(text),
            "text_truncated": False, "filing_url": None}


_CONFIDENT_PROSE = ("We delivered a record quarter with strong momentum and "
                    "robust growth across every segment. " * 40)
_HEDGED_PROSE = ("We face uncertainty and softness, with headwinds and a "
                 "challenging environment weighing on demand. " * 40)


def _patch_releases(monkeypatch, releases):
    import tools.web_search_server.sec_utils as su

    def fake(ticker, max_quarters=4, max_chars_per_release=50000):
        selected = releases[:max_quarters]
        return {"ticker": ticker, "success": True, "error": None,
                "source": "8-K Item 2.02", "releases": selected,
                "quarters_requested": max_quarters,
                "release_count": len(selected)}

    monkeypatch.setattr(su, "get_earnings_releases", fake)
    return su


def test_neither_sec_tool_is_named_for_a_call_transcript():
    """Both read 8-K Item 2.02 press releases. `get_earnings_transcripts`
    produced pairs_found=0 on CHWY because the "transcripts" were releases
    (docs/known_issues.md item 9), and a caveat did not stop that."""
    from tools.web_search_server.web_search import WebSearchServer

    names = set(_tool_map(WebSearchServer()))
    assert "get_earnings_releases" in names, sorted(names)
    assert "extract_earnings_release_sentiment" in names, sorted(names)
    assert "get_earnings_transcripts" not in names
    assert "extract_call_sentiment" not in names


def test_the_sec_server_does_not_claim_finnhub_served_the_document():
    """Both tools were pinned to `provider: "Finnhub"` in the sec server's
    own annotating() map while serving SEC EDGAR exhibits. A caller auditing
    provenance would go looking at the wrong vendor."""
    from tools.web_search_server.web_search import WebSearchServer

    per_tool = _annotation_config(WebSearchServer())["per_tool"]
    for tool in ("get_earnings_releases", "extract_earnings_release_sentiment"):
        provider = per_tool.get(tool, "SEC EDGAR")
        assert "finnhub" not in provider.lower(), f"{tool}: {provider}"
        assert "sec" in provider.lower(), f"{tool}: {provider}"


def test_a_one_quarter_gap_is_not_reported_as_year_over_year(monkeypatch):
    """WMT live: `yoy_shift.compared_periods` read "2026-08-20 vs
    2026-05-21". One quarter, labelled year-over-year, with a classifier
    tuned for an annual span sitting on top of it."""
    su = _patch_releases(monkeypatch, [
        _release("2026-08-20", _HEDGED_PROSE),
        _release("2026-05-21", _CONFIDENT_PROSE),
    ])

    out = su.extract_earnings_release_sentiment("WMT", 4)

    assert "yoy_shift" not in out, (
        "a one-quarter comparison is still carried in a field named yoy")
    shift = out["tone_shift"]
    assert shift["span_label"] == "quarter_over_quarter", shift
    assert 80 <= shift["span_days"] <= 100, shift


def test_a_genuine_year_apart_comparison_is_labelled_as_one(monkeypatch):
    su = _patch_releases(monkeypatch, [
        _release("2026-08-20", _HEDGED_PROSE),
        _release("2026-05-21", _CONFIDENT_PROSE),
        _release("2026-02-20", _CONFIDENT_PROSE),
        _release("2025-08-21", _CONFIDENT_PROSE),
    ])

    out = su.extract_earnings_release_sentiment("WMT", 4)

    assert out["tone_shift"]["span_label"] == "year_over_year", out["tone_shift"]


def test_asking_for_four_quarters_and_scoring_two_is_not_silent(monkeypatch):
    """WMT scored 2 of the 4 requested and said `quarters_scored: 2` with no
    statement of what was asked for. Tone over two quarters is a different
    statistic from tone over four."""
    su = _patch_releases(monkeypatch, [
        _release("2026-08-20", _HEDGED_PROSE),
        _release("2026-05-21", _CONFIDENT_PROSE),
    ])

    out = su.extract_earnings_release_sentiment("WMT", 4)

    assert out["quarters_requested"] == 4
    assert out["quarters_scored"] == 2
    codes = {w["code"] for w in (out.get("warnings") or [])}
    assert "fewer_quarters_than_requested" in codes, out.get("warnings")


def test_a_fully_answered_request_carries_no_shortfall_warning(monkeypatch):
    su = _patch_releases(monkeypatch, [
        _release("2026-08-20", _HEDGED_PROSE),
        _release("2026-05-21", _CONFIDENT_PROSE),
    ])

    out = su.extract_earnings_release_sentiment("WMT", 2)

    codes = {w["code"] for w in (out.get("warnings") or [])}
    assert "fewer_quarters_than_requested" not in codes, out.get("warnings")


def test_the_signal_names_the_span_it_was_measured_over(monkeypatch):
    su = _patch_releases(monkeypatch, [
        _release("2026-08-20", _HEDGED_PROSE),
        _release("2026-05-21", _CONFIDENT_PROSE),
    ])

    out = su.extract_earnings_release_sentiment("WMT", 4)

    assert "quarter" in out["signal_basis"].lower(), out["signal_basis"]


# ===========================================================================
# 3. get_policy_signals: the sector, the keywords and the count
# ===========================================================================

class _Resp:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"{self.status_code} error")

    def json(self):
        return self._payload


class _FakeYF:
    """yfinance double returning one fixed `info` dict."""

    def __init__(self, info):
        self._info = info

    def Ticker(self, symbol):
        info = self._info

        class _T:
            def __init__(self):
                self.info = info

        return _T()


def _fake_yfinance(monkeypatch, info):
    import sys
    monkeypatch.setitem(sys.modules, "yfinance", _FakeYF(info))


class _KeywordGet:
    """GovTrack double that answers each keyword with its own bill and
    records which keywords were actually queried."""

    def __init__(self, per_keyword=None, bills_per_keyword=1):
        self.queried = []
        self.per_keyword = per_keyword or {}
        self.bills_per_keyword = bills_per_keyword

    def __call__(self, url, params=None, **kwargs):
        keyword = (params or {}).get("q", "")
        self.queried.append(keyword)
        objects = self.per_keyword.get(keyword)
        if objects is None:
            objects = [{
                "title": f"{keyword.title()} Funding Act of 2026 no.{i}",
                "short_title": "",
                "current_status": "introduced",
                "introduced_date": "2026-08-01",
                "current_status_date": "2026-08-01",
                "link": f"https://govtrack.us/{keyword}/{i}",
            } for i in range(self.bills_per_keyword)]
        return _Resp({"objects": objects})


_LMT_INFO = {"longName": "Lockheed Martin Corporation",
             "sector": "Industrials",
             "industry": "Aerospace & Defense"}


def test_a_defense_prime_is_not_researched_as_generic_industrials(monkeypatch):
    """yfinance reports LMT as sector "Industrials" -- it has no "Defense"
    sector, and never will. The module already carries a Defense keyword set
    led by NDAA; nothing could reach it. LMT got "infrastructure",
    "reshoring" and "defense procurement" instead, and came back with the
    Restoring the Death Penalty in DC Act."""
    from tools.altdata_server import server as alt

    monkeypatch.delenv("CONGRESS_API_KEY", raising=False)
    _fake_yfinance(monkeypatch, _LMT_INFO)
    get = _KeywordGet()
    monkeypatch.setattr(requests, "get", get)

    out = alt._fetch_policy_signals("LMT", "", 180)

    assert out["sector"] == "Defense", out.get("sector")
    assert out["sector_reported_by_provider"] == "Industrials", out
    assert out["sector_source"], "nothing says why the sector was overridden"
    assert "NDAA" in out["keywords_searched"], out["keywords_searched"]


def test_an_explicit_sector_from_the_caller_is_not_overridden(monkeypatch):
    """The override exists to fill a gap in the provider's taxonomy, not to
    second-guess a caller who named the sector themselves."""
    from tools.altdata_server import server as alt

    monkeypatch.delenv("CONGRESS_API_KEY", raising=False)
    monkeypatch.setattr(requests, "get", _KeywordGet())

    out = alt._fetch_policy_signals("LMT", "Industrials", 180)

    assert out["sector"] == "Industrials"


def test_every_keyword_reported_as_searched_was_actually_searched(monkeypatch):
    """`keywords_searched` echoed the whole mapping while the fetchers used
    `keywords[:3]`. For Industrials the fourth entry is "NDAA": the response
    said it had looked for the most important defense bill of the year and it
    had not."""
    from tools.altdata_server import server as alt

    monkeypatch.delenv("CONGRESS_API_KEY", raising=False)
    _fake_yfinance(monkeypatch, _LMT_INFO)
    get = _KeywordGet()
    monkeypatch.setattr(requests, "get", get)

    out = alt._fetch_policy_signals("LMT", "", 180)

    reported = set(out["keywords_searched"])
    actually = set(get.queried)
    assert reported <= actually, (
        f"reported as searched but never queried: {sorted(reported - actually)}")


def test_the_bill_count_describes_the_set_and_the_page_says_so(monkeypatch):
    """`bill_count: 21` shipped with ten bills and no flag. get_congress_trades
    on this same server states "The row list is truncated"; this one did not,
    and the score in `total_score` is summed over rows the caller cannot see."""
    from tools.altdata_server import server as alt

    monkeypatch.delenv("CONGRESS_API_KEY", raising=False)
    monkeypatch.setattr(requests, "get", _KeywordGet(bills_per_keyword=8))

    out = alt._fetch_policy_signals("NVDA", "Technology", 180)

    assert out["bill_count"] > len(out["bills"]), (
        "fixture did not produce a truncated set")
    assert out["rows_returned"] == len(out["bills"])
    assert out["truncated"] is True, out


def test_an_untruncated_bill_set_is_not_flagged(monkeypatch):
    from tools.altdata_server import server as alt

    monkeypatch.delenv("CONGRESS_API_KEY", raising=False)
    monkeypatch.setattr(requests, "get", _KeywordGet(bills_per_keyword=1))

    out = alt._fetch_policy_signals("NVDA", "Technology", 180)

    assert out["truncated"] is False
    assert out["rows_returned"] == out["bill_count"] == len(out["bills"])


def test_each_bill_names_the_keyword_that_found_it(monkeypatch):
    """The verdict is an average over a set that includes bills matched on
    full text rather than subject. Naming the keyword per row is what lets a
    reader see that the Native American Housing Assistance Modernization Act
    arrived on "infrastructure" and discount it."""
    from tools.altdata_server import server as alt

    monkeypatch.delenv("CONGRESS_API_KEY", raising=False)
    monkeypatch.setattr(requests, "get", _KeywordGet())

    out = alt._fetch_policy_signals("NVDA", "Technology", 180)

    assert out["bills"], out
    for bill in out["bills"]:
        assert bill.get("matched_keyword"), bill


def test_the_signal_basis_names_how_many_bills_it_averaged(monkeypatch):
    from tools.altdata_server import server as alt

    monkeypatch.delenv("CONGRESS_API_KEY", raising=False)
    monkeypatch.setattr(requests, "get", _KeywordGet(bills_per_keyword=8))

    out = alt._fetch_policy_signals("NVDA", "Technology", 180)

    assert str(out["bill_count"]) in out["signal_basis"], out["signal_basis"]


# ===========================================================================
# 4. get_supply_chain cannot tell a supplier from a competitor
# ===========================================================================

_ITEM1_BODY = """
We utilize foundries, such as Taiwan Semiconductor Manufacturing Company
Limited, or TSMC, and Samsung Electronics Co., Ltd., or Samsung, to produce
our semiconductor wafers. We purchase memory from SK Hynix Inc., Micron
Technology, Inc., and Samsung.
Our competitors include suppliers of discrete and integrated computing
solutions offered for AI, such as Advanced Micro Devices, Inc., or AMD,
Huawei Technologies Co. Ltd., or Huawei, and Intel Corporation, or Intel.
Our customers include Microsoft Corporation and Amazon, Inc., which resell
our products inside their cloud offerings.
Follow us: NVIDIA LinkedIn (linkedin. com/company/nvidia) NVIDIA Facebook
(facebook. com/nvidia) NVIDIA Instagram (instagram. com/nvidia)
"""


class _FakeFiling:
    def __init__(self, text):
        self._text = text

    def text(self):
        return self._text


def _patch_item1(monkeypatch, body):
    """Serve `body` as 10-K Item 1 through get_latest_filing.

    get_supply_chain slices Item 1 out of the raw filing text: it skips the
    table of contents by starting at offset 7500 and stops at the Item 1A
    header, which it only looks for past offset 30000. The padding here
    reproduces that layout so the extractor sees the body it would see in a
    real filing.
    """
    import tools.web_search_server.sec_utils as su

    head = "TABLE OF CONTENTS " * 500          # >= 7500 chars
    head = head[:7500]
    filler = "\nGeneral corporate background paragraph. " * 800
    text = head + body + filler
    text = text[:30500] if len(text) > 30500 else text + " " * (30500 - len(text))
    text += "\nITEM 1A. RISK FACTORS\nRisks follow here.\n"

    monkeypatch.setattr(su, "get_latest_filing",
                        lambda ticker, form_type='10-K': {
                            "filing_object": _FakeFiling(text),
                            "filing_date": "2026-02-25"})
    return su


def _by_ticker(result):
    return {r["ticker"]: r for r in result["related_companies"]}


def test_a_supplier_and_a_competitor_are_told_apart(monkeypatch):
    """TSM and AMD arrived in one flat list with the same schema. A
    read-through workflow reading `related_companies` had no way to know that
    NVDA buying wafers from one and losing share to the other point opposite
    directions."""
    su = _patch_item1(monkeypatch, _ITEM1_BODY)

    rows = _by_ticker(su.get_supply_chain("NVDA"))

    assert rows["TSM"]["relationship"] == "supplier", rows["TSM"]
    assert rows["AMD"]["relationship"] == "competitor", rows["AMD"]
    assert rows["MU"]["relationship"] == "supplier", rows["MU"]


def test_the_direction_states_the_phrase_that_decided_it(monkeypatch):
    """A classification with no evidence beside it is another unauditable
    verdict. The cue that decided the direction rides along."""
    su = _patch_item1(monkeypatch, _ITEM1_BODY)

    rows = _by_ticker(su.get_supply_chain("NVDA"))

    assert rows["TSM"]["relationship_basis"], rows["TSM"]
    assert rows["AMD"]["relationship_basis"], rows["AMD"]


def test_the_read_through_handoff_is_machine_readable(monkeypatch):
    """The point of this tool inside /cross-company-readthrough is to hand
    the next step a list of who is upstream and who is a rival. Prose in
    `sample_context` is not a handoff."""
    su = _patch_item1(monkeypatch, _ITEM1_BODY)

    out = su.get_supply_chain("NVDA")

    assert "TSM" in out["suppliers"], out["suppliers"]
    assert "AMD" in out["competitors"], out["competitors"]
    assert "TSM" not in out["competitors"]
    assert "AMD" not in out["suppliers"]


def test_a_social_media_link_is_not_a_business_relationship(monkeypatch):
    """NVDA's Item 1 ends with a follow-us block. "NVIDIA Facebook
    (facebook. com/nvidia)" matched the curated name table and META was
    reported as a related company with mention_count 2."""
    su = _patch_item1(monkeypatch, _ITEM1_BODY)

    out = su.get_supply_chain("NVDA")

    assert "META" not in _by_ticker(out), (
        "a link in the follow-us footer is still being read as a "
        "supply-chain relationship")


def test_a_dropped_mention_is_reported_rather_than_vanishing(monkeypatch):
    """Silently dropping a match trades one invisible error for another. The
    exclusion is counted and the reason named."""
    su = _patch_item1(monkeypatch, _ITEM1_BODY)

    out = su.get_supply_chain("NVDA")

    excluded = out["excluded_mentions"]
    assert any(e["ticker"] == "META" for e in excluded), excluded
    assert all(e.get("reason") for e in excluded), excluded


def test_a_named_customer_is_not_filed_as_a_competitor(monkeypatch):
    su = _patch_item1(monkeypatch, _ITEM1_BODY)

    out = su.get_supply_chain("NVDA")

    assert "MSFT" in out["customers"], out["customers"]
    assert "MSFT" not in out["competitors"]


@requires_sec
def test_live_nvda_separates_its_foundry_from_its_rivals():
    """The fixture above is a reconstruction. This is the real filing."""
    from tools.web_search_server.sec_utils import get_supply_chain

    out = get_supply_chain("NVDA")
    if not out.get("success"):
        pytest.skip(f"NVDA 10-K unavailable: {out.get('error')}")

    rows = {r["ticker"]: r for r in out["related_companies"]}
    assert rows.get("TSM", {}).get("relationship") == "supplier", rows.get("TSM")
    assert rows.get("AMD", {}).get("relationship") == "competitor", rows.get("AMD")
    assert "META" not in rows, "the social-media footer is still matching"
