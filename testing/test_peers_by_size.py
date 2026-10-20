"""find_peers_by_sic ranks the whole SIC group by size before `limit` applies.

EDGAR lists an SIC group in name order, 100 filers to a page. The tool read one
page and took `limit` from its front, so the peer set was the start of the
alphabet. For NVDA (SIC 3674) limit 60 returned 24 peers, nearly all starting
with A, and of AVGO, INTC, MU and TXN only AVGO was present. Asked for NVDA's
three largest peers, an agent picked AMD, Amkor and Allegro and built its
EV/EBITDA medians from them.

The tool now reads every page and orders the peers by total assets before the
caller's limit applies. Both SEC endpoints are faked here: the browse list by
its `start` offset, the XBRL frames by the period in the URL.
"""
from datetime import date

import pytest
import requests

from tools.web_search_server import peers

_UA = {"User-Agent": "Test Analyst test@example.com"}
_NVDA = 1045810

# Name order across two pages, as EDGAR serves it. The three largest filers
# sort last, so a tool that reads one page, or does not sort, misses them.
_PAGE_1 = [11, 12, 13, 14, _NVDA, 99]       # ALPHA BRAVO CHARLIE DELTA, self, unlisted
_PAGE_2 = [21, 22, 23, 24]                  # MIKE XRAY YANKEE ZULU
_TICKERS = {
    11: ("ALFA", "ALPHA SEMI INC"),
    12: ("BRVO", "BRAVO DEVICES CORP"),
    13: ("CHRL", "CHARLIE MICRO INC"),
    14: ("DLTA", "DELTA CHIPS INC"),
    21: ("MIKE", "MIKE SILICON INC"),
    22: ("XRAY", "XRAY SEMICONDUCTOR CORP"),
    23: ("YANK", "YANKEE LOGIC INC"),
    24: ("ZULU", "ZULU WAFER CORP"),
    _NVDA: ("NVDA", "NVIDIA CORP"),
}
# DELTA and MIKE report no assets, so they rank nowhere and must still appear.
_ASSETS = {
    "CY2025Q2I": [(11, 1_000_000_000, "2025-06-30"), (12, 2_000_000_000, "2025-06-30"),
                  (22, 300_000_000_000, "2025-06-27"), (_NVDA, 4_000_000_000_000, "2025-07-25")],
    "CY2025Q1I": [(13, 3_000_000_000, "2025-03-31"), (23, 200_000_000_000, "2025-03-28"),
                  (24, 100_000_000_000, "2025-05-02")],
}


def _feed(ciks, more):
    entries = "".join(
        f'<entry title="ARRAY(0x5648a8af6810)"><content><cik>{c:010d}</cik>'
        f'</content></entry>' for c in ciks)
    nxt = '<link href="next-page" rel="next" type="application/atom+xml" />' if more else ""
    return ('<?xml version="1.0" encoding="ISO-8859-1" ?><feed '
            'xmlns="http://www.w3.org/2005/Atom"><link href="this-page" rel="self" />'
            f'{nxt}{entries}</feed>')


class _Response:
    def __init__(self, status=200, text="", payload=None):
        self.status_code = status
        self.text = text
        self._payload = payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Error")

    def json(self):
        return self._payload


class _FakeSec:
    """The browse list and the assets frames, with a hook to break one."""

    def __init__(self, pages, assets):
        self.pages = pages
        self.assets = assets
        self.broken = {}          # ("page", start) or ("frame", period) -> response
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=None):
        assert headers == _UA, "every SEC request carries the identity header"
        if url == peers._BROWSE_URL:
            assert params["SIC"] == "3674"
            start = int(params["start"])
            self.calls.append(("page", start))
            if ("page", start) in self.broken:
                return self.broken[("page", start)]
            index = start // int(params["count"])
            return _Response(text=_feed(self.pages[index], index + 1 < len(self.pages)))
        period = url.rsplit("/", 1)[1].removesuffix(".json")
        assert url == peers._FRAMES_URL.format(period=period)
        self.calls.append(("frame", period))
        if ("frame", period) in self.broken:
            return self.broken[("frame", period)]
        if period not in self.assets:
            return _Response(404, "<Error><Code>NoSuchKey</Code></Error>")
        return _Response(payload={"data": [
            {"cik": c, "val": v, "end": e} for c, v, e in self.assets[period]]})


@pytest.fixture
def sec(monkeypatch):
    fake = _FakeSec([_PAGE_1, _PAGE_2], _ASSETS)
    paced = []
    monkeypatch.setattr(peers.requests, "get", fake.get)
    monkeypatch.setattr(peers, "_headers", lambda: dict(_UA))
    monkeypatch.setattr(peers, "_throttle", lambda: paced.append(len(fake.calls)))
    monkeypatch.setattr(peers, "_ticker_map", lambda: dict(_TICKERS))
    monkeypatch.setattr(peers, "_company_sic", lambda t: ("3674", "Semiconductors", _NVDA))
    monkeypatch.setattr(peers, "_size_periods", lambda today: ["CY2025Q2I", "CY2025Q1I"])
    fake.paced = paced
    return fake


def _tickers(result):
    return [p["ticker"] for p in result["peers"]]


# ------------------------------------------------------------------ the list

def test_every_page_of_the_sic_list_is_read(sec):
    assert peers._fetch_sic_ciks("3674") == _PAGE_1 + _PAGE_2
    assert sec.calls == [("page", 0), ("page", 100)]


def test_limit_three_returns_the_three_largest_not_the_first_names(sec):
    result = peers.find_peers_by_sic("NVDA", limit=3)
    assert result["success"] is True
    assert _tickers(result) == ["XRAY", "YANK", "ZULU"]
    assert [p["assets_usd"] for p in result["peers"]] == [
        300_000_000_000, 200_000_000_000, 100_000_000_000]
    assert result["peers"][2]["assets_as_of"] == "2025-05-02"


def test_counts_cover_the_whole_group_not_the_page(sec):
    result = peers.find_peers_by_sic("NVDA", limit=3)
    assert result["filers_matched"] == 10
    assert result["peer_count"] == 8
    assert result["unresolved_count"] == 1
    assert result["peers_returned"] == 3
    assert result["truncated"] is True


def test_a_peer_with_no_assets_is_kept_after_the_ranked_ones(sec):
    result = peers.find_peers_by_sic("NVDA", limit=20)
    assert _tickers(result) == ["XRAY", "YANK", "ZULU", "CHRL", "BRVO", "ALFA",
                                "DLTA", "MIKE"]
    tail = result["peers"][-2:]
    assert [p["assets_usd"] for p in tail] == [None, None]
    assert [p["assets_as_of"] for p in tail] == [None, None]
    assert result["unranked_count"] == 2
    assert "2 peers" in result["note"]


def test_the_response_names_the_order_and_the_size_measure(sec):
    result = peers.find_peers_by_sic("NVDA", limit=3)
    assert "total assets, largest first" in result["order"]
    measure = result["size_measure"]
    assert measure["concept"] == "us-gaap:Assets"
    assert measure["unit"] == "USD"
    assert measure["periods"] == ["CY2025Q2I", "CY2025Q1I"]


def test_every_sec_request_is_paced(sec):
    peers.find_peers_by_sic("NVDA", limit=3)
    # One pause before each request: two pages, then two frames.
    assert sec.paced == [0, 1, 2, 3]
    assert len(sec.calls) == 4


@pytest.mark.parametrize("response", [
    _Response(503, "Service Unavailable"),
    _Response(200, "<html>Request Rate Threshold Exceeded</html>"),
], ids=["http-error", "not-a-feed"])
def test_a_failed_page_fails_the_call_rather_than_shortening_the_list(sec, response):
    """The missing page holds the large filers. Ranking the first page alone
    would return the old alphabetical answer under a size-ordered label."""
    sec.broken[("page", 100)] = response
    result = peers.find_peers_by_sic("NVDA", limit=3)
    assert result["success"] is False
    assert result["peers"] == []
    assert "start=100" in result["error"]
    assert "3674" in result["error"]


def test_the_sic_list_is_bounded(monkeypatch, sec):
    """A list that never ends is refused at the bound, not read forever."""
    bound = peers._SIC_MAX_PAGES

    def endless(url, params=None, headers=None, timeout=None):
        start = int(params["start"])
        assert start < (bound + 1) * peers._SIC_PAGE_SIZE, "read past the bound"
        sec.calls.append(("page", start))
        return _Response(text=_feed([start + 1], more=True))

    monkeypatch.setattr(peers.requests, "get", endless)
    with pytest.raises(RuntimeError, match=f"{bound} pages"):
        peers._fetch_sic_ciks("3674")
    assert len(sec.calls) == bound


# ------------------------------------------------------------------ the size

@pytest.mark.parametrize("status", [404, 500])
def test_a_failed_size_lookup_fails_the_call(sec, status):
    """The filers in a lost frame would sink to the unranked tail and read as
    filers that report no assets."""
    sec.broken[("frame", "CY2025Q1I")] = _Response(status, "error")
    result = peers.find_peers_by_sic("NVDA", limit=3)
    assert result["success"] is False
    assert result["peers"] == []
    assert "CY2025Q1I" in result["error"]
    assert "us-gaap:Assets" in result["error"]


@pytest.mark.parametrize("periods", [
    ["CY2025Q2I", "CY2024Q2I"], ["CY2024Q2I", "CY2025Q2I"]])
def test_each_filers_latest_balance_sheet_is_kept(sec, periods):
    sec.assets = {
        "CY2025Q2I": [(11, 5_000, "2025-06-30")],
        "CY2024Q2I": [(11, 4_000, "2024-06-28"), (12, 7_000, "2024-06-28")],
    }
    assets = peers._total_assets(periods)
    assert assets == {11: (5_000, "2025-06-30"), 12: (7_000, "2024-06-28")}


@pytest.mark.parametrize("today,expected", [
    (date(2026, 10, 20), ["CY2026Q2I", "CY2026Q1I", "CY2025Q4I", "CY2025Q3I"]),
    (date(2027, 1, 5), ["CY2026Q3I", "CY2026Q2I", "CY2026Q1I", "CY2025Q4I"]),
    (date(2026, 4, 1), ["CY2025Q4I", "CY2025Q3I", "CY2025Q2I", "CY2025Q1I"]),
])
def test_size_periods_skip_the_open_and_the_just_closed_quarter(today, expected):
    """SEC answers 404 for a frame no fact is dated in yet. Skipping the quarter
    in progress and the one just closed means every frame read exists, so a
    404 on any of them is a real failure. Four quarters hold every filer's
    balance sheet at least once."""
    assert peers._size_periods(today) == expected
