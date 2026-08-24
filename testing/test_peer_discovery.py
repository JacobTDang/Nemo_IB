"""Peer discovery by SIC code.

`comparable_company_analysis` required the analyst to supply the peer set,
which means the comps depend on already knowing the answer. SEC returns the SIC
code for every filer, so the peer set can come from the filings.

Coverage is partial by nature: an SIC query returns every filer in the class,
including deregistered and private ones that have no listed ticker. Measured
live for SIC 3674, 13 of 40 resolved. The tool reports the shortfall rather
than presenting 13 as though it were the universe.
"""
import os

import pytest

from tools.web_search_server import peers

SKIP_NETWORK = os.environ.get("SKIP_NETWORK_TESTS") == "1"


def network(func):
  """Apply the real `network` marker plus the offline skip.

  This name used to be bound to a bare pytest.mark.skipif. A skipif is not
  a registered marker, so `-m network` and `-m "not network"` collected
  nothing here -- the tests were selectable only by file path.
  """
  func = pytest.mark.network(func)
  return pytest.mark.skipif(SKIP_NETWORK, reason="live SEC test")(func)


def test_ciks_are_parsed_from_the_atom_feed():
    xml = "<x><cik>0000002488</cik><cik>1045810</cik></x>"
    assert peers._parse_ciks(xml) == [2488, 1045810]


def test_leaked_array_refs_do_not_become_company_names():
    """SEC's backend emits title="ARRAY(0x5648a8af6810)" instead of the company
    name. Names must come from the ticker map, never from the feed."""
    xml = '<entry title="ARRAY(0x5648a8af6810)"><cik>2488</cik></entry>'
    ciks = peers._parse_ciks(xml)
    assert ciks == [2488]


def test_unresolvable_ciks_are_counted_not_dropped_silently(monkeypatch):
    monkeypatch.setattr(peers, "_fetch_sic_ciks", lambda sic, limit: [1, 2, 3, 2488])
    monkeypatch.setattr(peers, "_ticker_map",
                        lambda: {2488: ("AMD", "ADVANCED MICRO DEVICES INC")})
    monkeypatch.setattr(peers, "_company_sic", lambda t: ("3674", "Semiconductors", 1045810))
    result = peers.find_peers_by_sic("NVDA")
    assert [p["ticker"] for p in result["peers"]] == ["AMD"]
    assert result["unresolved_count"] == 3
    assert result["filers_matched"] == 4


def test_the_query_company_is_not_listed_as_its_own_peer(monkeypatch):
    monkeypatch.setattr(peers, "_fetch_sic_ciks", lambda sic, limit: [1045810, 2488])
    monkeypatch.setattr(peers, "_ticker_map", lambda: {
        1045810: ("NVDA", "NVIDIA CORP"), 2488: ("AMD", "ADVANCED MICRO DEVICES INC")})
    monkeypatch.setattr(peers, "_company_sic", lambda t: ("3674", "Semiconductors", 1045810))
    result = peers.find_peers_by_sic("NVDA")
    assert [p["ticker"] for p in result["peers"]] == ["AMD"]


def test_missing_sic_is_an_explicit_failure(monkeypatch):
    monkeypatch.setattr(peers, "_company_sic", lambda t: (None, None, None))
    result = peers.find_peers_by_sic("NOSIC")
    assert result["success"] is False
    assert result["peers"] == []


def test_lookup_failure_is_reported_not_swallowed(monkeypatch):
    def explode(_):
        raise RuntimeError("edgar down")
    monkeypatch.setattr(peers, "_company_sic", explode)
    result = peers.find_peers_by_sic("BOOM")
    assert result["success"] is False
    assert "edgar down" in result["error"]


# ------------------------------------------------------------- live golden set

@pytest.fixture(scope="module", autouse=True)
def _load_env():
    from dotenv import load_dotenv
    load_dotenv()


@network
def test_nvda_sic_is_semiconductors():
    result = peers.get_sic_code("NVDA")
    assert result["success"] is True
    assert result["sic"] == "3674"
    assert "semiconductor" in result["industry"].lower()


@network
def test_amd_is_discovered_as_a_peer_of_nvda():
    """The obvious sanity check: the most comparable listed semiconductor
    company must appear."""
    result = peers.find_peers_by_sic("NVDA", limit=40)
    assert result["success"] is True
    tickers = [p["ticker"] for p in result["peers"]]
    assert "AMD" in tickers, f"AMD missing from {tickers}"
    assert "NVDA" not in tickers


@network
def test_unresolved_count_is_reported_honestly():
    result = peers.find_peers_by_sic("NVDA", limit=40)
    assert result["filers_matched"] >= len(result["peers"])
    assert result["unresolved_count"] == result["filers_matched"] - len(result["peers"])
