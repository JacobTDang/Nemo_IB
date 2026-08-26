"""The per-class breakdown must be deduplicated like the total it accompanies.

Biogen emits its share count twice with identical value, period and
context_ref. An XBRL context plus a concept defines exactly one fact, so
those two rows are one fact, and summing them reported roughly 295m shares
against a real ~148m.

`FilingPoint.total()` was fixed to deduplicate. `by_class` was not: it walks
`point.facts` raw. The headline number is therefore right while the
breakdown printed beside it lists every Biogen period twice -- and anyone
adding the classes up, which is exactly what a multi-class filer invites,
gets the original doubled figure back.
"""
import pytest

from tools.web_search_server.sec_series import ConceptFact, FilingPoint
import tools.web_search_server.dilution as dilution


def _biogen_point():
    """One filing, one fact, emitted twice -- Biogen's actual shape."""
    fact = dict(value=147_753_998.0, period="2026-07-27",
                context_ref="c-1", concept=dilution.SHARES_CONCEPT,
                unit="shares")
    return FilingPoint(
        filing_date="2026-07-29", form="10-Q", accession="0000875045-26-000050",
        facts=[ConceptFact(**fact), ConceptFact(**fact)])


def _alphabet_point():
    """Three genuinely distinct classes must survive deduplication."""
    def klass(member, value, ctx):
        return ConceptFact(value=value, period="2026-07-15", context_ref=ctx,
                           concept=dilution.SHARES_CONCEPT, unit="shares",
                           dimensions={dilution.CLASS_AXIS: member})
    return FilingPoint(
        filing_date="2026-07-23", form="10-Q", accession="0001652044-26-000070",
        facts=[klass("us-gaap:CommonClassAMember", 5_868_000_000.0, "c-1"),
               klass("us-gaap:CommonClassBMember", 835_000_000.0, "c-2"),
               klass("goog:CapitalClassCMember", 5_527_000_000.0, "c-3")])


@pytest.fixture
def _stub(monkeypatch):
    """The filings, offline.

    The quote provider goes with them: a multi-class filer now has its class
    weights checked against a market capitalisation, and these tests are about
    which facts survive deduplication rather than what the classes are worth
    to each other. Raising is what the offline suite actually is -- a source
    that cannot be reached -- and `_share_basis` reports that rather than
    assuming the classes are equivalent.
    """
    def install(point):
        monkeypatch.setattr(dilution, "fetch_concept_series",
                            lambda *a, **k: [point])
        monkeypatch.setattr(dilution, "fetch_market_share_count",
                            _no_quote_provider)
    return install


def _no_quote_provider(ticker):
    raise dilution.MarketShareCountUnavailable(
        f"no quote provider is reachable from the offline suite ({ticker})")


def test_a_duplicated_fact_appears_once_per_class(_stub):
    _stub(_biogen_point())
    result = dilution.get_share_count_series("BIIB")

    rows = result["by_class"]["Common"]
    assert len(rows) == 1, (
        f"Biogen's one share-count fact is listed {len(rows)} times; the "
        f"breakdown disagrees with latest_total beside it: {rows}")


def test_the_classes_sum_to_the_reported_total(_stub):
    """The invariant that keeps the two from drifting apart again."""
    _stub(_biogen_point())
    result = dilution.get_share_count_series("BIIB")

    newest = max(r["period"] for rows in result["by_class"].values() for r in rows)
    latest = sum(r["shares"] for rows in result["by_class"].values()
                 for r in rows if r["period"] == newest)
    assert latest == result["latest_total"], (
        f"by_class sums to {latest:,.0f} but latest_total is "
        f"{result['latest_total']:,.0f} -- summing the breakdown, which is "
        f"what a multi-class filer invites, reproduces the original bug")


def test_distinct_share_classes_are_not_collapsed(_stub):
    """Deduplication must key on the fact, never on the concept."""
    _stub(_alphabet_point())
    result = dilution.get_share_count_series("GOOGL")

    assert sorted(result["classes_found"]) == [
        "Capital Class C", "Class A", "Class B"]
    newest = max(r["period"] for rows in result["by_class"].values() for r in rows)
    assert sum(r["shares"] for rows in result["by_class"].values()
               for r in rows if r["period"] == newest) == \
        result["latest_total"] == 12_230_000_000.0
