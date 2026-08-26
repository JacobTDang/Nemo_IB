"""An empty Finnhub payload is a coverage fact, not a finding.

Asked about a symbol that does not exist, the finnhub server answers:

    get_company_profile   success: true, data: {}
    get_basic_financials  success: true, data: {"metric": {}, "series": {}}
    get_company_peers     success: true, data: []

Nothing in any of those says the lookup found nothing. A caller reads a
successful response with no content as "this company has no profile", which is
the same defect as reporting an outage as a non-disclosure -- the README's
first design rule.

Finnhub's free tier also returns an empty body for symbols outside the plan,
so an empty body cannot be read as "no such company" either. The honest label
is that Finnhub returned nothing and we cannot tell which. The server already
reasons this way for forward estimates, where a 403 is kept in the provider's
own words because "flattening that to 'no data' made an entitlement problem
look like a company with no analyst coverage".

`success` stays true: a news window with no articles is a real empty, not a
failure. What changes is that the emptiness is stated rather than left for the
caller to infer from a payload they have to measure.
"""
import pytest

from tools.news_agregator.finnhub_utils import build_envelope


@pytest.mark.parametrize("empty", [{}, [], {"metric": {}, "series": {}}])
def test_an_empty_payload_is_labelled_not_covered(empty):
    envelope = build_envelope(empty, "ZZZZNOTREAL", "get_company_profile")
    assert envelope.get("coverage") == "not_covered", (
        "an empty Finnhub response was returned with no coverage label")


def test_the_label_does_not_claim_the_company_lacks_the_data():
    envelope = build_envelope({}, "ZZZZNOTREAL", "get_company_profile")
    warnings = envelope.get("warnings") or []
    assert warnings, "nothing explains the empty response"
    message = " ".join(str(w.get("message", "")) for w in warnings).lower()
    assert "finnhub" in message
    for forbidden in ("does not disclose", "has no profile", "no such company"):
        assert forbidden not in message, (
            f"the label makes a claim about the company: {message[:160]}")


def test_a_payload_with_content_is_untouched():
    envelope = build_envelope({"name": "NVIDIA"}, "NVDA", "get_company_profile")
    assert envelope.get("coverage") != "not_covered"
    assert not envelope.get("warnings")


def test_a_dict_with_one_real_value_counts_as_content():
    """`{"metric": {}, "series": {"x": 1}}` has data in it."""
    envelope = build_envelope({"metric": {}, "series": {"x": 1}}, "NVDA",
                              "get_basic_financials")
    assert envelope.get("coverage") != "not_covered"


def test_zero_is_content_not_emptiness():
    """A metric that is genuinely zero is an answer."""
    envelope = build_envelope({"shortInterest": 0}, "NVDA",
                              "get_basic_financials")
    assert envelope.get("coverage") != "not_covered"


def test_the_envelope_shape_is_otherwise_unchanged():
    envelope = build_envelope({}, "X", "get_company_profile", api_calls_made=2,
                              errors=["boom"])
    assert envelope["domain"] == "market_intel"
    assert envelope["ticker"] == "X"
    assert envelope["metadata"] == {"api_calls_made": 2, "errors": ["boom"]}
    assert envelope["data"] == {}
