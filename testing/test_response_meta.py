"""Contract tests for additive response provenance metadata."""

from datetime import datetime

import pytest

from tools.response_meta import SCHEMA_VERSION, annotate, warning


def test_missing_metadata_is_added_without_mutating_the_payload():
    """Existing response contracts have many consumers, so provenance cannot
    rewrite their fields or alter the object a server handed us."""
    payload = {"symbol": "NEMO", "items": []}

    result = annotate(
        payload,
        provider="Example Feed",
        source_url="https://example.test/feed",
        data_as_of="2026-06-30",
        cached=True,
        coverage="partial",
    )

    assert payload == {"symbol": "NEMO", "items": []}
    assert result == {
        "symbol": "NEMO",
        "items": [],
        "success": True,
        "provider": "Example Feed",
        "source_url": "https://example.test/feed",
        "data_as_of": "2026-06-30",
        "retrieved_at": result["retrieved_at"],
        "cached": True,
        "coverage": "partial",
        "warnings": [],
        "schema_version": SCHEMA_VERSION,
    }


def test_existing_keys_are_never_overwritten():
    """Servers that already publish metadata remain authoritative during a
    gradual migration to the shared contract."""
    payload = {
        "success": False,
        "provider": "Original",
        "source_url": "original-url",
        "data_as_of": "2025",
        "retrieved_at": "2025-01-01T00:00:00Z",
        "cached": True,
        "coverage": "full",
        "schema_version": "legacy",
    }

    result = annotate(
        payload,
        provider="Replacement",
        source_url="replacement-url",
        data_as_of="2026",
        cached=False,
        coverage="partial",
        success=True,
    )

    assert result == {**payload, "warnings": []}


@pytest.mark.parametrize(
    ("payload", "explicit", "expected"),
    [
        ({}, None, True),
        ({"items": []}, None, True),
        ({"error": "upstream timeout"}, None, False),
        ({"error": ""}, None, True),
        ({}, False, False),
        ({"success": False}, True, False),
    ],
)
def test_success_inference_does_not_treat_empty_data_as_failure(
    payload, explicit, expected
):
    """A valid query may match no rows, while a non-empty error is an actual
    failure; conflating the two makes absence look like an outage."""
    assert annotate(payload, provider="Feed", success=explicit)["success"] is expected


def test_reannotation_is_idempotent():
    """Responses can cross more than one adapter, so a second annotation pass
    must not replace provenance or create nested metadata."""
    first = annotate({"value": 3}, provider="A")
    second = annotate(first, provider="B")

    assert second == first
    assert second["provider"] == "A"


def test_fred_shaped_envelope_round_trips_unchanged():
    """Envelope-based servers expose data and metadata directly to callers;
    moving or editing either object would break their established contract.

    Equality, not identity. Sharing the object with the caller is exactly what
    lets them rewrite the response a tool already returned, so annotate copies
    deeply -- the contract is that the content survives untouched, not that it
    is the same object.
    """
    data = {"series_id": "GDP", "observations": [{"date": "2026-01-01"}]}
    metadata = {"units": "Billions of Dollars", "frequency": "Quarterly"}
    payload = {
        "domain": "macro",
        "context": "economic_data",
        "tool": "get_series",
        "timestamp": "2026-08-25T12:00:00Z",
        "data": data,
        "metadata": metadata,
    }

    result = annotate(payload, provider="FRED")

    assert result["data"] == data
    assert result["metadata"] == metadata
    assert result["data"] == payload["data"]
    assert result["metadata"] == payload["metadata"]
    assert "provider" not in result["data"]


@pytest.mark.parametrize("payload", [[], "response"])
def test_non_dict_payloads_are_rejected_with_the_received_type(payload):
    """Silently inventing an envelope for raw lists or strings would choose a
    public API shape that the owning tool must decide deliberately."""
    with pytest.raises(TypeError, match=type(payload).__name__):
        annotate(payload, provider="Feed")


@pytest.mark.parametrize("coverage", ["full", "partial", "not_covered", "unknown"])
def test_each_documented_coverage_value_is_accepted(coverage):
    """Coverage has a closed vocabulary so downstream completeness checks do
    not need to interpret near-synonyms."""
    assert annotate({}, provider="Feed", coverage=coverage)["coverage"] == coverage


def test_invalid_coverage_names_the_value_and_allowed_set():
    """A typo must fail at its source instead of silently weakening a caller's
    completeness gate."""
    with pytest.raises(ValueError) as exc_info:
        annotate({}, provider="Feed", coverage="complete")

    message = str(exc_info.value)
    assert "complete" in message
    for allowed in ("full", "partial", "not_covered", "unknown"):
        assert allowed in message


def test_coverage_defaults_to_unknown():
    """Claiming full coverage without evidence recreates the silent omission
    this metadata contract is intended to expose."""
    assert annotate({}, provider="Feed")["coverage"] == "unknown"


def test_retrieved_at_is_parseable_utc_with_a_z_suffix():
    """Cross-region callers need an unambiguous retrieval instant rather than
    whichever local timezone the serving process happens to use."""
    retrieved_at = annotate({}, provider="Feed")["retrieved_at"]

    assert retrieved_at.endswith("Z")
    parsed = datetime.fromisoformat(retrieved_at)
    assert parsed.utcoffset().total_seconds() == 0


def test_warning_helper_preserves_structured_extra_fields():
    """Stable codes support automation while extra context lets operators act
    without parsing prose."""
    assert warning("STALE", "Observation is old", age_days=190) == {
        "code": "STALE",
        "message": "Observation is old",
        "age_days": 190,
    }


@pytest.mark.parametrize("entries", [["free text"], [{}], [{"code": "X"}]])
def test_unstructured_warning_entries_are_rejected(entries):
    """Bare prose and incomplete records cannot be routed or handled reliably
    by programmatic clients."""
    with pytest.raises(TypeError):
        annotate({}, provider="Feed", warnings=entries)


def test_existing_warnings_are_appended_without_mutating_the_original_list():
    """Earlier adapters may already have detected degradation, so later
    annotations must retain those warnings in their original order."""
    existing = [warning("STALE", "Old filing")]
    added = warning("PARTIAL", "One endpoint failed")
    payload = {"warnings": existing}

    result = annotate(payload, provider="Feed", warnings=[added])

    assert result["warnings"] == [existing[0], added]
    assert payload["warnings"] == [existing[0]]


def test_error_responses_still_receive_provenance():
    """A failed response without its upstream identity cannot be attributed or
    diagnosed during an outage."""
    result = annotate({"error": "rate limited"}, provider="Vendor API")

    assert result["success"] is False
    assert result["provider"] == "Vendor API"
    assert "retrieved_at" in result
    assert result["schema_version"] == SCHEMA_VERSION


def test_data_as_of_none_is_an_explicit_value_for_live_data():
    """A live quote has no reporting period, and callers must distinguish that
    fact from a producer that omitted the contract field entirely."""
    result = annotate({}, provider="Quotes", data_as_of=None)

    assert "data_as_of" in result
    assert result["data_as_of"] is None


# --------------------------------------------------------- re-annotation edges

def test_reannotating_with_the_same_warning_does_not_duplicate_it():
    """Rule 3 says re-annotation is safe, and a caller cannot always know.

    A server that annotates in a helper and again at the dispatch boundary
    would otherwise stack the same caveat twice, and a reader counting
    warnings would see a filing flagged twice as stale.
    """
    stale = warning("stale_data", "short interest is 2-3 weeks old")
    once = annotate({}, provider="Finnhub", warnings=[stale])
    twice = annotate(once, provider="Finnhub", warnings=[stale])

    assert twice["warnings"] == [stale], f"duplicated: {twice['warnings']}"


def test_distinct_warnings_still_accumulate():
    """Deduplication must not swallow a genuinely different caveat."""
    first = warning("stale_data", "2-3 weeks old")
    second = warning("partial_coverage", "3 of 8 quarters missing")
    once = annotate({}, provider="SEC EDGAR", warnings=[first])
    twice = annotate(once, provider="SEC EDGAR", warnings=[second])

    assert twice["warnings"] == [first, second]


def test_annotating_does_not_share_mutable_state_with_the_payload():
    """The docstring promises it mutates nothing; a shallow copy breaks that.

    `result["data"]` was the same object as `payload["data"]`, so a caller
    editing the annotated response silently rewrote the value the tool had
    already returned to someone else -- and with a cache in front of it, the
    cached entry too.
    """
    payload = {"data": {"revenue": 1}, "rows": [{"n": 1}]}
    annotated = annotate(payload, provider="SEC EDGAR")

    annotated["data"]["revenue"] = 999
    annotated["rows"][0]["n"] = 999

    assert payload["data"]["revenue"] == 1, "the payload's data was rewritten"
    assert payload["rows"][0]["n"] == 1, "the payload's rows were rewritten"


def test_a_payload_that_already_states_success_wins_over_the_caller():
    """Deliberate, and worth stating because the spec was ambiguous here.

    Rule 1 says never overwrite; the caller-supplied `success=` argument
    looked like an exception to it. It is not. The tool that built the payload
    knows whether its own call succeeded; an annotating wrapper further out
    does not, and letting the outer layer override would let a dispatch
    boundary mark a genuine failure as a success. `success=` therefore fills a
    gap and never contradicts.
    """
    assert annotate({"success": True}, provider="P", success=False)["success"] is True
    assert annotate({"success": False}, provider="P", success=True)["success"] is False
    # It still fills the gap when the payload says nothing.
    assert annotate({}, provider="P", success=False)["success"] is False
