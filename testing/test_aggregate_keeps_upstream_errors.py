"""When every source fails, say why each one failed.

`extract_forward_signals` reads two sources -- earnings releases and the MD&A
-- and when neither yields text it answers:

    No text sources available (earnings releases + MD&A both failed)

Both underlying calls returned a reason, and both reasons were discarded. The
sentence is true and useless: it cannot distinguish a filer with no MD&A from
an SEC outage, from a missing SEC_EMAIL, from a ticker that does not exist.
Observed live with credentials unset, where the real cause was named twice and
reported zero times.

This is the outage-as-finding failure wearing a different hat. The tool does
not claim the filer discloses nothing, but it leaves the caller no way to tell
that it hasn't.
"""
import pytest

import tools.web_search_server.sec_utils as su


@pytest.fixture
def both_sources_failing(monkeypatch):
    monkeypatch.setattr(su, "get_earnings_releases", lambda *a, **k: {
        "success": False,
        "error": "SEC returned 503 for the submissions index"})
    monkeypatch.setattr(su, "extract_mda", lambda *a, **k: {
        "success": False,
        "error": "SEC_EMAIL is not set"})


def test_the_reason_each_source_failed_survives(both_sources_failing):
    result = su.extract_forward_signals("TSM")

    assert result["success"] is False
    blob = repr(result)
    assert "503" in blob, "the earnings-release failure was discarded"
    assert "SEC_EMAIL" in blob, "the MD&A failure was discarded"


def test_the_failures_are_addressable_per_source(both_sources_failing):
    """A caller retrying needs to know which source to retry."""
    result = su.extract_forward_signals("TSM")
    failures = result.get("source_failures")

    assert isinstance(failures, dict), (
        "no per-source failure map, so a caller cannot tell which source broke")
    assert "503" in failures.get("earnings_releases", "")
    assert "SEC_EMAIL" in failures.get("mda", "")


def test_a_source_that_merely_had_nothing_is_not_called_a_failure(monkeypatch):
    """An empty MD&A is a finding about the filing; a failed one is not."""
    monkeypatch.setattr(su, "get_earnings_releases", lambda *a, **k: {
        "success": True, "releases": []})
    monkeypatch.setattr(su, "extract_mda", lambda *a, **k: {
        "success": True, "text": ""})

    result = su.extract_forward_signals("TSM")
    failures = result.get("source_failures") or {}
    assert not failures, (
        f"sources that succeeded with no content were reported as failures: "
        f"{failures}")
