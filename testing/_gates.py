"""Service gates for tests that need something this machine may not have.

Two rules:

1. A skip must name what is missing. "skipped" with no reason is how a broken
   test hides.
2. Every skip must be convertible into a failure. Set NEMO_REQUIRE_SERVICES=1
   and a gated test fails instead of skipping, so a run with full credentials
   proves the gated tests still work rather than merely still exist.

Gates are split by provider because providers are separately configured and
separately broken -- one key present and another empty is the ordinary case,
and a single `requires_llm` would skip tests that can actually run.

3. A gate must guard at least one test. One that guards none cannot fail
   either, and reads as coverage of a path that is gone -- which is what
   `requires_openrouter` became when the agent layer was retired and every
   test using it was deleted with the module. Enforced by test_gates.

SKIP_NETWORK_TESTS=1 counts as "unavailable" for every gate that reaches a
live service. That is the whole point of the offline run: an LLM call or a
SearXNG query is a network call whether or not a credential happens to be
present. Before this module existed nothing marked those tests, so an offline
run still spent real time and real money on live completions.
"""
import os

import pytest

# The reachability probe lives with the server that owns the dependency. Reused
# rather than copied so the two cannot drift apart on the default URL or the
# timeout.
from tools.web_search_server.web_search import _searxng_reachable

# Read at import: pytest markers are built at import, so a mid-session change
# could not take effect anyway. testing/test_gates.py drives strict mode
# through a subprocess for exactly this reason.
STRICT = os.environ.get("NEMO_REQUIRE_SERVICES") == "1"

# What the RAG suite needs to mean anything. Its own bootstrap smoke test asks
# for more than a hundred chunks, and the retrieval tests seed a handful of
# their own on top -- so an index below this answers their queries with an
# empty result and no error, which is a failure that looks like a finding.
_MIN_RAG_CHUNKS = 100


def _offline() -> bool:
    """True when the run has declared itself offline."""
    return os.environ.get("SKIP_NETWORK_TESTS", "0") == "1"


# Credentials live in .env, and every LLM template calls load_dotenv() in its
# own constructor. Reading os.environ alone therefore made the gates blinder
# than the code they gate: pytest loads no .env, so OpenRouter tests were
# skipped while the code under them would have run fine. A gate more
# conservative than reality is not safe -- it is the invisible-failure case
# these gates exist to prevent, arriving by another door.
_ENV_LOADED = False


def _load_env_once() -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:  # noqa: BLE001 - absent .env is normal in CI
        pass
    _ENV_LOADED = True


def _env_key(name: str) -> bool:
    _load_env_once()
    # An empty value is a missing credential, not a configured one.
    return bool(os.environ.get(name, "").strip())


def _playbook_present() -> bool:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.exists(os.path.join(root, "CLAUDE.md"))


_OFFLINE_REASON = (
    "SKIP_NETWORK_TESTS=1: this test calls {service} over the network. "
    "Unset SKIP_NETWORK_TESTS to run it."
)


def service_missing(name: str) -> str | None:
    _load_env_once()
    """Return a human-readable reason the service is unavailable, or None."""
    if name == "searxng":
        if _offline():
            return _OFFLINE_REASON.format(service="SearXNG")
        return None if _searxng_reachable() else (
            "SearXNG is not reachable at SEARXNG_URL (default http://localhost:8888); "
            "start it with: docker compose up -d searxng")
    if name == "playbook":
        return None if _playbook_present() else (
            "CLAUDE.md is absent. It is gitignored, so this lint only runs on a "
            "machine that has the local playbook")
    if name == "sec":
        # Every other service gate refuses when the run declares itself
        # offline; this one did not, so SKIP_NETWORK_TESTS=1 still made live
        # EDGAR calls and the "offline, no network" suite was neither.
        if _offline():
            return _OFFLINE_REASON.format(service="SEC EDGAR")
        return None if _env_key("SEC_EMAIL") else (
            "SEC_EMAIL is unset; SEC fair access requires a real contact identity")
    if name == "finnhub":
        if _offline():
            return _OFFLINE_REASON.format(service="the Finnhub API")
        return None if _env_key("FINNHUB_API_KEY") else (
            "FINNHUB_API_KEY is unset or empty; set it in .env to run this test")
    if name == "fred":
        if _offline():
            return _OFFLINE_REASON.format(service="the FRED API")
        return None if _env_key("FRED_API_KEY") else (
            "FRED_API_KEY is unset or empty; set it in .env to run this test")
    if name == "rag":
        # Not a credential and not a network service: a corpus somebody
        # ingested. `rag_search` and `rag_ingest` are declared tools, hidden in
        # the image only because the RAG stack is not installed there, so a
        # machine that has both the package and an index can really run these.
        # Sized rather than merely present, because an empty index answers
        # every query with an empty result and no error -- which is what made
        # these five fail rather than skip.
        try:
            from agent.rag import store
        except Exception as exc:      # noqa: BLE001 - reported, not hidden
            return (f"the RAG stack is not importable ({type(exc).__name__}); "
                    f"install it to run this test")
        try:
            chunks = store.count_chunks()
        except Exception as exc:      # noqa: BLE001 - a missing table is a
            # missing corpus, and saying which is more use than a traceback.
            return (f"the RAG index cannot be read ({type(exc).__name__}: "
                    f"{exc}); ingest a corpus to run this test")
        if chunks < _MIN_RAG_CHUNKS:
            return (f"the RAG corpus holds {chunks} chunks, under the "
                    f"{_MIN_RAG_CHUNKS} these tests need; run the ingest "
                    f"bootstrap to populate it")
        return None
    raise ValueError(f"unknown service gate: {name!r}")


def _gate(name: str):
    reason = service_missing(name)
    if reason is None:
        return pytest.mark.skipif(False, reason="")
    if STRICT:
        # A gate that cannot fail is not a gate. The marker is consumed by the
        # pytest_pyfunc_call hook in conftest.py, which fails during the call
        # phase. Failing in setup instead would report an error, not a failure.
        return pytest.mark.fail_missing_service(name=name, reason=reason)
    return pytest.mark.skipif(True, reason=reason)


requires_searxng = _gate("searxng")
requires_playbook = _gate("playbook")
requires_sec = _gate("sec")
requires_finnhub = _gate("finnhub")
requires_fred = _gate("fred")
requires_rag = _gate("rag")


def skip_if_provider_unavailable(result: dict, provider: str = "") -> None:
    """Skip when the upstream refused to answer at all.

    A live test asserting "GovTrack returns bills for Technology" tests
    GovTrack, not us. When the provider is genuinely down -- and these servers
    report that distinctly, with coverage "not_covered" and reason
    "provider_unavailable" -- the assertion is unanswerable, and failing it
    reports our tool as broken when the tool did exactly the right thing.

    Deliberately narrow: only the server's own "no provider answered" signal
    skips. A provider that answered with nothing still fails the test, which is
    what the test is for.
    """
    if not isinstance(result, dict):
        return
    if result.get("reason") != "provider_unavailable":
        return
    detail = str(result.get("error") or "")[:300]
    pytest.skip(f"{provider or 'the upstream provider'} did not answer, so "
                f"this assertion is unanswerable: {detail}")
