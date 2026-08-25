"""Service gates for tests that need something this machine may not have.

Two rules:

1. A skip must name what is missing. "skipped" with no reason is how a broken
   test hides.
2. Every skip must be convertible into a failure. Set NEMO_REQUIRE_SERVICES=1
   and a gated test fails instead of skipping, so a run with full credentials
   proves the gated tests still work rather than merely still exist.

The LLM gates are split by provider because the providers are separately
configured and separately broken: OPENROUTER_API_KEY is set on this machine
while GROQ_API_KEY is present with an empty value. A single `requires_llm`
would skip OpenRouter-backed tests that can actually run.

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
    if name == "groq":
        if _offline():
            return _OFFLINE_REASON.format(service="the Groq API")
        return None if _env_key("GROQ_API_KEY") else (
            "GROQ_API_KEY is unset or empty; set it in .env to run this test")
    if name == "openrouter":
        if _offline():
            return _OFFLINE_REASON.format(service="the OpenRouter API")
        return None if _env_key("OPENROUTER_API_KEY") else (
            "OPENROUTER_API_KEY is unset or empty; set it in .env to run this test")
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


requires_groq = _gate("groq")
requires_openrouter = _gate("openrouter")
requires_searxng = _gate("searxng")
requires_playbook = _gate("playbook")
requires_sec = _gate("sec")
requires_finnhub = _gate("finnhub")
requires_fred = _gate("fred")
