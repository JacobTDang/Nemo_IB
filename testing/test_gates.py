"""A gate that cannot fail is not a gate.

Skipping a test when its dependency is absent is only honest if the skip can
be turned back into a failure on demand. Otherwise "skipped" silently decays
into "deleted" and nobody notices that dozens of tests stopped running.
"""
import os
import pathlib
import subprocess
import sys
import tempfile
import textwrap

import pytest


def _online(monkeypatch):
    """These tests exercise credential resolution, so the offline gate -- which
    is checked first and would mask it -- has to be off."""
    monkeypatch.delenv("SKIP_NETWORK_TESTS", raising=False)


def test_missing_key_is_reported(monkeypatch):
    from testing import _gates
    _online(monkeypatch)
    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
    assert _gates.service_missing("finnhub") is not None


def test_present_key_is_not_reported(monkeypatch):
    from testing import _gates
    _online(monkeypatch)
    monkeypatch.setenv("FINNHUB_API_KEY", "test-key-not-real")
    assert _gates.service_missing("finnhub") is None


def test_empty_key_counts_as_missing(monkeypatch):
    """A key present in .env with an empty value is a missing credential, not
    a configured one. This was written against GROQ_API_KEY, which was empty
    in .env for exactly this reason; Groq is gone and the rule is not."""
    from testing import _gates
    _online(monkeypatch)
    monkeypatch.setenv("FINNHUB_API_KEY", "")
    assert _gates.service_missing("finnhub") is not None


def test_an_unknown_service_is_refused_rather_than_passed(monkeypatch):
    """Removing the groq branch turned service_missing("groq") into a
    ValueError rather than a silent None. A gate that quietly approves a
    service nobody defined is not a gate."""
    import pytest

    from testing import _gates
    _online(monkeypatch)
    with pytest.raises(ValueError):
        _gates.service_missing("groq")


def test_offline_run_counts_as_missing(monkeypatch):
    """A live completion is a network call whether or not a key is configured.
    Nothing marked these tests before, so SKIP_NETWORK_TESTS=1 did not gate
    them and an offline run still spent real time on live LLM calls."""
    from testing import _gates
    monkeypatch.setenv("SKIP_NETWORK_TESTS", "1")
    monkeypatch.setenv("FINNHUB_API_KEY", "test-key-not-real")
    reason = _gates.service_missing("finnhub")
    assert reason is not None and "SKIP_NETWORK_TESTS" in reason


def test_playbook_gate_ignores_the_offline_flag(monkeypatch, tmp_path):
    """CLAUDE.md is a local file, not a service. An offline run must still
    run the playbook lint when the playbook is there."""
    from testing import _gates
    monkeypatch.setenv("SKIP_NETWORK_TESTS", "1")
    monkeypatch.setattr(_gates, "_playbook_present", lambda: True)
    assert _gates.service_missing("playbook") is None


def test_reason_names_the_missing_dependency(monkeypatch):
    """A skip with no reason is how a broken test hides."""
    from testing import _gates
    _online(monkeypatch)
    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
    assert "FINNHUB_API_KEY" in _gates.service_missing("finnhub")


def test_unknown_service_raises(monkeypatch):
    """A typo in a gate name must not silently mean 'available'."""
    from testing import _gates
    with pytest.raises(ValueError, match="unknown service gate"):
        _gates.service_missing("no_such_service")


def _run_probe(env_extra):
    """Run a one-test probe in a subprocess with a controlled environment.

    A subprocess is required because _gates.STRICT is read at import time, so
    NEMO_REQUIRE_SERVICES cannot be toggled within a running session.

    The probe file is written into testing/ rather than a tempdir on purpose:
    testing/conftest.py owns the strict-mode hook, and a file outside that
    directory would not pick the conftest up. The gate would then silently
    never fire and this test would pass for the wrong reason.
    """
    probe = textwrap.dedent("""
        import pytest
        from testing._gates import requires_finnhub

        pytestmark = requires_finnhub

        def test_needs_finnhub():
            assert True
    """)
    testing_dir = pathlib.Path(__file__).resolve().parent
    handle, name = tempfile.mkstemp(
        prefix="test_probe_gate_", suffix=".py", dir=testing_dir)
    os.close(handle)
    path = pathlib.Path(name)
    path.write_text(probe)
    try:
        env = {**os.environ, **env_extra}
        env.pop("GROQ_API_KEY", None)
        env["PYTHONPATH"] = str(testing_dir.parent)
        env["SKIP_NETWORK_TESTS"] = "1"
        return subprocess.run(
            [sys.executable, "-m", "pytest", str(path),
             "-q", "--no-header", "-p", "no:randomly"],
            capture_output=True, text=True, env=env,
        )
    finally:
        path.unlink(missing_ok=True)


def test_gate_skips_when_service_absent():
    result = _run_probe({})
    assert "1 skipped" in result.stdout, result.stdout


def test_gate_fails_when_strict_mode_is_on():
    result = _run_probe({"NEMO_REQUIRE_SERVICES": "1"})
    assert "1 failed" in result.stdout, result.stdout


def test_strict_mode_reports_a_failure_not_an_error():
    """pytest.fail() inside pytest_runtest_setup reports an ERROR, which muddies
    the '0 failed, 0 errors' target. The hook must fire in the call phase."""
    result = _run_probe({"NEMO_REQUIRE_SERVICES": "1"})
    assert "1 failed" in result.stdout, result.stdout
    assert "error" not in result.stdout.lower(), result.stdout


# --------------------------------------------------------------------------
# The gates must resolve credentials the same way the code they gate does.
#
# _gates read os.environ directly. The credentials live in .env, and every LLM
# template calls load_dotenv() in its own constructor -- so the code could see
# a key the gate could not, and pytest (which loads no .env) skipped tests that
# would have passed. A gate more conservative than reality is not "safe": it is
# the invisible-failure case these gates exist to prevent, arriving by a
# different door.
# --------------------------------------------------------------------------

def test_gates_resolve_dotenv_like_the_code_they_gate(tmp_path, monkeypatch):
    """A key present only in .env must count as available."""
    import importlib
    import os

    env = tmp_path / ".env"
    env.write_text("GATEPROBE_KEY=a-real-looking-value\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GATEPROBE_KEY", raising=False)

    from dotenv import load_dotenv
    load_dotenv(env)
    assert os.environ.get("GATEPROBE_KEY") == "a-real-looking-value", (
        "dotenv did not populate the environment; the premise of this test is wrong")

    import testing._gates as gates
    importlib.reload(gates)
    assert hasattr(gates, "_load_env_once"), (
        "_gates does not load .env, so it cannot see credentials the code it "
        "gates resolves through load_dotenv()")


def test_a_gate_sees_a_dotenv_key(monkeypatch):
    """The concrete case: FINNHUB_API_KEY is set in this repo's .env, and
    the offline suite was skipping OpenRouter tests that would have run."""
    import importlib
    import pathlib

    repo_env = pathlib.Path(__file__).resolve().parent.parent / ".env"
    if not repo_env.exists():
        pytest.skip("no .env in this checkout")
    if "FINNHUB_API_KEY" not in repo_env.read_text():
        pytest.skip("FINNHUB_API_KEY not configured in .env")

    # SKIP_NETWORK_TESTS deliberately counts as "service unavailable" for
    # network gates, so it has to be cleared to test credential resolution.
    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
    monkeypatch.delenv("SKIP_NETWORK_TESTS", raising=False)
    import testing._gates as gates
    importlib.reload(gates)
    assert gates.service_missing("finnhub") is None, (
        "gate reports OpenRouter unavailable while .env configures it")


def test_finnhub_gate_reports_missing_key(monkeypatch):
    from testing import _gates
    _online(monkeypatch)
    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
    assert "FINNHUB_API_KEY" in _gates.service_missing("finnhub")


def test_fred_gate_reports_missing_key(monkeypatch):
    from testing import _gates
    _online(monkeypatch)
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    assert "FRED_API_KEY" in _gates.service_missing("fred")


def test_finnhub_and_fred_gates_respect_the_offline_flag(monkeypatch):
    """Both reach a live HTTP API, so an offline run must gate them even when
    the credential is configured."""
    from testing import _gates
    monkeypatch.setenv("SKIP_NETWORK_TESTS", "1")
    monkeypatch.setenv("FINNHUB_API_KEY", "test-key-not-real")
    monkeypatch.setenv("FRED_API_KEY", "test-key-not-real")
    for service in ("finnhub", "fred"):
        reason = _gates.service_missing(service)
        assert reason is not None and "SKIP_NETWORK_TESTS" in reason, service


def test_every_gate_guards_at_least_one_test():
    """A gate nothing applies is the same defect this module exists to stop.

    `_gates` opens by saying a gate that cannot fail is not a gate. One that
    guards no test cannot fail either: it passes forever, it is exercised only
    by the tests written for it, and it reads as coverage of a path that is
    gone. `requires_openrouter` became exactly that when the agent layer was
    retired and every test using it was deleted with the module.
    """
    import importlib
    import pathlib
    import re

    _gates = importlib.import_module("testing._gates")
    here = pathlib.Path(__file__).resolve().parent
    gates = {name for name in dir(_gates) if name.startswith("requires_")}
    assert gates, "no gates found; this test is looking in the wrong place"

    used = set()
    for path in here.glob("test_*.py"):
        if path.name in ("test_gates.py",):
            continue          # its own tests name every gate by construction
        text = path.read_text(encoding="utf-8")
        for name in gates:
            if re.search(rf"\b{name}\b", text):
                used.add(name)

    unused = sorted(gates - used)
    assert not unused, (
        f"these gates guard no test, so they cannot fail and stand for "
        f"coverage that no longer exists: {unused}. Remove the gate and its "
        f"env var, or apply it.")
