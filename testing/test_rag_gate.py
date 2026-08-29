"""The RAG gate names what is missing and can be turned into a failure.

`test_rag_search` and `test_rag_ingest` need an ingested corpus. Without one
they do not skip -- they fail, on `top_ids=[]` and on 23 chunks against a
`>100` bar. Five permanently-red tests train the eye to ignore red, which is
worse than the coverage they were providing.

They are gated rather than deleted because the corpus is a real thing a
machine may have: `rag_search` and `rag_ingest` are declared tools, hidden in
the image only because the RAG stack is not installed there. So this follows
the rule the other gates follow -- a skip must say what is absent, and
NEMO_REQUIRE_SERVICES=1 must turn it into a failure, or the gate cannot fail
and is not a gate.
"""
import os
import subprocess
import sys

from testing import _gates


def test_the_gate_exists_and_names_what_is_missing():
    reason = _gates.service_missing("rag")
    assert reason is None or ("corpus" in reason.lower()
                              or "rag" in reason.lower()), (
        f"a skip must name what is absent; got {reason!r}")


def test_an_unknown_service_still_raises():
    """The factory must not answer for a name it does not know -- that is how
    a typo becomes a gate that silently never fires."""
    try:
        _gates.service_missing("not_a_service")
    except ValueError:
        return
    raise AssertionError("service_missing accepted an unknown gate name")


def test_strict_mode_turns_the_skip_into_a_failure():
    """The property the whole gate module exists for, exercised end to end.

    Run in a subprocess because STRICT is read at import time, and asserting
    it in-process would only prove the constant was patched.
    """
    probe = os.path.join(os.path.dirname(__file__), "test_zz_rag_gate_probe.py")
    with open(probe, "w") as fh:
        fh.write("from testing._gates import requires_rag\n\n\n"
                 "@requires_rag\n"
                 "def test_probe():\n"
                 "    assert True\n")
    try:
        env = {**os.environ, "NEMO_REQUIRE_SERVICES": "1"}
        strict = subprocess.run(
            [sys.executable, "-m", "pytest", probe, "-q", "-p", "no:randomly"],
            capture_output=True, text=True, env=env, timeout=300, cwd=os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))))
    finally:
        os.path.exists(probe) and os.remove(probe)

    if _gates.service_missing("rag") is None:
        assert strict.returncode == 0, "the corpus is present; the gate should pass"
    else:
        assert strict.returncode != 0, (
            "NEMO_REQUIRE_SERVICES=1 left the gated test skipping, so the gate "
            "cannot fail:\n" + strict.stdout[-1500:])
