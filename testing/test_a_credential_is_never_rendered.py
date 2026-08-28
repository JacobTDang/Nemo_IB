"""A credential must not be renderable, by anything, ever.

`_verify_model_alive` took the OpenRouter key as a positional parameter, and
pytest prints a frame's arguments at the head of every traceback entry. So any
failing run of testing/test_bugfix_01_model_resolution.py wrote the live key
straight to stdout -- into CI logs, pasted terminals and captured artefacts.
Issue #17.

Suppressing frame locals through pytest configuration would have covered
pytest and nothing else. A credential also reaches a log line, a debugger, a
crash reporter and any f-string written next year, and each of those is the
same disclosure by a different route. So the value is kept behind `Secret`
instead: there is nothing renderable left to render, which closes all of them
at once.

The sentinel below is synthetic and constructed here. The real key is never
read, compared against, or asserted on -- a test that loads the live
credential in order to prove it does not escape is itself the escape.
"""
import os
import subprocess
import sys

import pytest

import agent.openrouter_template as ort
from agent.openrouter_template import Secret


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Shaped like a real OpenRouter key so that anything matching on the 'sk-or-v1-'
# prefix treats it the way it would treat the live one, but with a body that
# could only have come from this file.
_SENTINEL = "sk-or-v1-synthetic0000sentinel0000donotuse"


# ---------------------------------------------------------------------------
# Secret itself
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
  "render",
  [
    repr,
    str,
    lambda s: f"{s}",
    lambda s: f"{s!r}",
    lambda s: f"{s!s}",
    lambda s: "{}".format(s),                    # noqa: UP032 - the point is .format
    lambda s: "%s" % (s,),
    lambda s: str([s]),                          # repr, via a container
    lambda s: str({"key": s}),
  ],
  ids=["repr", "str", "fstring", "fstring_r", "fstring_s",
       "format", "percent", "in_list", "in_dict"],
)
def test_no_way_of_rendering_a_secret_shows_its_value(render):
  """Every route a traceback, a log line or a print could take."""
  rendered = render(Secret(_SENTINEL))
  assert _SENTINEL not in rendered
  assert "synthetic" not in rendered, "part of the value survived rendering"


def test_a_secret_reveals_its_value_only_when_asked():
  """Redacting is worthless if the credential can no longer be used."""
  assert Secret(_SENTINEL).reveal() == _SENTINEL


def test_an_empty_secret_is_falsy():
  """Callers test the credential for presence; an unset key must read absent."""
  assert not Secret("")
  assert Secret(_SENTINEL)


def test_a_secret_scrubs_its_value_out_of_provider_text():
  """Provider error bodies get quoted into our own messages. A provider that
  echoes the offending credential back would otherwise put it in the text we
  raise and print."""
  echoed = f"401 Unauthorized: key {_SENTINEL} is not recognised"
  scrubbed = Secret(_SENTINEL).scrub(echoed)
  assert _SENTINEL not in scrubbed
  assert "401 Unauthorized" in scrubbed, "scrubbing ate the diagnosis"


# ---------------------------------------------------------------------------
# The probe
# ---------------------------------------------------------------------------

def test_the_probe_refuses_a_bare_string_credential():
  """Wrapping a bare string on the callee's side would be no fix at all: the
  raw value would still sit in the *caller's* frame, which is where it leaked
  from. So the bare string is refused, and the refusal names the type only."""
  with pytest.raises(TypeError) as caught:
    ort._verify_model_alive("vendor/model:free", _SENTINEL)
  assert _SENTINEL not in str(caught.value)


def test_the_probe_refuses_to_report_on_a_credential_it_does_not_have():
  """An unset key must raise, not sail into the client.

  The generic handler below reads any non-404 as "not a 404, so keep it in the
  pool". An empty key makes the SDK's own constructor raise, which that handler
  would have read as a live model -- a model marked verified that was never
  probed, which is the exact failure CredentialRejected exists to prevent."""
  from agent.groq_template import CredentialsMissing

  with pytest.raises(CredentialsMissing):
    ort._verify_model_alive("vendor/model:free", Secret(""))


# ---------------------------------------------------------------------------
# End to end: a real pytest run over a deliberately failing probe
# ---------------------------------------------------------------------------

# Run in a subprocess because the thing under test is pytest's own traceback
# rendering, which cannot be observed from inside the run it is rendering.
# --showlocals is the strict setting: without it pytest prints only a frame's
# arguments, so the run would pass on a fix that moved the key from a parameter
# into a local and called it done.
_PROBE_MODULE = '''\
"""Written by testing/test_a_credential_is_never_rendered.py. Not a fixture.

Every test here is meant to fail. The parent asserts on what the failures
print.
"""
import sys

sys.path.insert(0, {root!r})

import openai
import pytest

import agent.openrouter_template as ort


def _rejecting_openai(**kwargs):
  """An OpenAI stand-in that answers every request with a 401.

  Mocked rather than live so this reaches the credential-rejection branch
  offline and on every run -- that branch is the one that raises, and only a
  frame that raises gets rendered.
  """
  class _Completions:
    def create(self, **kw):
      err = openai.AuthenticationError.__new__(openai.AuthenticationError)
      Exception.__init__(err, "Missing Authentication header")
      raise err

  class _Chat:
    completions = _Completions()

  class _Client:
    chat = _Chat()

  return _Client()


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
  # No .env: the credential must be the sentinel the parent exported and
  # nothing else.
  monkeypatch.setattr(ort, "load_dotenv", lambda *a, **k: None)
  monkeypatch.setattr(ort, "OpenAI", _rejecting_openai)


def test_probe_reading_the_environment():
  ort._verify_model_alive("vendor/model:free")


def test_probe_handed_the_credential():
  ort._verify_model_alive("vendor/model:free", ort._openrouter_credential())


def test_pool_refuses_and_reports():
  ort.primary_reasoning_model()
'''


@pytest.fixture
def _failing_probe_run(tmp_path):
  """A pytest run whose every test fails inside the liveness probe."""
  probe = tmp_path / "test_probe_that_must_fail.py"
  probe.write_text(_PROBE_MODULE.format(root=_PROJECT_ROOT))

  env = dict(os.environ)
  env["OPENROUTER_API_KEY"] = _SENTINEL
  env["OPENROUTER_GLM"] = _SENTINEL
  env["OPENROUTER_NEMOTRON"] = _SENTINEL
  env["PYTHONDONTWRITEBYTECODE"] = "1"

  result = subprocess.run(
    [sys.executable, "-m", "pytest", str(probe),
     "-p", "no:randomly", "-p", "no:cacheprovider", "--showlocals", "-q"],
    cwd=_PROJECT_ROOT, env=env, capture_output=True, text=True, timeout=300,
  )
  return result.stdout + result.stderr


def test_the_probe_still_refuses_loudly(_failing_probe_run):
  """The guard on the assertion below.

  "no key in the output" is trivially true of a run that never reached the
  credential -- an import error, a TypeError on the call, a skip. This pins
  that all three probes did reach it and all three refused, so the assertion
  that follows is made about a run where the key really was in play.
  """
  assert "3 failed" in _failing_probe_run, (
    "the probe run did not fail in the three expected places, so it never "
    f"exercised the credential:\n{_failing_probe_run[-3000:]}")
  assert _failing_probe_run.count("CredentialRejected") >= 3, (
    "a probe returned instead of refusing a rejected credential; "
    "CredentialRejected must not be weakened")


def test_a_failing_probe_writes_no_key_material(_failing_probe_run):
  """Issue #17: this is what used to write the live key to stdout."""
  assert _SENTINEL not in _failing_probe_run
  assert "synthetic0000sentinel" not in _failing_probe_run, \
    "part of the credential reached the traceback"


def test_the_probe_takes_no_bare_credential_parameter():
  """Belt and braces on the signature itself, so the parameter cannot come
  back under a different name."""
  import inspect

  params = inspect.signature(ort._verify_model_alive).parameters
  assert "api_key" not in params, \
    "_verify_model_alive grew a bare-string credential parameter again"
  annotations = [str(p.annotation) for p in params.values()]
  assert any("Secret" in a for a in annotations), \
    "the credential parameter is no longer typed as Secret"


def test_the_module_binds_no_unwrapped_credential():
  """Nothing in this module may bind a raw credential to a name.

  Under --showlocals a local is rendered exactly like a parameter, so moving
  the key out of the signature and into a variable would have looked like a fix
  and disclosed the same value. The rule enforced here is narrow and mechanical:
  a credential read out of the environment has to be inside a `Secret(...)`
  before it is assigned to anything.
  """
  import ast

  credentialish = ("KEY", "SECRET", "TOKEN", "PASSWORD")

  def _is_credential_read(node):
    return (isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "getenv"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
            and any(w in node.args[0].value.upper() for w in credentialish))

  source = ast.parse(open(ort.__file__).read())
  offenders = []
  for assignment in ast.walk(source):
    if not isinstance(assignment, ast.Assign):
      continue
    # Reads already enclosed in Secret(...) are the whole point of Secret.
    wrapped = set()
    for node in ast.walk(assignment.value):
      if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
          and node.func.id == "Secret"):
        wrapped.update(id(n) for n in ast.walk(node))
    for node in ast.walk(assignment.value):
      if _is_credential_read(node) and id(node) not in wrapped:
        offenders.append((node.args[0].value, assignment.lineno))

  assert not offenders, (
    f"a credential is assigned unwrapped in {ort.__file__} "
    f"(env var, line): {offenders}. Wrap it in Secret(...) at the read.")
