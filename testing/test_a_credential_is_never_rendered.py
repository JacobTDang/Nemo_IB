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

**What issue #63 changed.** `agent/openrouter_template.py` -- the module this
file was written about, and the one the probe and the end-to-end
`--showlocals` run exercised -- was deleted with the LangGraph/OpenRouter
layer. Those tests went with it: a probe function that no longer exists cannot
leak a key. What survives is the rule, and the module it now applies to is
`agent/groq_template.py`, the remaining LLM template. It is not a museum
piece: the news daemons build it on every article through
`Materiality_Classifier`, so it holds a live credential in a loop.

The end-to-end proof that a *failing pytest run* discloses nothing is held for
the credentials that can move money by
`testing/test_a_broker_credential_is_never_rendered.py`, which runs a
deliberately-failing suite under `--showlocals` in a subprocess. It is not
duplicated here.

The sentinel below is synthetic and constructed here. The real key is never
read, compared against, or asserted on -- a test that loads the live
credential in order to prove it does not escape is itself the escape.
"""
import ast
import os

import pytest

import agent.groq_template as gt
from agent.groq_template import CredentialsMissing, Secret


# Shaped like a real Groq key so that anything matching on the 'gsk_' prefix
# treats it the way it would treat the live one, but with a body that could
# only have come from this file.
_SENTINEL = "gsk_synthetic0000sentinel0000donotuse"


# ---------------------------------------------------------------------------
# Secret itself, reached through the module that holds the credential
# ---------------------------------------------------------------------------
#
# Imported from `agent.groq_template` rather than from `common.secret` on
# purpose: `mod.Secret` is the name this module's own code and its callers
# reach for, so a module that quietly grew a local one back fails here.

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
# The read
# ---------------------------------------------------------------------------

def test_the_credential_leaves_the_read_already_wrapped(monkeypatch):
  """The value must never exist as a bare `str` in any frame, including this
  one. Returning it wrapped is what makes that true of every caller at once."""
  monkeypatch.setattr(gt, "load_dotenv", lambda *a, **k: None)
  monkeypatch.setenv("GROQ_API_KEY", _SENTINEL)

  resolved = gt.GroqModel()._resolve_credential()
  assert isinstance(resolved, Secret)
  assert _SENTINEL not in repr(resolved)


def test_the_read_is_annotated_as_returning_a_secret():
  """Belt and braces on the signature, so a bare `str` cannot come back under
  the same name."""
  import inspect

  annotation = inspect.signature(gt.GroqModel._resolve_credential).return_annotation
  assert "Secret" in str(annotation), (
    "_resolve_credential no longer declares that it returns a Secret")


def test_the_template_refuses_to_report_on_a_credential_it_does_not_have(monkeypatch):
  """An unset key must raise, not sail into the SDK constructor.

  A client built from an empty string is a client that fails later, somewhere
  else, with a provider error that names neither the cause nor the fix."""
  monkeypatch.setattr(gt, "load_dotenv", lambda *a, **k: None)
  monkeypatch.setenv("GROQ_API_KEY", "")

  with pytest.raises(CredentialsMissing) as caught:
    _ = gt.GroqModel().client
  assert _SENTINEL not in str(caught.value)


# ---------------------------------------------------------------------------
# The rule, enforced mechanically on the module that holds the credential
# ---------------------------------------------------------------------------

_CREDENTIALISH = ("KEY", "SECRET", "TOKEN", "PASSWORD")


def _is_credential_read(node):
  """True for `os.getenv("...KEY...")` and `os.getenv(self._api_key_env)`.

  The indirect form matters here: `GroqModel` reads through
  `self._api_key_env`, so a check that only understood string literals would
  pass this module without looking at anything.
  """
  if not (isinstance(node, ast.Call)
          and isinstance(node.func, ast.Attribute)
          and node.func.attr == "getenv"
          and node.args):
    return False
  target = node.args[0]
  if isinstance(target, ast.Constant) and isinstance(target.value, str):
    name = target.value
  elif isinstance(target, ast.Attribute):
    name = target.attr        # self._api_key_env
  elif isinstance(target, ast.Name):
    name = target.id
  else:
    return False
  return any(word in name.upper() for word in _CREDENTIALISH)


def test_the_module_binds_no_unwrapped_credential():
  """Nothing in the module may bind a raw credential to a name.

  Under --showlocals a local is rendered exactly like a parameter, so moving
  the key out of a signature and into a variable would have looked like a fix
  and disclosed the same value. The rule enforced here is narrow and
  mechanical: a credential read out of the environment has to be inside a
  `Secret(...)` before it is assigned to anything.
  """
  source = ast.parse(open(gt.__file__, encoding="utf-8").read())

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
        offenders.append((ast.unparse(node), assignment.lineno))

  assert not offenders, (
    f"a credential is assigned unwrapped in {gt.__file__} "
    f"(read, line): {offenders}. Wrap it in Secret(...) at the read.")


def test_the_check_above_can_actually_fail():
  """A mechanical rule that matches nothing passes everything.

  `GroqModel` reads its key indirectly, so this pins that the detector sees
  that form -- otherwise the test above would be green on a module it never
  looked inside.
  """
  tree = ast.parse(
    "import os\n"
    "cred = os.getenv(self._api_key_env)\n"
    "safe = Secret(os.getenv(self._api_key_env) or '')\n")
  reads = [n for n in ast.walk(tree) if _is_credential_read(n)]
  assert len(reads) == 2, (
    "the detector no longer recognises `os.getenv(self._api_key_env)`, so the "
    "rule above is vacuous on every module that reads its key indirectly")
  assert os.path.exists(gt.__file__)
