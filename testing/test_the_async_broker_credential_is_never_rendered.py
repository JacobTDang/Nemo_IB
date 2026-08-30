"""The async broker's credentials must not be renderable either.

Issue #64, split out of #61. `tools/alpaca/async_broker.py` is the last
credential site in the repo still holding its key and secret as raw strings on
the instance. It was left out of the #59 and #61 sweeps for a mechanical
reason rather than a judgement about risk: `testing/test_phase_B3a_alpaca_env_
fallback.py` asserts on `broker.key` and `broker.secret` as strings, so closing
this one needs a test change as well as a source change and sat outside the
file list of both.

The exposure is the same one `common/secret.py` is written around, with two
routes of its own:

  * the instance dict. `AsyncBroker` keeps the credentials for the lifetime of
    the object, so a debugger, a crash reporter or anything that dumps
    `vars(broker)` finds them there long after the constructor returned.
  * the provider's error text. `_raise_for_status` puts the response body into
    `AsyncBrokerError`, and `tools/alpaca/server.py` puts `str(e)` straight
    into the `error` field of the MCP result it returns -- so a 401 body that
    quoted the offending key would carry it out of the process entirely rather
    than merely into a log.

These are paper *and* live broker credentials: `AsyncBroker(paper=False)` reads
ALPACA_LIVE_KEY, which can move real money.

Every credential below is a synthetic sentinel constructed in this file. The
real keys are never read, compared against, or asserted on -- a test that loads
a live credential in order to prove it does not escape is itself the escape.
The sentinels are long on purpose: a short one is found inside ordinary prose,
and scrubbing then rewrites the diagnosis it was supposed to leave alone.
"""
import ast
import os
import subprocess
import sys

import httpx
import pytest

# The same rule, not a second copy of it. `_is_credential_read` is the
# mechanical check issue #59 landed for the five modules it covered; async_broker
# is the sixth, and a rule that is re-typed here would drift from the one those
# five are held to.
from testing.test_a_broker_credential_is_never_rendered import _is_credential_read


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Shaped like the real thing so anything matching on a prefix treats them the
# way it would treat a live key, with bodies that could only have come from
# this file.
_ALPACA_KEY = "PKasyncbroker0000synthetic0000key"
_ALPACA_SECRET = "asyncbroker0000synthetic0000secret0000donotuse"

# The body fragment every assertion below looks for. Matching on this rather
# than on the whole value catches a partial disclosure -- a truncated error
# message, a key split across a wrapped traceback line.
_BODY = "asyncbroker0000synthetic"


def _alpaca_env(**overrides):
  """All six Alpaca names cleared, then the given ones set.

  Cleared rather than left alone because `testing/conftest.py` loads the
  project `.env` at session start, and async_broker loads it again at import.
  Without this a developer's real credentials stand in for a sentinel and the
  assertions below are made about a value this file never constructed.
  """
  env = {k: "" for k in (
    "ALPACA_PAPER_KEY", "ALPACA_PAPER_SECRET",
    "ALPACA_LIVE_KEY", "ALPACA_LIVE_SECRET",
    "ALPACA_API_KEY", "ALPACA_SECRET",
  )}
  env.update(overrides)
  return env


@pytest.fixture
def paper_broker(monkeypatch):
  """The module and a paper broker built on the synthetic credentials."""
  from tools.alpaca import async_broker as ab

  for name, value in _alpaca_env(ALPACA_PAPER_KEY=_ALPACA_KEY,
                                 ALPACA_PAPER_SECRET=_ALPACA_SECRET).items():
    monkeypatch.setenv(name, value)
  return ab, ab.AsyncBroker(paper=True)


# ---------------------------------------------------------------------------
# The credentials on the instance
# ---------------------------------------------------------------------------

def test_the_broker_names_the_shared_secret_type():
  """One definition, for the reason `common/secret.py` exists.

  Two implementations of a redaction primitive drift, and the drift shows up
  as the leak."""
  from common.secret import Secret as Shared
  from tools.alpaca import async_broker as ab

  assert getattr(ab, "Secret", None) is Shared


def test_the_broker_holds_its_credentials_as_secrets(paper_broker):
  """The attributes outlive the constructor, so they are renderable for as
  long as the broker is."""
  ab, broker = paper_broker

  assert isinstance(broker._key, ab.Secret)
  assert isinstance(broker._secret, ab.Secret)
  assert _BODY not in f"{broker._key} {broker._secret}"
  assert _BODY not in f"{vars(broker)}", \
    "the instance dict renders the credentials"


async def test_the_broker_still_sends_a_usable_credential(paper_broker):
  """Redaction that also breaks authentication is not a fix.

  The values asserted on are the synthetic ones the fixture set, so this
  discloses nothing."""
  _ab, broker = paper_broker

  async with broker:
    assert broker._client.headers["APCA-API-KEY-ID"] == _ALPACA_KEY, \
      "the broker went out with a redacted key"
    assert broker._client.headers["APCA-API-SECRET-KEY"] == _ALPACA_SECRET


# ---------------------------------------------------------------------------
# The provider's error text, which leaves the process
# ---------------------------------------------------------------------------

async def test_a_failed_request_carries_no_key_material(paper_broker):
  """`str(e)` from here becomes the `error` field the MCP caller receives."""
  from tools.alpaca.async_broker import AsyncBrokerError

  _ab, broker = paper_broker

  def handler(request):
    # Alpaca's own 401 body does not quote the key today. This one does,
    # which is the case the scrubbing exists for -- nothing here should
    # depend on a provider never starting to echo what it rejected.
    return httpx.Response(401, json=dict(
      message="access key verification failed for "
              + request.headers["APCA-API-KEY-ID"]))

  async with broker:
    broker._client._transport = httpx.MockTransport(handler)
    with pytest.raises(AsyncBrokerError) as caught:
      await broker.get_account()

  message = str(caught.value)
  assert "401" in message, \
    f"the request failed somewhere other than the status branch: {message}"
  assert _BODY not in message, \
    "the provider's echo of the key was handed back to the caller"


# ---------------------------------------------------------------------------
# The refusals, which must stay loud
# ---------------------------------------------------------------------------

def test_the_broker_still_refuses_a_half_configured_paper_account(monkeypatch):
  """A key with no secret must raise, naming what to set and neither value."""
  from tools.alpaca.async_broker import AsyncBroker

  for name, value in _alpaca_env(ALPACA_PAPER_KEY=_ALPACA_KEY).items():
    monkeypatch.setenv(name, value)

  with pytest.raises(RuntimeError) as caught:
    AsyncBroker(paper=True)
  message = str(caught.value)
  assert "paper" in message, "the refusal no longer says which account it means"
  assert "ALPACA_PAPER_KEY" in message, "the refusal no longer names what to set"
  assert _BODY not in message, "the refusal quoted the credential"


def test_the_broker_still_refuses_a_half_configured_live_account(monkeypatch):
  """The live account is the one that can move real money."""
  from tools.alpaca.async_broker import AsyncBroker

  for name, value in _alpaca_env(ALPACA_LIVE_KEY=_ALPACA_KEY).items():
    monkeypatch.setenv(name, value)

  with pytest.raises(RuntimeError) as caught:
    AsyncBroker(paper=False)
  message = str(caught.value)
  assert "LIVE" in message, "the refusal no longer says which account it means"
  assert "ALPACA_LIVE_SECRET" in message, "the refusal no longer names what to set"
  assert _BODY not in message, "the refusal quoted the credential"


# ---------------------------------------------------------------------------
# The rule, enforced mechanically
# ---------------------------------------------------------------------------

def test_async_broker_binds_no_unwrapped_credential():
  """A credential read out of the environment must be inside `Secret(...)`
  before it is assigned to anything -- an attribute included.

  Under --showlocals a local is rendered exactly like a parameter, so moving a
  key out of a signature and into a variable looks like a fix and discloses the
  same value. The rule is narrow and mechanical on purpose: it is the one that
  can be checked without running the code that leaks.
  """
  path = os.path.join(_PROJECT_ROOT, "tools/alpaca/async_broker.py")
  source = ast.parse(open(path).read())
  offenders = []
  for assignment in ast.walk(source):
    if not isinstance(assignment, (ast.Assign, ast.AnnAssign)):
      continue
    if assignment.value is None:
      continue
    wrapped = set()
    for node in ast.walk(assignment.value):
      if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
          and node.func.id == "Secret"):
        wrapped.update(id(n) for n in ast.walk(node))
    for node in ast.walk(assignment.value):
      if _is_credential_read(node) and id(node) not in wrapped:
        offenders.append(assignment.lineno)

  assert not offenders, (
    "a credential is assigned unwrapped in tools/alpaca/async_broker.py at "
    f"line(s) {offenders}. Wrap it in Secret(...) at the read.")


# ---------------------------------------------------------------------------
# End to end: a real pytest run, with the strict rendering setting
# ---------------------------------------------------------------------------

# Run in a subprocess because the thing under test is pytest's own traceback
# rendering, which cannot be observed from inside the run it is rendering.
# --showlocals is the strict setting: without it pytest prints only a frame's
# arguments, so the run would pass on a fix that moved a key from a parameter
# into a local and called it done. --tb=long because the default `auto` style
# shortens the middle entries of a traceback, and the frame that reads the
# provider's error body is exactly such an entry.
#
# The sentinels reach the probe through the environment and are read inline,
# never written into the probe's source and never bound to one of its names.
# A test file that spells its own sentinel out would fail on its own source
# line, which says nothing about the code under test.
_PROBE_KEY_VAR = "_PROBE_ASYNC_BROKER_KEY"
_PROBE_SECRET_VAR = "_PROBE_ASYNC_BROKER_SECRET"

_PROBE_MODULE = '''\
"""Written by testing/test_the_async_broker_credential_is_never_rendered.py.

Every test here is meant to fail with a credential in play. The parent asserts
on what those failures print.
"""
import asyncio
import os
import sys

sys.path.insert(0, {root!r})

import httpx

from tools.alpaca.async_broker import AsyncBroker

KEY_VAR = {key_var!r}
SECRET_VAR = {secret_var!r}


def _configured_paper_broker(monkeypatch):
  # Returns rather than raises, so this helper is never a traceback frame.
  monkeypatch.setenv("ALPACA_PAPER_KEY", os.environ[KEY_VAR])
  monkeypatch.setenv("ALPACA_PAPER_SECRET", os.environ[SECRET_VAR])
  return AsyncBroker(paper=True)


def test_paper_key_without_its_secret(monkeypatch):
  """The half-configured account: key read, secret absent, refuse."""
  monkeypatch.setenv("ALPACA_PAPER_KEY", os.environ[KEY_VAR])
  monkeypatch.setenv("ALPACA_PAPER_SECRET", "")
  AsyncBroker(paper=True)


def test_live_key_without_its_secret(monkeypatch):
  """The same shape on the account that can move real money."""
  monkeypatch.setenv("ALPACA_LIVE_KEY", os.environ[KEY_VAR])
  monkeypatch.setenv("ALPACA_LIVE_SECRET", "")
  AsyncBroker(paper=False)


def test_a_dump_of_the_broker_state(monkeypatch):
  """What a debugger or a crash reporter shows for a live broker object."""
  broker = _configured_paper_broker(monkeypatch)
  state = vars(broker)
  assert state["base_url"] == "deliberate failure, with the state on a local"


def test_a_provider_error_that_quotes_the_key(monkeypatch):
  """A 401 whose body echoes the credential back at us."""
  broker = _configured_paper_broker(monkeypatch)

  def handler(request):
    return httpx.Response(401, json=dict(
      message="access key verification failed for "
              + request.headers["APCA-API-KEY-ID"]
              + " with secret "
              + request.headers["APCA-API-SECRET-KEY"]))

  async def run():
    async with broker:
      broker._client._transport = httpx.MockTransport(handler)
      await broker.get_account()

  asyncio.run(run())
'''


@pytest.fixture(scope="module")
def failing_broker_run(tmp_path_factory):
  """A pytest run whose every test fails with a broker credential in play."""
  probe = tmp_path_factory.mktemp("probe") / "test_probe_that_must_fail.py"
  probe.write_text(_PROBE_MODULE.format(root=_PROJECT_ROOT,
                                        key_var=_PROBE_KEY_VAR,
                                        secret_var=_PROBE_SECRET_VAR))

  env = dict(os.environ)
  env[_PROBE_KEY_VAR] = _ALPACA_KEY
  env[_PROBE_SECRET_VAR] = _ALPACA_SECRET
  # Cleared here as well so that nothing the developer's own .env put in the
  # environment can stand in for a sentinel and make this run pass on a real
  # credential it never asserted about. Empty rather than absent: async_broker
  # calls load_dotenv() at import, and load_dotenv leaves a name that is
  # already present alone.
  for name in _alpaca_env():
    env[name] = ""
  env["PYTHONDONTWRITEBYTECODE"] = "1"

  result = subprocess.run(
    [sys.executable, "-m", "pytest", str(probe),
     "-p", "no:randomly", "-p", "no:cacheprovider",
     "--showlocals", "--tb=long", "-q"],
    cwd=_PROJECT_ROOT, env=env, capture_output=True, text=True, timeout=300,
  )
  return result.stdout + result.stderr


def test_every_probe_failure_still_happened_where_it_should(failing_broker_run):
  """The guard on the assertion below.

  "no key in the output" is trivially true of a run that never reached the
  credential -- an import error, a skip, a constructor that succeeded. This
  pins that all four probes did read a credential and that the two refusals
  still refuse in their own words, so the assertion that follows is made about
  a run where a key really was in play.
  """
  assert "4 failed" in failing_broker_run, (
    "the probe run did not fail in the four expected places, so it never "
    f"exercised the credentials:\n{failing_broker_run[-3000:]}")
  assert failing_broker_run.count("Missing Alpaca") >= 2, (
    "AsyncBroker stopped refusing a half-configured account; that refusal "
    "must not be weakened")
  assert "ALPACA_PAPER_KEY" in failing_broker_run, (
    "the paper refusal no longer names what to set")
  assert "ALPACA_LIVE_KEY" in failing_broker_run, (
    "the live refusal no longer names what to set")
  assert "HTTP 401" in failing_broker_run, (
    "the request probe never reached the provider-error branch, so 'no key in "
    "the error' is trivially true of it")


def test_a_failing_broker_writes_no_key_material(failing_broker_run):
  """Issue #64: this is the run that writes the broker key to stdout."""
  assert _ALPACA_KEY not in failing_broker_run
  assert _ALPACA_SECRET not in failing_broker_run
  assert _BODY not in failing_broker_run, \
    "part of a credential reached the traceback"


def test_there_is_no_public_raw_accessor(monkeypatch):
    """`broker.key` must not hand back the credential.

    It survived the first pass as a property returning `.reveal()`, which
    keeps it out of the instance dict -- so `--showlocals` and a debugger see
    only a Secret. But a public attribute that looks like a plain string and
    returns one is the shape that put the raw value on a local in the first
    place, and nothing about the call site warns that it is a reveal. The two
    readers were both in one test; `_key.reveal()` says what it does.
    """
    from tools.alpaca.async_broker import AsyncBroker

    monkeypatch.setenv("ALPACA_PAPER_KEY", _ALPACA_KEY)
    monkeypatch.setenv("ALPACA_PAPER_SECRET", _ALPACA_SECRET)
    broker = AsyncBroker(paper=True)

    for name in ("key", "secret"):
        assert not hasattr(broker, name), (
            f"AsyncBroker.{name} is a public accessor returning the raw "
            f"credential; read `_{name}.reveal()` at the point of use instead")
