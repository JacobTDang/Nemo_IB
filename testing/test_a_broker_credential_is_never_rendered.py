"""Broker and data-provider credentials must not be renderable either.

Issue #17 closed one disclosure route -- the OpenRouter key sitting in a
parameter of `_verify_model_alive`. Issue #59 is the same shape in five more
places, and one of them is worse than the original: `Execution_Agent.__init__`
binds the *live* Alpaca key and secret as plain locals, and a partially
configured `.env` (key present, secret absent) makes that constructor raise
with both still live in the frame. pytest prints every local under
`--showlocals`, so the credential that can move real money is written to
whatever captured that run.

The fix is the `Secret` type from `agent/openrouter_template.py`: the value
lives behind `reveal()`, so there is nothing renderable left to render, and
`scrub()` takes it back out of provider error text before that text is printed
or returned to a caller.

Every credential below is a synthetic sentinel constructed in this file. The
real keys are never read, compared against, or asserted on -- a test that loads
a live credential in order to prove it does not escape is itself the escape.
"""
import ast
import importlib
import os
import subprocess
import sys
import types

import aiohttp
import pytest


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Shaped like the real thing so that anything matching on a prefix treats them
# the way it would treat a live key, with bodies that could only have come from
# this file.
_ALPACA_KEY = "PKsynthetic0000alpaca0000key"
_ALPACA_SECRET = "synthetic0000alpaca0000secret0000donotuse"
_GROQ_KEY = "gsk_synthetic0000groq0000sentinel0000donotuse"
_FINNHUB_KEY = "synthetic0000finnhub0000sentinel"
_FRED_KEY = "synthetic0000fred0000sentinel"

# The body fragment every assertion below looks for. Matching on this rather
# than on the whole value catches a partial disclosure -- a truncated error
# message, a key split across a wrapped traceback line.
_BODY = "synthetic0000"


# ---------------------------------------------------------------------------
# Secret, in each module that carries one
# ---------------------------------------------------------------------------
# `Secret` is mirrored into each module rather than imported: agent/groq_template
# is imported *by* agent/openrouter_template (importing back is a cycle),
# Execution_Agent is a 112-module import that must not grow the 1146-module LLM
# layer to reach a value class, and tools/news_agregator + tools/alpaca_server
# import nothing from agent/ today. Mirrored or not, every copy must behave
# identically, so they are tested as one.

_SECRET_HOMES = [
  "agent.Execution_Agent",
  "agent.groq_template",
  "tools.alpaca_server.alpaca_server",
  "tools.news_agregator.finnhub_utils",
  "tools.news_agregator.fred_utils",
]


@pytest.fixture(params=_SECRET_HOMES)
def secret_cls(request):
  """The `Secret` mirror carried by one of the modules in the issue.

  Imported here rather than at module scope so a module that has not grown one
  yet fails this fixture's tests instead of erroring collection for the file.
  """
  module = importlib.import_module(request.param)
  cls = getattr(module, "Secret", None)
  assert cls is not None, f"{request.param} carries no Secret"
  return cls


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
def test_no_way_of_rendering_a_secret_shows_its_value(secret_cls, render):
  """Every route a traceback, a log line or a print could take."""
  rendered = render(secret_cls(_ALPACA_SECRET))
  assert _ALPACA_SECRET not in rendered
  assert _BODY not in rendered, "part of the value survived rendering"


def test_a_secret_reveals_its_value_only_when_asked(secret_cls):
  """Redacting is worthless if the credential can no longer be used."""
  assert secret_cls(_ALPACA_KEY).reveal() == _ALPACA_KEY


def test_an_empty_secret_is_falsy(secret_cls):
  """Callers test the credential for presence; an unset key must read absent."""
  assert not secret_cls("")
  assert secret_cls(_ALPACA_KEY)


def test_a_secret_scrubs_its_value_out_of_provider_text(secret_cls):
  """Provider error text is printed and, in the data servers, returned to the
  caller. A provider that quotes the offending credential back -- or simply
  echoes the request URL it was a query parameter of -- would otherwise put it
  in our own diagnostics."""
  echoed = f"401 Unauthorized: token={_FINNHUB_KEY} was rejected"
  scrubbed = secret_cls(_FINNHUB_KEY).scrub(echoed)
  assert _BODY not in scrubbed
  assert "401 Unauthorized" in scrubbed, "scrubbing ate the diagnosis"


# ---------------------------------------------------------------------------
# Execution_Agent -- the live broker credentials
# ---------------------------------------------------------------------------

def _alpaca_env(**overrides):
  """All six Alpaca names cleared, then the given ones set."""
  env = {k: "" for k in (
    "ALPACA_PAPER_KEY", "ALPACA_PAPER_SECRET",
    "ALPACA_LIVE_KEY", "ALPACA_LIVE_SECRET",
    "ALPACA_API_KEY", "ALPACA_SECRET",
  )}
  env.update(overrides)
  return env


def test_execution_agent_still_refuses_a_half_configured_live_account(monkeypatch):
  """The refusal this whole test file is built around must stay loud.

  A live key with no live secret is the configuration that makes the
  constructor raise, and it must keep raising -- naming the variables to set
  and naming neither of their values."""
  from agent import Execution_Agent as ea

  monkeypatch.setattr(ea, "load_dotenv", lambda *a, **k: None)
  for name, value in _alpaca_env(ALPACA_LIVE_KEY=_ALPACA_KEY).items():
    monkeypatch.setenv(name, value)

  with pytest.raises(RuntimeError) as caught:
    ea.Execution_Agent(paper=False)
  message = str(caught.value)
  assert "LIVE" in message, "the refusal no longer says which account it means"
  assert "ALPACA_LIVE_KEY" in message, "the refusal no longer names what to set"
  assert _BODY not in message, "the refusal quoted the credential"


def test_execution_agent_holds_its_credentials_as_secrets(monkeypatch):
  """The attributes survive the constructor, so they are renderable too."""
  from agent import Execution_Agent as ea

  monkeypatch.setattr(ea, "load_dotenv", lambda *a, **k: None)
  for name, value in _alpaca_env(ALPACA_LIVE_KEY=_ALPACA_KEY,
                                 ALPACA_LIVE_SECRET=_ALPACA_SECRET).items():
    monkeypatch.setenv(name, value)

  agent = ea.Execution_Agent(paper=False)
  assert isinstance(agent._key, ea.Secret)
  assert isinstance(agent._secret, ea.Secret)
  assert _BODY not in f"{agent._key} {agent._secret}"
  assert _BODY not in f"{vars(agent)}", "the instance dict renders the credentials"


def test_execution_agent_still_hands_the_broker_a_usable_credential(monkeypatch):
  """Redaction that also breaks authentication is not a fix.

  The values captured here are the synthetic ones set above, so asserting on
  them discloses nothing."""
  from agent import Execution_Agent as ea

  monkeypatch.setattr(ea, "load_dotenv", lambda *a, **k: None)
  for name, value in _alpaca_env(ALPACA_LIVE_KEY=_ALPACA_KEY,
                                 ALPACA_LIVE_SECRET=_ALPACA_SECRET).items():
    monkeypatch.setenv(name, value)

  captured = {}

  class _RecordingTradingClient:
    def __init__(self, api_key=None, secret_key=None, paper=None):
      captured.update(api_key=api_key, secret_key=secret_key, paper=paper)

  fake_module = types.ModuleType("alpaca.trading.client")
  fake_module.TradingClient = _RecordingTradingClient
  monkeypatch.setitem(sys.modules, "alpaca.trading.client", fake_module)

  ea.Execution_Agent(paper=False)._get_client()
  assert captured["api_key"] == _ALPACA_KEY, "the broker got a redacted key"
  assert captured["secret_key"] == _ALPACA_SECRET
  assert captured["paper"] is False


# ---------------------------------------------------------------------------
# alpaca_server -- the same credentials, read in a data server
# ---------------------------------------------------------------------------

def test_alpaca_server_refuses_a_half_configured_account(monkeypatch):
  """setup_clients must say which variable is missing rather than handing
  `None` to alpaca-py and letting the request come back 401."""
  from tools.alpaca_server import alpaca_server as srv

  monkeypatch.setattr(srv, "load_dotenv", lambda *a, **k: None)
  for name, value in _alpaca_env(ALPACA_API_KEY=_ALPACA_KEY).items():
    monkeypatch.setenv(name, value)

  with pytest.raises(RuntimeError) as caught:
    srv.alpaca_client()
  message = str(caught.value)
  assert "ALPACA_SECRET" in message, "the refusal no longer names what to set"
  assert _BODY not in message, "the refusal quoted the credential"


# ---------------------------------------------------------------------------
# groq_template -- a credential handed back across a function boundary
# ---------------------------------------------------------------------------

def test_groq_resolves_to_a_secret_not_a_string(monkeypatch):
  """A `-> str` return type is a renderable credential in every caller's frame."""
  import agent.groq_template as gt

  monkeypatch.setattr(gt, "load_dotenv", lambda *a, **k: None)
  monkeypatch.setenv("GROQ_API_KEY", _GROQ_KEY)

  credential = gt.GroqModel()._resolve_credential()
  assert isinstance(credential, gt.Secret)
  assert credential.reveal() == _GROQ_KEY
  assert _BODY not in f"{credential!r}"


def test_groq_still_refuses_a_missing_credential(monkeypatch):
  """CredentialsMissing exists so a daemon looping over articles reports the
  right cause. Wrapping the value must not soften that."""
  import agent.groq_template as gt

  monkeypatch.setattr(gt, "load_dotenv", lambda *a, **k: None)
  monkeypatch.delenv("GROQ_API_KEY", raising=False)
  with pytest.raises(gt.CredentialsMissing):
    gt.GroqModel()._resolve_credential()


def test_groq_scrubs_the_key_out_of_a_provider_error_it_prints(monkeypatch, capsys):
  """Groq's error text is printed verbatim on every retry and fallback hop.

  A provider that echoes the offending credential back -- or an SDK that quotes
  the request it built -- puts the key on stderr once per attempt.
  """
  import agent.groq_template as gt
  from openai import APIError

  monkeypatch.setattr(gt, "load_dotenv", lambda *a, **k: None)
  monkeypatch.setenv("GROQ_API_KEY", _GROQ_KEY)
  monkeypatch.setattr(gt.GroqModel, "MAX_RETRIES", 2)
  monkeypatch.setattr(gt.GroqModel, "RETRY_BASE_DELAY", 0)

  def _echoing_openai(**kwargs):
    """An OpenAI stand-in whose failures quote the credential.

    Built so that nothing in *this* helper raises: a stub that took the key as
    an argument and then raised would render it from its own frame, and the
    assertion below would be about the stub rather than about the fix.
    """
    class _Completions:
      def create(self, **kw):
        err = APIError.__new__(APIError)
        Exception.__init__(
          err, f"upstream rejected Authorization: Bearer {_GROQ_KEY}")
        raise err

    class _Chat:
      completions = _Completions()

    class _Client:
      chat = _Chat()

    return _Client()

  monkeypatch.setattr(gt, "OpenAI", _echoing_openai)
  monkeypatch.setattr(gt, "ollama_chat",
                      lambda **kw: (_ for _ in ()).throw(RuntimeError("no ollama")))

  model = gt.GroqModel()
  with pytest.raises(APIError):
    model.generate_response("anything")

  printed = capsys.readouterr()
  combined = printed.out + printed.err
  assert "[Retry 1/2]" in combined, (
    "the call never reached the retry path, so 'no key in the output' is "
    f"trivially true here:\n{combined[-2000:]}")
  assert _BODY not in combined, "the provider's echo of the key was printed"


# ---------------------------------------------------------------------------
# finnhub_utils / fred_utils -- a key that travels as a query parameter
# ---------------------------------------------------------------------------

class _SessionThatFailsQuotingTheUrl:
  """aiohttp stand-in whose error text quotes the URL it was given.

  This is the real shape, not a contrivance: the credential is a *query
  parameter*, so the URL aiohttp builds contains it, and several aiohttp errors
  render that URL. `get()` puts `str(e)` into the dict it returns to the MCP
  caller, which is how a key leaves the process entirely rather than merely
  reaching a log.
  """
  closed = False

  def __init__(self, secret_in_url: str):
    self._secret_in_url = secret_in_url

  def get(self, url, **kwargs):
    raise aiohttp.ClientError(
      f"Cannot connect to host for {url}?token={self._secret_in_url}")

  async def close(self):
    pass


class _SessionRecordingParams:
  """aiohttp stand-in that records the query it was asked to send."""
  closed = False

  def __init__(self, sink):
    self._sink = sink

  def get(self, url, **kwargs):
    self._sink.update(kwargs.get("params") or {})
    raise aiohttp.ClientError("recorded")

  async def close(self):
    pass


@pytest.fixture(params=["finnhub", "fred"])
def data_client(request, monkeypatch):
  """A Finnhub or FRED client built on a synthetic credential."""
  if request.param == "finnhub":
    from tools.news_agregator import finnhub_utils as mod
    monkeypatch.setattr(mod, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("FINNHUB_API_KEY", _FINNHUB_KEY)
    return mod, mod.FinnhubClient(), _FINNHUB_KEY, "token", "/company-news"
  from tools.news_agregator import fred_utils as mod
  monkeypatch.setattr(mod, "load_dotenv", lambda *a, **k: None)
  monkeypatch.setenv("FRED_API_KEY", _FRED_KEY)
  return mod, mod.FredClient(), _FRED_KEY, "api_key", "/series/observations"


def test_get_api_key_returns_a_secret(data_client):
  """`-> str` hands a renderable credential to every caller."""
  mod, client, sentinel, _param, _endpoint = data_client
  credential = mod.get_api_key()
  assert isinstance(credential, mod.Secret)
  assert credential.reveal() == sentinel
  assert _BODY not in f"{credential!r}"
  assert isinstance(client._api_key, mod.Secret), \
    "the client still holds a renderable credential on the instance"


def test_get_api_key_still_refuses_when_unset(data_client, monkeypatch):
  """An unset key must raise, not sail into the request as an empty token."""
  mod, _client, _sentinel, _param, _endpoint = data_client
  monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
  monkeypatch.delenv("FRED_API_KEY", raising=False)
  with pytest.raises(RuntimeError) as caught:
    mod.get_api_key()
  assert "not found in environment" in str(caught.value)


async def test_a_failed_request_returns_no_key_material(data_client, monkeypatch):
  """The returned error goes back to the MCP caller, so it must carry no key."""
  _mod, client, sentinel, _param, endpoint = data_client
  session = _SessionThatFailsQuotingTheUrl(sentinel)

  async def _fake_session():
    return session

  monkeypatch.setattr(client, "_get_session", _fake_session)

  result = await client.get(endpoint, {"symbol": "TST"})
  assert "error" in result, (
    "the request did not fail where this test assumes it does, so 'no key in "
    f"the error' is trivially true: {result}")
  assert "HTTP client error" in result["error"], \
    f"failed somewhere other than the client-error branch: {result['error']}"
  assert _BODY not in result["error"], \
    "the provider's error text carried the credential back to the caller"


async def test_the_request_still_carries_a_usable_credential(data_client, monkeypatch):
  """Redaction that breaks authentication is not a fix."""
  _mod, client, sentinel, param, endpoint = data_client
  sent = {}
  session = _SessionRecordingParams(sent)

  async def _fake_session():
    return session

  monkeypatch.setattr(client, "_get_session", _fake_session)

  await client.get(endpoint, {"symbol": "TST"})
  assert sent.get(param) == sentinel, \
    f"the request went out with {param}={sent.get(param)!r}"


# ---------------------------------------------------------------------------
# The rule, enforced mechanically on every module in the issue
# ---------------------------------------------------------------------------

_CREDENTIAL_MODULES = [
  "agent/Execution_Agent.py",
  "agent/groq_template.py",
  "tools/alpaca_server/alpaca_server.py",
  "tools/news_agregator/finnhub_utils.py",
  "tools/news_agregator/fred_utils.py",
]

_CREDENTIALISH = ("KEY", "SECRET", "TOKEN", "PASSWORD")


def _is_credential_read(node):
  """True for `os.getenv("...KEY...")` and `os.getenv(self._api_key_env)`."""
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


@pytest.mark.parametrize("relative_path", _CREDENTIAL_MODULES)
def test_no_module_binds_an_unwrapped_credential(relative_path):
  """A credential read out of the environment must be inside `Secret(...)`
  before it is assigned to anything.

  Under --showlocals a local is rendered exactly like a parameter, so moving a
  key out of a signature and into a variable looks like a fix and discloses the
  same value. The rule is narrow and mechanical on purpose: it is the one that
  can be checked without running the code that leaks.
  """
  path = os.path.join(_PROJECT_ROOT, relative_path)
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
    f"a credential is assigned unwrapped in {relative_path} at line(s) "
    f"{offenders}. Wrap it in Secret(...) at the read.")


# ---------------------------------------------------------------------------
# End to end: a real pytest run, with the strict rendering setting
# ---------------------------------------------------------------------------

# Run in a subprocess because the thing under test is pytest's own traceback
# rendering, which cannot be observed from inside the run it is rendering.
# --showlocals is the strict setting: without it pytest prints only a frame's
# arguments, so the run would pass on a fix that moved a key from a parameter
# into a local and called it done.
# Run in a subprocess because the thing under test is pytest's own traceback
# rendering, which cannot be observed from inside the run it is rendering.
# --showlocals is the strict setting: without it pytest prints only a frame's
# arguments, so the run would pass on a fix that moved a key from a parameter
# into a local and called it done.
#
# The sentinels reach the probe through the environment and are read inline,
# never written into the probe's source and never bound to one of its names.
# A test file that spells its own sentinel out would fail on its own source
# line, which says nothing about the code under test.
_PROBE_KEY_VAR = "_PROBE_ALPACA_KEY"
_PROBE_SECRET_VAR = "_PROBE_ALPACA_SECRET"

_PROBE_MODULE = '''\
"""Written by testing/test_a_broker_credential_is_never_rendered.py.

Every test here is meant to fail, in a constructor that has just read a
credential. The parent asserts on what those failures print.
"""
import os
import sys

sys.path.insert(0, {root!r})

import pytest

from agent import Execution_Agent as ea
from tools.alpaca_server import alpaca_server as srv

KEY_VAR = {key_var!r}
SECRET_VAR = {secret_var!r}


@pytest.fixture(autouse=True)
def _no_dotenv(monkeypatch):
  # The credentials must be the sentinels the parent exported and nothing else.
  monkeypatch.setattr(ea, "load_dotenv", lambda *a, **k: None)
  monkeypatch.setattr(srv, "load_dotenv", lambda *a, **k: None)


def test_live_key_without_its_secret(monkeypatch):
  """The half-configured live account: key read, secret absent, raise."""
  monkeypatch.setenv("ALPACA_LIVE_KEY", os.environ[KEY_VAR])
  monkeypatch.setenv("ALPACA_LIVE_SECRET", "")
  ea.Execution_Agent(paper=False)


def test_live_secret_without_its_key(monkeypatch):
  monkeypatch.setenv("ALPACA_LIVE_KEY", "")
  monkeypatch.setenv("ALPACA_LIVE_SECRET", os.environ[SECRET_VAR])
  ea.Execution_Agent(paper=False)


def test_paper_key_without_its_secret(monkeypatch):
  monkeypatch.setenv("ALPACA_PAPER_KEY", os.environ[KEY_VAR])
  monkeypatch.setenv("ALPACA_PAPER_SECRET", "")
  ea.Execution_Agent(paper=True)


def test_server_key_without_its_secret(monkeypatch):
  monkeypatch.setenv("ALPACA_API_KEY", os.environ[KEY_VAR])
  monkeypatch.setenv("ALPACA_SECRET", "")
  srv.alpaca_client()
'''


@pytest.fixture(scope="module")
def _failing_constructor_run(tmp_path_factory):
  """A pytest run whose every test fails inside a credential-reading constructor."""
  probe = tmp_path_factory.mktemp("probe") / "test_probe_that_must_fail.py"
  probe.write_text(_PROBE_MODULE.format(root=_PROJECT_ROOT,
                                        key_var=_PROBE_KEY_VAR,
                                        secret_var=_PROBE_SECRET_VAR))

  env = dict(os.environ)
  env[_PROBE_KEY_VAR] = _ALPACA_KEY
  env[_PROBE_SECRET_VAR] = _ALPACA_SECRET
  # Cleared here as well so that nothing the developer's own .env put in the
  # environment can stand in for a sentinel and make this run pass on a real
  # credential it never asserted about.
  for name in ("ALPACA_PAPER_KEY", "ALPACA_PAPER_SECRET",
               "ALPACA_LIVE_KEY", "ALPACA_LIVE_SECRET",
               "ALPACA_API_KEY", "ALPACA_SECRET"):
    env[name] = ""
  env["PYTHONDONTWRITEBYTECODE"] = "1"

  result = subprocess.run(
    [sys.executable, "-m", "pytest", str(probe),
     "-p", "no:randomly", "-p", "no:cacheprovider", "--showlocals", "-q"],
    cwd=_PROJECT_ROOT, env=env, capture_output=True, text=True, timeout=300,
  )
  return result.stdout + result.stderr


def test_every_constructor_still_refuses_loudly(_failing_constructor_run):
  """The guard on the assertion below.

  "no key in the output" is trivially true of a run that never reached the
  credential -- an import error, a skip, a constructor that succeeded. This
  pins that all four constructors did read a credential and all four refused
  in their own words, so the assertion that follows is made about a run where
  a key really was in play.
  """
  assert "4 failed" in _failing_constructor_run, (
    "the probe run did not fail in the four expected places, so it never "
    f"exercised the credentials:\n{_failing_constructor_run[-3000:]}")
  assert _failing_constructor_run.count("Missing Alpaca") >= 3, (
    "Execution_Agent stopped refusing a half-configured account; that "
    "refusal must not be weakened")
  assert "ALPACA_SECRET in .env" in _failing_constructor_run, (
    "alpaca_server did not name the variable to set; it must refuse rather "
    "than hand a half credential to alpaca-py")


def test_a_failing_constructor_writes_no_key_material(_failing_constructor_run):
  """Issue #59: this is what writes the live broker key to stdout."""
  assert _ALPACA_KEY not in _failing_constructor_run
  assert _ALPACA_SECRET not in _failing_constructor_run
  assert _BODY not in _failing_constructor_run, \
    "part of a credential reached the traceback"
