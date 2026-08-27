"""Model-id validation for the OpenRouter reasoning pool.

Regression cover for the malformed-override defect: `.env.example` shipped
`PRIMARY_REASONING_MODEL=  # optional override; ...` on one line, so dotenv read
the trailing comment as the VALUE. That garbage string was pushed to position 0
of the pool and `_verify_model_alive` swallowed the resulting error, declaring it
"alive". Every OpenRouterModel built without an explicit model name then defaulted
to the comment text.
"""
import os
import sys

import httpx
import pytest
from openai import AuthenticationError, NotFoundError, RateLimitError

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent.openrouter_template as ort

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _http_error(exc_cls, status: int):
  """Build a real openai exception instance for the given status code."""
  response = httpx.Response(status, request=httpx.Request("POST", "https://openrouter.ai"))
  return exc_cls("simulated", response=response, body=None)


class _ExplodingOpenAI:
  """Stand-in for openai.OpenAI that fails the test if it is constructed."""

  def __init__(self, *args, **kwargs):
    raise AssertionError(
      "_verify_model_alive built a network client for a malformed model id"
    )


def _client_raising(exc):
  """Return a fake OpenAI class whose completions call raises `exc`."""

  class _FakeCompletions:
    def create(self, **kwargs):
      raise exc

  class _FakeChat:
    completions = _FakeCompletions()

  class _FakeOpenAI:
    def __init__(self, *args, **kwargs):
      self.chat = _FakeChat()

  return _FakeOpenAI


# ---------------------------------------------------------------------------
# _is_valid_model_id
# ---------------------------------------------------------------------------

MALFORMED_IDS = [
  None,
  "",
  "   ",
  "\t\n",
  "# optional override; if unset, pool auto-resolves",
  "  # leading whitespace then comment",
  "deepseek-chat-v3.1",          # no vendor separator
  "no-slash-at-all:free",        # tag but still no vendor
  "/leading-slash",              # empty vendor
  "vendor/",                     # empty model
  "vendor/model:",               # empty tag
]


@pytest.mark.parametrize("bad", MALFORMED_IDS)
def test_malformed_model_ids_rejected(bad):
  assert ort._is_valid_model_id(bad) is False, f"should have rejected {bad!r}"


WELL_FORMED_IDS = [
  "z-ai/glm-4.5-air:free",
  "deepseek/deepseek-chat-v3.1:free",
  "deepseek/deepseek-r1-distill-llama-70b:free",
  "qwen/qwq-32b-preview:free",
  "meta-llama/llama-3.3-70b-instruct:free",
  "openai/gpt-4o",               # no tag is fine
  "anthropic/claude-3.5-sonnet",
]


@pytest.mark.parametrize("good", WELL_FORMED_IDS)
def test_well_formed_model_ids_accepted(good):
  assert ort._is_valid_model_id(good) is True, f"should have accepted {good!r}"


def test_every_hardcoded_pool_candidate_is_valid():
  """The built-in fallbacks must all pass their own validator."""
  for candidate in WELL_FORMED_IDS[:5]:
    assert ort._is_valid_model_id(candidate) is True


# ---------------------------------------------------------------------------
# _verify_model_alive
# ---------------------------------------------------------------------------

def test_malformed_id_is_not_alive_and_makes_no_network_call(monkeypatch):
  """A malformed id must be rejected before any client is constructed."""
  monkeypatch.setattr(ort, "OpenAI", _ExplodingOpenAI)
  bad = "# optional override; if unset, pool auto-resolves"
  assert ort._verify_model_alive(bad, "fake-key") is False


@pytest.mark.parametrize("bad", MALFORMED_IDS)
def test_no_malformed_id_reaches_the_network(monkeypatch, bad):
  monkeypatch.setattr(ort, "OpenAI", _ExplodingOpenAI)
  assert ort._verify_model_alive(bad, "fake-key") is False


def test_not_found_means_dead(monkeypatch):
  """Existing behaviour: an explicit 404 proves the model does not exist."""
  monkeypatch.setattr(
    ort, "OpenAI", _client_raising(_http_error(NotFoundError, 404))
  )
  assert ort._verify_model_alive("vendor/model:free", "fake-key") is False


@pytest.mark.parametrize(
  "exc",
  [_http_error(RateLimitError, 429)],
  ids=["rate_limit"],
)
def test_transient_errors_still_count_as_alive(monkeypatch, exc):
  """A 429 says the model is known and busy, not absent — keep it in the pool."""
  monkeypatch.setattr(ort, "OpenAI", _client_raising(exc))
  assert ort._verify_model_alive("vendor/model:free", "fake-key") is True


def test_a_rejected_credential_is_not_a_live_model(monkeypatch):
  """401 used to be grouped with 429 as "says nothing about the model". It is
  not the same: a rate limit is a fact about the model, while a rejected key
  means the probe never reached one. Grouped together, a bad key marked every
  candidate alive and the pool was assembled entirely out of 401s -- observed
  live, a rejected key still logged "Pool initialized with 5 models" without
  one of them having been reached."""
  monkeypatch.setattr(
    ort, "OpenAI", _client_raising(_http_error(AuthenticationError, 401))
  )
  with pytest.raises(ort.CredentialRejected):
    ort._verify_model_alive("vendor/model:free", "fake-key")


def test_successful_ping_is_alive(monkeypatch):
  class _OkCompletions:
    def create(self, **kwargs):
      return object()

  class _OkChat:
    completions = _OkCompletions()

  class _OkOpenAI:
    def __init__(self, *args, **kwargs):
      self.chat = _OkChat()

  monkeypatch.setattr(ort, "OpenAI", _OkOpenAI)
  assert ort._verify_model_alive("vendor/model:free", "fake-key") is True


# ---------------------------------------------------------------------------
# Pool construction
# ---------------------------------------------------------------------------

def test_pool_drops_a_malformed_override(monkeypatch):
  """A garbage PRIMARY_REASONING_MODEL must never reach position 0."""
  monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")
  monkeypatch.setenv(
    "PRIMARY_REASONING_MODEL", "# optional override; if unset, pool auto-resolves"
  )
  monkeypatch.setattr(ort, "load_dotenv", lambda *a, **k: None)
  monkeypatch.setattr(ort, "_verify_model_alive", lambda m, k, **kw: ort._is_valid_model_id(m))

  pool = ort._build_reasoning_pool()
  assert pool, "pool must never be empty"
  assert all(ort._is_valid_model_id(m) for m in pool), f"pool holds junk: {pool}"
  assert not pool[0].startswith("#")


def test_pool_honours_a_well_formed_override(monkeypatch):
  monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")
  monkeypatch.setenv("PRIMARY_REASONING_MODEL", "vendor/custom-model:free")
  monkeypatch.setattr(ort, "load_dotenv", lambda *a, **k: None)
  monkeypatch.setattr(ort, "_verify_model_alive", lambda m, k, **kw: ort._is_valid_model_id(m))

  pool = ort._build_reasoning_pool()
  assert pool[0] == "vendor/custom-model:free"


def test_module_primary_reasoning_model_is_well_formed():
  """The import-time constant that constructor defaults use must be a real id."""
  assert ort._is_valid_model_id(ort.PRIMARY_REASONING_MODEL), \
    f"PRIMARY_REASONING_MODEL is malformed: {ort.PRIMARY_REASONING_MODEL!r}"


# ---------------------------------------------------------------------------
# .env.example hygiene
# ---------------------------------------------------------------------------

def test_env_example_ships_no_comment_as_a_value():
  """No .env.example key may parse to a value that is really a comment.

  python-dotenv strips a trailing `# ...` only when a real value precedes it
  (`FOO=true  # note` parses to "true"). When the value is EMPTY the comment
  becomes the whole value -- that is the defect this guards.
  Parsed with dotenv itself so the test tracks the real parser, not a guess.
  """
  from dotenv import dotenv_values

  env_example = os.path.join(REPO_ROOT, ".env.example")
  assert os.path.exists(env_example), ".env.example is missing"

  parsed = dotenv_values(env_example)
  offenders = {
    key: value for key, value in parsed.items()
    if value is not None and value.strip().startswith("#")
  }
  assert not offenders, (
    "these .env.example keys parse to a comment instead of a value "
    "(move the comment to its own line): " + repr(offenders)
  )


def test_env_example_primary_reasoning_model_is_empty():
  env_example = os.path.join(REPO_ROOT, ".env.example")
  with open(env_example, "r", encoding="utf-8") as fh:
    for raw in fh:
      if raw.strip().startswith("PRIMARY_REASONING_MODEL="):
        _, _, value = raw.strip().partition("=")
        assert value.strip() == "", \
          f"PRIMARY_REASONING_MODEL ships a non-empty default: {value!r}"
        return
  pytest.fail("PRIMARY_REASONING_MODEL not found in .env.example")


# --------------------------------------------------------------------------
# Import-time network access.
#
# _build_reasoning_pool() ran at module import, so merely importing
# openrouter_template pinged OpenRouter five times and cost ~0.7s. That makes
# an "offline" test run not strictly offline, and it charges a caller who only
# wanted the module for a type or a helper.
# --------------------------------------------------------------------------

def test_importing_the_module_makes_no_network_call():
    """Importing must be free. Resolving a model is what costs."""
    import subprocess
    import sys

    probe = (
        "import sys\n"
        "class Boom:\n"
        "    def __init__(self, *a, **k):\n"
        "        raise AssertionError('network client built at import')\n"
        "import openai\n"
        "openai.OpenAI = Boom\n"
        "import agent.openrouter_template\n"
        "print('imported clean')\n"
    )
    out = subprocess.run([sys.executable, "-c", probe],
                         capture_output=True, text=True, cwd=".")
    assert "imported clean" in out.stdout, out.stderr[-600:]


def test_the_constant_still_resolves_when_read():
    """Consumers do `from openrouter_template import PRIMARY_REASONING_MODEL`.
    Laziness must not break that -- reading it builds the pool on demand."""
    import agent.openrouter_template as ort
    value = ort.PRIMARY_REASONING_MODEL
    assert isinstance(value, str) and "/" in value


def test_unknown_attribute_still_raises():
    import agent.openrouter_template as ort
    with pytest.raises(AttributeError):
        ort.NoSuchAttribute
