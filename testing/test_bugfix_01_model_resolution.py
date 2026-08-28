"""Bug 1: PRIMARY_REASONING_MODEL resolves to a live OpenRouter endpoint.

Requires OPENROUTER_API_KEY in the environment. The previous default
'deepseek/deepseek-r1-0528:free' was retired -- this test confirms the
replacement is alive AND confirms the old one is still dead (sanity check).
"""
import os, sys

import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.openrouter_template import _verify_model_alive

def _primary_model():
    """Resolved inside the test, not at import.

    `PRIMARY_REASONING_MODEL` is a module attribute backed by a live probe, so
    a module-level from-import made collecting this file cost a network round
    trip -- and, once a rejected credential started refusing instead of
    reporting every model alive, made the whole suite uncollectable over a key
    that almost no test needs.
    """
    import pytest

    from agent.openrouter_template import CredentialRejected, primary_reasoning_model
    try:
        return primary_reasoning_model()
    except CredentialRejected as exc:
        pytest.skip(f"OpenRouter credential rejected: {exc}")

from testing._gates import requires_openrouter
from dotenv import load_dotenv
load_dotenv()


def test_a_model_is_resolved():
  """The one half of this file that needs no network."""
  assert _primary_model(), "no model resolved"


# A well-formed id no vendor will ever publish. _verify_model_alive returns
# False without a network call for a *malformed* id, so this has to look real:
# `vendor/model:tag`, matching _MODEL_ID_RE.
_CONTROL_DEAD_MODEL = "nemo-control/definitely-not-a-model:free"


def _probe_can_tell_dead_from_alive(api_key: str) -> bool:
  """Does _verify_model_alive discriminate on this machine right now?

  It answers True for any non-404 -- a 401, a rate limit, a timeout -- because
  none of those prove a model is dead. That is the right rule for deciding
  pool membership and a useless one for a test, since a rejected API key makes
  every model on earth read as alive. Measured on this machine: OPENROUTER_API_KEY
  is set but OpenRouter answers 401 Missing Authentication header, so every
  probe returned True and both live tests below were green without ever
  reaching a real verdict.

  So probe a control id that cannot exist. If even that reads as alive, the
  probe is telling us nothing and the tests below skip saying so, rather than
  passing on a 401 or failing on one.
  """
  return not _verify_model_alive(_CONTROL_DEAD_MODEL, api_key)


@requires_openrouter
def test_resolved_model_is_alive():
  """Live OpenRouter probe. Gated, because offline it proves nothing.

  _verify_model_alive treats any non-404 exception as "alive" on purpose --
  a rate limit or a timeout does not prove a model is dead. Under
  SKIP_NETWORK_TESTS=1 the conftest socket guard raises, that guard catches
  it, and the probe returns True. So this assertion passed on every offline
  run without a single packet leaving the machine. requires_openrouter skips
  it offline and, under NEMO_REQUIRE_SERVICES=1, fails instead of skipping.
  """
  api_key = os.getenv("OPENROUTER_API_KEY")
  assert api_key, "requires_openrouter let through a run with no key"
  if not _probe_can_tell_dead_from_alive(api_key):
    pytest.skip(f"the liveness probe reports {_CONTROL_DEAD_MODEL} as alive, "
                "so it is not answering 404 for anything -- most likely the "
                "OpenRouter key is being rejected. Nothing here is decidable.")
  # Resolved once, so the assertion and the message name the same model. The
  # module-level import these two lines used to read was removed on purpose --
  # see `_primary_model` -- and they were missed, so this raised NameError
  # rather than checking anything. Being network-gated, it skipped instead of
  # failing, for as long as the name was wrong.
  model = _primary_model()
  assert _verify_model_alive(model, api_key), \
    f"resolved model {model} is not alive (404)"
  print(f"PASS: resolved model {model!r} is alive")


@requires_openrouter
def test_old_dead_model_still_dead():
  """If the R1 endpoint comes back, we want to know -- so this has to fail.

  It used to print "NOTE: ... is alive again" down either branch and return
  None, which is not a way of being told anything: pytest reported it green
  whichever answer came back. And offline it printed the "alive again" note
  every run, because the probe's non-404-is-alive rule turns a refused
  connection into True.

  A failure here is good news, not a defect: the retired free R1 endpoint is
  answering again and belongs back in the candidate list in
  agent/openrouter_template.py. Delete this test when you re-add it.
  """
  api_key = os.getenv("OPENROUTER_API_KEY")
  assert api_key, "requires_openrouter let through a run with no key"
  if not _probe_can_tell_dead_from_alive(api_key):
    pytest.skip(f"the liveness probe reports {_CONTROL_DEAD_MODEL} as alive, "
                "so it is not answering 404 for anything -- most likely the "
                "OpenRouter key is being rejected. Nothing here is decidable.")
  alive = _verify_model_alive("deepseek/deepseek-r1-0528:free", api_key)
  assert not alive, (
    "deepseek/deepseek-r1-0528:free answers again -- this is good news. "
    "Consider re-adding it to the candidate list in "
    "agent/openrouter_template.py, then delete this test.")


def test_constructor_default_uses_resolved_model():
  from agent.Financial_Analysis_Agent import Financial_Analysis_Agent
  expected = _primary_model()   # skips when the credential was rejected
  agent = Financial_Analysis_Agent()
  assert agent.model_name == expected, \
    f"agent uses {agent.model_name!r}, expected {_primary_model()!r}"
  print(f"PASS: Financial_Analysis_Agent picks up {agent.model_name!r}")


def test_explicit_override_wins():
  """Passing a model_name explicitly should override the auto-resolved one."""
  from agent.Financial_Analysis_Agent import Financial_Analysis_Agent
  agent = Financial_Analysis_Agent(model_name='z-ai/glm-4.5-air:free')
  assert agent.model_name == 'z-ai/glm-4.5-air:free'
  print("PASS: explicit model_name override respected")


if __name__ == "__main__":
  test_resolved_model_is_alive()
  test_a_model_is_resolved()
  test_resolved_model_is_alive()
  test_old_dead_model_still_dead()
  test_constructor_default_uses_resolved_model()
  test_explicit_override_wins()
  print("\nAll tests passed.")
