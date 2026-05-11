"""Bug 1: PRIMARY_REASONING_MODEL resolves to a live OpenRouter endpoint.

Requires OPENROUTER_API_KEY in the environment. The previous default
'deepseek/deepseek-r1-0528:free' was retired -- this test confirms the
replacement is alive AND confirms the old one is still dead (sanity check).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.openrouter_template import PRIMARY_REASONING_MODEL, _verify_model_alive
from dotenv import load_dotenv
load_dotenv()


def test_resolved_model_is_alive():
  assert PRIMARY_REASONING_MODEL, "no model resolved"
  api_key = os.getenv("OPENROUTER_API_KEY")
  if not api_key:
    print("SKIP: OPENROUTER_API_KEY not set")
    return
  assert _verify_model_alive(PRIMARY_REASONING_MODEL, api_key), \
    f"resolved model {PRIMARY_REASONING_MODEL} is not alive (404)"
  print(f"PASS: resolved model {PRIMARY_REASONING_MODEL!r} is alive")


def test_old_dead_model_still_dead():
  """If the R1 endpoint comes back, we want to know."""
  api_key = os.getenv("OPENROUTER_API_KEY")
  if not api_key:
    print("SKIP: OPENROUTER_API_KEY not set")
    return
  alive = _verify_model_alive("deepseek/deepseek-r1-0528:free", api_key)
  if alive:
    print(f"NOTE: deepseek/deepseek-r1-0528:free is alive again -- "
          "consider re-adding to candidate list")
  else:
    print("PASS: dead endpoint confirmed still dead")


def test_constructor_default_uses_resolved_model():
  from agent.Financial_Analysis_Agent import Financial_Analysis_Agent
  agent = Financial_Analysis_Agent()
  assert agent.model_name == PRIMARY_REASONING_MODEL, \
    f"agent uses {agent.model_name!r}, expected {PRIMARY_REASONING_MODEL!r}"
  print(f"PASS: Financial_Analysis_Agent picks up {agent.model_name!r}")


def test_explicit_override_wins():
  """Passing a model_name explicitly should override the auto-resolved one."""
  from agent.Financial_Analysis_Agent import Financial_Analysis_Agent
  agent = Financial_Analysis_Agent(model_name='z-ai/glm-4.5-air:free')
  assert agent.model_name == 'z-ai/glm-4.5-air:free'
  print("PASS: explicit model_name override respected")


if __name__ == "__main__":
  test_resolved_model_is_alive()
  test_old_dead_model_still_dead()
  test_constructor_default_uses_resolved_model()
  test_explicit_override_wins()
  print("\nAll tests passed.")
