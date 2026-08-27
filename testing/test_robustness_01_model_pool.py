"""Item 1: rotating reasoning model pool with LRU pick and 429 demotion."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent.openrouter_template as ort
from agent.openrouter_template import (
  _pick_next_model, _demote_model,
)

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



def test_pool_initialized():
  _primary_model()          # skips when the credential was rejected
  assert len(ort._MODEL_POOL) >= 1, "pool must have at least one model"
  assert _primary_model() == ort._MODEL_POOL[0]
  print(f"PASS: pool has {len(ort._MODEL_POOL)} models, primary = {_primary_model()!r}")


def test_lru_round_robin():
  """With N alive models, calling _pick_next_model N times picks each once."""
  if len(ort._MODEL_POOL) < 2:
    print(f"SKIP: pool only has {len(ort._MODEL_POOL)} model — round-robin needs 2+")
    return
  # Clear state and force a known small pool
  saved_pool = ort._MODEL_POOL
  saved_lru = dict(ort._MODEL_LAST_USED)
  saved_demote = dict(ort._MODEL_DEMOTED_UNTIL)
  try:
    ort._MODEL_POOL = ['model_a', 'model_b', 'model_c']
    ort._MODEL_LAST_USED = {}
    ort._MODEL_DEMOTED_UNTIL = {}
    picks = [_pick_next_model() for _ in range(3)]
    assert set(picks) == {'model_a', 'model_b', 'model_c'}, f"got {picks}"
    print(f"PASS: round-robin picked all 3 distinct models: {picks}")
  finally:
    ort._MODEL_POOL = saved_pool
    ort._MODEL_LAST_USED = saved_lru
    ort._MODEL_DEMOTED_UNTIL = saved_demote


def test_demote_skips_until_expiry():
  saved_pool = ort._MODEL_POOL
  saved_lru = dict(ort._MODEL_LAST_USED)
  saved_demote = dict(ort._MODEL_DEMOTED_UNTIL)
  try:
    ort._MODEL_POOL = ['model_a', 'model_b']
    ort._MODEL_LAST_USED = {}
    ort._MODEL_DEMOTED_UNTIL = {}
    _demote_model('model_a', seconds=60)
    picks = [_pick_next_model() for _ in range(3)]
    assert all(p == 'model_b' for p in picks), \
      f"demoted model_a but it was picked: {picks}"
    print("PASS: demoted model is skipped until expiry")
  finally:
    ort._MODEL_POOL = saved_pool
    ort._MODEL_LAST_USED = saved_lru
    ort._MODEL_DEMOTED_UNTIL = saved_demote


def test_all_demoted_picks_least_demoted():
  """When every model is demoted, fall back to the one banning soonest."""
  saved_pool = ort._MODEL_POOL
  saved_lru = dict(ort._MODEL_LAST_USED)
  saved_demote = dict(ort._MODEL_DEMOTED_UNTIL)
  try:
    ort._MODEL_POOL = ['model_a', 'model_b']
    ort._MODEL_LAST_USED = {}
    ort._MODEL_DEMOTED_UNTIL = {}
    _demote_model('model_a', seconds=10)   # expires sooner
    _demote_model('model_b', seconds=100)  # expires later
    pick = _pick_next_model()
    assert pick == 'model_a', f"expected least-demoted model_a, got {pick}"
    print("PASS: when all demoted, picks the one demoting soonest")
  finally:
    ort._MODEL_POOL = saved_pool
    ort._MODEL_LAST_USED = saved_lru
    ort._MODEL_DEMOTED_UNTIL = saved_demote


def test_backcompat_primary_reasoning_model_constant():
  """Existing constructor defaults import PRIMARY_REASONING_MODEL — must still work."""
  from agent.openrouter_template import PRIMARY_REASONING_MODEL as prm
  assert isinstance(prm, str) and len(prm) > 0
  assert ':' in prm or '/' in prm, f"unexpected model id format: {prm!r}"
  print(f"PASS: PRIMARY_REASONING_MODEL constant intact = {prm!r}")


if __name__ == "__main__":
  test_pool_initialized()
  test_lru_round_robin()
  test_demote_skips_until_expiry()
  test_all_demoted_picks_least_demoted()
  test_backcompat_primary_reasoning_model_constant()
  print("\nAll tests passed.")
