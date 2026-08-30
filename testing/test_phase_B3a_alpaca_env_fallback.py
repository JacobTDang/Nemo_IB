"""Phase B3a: AsyncBroker accepts legacy ALPACA_API_KEY/SECRET as fallback.

Triggered by the B3 pilot finding: the project `.env` uses the legacy
ALPACA_API_KEY / ALPACA_SECRET names (Alpaca's documented defaults), while
AsyncBroker only reads ALPACA_PAPER_KEY / ALPACA_PAPER_SECRET. This caused
the reconcile-halt to trip in the B3 pilot. The fallback makes the broker
accept either name pair so the next pilot can exercise the trade path.

The fix is paper-side only — live trading is intentionally unsupported and
must continue to require explicit ALPACA_LIVE_KEY / ALPACA_LIVE_SECRET.

The credentials are `Secret` rather than strings since issue #64, so the
assertions read them through `.reveal()`. What each test proves is unchanged:
every one still pins *which* environment variable the broker resolved to, by
comparing the resolved value against a sentinel only one of the names carries.
Asserting merely that a `Secret` arrived would pass with the precedence
inverted, which is the one thing this file exists to catch — so the values are
still compared, and the sentinels below are synthetic strings written here
rather than credentials of any kind.
"""
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


_PAPER_KEY = 'paper0000sentinel0000key'
_PAPER_SECRET = 'paper0000sentinel0000secret'
_LEGACY_KEY = 'legacy0000sentinel0000key'
_LEGACY_SECRET = 'legacy0000sentinel0000secret'


def _env(**vals):
  """Return a dict with the given keys set, all four ALPACA names cleared first."""
  base = {k: '' for k in (
    'ALPACA_PAPER_KEY', 'ALPACA_PAPER_SECRET',
    'ALPACA_API_KEY', 'ALPACA_SECRET',
    'ALPACA_LIVE_KEY', 'ALPACA_LIVE_SECRET',
  )}
  base.update(vals)
  return base


def test_paper_with_preferred_names():
  from tools.alpaca.async_broker import AsyncBroker
  env = _env(ALPACA_PAPER_KEY=_PAPER_KEY, ALPACA_PAPER_SECRET=_PAPER_SECRET)
  with patch.dict(os.environ, env, clear=False), \
       patch('tools.alpaca.async_broker.load_dotenv', lambda: None):
    b = AsyncBroker(paper=True)
    assert b._key.reveal() == _PAPER_KEY and b._secret.reveal() == _PAPER_SECRET
  print("PASS: preferred ALPACA_PAPER_* names still work")


def test_paper_falls_back_to_legacy_names():
  """Pilot scenario: .env has ALPACA_API_KEY/SECRET but no ALPACA_PAPER_*."""
  from tools.alpaca.async_broker import AsyncBroker
  env = _env(ALPACA_API_KEY=_LEGACY_KEY, ALPACA_SECRET=_LEGACY_SECRET)
  with patch.dict(os.environ, env, clear=False), \
       patch('tools.alpaca.async_broker.load_dotenv', lambda: None):
    b = AsyncBroker(paper=True)
    # The failure message names the variable the broker *should* have read,
    # not the value it did read: a diagnostic that prints the resolved
    # credential is the disclosure issue #64 closed.
    assert b._key.reveal() == _LEGACY_KEY and b._secret.reveal() == _LEGACY_SECRET, \
      "legacy names not picked up: the broker resolved neither ALPACA_API_KEY " \
      "nor ALPACA_SECRET"
  print("PASS: legacy ALPACA_API_KEY/SECRET picked up when PAPER_* absent")


def test_preferred_wins_over_legacy_when_both_set():
  from tools.alpaca.async_broker import AsyncBroker
  env = _env(
    ALPACA_PAPER_KEY=_PAPER_KEY, ALPACA_PAPER_SECRET=_PAPER_SECRET,
    ALPACA_API_KEY=_LEGACY_KEY,  ALPACA_SECRET=_LEGACY_SECRET,
  )
  with patch.dict(os.environ, env, clear=False), \
       patch('tools.alpaca.async_broker.load_dotenv', lambda: None):
    b = AsyncBroker(paper=True)
    # Both pairs are set and the two sentinels differ, so this comparison is
    # the whole point of the file: it fails if the fallback ever wins.
    assert b._key.reveal() == _PAPER_KEY and b._secret.reveal() == _PAPER_SECRET, \
      "PAPER_* should win, but the broker resolved the legacy ALPACA_API_KEY pair"
    assert b._key.reveal() != _LEGACY_KEY and b._secret.reveal() != _LEGACY_SECRET
  print("PASS: ALPACA_PAPER_* takes priority over legacy when both set")


def test_paper_still_raises_when_nothing_set():
  from tools.alpaca.async_broker import AsyncBroker
  env = _env()  # all six cleared
  with patch.dict(os.environ, env, clear=False), \
       patch('tools.alpaca.async_broker.load_dotenv', lambda: None):
    try:
      AsyncBroker(paper=True)
    except RuntimeError as e:
      assert 'ALPACA' in str(e) and 'PAPER' in str(e), f"error msg unhelpful: {e}"
      print(f"PASS: missing both pairs still raises with helpful msg")
      return
  raise AssertionError("expected RuntimeError when no Alpaca creds set")


def test_live_does_NOT_fall_back_to_legacy():
  """Live trading must require explicit LIVE_* names. ALPACA_API_KEY in the
  env should NEVER auto-enable live trading — that would be a silent footgun."""
  from tools.alpaca.async_broker import AsyncBroker
  env = _env(ALPACA_API_KEY=_LEGACY_KEY, ALPACA_SECRET=_LEGACY_SECRET)  # only legacy paper-ish
  with patch.dict(os.environ, env, clear=False), \
       patch('tools.alpaca.async_broker.load_dotenv', lambda: None):
    try:
      AsyncBroker(paper=False)
    except RuntimeError as e:
      assert 'LIVE' in str(e)
      print("PASS: live mode does NOT fall back to legacy names")
      return
  raise AssertionError("live mode silently picked up legacy names — safety violation")


if __name__ == "__main__":
  test_paper_with_preferred_names()
  test_paper_falls_back_to_legacy_names()
  test_preferred_wins_over_legacy_when_both_set()
  test_paper_still_raises_when_nothing_set()
  test_live_does_NOT_fall_back_to_legacy()
  print("\nAll Phase B3a env-fallback tests passed.")
