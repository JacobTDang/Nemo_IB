"""Phase C3: send_brief delivers paginated markdown to Discord.

Mocks the discord_webhook layer so the test makes no network call.
"""
import os
import sys
import types
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _install_fake_discord_module(hook_cls):
  """Ensure `import discord_webhook` works in send_brief, with our fake class."""
  mod = types.ModuleType('discord_webhook')
  mod.DiscordWebhook = hook_cls
  class _DummyEmbed:
    def __init__(self, *a, **kw): pass
    def add_embed_field(self, **kw): pass
  mod.DiscordEmbed = _DummyEmbed
  sys.modules['discord_webhook'] = mod
  return mod


SHORT_BRIEF = """## Pre-Market Brief - 2026-05-20

### Positions
- AAPL: thesis intact, +1.2% premarket on services beat.

### Watchlist
- NVDA: new Blackwell SKU announced overnight.

### Quiet overnight
MSFT, GOOGL, AMZN had no material events.
"""

LONG_BRIEF = ("## Pre-Market Brief - 2026-05-20\n\n"
              + "### Section A\n" + ("Lorem ipsum dolor sit amet. " * 80) + "\n\n"
              + "### Section B\n" + ("Consectetur adipiscing elit. " * 80) + "\n\n"
              + "### Section C\n" + ("Sed do eiusmod tempor. " * 80))


def test_build_brief_chunks_short_brief_single_message():
  from agent.Pre_Market_Briefer import build_brief_chunks
  chunks = build_brief_chunks(SHORT_BRIEF)
  assert len(chunks) == 1, f"short brief should fit in 1 chunk, got {len(chunks)}"
  assert 'AAPL' in chunks[0] and 'NVDA' in chunks[0]
  print(f"PASS: short brief -> 1 chunk ({len(chunks[0])} chars)")


def test_build_brief_chunks_long_brief_paginates():
  from agent.Pre_Market_Briefer import build_brief_chunks
  chunks = build_brief_chunks(LONG_BRIEF)
  assert len(chunks) > 1, f"long brief should paginate, got {len(chunks)} chunks"
  for i, c in enumerate(chunks):
    assert len(c) <= 2000, f"chunk {i} exceeds Discord 2000-char limit: {len(c)}"
  # Round-trip: concatenated chunks should contain all original section markers
  joined = '\n'.join(chunks)
  for marker in ('Section A', 'Section B', 'Section C'):
    assert marker in joined, f"section {marker} lost in pagination"
  print(f"PASS: long brief -> {len(chunks)} chunks, each <=2000 chars")


def test_build_brief_chunks_prefers_section_boundaries():
  from agent.Pre_Market_Briefer import build_brief_chunks
  chunks = build_brief_chunks(LONG_BRIEF)
  # If we have multiple chunks, each chunk after the first should start with
  # a "###" or "##" header — meaning we split on section, not mid-paragraph
  for c in chunks[1:]:
    stripped = c.lstrip()
    starts_with_header = stripped.startswith('##') or stripped.startswith('###')
    assert starts_with_header, f"chunk does not start on a header: {stripped[:60]!r}"
  print(f"PASS: pagination respects section boundaries")


def test_send_brief_no_webhook_returns_false():
  from agent.Pre_Market_Briefer import send_brief
  with patch.dict(os.environ, {'DISCORD_WEBHOOK_URL': ''}, clear=False):
    # Ensure DISCORD_WEBHOOK_URL is empty even after load_dotenv
    with patch('alerts.discord.load_dotenv', lambda: None):
      result = send_brief(SHORT_BRIEF, webhook_url=None)
  assert result is False, "missing webhook should return False, not raise"
  print(f"PASS: missing webhook returns False without raising")


def test_send_brief_posts_each_chunk():
  """Mock DiscordWebhook so we capture every execute() call."""
  posted = []
  class FakeResp:
    status_code = 204
  class FakeHook:
    def __init__(self, *a, **kw):
      self.content = kw.get('content', '')
    def execute(self):
      posted.append(self.content)
      return FakeResp()
    def set_content(self, c):
      self.content = c

  _install_fake_discord_module(FakeHook)
  from agent.Pre_Market_Briefer import send_brief
  result = send_brief(LONG_BRIEF, webhook_url='https://discord.example/webhook/test')
  assert result is True, f"send_brief returned False: {result}"
  assert len(posted) >= 2, f"long brief should post >= 2 messages, got {len(posted)}"
  joined = '\n'.join(posted)
  for marker in ('Section A', 'Section B', 'Section C'):
    assert marker in joined
  print(f"PASS: long brief posted as {len(posted)} discord messages")


def test_send_brief_failure_returns_false():
  class FakeResp:
    status_code = 500
  class FakeHook:
    def __init__(self, *a, **kw):
      self.content = kw.get('content', '')
    def execute(self):
      return FakeResp()
    def set_content(self, c):
      self.content = c

  _install_fake_discord_module(FakeHook)
  from agent.Pre_Market_Briefer import send_brief
  result = send_brief(SHORT_BRIEF, webhook_url='https://discord.example/webhook/test')
  assert result is False, "non-2xx response should yield False"
  print(f"PASS: send failure surfaced as False")


def test_send_brief_handles_empty_string():
  from agent.Pre_Market_Briefer import send_brief
  result = send_brief("", webhook_url='https://x.example')
  # Empty input -> nothing to send -> False (or True with no posts)
  assert result is False
  print(f"PASS: empty brief returns False without crashing")


if __name__ == "__main__":
  test_build_brief_chunks_short_brief_single_message()
  test_build_brief_chunks_long_brief_paginates()
  test_build_brief_chunks_prefers_section_boundaries()
  test_send_brief_no_webhook_returns_false()
  test_send_brief_posts_each_chunk()
  test_send_brief_failure_returns_false()
  test_send_brief_handles_empty_string()
  print("\nAll Phase C3 brief delivery tests passed.")
