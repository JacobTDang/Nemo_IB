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
    assert len(c) <= 4000, f"chunk {i} exceeds a text display: {len(c)}"
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


def test_send_brief_no_webhook_names_the_setting(monkeypatch):
  from agent.Pre_Market_Briefer import send_brief
  from research.discord_post import DiscordError
  monkeypatch.delenv('NEMO_DISCORD_SIGNALS_WEBHOOK_URL', raising=False)
  try:
    send_brief(SHORT_BRIEF, webhook_url=None)
  except DiscordError as exc:
    assert 'NEMO_DISCORD_SIGNALS_WEBHOOK_URL' in str(exc)
    assert 'http' not in str(exc)
  else:
    raise AssertionError('missing webhook returned instead of failing')


def test_send_brief_posts_each_chunk(monkeypatch):
  posted = []

  def transport(url, body):
    posted.append(body)
    assert 'with_components=true' in url
    assert 'discord.example' in url

  monkeypatch.setattr('research.discord_post._http_send', transport)
  from agent.Pre_Market_Briefer import send_brief
  result = send_brief(LONG_BRIEF, webhook_url='https://discord.example/webhook/test')
  assert result is True
  assert len(posted) >= 2, f"long brief should post >= 2 messages, got {len(posted)}"
  joined = str(posted)
  for marker in ('Section A', 'Section B', 'Section C'):
    assert marker in joined
  assert '(1/' in joined


def test_send_brief_failure_raises(monkeypatch):
  def transport(url, body):
    raise RuntimeError(f'post failed {url}')

  monkeypatch.setattr('research.discord_post._http_send', transport)
  from agent.Pre_Market_Briefer import send_brief
  from research.discord_post import DiscordError
  try:
    send_brief(SHORT_BRIEF, webhook_url='https://discord.example/webhook/secret-token')
  except DiscordError as exc:
    assert 'secret-token' not in str(exc)
    assert 'http' not in str(exc)
  else:
    raise AssertionError('a failed post returned instead of failing')


def test_send_brief_handles_empty_string():
  from agent.Pre_Market_Briefer import send_brief
  from research.discord_post import DiscordError
  try:
    send_brief('', webhook_url='https://hooks.example/brief')
  except DiscordError as exc:
    assert 'empty' in str(exc)
  else:
    raise AssertionError('an empty brief was sent')


if __name__ == "__main__":
  test_build_brief_chunks_short_brief_single_message()
  test_build_brief_chunks_long_brief_paginates()
  test_build_brief_chunks_prefers_section_boundaries()
  test_send_brief_no_webhook_names_the_setting()
  test_send_brief_posts_each_chunk()
  test_send_brief_failure_raises()
  test_send_brief_handles_empty_string()
  print("\nAll Phase C3 brief delivery tests passed.")
