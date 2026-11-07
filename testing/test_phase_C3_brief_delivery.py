"""Phase C3: send_brief delivers paginated markdown to Discord.

The HTTP call is replaced, so the test makes no network call.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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


def _details(pages):
  """The brief text of each page: the text displays between headline and footer."""
  out = []
  for payload in pages:
    texts = [n['content'] for n in payload['components'][0]['components']
             if n.get('type') == 10]
    out.append('\n'.join(texts[1:-1]))
  return out


def test_short_brief_is_a_single_message():
  from research.discord_post import brief_messages
  pages = brief_messages(SHORT_BRIEF)
  assert len(pages) == 1, f"short brief should fit in 1 message, got {len(pages)}"
  detail = _details(pages)[0]
  assert 'AAPL' in detail and 'NVDA' in detail
  assert '(1/' not in str(pages)


def test_long_brief_paginates_and_every_page_fits_a_text_display():
  from research.discord_post import brief_messages
  pages = brief_messages(LONG_BRIEF)
  assert len(pages) > 1, f"long brief should paginate, got {len(pages)} messages"
  for i, detail in enumerate(_details(pages)):
    assert len(detail) <= 4000, f"page {i} exceeds a text display: {len(detail)}"
  joined = '\n'.join(_details(pages))
  for marker in ('Section A', 'Section B', 'Section C'):
    assert marker in joined, f"section {marker} lost in pagination"


def test_pagination_prefers_section_boundaries():
  from research.discord_post import brief_messages
  for detail in _details(brief_messages(LONG_BRIEF))[1:]:
    assert detail.lstrip().startswith('##'), (
      f"page does not start on a header: {detail.lstrip()[:60]!r}")


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
  test_short_brief_is_a_single_message()
  test_long_brief_paginates_and_every_page_fits_a_text_display()
  test_pagination_prefers_section_boundaries()
  test_send_brief_no_webhook_names_the_setting()
  test_send_brief_posts_each_chunk()
  test_send_brief_failure_raises()
  test_send_brief_handles_empty_string()
  print("\nAll Phase C3 brief delivery tests passed.")
