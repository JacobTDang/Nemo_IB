"""Phase 2c: Discord alert payload shape (no real webhook calls).

Tests build_thesis_payload, the pure function exposed for testability. Also
tests the unset-webhook graceful-noop path so daemons don't crash without it.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alerts.discord import (
  build_thesis_payload, send_thesis_alert, send_order_alert,
  send_health_alert, SEVERITY_COLORS,
)


def test_thesis_payload_shape_broken():
  p = build_thesis_payload(
    ticker="AAPL", severity="broken",
    headline="Apple posts surprise revenue miss",
    reasoning="Miss + guide-down contradicts growth assumptions",
    action="trigger_reanalysis",
    thesis_id=42
  )
  assert p['title'] == "AAPL: thesis broken"
  assert p['color'] == SEVERITY_COLORS['broken']
  field_names = [f['name'] for f in p['fields']]
  assert 'Event' in field_names
  assert 'Reasoning' in field_names
  assert 'Action' in field_names
  assert 'Thesis' in field_names
  # Find the action field and verify value
  action_field = next(f for f in p['fields'] if f['name'] == 'Action')
  assert action_field['value'] == 'trigger_reanalysis'
  thesis_field = next(f for f in p['fields'] if f['name'] == 'Thesis')
  assert thesis_field['value'] == '#42'
  print("PASS: broken thesis payload has correct title, color, and fields")


def test_thesis_payload_severity_colors_distinct():
  for sev in ('intact', 'weakened', 'broken'):
    p = build_thesis_payload("X", sev, "h", "r", "a")
    assert p['color'] == SEVERITY_COLORS[sev]
  # Confirm the three colors are all different (we want visual distinction)
  assert len({SEVERITY_COLORS[s] for s in ('intact', 'weakened', 'broken')}) == 3
  print("PASS: each severity maps to a distinct color")


def test_thesis_payload_truncation():
  long_reason = "x" * 2000
  long_headline = "y" * 2000
  p = build_thesis_payload("X", "weakened", long_headline, long_reason, "a")
  evt = next(f for f in p['fields'] if f['name'] == 'Event')
  reason = next(f for f in p['fields'] if f['name'] == 'Reasoning')
  assert len(evt['value']) <= 1000, f"event truncated to <=1000, got {len(evt['value'])}"
  assert len(reason['value']) <= 1000
  print(f"PASS: long event ({len(long_headline)}) + reason ({len(long_reason)}) truncated safely")


def test_payload_omits_thesis_when_no_id():
  p = build_thesis_payload("X", "intact", "h", "r", "no_action")
  assert not any(f['name'] == 'Thesis' for f in p['fields'])
  print("PASS: Thesis field omitted when thesis_id=None")


def test_unknown_severity_falls_back_to_info():
  p = build_thesis_payload("X", "FROBNICATED", "h", "r", "a")
  assert p['color'] == SEVERITY_COLORS['info']
  print("PASS: unknown severity falls back to info color")


def test_send_thesis_alert_unset_webhook_returns_false():
  """Without DISCORD_WEBHOOK_URL, send_thesis_alert should log + return False."""
  import os as _os
  prev = _os.environ.pop('DISCORD_WEBHOOK_URL', None)
  try:
    # Also clear what .env loaded
    import alerts.discord as ad
    import importlib
    importlib.reload(ad)
    # Stub _webhook_url to return None for this test
    ad._webhook_url = lambda: None
    ok = ad.send_thesis_alert("X", "broken", "h", "r", "a", thesis_id=1)
    assert ok is False, "should return False when webhook unset"
  finally:
    if prev:
      _os.environ['DISCORD_WEBHOOK_URL'] = prev
  print("PASS: unset webhook returns False (graceful noop)")


def test_send_thesis_alert_with_mock_executes_embed():
  """Replace discord_webhook.DiscordWebhook with a mock and assert the embed
  payload reaches it correctly."""
  from unittest.mock import patch, MagicMock
  import alerts.discord as ad
  ad._webhook_url = lambda: "https://fake.url/hook"

  calls = []

  class FakeHook:
    def __init__(self, url, rate_limit_retry=True):
      calls.append(('init', url))
      self._embeds = []
    def add_embed(self, embed):
      calls.append(('add_embed', embed))
      self._embeds.append(embed)
    def execute(self):
      calls.append(('execute', self._embeds))
      resp = MagicMock(); resp.status_code = 204
      return resp

  class FakeEmbed:
    def __init__(self, title, color):
      self.title = title; self.color = color; self.fields = []
    def add_embed_field(self, name, value, inline=False):
      self.fields.append({'name': name, 'value': value, 'inline': inline})

  with patch.dict('sys.modules', {'discord_webhook': MagicMock(
    DiscordWebhook=FakeHook, DiscordEmbed=FakeEmbed
  )}):
    ok = ad.send_thesis_alert("AAPL", "broken", "Apple miss", "miss + guide", "trigger_reanalysis",
                              thesis_id=42)
  assert ok is True
  assert any(c[0] == 'execute' for c in calls)
  embed_call = next(c for c in calls if c[0] == 'add_embed')
  embed = embed_call[1]
  assert embed.title == "AAPL: thesis broken"
  assert embed.color == SEVERITY_COLORS['broken']
  field_names = [f['name'] for f in embed.fields]
  assert 'Event' in field_names and 'Reasoning' in field_names and 'Thesis' in field_names
  print("PASS: send_thesis_alert produces correct embed for the webhook client")


def test_send_order_alert_mock():
  from unittest.mock import patch, MagicMock
  import alerts.discord as ad
  ad._webhook_url = lambda: "https://fake.url/hook"

  class FakeHook:
    def __init__(self, url, rate_limit_retry=True):
      self._embeds = []
    def add_embed(self, e): self._embeds.append(e)
    def execute(self):
      r = MagicMock(); r.status_code = 200; return r
  class FakeEmbed:
    def __init__(self, title, color):
      self.title = title; self.color = color; self.fields = []
    def add_embed_field(self, name, value, inline=False):
      self.fields.append({'name': name, 'value': value, 'inline': inline})

  with patch.dict('sys.modules', {'discord_webhook': MagicMock(
    DiscordWebhook=FakeHook, DiscordEmbed=FakeEmbed
  )}):
    ok = ad.send_order_alert("AAPL", "buy", 10, "market", None, True, thesis_id=42, order_id="abc")
  assert ok
  print("PASS: send_order_alert handles paper-order embed")


if __name__ == "__main__":
  test_thesis_payload_shape_broken()
  test_thesis_payload_severity_colors_distinct()
  test_thesis_payload_truncation()
  test_payload_omits_thesis_when_no_id()
  test_unknown_severity_falls_back_to_info()
  test_send_thesis_alert_unset_webhook_returns_false()
  test_send_thesis_alert_with_mock_executes_embed()
  test_send_order_alert_mock()
  print("\nAll Phase 2c discord alert tests passed.")
