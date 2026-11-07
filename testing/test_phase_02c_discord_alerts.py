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


def _body(payload):
  parts = []
  for node in payload['components'][0]['components']:
    if node.get('type') == 10:
      parts.append(node['content'])
  return '\n'.join(parts)


def test_thesis_payload_shape_broken():
  p = build_thesis_payload(
    ticker="AAPL", severity="broken",
    headline="Apple posts surprise revenue miss",
    reasoning="Miss + guide-down contradicts growth assumptions",
    action="trigger_reanalysis",
    thesis_id=42
  )
  assert p['components'][0]['accent_color'] == SEVERITY_COLORS['broken']
  body = _body(p)
  assert 'AAPL' in body and 'broken' in body
  assert 'trigger_reanalysis' in body.replace('\\', '')
  assert '#42' in body
  assert 'content' not in p and 'embeds' not in p
  print("PASS: broken thesis payload is a Components V2 container")


def test_thesis_payload_severity_colors_distinct():
  colors = set()
  for sev in ('intact', 'weakened', 'broken'):
    p = build_thesis_payload("X", sev, "h", "r", "a")
    color = p['components'][0]['accent_color']
    assert color == SEVERITY_COLORS[sev]
    colors.add(color)
  assert len(colors) == 3
  print("PASS: each severity maps to a distinct color")


def test_thesis_payload_truncation():
  long_reason = "x" * 2000
  long_headline = "y" * 2000
  p = build_thesis_payload("X", "weakened", long_headline, long_reason, "a")
  body = _body(p)
  assert len(body) <= 4000
  print("PASS: long event and reason stay inside one text budget")


def test_payload_omits_thesis_when_no_id():
  p = build_thesis_payload("X", "intact", "h", "r", "no_action")
  assert '#None' not in _body(p)
  assert 'Thesis' not in _body(p)
  print("PASS: Thesis line omitted when thesis_id=None")


def test_unknown_severity_raises_instead_of_guessing_a_color():
  import pytest
  from research.discord_post import DiscordError
  with pytest.raises(DiscordError) as caught:
    build_thesis_payload("X", "FROBNICATED", "h", "r", "a")
  assert 'FROBNICATED' in str(caught.value)
  print("PASS: unknown severity raises")


def test_every_known_severity_has_its_own_emoji_and_color():
  for severity in SEVERITY_COLORS:
    p = build_thesis_payload("X", severity, "h", "r", "a")
    assert p['components'][0]['accent_color'] == SEVERITY_COLORS[severity]
  assert '❌' in _body(build_thesis_payload("X", "error", "h", "r", "a"))


def test_send_thesis_alert_unset_webhook_names_the_setting(monkeypatch):
  """A missing webhook is a visible failure, not a silent False."""
  import alerts.discord as ad
  from research.discord_post import DiscordError
  monkeypatch.delenv('NEMO_DISCORD_SIGNALS_WEBHOOK_URL', raising=False)
  try:
    ad.send_thesis_alert("X", "broken", "h", "r", "a", thesis_id=1)
  except DiscordError as exc:
    assert 'NEMO_DISCORD_SIGNALS_WEBHOOK_URL' in str(exc)
    assert 'http' not in str(exc)
  else:
    raise AssertionError('missing webhook returned instead of failing')


def test_send_thesis_alert_posts_components(monkeypatch):
  import alerts.discord as ad
  posted = []
  monkeypatch.setattr(ad, 'deliver', lambda channel, payloads: posted.append((channel, payloads)))
  ok = ad.send_thesis_alert(
      "AAPL", "broken", "Apple miss", "miss + guide", "trigger_reanalysis",
      thesis_id=42)
  assert ok is True
  channel, payloads = posted[0]
  assert channel == 'signals'
  body = _body(payloads[0])
  assert 'AAPL' in body and 'Thesis' in body and '#42' in body


def test_send_order_alert_mock(monkeypatch):
  import alerts.discord as ad
  posted = []
  monkeypatch.setattr(ad, 'deliver', lambda channel, payloads: posted.append(channel))
  ok = ad.send_order_alert("AAPL", "buy", 10, "market", None, True, thesis_id=42, order_id="abc")
  assert ok
  assert posted == ['signals']


if __name__ == "__main__":
  test_thesis_payload_shape_broken()
  test_thesis_payload_severity_colors_distinct()
  test_thesis_payload_truncation()
  test_payload_omits_thesis_when_no_id()
  test_unknown_severity_raises_instead_of_guessing_a_color()
  test_every_known_severity_has_its_own_emoji_and_color()
  test_send_thesis_alert_unset_webhook_names_the_setting()
  test_send_thesis_alert_posts_components()
  test_send_order_alert_mock()
  print("\nAll Phase 2c discord alert tests passed.")
