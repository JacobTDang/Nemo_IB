"""Discord webhook alerts. Used by Phase 2 (thesis status) and Phase 6 (orders).

Webhook URL is read from DISCORD_WEBHOOK_URL env. If unset, send_alert silently
no-ops with a log line — daemons don't crash for want of a webhook.
"""
import os
import sys
from typing import Optional, Dict, Any
from dotenv import load_dotenv


SEVERITY_COLORS = {
  'broken':    0xFF3B30,  # red
  'weakened':  0xFFAA00,  # amber
  'intact':    0x00C853,  # green
  'info':      0x4A90E2,  # blue
  'order':     0x7B61FF,  # purple
  'error':     0xB71C1C,  # dark red
}


def _webhook_url() -> Optional[str]:
  load_dotenv()
  return os.getenv("DISCORD_WEBHOOK_URL")


def send_thesis_alert(
  ticker: str,
  severity: str,
  headline: str,
  reasoning: str,
  action: str,
  thesis_id: Optional[int] = None,
) -> bool:
  """Sends an embed describing a thesis status change. Returns True on success."""
  url = _webhook_url()
  if not url:
    print(f"[alerts] DISCORD_WEBHOOK_URL not set; would have alerted "
          f"{ticker} {severity}: {headline[:60]}", file=sys.stderr, flush=True)
    return False
  return _send_embed(
    url=url,
    title=f"{ticker}: thesis {severity}",
    color=SEVERITY_COLORS.get(severity, SEVERITY_COLORS['info']),
    fields=[
      {"name": "Event", "value": (headline or "")[:1000], "inline": False},
      {"name": "Reasoning", "value": (reasoning or "")[:1000], "inline": False},
      {"name": "Action", "value": action or "no_action", "inline": True},
      *([{"name": "Thesis", "value": f"#{thesis_id}", "inline": True}]
        if thesis_id else []),
    ],
  )


def send_order_alert(
  ticker: str, side: str, qty: float, order_type: str,
  limit_price: Optional[float], paper: bool,
  thesis_id: Optional[int] = None,
  order_id: Optional[str] = None,
) -> bool:
  url = _webhook_url()
  if not url:
    print(f"[alerts] DISCORD_WEBHOOK_URL not set; would have alerted order "
          f"{side} {qty} {ticker}", file=sys.stderr, flush=True)
    return False
  return _send_embed(
    url=url,
    title=f"{'PAPER' if paper else 'LIVE'} order: {side.upper()} {qty} {ticker}",
    color=SEVERITY_COLORS['order'],
    fields=[
      {"name": "Type", "value": order_type, "inline": True},
      {"name": "Limit", "value": str(limit_price) if limit_price else "market",
       "inline": True},
      *([{"name": "Thesis", "value": f"#{thesis_id}", "inline": True}]
        if thesis_id else []),
      *([{"name": "Broker ID", "value": str(order_id), "inline": True}]
        if order_id else []),
    ],
  )


def send_health_alert(component: str, status: str, detail: str = "") -> bool:
  url = _webhook_url()
  if not url:
    print(f"[alerts] DISCORD_WEBHOOK_URL not set; health: {component} {status}",
          file=sys.stderr, flush=True)
    return False
  return _send_embed(
    url=url,
    title=f"Health: {component} {status}",
    color=SEVERITY_COLORS['error' if status.lower() in ('down', 'error') else 'info'],
    fields=[{"name": "Detail", "value": (detail or "")[:1500], "inline": False}],
  )


def _send_embed(url: str, title: str, color: int, fields: list) -> bool:
  """Lazy-import discord_webhook to keep the package optional at import time."""
  try:
    from discord_webhook import DiscordWebhook, DiscordEmbed
  except ImportError as e:
    print(f"[alerts] discord_webhook not installed: {e}", file=sys.stderr, flush=True)
    return False
  try:
    hook = DiscordWebhook(url=url, rate_limit_retry=True)
    embed = DiscordEmbed(title=title[:256], color=color)
    for f in fields:
      embed.add_embed_field(
        name=f["name"][:256], value=f["value"][:1024],
        inline=f.get("inline", False)
      )
    hook.add_embed(embed)
    resp = hook.execute()
    # discord_webhook returns Response or list of Response
    if isinstance(resp, list):
      ok = all(getattr(r, 'status_code', 500) < 300 for r in resp)
    else:
      ok = getattr(resp, 'status_code', 500) < 300
    if not ok:
      print(f"[alerts] webhook returned non-2xx: {resp}", file=sys.stderr, flush=True)
    return ok
  except Exception as e:
    print(f"[alerts] webhook send failed: {e}", file=sys.stderr, flush=True)
    return False


def build_thesis_payload(ticker: str, severity: str, headline: str,
                          reasoning: str, action: str,
                          thesis_id: Optional[int] = None) -> Dict[str, Any]:
  """Pure function exposed for testability — returns the embed payload that
  WOULD be sent to Discord. Used by tests so we can assert payload shape
  without making a network call."""
  return {
    "title": f"{ticker}: thesis {severity}",
    "color": SEVERITY_COLORS.get(severity, SEVERITY_COLORS['info']),
    "fields": [
      {"name": "Event", "value": (headline or "")[:1000], "inline": False},
      {"name": "Reasoning", "value": (reasoning or "")[:1000], "inline": False},
      {"name": "Action", "value": action or "no_action", "inline": True},
      *([{"name": "Thesis", "value": f"#{thesis_id}", "inline": True}]
        if thesis_id else []),
    ],
  }
