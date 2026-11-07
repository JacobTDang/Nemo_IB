"""Discord alerts as Components V2 messages.

A missing webhook is a DiscordError that names the setting. It is not a
silent False. The webhook URL is never included in the error.
"""
from typing import Any, Dict, Optional

from research.discord_post import (
    BLURPLE, GREEN, RED, YELLOW, DiscordError, deliver, escape_text,
    format_money, signal_message,
)

# The colour and the emoji of each severity. A severity that is not here is a
# caller's mistake, and the message says so instead of choosing a colour.
_SEVERITIES = {
    'broken': (RED, '❌'),
    'weakened': (YELLOW, '⚠️'),
    'intact': (GREEN, '✅'),
    'info': (BLURPLE, '📋'),
    'order': (BLURPLE, '📋'),
    'error': (RED, '❌'),
}
SEVERITY_COLORS = {name: color for name, (color, _) in _SEVERITIES.items()}


def build_thesis_payload(
    ticker: str, severity: str, headline: str, reasoning: str, action: str,
    thesis_id: Optional[int] = None,
) -> Dict[str, Any]:
    """The Components V2 message a thesis change would send."""
    if severity not in _SEVERITIES:
        raise DiscordError(
            f"unknown severity {severity!r}; use one of {', '.join(_SEVERITIES)}")
    color, emoji = _SEVERITIES[severity]
    lines = [
        f"**{escape_text(ticker)}** {escape_text(severity)}",
        escape_text(headline or '')[:300],
        escape_text(reasoning or '')[:300],
        f"**Action** {escape_text(action or 'no_action')}",
    ]
    if thesis_id:
        lines.append(f"**Thesis** #{thesis_id}")
    return signal_message(
        f"{ticker}: thesis {severity}", lines, emoji=emoji, color=color,
        source='nemo thesis', suppress=False)


def send_thesis_alert(
    ticker: str, severity: str, headline: str, reasoning: str, action: str,
    thesis_id: Optional[int] = None,
) -> bool:
    deliver('signals', [build_thesis_payload(
        ticker, severity, headline, reasoning, action, thesis_id)])
    return True


def send_order_alert(
    ticker: str, side: str, qty: float, order_type: str,
    limit_price: Optional[float], paper: bool,
    thesis_id: Optional[int] = None, order_id: Optional[str] = None,
) -> bool:
    price = format_money(limit_price) if limit_price else 'market'
    lines = [
        f"**{escape_text(side.upper())}** {qty:g} **{escape_text(ticker)}**",
        f"**Type** {escape_text(order_type)} · **Limit** {price}",
        '**Paper**' if paper else '**Live**',
    ]
    if thesis_id:
        lines.append(f"**Thesis** #{thesis_id}")
    if order_id:
        lines.append(f"**Broker** {escape_text(str(order_id))}")
    payload = signal_message(
        f"{'Paper' if paper else 'Live'} order", lines,
        emoji='🧪' if paper else '🛒',
        color=YELLOW if paper else GREEN, source='nemo order')
    deliver('signals', [payload])
    return True


def send_health_alert(component: str, status: str, detail: str = '') -> bool:
    bad = status.lower() in ('down', 'error')
    payload = signal_message(
        f"Health: {component} {status}",
        [escape_text(detail or status)[:1500]],
        emoji='❌' if bad else '📋',
        color=RED if bad else BLURPLE, source='nemo health')
    deliver('ops', [payload])
    return True


__all__ = [
    'DiscordError', 'SEVERITY_COLORS', 'build_thesis_payload',
    'send_thesis_alert', 'send_order_alert', 'send_health_alert',
]
