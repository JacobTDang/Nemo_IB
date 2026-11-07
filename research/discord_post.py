"""Discord Components V2 posts for the scheduled jobs.

One container per message. A plain webhook can send this layout when the
request sets ``with_components=true`` and the message flag ``1 << 15``.
The webhook URL is never put in an exception, a log line, or a payload.
"""
from __future__ import annotations

import os
import re
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

IS_COMPONENTS_V2 = 1 << 15
SUPPRESS_NOTIFICATIONS = 1 << 12

CONTAINER = 17
TEXT = 10
SEPARATOR = 14
ACTION_ROW = 1
BUTTON = 2

GREEN = 0x57F287
YELLOW = 0xFEE75C
RED = 0xED4245
BLURPLE = 0x5865F2

TEXT_LIMIT = 4000
COMPONENT_LIMIT = 40
LABEL_LIMIT = 80
URL_LIMIT = 512

# A 429 waits out Discord's retry_after, up to this many seconds, and the post
# is tried again. After SEND_ATTEMPTS answers the 429 is a failed post.
RETRY_AFTER_CAP = 10.0
SEND_ATTEMPTS = 3
ERROR_DETAIL_LIMIT = 300

OPS_SETTING = "NEMO_DISCORD_OPS_WEBHOOK_URL"
SIGNALS_SETTING = "NEMO_DISCORD_SIGNALS_WEBHOOK_URL"
_SETTINGS = {"ops": OPS_SETTING, "signals": SIGNALS_SETTING}

_WEBHOOK_URL = re.compile(
    r"https://(?:ptb\.|canary\.)?discord(?:app)?\.com/api/webhooks/\S+",
    re.I)
_MARKDOWN = re.compile(r"([\\*_`~|>\[\]])")


class DiscordError(Exception):
    """A Discord post cannot be sent. The message names a setting or a status code."""


def escape_text(value: str) -> str:
    """Escape markdown in text the job did not write as markup."""
    return _MARKDOWN.sub(r"\\\1", value)


def redact(value: str) -> str:
    return _WEBHOOK_URL.sub("[webhook]", value)


def format_money(amount: float) -> str:
    return f"${amount:,.2f}"


def format_percent(fraction: float) -> str:
    return f"{fraction * 100:.0f}%"


def format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 10:
        return f"{seconds:.1f}s"
    if seconds < 60:
        return f"{int(round(seconds))}s"
    minutes, sec = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}m {sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def format_gib(nbytes: float) -> str:
    return f"{nbytes / (1024 ** 3):.1f} GiB"


def mask_link(label: str, url: str) -> str:
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise DiscordError("a link needs an http(s) URL with a host")
    if parts.username or parts.password:
        raise DiscordError("a link must not carry a user or a password")
    safe = "".join(
        f"%{ord(ch):02X}" if ch in "()<>[] " or ord(ch) < 32 else ch
        for ch in url)
    return f"[{escape_text(label)}](<{safe}>)"


def _ops_webhook() -> str:
    return os.environ.get("NEMO_DISCORD_OPS_WEBHOOK_URL", "").strip()


def _signals_webhook() -> str:
    return os.environ.get("NEMO_DISCORD_SIGNALS_WEBHOOK_URL", "").strip()


def missing_settings() -> List[str]:
    """Settings a job needs and does not have. Empty when both webhooks are set."""
    missing = []
    if not _signals_webhook():
        missing.append(f"{SIGNALS_SETTING} is not set")
    if not _ops_webhook():
        missing.append(f"{OPS_SETTING} is not set")
    return missing


def webhook_url(channel: str) -> str:
    value = _signals_webhook() if channel == "signals" else _ops_webhook()
    name = _SETTINGS[channel]
    if not value:
        raise DiscordError(f"{name} is not set")
    return value


def webhook_endpoint(url: str) -> str:
    """The URL to POST to: the webhook plus ``with_components=true&wait=true``.

    ``wait=true`` makes Discord answer with the saved message. Without it a
    message that Discord does not save returns no error. Other query
    parameters stay.
    """
    parts = urlsplit(url)
    query = [(key, value) for key, value in parse_qsl(parts.query, keep_blank_values=True)
             if key not in ("with_components", "wait")]
    query.extend([("with_components", "true"), ("wait", "true")])
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), ""))


def _text_length(nodes: Sequence[dict]) -> int:
    total = 0
    for node in nodes:
        if node.get("type") == TEXT:
            total += len(node.get("content") or "")
        total += _text_length(node.get("components") or [])
    return total


def _component_count(nodes: Sequence[dict]) -> int:
    total = 0
    for node in nodes:
        total += 1
        total += _component_count(node.get("components") or [])
    return total


def _check(payload: dict) -> None:
    nodes = payload.get("components") or []
    if _component_count(nodes) > COMPONENT_LIMIT:
        raise DiscordError(
            f"a Discord message has {_component_count(nodes)} components; "
            f"the limit is {COMPONENT_LIMIT}")
    length = _text_length(nodes)
    if length > TEXT_LIMIT:
        raise DiscordError(
            f"a Discord message has {length} characters of text; "
            f"the limit is {TEXT_LIMIT}")
    for node in _walk(nodes):
        if node.get("type") != BUTTON:
            continue
        label = node.get("label") or ""
        url = node.get("url") or ""
        if len(label) > LABEL_LIMIT:
            raise DiscordError(f"a button label is {len(label)} characters; the limit is {LABEL_LIMIT}")
        if len(url) > URL_LIMIT:
            raise DiscordError(f"a button URL is {len(url)} characters; the limit is {URL_LIMIT}")


def _walk(nodes: Sequence[dict]):
    for node in nodes:
        yield node
        yield from _walk(node.get("components") or [])


def _footer(source: str, when: int) -> str:
    return f"-# {source} · <t:{when}:f> (<t:{when}:R>)"


def _fit_error(error: str, budget: int) -> str:
    body = redact(error).strip()
    fence = f"```\n{body}\n```"
    if len(fence) <= budget:
        return fence
    keep = max(0, budget - len("```\n\n```"))
    return f"```\n{body[:keep]}\n```"


def _assemble(*, emoji: str, headline: str, color: int, facts: str, detail: str,
              error: str, links: Sequence[Tuple[str, str]], source: str,
              when: int, suppress: bool) -> dict:
    children: List[dict] = [
        {"type": TEXT, "content": f"## {emoji} {headline}"},
    ]
    if facts:
        children.append({"type": TEXT, "content": facts})
    extras: List[dict] = []
    if detail:
        extras.append({"type": TEXT, "content": detail})
    if error:
        used = _text_length(children) + _text_length(extras) + len(_footer(source, when))
        extras.append({"type": TEXT, "content": _fit_error(error, TEXT_LIMIT - used - 1)})
    buttons = []
    for label, url in links:
        if len(label) > LABEL_LIMIT or len(url) > URL_LIMIT:
            raise DiscordError("a link button exceeds Discord's label or URL limit")
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            continue
        buttons.append({"type": BUTTON, "style": 5, "label": label, "url": url})
    if extras or buttons:
        children.append({"type": SEPARATOR, "divider": True, "spacing": 1})
        children.extend(extras)
    while buttons:
        children.append({"type": ACTION_ROW, "components": buttons[:5]})
        buttons = buttons[5:]
    children.append({"type": TEXT, "content": _footer(source, when)})
    payload = {
        "flags": IS_COMPONENTS_V2 | (SUPPRESS_NOTIFICATIONS if suppress else 0),
        "components": [{
            "type": CONTAINER,
            "accent_color": color,
            "components": children,
        }],
        "allowed_mentions": {"parse": []},
    }
    _check(payload)
    return payload


def _bold_first_number(summary: str) -> str:
    if "**" in summary:
        return summary
    return re.sub(r"(\d[\d,]*)", r"**\1**", summary, count=1)


def ops_message(job: str, summary: str, *, ok: bool, quiet: bool = False,
                seconds: Optional[float] = None, error: str = "",
                when: Optional[int] = None, warning: bool = False) -> dict:
    when = int(time.time()) if when is None else int(when)
    if not ok:
        emoji, color, suppress = "❌", RED, False
    elif warning:
        emoji, color, suppress = "⚠️", YELLOW, False
    elif quiet:
        emoji, color, suppress = "💤", BLURPLE, True
    else:
        emoji, color, suppress = "✅", GREEN, True
    facts = _bold_first_number(summary)
    if seconds is not None:
        facts = f"{facts}\n**Duration** **{format_duration(seconds)}**"
    return _assemble(
        emoji=emoji, headline=job, color=color, facts=facts, detail="",
        error=error, links=(), source=f"nemo {job}", when=when, suppress=suppress)


def signal_message(headline: str, lines: Sequence[str], *, emoji: str = "📋",
                   color: int = BLURPLE, source: str = "nemo",
                   links: Sequence[Tuple[str, str]] = (),
                   detail: str = "", when: Optional[int] = None,
                   suppress: bool = False) -> dict:
    when = int(time.time()) if when is None else int(when)
    return _assemble(
        emoji=emoji, headline=headline, color=color,
        facts="\n".join(lines), detail=detail, error="", links=tuple(links),
        source=source, when=when, suppress=suppress)


def brief_messages(brief_md: str, *, when: Optional[int] = None) -> List[dict]:
    """One container per page. The headline carries ``(1/3)`` once the brief splits."""
    when = int(time.time()) if when is None else int(when)
    text = (brief_md or "").strip()
    if not text:
        raise DiscordError("the pre-market brief is empty")
    source = "nemo pre-market-brief"
    footer = _footer(source, when)
    # Room for the headline line and the footer inside the 4000-character budget.
    budget = TEXT_LIMIT - len(footer) - 80
    pages = _pages(text, budget)
    total = len(pages)
    built = []
    for index, page in enumerate(pages, start=1):
        headline = "Pre-market brief" if total == 1 else f"Pre-market brief ({index}/{total})"
        built.append(_assemble(
            emoji="📋", headline=headline, color=BLURPLE, facts="",
            detail=page, error="", links=(), source=source, when=when,
            suppress=False))
    return built


def _pages(text: str, budget: int) -> List[str]:
    if len(text) <= budget:
        return [text]
    pages: List[str] = []
    current = ""
    for block in re.split(r"(?=\n## |\n### )", text):
        piece = block.lstrip("\n")
        if not piece:
            continue
        if len(piece) > budget:
            if current:
                pages.append(current)
                current = ""
            for start in range(0, len(piece), budget):
                pages.append(piece[start:start + budget])
            continue
        candidate = f"{current}\n\n{piece}" if current else piece
        if len(candidate) <= budget:
            current = candidate
        else:
            pages.append(current)
            current = piece
    if current:
        pages.append(current)
    return pages or [text[:budget]]


def messages_for(job: str, code: int, result: Optional[dict], *,
                 seconds: Optional[float] = None, when: Optional[int] = None
                 ) -> List[Tuple[str, dict]]:
    """The messages one job run should post. ``code`` is the job's own exit code."""
    result = result or {}
    when = int(time.time()) if when is None else int(when)
    error = str(result.get("error") or "")
    failed = code != 0 or result.get("status") == "failed"

    if job == "research-copy":
        return _copy(result, failed, seconds, when, error)
    if job == "research-scan":
        return _scan(result, failed, seconds, when, error)
    if job == "research-watch":
        return _watch(result, failed, seconds, when, error)
    if job == "research-score":
        return _score(result, failed, seconds, when, error)
    if job == "research-announce":
        return _announce(result, failed, seconds, when, error)
    if job == "research-fills":
        return _fills(result, failed, seconds, when, error)
    if job == "research-daily":
        return _daily(result, failed, seconds, when, error)
    if job == "research-seed":
        written = int(result.get("written") or 0)
        quiet = written == 0
        return [("ops", ops_message(
            job, f"**{written:,}** quarters recorded", ok=not failed, quiet=quiet,
            seconds=seconds, error=error, when=when))]
    if job == "congress-sync":
        parsed = int(result.get("parsed") or 0)
        return [("ops", ops_message(
            job, f"**{parsed:,}** filings parsed", ok=not failed, quiet=False,
            seconds=seconds, error=error, when=when))]
    if job == "research-backup":
        removed = int(result.get("removed") or 0)
        size = format_gib(float(result.get("bytes") or 0))
        return [("ops", ops_message(
            job, f"**{size}** written, **{removed}** older copies removed",
            ok=not failed, quiet=False, seconds=seconds, error=error, when=when))]
    if job == "research-status":
        attention = [str(item) for item in result.get("attention") or []]
        if attention:
            return [("ops", ops_message(
                job, f"**{len(attention)}** items need attention", ok=True,
                warning=True, seconds=seconds,
                error="\n".join(attention), when=when))]
        return [("ops", ops_message(
            job, "**0** items need attention", ok=True, quiet=True,
            seconds=seconds, when=when))]
    if job == "docker-prune":
        ok = bool(result.get("ok", code == 0)) and not failed
        return [("ops", ops_message(
            job, "containers older than 24h pruned" if ok else "container prune failed",
            ok=ok, quiet=ok, seconds=seconds, error=error, when=when))]
    summary = f"**{code}** exit"
    return [("ops", ops_message(
        job, summary, ok=not failed, quiet=not failed, seconds=seconds,
        error=error, when=when))]


def _copy(result, failed, seconds, when, error):
    rows = int(result.get("rows_written") or 0)
    if result.get("status") == "off":
        return [("ops", ops_message(
            "research-copy", "copy trading is **off**", ok=True, quiet=True,
            seconds=seconds, when=when))]
    messages = [("ops", ops_message(
        "research-copy", f"**{rows:,}** decisions recorded",
        ok=not failed, quiet=rows == 0, seconds=seconds, error=error, when=when))]
    preview = result.get("preview") or []
    if preview and not failed:
        lines = []
        for item in preview[:8]:
            lines.append(
                f"**{escape_text(str(item.get('ticker') or ''))}** "
                f"{escape_text(str(item.get('member') or ''))} · "
                f"**{format_money(float(item.get('dollars') or 0))}**")
        messages.append(("signals", signal_message(
            "Congressional copy", lines, emoji="🛒", color=GREEN,
            source="nemo research-copy", when=when)))
    return messages


def _scan(result, failed, seconds, when, error):
    candidates = result.get("candidates") or []
    written = int(result.get("written") or len(candidates) or 0)
    messages = [("ops", ops_message(
        "research-scan", f"**{written:,}** orders filed",
        ok=not failed, quiet=written == 0 and not failed,
        seconds=seconds, error=error, when=when))]
    if candidates and not failed:
        lines = []
        for row in candidates[:8]:
            lines.append(
                f"**{escape_text(str(row.get('ticker') or ''))}** "
                f"{escape_text(str(row.get('side') or ''))} · "
                f"SUE **{row.get('sue')}**")
        messages.append(("signals", signal_message(
            "Scanner book", lines, emoji="📋", color=BLURPLE,
            source="nemo research-scan", when=when)))
    return messages


def _watch(result, failed, seconds, when, error):
    events = result.get("events") or []
    count = int(result.get("new_events") if result.get("new_events") is not None else len(events))
    messages = [("ops", ops_message(
        "research-watch", f"**{count:,}** new 13D filings",
        ok=not failed, quiet=count == 0 and not failed,
        seconds=seconds, error=error, when=when))]
    if count and not failed:
        lines = []
        links = []
        for event in events[:8]:
            if isinstance(event, dict):
                lines.append(
                    f"**{escape_text(str(event.get('subject_ticker') or ''))}** "
                    f"{escape_text(str(event.get('form') or '13D'))} · "
                    f"{escape_text(str(event.get('filer_name') or ''))}")
                url = event.get("url") or ""
                if url and not links:
                    links.append(("Filing", str(url)))
            else:
                lines.append(escape_text(str(event)))
        messages.append(("signals", signal_message(
            "Activist filing", lines or [f"**{count}** new filings"],
            emoji="📋", color=BLURPLE, source="nemo research-watch",
            links=links, when=when)))
    return messages


def _score(result, failed, seconds, when, error):
    sample = int(result.get("sample") or 0)
    usable = bool(result.get("coefficient_usable"))
    drift = result.get("drift_bps_per_sue")
    if drift is None:
        drift = result.get("drift_bps_per_tail")
    drift_text = "no coefficient" if drift is None else f"**{float(drift):.1f} bp** per SUE"
    hit = result.get("hit_rate")
    hit_text = "" if hit is None else f" · hit rate **{format_percent(float(hit))}**"
    messages = [("ops", ops_message(
        "research-score", f"**{sample:,}** trades scored",
        ok=not failed, quiet=not failed, seconds=seconds, error=error, when=when))]
    if not failed:
        messages.append(("signals", signal_message(
            "Score",
            [f"{drift_text}{hit_text}",
             "**usable**" if usable else "**not usable** yet"],
            emoji="✅" if usable else "⚠️",
            color=GREEN if usable else YELLOW,
            source="nemo research-score", when=when)))
    return messages


def _announce(result, failed, seconds, when, error):
    written = int(result.get("written") or 0)
    messages = [("ops", ops_message(
        "research-announce", f"**{written:,}** announcement dates recorded",
        ok=not failed, quiet=written == 0 and not failed,
        seconds=seconds, error=error, when=when))]
    if written and not failed:
        messages.append(("signals", signal_message(
            "Earnings dates",
            [f"**{written:,}** dates recorded from Item 2.02"],
            emoji="📋", color=BLURPLE, source="nemo research-announce", when=when)))
    return messages


def _fills(result, failed, seconds, when, error):
    submitted = int(result.get("submitted") or 0)
    rejected = int(result.get("rejected") or 0)
    updated = int(result.get("updated") or 0)
    action = str(result.get("action") or "fills")
    count = submitted or updated
    warning = rejected > 0 and not failed
    messages = [("ops", ops_message(
        "research-fills",
        f"**{count:,}** paper {action}s, **{rejected:,}** rejected",
        ok=not failed, quiet=count == 0 and not warning and not failed,
        warning=warning, seconds=seconds, error=error, when=when))]
    if count and not failed:
        messages.append(("signals", signal_message(
            "Paper fills",
            [f"**{count:,}** {escape_text(action)}",
             f"**{rejected:,}** rejected" if rejected else "**0** rejected"],
            emoji="🧪" if warning else "🛒",
            color=YELLOW if warning else GREEN,
            source="nemo research-fills", when=when)))
    return messages


def _daily(result, failed, seconds, when, error):
    if "days" in result and isinstance(result["days"], dict):
        days = result["days"]
        return [("ops", ops_message(
            "research-daily", f"**{len(days):,}** backfill days",
            ok=not failed, quiet=False, seconds=seconds, error=error, when=when))]
    stages = {key: value for key, value in result.items()
              if isinstance(value, dict) and "status" in value}
    written = sum(int(value.get("written") or 0) for value in stages.values())
    failed_names = [name for name, value in stages.items() if value.get("status") == "failed"]
    gaps = result.get("gaps") or []
    summary = f"**{written:,}** rows recorded"
    if failed_names:
        summary += f", **{len(failed_names)}** stages failed"
    detail_error = error
    if failed_names and not detail_error:
        bits = []
        for name in failed_names:
            stage_error = stages[name].get("error") or "failed"
            bits.append(f"{name}: {stage_error}")
        detail_error = "\n".join(bits)
    if gaps:
        detail_error = (detail_error + "\n" if detail_error else "") + (
            f"{len(gaps)} coverage gaps")
    return [("ops", ops_message(
        "research-daily", summary, ok=not failed_names and not failed,
        quiet=False, seconds=seconds, error=detail_error, when=when))]


def deliver(channel: str, payloads: Sequence[dict], *,
            transport: Optional[Callable[[str, dict], Any]] = None,
            url_for: Optional[Callable[[str], str]] = None) -> None:
    """POST each payload. ``transport`` is the test seam. It receives the URL.

    Every payload is checked against the limits before the first one is sent,
    so a long digest never posts half.
    """
    url = (url_for or webhook_url)(channel)
    endpoint = webhook_endpoint(url)
    send = transport or _http_send
    for payload in payloads:
        _check(payload)
    for payload in payloads:
        try:
            send(endpoint, payload)
        except DiscordError:
            raise
        except Exception as exc:
            raise DiscordError(
                f"Discord post failed: {type(exc).__name__}") from None


def finish(job: str, code: int, result: Optional[dict] = None, *,
           seconds: Optional[float] = None,
           transport: Optional[Callable[[str, dict], Any]] = None) -> int:
    """Post after the job has recorded its data. A failed post returns 1.

    The recorded rows stay. This function only changes the exit code.
    """
    try:
        grouped: Dict[str, List[dict]] = {}
        for channel, payload in messages_for(job, code, result, seconds=seconds):
            grouped.setdefault(channel, []).append(payload)
        for channel, payloads in grouped.items():
            deliver(channel, payloads, transport=transport)
    except DiscordError as exc:
        print(redact(str(exc)), flush=True)
        return 1 if code == 0 else code
    return code


def _retry_after(response) -> Optional[float]:
    try:
        value = response.json().get("retry_after")
    except (ValueError, AttributeError):
        value = None
    if value is None:
        value = response.headers.get("retry-after")
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return None


def _discord_detail(response) -> str:
    """Discord's JSON ``code`` and ``message``. The body holds no secret."""
    try:
        body = response.json()
    except ValueError:
        return ""
    if not isinstance(body, dict):
        return ""
    bits = []
    if body.get("code") is not None:
        bits.append(f"code {body['code']}")
    if body.get("message"):
        bits.append(str(body["message"]))
    return redact(": ".join(bits))[:ERROR_DETAIL_LIMIT]


def _refusal(response) -> DiscordError:
    detail = _discord_detail(response)
    suffix = f": {detail}" if detail else ""
    return DiscordError(f"Discord post returned {response.status_code}{suffix}")


def _http_send(url: str, body: dict) -> None:
    import httpx

    for attempt in range(1, SEND_ATTEMPTS + 1):
        try:
            response = httpx.post(url, json=body, timeout=30)
        except Exception as exc:
            raise DiscordError(
                f"Discord post failed: {type(exc).__name__}") from None
        if response.status_code == 429 and attempt < SEND_ATTEMPTS:
            wait = _retry_after(response)
            if wait is None:
                raise _refusal(response)
            time.sleep(min(wait, RETRY_AFTER_CAP))
            continue
        if response.status_code >= 300:
            raise _refusal(response)
        try:
            saved = response.json()
        except ValueError:
            saved = None
        if not isinstance(saved, dict) or not saved.get("id"):
            raise DiscordError(
                f"Discord answered {response.status_code} without a message id; "
                f"the message may not be saved")
        return


def log_tail(path: str, lines: int = 100) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            content = handle.readlines()
    except OSError as exc:
        return f"{path} cannot be read: {exc.strerror or exc}"
    return redact("".join(content[-lines:]))
