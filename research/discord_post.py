"""Discord Components V2 posts for the scheduled jobs.

One container per message. A plain webhook can send this layout when the
request sets ``with_components=true`` and the message flag ``1 << 15``.
The webhook URL is never put in an exception, a log line, or a payload.

A run that found nothing and did not fail posts nothing. Each job's own
process posts its result through ``finish``. After a non-zero exit that it
reported, the process prints ``REPORTED`` as its last line, so the cron
failure branch (``research.notify``) does not post the same failure again.
"""
from __future__ import annotations

import collections
import json
import os
import re
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

IS_COMPONENTS_V2 = 1 << 15
SUPPRESS_NOTIFICATIONS = 1 << 12
EPHEMERAL = 1 << 6

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

# Exit codes the cron lines read. A job that ran and could not post exits 75
# (EX_TEMPFAIL), and the chain still goes on to the scan. `flock -n -E` exits
# 99 when the lock is held.
POST_FAILED = 75
LOCK_HELD = 99

# The last line a process prints after a non-zero exit that it already
# reported to Discord, or deliberately did not repeat.
REPORTED = "nemo-discord: failure reported"

INTERACTION_API = "https://discord.com/api/v10"

OPS_SETTING = "NEMO_DISCORD_OPS_WEBHOOK_URL"
SIGNALS_SETTING = "NEMO_DISCORD_SIGNALS_WEBHOOK_URL"

# The channels each job posts to. The compose file gives each service exactly
# the webhook settings for these channels, and `messages_for` refuses a
# message to any other channel.
JOB_CHANNELS: Dict[str, Tuple[str, ...]] = {
    "research-copy": ("ops", "signals"),
    "research-scan": ("ops", "signals"),
    "research-watch": ("ops", "signals"),
    "research-score": ("ops", "signals"),
    "research-announce": ("ops", "signals"),
    "research-fills": ("ops", "signals"),
    "research-portfolio": ("ops", "signals"),
    "research-daily": ("ops",),
    "research-seed": ("ops",),
    "research-quiver": ("ops",),
    "congress-sync": ("ops",),
    "research-backup": ("ops",),
    "research-status": ("ops",),
    "docker-prune": ("ops",),
}

WATCH_EVENTS_SHOWN = 8
BUTTONS_PER_ROW = 5
FAILURES_SHOWN = 10
CUT_MARK = "[earlier lines cut]"
ZERO_WIDTH_SPACE = "​"

# A webhook URL, and an interaction's (/api/v10/webhooks/<app>/<token>), whose
# token answers for the interaction for 15 minutes.
_WEBHOOK_URL = re.compile(
    r"https://(?:ptb\.|canary\.)?discord(?:app)?\.com/api/(?:v\d+/)?webhooks/\S+",
    re.I)
_MARKDOWN = re.compile(r"([\\*_`~|>\[\]])")
_FENCE = re.compile(r"^\s*```")
_HEADER = re.compile(r"^#{1,3} ")
_RUN_OF_BACKTICKS = re.compile(r"`(?=``)")


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


def format_signed_money(amount: float) -> str:
    """``+$412.10`` or ``-$1,000.00``: a change, with its sign always shown."""
    return ("+" if amount >= 0 else "-") + format_money(abs(amount))


def format_signed_percent(fraction: float) -> str:
    """``+0.41%`` or ``-0.85%``: a return, to two places."""
    return f"{fraction * 100:+.2f}%"


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


def _count(n: int, singular: str, plural: Optional[str] = None, *,
           bold: bool = True) -> str:
    """``**1** order`` or ``**1,200** orders``: the noun agrees with the count."""
    noun = singular if n == 1 else (plural or f"{singular}s")
    number = f"{n:,}"
    return f"**{number}** {noun}" if bold else f"{number} {noun}"


def _fixed(value: Any, places: int) -> str:
    try:
        return f"{float(value):.{places}f}"
    except (TypeError, ValueError):
        return escape_text(str(value))


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


def webhook_url(channel: str) -> str:
    if channel == "ops":
        name, value = OPS_SETTING, os.environ.get(OPS_SETTING, "")
    elif channel == "signals":
        name, value = SIGNALS_SETTING, os.environ.get(SIGNALS_SETTING, "")
    else:
        raise DiscordError(f"{channel!r} is not a Discord channel this project posts to")
    if not value.strip():
        raise DiscordError(f"{name} is not set")
    return value.strip()


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


def _break_fences(text: str) -> str:
    """Stop a run of backticks in log text from closing the code block."""
    return _RUN_OF_BACKTICKS.sub("`" + ZERO_WIDTH_SPACE, text)


def _fit_error(error: str, budget: int) -> str:
    """The end of ``error`` in a code block of at most ``budget`` characters.

    The last lines of a log hold the cause, so the head is the part that goes.
    The cut falls on a line boundary unless the last line alone is too long.
    """
    body = _break_fences(redact(error)).strip()
    fence = f"```\n{body}\n```"
    if len(fence) <= budget:
        return fence
    keep = max(0, budget - len("```\n\n```") - len(CUT_MARK) - 1)
    tail = body[len(body) - keep:] if keep else ""
    if tail and body[len(body) - keep - 1] != "\n" and "\n" in tail:
        tail = tail.split("\n", 1)[1]
    return f"```\n{CUT_MARK}\n{tail}\n```"


def _link_button(label: Any, url: Any) -> dict:
    if not isinstance(label, str) or not label:
        raise DiscordError("a link button needs a label")
    if len(label) > LABEL_LIMIT:
        raise DiscordError(f"a button label is {len(label)} characters; the limit is {LABEL_LIMIT}")
    if not isinstance(url, str) or not url:
        raise DiscordError(f"the link button {label!r} has no URL")
    if len(url) > URL_LIMIT:
        raise DiscordError(f"the link button {label!r} has a URL of {len(url)} characters; the limit is {URL_LIMIT}")
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise DiscordError(f"the link button {label!r} needs an http(s) URL with a host")
    return {"type": BUTTON, "style": 5, "label": label, "url": url}


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
    buttons = [_link_button(label, url) for label, url in links]
    if extras or buttons:
        children.append({"type": SEPARATOR, "divider": True, "spacing": 1})
        children.extend(extras)
    while buttons:
        children.append({"type": ACTION_ROW, "components": buttons[:BUTTONS_PER_ROW]})
        buttons = buttons[BUTTONS_PER_ROW:]
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
                when: Optional[int] = None, warning: bool = False,
                detail: str = "") -> dict:
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
        emoji=emoji, headline=job, color=color, facts=facts, detail=detail,
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


def _blocks(text: str) -> List[str]:
    """Paragraphs, with a header starting a new one. A fenced block is one block."""
    blocks: List[str] = []
    current: List[str] = []

    def close() -> None:
        if current:
            blocks.append("\n".join(current))
            current.clear()

    fenced = False
    for line in text.splitlines():
        if fenced:
            current.append(line)
            if _FENCE.match(line):
                fenced = False
                close()
        elif _FENCE.match(line):
            close()
            current.append(line)
            fenced = True
        elif not line.strip():
            close()
        elif _HEADER.match(line):
            close()
            current.append(line)
        else:
            current.append(line)
    close()
    return blocks


def _chunk_lines(lines: Sequence[str], limit: int) -> List[List[str]]:
    chunks: List[List[str]] = []
    current: List[str] = []
    size = 0
    for line in lines:
        pieces = [line[i:i + limit] for i in range(0, len(line), limit)] or [""]
        for piece in pieces:
            added = len(piece) + (1 if current else 0)
            if current and size + added > limit:
                chunks.append(current)
                current, size, added = [], 0, len(piece)
            current.append(piece)
            size += added
    if current:
        chunks.append(current)
    return chunks


def _split_block(block: str, budget: int) -> List[str]:
    """Cut one block that is longer than a page. A fenced block is closed at the
    end of each piece and opened again at the start of the next."""
    lines = block.splitlines()
    if _FENCE.match(lines[0]):
        opener = lines[0].strip()
        inner = lines[1:]
        if inner and _FENCE.match(inner[-1]):
            inner = inner[:-1]
        room = budget - len(opener) - len("```") - 2
        return [f"{opener}\n" + "\n".join(chunk) + "\n```"
                for chunk in _chunk_lines(inner, room)]
    return ["\n".join(chunk) for chunk in _chunk_lines(lines, budget)]


def _pages(text: str, budget: int) -> List[str]:
    if len(text) <= budget:
        return [text]
    pieces: List[str] = []
    for block in _blocks(text):
        pieces.extend(_split_block(block, budget) if len(block) > budget else [block])
    pages: List[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current}\n\n{piece}" if current else piece
        if len(candidate) <= budget:
            current = candidate
        else:
            pages.append(current)
            current = piece
    if current:
        pages.append(current)
    return pages


# ------------------------------------------------------------ what each job says

def _failure_lines(failed: Sequence[Any], limit: int = FAILURES_SHOWN) -> str:
    items = [str(item) for item in failed]
    shown = items[:limit]
    if len(items) > limit:
        shown.append(f"and {len(items) - limit} more")
    return "\n".join(shown)


def _guard(job: str, messages: List[Tuple[str, dict]]) -> List[Tuple[str, dict]]:
    allowed = JOB_CHANNELS[job]
    for channel, _ in messages:
        if channel not in allowed:
            raise DiscordError(f"{job} cannot post to the {channel} channel")
    return messages


def messages_for(job: str, code: int, result: Optional[dict], *,
                 seconds: Optional[float] = None, when: Optional[int] = None,
                 watch_state: Optional[dict] = None) -> List[Tuple[str, dict]]:
    """The messages one job run should post. ``code`` is the job's own exit code.

    A run that found nothing and did not fail returns no messages.
    """
    if job not in JOB_CHANNELS:
        raise DiscordError(f"no Discord message is defined for the job {job!r}")
    if job == "research-watch":
        return watch_messages(
            code, result, watch_state or {}, seconds=seconds, when=when)[0]
    result = result or {}
    when = int(time.time()) if when is None else int(when)
    failed = code != 0 or result.get("status") == "failed"
    error = str(result.get("error") or "") if failed else ""

    if job == "research-fills":
        messages = _fills(result, seconds, when)
    elif job == "research-portfolio":
        messages = _portfolio(result, failed, seconds, when, error)
    elif job == "research-copy":
        messages = _copy(result, failed, seconds, when, error)
    elif job == "research-scan":
        messages = _scan(result, failed, seconds, when, error)
    elif job == "research-score":
        messages = _score(result, failed, seconds, when, error)
    elif job == "research-announce":
        messages = _announce(result, failed, seconds, when, error)
    elif job == "research-daily":
        messages = _daily(result, failed, seconds, when, error)
    elif job == "research-seed":
        messages = _seed(result, failed, seconds, when, error)
    elif job == "congress-sync":
        messages = _sync(result, failed, seconds, when, error)
    elif job == "research-quiver":
        messages = _quiver(result, failed, seconds, when, error)
    elif job == "research-backup":
        messages = _backup(result, failed, seconds, when, error)
    elif job == "research-status":
        messages = _status(result, seconds, when)
    else:
        messages = [("ops", ops_message(
            job,
            "containers older than 24h pruned" if not failed
            else f"exited with code **{code}**",
            ok=not failed, quiet=not failed, seconds=seconds, error=error,
            when=when))]
    return _guard(job, messages)


def _copy(result, failed, seconds, when, error):
    rows = int(result.get("rows_written") or 0)
    if not failed and (result.get("status") == "off" or rows == 0):
        return []
    messages = [("ops", ops_message(
        "research-copy", f"{_count(rows, 'decision')} recorded",
        ok=not failed, seconds=seconds, error=error, when=when))]
    preview = result.get("preview") or []
    if preview and not failed:
        lines = []
        for item in preview[:WATCH_EVENTS_SHOWN]:
            lines.append(
                f"**{escape_text(str(item.get('ticker') or ''))}** "
                f"{escape_text(str(item.get('member') or ''))} · "
                f"**{format_money(float(item.get('dollars') or 0))}**")
        accepted = int(result.get("accepted") or len(preview))
        if accepted > len(lines):
            lines.append(f"and **{accepted - len(lines)}** more")
        messages.append(("signals", signal_message(
            "Congressional copy", lines, emoji="🛒", color=GREEN,
            source="nemo research-copy", when=when)))
    return messages


def _scan(result, failed, seconds, when, error):
    candidates = result.get("candidates") or []
    superseded = result.get("superseded") or {}
    dropped = set(superseded.get("dropped") or [])
    filed = [row for row in candidates if row.get("ticker") not in dropped]
    if not failed and not filed and not superseded:
        return []
    warning = bool(superseded) and not failed
    note = (f"⚠️ {escape_text(str(superseded.get('note') or 'an earlier decision stands'))}"
            if superseded else "")
    messages = [("ops", ops_message(
        "research-scan", f"{_count(len(filed), 'order')} filed",
        ok=not failed, warning=warning, seconds=seconds, error=error,
        detail=note, when=when))]
    if filed and not failed:
        lines = []
        for row in filed[:WATCH_EVENTS_SHOWN]:
            lines.append(
                f"**{escape_text(str(row.get('ticker') or ''))}** "
                f"{escape_text(str(row.get('side') or ''))} · "
                f"SUE **{_fixed(row.get('sue'), 2)}**")
        if len(filed) > len(lines):
            lines.append(f"and **{len(filed) - len(lines)}** more")
        if note:
            lines.append(note)
        messages.append(("signals", signal_message(
            "Scanner book", lines, emoji="⚠️" if warning else "📋",
            color=YELLOW if warning else BLURPLE,
            source="nemo research-scan", when=when)))
    return messages


def _events_message(events: Sequence[Any], count: int, when: int) -> dict:
    lines = []
    links = []
    shown = events[:WATCH_EVENTS_SHOWN]
    for event in shown:
        if isinstance(event, dict):
            ticker = str(event.get("subject_ticker") or "")
            form = str(event.get("form") or "13D")
            lines.append(
                f"**{escape_text(ticker)}** {escape_text(form)} · "
                f"{escape_text(str(event.get('filer_name') or ''))}")
            links.append((f"{ticker} {form}"[:LABEL_LIMIT], event.get("url")))
        else:
            lines.append(escape_text(str(event)))
    total = max(count, len(events))
    if shown and total > len(shown):
        lines.append(f"and **{total - len(shown)}** more")
    return signal_message(
        "Activist filing", lines or [_count(count, "new filing")],
        emoji="📋", color=BLURPLE, source="nemo research-watch",
        links=links, when=when)


def watch_messages(code: int, result: Optional[dict], state: dict, *,
                   seconds: Optional[float] = None, when: Optional[int] = None
                   ) -> Tuple[List[Tuple[str, dict]], dict]:
    """The messages for one research-watch pass, and the alert state after it.

    #ops hears about a change of state only. The first failed pass notifies,
    later failed passes in a row post nothing, and the first good pass after a
    failure posts one silent line. The pass itself is `partial` in the steady
    state, so a coverage note is never a reason to post.
    """
    result = result or {}
    when = int(time.time()) if when is None else int(when)
    failing_since = int(state.get("failing_since") or 0)
    passes = int(state.get("failed_passes") or 0)
    failed = code != 0 or result.get("status") == "failed"
    messages: List[Tuple[str, dict]] = []
    if failed:
        if not failing_since:
            messages.append(("ops", ops_message(
                "research-watch", "the pass failed", ok=False, seconds=seconds,
                error=str(result.get("error") or ""), when=when)))
        new_state = {"failing_since": failing_since or when,
                     "failed_passes": passes + 1}
        return _guard("research-watch", messages), new_state
    if failing_since:
        messages.append(("ops", ops_message(
            "research-watch",
            f"recovered after {_count(passes, 'failed pass')}, "
            f"failing since <t:{failing_since}:R>",
            ok=True, seconds=seconds, when=when)))
    events = result.get("events") or []
    count = int(result["new_events"] if result.get("new_events") is not None
                else len(events))
    if count:
        messages.append(("signals", _events_message(events, count, when)))
    return _guard("research-watch", messages), {}


def _score(result, failed, seconds, when, error):
    sample = int(result.get("sample") or 0)
    if not failed and sample == 0:
        return []
    messages = [("ops", ops_message(
        "research-score", f"{_count(sample, 'trade')} scored",
        ok=not failed, seconds=seconds, error=error, when=when))]
    if not failed:
        usable = bool(result.get("coefficient_usable"))
        if result.get("drift_bps_per_sue") is not None:
            drift_text = f"**{float(result['drift_bps_per_sue']):.1f} bp** per SUE"
        elif result.get("drift_bps_per_tail") is not None:
            drift_text = f"**{float(result['drift_bps_per_tail']):.1f} bp** per tail"
        else:
            drift_text = "no coefficient"
        hit = result.get("hit_rate")
        hit_text = "" if hit is None else f" · hit rate **{format_percent(float(hit))}**"
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
    failures = [str(item) for item in result.get("failed") or []]
    if failed:
        return [("ops", ops_message(
            "research-announce", f"{_count(written, 'announcement date')} recorded",
            ok=False, seconds=seconds, error=error or _failure_lines(failures),
            when=when))]
    if written == 0 and not failures:
        return []
    warning = bool(failures)
    messages = [("ops", ops_message(
        "research-announce", f"{_count(written, 'announcement date')} recorded",
        ok=True, warning=warning, seconds=seconds,
        detail=f"⚠️ {_count(len(failures), 'lookup')} failed" if warning else "",
        error=_failure_lines(failures), when=when))]
    if written:
        messages.append(("signals", signal_message(
            "Earnings dates",
            [f"{_count(written, 'date')} recorded from Item 2.02"],
            emoji="📋", color=BLURPLE, source="nemo research-announce", when=when)))
    return messages


def _seed(result, failed, seconds, when, error):
    written = int(result.get("written") or 0)
    failures = [str(item) for item in result.get("failed") or []]
    summary = f"{_count(written, 'quarter')} recorded"
    if failed:
        return [("ops", ops_message(
            "research-seed", summary, ok=False, seconds=seconds,
            error=error or _failure_lines(failures), when=when))]
    if failures:
        return [("ops", ops_message(
            "research-seed", summary, ok=True, warning=True, seconds=seconds,
            detail=f"⚠️ {_count(len(failures), 'name')} failed",
            error=_failure_lines(failures), when=when))]
    if written == 0:
        return []
    return [("ops", ops_message(
        "research-seed", summary, ok=True, seconds=seconds, when=when))]


def _sync(result, failed, seconds, when, error):
    parsed = int(result.get("parsed") or 0)
    if not failed and parsed == 0:
        return []
    return [("ops", ops_message(
        "congress-sync", f"{_count(parsed, 'filing')} parsed", ok=not failed,
        seconds=seconds, error=error, when=when))]


def _quiver(result, failed, seconds, when, error):
    written = int(result.get("written") or 0)
    failures = result.get("failed") or {}
    summary = f"{_count(written, 'new record')} from QuiverQuant"
    if failures:
        summary += f", {_count(len(failures), 'feed')} failed"
    detail = error or "\n".join(
        f"{name}: {reason}" for name, reason in failures.items())
    return [("ops", ops_message(
        "research-quiver", summary, ok=not failed, quiet=not failed,
        seconds=seconds, error=detail if failed else "", when=when))]


def _backup(result, failed, seconds, when, error):
    removed = int(result.get("removed") or 0)
    size = format_gib(float(result.get("bytes") or 0))
    return [("ops", ops_message(
        "research-backup",
        f"**{size}** written, "
        f"{_count(removed, 'older copy', 'older copies')} removed",
        ok=not failed, seconds=seconds, error=error, when=when))]


def _status(result, seconds, when):
    """The one heartbeat. It always posts, silent when nothing needs attention."""
    attention = [str(item) for item in result.get("attention") or []]
    if attention:
        return [("ops", ops_message(
            "research-status",
            f"{_count(len(attention), 'item')} "
            f"{'needs' if len(attention) == 1 else 'need'} attention",
            ok=True, warning=True, seconds=seconds,
            error="\n".join(attention), when=when))]
    return [("ops", ops_message(
        "research-status", "**All clear**: nothing needs attention", ok=True,
        seconds=seconds, when=when))]


def _fills(result, seconds, when):
    """Both books summed. A rejection is a warning, not a failed run."""
    status = result.get("status")
    if status not in ("ok", "closed", "partial"):
        return [("ops", ops_message(
            "research-fills", "the run failed", ok=False, seconds=seconds,
            error=str(result.get("error") or f"unexpected status {status!r}"),
            when=when))]
    action = result.get("action")
    if action not in ("submit", "collect"):
        raise DiscordError(f"research-fills posts submit or collect, not {action!r}")
    copy = result.get("copy") or {}
    key, verb = ("submitted", "submitted") if action == "submit" else ("updated", "updated")
    earnings = int(result.get(key) or 0)
    copy_count = int(copy.get(key) or 0)
    count = earnings + copy_count
    rejected = int(result.get("rejected") or 0) + int(copy.get("rejected") or 0)
    skipped = int(result.get("skipped") or 0) + int(copy.get("skipped") or 0)
    missed = int(result.get("missed") or 0) + int(copy.get("missed") or 0)
    warning = status == "partial" or rejected > 0 or missed > 0
    if not warning and count == 0:
        return []
    lines = [f"{_count(count, 'order')} {verb}"]
    if copy_count:
        lines.append(f"Earnings **{earnings:,}** · Copy **{copy_count:,}**")
    if skipped:
        lines.append(f"**{skipped:,}** skipped")
    if rejected:
        lines.append(f"⚠️ **{rejected:,}** rejected")
    if missed:
        lines.append(f"⚠️ **{missed:,}** missed")
    notes = [str(note) for note in result.get("notes") or []]
    if not notes:
        notes = [str(text) for text in (result.get("error"), copy.get("error")) if text]
    messages = [("ops", ops_message(
        "research-fills", "\n".join(lines), ok=True, warning=warning,
        seconds=seconds, error="\n".join(notes) if warning else "", when=when))]
    if count:
        messages.append(("signals", signal_message(
            "Paper fills", lines, emoji="⚠️" if warning else "🛒",
            color=YELLOW if warning else GREEN,
            source="nemo research-fills", when=when)))
    return messages


def _portfolio(result, failed, seconds, when, error):
    """The card every weekday, silent. A failed read goes to #ops instead."""
    if failed:
        return [("ops", ops_message(
            "research-portfolio", "the paper account could not be read",
            ok=False, seconds=seconds, error=error, when=when))]
    return [("signals", portfolio_message(result, when=when))]


def _position_line(position: dict) -> str:
    symbol = f"**{escape_text(str(position.get('symbol') or ''))}**"
    if position.get("side") == "short":
        symbol += " short"
    books = " · ".join(position.get("books") or []) or "no book"
    value = float(position.get("market_value") or 0)
    money = ("-" if value < 0 else "") + format_money(abs(value))
    gain = format_signed_percent(float(position.get("unrealized_plpc") or 0))
    return f"{symbol} · {books} · {money} · **{gain}**"


def portfolio_message(result: dict, *, when: Optional[int] = None,
                      source: str = "nemo research-portfolio",
                      suppress: bool = True) -> dict:
    """The paper account: its value and the day's change, its open positions
    largest first with the books that hold them, and each book's closed
    trades. The daily post and the /portfolio reply are this one card."""
    equity = float(result.get("equity") or 0)
    last = float(result.get("last_equity") or 0)
    positions = result.get("positions") or []
    facts = []
    # A new account has no previous close to measure the day from.
    if last > 0:
        change = equity - last
        facts.append(f"**Today** **{format_signed_money(change)}** "
                      f"(**{format_signed_percent(change / last)}**)")
    unrealized = sum(float(p.get("unrealized_pl") or 0) for p in positions)
    facts.append(f"**Cash** **{format_money(float(result.get('cash') or 0))}**"
                 f" · **Unrealized** **{format_signed_money(unrealized)}**")
    lines = [_count(len(positions), "open position")]
    lines.extend(_position_line(p) for p in positions[:WATCH_EVENTS_SHOWN])
    if len(positions) > WATCH_EVENTS_SHOWN:
        lines.append(f"and **{len(positions) - WATCH_EVENTS_SHOWN}** more")
    closed = []
    for name, book in (result.get("books") or {}).items():
        part = f"{escape_text(name)} {_count(int(book.get('closed') or 0), 'trade')}"
        if book.get("mean_return") is not None:
            part += f", mean **{format_signed_percent(book['mean_return'])}**"
        closed.append(part)
    if closed:
        lines.append("-# Closed: " + " · ".join(closed))
    return signal_message(
        f"Paper portfolio · {format_money(equity)}", facts, emoji="📋",
        color=BLURPLE, source=source, detail="\n".join(lines), when=when,
        suppress=suppress)


# ------------------------------------------------- the bot's on-demand cards
#
# Read by the owner on a phone, not by an engineer: plain words throughout,
# and none of Nemo's own terms (SUE, gate, drawdown, issuer, job names).

JOB_NAMES = {
    "congress_copy": "Congress copying",
    "daily_bars": "Price update",
    "newcomers": "Price history for new stocks",
    "universe": "Stock list update",
    "consensus": "Analyst estimate update",
    "scan": "Stock picking",
    "activist_watch": "Activist watch",
    "score": "Weekly scoring",
    "announce": "Earnings-date check",
    "seed": "Monthly estimate history",
    "fill_submit": "Sending orders",
    "fill_collect": "Checking orders",
    "quiver_feeds": "Trades and filings from QuiverQuant",
    "quiver_lists": "Weekly lists from QuiverQuant",
    # Not a fee, and the name says a count: the card prints this beside the
    # borrow line, and "Short interest" is the only wording that does not read
    # as though the book could price a short.
    "short_interest": "Short interest",
}

_GLANCE = {"never": "hasn't run yet", "running": "is still running",
           "off": "is turned off", "partial": "ran with gaps"}

COPY_REASONS = {
    "no_ticker": "no stock symbol",
    "below_amount_floor": "too small",
    "member_not_followed": "member not followed",
    "trade_too_old": "reported too late",
    "not_eligible": "stock not tracked",
    "already_held": "already owned",
    "duplicate_issuer": "same company twice",
    "book_full": "at the limit of open trades",
}

# The scanner's reasons, which are written for the record, said plainly.
# First match wins; a reason with no match is shown as written.
PICK_REASONS = (
    (r"^\|SUE\| ", "the surprise was too small"),
    (r"^surprise of .* under the", "the surprise was too small"),
    (r"into a tail, under the", "the surprise was too small"),
    (r"^sigma from \d+ quarters", "not enough earnings history"),
    (r"lag is .* under the", "its report timing does not fit the strategy"),
    (r"^already acted on", "already traded this earnings report"),
    (r"^signal unavailable", "its earnings could not be read"),
    (r"^no percentile computed|^no scaled surprise",
     "its earnings could not be compared"),
    (r"^(sizing|cost|borrow) unavailable", "the trade could not be priced"),
    (r"does not clear", "the expected gain was smaller than the cost"),
    (r"is the same issuer", "another pick is the same company"),
    (r"^ranked below the top", "other picks ranked higher"),
)

_TIMING = {"bmo": "before the market opened", "amc": "after the market closed",
           "dmh": "during the trading day"}

SURPRISE_NOTE = ("-# Surprise: how far earnings beat (+) or missed (−) what "
                 "was expected. The bigger it is, the bigger the surprise.")


def day(iso: Any, *, this_year: Optional[int] = None) -> str:
    """``Nov 10``, with the year only when it is not this one."""
    when = date.fromisoformat(str(iso)[:10])
    if this_year is None:
        this_year = datetime.now(timezone.utc).year
    text = f"{when.strftime('%b')} {when.day}"
    return text if when.year == this_year else f"{text}, {when.year}"


def _year(when: int) -> int:
    return datetime.fromtimestamp(when, timezone.utc).year


def _dollars(amount: Any) -> str:
    return f"${float(amount or 0):,.0f}"


def _surprise(value: Any) -> str:
    return f"surprise **{float(value):+.2f}**"


def _quarter(period: Any) -> str:
    match = re.fullmatch(r"(\d{4})Q(\d)", str(period or ""))
    return f"Q{match[2]} {match[1]}" if match else escape_text(str(period or ""))


def _pick_reason(reason: Any) -> str:
    text = str(reason or "")
    for pattern, plain in PICK_REASONS:
        if re.search(pattern, text):
            return plain
    return escape_text(text) if text else "no reason given"


def _copy_reason(reason: Any) -> str:
    text = str(reason or "")
    return COPY_REASONS.get(text, escape_text(text.replace("_", " ")))


def _job_is_clean(job: dict) -> bool:
    return (job.get("state") in ("ok", "closed")
            or (job.get("state") == "partial" and job.get("partial_is_normal")))


def _job_name(job: dict) -> str:
    return JOB_NAMES.get(job["job"], escape_text(job["job"].replace("_", " ")))


def _trouble(job: dict, year: int) -> str:
    """What went wrong with a job that needs a person, in one line."""
    name, state = _job_name(job), job.get("state")
    last = day(job["last_run"], this_year=year) if job.get("last_run") else "?"
    if state == "failed":
        error = (str(job.get("error") or "").strip().splitlines() or [""])[0]
        if len(error) > 120:
            error = error[:119] + "…"
        return f"❌ {name} failed on {last}" + (
            f": {escape_text(redact(error))}" if error else "")
    if state == "stale":
        return f"❌ {name} hasn't run since {last}"
    if state == "crashed":
        return f"❌ {name} stopped partway through on {last}"
    if state == "never":
        return f"❌ {name} has never run"
    return f"❌ {name}: {escape_text(str(state))}"


def status_card(report: dict, *, when: Optional[int] = None,
                source: str = "nemo /status") -> dict:
    """`research.status.collect` in plain words: what Nemo picked last, how
    close the safety stop is, how far the strategy is from being judged, and
    whether every job ran."""
    when = int(time.time()) if when is None else int(when)
    year = _year(when)
    attention = [str(item) for item in report.get("attention") or []]
    jobs = report.get("jobs") or []
    raised = {job["job"] for job in jobs
              if any(item.startswith(job["job"] + " ") for item in attention)}

    book = report.get("book") or {}
    if book.get("as_of"):
        shorts = int(book.get("shorts") or 0)
        longs = int(book.get("accepted") or 0) - shorts
        parts = []
        if longs:
            parts.append(f"**{longs:,}** stock{'s' if longs != 1 else ''} to buy")
        if shorts:
            parts.append(f"**{shorts:,}** to bet against" if longs else
                         f"**{shorts:,}** stock{'s' if shorts != 1 else ''} "
                         f"to bet against")
        facts = [f"**Latest picks** {' and '.join(parts) or 'none'}, chosen "
                 f"{day(book['as_of'], this_year=year)}"]
    else:
        facts = ["**Latest picks** none yet"]

    risk = report.get("risk") or {}
    latest = risk.get("latest")
    if risk.get("halted"):
        tripped = (risk.get("tripped") or {}).get("as_of_date")
        facts.append(f"**Safety stop** ❌ on since {day(tripped, this_year=year)}"
                     f": no new trades until it is reset")
    elif latest:
        down = float(risk.get("drawdown_dollars") or 0)
        where = (f"down **{_dollars(down)}** from its best" if down > 0
                 else "at its best so far")
        facts.append(f"**Safety stop** not triggered · {where}; it pauses new "
                     f"trades at **{_dollars(latest['limit_dollars'])}** down")
    else:
        facts.append("**Safety stop** nothing traded yet")

    gate = report.get("gate")
    if not gate:
        facts.append("**Track record** not checked yet; it is checked each "
                     "Saturday")
    elif gate.get("passed"):
        facts.append(f"**Track record** proven on "
                     f"**{int(gate['measured_trades']):,}** finished trades")
    else:
        facts.append(f"**Track record** **{int(gate['measured_trades']):,}** of "
                     f"the **{int(gate['min_trades']):,}** finished trades "
                     f"needed to judge the strategy")

    copy = report.get("copy") or {}
    line = (f"**Congress copying** **{int(copy.get('open') or 0):,}** open "
            f"trades · **{int(copy.get('closed') or 0):,}** closed")
    if copy.get("mean_return") is not None:
        line += f", average **{format_signed_percent(copy['mean_return'])}**"
    facts.append(line)

    clean = [job for job in jobs if _job_is_clean(job)]
    detail = []
    if clean and len(clean) == len(jobs):
        detail.append(f"✅ All **{len(jobs):,}** jobs ran on time")
    elif clean:
        detail.append(f"✅ **{len(clean):,}** of **{len(jobs):,}** jobs ran "
                      f"on time")
    glance = [f"{_job_name(job)} {_GLANCE.get(job['state'], escape_text(str(job['state'])))}"
              for job in jobs
              if job["job"] not in raised and not _job_is_clean(job)]
    if glance:
        detail.append("-# " + " · ".join(glance))
    detail.extend(_trouble(job, year) for job in jobs if job["job"] in raised)
    for item in attention:
        if any(item.startswith(job + " ") for job in raised):
            continue
        if item.startswith("risk switch"):
            continue    # the safety stop line above says it
        detail.append(f"❌ {escape_text(item)}")

    if attention:
        count = len(attention)
        headline = (f"**{count:,}** thing{'s' if count != 1 else ''} "
                    f"need{'s' if count == 1 else ''} a look")
        emoji, color = "⚠️", YELLOW
    else:
        headline, emoji, color = "All good", "✅", GREEN
    return signal_message(headline, facts, emoji=emoji, color=color,
                          source=source, detail="\n".join(detail), when=when)


def _price(value: Any) -> str:
    return format_money(float(value))


def _leg_outcome(leg: Optional[dict]) -> str:
    """Which count a leg adds to on the /orders card."""
    if leg is None:
        return "not sent"
    status = leg.get("status")
    filled = float(leg.get("filled_qty") or 0)
    if status == "filled":
        return "done"
    if status in ("expired", "canceled"):
        return "partly done" if filled > 0 else "not done"
    if status == "rejected":
        return "refused"
    if status == "skipped":
        return "skipped"
    return "waiting"


def _verbs(leg: dict) -> Tuple[str, str]:
    """What the leg does, done and not done: buy, bet against, sell, buy back."""
    entry = leg.get("leg", "entry") == "entry"
    if leg.get("side") == "sell":
        return ("sold short", "not sold short") if entry else ("sold", "not sold")
    return ("bought", "not bought") if entry else ("bought back", "not bought back")


def _shares(amount: Any) -> str:
    value = abs(float(amount or 0))
    number = f"{value:,.0f}" if value.is_integer() else f"{value:,.4f}".rstrip("0")
    return f"**{number}** share" + ("" if value == 1 else "s")


def _leg_text(leg: Optional[dict]) -> str:
    """One leg in plain words: what went through, at what price, or why not."""
    outcome = _leg_outcome(leg)
    if outcome == "not sent":
        return "not sent yet"
    done, not_done = _verbs(leg)
    filled = float(leg.get("filled_qty") or 0)
    if outcome == "done":
        return f"{done} {_shares(filled)} at **{_price(leg['filled_price'])}**"
    rest = "expired" if leg.get("status") == "expired" else "was canceled"
    if outcome == "partly done":
        return (f"{done} **{filled:,.0f}** of {_shares(leg['qty'])} at "
                f"**{_price(leg['filled_price'])}**; the rest {rest}")
    if outcome == "not done":
        return (f"{not_done}: the order expired at the open"
                if leg.get("status") == "expired"
                else f"{not_done}: the order was canceled")
    if outcome in ("refused", "skipped"):
        reason = escape_text(str(leg.get("reason") or "no reason given"))
        return ("the broker refused it: " if outcome == "refused"
                else "skipped: ") + reason
    text = "sent, waiting for the market"
    if filled > 0:
        text += f" · {done} **{filled:,.0f}** of {_shares(leg['qty'])} so far"
    return text


_TALLY = ("done", "partly done", "not done", "refused", "skipped", "waiting",
          "not sent")


def _shown(lines: List[str], items: Sequence[Any], line) -> None:
    lines.extend(line(item) for item in items[:WATCH_EVENTS_SHOWN])
    if len(items) > WATCH_EVENTS_SHOWN:
        lines.append(f"and **{len(items) - WATCH_EVENTS_SHOWN}** more")


def _side_word(side: Any) -> str:
    return "bet against" if side == "short" else "buy"


def orders_card(data: dict, *, when: Optional[int] = None,
                source: str = "nemo /orders") -> dict:
    """What each strategy picked last, and what each order did at the
    broker, in plain words."""
    when = int(time.time()) if when is None else int(when)
    year = _year(when)
    earnings, copy = data.get("earnings"), data.get("copy")
    if not earnings and not copy:
        return signal_message("No orders yet",
                              ["Nemo has not picked any trades yet."],
                              emoji="💤", source=source, when=when)
    facts: List[str] = []
    detail: List[str] = []
    surprises = False
    if earnings:
        orders = earnings.get("orders") or []
        exits = earnings.get("exits") or []
        line = (f"**Earnings strategy** **{len(orders):,}** "
                f"stock{'s' if len(orders) != 1 else ''} picked on "
                f"{day(earnings['decided'], this_year=year)}")
        if exits:
            line += (f" · **{len(exits):,}** earlier "
                     f"trade{'s' if len(exits) != 1 else ''} closing")
        facts.append(line)
        tally = {outcome: 0 for outcome in _TALLY}
        for order in orders:
            tally[_leg_outcome(order.get("entry"))] += 1
        all_buys = all(order.get("side") != "short" for order in orders)
        words = {"done": "bought", "partly done": "partly bought",
                 "not done": "not bought", "not sent": "not sent yet"} \
            if all_buys else {"not sent": "not sent yet"}
        facts.append(" · ".join(f"**{n:,}** {words.get(outcome, outcome)}"
                                for outcome, n in tally.items() if n))

        def order_line(order):
            nonlocal surprises
            sue = ""
            if order.get("sue") is not None:
                sue, surprises = f" · {_surprise(order['sue'])}", True
            return (f"**{escape_text(str(order['ticker']))}** "
                    f"{_side_word(order.get('side'))} "
                    f"**{_dollars(order.get('target_dollars'))}**{sue} · "
                    f"{_leg_text(order.get('entry'))}")

        _shown(detail, orders, order_line)
        _shown(detail, exits, lambda leg: (
            f"**{escape_text(str(leg['ticker']))}** closing an earlier trade · "
            f"{_leg_text(leg)}"))
    if copy:
        copied = copy.get("orders") or []
        seen = int(copy.get("seen") or 0)
        trades = ("trade by a member of Congress" if seen == 1
                  else "trades by members of Congress")
        facts.append(
            f"**Congress copying** **{seen:,}** {trades} seen "
            f"{day(copy['decided'], this_year=year)} · "
            + (f"**{len(copied):,}** copied" if copied else "none copied"))
        _shown(detail, copied, lambda order: (
            f"**{escape_text(str(order['ticker']))}** copying "
            f"{escape_text(str(order.get('member') or 'a member'))} · "
            f"**{_dollars(order.get('target_dollars'))}** · "
            f"{_leg_text(order.get('entry'))}"))
        refused = sorted((copy.get("refused") or {}).items(),
                         key=lambda item: (-item[1], str(item[0])))
        if refused:
            detail.append("-# Not copied: " + " · ".join(
                f"{_copy_reason(reason)} **{n:,}**" for reason, n in refused))
    if surprises:
        detail.append(SURPRISE_NOTE)
    session = earnings["session"] if earnings else data.get("as_of")
    return signal_message(f"Orders for {day(session, this_year=year)}", facts,
                          source=source, detail="\n".join(detail), when=when)


def _held_by(books: Sequence[str]) -> str:
    names = {"Earnings": "the earnings strategy", "Copy": "congress copying"}
    books = list(books or [])
    if not books:
        return "not from either strategy"
    if len(books) > 1:
        return "from both strategies"
    return f"from {names.get(books[0], escape_text(books[0]))}"


def _filing_said(form: str) -> str:
    if "/A" in form:
        return "updated their stake"
    if "13G" in form:
        return "reported a passive stake of 5% or more"
    return "reported a stake of 5% or more"


def ticker_card(data: dict, *, when: Optional[int] = None,
                source: str = "nemo /ticker") -> dict:
    """What Nemo has on one stock, in plain words: what you own, whether
    Nemo picked it and why or why not, its orders, its earnings reports,
    activist investors and congress trades."""
    when = int(time.time()) if when is None else int(when)
    year = _year(when)
    symbol = escape_text(str(data.get("symbol") or ""))
    position, error = data.get("position"), data.get("position_error")
    signal = data.get("signal")
    orders = data.get("orders") or []
    releases = data.get("releases") or []
    filings = data.get("filings") or []
    congress = data.get("congress") or []

    if not (position or error or signal or orders or releases or filings
            or congress):
        return signal_message(
            f"{symbol} · Nemo has nothing on it",
            [f"No picks, orders, earnings reports, activist filings or "
             f"congress trades for **{symbol}**."], emoji="💤", source=source,
            when=when)

    facts: List[str] = []
    if position:
        worth = format_money(abs(float(position.get("market_value") or 0)))
        holds = ("you're betting against" if position.get("side") == "short"
                 else "you own")
        headline = (f"{symbol} · {holds} {_shares(position.get('qty'))} "
                    f"worth **{worth}**")
        facts.append(
            f"**Gain/loss** "
            f"**{format_signed_money(float(position.get('unrealized_pl') or 0))}** "
            f"(**{format_signed_percent(float(position.get('unrealized_plpc') or 0))}**)"
            f" · {_held_by(position.get('books'))}")
    elif error:
        headline = f"{symbol} · couldn't check your account"
        facts.append(f"⚠️ The paper account could not be read: "
                     f"{escape_text(redact(str(error)))}")
    else:
        headline = f"{symbol} · you don't own it"

    surprises = False
    if signal:
        sue = ""
        if signal.get("sue") is not None:
            sue, surprises = f" · {_surprise(signal['sue'])}", True
        outcome = (f"picked to {_side_word(signal.get('side'))}"
                   if signal.get("accepted")
                   else f"not traded: {_pick_reason(signal.get('reason'))}")
        facts.append(f"**Latest check** "
                     f"{day(signal['as_of_date'], this_year=year)}{sue} · "
                     f"{outcome}")
    if releases:
        last = releases[0]
        line = (f"**Last earnings report** "
                f"{day(last['announced_date'], this_year=year)} "
                f"({_quarter(last['fiscal_period'])})")
        if _TIMING.get(last.get("timing")):
            line += f", {_TIMING[last['timing']]}"
        facts.append(line)

    detail: List[str] = []
    if len(releases) > 1:
        detail.append("-# Earlier reports: " + " · ".join(
            day(r["announced_date"], this_year=year) for r in releases[1:]))
    if orders:
        detail.append("**Orders**")
        for order in orders:
            line = (f"{day(order['as_of_date'], this_year=year)} "
                    f"{_side_word(order.get('side'))} "
                    f"**{_dollars(order.get('target_dollars'))}** · "
                    f"{_leg_text(order.get('entry'))}")
            if order.get("exit"):
                line += f" · then {_leg_text(order['exit'])}"
            detail.append(line)
    links = []
    if filings:
        detail.append("**Activist investors**")
        for filing in filings:
            form = str(filing.get("form") or "13D")
            when_filed = day(filing["filing_date"], this_year=year)
            detail.append(
                f"{when_filed} · "
                f"{escape_text(str(filing.get('filer_name') or 'an investor'))} · "
                f"{_filing_said(form)}")
            if filing.get("url"):
                links.append((f"Filing {when_filed}"[:LABEL_LIMIT],
                              filing["url"]))
    if congress:
        detail.append("**Congress**")
        for trade in congress:
            verdict = (f"copied with **{_dollars(trade.get('target_dollars'))}**"
                       if trade.get("accepted")
                       else f"not copied: {_copy_reason(trade.get('reason'))}")
            amount = ""
            if trade.get("amount_min") is not None:
                amount = (f" · ${int(trade['amount_min']):,}"
                          f"–${int(trade.get('amount_max') or 0):,}")
            traded = (day(trade["transaction_date"], this_year=year)
                      if trade.get("transaction_date") else "?")
            detail.append(
                f"{traded} · {escape_text(str(trade.get('member') or ''))} "
                f"({escape_text(str(trade.get('chamber') or '').title())})"
                f"{amount} · {verdict}")
    if surprises:
        detail.append(SURPRISE_NOTE)
    return signal_message(headline, facts, source=source, links=links,
                          detail="\n".join(detail), when=when)


ANSWER_TOOLS_NAMED = 4


def answer_messages(result: dict, *, when: Optional[int] = None,
                    source: str = "nemo /ask") -> List[dict]:
    """Claude's answer to /ask: the question, the answer as Claude wrote it,
    and in the footer the model, the tools it called and the tokens it used.

    A long answer is split at paragraphs into numbered cards, never cut. An
    answer that was declined or stopped short says so above what it had.
    """
    when = int(time.time()) if when is None else int(when)
    stop = result.get("stop_reason")
    text = (result.get("answer") or "").strip()
    notes = []
    if stop == "refusal":
        notes.append("⚠️ Claude declined to answer this question.")
    elif stop == "max_turns":
        notes.append(f"⚠️ Stopped after **{int(result.get('rounds') or 0):,}** "
                     f"turns before a final answer; this is what it had.")
    elif stop == "max_tokens":
        notes.append("⚠️ The answer reached its length limit and stops "
                     "part-way.")
    if not text and not notes:
        notes.append("⚠️ Claude returned no text.")
    warning = bool(notes)

    tools = list(result.get("tools") or [])
    named = (" · ".join(tools) if len(tools) <= ANSWER_TOOLS_NAMED
             else _count(len(tools), "tool", bold=False))
    footer_source = (
        f"{source} · {result.get('model')} · {named or 'no tools'} · "
        f"{int(result.get('input_tokens') or 0):,} tokens in · "
        f"{int(result.get('output_tokens') or 0):,} out")
    asked = f"-# Asked: {escape_text(' '.join(str(result.get('question') or '').split()))}"
    body = "\n\n".join(part for part in ("\n".join(notes), text) if part)
    budget = (TEXT_LIMIT - len(_footer(footer_source, when)) - len(asked)
              - 80)
    pages = _pages(body, budget)
    total = len(pages)
    emoji, color = ("⚠️", YELLOW) if warning else ("📋", BLURPLE)
    return [_assemble(
        emoji=emoji,
        headline="Nemo's answer" if total == 1 else f"Nemo's answer ({i}/{total})",
        color=color, facts=asked if i == 1 else "", detail=page, error="",
        links=(), source=footer_source, when=when, suppress=False)
        for i, page in enumerate(pages, start=1)]


def _daily(result, failed, seconds, when, error):
    if "days" in result and isinstance(result["days"], dict):
        days = result["days"]
        return [("ops", ops_message(
            "research-daily", f"{_count(len(days), 'backfill day')}",
            ok=not failed, quiet=False, seconds=seconds, error=error, when=when))]
    stages = {key: value for key, value in result.items()
              if isinstance(value, dict) and "status" in value}
    written = sum(int(value.get("written") or 0) for value in stages.values())
    failed_names = [name for name, value in stages.items() if value.get("status") == "failed"]
    gaps = result.get("gaps") or []
    summary = f"{_count(written, 'row')} recorded"
    if failed_names:
        summary += f", {_count(len(failed_names), 'stage')} failed"
    detail_error = error
    if failed_names and not detail_error:
        bits = []
        for name in failed_names:
            stage_error = stages[name].get("error") or "failed"
            bits.append(f"{name}: {stage_error}")
        detail_error = "\n".join(bits)
    if gaps:
        detail_error = (detail_error + "\n" if detail_error else "") + (
            f"{_count(len(gaps), 'coverage gap', bold=False)}")
    return [("ops", ops_message(
        "research-daily", summary, ok=not failed_names and not failed,
        quiet=False, seconds=seconds, error=detail_error, when=when))]


def cron_message(job: str, code: int, tail: str, *,
                 when: Optional[int] = None) -> dict:
    """What the cron failure branch posts for a job that did not post itself."""
    if code == LOCK_HELD:
        return ops_message(
            job, "skipped: the previous run still holds the lock", ok=True,
            warning=True, when=when)
    if code == POST_FAILED:
        return ops_message(
            job, "ran, but its Discord post failed", ok=True, warning=True,
            error=tail, when=when)
    return ops_message(
        job, f"exited with code **{code}**", ok=False, error=tail, when=when)


# --------------------------------------------------------- alert state, watch only

def watch_state_path() -> Path:
    """Beside the store, on the volume that outlives a run."""
    from research import pit_store

    return Path(pit_store.db_path()).with_name("discord_watch_state.json")


def load_watch_state(path) -> dict:
    path = Path(path)
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError as exc:
        raise DiscordError(f"{path} cannot be read: {exc.strerror or exc}") from None
    try:
        state = json.loads(raw)
    except ValueError:
        state = None
    if not isinstance(state, dict):
        raise DiscordError(
            f"{path} is not a JSON object; delete it to reset the "
            f"research-watch alert state")
    return state


def save_watch_state(path, state: dict) -> None:
    path = Path(path)
    try:
        if not state:
            path.unlink(missing_ok=True)
            return
        scratch = path.with_name(path.name + ".tmp")
        scratch.write_text(json.dumps(state), encoding="utf-8")
        os.replace(scratch, path)
    except OSError as exc:
        raise DiscordError(f"{path} cannot be written: {exc.strerror or exc}") from None


# ------------------------------------------------------------------ posting

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


def stopwatch() -> Callable[[], float]:
    """Start timing now. Call the result for the seconds since."""
    started = time.monotonic()
    return lambda: time.monotonic() - started


def finish(job: str, code: int, result: Optional[dict] = None, *,
           seconds: Optional[float] = None,
           transport: Optional[Callable[[str, dict], Any]] = None,
           state_path=None) -> int:
    """Post after the job has recorded its data, and return the exit code.

    The recorded rows stay. This function only changes the exit code: a job
    that succeeded and could not post exits ``POST_FAILED``. After a non-zero
    exit that it posted (or chose not to repeat), the process prints
    ``REPORTED`` last, so the cron failure branch posts nothing more.
    """
    when = int(time.time())
    try:
        new_state = None
        if job == "research-watch":
            path = state_path or watch_state_path()
            messages, new_state = watch_messages(
                code, result, load_watch_state(path), seconds=seconds, when=when)
        else:
            messages = messages_for(job, code, result, seconds=seconds, when=when)
        grouped: Dict[str, List[dict]] = {}
        for channel, payload in messages:
            grouped.setdefault(channel, []).append(payload)
        for channel, payloads in grouped.items():
            deliver(channel, payloads, transport=transport)
        if new_state is not None:
            save_watch_state(path, new_state)
    except DiscordError as exc:
        print(f"{job}: {redact(str(exc))}", file=sys.stderr, flush=True)
        return code if code != 0 else POST_FAILED
    if code != 0:
        print(REPORTED, file=sys.stderr, flush=True)
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


def edit_interaction_reply(application_id: str, token: str,
                           payload: dict) -> None:
    """Replace a deferred interaction's loading message with ``payload``.

    The token is the interaction's, and like a webhook URL it is never put
    in an error. Only a 200 that carries the saved message's id is an edit.
    """
    import httpx

    _check(payload)
    url = (f"{INTERACTION_API}/webhooks/{application_id}/{token}"
           f"/messages/@original?with_components=true")
    try:
        response = httpx.patch(url, json=payload, timeout=30)
    except Exception as exc:
        raise DiscordError(f"Discord edit failed: {type(exc).__name__}") from None
    if response.status_code >= 300:
        detail = _discord_detail(response)
        raise DiscordError(f"Discord edit returned {response.status_code}"
                           + (f": {detail}" if detail else ""))
    try:
        saved = response.json()
    except ValueError:
        saved = None
    if not isinstance(saved, dict) or not saved.get("id"):
        raise DiscordError(f"Discord answered the edit with {response.status_code} "
                           f"and no message id")


def follow_up_interaction(application_id: str, token: str,
                          payload: dict) -> None:
    """Post ``payload`` as a further private message on an interaction.

    For an answer longer than one card. Discord keeps the message only for
    the person who asked when it carries EPHEMERAL, so the flag is set here
    rather than trusted to the caller.
    """
    import httpx

    _check(payload)
    body = {**payload, "flags": EPHEMERAL | IS_COMPONENTS_V2}
    url = (f"{INTERACTION_API}/webhooks/{application_id}/{token}"
           f"?with_components=true")
    try:
        response = httpx.post(url, json=body, timeout=30)
    except Exception as exc:
        raise DiscordError(
            f"Discord follow-up failed: {type(exc).__name__}") from None
    if response.status_code >= 300:
        detail = _discord_detail(response)
        raise DiscordError(f"Discord follow-up returned {response.status_code}"
                           + (f": {detail}" if detail else ""))
    try:
        saved = response.json()
    except ValueError:
        saved = None
    if not isinstance(saved, dict) or not saved.get("id"):
        raise DiscordError(f"Discord answered the follow-up with "
                           f"{response.status_code} and no message id")


# ------------------------------------------------------------------- the log

def _log_lines(path: str, lines: int) -> List[str]:
    with open(path, encoding="utf-8", errors="replace") as handle:
        return list(collections.deque(handle, maxlen=lines))


def log_tail(path: str, lines: int = 100) -> str:
    try:
        content = _log_lines(path, lines)
    except OSError as exc:
        return f"{path} cannot be read: {exc.strerror or exc}"
    return redact("".join(content))


def already_reported(path: str) -> bool:
    """Whether the last run in the log ended by reporting its own failure."""
    try:
        content = _log_lines(path, 5)
    except OSError:
        return False
    last = [line.strip() for line in content if line.strip()]
    return bool(last) and last[-1] == REPORTED
