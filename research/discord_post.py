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
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
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

# Exit codes the cron lines read. A job that ran and could not post exits 75
# (EX_TEMPFAIL), and the chain still goes on to the scan. `flock -n -E` exits
# 99 when the lock is held.
POST_FAILED = 75
LOCK_HELD = 99

# The last line a process prints after a non-zero exit that it already
# reported to Discord, or deliberately did not repeat.
REPORTED = "nemo-discord: failure reported"

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
    "research-daily": ("ops",),
    "research-seed": ("ops",),
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

_WEBHOOK_URL = re.compile(
    r"https://(?:ptb\.|canary\.)?discord(?:app)?\.com/api/webhooks/\S+",
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
        "research-copy", f"**{rows:,}** decisions recorded",
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
        "research-scan", f"**{len(filed):,}** orders filed",
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
    if total > len(shown):
        lines.append(f"and **{total - len(shown)}** more")
    return signal_message(
        "Activist filing", lines or [f"**{count}** new filings"],
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
            f"recovered after **{passes}** failed passes, "
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
        "research-score", f"**{sample:,}** trades scored",
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
            "research-announce", f"**{written:,}** announcement dates recorded",
            ok=False, seconds=seconds, error=error or _failure_lines(failures),
            when=when))]
    if written == 0 and not failures:
        return []
    warning = bool(failures)
    messages = [("ops", ops_message(
        "research-announce", f"**{written:,}** announcement dates recorded",
        ok=True, warning=warning, seconds=seconds,
        detail=f"⚠️ **{len(failures)}** lookups failed" if warning else "",
        error=_failure_lines(failures), when=when))]
    if written:
        messages.append(("signals", signal_message(
            "Earnings dates",
            [f"**{written:,}** dates recorded from Item 2.02"],
            emoji="📋", color=BLURPLE, source="nemo research-announce", when=when)))
    return messages


def _seed(result, failed, seconds, when, error):
    written = int(result.get("written") or 0)
    failures = [str(item) for item in result.get("failed") or []]
    summary = f"**{written:,}** quarters recorded"
    if failed:
        return [("ops", ops_message(
            "research-seed", summary, ok=False, seconds=seconds,
            error=error or _failure_lines(failures), when=when))]
    if failures:
        return [("ops", ops_message(
            "research-seed", summary, ok=True, warning=True, seconds=seconds,
            detail=f"⚠️ **{len(failures)}** names failed",
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
        "congress-sync", f"**{parsed:,}** filings parsed", ok=not failed,
        seconds=seconds, error=error, when=when))]


def _backup(result, failed, seconds, when, error):
    removed = int(result.get("removed") or 0)
    size = format_gib(float(result.get("bytes") or 0))
    return [("ops", ops_message(
        "research-backup",
        f"**{size}** written, **{removed}** older copies removed",
        ok=not failed, seconds=seconds, error=error, when=when))]


def _status(result, seconds, when):
    """The one heartbeat. It always posts, silent when nothing needs attention."""
    attention = [str(item) for item in result.get("attention") or []]
    if attention:
        return [("ops", ops_message(
            "research-status", f"**{len(attention)}** items need attention",
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
    lines = [f"**{count:,}** orders {verb}"]
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
