"""Components V2 posts for the scheduled jobs.

No network. A live check against the preview webhook is
`test_preview_accepts_every_message_type`, and it stays skipped until
DISCORD_PREVIEW_WEBHOOK_URL is set.
"""
from __future__ import annotations

import json
import os

import pytest

import research.discord_post as post


WHEN = 1_700_000_000
OPS = "https://hooks.example/ops"
SIGNALS = "https://hooks.example/signals"


@pytest.fixture
def webhooks(monkeypatch):
    monkeypatch.setenv("NEMO_DISCORD_OPS_WEBHOOK_URL", OPS)
    monkeypatch.setenv("NEMO_DISCORD_SIGNALS_WEBHOOK_URL", SIGNALS)


def _texts(payload):
    found = []

    def walk(nodes):
        for node in nodes:
            if node.get("type") == post.TEXT:
                found.append(node["content"])
            walk(node.get("components") or [])

    walk(payload["components"])
    return found


def _count(payload):
    total = 0

    def walk(nodes):
        nonlocal total
        for node in nodes:
            total += 1
            walk(node.get("components") or [])

    walk(payload["components"])
    return total


def test_a_routine_success_is_one_container_and_does_not_notify():
    payload = post.ops_message(
        "research-seed", "312 rows recorded", ok=True, quiet=True,
        seconds=31, when=WHEN)
    assert payload["flags"] == post.IS_COMPONENTS_V2 | post.SUPPRESS_NOTIFICATIONS
    assert "content" not in payload and "embeds" not in payload
    assert payload["allowed_mentions"] == {"parse": []}
    box = payload["components"][0]
    assert box["type"] == post.CONTAINER
    assert box["accent_color"] == post.BLURPLE
    text = "\n".join(_texts(payload))
    assert text.startswith("## 💤 research-seed")
    assert "**312** rows recorded" in text
    assert "**31s**" in text
    assert f"<t:{WHEN}:f>" in text and f"<t:{WHEN}:R>" in text
    assert "UTC" not in text
    assert _count(payload) <= 40


def test_a_failure_notifies_and_puts_the_error_in_a_code_block():
    payload = post.ops_message(
        "research-daily", "2 stages failed", ok=False, seconds=130,
        error="stage consensus: HTTP 503\n" + ("x" * 5000), when=WHEN)
    assert payload["flags"] == post.IS_COMPONENTS_V2
    assert payload["components"][0]["accent_color"] == post.RED
    text = "\n".join(_texts(payload))
    assert "## ❌ research-daily" in text
    assert "```" in text
    assert sum(len(item) for item in _texts(payload)) <= 4000


def test_a_signal_uses_a_link_button_inside_the_container():
    payload = post.signal_message(
        "Congressional copy",
        ["**AAA** bought", "**$5,000.00**"],
        emoji="🛒", color=post.GREEN, source="nemo research-copy",
        links=[("Filing", "https://www.sec.gov/Archives/edgar/data/1/a.htm")],
        when=WHEN)
    box = payload["components"][0]
    row = next(node for node in box["components"] if node["type"] == post.ACTION_ROW)
    button = row["components"][0]
    assert button == {
        "type": post.BUTTON, "style": 5, "label": "Filing",
        "url": "https://www.sec.gov/Archives/edgar/data/1/a.htm"}
    assert box["accent_color"] == post.GREEN
    assert payload["flags"] == post.IS_COMPONENTS_V2


def test_limits_are_refused_before_a_post():
    with pytest.raises(post.DiscordError):
        post.signal_message(
            "too many", ["line"], links=[(f"label{i}", "https://example.com/")
                                         for i in range(40)], when=WHEN)
    with pytest.raises(post.DiscordError):
        post.signal_message(
            "long label", ["line"],
            links=[("L" * 81, "https://example.com/")], when=WHEN)
    with pytest.raises(post.DiscordError):
        post.mask_link("docs", "https://user:pass@example.com/a")
    with pytest.raises(post.DiscordError):
        post.mask_link("docs", "ftp://example.com/a")


def test_a_masked_link_cannot_be_closed_by_the_url():
    masked = post.mask_link("filing", "https://example.com/a(b)<c> d[e]")
    assert "](<https://example.com/a%28b%29%3Cc%3E%20d%5Be%5D>)" in masked
    assert "(" not in masked.split("](", 1)[1]


def test_public_text_is_escaped_and_amounts_use_the_house_units():
    assert r"\*" in post.escape_text("2 * 2")
    assert post.format_money(12.336) == "$12.34"
    assert post.format_percent(0.38) == "38%"
    assert post.format_duration(0.41) == "0.4s"
    assert post.format_duration(31) == "31s"
    assert post.format_duration(130) == "2m 10s"
    assert post.format_duration(3720) == "1h 02m"
    assert post.format_gib(1.25 * 1024 ** 3) == "1.2 GiB"


def test_the_webhook_query_keeps_existing_parameters(webhooks):
    seen = {}

    def transport(url, body):
        seen["url"] = url
        seen["body"] = body

    post.deliver("ops", [post.ops_message(
        "research-seed", "1 row recorded", ok=True, quiet=True, when=WHEN)],
        transport=transport,
        url_for=lambda channel: OPS + "?thread_id=9")
    assert seen["url"] == OPS + "?thread_id=9&with_components=true&wait=true"
    assert seen["body"]["flags"] & post.IS_COMPONENTS_V2


def test_a_missing_webhook_names_the_setting_and_not_a_url(monkeypatch):
    monkeypatch.delenv("NEMO_DISCORD_OPS_WEBHOOK_URL", raising=False)
    with pytest.raises(post.DiscordError) as caught:
        post.deliver("ops", [post.ops_message(
            "research-seed", "1 row recorded", ok=True, quiet=True, when=WHEN)])
    assert "NEMO_DISCORD_OPS_WEBHOOK_URL" in str(caught.value)
    assert "http" not in str(caught.value)


def test_a_failed_post_does_not_repeat_the_webhook(webhooks):
    def transport(url, body):
        raise RuntimeError(f"boom {url} secret-token")

    with pytest.raises(post.DiscordError) as caught:
        post.deliver("ops", [post.ops_message(
            "research-seed", "1 row recorded", ok=True, quiet=True, when=WHEN)],
            transport=transport)
    text = str(caught.value)
    assert "hooks.example" not in text
    assert "secret-token" not in text
    assert "http" not in text


def test_finish_keeps_a_successful_exit_only_when_the_post_works(webhooks):
    assert post.finish("research-seed", 0, {"written": 4, "failed": []},
                       transport=lambda url, body: None) == 0

    def boom(url, body):
        raise RuntimeError("nope")

    assert post.finish("research-seed", 0, {"written": 4, "failed": []},
                       transport=boom) == post.POST_FAILED == 75


def test_each_job_renders_a_phone_message(webhooks):
    cases = {
        "research-copy": {"status": "ok", "rows_written": 1, "accepted": 1,
                          "preview": [{"ticker": "AAA", "member": "Ada Lovelace",
                                       "dollars": 5000}]},
        "research-scan": {"written": 2, "candidates": [
            {"ticker": "AAA", "side": "long", "sue": 1.5},
            {"ticker": "BBB", "side": "short", "sue": -0.4}]},
        "research-watch": {"status": "ok", "new_events": 1, "events": [{
            "subject_ticker": "INTC", "filer_name": "Example LP",
            "form": "SC 13D",
            "url": "https://www.sec.gov/Archives/edgar/data/1/x.htm"}]},
        "research-score": {"sample": 40, "hit_rate": 0.55,
                           "coefficient_usable": False,
                           "drift_bps_per_sue": 12.0},
        "research-announce": {"written": 3, "failed": [], "tickers": 10},
        "research-fills": {"status": "ok", "submitted": 2, "rejected": 0,
                           "action": "submit"},
        "research-daily": {"bars": {"status": "ok", "written": 300},
                           "consensus": {"status": "failed", "written": 0,
                                         "error": "HTTP 503"},
                           "gaps": ["2026-11-06"]},
        "research-seed": {"written": 80, "failed": [], "tickers": 20},
        "congress-sync": {"parsed": 40, "failures": 0},
        "research-backup": {"bytes": 2_200_000_000, "removed": 1},
        "research-status": {"attention": ["daily_bars last ran 4 days ago"]},
        "docker-prune": {"ok": True},
    }
    for job, result in cases.items():
        messages = post.messages_for(job, 0 if job != "research-daily" else 1,
                                     result, when=WHEN)
        assert messages, job
        encoded = json.dumps(messages)
        assert "discord.com/api/webhooks" not in encoded
        for channel, payload in messages:
            assert channel in ("ops", "signals")
            assert payload["flags"] & post.IS_COMPONENTS_V2
            assert _count(payload) <= 40
            assert sum(len(item) for item in _texts(payload)) <= 4000


def test_a_long_brief_splits_into_numbered_containers():
    body = "\n\n".join(
        f"## Part {i}\n" + ("Sentence about the tape. " * 80) for i in range(4))
    pages = post.brief_messages(body, when=WHEN)
    assert len(pages) > 1
    for index, payload in enumerate(pages, start=1):
        text = "\n".join(_texts(payload))
        assert f"({index}/{len(pages)})" in text
        assert sum(len(item) for item in _texts(payload)) <= 4000
        assert payload["components"][0]["type"] == post.CONTAINER


REAL_SEND = post._http_send  # the conftest fixture replaces the module attribute


def _examples():
    """One of every message type, built from fixtures, for the preview webhook."""
    examples = []
    examples.append(post.ops_message(
        "research-seed", "4 rows recorded", ok=True, quiet=True,
        seconds=2, when=WHEN))
    examples.append(post.ops_message(
        "research-daily", "1 stage failed", ok=False, seconds=12,
        error="consensus: HTTP 503", when=WHEN))

    def messages(job, code, result, **kwargs):
        return [payload for _, payload in post.messages_for(
            job, code, result, when=WHEN, **kwargs)]

    watch_events = [{
        "subject_ticker": f"T{i}", "filer_name": "Example LP", "form": "SC 13D",
        "url": f"https://www.sec.gov/Archives/edgar/data/{i}/x.htm"}
        for i in range(9)]
    examples += messages("research-copy", 0, {
        "status": "ok", "rows_written": 1, "accepted": 1,
        "preview": [{"ticker": "AAA", "member": "Ada Lovelace",
                     "dollars": 5000}]})
    examples += messages("research-scan", 0, {"written": 1, "candidates": [
        {"ticker": "AAA", "side": "long", "sue": 1.2}]})
    examples += messages("research-scan", 0, {
        "written": 2, "rejected": [],
        "candidates": [{"ticker": "AAA", "side": "long", "sue": 1.2},
                       {"ticker": "BBB", "side": "short", "sue": -1.1}],
        "superseded": {"dropped": ["BBB"],
                       "note": "2026-11-06 already holds a row for BBB"}})
    examples += messages("research-watch", 0, {
        "status": "partial", "new_events": 9, "events": watch_events})
    def watch(code, result, state):
        return [payload for _, payload in post.watch_messages(
            code, result, state, when=WHEN)[0]]

    examples += watch(1, {"status": "failed", "error": "EDGAR returned 503"}, {})
    examples += watch(0, {"status": "partial", "new_events": 0, "events": []},
                      {"failing_since": WHEN - 3600, "failed_passes": 3})
    examples += messages("research-score", 0, {
        "sample": 40, "hit_rate": 0.55, "coefficient_usable": True,
        "drift_bps_per_sue": 12.0})
    examples += messages("research-announce", 0, {
        "written": 3, "tickers": 5, "covered": 4,
        "failed": ["AAA: ReadTimeout: slow"], "outcomes": {}})
    examples += messages("research-fills", 1, {
        "action": "submit", "status": "partial", "submitted": 2, "skipped": 0,
        "rejected": 1, "missed": 0, "notes": ["1 rejected", "copy: 1 submitted"],
        "copy": {"status": "ok", "submitted": 1, "skipped": 0, "rejected": 0,
                 "missed": 0}})
    examples += messages("congress-sync", 0, {"parsed": 12, "failures": 0})
    examples += messages("research-backup", 0, {"bytes": 2_200_000_000,
                                                "removed": 1})
    examples += messages("research-status", 1, {
        "attention": ["score has no usable coefficient"]})
    examples += messages("research-status", 0, {"attention": []})
    examples.append(post.cron_message(
        "research-watch", post.LOCK_HELD, "", when=WHEN))
    examples.append(post.cron_message(
        "research-copy", 137, "Killed\nexit 137", when=WHEN))
    brief = ("## Pre-market\n\nAAA is unchanged.\n\n```python\n"
             + "\n".join(f"x = {i}" for i in range(700)) + "\n```")
    examples.extend(post.brief_messages(brief, when=WHEN))
    return examples


def test_every_example_message_fits_the_limits():
    examples = _examples()
    assert len(examples) >= 20
    for payload in examples:
        assert payload["flags"] & post.IS_COMPONENTS_V2
        assert _count(payload) <= post.COMPONENT_LIMIT
        assert sum(len(item) for item in _texts(payload)) <= post.TEXT_LIMIT


@pytest.mark.network
@pytest.mark.live_discord
@pytest.mark.skipif(not os.environ.get("DISCORD_PREVIEW_WEBHOOK_URL"),
                    reason="DISCORD_PREVIEW_WEBHOOK_URL is not set")
def test_preview_accepts_every_message_type():
    """One example of every message, to the preview webhook only, through the
    real transport: wait=true, a message id in every answer, 429 retried."""
    import time

    endpoint = post.webhook_endpoint(os.environ["DISCORD_PREVIEW_WEBHOOK_URL"])
    for payload in _examples():
        REAL_SEND(endpoint, payload)
        time.sleep(0.4)
