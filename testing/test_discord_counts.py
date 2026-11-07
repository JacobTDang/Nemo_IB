"""A count of one reads as singular in every Discord message.

``"**1** items need attention"`` and ``"**1** older copies removed"`` showed in
the live preview. Every message type is built here with each of its counts set
to one, and no noun after a count of one may be a plural.
"""
from __future__ import annotations

import re

import pytest

import research.discord_post as post

WHEN = 1_700_000_000

# Words that end in s and are not a plural noun after a count of one.
NOT_PLURAL = {"needs", "was", "has", "pass"}

BOLD_ONE = re.compile(r"\*\*1\*\*([ \w]*)")
PLAIN_ONE = re.compile(r"(?<![\w.,*:<-])1 ([a-z][a-z ]*)")


def texts(payload):
    found = []

    def walk(nodes):
        for node in nodes:
            if node.get("type") == post.TEXT:
                found.append(node["content"])
            walk(node.get("components") or [])

    walk(payload["components"])
    return found


def plurals_after_one(text):
    """Words ending in s that follow a count of one, bold or plain."""
    found = []
    for pattern in (BOLD_ONE, PLAIN_ONE):
        for match in pattern.finditer(text):
            found += [word for word in match.group(1).split()
                      if word.endswith("s") and word not in NOT_PLURAL]
    return found


def _messages(job, code, result, **kwargs):
    return post.messages_for(job, code, result, when=WHEN, **kwargs)


def _watch(code, result, state):
    return post.watch_messages(code, result, state, when=WHEN)[0]


EVENT = {"subject_ticker": "INTC", "form": "SC 13D", "filer_name": "Example LP",
         "url": "https://www.sec.gov/Archives/edgar/data/1/x.htm"}
COPY_BOOK = {"status": "ok", "submitted": 1, "skipped": 1, "rejected": 1,
             "missed": 1}

# name -> (messages built with every count at one, phrases that must appear)
CASES = {
    "copy": (lambda: _messages("research-copy", 0, {
        "status": "ok", "rows_written": 1, "accepted": 1,
        "preview": [{"ticker": "AAA", "member": "Ada", "dollars": 5000}]}),
        ["**1** decision recorded"]),
    "scan": (lambda: _messages("research-scan", 0, {
        "written": 1, "rejected": [], "candidates": [
            {"ticker": "AAA", "side": "long", "sue": 1.0}]}),
        ["**1** order filed"]),
    "watch events": (lambda: _watch(0, {
        "status": "ok", "new_events": 1, "events": [EVENT]}, {}), []),
    "watch events without a list": (lambda: _watch(0, {
        "status": "ok", "new_events": 1, "events": []}, {}),
        ["**1** new filing"]),
    "watch recovery": (lambda: _watch(0, {
        "status": "ok", "new_events": 0, "events": []},
        {"failing_since": WHEN - 600, "failed_passes": 1}),
        ["**1** failed pass"]),
    "score": (lambda: _messages("research-score", 0, {
        "sample": 1, "hit_rate": 1.0, "coefficient_usable": False,
        "drift_bps_per_sue": 1.0}),
        ["**1** trade scored"]),
    "announce": (lambda: _messages("research-announce", 0, {
        "written": 1, "failed": ["AAA: Boom: x"], "tickers": 2}),
        ["**1** announcement date recorded", "**1** lookup failed",
         "**1** date recorded from Item 2.02"]),
    "seed": (lambda: _messages("research-seed", 0, {
        "written": 1, "failed": ["AAA: Boom: x"], "tickers": 2}),
        ["**1** quarter recorded", "**1** name failed"]),
    "congress-sync": (lambda: _messages("congress-sync", 0, {"parsed": 1}),
                      ["**1** filing parsed"]),
    "backup": (lambda: _messages("research-backup", 0, {
        "bytes": 1024 ** 3, "removed": 1}),
        ["**1** older copy removed"]),
    "status": (lambda: _messages("research-status", 1, {
        "attention": ["score failed"]}),
        ["**1** item needs attention"]),
    "fills submit": (lambda: _messages("research-fills", 1, {
        "action": "submit", "status": "partial", "submitted": 1, "skipped": 1,
        "rejected": 1, "missed": 1, "notes": [], "copy": COPY_BOOK}),
        ["**2** orders submitted"]),
    "fills submit one order": (lambda: _messages("research-fills", 1, {
        "action": "submit", "status": "partial", "submitted": 1, "skipped": 0,
        "rejected": 1, "missed": 0, "notes": [],
        "copy": {"status": "ok", "submitted": 0, "skipped": 0, "rejected": 0,
                 "missed": 0}}),
        ["**1** order submitted"]),
    "fills collect": (lambda: _messages("research-fills", 0, {
        "action": "collect", "status": "ok", "asked": 1, "updated": 1,
        "notes": [], "copy": {"status": "ok", "asked": 0, "updated": 0}}),
        ["**1** order updated"]),
    "daily": (lambda: _messages("research-daily", 1, {
        "as_of": "2026-11-06",
        "daily_bars": {"status": "ok", "written": 1},
        "consensus": {"status": "failed", "written": 0, "error": "HTTP 503"},
        "gaps": ["2026-11-05"]}),
        ["**1** row recorded", "**1** stage failed", "1 coverage gap"]),
    "daily backfill": (lambda: _messages("research-daily", 0, {
        "gaps": ["2026-11-05"], "days": {"2026-11-05": {}}}),
        ["**1** backfill day"]),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_a_count_of_one_reads_as_singular(name):
    build, phrases = CASES[name]
    built = build()
    assert built, name
    combined = []
    for item in built:
        payload = item[1] if isinstance(item, tuple) else item
        combined += texts(payload)
    text = "\n".join(combined)
    assert plurals_after_one(text) == [], text
    for phrase in phrases:
        assert phrase in text, (phrase, text)


def test_the_helper_pluralises_and_groups_thousands():
    assert post._count(1, "order") == "**1** order"
    assert post._count(2, "order") == "**2** orders"
    assert post._count(0, "order") == "**0** orders"
    assert post._count(1200, "order") == "**1,200** orders"
    assert post._count(1, "older copy", "older copies") == "**1** older copy"
    assert post._count(3, "older copy", "older copies") == "**3** older copies"
    assert post._count(1, "coverage gap", bold=False) == "1 coverage gap"
    assert post._count(2, "coverage gap", bold=False) == "2 coverage gaps"


def test_a_count_above_a_thousand_keeps_its_separator_in_a_message():
    result = {"candidates": [{"ticker": f"T{i}", "side": "long", "sue": 1.0}
                             for i in range(1200)]}
    text = "\n".join(texts(_messages("research-scan", 0, result)[0][1]))
    assert "**1,200** orders filed" in text


def test_the_checker_catches_the_bugs_that_were_seen():
    assert plurals_after_one("**1** items need attention") == ["items"]
    assert plurals_after_one("**1** older copies removed") == ["copies"]
    assert plurals_after_one("1 coverage gaps") == ["gaps"]
    assert plurals_after_one("**1** announcement dates recorded") == ["dates"]
    assert plurals_after_one("**1** item needs attention") == []
    assert plurals_after_one("**2** orders filed") == []
