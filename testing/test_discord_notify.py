"""The cron failure branch: ``python -m research.notify``.

The job's own process posts the failure it knows about and prints a marker
as its last line. This command posts only when the process did not: a crash
before the post, an out-of-memory kill, a docker error. It also says when a
run was skipped because its lock was held.
"""
from __future__ import annotations

import pytest

import research.discord_post as post
from research import notify


class Recorder:
    def __init__(self):
        self.posts = []

    def __call__(self, url, body):
        assert url.startswith("https://hooks.example/ops"), "only #ops is posted to"
        self.posts.append(body)

    def text(self):
        found = []

        def walk(nodes):
            for node in nodes:
                if node.get("type") == post.TEXT:
                    found.append(node["content"])
                walk(node.get("components") or [])

        for body in self.posts:
            walk(body["components"])
        return "\n".join(found)

    @property
    def color(self):
        return self.posts[-1]["components"][0]["accent_color"]


@pytest.fixture
def sent(monkeypatch):
    recorder = Recorder()
    monkeypatch.setattr(post, "_http_send", recorder)
    return recorder


def _log(tmp_path, *lines):
    path = tmp_path / "job.log"
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def test_a_crashed_run_posts_the_log_tail_as_a_failure(sent, tmp_path):
    log = _log(tmp_path, "Mon Nov  6 12:30:00 UTC 2026", "starting",
               "Traceback (most recent call last):", "KeyError: 'member_id'")
    assert notify.main(["--job", "research-copy", "--log", log]) == 0
    assert len(sent.posts) == 1
    text = sent.text()
    assert "## ❌ research-copy" in text
    assert "KeyError: 'member_id'" in text
    assert sent.color == post.RED
    assert not sent.posts[0]["flags"] & post.SUPPRESS_NOTIFICATIONS


def test_a_failure_the_process_already_reported_is_not_posted_again(sent, tmp_path):
    log = _log(tmp_path, "Mon Nov  6 12:30:00 UTC 2026", "{}", post.REPORTED)
    assert notify.main(["--job", "research-copy", "--log", log]) == 0
    assert sent.posts == []


def test_a_marker_from_an_earlier_run_does_not_hide_a_new_crash(sent, tmp_path):
    log = _log(tmp_path, "Mon Nov  6 12:30:00 UTC 2026", post.REPORTED,
               "Tue Nov  7 12:30:00 UTC 2026", "Killed")
    assert notify.main(["--job", "research-copy", "--log", log]) == 0
    assert len(sent.posts) == 1 and "Killed" in sent.text()


def test_trailing_blank_lines_after_the_marker_still_count_as_reported(sent, tmp_path):
    path = tmp_path / "job.log"
    path.write_text(f"date\n{post.REPORTED}\n\n\n")
    assert notify.main(["--job", "research-copy", "--log", str(path)]) == 0
    assert sent.posts == []


def test_an_unreadable_log_is_posted_as_the_error(sent, tmp_path):
    missing = str(tmp_path / "gone.log")
    assert notify.main(["--job", "research-copy", "--log", missing]) == 0
    assert "gone.log cannot be read" in sent.text()
    assert sent.color == post.RED


def test_no_log_still_posts_a_failure_that_says_so(sent):
    assert notify.main(["--job", "research-copy"]) == 0
    assert "left no log tail" in sent.text()


def test_only_the_last_hundred_lines_are_posted(sent, tmp_path):
    log = _log(tmp_path, *[f"line {i}" for i in range(300)])
    notify.main(["--job", "research-copy", "--log", log])
    text = sent.text()
    assert "line 299" in text and "line 150" not in text


def test_a_webhook_in_the_log_is_not_posted(sent, tmp_path):
    secret = "https://discord.com/api/webhooks/1/secret-token"
    log = _log(tmp_path, f"POST {secret} failed")
    notify.main(["--job", "research-copy", "--log", log])
    assert "secret-token" not in sent.text()


def test_a_held_lock_is_a_yellow_skip_and_not_a_failure(sent, tmp_path):
    log = _log(tmp_path, "Mon Nov  6 12:30:00 UTC 2026")
    assert notify.main(
        ["--job", "research-watch", "--log", log, "--code", str(post.LOCK_HELD)]) == 0
    text = sent.text()
    assert "skipped: the previous run still holds the lock" in text
    assert "❌" not in text and "```" not in text
    assert sent.color == post.YELLOW
    assert "⚠️" in text
    assert not sent.posts[0]["flags"] & post.SUPPRESS_NOTIFICATIONS


def test_a_job_whose_own_post_failed_is_not_called_a_failed_job(sent, tmp_path):
    log = _log(tmp_path, "date", "Discord post failed: ConnectError")
    assert notify.main(
        ["--job", "docker-prune", "--log", log, "--code", str(post.POST_FAILED)]) == 0
    text = sent.text()
    assert "container prune failed" not in text
    assert "Discord post failed" in text
    assert sent.color == post.YELLOW


def test_the_exit_code_of_a_failure_is_named(sent, tmp_path):
    log = _log(tmp_path, "date", "Killed")
    notify.main(["--job", "research-daily", "--log", log, "--code", "137"])
    assert "**137**" in sent.text()


def test_ok_posts_a_silent_success_for_the_container_prune(sent):
    assert notify.main(["--job", "docker-prune", "--ok"]) == 0
    assert len(sent.posts) == 1
    assert sent.posts[0]["flags"] & post.SUPPRESS_NOTIFICATIONS
    assert "docker-prune" in sent.text()
    assert sent.color in (post.GREEN, post.BLURPLE)


def test_ok_exits_75_when_the_post_fails(monkeypatch, capsys):
    def refuse(url, body):
        raise OSError("down")

    monkeypatch.setattr(post, "_http_send", refuse)
    assert notify.main(["--job", "docker-prune", "--ok"]) == post.POST_FAILED


def test_a_failure_post_that_cannot_be_sent_exits_75_and_says_why(
        monkeypatch, tmp_path, capsys):
    def refuse(url, body):
        raise OSError("down")

    monkeypatch.setattr(post, "_http_send", refuse)
    log = _log(tmp_path, "date", "boom")
    assert notify.main(["--job", "research-copy", "--log", log]) == post.POST_FAILED
    assert "OSError" in capsys.readouterr().err


def test_a_missing_webhook_exits_75_and_names_the_setting(
        monkeypatch, tmp_path, capsys):
    monkeypatch.delenv(post.OPS_SETTING)
    log = _log(tmp_path, "date", "boom")
    assert notify.main(["--job", "research-copy", "--log", log]) == post.POST_FAILED
    assert post.OPS_SETTING in capsys.readouterr().err


def test_the_unused_exit_code_option_is_gone():
    with pytest.raises(SystemExit):
        notify.main(["--job", "research-copy", "--exit-code", "0"])
