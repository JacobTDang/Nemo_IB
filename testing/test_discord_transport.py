"""The HTTP side of a Discord post: what is sent and how each answer is read.

No network. ``httpx.post`` is replaced with a function that returns a canned
response, so the 2xx, 4xx, 5xx and 429 paths run through the real
``_http_send``. The webhook URL is a secret: no test here lets it into a
message.
"""
from __future__ import annotations

import httpx
import pytest

import research.discord_post as post

SECRET_URL = "https://discord.com/api/webhooks/123/secret-token"

# The conftest fixture replaces ``post._http_send`` for every test. This module
# is imported first, so it keeps the real function.
REAL_SEND = post._http_send


def _response(status, body=None, headers=None, text=None):
    request = httpx.Request("POST", SECRET_URL)
    if text is not None:
        return httpx.Response(status, text=text, headers=headers, request=request)
    return httpx.Response(status, json=body, headers=headers, request=request)


@pytest.fixture
def answers(monkeypatch):
    """Queue the responses ``httpx.post`` returns. Returns the list of calls."""
    queue = []
    calls = []
    sleeps = []

    def fake_post(url, json=None, timeout=None):
        calls.append({"url": url, "json": json, "timeout": timeout})
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(post.time, "sleep", sleeps.append)
    queue_api = type("Answers", (), {})()
    queue_api.queue = queue
    queue_api.calls = calls
    queue_api.sleeps = sleeps
    return queue_api


def _send():
    REAL_SEND(post.webhook_endpoint(SECRET_URL), {"flags": 1 << 15})


def test_the_endpoint_asks_discord_to_wait_and_to_accept_components():
    endpoint = post.webhook_endpoint(SECRET_URL)
    assert endpoint.endswith("?with_components=true&wait=true")


def test_the_endpoint_keeps_other_parameters_and_does_not_repeat_its_own():
    endpoint = post.webhook_endpoint(
        SECRET_URL + "?thread_id=9&wait=false&with_components=false")
    assert endpoint.endswith("?thread_id=9&with_components=true&wait=true")


def test_a_saved_message_has_an_id_and_passes(answers):
    answers.queue.append(_response(200, {"id": "111", "channel_id": "2"}))
    _send()
    assert answers.calls[0]["json"] == {"flags": 1 << 15}


def test_a_2xx_answer_without_a_message_id_is_a_failed_post(answers):
    answers.queue.append(_response(204, text=""))
    with pytest.raises(post.DiscordError) as caught:
        _send()
    assert "message id" in str(caught.value)
    assert "secret-token" not in str(caught.value)


def test_a_2xx_answer_that_is_not_json_is_a_failed_post(answers):
    answers.queue.append(_response(200, text="<html>proxy</html>"))
    with pytest.raises(post.DiscordError) as caught:
        _send()
    assert "message id" in str(caught.value)


def test_a_4xx_answer_names_discords_code_and_message(answers):
    answers.queue.append(_response(400, {
        "code": 50035, "message": "Invalid Form Body"}))
    with pytest.raises(post.DiscordError) as caught:
        _send()
    text = str(caught.value)
    assert "400" in text and "50035" in text and "Invalid Form Body" in text
    assert "secret-token" not in text


def test_a_4xx_answer_that_is_not_json_still_names_the_status(answers):
    answers.queue.append(_response(404, text="Not Found"))
    with pytest.raises(post.DiscordError) as caught:
        _send()
    assert "404" in str(caught.value)


def test_a_5xx_answer_raises_and_is_not_retried(answers):
    answers.queue.append(_response(503, {"code": 0, "message": "overloaded"}))
    with pytest.raises(post.DiscordError) as caught:
        _send()
    assert "503" in str(caught.value) and "overloaded" in str(caught.value)
    assert len(answers.calls) == 1


def test_a_429_waits_out_retry_after_and_posts_again(answers):
    answers.queue.append(_response(429, {
        "message": "You are being rate limited.", "retry_after": 0.5,
        "global": False}))
    answers.queue.append(_response(200, {"id": "7"}))
    _send()
    assert answers.sleeps == [0.5]
    assert len(answers.calls) == 2


def test_a_long_retry_after_is_capped(answers):
    answers.queue.append(_response(429, {"retry_after": 600}))
    answers.queue.append(_response(200, {"id": "7"}))
    _send()
    assert answers.sleeps == [post.RETRY_AFTER_CAP]


def test_a_429_that_never_clears_raises_with_the_status(answers):
    for _ in range(post.SEND_ATTEMPTS):
        answers.queue.append(_response(429, {
            "message": "You are being rate limited.", "retry_after": 1}))
    with pytest.raises(post.DiscordError) as caught:
        _send()
    assert "429" in str(caught.value)
    assert len(answers.calls) == post.SEND_ATTEMPTS


def test_a_429_without_retry_after_raises_instead_of_guessing(answers):
    answers.queue.append(_response(429, {"message": "slow down"}))
    with pytest.raises(post.DiscordError) as caught:
        _send()
    assert "429" in str(caught.value)
    assert answers.sleeps == []


def test_a_connection_error_names_its_type_and_not_the_url(answers):
    answers.queue.append(httpx.ConnectError(f"cannot reach {SECRET_URL}"))
    with pytest.raises(post.DiscordError) as caught:
        _send()
    text = str(caught.value)
    assert "ConnectError" in text
    assert "secret-token" not in text and "discord.com" not in text


def test_deliver_names_the_exception_type_of_a_failing_transport(monkeypatch):
    monkeypatch.setenv("NEMO_DISCORD_OPS_WEBHOOK_URL", SECRET_URL)

    def transport(url, body):
        raise TimeoutError(f"slow {url}")

    payload = post.ops_message("research-seed", "1 row recorded", ok=True,
                               when=1_700_000_000)
    with pytest.raises(post.DiscordError) as caught:
        post.deliver("ops", [payload], transport=transport)
    text = str(caught.value)
    assert "TimeoutError" in text
    assert "secret-token" not in text


def test_deliver_checks_every_page_before_it_posts_the_first(monkeypatch):
    monkeypatch.setenv("NEMO_DISCORD_OPS_WEBHOOK_URL", SECRET_URL)
    sent = []
    fine = post.ops_message("research-seed", "1 row recorded", ok=True,
                            when=1_700_000_000)
    too_long = {
        "flags": post.IS_COMPONENTS_V2,
        "components": [{"type": post.CONTAINER, "components": [
            {"type": post.TEXT, "content": "x" * (post.TEXT_LIMIT + 1)}]}],
        "allowed_mentions": {"parse": []},
    }
    with pytest.raises(post.DiscordError):
        post.deliver("ops", [fine, too_long],
                     transport=lambda url, body: sent.append(body))
    assert sent == []
