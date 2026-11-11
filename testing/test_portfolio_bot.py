"""/portfolio: the daily portfolio card on demand, seen only by whoever asks.

The command answers with a deferred, ephemeral reply at once and then
replaces it with the card through the interaction's webhook. These tests run
the command's callback with a stand-in interaction; nothing connects to
Discord.
"""
from __future__ import annotations

import asyncio

import httpx
import pytest

import research.discord_post as post

discord = pytest.importorskip("discord")

from tools.alpaca import portfolio, portfolio_bot  # noqa: E402

TOKEN = "interaction-token-that-is-a-secret"
APP = "123456789012345678"
GUILD = 987654321098765432


def _texts(payload):
    found = []

    def walk(nodes):
        for node in nodes:
            if node.get("type") == post.TEXT:
                found.append(node["content"])
            walk(node.get("components") or [])

    walk(payload["components"])
    return "\n".join(found)


def _ok():
    return {"status": "ok", "equity": 1000.0, "last_equity": 990.0,
            "cash": 400.0,
            "positions": [{"symbol": "NVDA", "side": "long",
                           "market_value": 600.0, "unrealized_pl": 6.0,
                           "unrealized_plpc": 0.01, "books": ["Earnings"]}],
            "books": {"Earnings": {"open": 1, "closed": 0,
                                   "mean_return": None}}}


# ------------------------------------------------------------- the reply

def test_the_reply_is_the_daily_card_with_only_the_flag_an_edit_takes():
    payload = portfolio_bot.reply(_ok())

    assert payload["flags"] == post.IS_COMPONENTS_V2
    text = _texts(payload)
    assert "## 📋 Paper portfolio · $1,000.00" in text
    assert "nemo /portfolio" in text


def test_a_failed_read_replies_with_its_error():
    payload = portfolio_bot.reply(
        {"status": "failed", "error": "AsyncBrokerError: 401 unauthorized"})

    assert payload["flags"] == post.IS_COMPONENTS_V2
    text = _texts(payload)
    assert "❌" in text
    assert "401 unauthorized" in text


class Interaction:
    def __init__(self):
        self.application_id = int(APP)
        self.token = TOKEN
        self.deferred = None

        class Response:
            async def defer(inner, **kwargs):
                self.deferred = kwargs

        self.response = Response()


def _edits(monkeypatch):
    edits = []
    monkeypatch.setattr(
        post, "edit_interaction_reply",
        lambda app, token, payload: edits.append((app, token, payload)))
    return edits


def test_the_command_defers_privately_then_replaces_it_with_the_card(
        monkeypatch):
    async def read():
        return _ok()

    monkeypatch.setattr(portfolio, "read", read)
    edits = _edits(monkeypatch)
    command = portfolio_bot.build(GUILD).tree.get_command(
        "portfolio", guild=discord.Object(id=GUILD))
    interaction = Interaction()

    asyncio.run(command.callback(interaction))

    assert interaction.deferred == {"ephemeral": True, "thinking": True}
    [(app, token, payload)] = edits
    assert (app, token) == (APP, TOKEN)
    assert "**NVDA** · Earnings" in _texts(payload)


def test_a_broker_that_raises_still_gets_an_answer(monkeypatch):
    async def read():
        raise RuntimeError("Missing Alpaca paper credentials")

    monkeypatch.setattr(portfolio, "read", read)
    edits = _edits(monkeypatch)

    asyncio.run(portfolio_bot.answer(Interaction()))

    [(_, _, payload)] = edits
    assert "Missing Alpaca paper credentials" in _texts(payload)


def test_a_failed_edit_names_no_token(monkeypatch, capsys):
    async def read():
        return _ok()

    def refuse(app, token, payload):
        raise post.DiscordError("Discord edit returned 404: code 10015: Unknown Webhook")

    monkeypatch.setattr(portfolio, "read", read)
    monkeypatch.setattr(post, "edit_interaction_reply", refuse)

    asyncio.run(portfolio_bot.answer(Interaction()))

    err = capsys.readouterr().err
    assert "404" in err
    assert TOKEN not in err


# ------------------------------------------------------------- the command

def test_the_command_is_registered_to_one_server_for_its_administrators():
    client = portfolio_bot.build(GUILD)

    command = client.tree.get_command("portfolio", guild=discord.Object(id=GUILD))
    assert command is not None
    assert client.tree.get_command("portfolio") is None
    assert command.default_permissions.administrator is True


def test_a_missing_token_stops_the_bot_and_names_the_setting(
        monkeypatch, capsys):
    monkeypatch.delenv(portfolio_bot.TOKEN_SETTING, raising=False)
    monkeypatch.setenv(portfolio_bot.GUILD_SETTING, str(GUILD))

    assert portfolio_bot.main() == 1
    assert portfolio_bot.TOKEN_SETTING in capsys.readouterr().err


@pytest.mark.parametrize("value", ["", "jd-llc"])
def test_a_server_id_that_is_not_a_number_stops_the_bot(
        monkeypatch, capsys, value):
    monkeypatch.setenv(portfolio_bot.TOKEN_SETTING, "a-bot-token")
    monkeypatch.setenv(portfolio_bot.GUILD_SETTING, value)

    assert portfolio_bot.main() == 1
    assert portfolio_bot.GUILD_SETTING in capsys.readouterr().err


# ------------------------------------------------------------- the edit

def _patch_with(monkeypatch, handler):
    def patch(url, **kwargs):
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            return client.patch(url, **kwargs)

    monkeypatch.setattr(httpx, "patch", patch)


def test_the_edit_replaces_the_original_message_with_components(monkeypatch):
    seen = {}

    def handler(request):
        seen["method"] = request.method
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"id": "1"})

    _patch_with(monkeypatch, handler)
    post.edit_interaction_reply(APP, TOKEN, portfolio_bot.reply(_ok()))

    assert seen["method"] == "PATCH"
    assert seen["url"] == (f"https://discord.com/api/v10/webhooks/{APP}/{TOKEN}"
                           f"/messages/@original?with_components=true")


def test_an_edit_discord_refuses_raises_without_the_token(monkeypatch):
    def handler(request):
        return httpx.Response(400, json={"code": 50035,
                                         "message": "Invalid Form Body"})

    _patch_with(monkeypatch, handler)
    with pytest.raises(post.DiscordError) as caught:
        post.edit_interaction_reply(APP, TOKEN, portfolio_bot.reply(_ok()))

    assert "400" in str(caught.value)
    assert "Invalid Form Body" in str(caught.value)
    assert TOKEN not in str(caught.value)


def test_an_edit_answered_without_a_message_id_is_a_failure(monkeypatch):
    _patch_with(monkeypatch, lambda request: httpx.Response(200, json={}))

    with pytest.raises(post.DiscordError):
        post.edit_interaction_reply(APP, TOKEN, portfolio_bot.reply(_ok()))


def test_an_interaction_url_is_redacted_like_a_webhook():
    text = f"failed at https://discord.com/api/v10/webhooks/{APP}/{TOKEN}/messages/@original"

    assert TOKEN not in post.redact(text)
