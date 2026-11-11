"""The nemo bot: /portfolio, /status, /orders and /ticker, each seen only by
whoever asks.

Every command answers with a deferred, ephemeral reply at once and then
replaces it with its card through the interaction's webhook. These tests run
the commands' callbacks with a stand-in interaction; nothing connects to
Discord.
"""
from __future__ import annotations

import asyncio

import httpx
import pytest

import research.discord_post as post

discord = pytest.importorskip("discord")

from research import lookup, status  # noqa: E402
from tools.alpaca import ask, bot, portfolio  # noqa: E402

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

def test_the_reply_is_the_daily_card_named_for_the_command():
    payload = bot.reply(_ok())

    text = _texts(payload)
    assert "## 📋 Paper portfolio · $1,000.00" in text
    assert "nemo /portfolio" in text


def test_a_failed_read_replies_with_its_error():
    payload = bot.reply(
        {"status": "failed", "error": "AsyncBrokerError: 401 unauthorized"})

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


def _command(name):
    return bot.build(GUILD).tree.get_command(
        name, guild=discord.Object(id=GUILD))


def test_the_command_defers_privately_then_replaces_it_with_the_card(
        monkeypatch):
    async def read():
        return _ok()

    monkeypatch.setattr(portfolio, "read", read)
    edits = _edits(monkeypatch)
    interaction = Interaction()

    asyncio.run(_command("portfolio").callback(interaction))

    assert interaction.deferred == {"ephemeral": True, "thinking": True}
    [(app, token, payload)] = edits
    assert (app, token) == (APP, TOKEN)
    assert payload["flags"] == post.IS_COMPONENTS_V2
    assert "**NVDA** · Earnings" in _texts(payload)


def test_a_broker_that_raises_still_gets_an_answer(monkeypatch):
    async def read():
        raise RuntimeError("Missing Alpaca paper credentials")

    monkeypatch.setattr(portfolio, "read", read)
    edits = _edits(monkeypatch)

    asyncio.run(_command("portfolio").callback(Interaction()))

    [(_, _, payload)] = edits
    text = _texts(payload)
    assert "❌ /portfolio" in text
    assert "Missing Alpaca paper credentials" in text


def test_a_failed_edit_names_no_token(monkeypatch, capsys):
    async def read():
        return _ok()

    def refuse(app, token, payload):
        raise post.DiscordError("Discord edit returned 404: code 10015: Unknown Webhook")

    monkeypatch.setattr(portfolio, "read", read)
    monkeypatch.setattr(post, "edit_interaction_reply", refuse)

    asyncio.run(_command("portfolio").callback(Interaction()))

    err = capsys.readouterr().err
    assert "404" in err
    assert TOKEN not in err


def test_status_answers_with_the_status_report(monkeypatch):
    report = {"as_of": "2026-11-11", "jobs": [], "book": {}, "risk": {},
              "gate": None, "copy": {}, "attention": []}
    monkeypatch.setattr(status, "collect", lambda: report)
    edits = _edits(monkeypatch)
    interaction = Interaction()

    asyncio.run(_command("status").callback(interaction))

    assert interaction.deferred == {"ephemeral": True, "thinking": True}
    [(_, _, payload)] = edits
    assert payload["flags"] == post.IS_COMPONENTS_V2
    assert "All good" in _texts(payload)
    assert "nemo /status" in _texts(payload)


def test_orders_reads_the_store_as_of_today_in_utc(monkeypatch):
    seen = []

    def orders(as_of):
        seen.append(as_of)
        return {"as_of": as_of, "earnings": None, "copy": None}

    monkeypatch.setattr(lookup, "orders", orders)
    monkeypatch.setattr(bot, "_today", lambda: "2026-11-11")
    edits = _edits(monkeypatch)

    asyncio.run(_command("orders").callback(Interaction()))

    assert seen == ["2026-11-11"]
    [(_, _, payload)] = edits
    assert "No orders yet" in _texts(payload)


def _found(name, as_of):
    return {"symbol": name, "as_of": as_of, "signal": None, "orders": [],
            "releases": [{"fiscal_period": "2026Q3",
                          "announced_date": "2026-10-21", "timing": "bmo"}],
            "filings": [], "congress": []}


def test_ticker_reads_the_symbol_and_puts_the_position_beside_it(monkeypatch):
    seen = []

    def ticker(name, as_of):
        seen.append(name)
        return _found(name, as_of)

    async def read():
        return _ok()

    monkeypatch.setattr(lookup, "ticker", ticker)
    monkeypatch.setattr(portfolio, "read", read)
    edits = _edits(monkeypatch)

    asyncio.run(_command("ticker").callback(Interaction(), "nvda"))

    assert seen == ["NVDA"]
    [(_, _, payload)] = edits
    text = _texts(payload)
    assert "NVDA · you own" in text
    assert "**Last earnings report** Oct 21" in text


def test_ticker_still_answers_from_the_store_when_the_account_fails(
        monkeypatch):
    async def read():
        raise RuntimeError("Missing Alpaca paper credentials")

    monkeypatch.setattr(lookup, "ticker", _found)
    monkeypatch.setattr(portfolio, "read", read)
    edits = _edits(monkeypatch)

    asyncio.run(_command("ticker").callback(Interaction(), "NVDA"))

    [(_, _, payload)] = edits
    text = _texts(payload)
    assert "NVDA · couldn't check your account" in text
    assert "Missing Alpaca paper credentials" in text
    assert "**Last earnings report** Oct 21" in text


def test_ticker_refuses_what_is_not_a_ticker_without_reading_anything(
        monkeypatch):
    def ticker(name, as_of):
        raise AssertionError("the store was read")

    monkeypatch.setattr(lookup, "ticker", ticker)
    edits = _edits(monkeypatch)

    asyncio.run(_command("ticker").callback(Interaction(), "$NVDA"))

    [(_, _, payload)] = edits
    text = _texts(payload)
    assert "❌ /ticker" in text
    assert "'$NVDA' is not a ticker" in text


# ------------------------------------------------------------- the command

def test_every_command_is_registered_to_one_server_for_its_administrators():
    client = bot.build(GUILD)
    guild = discord.Object(id=GUILD)

    names = sorted(c.name for c in client.tree.get_commands(guild=guild))
    assert names == ["ask", "orders", "portfolio", "status", "ticker"]
    assert client.tree.get_commands() == []
    for name in names:
        command = client.tree.get_command(name, guild=guild)
        assert command.default_permissions.administrator is True


def test_ticker_takes_one_required_symbol():
    [param] = _command("ticker").parameters

    assert param.name == "symbol"
    assert param.required is True


def test_ask_takes_one_required_question_of_bounded_length():
    [param] = _command("ask").parameters

    assert param.name == "question"
    assert param.required is True
    assert param.max_value == ask.QUESTION_LIMIT


def test_a_missing_token_stops_the_bot_and_names_the_setting(
        monkeypatch, capsys):
    monkeypatch.delenv(bot.TOKEN_SETTING, raising=False)
    monkeypatch.setenv(bot.GUILD_SETTING, str(GUILD))

    assert bot.main() == 1
    assert bot.TOKEN_SETTING in capsys.readouterr().err


@pytest.mark.parametrize("value", ["", "jd-llc"])
def test_a_server_id_that_is_not_a_number_stops_the_bot(
        monkeypatch, capsys, value):
    monkeypatch.setenv(bot.TOKEN_SETTING, "a-bot-token")
    monkeypatch.setenv(bot.GUILD_SETTING, value)

    assert bot.main() == 1
    assert bot.GUILD_SETTING in capsys.readouterr().err


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
    post.edit_interaction_reply(APP, TOKEN, bot.reply(_ok()))

    assert seen["method"] == "PATCH"
    assert seen["url"] == (f"https://discord.com/api/v10/webhooks/{APP}/{TOKEN}"
                           f"/messages/@original?with_components=true")


def test_an_edit_discord_refuses_raises_without_the_token(monkeypatch):
    def handler(request):
        return httpx.Response(400, json={"code": 50035,
                                         "message": "Invalid Form Body"})

    _patch_with(monkeypatch, handler)
    with pytest.raises(post.DiscordError) as caught:
        post.edit_interaction_reply(APP, TOKEN, bot.reply(_ok()))

    assert "400" in str(caught.value)
    assert "Invalid Form Body" in str(caught.value)
    assert TOKEN not in str(caught.value)


def test_an_edit_answered_without_a_message_id_is_a_failure(monkeypatch):
    _patch_with(monkeypatch, lambda request: httpx.Response(200, json={}))

    with pytest.raises(post.DiscordError):
        post.edit_interaction_reply(APP, TOKEN, bot.reply(_ok()))


def test_an_interaction_url_is_redacted_like_a_webhook():
    text = f"failed at https://discord.com/api/v10/webhooks/{APP}/{TOKEN}/messages/@original"

    assert TOKEN not in post.redact(text)


def test_the_bot_reads_the_copy_switch_the_copy_job_reads():
    """/status reports the copy job as off unless the bot sees the same
    NEMO_COPY_TRADING the job does."""
    import yaml

    from pathlib import Path

    compose = yaml.safe_load((Path(__file__).resolve().parents[1] / "deploy"
                              / "docker-compose.yml").read_text())
    services = compose["services"]

    assert (services["nemo-bot"]["environment"]["NEMO_COPY_TRADING"]
            == services["research-copy"]["environment"]["NEMO_COPY_TRADING"])


# ------------------------------------------------------------- /ask

def _asked(answer):
    async def fake(question):
        return {"question": question, "answer": answer, "tools": [],
                "rounds": 1, "input_tokens": 10, "output_tokens": 5,
                "stop_reason": "end_turn", "model": "claude-opus-5-5"}
    return fake


def _follow_ups(monkeypatch):
    sent = []
    monkeypatch.setattr(
        post, "follow_up_interaction",
        lambda app, token, payload: sent.append((app, token, payload)))
    return sent


def test_ask_answers_in_one_card_when_the_answer_fits(monkeypatch):
    monkeypatch.setattr(ask, "ask", _asked("NVDA closed at $181.20."))
    edits = _edits(monkeypatch)
    follows = _follow_ups(monkeypatch)
    interaction = Interaction()

    asyncio.run(_command("ask").callback(interaction, "What did NVDA close at?"))

    assert interaction.deferred == {"ephemeral": True, "thinking": True}
    [(_, _, payload)] = edits
    assert "NVDA closed at $181.20." in _texts(payload)
    assert follows == []


def test_a_long_answer_edits_the_first_card_and_follows_up_with_the_rest(
        monkeypatch):
    long_answer = "\n\n".join(f"Paragraph {i}: " + "word " * 120
                               for i in range(12))
    monkeypatch.setattr(ask, "ask", _asked(long_answer))
    edits = _edits(monkeypatch)
    follows = _follow_ups(monkeypatch)

    asyncio.run(_command("ask").callback(Interaction(), "Tell me everything"))

    [(_, _, first)] = edits
    assert "(1/" in _texts(first)
    assert follows
    assert all((app, token) == (APP, TOKEN) for app, token, _ in follows)
    assert "Paragraph 11:" in _texts(follows[-1][2])


def test_an_answer_that_takes_too_long_says_so(monkeypatch):
    async def slow(question):
        await asyncio.sleep(5)

    monkeypatch.setattr(ask, "ask", slow)
    monkeypatch.setattr(ask, "TIMEOUT_SECONDS", 0.01)
    edits = _edits(monkeypatch)

    asyncio.run(_command("ask").callback(Interaction(), "What did NVDA close at?"))

    [(_, _, payload)] = edits
    text = _texts(payload)
    assert "❌ /ask" in text
    assert "no answer within" in text


def test_a_follow_up_is_private_and_names_no_token(monkeypatch):
    seen = {}

    def handler(request):
        import json
        seen["method"] = request.method
        seen["url"] = str(request.url)
        seen["flags"] = json.loads(request.content)["flags"]
        return httpx.Response(400, json={"code": 50035,
                                         "message": "Invalid Form Body"})

    def fake_post(url, **kwargs):
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            return client.post(url, **kwargs)

    monkeypatch.setattr(httpx, "post", fake_post)
    [payload] = post.answer_messages({"question": "q", "answer": "a",
                                      "stop_reason": "end_turn"})

    with pytest.raises(post.DiscordError) as caught:
        post.follow_up_interaction(APP, TOKEN, payload)

    assert seen["method"] == "POST"
    assert seen["url"] == (f"https://discord.com/api/v10/webhooks/{APP}/{TOKEN}"
                           f"?with_components=true")
    assert seen["flags"] == post.EPHEMERAL | post.IS_COMPONENTS_V2
    assert "Invalid Form Body" in str(caught.value)
    assert TOKEN not in str(caught.value)


def test_only_the_bot_is_handed_the_anthropic_key_and_never_the_operator_token():
    """/ask reaches the gateway as its own agent, so its access can be
    withdrawn alone, and no other service can spend the Anthropic key."""
    import yaml

    from pathlib import Path

    compose = yaml.safe_load((Path(__file__).resolve().parents[1] / "deploy"
                              / "docker-compose.yml").read_text())
    services = compose["services"]
    holders = sorted(name for name, service in services.items()
                     if "ANTHROPIC_API_KEY" in (service.get("environment") or {}))
    env = services["nemo-bot"]["environment"]

    assert holders == ["nemo-bot"]
    assert env["NEMO_MCP_BOT_TOKEN"] == "${NEMO_MCP_BOT_TOKEN:-}"
    assert not any(value and "NEMO_MCP_TOKEN" in str(value).replace(
        "NEMO_MCP_BOT_TOKEN", "") for value in env.values())
