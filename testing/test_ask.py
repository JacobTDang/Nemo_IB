"""/ask: a question answered by Claude with Nemo's own tools.

The model and the MCP session are stand-ins here: nothing reaches Anthropic
or a Nemo server. What is pinned is what the bot sends, how it reads the
answer back, and the card a reader sees.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

import research.discord_post as post

pytest.importorskip("anthropic")
mcp_types = pytest.importorskip("mcp.types")

from tools.alpaca import ask  # noqa: E402

WHEN = 1_794_000_000


def _texts(payload):
    found = []

    def walk(nodes):
        for node in nodes:
            if node.get("type") == post.TEXT:
                found.append(node["content"])
            walk(node.get("components") or [])

    walk(payload["components"])
    return "\n".join(found)


def _tool(name):
    return mcp_types.Tool(name=name, description=f"{name} tool",
                          inputSchema={"type": "object", "properties": {}})


class Session:
    def __init__(self, names=("get_market_data", "get_sic_code")):
        self.names = names

    async def list_tools(self):
        return SimpleNamespace(tools=[_tool(n) for n in self.names])


def _block(kind, **fields):
    return SimpleNamespace(type=kind, **fields)


def _message(content, stop="end_turn", input_tokens=1000, output_tokens=50,
             cached=0):
    return SimpleNamespace(
        content=content, stop_reason=stop, model="claude-opus-5-5",
        usage=SimpleNamespace(input_tokens=input_tokens,
                              output_tokens=output_tokens,
                              cache_read_input_tokens=cached,
                              cache_creation_input_tokens=0))


class Runner:
    def __init__(self, messages):
        self.messages = messages

    def __aiter__(self):
        async def gen():
            for message in self.messages:
                yield message
        return gen()


class Client:
    def __init__(self, messages):
        self.sent = None
        outer = self

        class Messages:
            def tool_runner(inner, **kwargs):
                outer.sent = kwargs
                return Runner(messages)

        self.beta = SimpleNamespace(messages=Messages())


def _answered(text="NVDA closed at **$181.20** (get_market_data, 2026-11-10)."):
    return [
        _message([_block("server_tool_use", name="tool_search_tool_bm25"),
                  _block("tool_use", name="get_market_data", input={})],
                 stop="tool_use", input_tokens=3000, output_tokens=80),
        _message([_block("text", text=text)], input_tokens=500,
                 output_tokens=40, cached=2500),
    ]


# ------------------------------------------------------------- the request

def test_every_nemo_tool_is_deferred_behind_tool_search():
    client = Client(_answered())

    asyncio.run(ask.answer("What did NVDA close at?", Session(), client))

    tools = list(client.sent["tools"])
    assert tools[0] == ask.SEARCH_TOOL
    assert [t.to_dict()["name"] for t in tools[1:]] == ["get_market_data",
                                                         "get_sic_code"]
    assert all(t.to_dict().get("defer_loading") is True for t in tools[1:])


def test_the_request_names_the_model_effort_and_its_limits():
    client = Client(_answered())

    asyncio.run(ask.answer("What did NVDA close at?", Session(), client))

    sent = client.sent
    assert sent["model"] == "claude-opus-5-5"
    assert sent["output_config"] == {"effort": ask.EFFORT}
    assert sent["max_iterations"] == ask.MAX_ITERATIONS
    assert sent["cache_control"] == {"type": "ephemeral"}
    assert sent["fallbacks"] == "default"
    assert "server-side-fallback-2026-07-01" in sent["betas"]
    assert sent["messages"] == [{"role": "user",
                                 "content": "What did NVDA close at?"}]


def test_the_system_prompt_carries_todays_date_in_utc(monkeypatch):
    monkeypatch.setattr(ask, "_today", lambda: "2026-11-11")
    client = Client(_answered())

    asyncio.run(ask.answer("What did NVDA close at?", Session(), client))

    assert "2026-11-11" in client.sent["system"]


# ------------------------------------------------------------- the answer

def test_the_answer_is_the_final_text_with_the_tools_it_used_and_its_tokens():
    result = asyncio.run(ask.answer("What did NVDA close at?", Session(),
                                    Client(_answered())))

    assert result["answer"].startswith("NVDA closed at")
    assert result["tools"] == ["get_market_data"]
    assert result["rounds"] == 2
    assert result["input_tokens"] == 3000 + 500 + 2500
    assert result["output_tokens"] == 120
    assert result["stop_reason"] == "end_turn"


def test_an_empty_question_is_refused_before_anything_is_called():
    with pytest.raises(ValueError, match="empty"):
        asyncio.run(ask.ask("   "))


@pytest.mark.parametrize("missing", [ask.KEY_SETTING, ask.MCP_TOKEN_SETTING])
def test_a_missing_setting_is_named(monkeypatch, missing):
    monkeypatch.setenv(ask.KEY_SETTING, "sk-test")
    monkeypatch.setenv(ask.MCP_TOKEN_SETTING, "mcp-test")
    monkeypatch.delenv(missing)

    with pytest.raises(RuntimeError, match=missing):
        asyncio.run(ask.ask("What did NVDA close at?"))


# ------------------------------------------------------------- the card

def _result(**extra):
    result = {"question": "What did NVDA close at?",
              "answer": "NVDA closed at **$181.20**.", "tools": ["get_market_data"],
              "rounds": 2, "input_tokens": 6000, "output_tokens": 120,
              "stop_reason": "end_turn", "model": "claude-opus-5-5"}
    result.update(extra)
    return result


def test_the_answer_card_shows_the_question_the_answer_and_what_it_used():
    [payload] = post.answer_messages(_result(), when=WHEN)

    text = _texts(payload)
    assert "## 📋 Nemo's answer" in text
    assert "-# Asked: What did NVDA close at?" in text
    assert "NVDA closed at **$181.20**." in text
    assert ("nemo /ask · claude-opus-5-5 · get_market_data · 6,000 tokens in · "
            "120 out" in text)
    assert payload["components"][0]["accent_color"] == post.BLURPLE


def test_the_question_is_escaped_but_the_answer_keeps_its_markdown():
    [payload] = post.answer_messages(_result(question="is *this* `ok`?"),
                                     when=WHEN)

    text = _texts(payload)
    assert "is \\*this\\* \\`ok\\`?" in text
    assert "**$181.20**" in text


def test_a_long_answer_is_split_into_numbered_cards_without_losing_text():
    paragraphs = [f"Paragraph {i}: " + "word " * 120 for i in range(12)]
    pages = post.answer_messages(_result(answer="\n\n".join(paragraphs)),
                                 when=WHEN)

    assert len(pages) > 1
    texts = [_texts(page) for page in pages]
    assert f"Nemo's answer (1/{len(pages)})" in texts[0]
    assert f"Nemo's answer ({len(pages)}/{len(pages)})" in texts[-1]
    for i in range(12):
        assert sum(f"Paragraph {i}:" in t for t in texts) == 1
    assert "-# Asked:" in texts[0]


def test_a_declined_question_says_so_in_yellow():
    [payload] = post.answer_messages(_result(answer="", stop_reason="refusal"),
                                     when=WHEN)

    assert "⚠️ Claude declined to answer this question." in _texts(payload)
    assert payload["components"][0]["accent_color"] == post.YELLOW


def test_an_answer_cut_short_says_why_and_keeps_what_it_had():
    [payload] = post.answer_messages(
        _result(answer="Partial: NVDA closed at $181.20.", stop_reason="tool_use",
                rounds=ask.MAX_ITERATIONS), when=WHEN)

    text = _texts(payload)
    assert "⚠️ Stopped after **10** rounds before a final answer" in text
    assert "Partial: NVDA closed at $181.20." in text
    assert payload["components"][0]["accent_color"] == post.YELLOW
