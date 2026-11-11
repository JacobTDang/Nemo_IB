"""/ask: a question answered by Claude Code, signed in as the bot's owner.

The bot runs the unmodified `claude` program in print mode with the owner's
own subscription token. Here the program is a stand-in that prints what the
real one prints: nothing reaches Anthropic or a Nemo server. What is pinned
is what the bot runs, what it hands the program, and how it reads the answer.
"""
from __future__ import annotations

import asyncio
import json

import pytest

import research.discord_post as post
from tools.alpaca import ask

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


def _init(tools=("ToolSearch", "mcp__nemo__get_market_data"),
          status="connected", model="claude-opus-5-5"):
    return {"type": "system", "subtype": "init", "model": model,
            "tools": list(tools),
            "mcp_servers": [{"name": "nemo", "status": status}]}


def _assistant(*blocks):
    return {"type": "assistant", "message": {"content": list(blocks)}}


def _result(text="NVDA closed at **$181.20** (get_market_data, Nov 10).",
            subtype="success", turns=3, stop="end_turn", is_error=False):
    return {"type": "result", "subtype": subtype, "is_error": is_error,
            "result": text if subtype == "success" else None,
            "num_turns": turns, "stop_reason": stop,
            "usage": {"input_tokens": 500, "output_tokens": 120,
                      "cache_read_input_tokens": 4000,
                      "cache_creation_input_tokens": 1500}}


def _answered():
    return [_init(),
            _assistant({"type": "tool_use", "name": "ToolSearch", "input": {}}),
            _assistant({"type": "tool_use", "name": "mcp__nemo__get_market_data",
                        "input": {"ticker": "NVDA"}}),
            _assistant({"type": "text", "text": "NVDA closed at $181.20."}),
            _result()]


class Program:
    """Stands in for `claude`: records how it was run, prints its lines."""

    def __init__(self, lines, code=0, stderr=b"", hang=False):
        self.lines, self.code, self.stderr, self.hang = lines, code, stderr, hang
        self.argv = self.env = self.cwd = self.stdin = None
        self.killed = False
        self.config = None

    async def exec(self, *argv, stdin=None, stdout=None, stderr=None,
                   cwd=None, env=None):
        self.argv, self.env, self.cwd = list(argv), env, cwd
        config = self.argv[self.argv.index("--mcp-config") + 1]
        with open(config) as handle:
            self.config = json.load(handle)
        program = self

        class Process:
            returncode = None

            async def communicate(inner, data=None):
                program.stdin = data
                if program.hang:
                    await asyncio.sleep(5)
                inner.returncode = program.code
                out = "\n".join(json.dumps(line) for line in program.lines)
                return out.encode(), program.stderr

            def kill(inner):
                program.killed = True

            async def wait(inner):
                return -9

        return Process()


@pytest.fixture
def settings(monkeypatch):
    monkeypatch.setenv(ask.OAUTH_SETTING, "sk-ant-oat-owner")
    monkeypatch.setenv(ask.MCP_TOKEN_SETTING, "mcp-bot-token")
    monkeypatch.setenv("ALPACA_PAPER_KEY", "paper-key")
    monkeypatch.setenv("NEMO_DISCORD_BOT_TOKEN", "discord-token")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api-would-bill-the-api")


def _run(monkeypatch, program, question="What did NVDA close at?"):
    monkeypatch.setattr(asyncio, "create_subprocess_exec", program.exec)
    return asyncio.run(ask.ask(question))


# ------------------------------------------------------------- what it runs

def test_the_question_goes_in_on_stdin_never_on_the_command_line(
        monkeypatch, settings):
    program = Program(_answered())

    _run(monkeypatch, program, "-5% movers today?")

    assert program.stdin == b"-5% movers today?"
    assert "-5% movers today?" not in program.argv


def test_claude_code_gets_only_tool_search_and_nemo_tools(monkeypatch, settings):
    program = Program(_answered())

    _run(monkeypatch, program)

    argv = program.argv
    assert argv[:2] == [ask.CLAUDE, "-p"]
    assert argv[argv.index("--tools") + 1] == "ToolSearch"
    assert "--strict-mcp-config" in argv
    assert argv[argv.index("--allowedTools") + 1] == "mcp__nemo__*"
    assert argv[argv.index("--permission-mode") + 1] == "dontAsk"
    assert argv[argv.index("--max-turns") + 1] == str(ask.MAX_TURNS)
    assert "--no-session-persistence" in argv
    denied = argv[argv.index("--disallowedTools") + 1:]
    for tool in ("Bash", "Read", "Write", "Edit", "WebFetch", "WebSearch",
                 "Agent"):
        assert tool in denied


def test_the_nemo_server_is_reached_with_the_bots_own_token_by_reference(
        monkeypatch, settings, tmp_path):
    monkeypatch.setenv(ask.MCP_URL_SETTING, "http://gateway:8080/mcp/")
    program = Program(_answered())

    _run(monkeypatch, program)

    server = program.config["mcpServers"]["nemo"]
    assert server == {"type": "http", "url": "http://gateway:8080/mcp/",
                      "headers": {"Authorization": "Bearer ${NEMO_MCP_BOT_TOKEN}"}}


def test_the_program_sees_no_secret_but_its_own_two(monkeypatch, settings):
    program = Program(_answered())

    _run(monkeypatch, program)

    env = program.env
    assert env[ask.OAUTH_SETTING] == "sk-ant-oat-owner"
    assert env[ask.MCP_TOKEN_SETTING] == "mcp-bot-token"
    for secret in ("ANTHROPIC_API_KEY", "ALPACA_PAPER_KEY",
                   "NEMO_DISCORD_BOT_TOKEN"):
        assert secret not in env
    assert env["DISABLE_UPDATES"] == "1"
    assert env["HOME"] == program.cwd


def test_it_runs_in_an_empty_directory_that_is_removed_after(
        monkeypatch, settings):
    import os

    program = Program(_answered())

    _run(monkeypatch, program)

    assert not os.path.exists(program.cwd)


# ------------------------------------------------------------- what it reads

def test_the_answer_is_the_result_with_the_nemo_tools_it_called():
    result = ask.read_stream(
        "\n".join(json.dumps(line) for line in _answered()), "q")

    assert result["answer"].startswith("NVDA closed at")
    assert result["tools"] == ["get_market_data"]
    assert result["rounds"] == 3
    assert result["input_tokens"] == 500 + 4000 + 1500
    assert result["output_tokens"] == 120
    assert result["model"] == "claude-opus-5-5"
    assert result["stop_reason"] == "end_turn"


def test_a_built_in_tool_left_in_reach_stops_the_answer():
    lines = [_init(tools=("ToolSearch", "Bash", "mcp__nemo__get_sic_code")),
             _result()]

    with pytest.raises(RuntimeError, match="Bash"):
        ask.read_stream("\n".join(json.dumps(line) for line in lines), "q")


def test_nemo_tools_that_could_not_be_reached_are_said():
    lines = [_init(status="failed"), _result()]

    with pytest.raises(RuntimeError, match="could not reach Nemo's tools"):
        ask.read_stream("\n".join(json.dumps(line) for line in lines), "q")


def test_an_answer_stopped_at_the_turn_limit_is_kept_and_marked():
    lines = [_init(), _result(subtype="error_max_turns", turns=10)]

    result = ask.read_stream("\n".join(json.dumps(line) for line in lines), "q")

    assert result["stop_reason"] == "max_turns"
    assert result["rounds"] == 10


def test_a_failed_run_is_an_error_naming_what_went_wrong(monkeypatch, settings):
    lines = [_init(), {**_result(subtype="error_during_execution",
                                 is_error=True), "errors": ["rate limited"]}]

    with pytest.raises(RuntimeError, match="rate limited"):
        _run(monkeypatch, Program(lines, code=1))


def test_a_program_that_prints_nothing_is_an_error_with_its_stderr(
        monkeypatch, settings):
    with pytest.raises(RuntimeError, match="Invalid API key"):
        _run(monkeypatch, Program([], code=1,
                                  stderr=b"Invalid API key \xc2\xb7 Please run /login"))


def test_a_program_that_runs_too_long_is_killed(monkeypatch, settings):
    monkeypatch.setattr(ask, "TIMEOUT_SECONDS", 0.01)
    program = Program(_answered(), hang=True)

    with pytest.raises(RuntimeError, match="no answer within"):
        _run(monkeypatch, program)

    assert program.killed


def test_an_empty_question_is_refused_before_anything_runs(settings):
    with pytest.raises(ValueError, match="empty"):
        asyncio.run(ask.ask("   "))


@pytest.mark.parametrize("missing", ["CLAUDE_CODE_OAUTH_TOKEN",
                                     "NEMO_MCP_BOT_TOKEN"])
def test_a_missing_setting_is_named(monkeypatch, settings, missing):
    monkeypatch.delenv(missing)

    with pytest.raises(RuntimeError, match=missing):
        asyncio.run(ask.ask("What did NVDA close at?"))


# ------------------------------------------------------------- the card

def _card(**extra):
    result = {"question": "What did NVDA close at?",
              "answer": "NVDA closed at **$181.20**.", "tools": ["get_market_data"],
              "rounds": 3, "input_tokens": 6000, "output_tokens": 120,
              "stop_reason": "end_turn", "model": "claude-opus-5-5"}
    result.update(extra)
    return result


def test_the_answer_card_shows_the_question_the_answer_and_what_it_used():
    [payload] = post.answer_messages(_card(), when=WHEN)

    text = _texts(payload)
    assert "## 📋 Nemo's answer" in text
    assert "-# Asked: What did NVDA close at?" in text
    assert "NVDA closed at **$181.20**." in text
    assert ("nemo /ask · claude-opus-5-5 · get_market_data · 6,000 tokens in · "
            "120 out" in text)
    assert payload["components"][0]["accent_color"] == post.BLURPLE


def test_the_question_is_escaped_but_the_answer_keeps_its_markdown():
    [payload] = post.answer_messages(_card(question="is *this* `ok`?"),
                                     when=WHEN)

    text = _texts(payload)
    assert "is \\*this\\* \\`ok\\`?" in text
    assert "**$181.20**" in text


def test_a_long_answer_is_split_into_numbered_cards_without_losing_text():
    paragraphs = [f"Paragraph {i}: " + "word " * 120 for i in range(12)]
    pages = post.answer_messages(_card(answer="\n\n".join(paragraphs)),
                                 when=WHEN)

    assert len(pages) > 1
    texts = [_texts(page) for page in pages]
    assert f"Nemo's answer (1/{len(pages)})" in texts[0]
    assert f"Nemo's answer ({len(pages)}/{len(pages)})" in texts[-1]
    for i in range(12):
        assert sum(f"Paragraph {i}:" in t for t in texts) == 1
    assert "-# Asked:" in texts[0]


def test_a_declined_question_says_so_in_yellow():
    [payload] = post.answer_messages(_card(answer="", stop_reason="refusal"),
                                     when=WHEN)

    assert "⚠️ Claude declined to answer this question." in _texts(payload)
    assert payload["components"][0]["accent_color"] == post.YELLOW


def test_an_answer_cut_short_says_why_and_keeps_what_it_had():
    [payload] = post.answer_messages(
        _card(answer="Partial: NVDA closed at $181.20.", stop_reason="max_turns",
              rounds=ask.MAX_TURNS), when=WHEN)

    text = _texts(payload)
    assert "⚠️ Stopped after **10** turns before a final answer" in text
    assert "Partial: NVDA closed at $181.20." in text
    assert payload["components"][0]["accent_color"] == post.YELLOW
