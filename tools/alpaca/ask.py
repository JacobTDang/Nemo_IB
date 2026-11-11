"""/ask: a question answered by Claude Code, signed in as the bot's owner.

The bot runs the unmodified `claude` program in print mode, signed in with
the owner's own subscription through `CLAUDE_CODE_OAUTH_TOKEN` (made once
with `claude setup-token`). Anthropic allows that for the owner's own use and
not for answering anybody else, so the bot answers /ask for its owner only
(`tools.alpaca.bot`).

Claude Code gets one MCP server, the gateway, which serves every data
server's tools behind one URL, reached with the bot's own agent token so
the servers rate it like any other agent. Of its own tools it keeps only
tool search, which loads a Nemo tool when a question needs it: no shell, no
files, no web. The program runs in an empty directory with an environment
holding only its two tokens, so the broker keys and the Discord token in
the bot's environment never reach it, and no API key can quietly bill the
API instead of the subscription.

Read only: Nemo's servers read data and place no order.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, List

CLAUDE = "claude"
MAX_TURNS = 10
TIMEOUT_SECONDS = 600
QUESTION_LIMIT = 500

OAUTH_SETTING = "CLAUDE_CODE_OAUTH_TOKEN"
MCP_TOKEN_SETTING = "NEMO_MCP_BOT_TOKEN"
MCP_URL_SETTING = "NEMO_MCP_URL"
GATEWAY_URL = "http://nemo:8080/mcp/"
SERVER = "nemo"

# Tool search reads the deferred tool definitions and nothing else. Without
# it Claude Code loads all 96 Nemo tools into every turn.
TOOLS = "ToolSearch"
DENIED = ("Agent", "Bash", "Edit", "Glob", "Grep", "ListMcpResourcesTool",
          "Monitor", "NotebookEdit", "Read", "ReadMcpResourceTool", "Skill",
          "TaskCreate", "TaskGet", "TaskList", "TaskStop", "TaskUpdate",
          "TodoWrite", "WebFetch", "WebSearch", "Write")

SYSTEM = """\
You are answering a question from the owner of Nemo, in a private Discord
reply. Nemo reads financial data from primary sources: SEC filings, market
data, Finnhub, FRED, and congressional disclosures. Use the nemo tools; find
the one you need with tool search.

Base every number on a tool result and say which tool and date it came from.
If a tool fails or returns nothing, say so plainly; never fill the gap with
a guess. The owner is not an engineer: use plain words.

Write under 1,500 characters of Discord markdown: short paragraphs or
bullets, no tables, no heading larger than ###."""


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def command(config_path: str) -> List[str]:
    """`claude` in print mode with only tool search and the nemo tools."""
    return [CLAUDE, "-p",
            "--strict-mcp-config", "--mcp-config", config_path,
            "--tools", TOOLS,
            "--allowedTools", f"mcp__{SERVER}__*",
            "--permission-mode", "dontAsk",
            "--max-turns", str(MAX_TURNS),
            "--append-system-prompt",
            f"{SYSTEM}\n\nToday's date is {_today()} (UTC).",
            "--no-session-persistence",
            "--output-format", "stream-json", "--verbose",
            "--disallowedTools", *DENIED]


def environment(home: str, oauth: str, token: str) -> Dict[str, str]:
    """Only what the program needs. Nothing is inherited from the bot."""
    return {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": home,
        "CLAUDE_CONFIG_DIR": os.path.join(home, ".claude"),
        "TZ": "UTC",
        OAUTH_SETTING: oauth,
        MCP_TOKEN_SETTING: token,
        "DISABLE_UPDATES": "1",
        "DISABLE_AUTOUPDATER": "1",
        "DISABLE_TELEMETRY": "1",
        "DISABLE_ERROR_REPORTING": "1",
        "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
        "CLAUDE_CODE_SKIP_PROMPT_HISTORY": "1",
    }


def _check_init(event: dict) -> str:
    """Refuse a run that can reach anything but tool search and Nemo."""
    extra = [tool for tool in event.get("tools") or []
             if tool != TOOLS and not tool.startswith(f"mcp__{SERVER}__")]
    if extra:
        raise RuntimeError("Claude Code kept tools it must not have: "
                           + ", ".join(sorted(extra)))
    servers = {s.get("name"): s.get("status")
               for s in event.get("mcp_servers") or []}
    if servers.get(SERVER) != "connected":
        raise RuntimeError(f"Claude Code could not reach Nemo's tools "
                           f"(the {SERVER} server is "
                           f"{servers.get(SERVER) or 'missing'})")
    return str(event.get("model") or "")


def read_stream(stdout: str, question: str) -> Dict[str, Any]:
    """The answer, the Nemo tools called and the tokens used, from the
    program's stream-json output."""
    model = ""
    used: List[str] = []
    final = None
    prefix = f"mcp__{SERVER}__"
    for line in stdout.splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("type") == "system" and event.get("subtype") == "init":
            model = _check_init(event)
        elif event.get("type") == "assistant":
            for block in (event.get("message") or {}).get("content") or []:
                name = block.get("name") or ""
                if (block.get("type") == "tool_use" and name.startswith(prefix)
                        and name[len(prefix):] not in used):
                    used.append(name[len(prefix):])
        elif event.get("type") == "result":
            final = event
    if final is None:
        raise RuntimeError("Claude Code printed no result")
    subtype = final.get("subtype")
    if subtype == "error_max_turns":
        stop = "max_turns"
    elif final.get("is_error") or subtype != "success":
        errors = "; ".join(str(e) for e in final.get("errors") or [])
        raise RuntimeError(f"Claude Code stopped with {subtype}"
                           + (f": {errors}" if errors else ""))
    else:
        stop = final.get("stop_reason") or "end_turn"
    usage = final.get("usage") or {}
    return {"question": question, "answer": str(final.get("result") or ""),
            "tools": used, "rounds": int(final.get("num_turns") or 0),
            "input_tokens": int(usage.get("input_tokens") or 0)
            + int(usage.get("cache_read_input_tokens") or 0)
            + int(usage.get("cache_creation_input_tokens") or 0),
            "output_tokens": int(usage.get("output_tokens") or 0),
            "stop_reason": stop, "model": model}


async def ask(question: str) -> Dict[str, Any]:
    """Answer one question with Claude Code and Nemo's tools."""
    question = (question or "").strip()
    if not question:
        raise ValueError("the question is empty")
    oauth = os.environ.get(OAUTH_SETTING, "").strip()
    if not oauth:
        raise RuntimeError(f"{OAUTH_SETTING} is not set")
    token = os.environ.get(MCP_TOKEN_SETTING, "").strip()
    if not token:
        raise RuntimeError(f"{MCP_TOKEN_SETTING} is not set")

    home = tempfile.mkdtemp(prefix="nemo-ask-")
    try:
        config = os.path.join(home, "mcp.json")
        url = os.environ.get(MCP_URL_SETTING, "").strip() or GATEWAY_URL
        with open(config, "w") as handle:
            json.dump({"mcpServers": {SERVER: {
                "type": "http", "url": url,
                "headers": {"Authorization": f"Bearer ${{{MCP_TOKEN_SETTING}}}"},
            }}}, handle)
        process = await asyncio.create_subprocess_exec(
            *command(config), stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            cwd=home, env=environment(home, oauth, token))
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(question.encode()), TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            raise RuntimeError(
                f"no answer within {TIMEOUT_SECONDS:g} seconds") from None
        text = stdout.decode(errors="replace")
        if not text.strip():
            tail = stderr.decode(errors="replace").strip()[-300:]
            raise RuntimeError(f"Claude Code exited with {process.returncode}"
                               + (f": {tail}" if tail else ""))
        return read_stream(text, question)
    finally:
        shutil.rmtree(home, ignore_errors=True)
