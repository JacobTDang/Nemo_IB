"""/ask: a question answered by Claude with Nemo's own tools.

The bot opens one MCP session to the gateway, which serves every data
server's tools behind one URL, and hands them to Claude deferred behind tool
search, so a question loads the few tools it needs rather than all of them.
The SDK's tool runner calls the tools Claude asks for through that session
until Claude answers or the round limit is reached.

The gateway is reached with the bot's own agent token, so the servers rate
it like any other agent and removing the token withdraws it alone.

Read only: Nemo's servers read data and place no order.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, List

MODEL = "claude-opus-5-5"
EFFORT = "medium"
MAX_ITERATIONS = 10
MAX_TOKENS = 16000
TIMEOUT_SECONDS = 600
QUESTION_LIMIT = 500

KEY_SETTING = "ANTHROPIC_API_KEY"
MCP_TOKEN_SETTING = "NEMO_MCP_BOT_TOKEN"
MCP_URL_SETTING = "NEMO_MCP_URL"
GATEWAY_URL = "http://nemo:8080/mcp/"

SEARCH_TOOL = {"type": "tool_search_tool_bm25_20251119",
               "name": "tool_search_tool_bm25"}

SYSTEM = """\
You answer questions from the owner of Nemo, in a private Discord reply.
Nemo reads financial data from primary sources: SEC filings, market data,
Finnhub, FRED, and congressional disclosures. Its tools are deferred: search
for the tool you need, then call it.

Base every number on a tool result and say which tool and date it came from.
If a tool fails or returns nothing, say so plainly; never fill the gap with a
guess.

Write under 1,500 characters of Discord markdown: short paragraphs or
bullets, no tables, no heading larger than ###."""


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def tools_for(listed, session) -> List[Any]:
    """Tool search first, then every Nemo tool, deferred until searched for."""
    from anthropic.lib.tools.mcp import async_mcp_tool

    return [SEARCH_TOOL] + [async_mcp_tool(tool, session, defer_loading=True)
                            for tool in listed]


async def answer(question: str, session, client) -> Dict[str, Any]:
    """Run the question through the tool runner and read the answer back."""
    listed = (await session.list_tools()).tools
    runner = client.beta.messages.tool_runner(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        max_iterations=MAX_ITERATIONS,
        system=f"{SYSTEM}\n\nToday's date is {_today()} (UTC).",
        tools=tools_for(listed, session),
        messages=[{"role": "user", "content": question}],
        output_config={"effort": EFFORT},
        cache_control={"type": "ephemeral"},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    final = None
    rounds = input_tokens = output_tokens = 0
    used: List[str] = []
    async for message in runner:
        final = message
        rounds += 1
        usage = message.usage
        input_tokens += (usage.input_tokens
                         + (usage.cache_read_input_tokens or 0)
                         + (usage.cache_creation_input_tokens or 0))
        output_tokens += usage.output_tokens
        for block in message.content:
            if block.type == "tool_use" and block.name not in used:
                used.append(block.name)
    if final is None:
        raise RuntimeError("Claude returned no message")
    text = "".join(block.text for block in final.content
                   if block.type == "text").strip()
    return {"question": question, "answer": text, "tools": used,
            "rounds": rounds, "input_tokens": input_tokens,
            "output_tokens": output_tokens, "stop_reason": final.stop_reason,
            "model": final.model}


async def ask(question: str) -> Dict[str, Any]:
    """Answer one question with Nemo's tools through the gateway."""
    question = (question or "").strip()
    if not question:
        raise ValueError("the question is empty")
    if not os.environ.get(KEY_SETTING, "").strip():
        raise RuntimeError(f"{KEY_SETTING} is not set")
    token = os.environ.get(MCP_TOKEN_SETTING, "").strip()
    if not token:
        raise RuntimeError(f"{MCP_TOKEN_SETTING} is not set")

    import httpx
    from anthropic import AsyncAnthropic
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    url = os.environ.get(MCP_URL_SETTING, "").strip() or GATEWAY_URL
    timeout = httpx.Timeout(30.0, read=float(TIMEOUT_SECONDS))
    async with httpx.AsyncClient(
            headers={"Authorization": f"Bearer {token}"},
            timeout=timeout) as http:
        async with streamable_http_client(url, http_client=http) as (
                read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                async with AsyncAnthropic() as client:
                    return await answer(question, session, client)
