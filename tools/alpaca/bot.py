"""The nemo bot in Discord: Nemo's cards on demand.

    /portfolio      the paper account, as the daily card shows it
    /status         the jobs, the book, the risk switch and the gate
    /orders         each book's latest decision and what its orders did
    /ticker SYMBOL  everything Nemo holds on one name
    /ask QUESTION   Claude's answer, from Nemo's own tools

The client connects out to Discord's gateway, so nothing on the host listens
for Discord. The commands are registered to one server, `NEMO_DISCORD_GUILD_ID`,
and Discord shows them only to that server's administrators. Each answer is
seen only by whoever asked.

Reading the broker or the store can take longer than the three seconds
Discord allows for an answer, so each command defers at once and then
replaces the loading message with its card, through the interaction's own
webhook.

    python -m tools.alpaca.bot

Paper only, and read only: it reads the store and calls `portfolio.read`,
which never places or cancels an order.
"""
from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Union

import discord
from discord import app_commands

from research import discord_post, lookup, pit_store, status
from tools.alpaca import ask, portfolio

TOKEN_SETTING = "NEMO_DISCORD_BOT_TOKEN"
GUILD_SETTING = "NEMO_DISCORD_GUILD_ID"


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def reply(result: Dict[str, Any]) -> dict:
    """The portfolio card, or the error that stopped it."""
    if result.get("status") == "ok":
        return discord_post.portfolio_message(result, source="nemo /portfolio")
    return discord_post.ops_message(
        "/portfolio", "the paper account could not be read", ok=False,
        error=str(result.get("error") or ""))


async def portfolio_card() -> dict:
    return reply(await portfolio.read())


async def status_card() -> dict:
    return discord_post.status_card(await asyncio.to_thread(status.collect))


async def orders_card() -> dict:
    return discord_post.orders_card(
        await asyncio.to_thread(lookup.orders, _today()))


async def ticker_card(raw: str) -> dict:
    """The store's record of one name, with its position beside it.

    An account that cannot be read is said on the card; the store's record
    still answers, because it does not depend on the broker.
    """
    name = lookup.symbol(raw)
    found = await asyncio.to_thread(lookup.ticker, name, _today())
    try:
        account = await portfolio.read()
    except Exception as exc:  # noqa: BLE001 - shown on the card
        found["position_error"] = f"{type(exc).__name__}: {exc}"
    else:
        found["position"] = next((p for p in account["positions"]
                                  if p["symbol"] == name), None)
    return discord_post.ticker_card(found)


async def ask_cards(question: str) -> List[dict]:
    """Claude's answer, as one card or several when it is long."""
    try:
        result = await asyncio.wait_for(ask.ask(question), ask.TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        raise RuntimeError(
            f"no answer within {ask.TIMEOUT_SECONDS:g} seconds") from None
    return discord_post.answer_messages(result)


Cards = Union[dict, List[dict]]


async def answer(interaction, command: str,
                 build: Callable[[], Awaitable[Cards]]) -> None:
    """Build the card, then replace the deferred message with it.

    Whatever stops the card is the answer instead, so a command never leaves
    the loading message standing. An edit takes only IS_COMPONENTS_V2 of the
    flags a post sets; the daily portfolio card also sets the silent flag.
    Cards after the first are private follow-ups to the same interaction.
    """
    try:
        built = await build()
    except Exception as exc:  # noqa: BLE001 - shown in the reply and logged
        error = f"{type(exc).__name__}: {exc}"
        print(f"/{command}: {error}", file=sys.stderr, flush=True)
        built = discord_post.ops_message(
            f"/{command}", "Nemo could not answer", ok=False, error=error)
    first, *rest = built if isinstance(built, list) else [built]
    first["flags"] = discord_post.IS_COMPONENTS_V2
    app, token = str(interaction.application_id), interaction.token
    try:
        await asyncio.to_thread(discord_post.edit_interaction_reply,
                                app, token, first)
        for payload in rest:
            await asyncio.to_thread(discord_post.follow_up_interaction,
                                    app, token, payload)
    except discord_post.DiscordError as exc:
        print(f"/{command}: {discord_post.redact(str(exc))}", file=sys.stderr,
              flush=True)


class NemoClient(discord.Client):
    """A client with Nemo's commands, synced to one server when it connects."""

    def __init__(self, guild_id: int):
        # Slash commands arrive whatever the intents; the bot reads nothing else.
        super().__init__(intents=discord.Intents.none())
        self.guild = discord.Object(id=guild_id)
        self.tree = app_commands.CommandTree(self)

    async def setup_hook(self) -> None:
        await self.tree.sync(guild=self.guild)


def build(guild_id: int) -> NemoClient:
    client = NemoClient(guild_id)

    def command(name: str, description: str):
        return client.tree.command(name=name, description=description,
                                   guild=client.guild)

    admins = app_commands.default_permissions(administrator=True)

    async def respond(interaction, name, card):
        await interaction.response.defer(ephemeral=True, thinking=True)
        await answer(interaction, name, card)

    @command("portfolio", "The paper account: value, today's change, "
                          "positions, and each book's closed trades")
    @admins
    async def portfolio_command(interaction: discord.Interaction) -> None:
        await respond(interaction, "portfolio", portfolio_card)

    @command("status", "The jobs, the book, the risk switch and the gate")
    @admins
    async def status_command(interaction: discord.Interaction) -> None:
        await respond(interaction, "status", status_card)

    @command("orders", "Each book's latest orders and what they filled")
    @admins
    async def orders_command(interaction: discord.Interaction) -> None:
        await respond(interaction, "orders", orders_card)

    @command("ticker", "Everything Nemo holds on one name")
    @app_commands.describe(symbol="The ticker, for example TMO")
    @admins
    async def ticker_command(interaction: discord.Interaction,
                             symbol: str) -> None:
        await respond(interaction, "ticker", lambda: ticker_card(symbol))

    @command("ask", "A question answered by Claude from Nemo's own tools")
    @app_commands.describe(question="For example: what did NVDA report "
                                    "last quarter?")
    @admins
    async def ask_command(
            interaction: discord.Interaction,
            question: app_commands.Range[str, 1, ask.QUESTION_LIMIT]) -> None:
        await respond(interaction, "ask", lambda: ask_cards(question))

    return client


def main() -> int:
    token = os.environ.get(TOKEN_SETTING, "").strip()
    guild = os.environ.get(GUILD_SETTING, "").strip()
    if not token:
        print(f"nemo-bot: {TOKEN_SETTING} is not set", file=sys.stderr)
        return 1
    if not guild.isdigit():
        print(f"nemo-bot: {GUILD_SETTING} must be the server's numeric id",
              file=sys.stderr)
        return 1
    pit_store.init_schema()
    build(int(guild)).run(token)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via main()
    raise SystemExit(main())
