"""/portfolio in Discord: the daily portfolio card on demand.

The client connects out to Discord's gateway, so nothing on the host listens
for Discord. The command is registered to one server, `NEMO_DISCORD_GUILD_ID`,
and Discord shows it only to that server's administrators. The answer is
seen only by whoever asked.

Reading the broker can take longer than the three seconds Discord allows for
an answer, so the command defers at once and then replaces the loading
message with the card, through the interaction's own webhook.

    python -m tools.alpaca.portfolio_bot

Paper only, and read only: it calls `portfolio.read`, which never places or
cancels an order.
"""
from __future__ import annotations

import asyncio
import os
import sys
from typing import Any, Dict

import discord
from discord import app_commands

from research import discord_post, pit_store
from tools.alpaca import portfolio

TOKEN_SETTING = "NEMO_DISCORD_BOT_TOKEN"
GUILD_SETTING = "NEMO_DISCORD_GUILD_ID"


def reply(result: Dict[str, Any]) -> dict:
    """The card for the deferred message, or the error that stopped it.

    An edit takes only IS_COMPONENTS_V2 of the flags the daily post sets.
    """
    if result.get("status") == "ok":
        payload = discord_post.portfolio_message(result, source="nemo /portfolio")
    else:
        payload = discord_post.ops_message(
            "/portfolio", "the paper account could not be read", ok=False,
            error=str(result.get("error") or ""))
    payload["flags"] = discord_post.IS_COMPONENTS_V2
    return payload


async def answer(interaction) -> None:
    """Read the account, then replace the deferred message with the card."""
    try:
        result = await portfolio.read()
    except Exception as exc:  # noqa: BLE001 - shown in the reply and logged
        result = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        print(f"/portfolio: {result['error']}", file=sys.stderr, flush=True)
    try:
        await asyncio.to_thread(
            discord_post.edit_interaction_reply,
            str(interaction.application_id), interaction.token, reply(result))
    except discord_post.DiscordError as exc:
        print(f"/portfolio: {discord_post.redact(str(exc))}", file=sys.stderr,
              flush=True)


class PortfolioClient(discord.Client):
    """A client with one command, synced to one server when it connects."""

    def __init__(self, guild_id: int):
        # Slash commands arrive whatever the intents; the bot reads nothing else.
        super().__init__(intents=discord.Intents.none())
        self.guild = discord.Object(id=guild_id)
        self.tree = app_commands.CommandTree(self)

    async def setup_hook(self) -> None:
        await self.tree.sync(guild=self.guild)


def build(guild_id: int) -> PortfolioClient:
    client = PortfolioClient(guild_id)

    @client.tree.command(
        name="portfolio", guild=client.guild,
        description="The paper account: value, today's change, positions, "
                    "and each book's closed trades")
    @app_commands.default_permissions(administrator=True)
    async def portfolio_command(interaction: discord.Interaction) -> None:
        await interaction.response.defer(ephemeral=True, thinking=True)
        await answer(interaction)

    return client


def main() -> int:
    token = os.environ.get(TOKEN_SETTING, "").strip()
    guild = os.environ.get(GUILD_SETTING, "").strip()
    if not token:
        print(f"portfolio-bot: {TOKEN_SETTING} is not set", file=sys.stderr)
        return 1
    if not guild.isdigit():
        print(f"portfolio-bot: {GUILD_SETTING} must be the server's numeric id",
              file=sys.stderr)
        return 1
    pit_store.init_schema()
    build(int(guild)).run(token)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via main()
    raise SystemExit(main())
