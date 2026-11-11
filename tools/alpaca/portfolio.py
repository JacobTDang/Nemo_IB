"""The paper portfolio, posted to #signals after each weekday's close.

The account and the positions are the paper broker's. Each position is
labelled with the books whose fills hold it open -- the earnings book's
`paper_fill` and the congressional copy book's `copy_fill` -- and each book's
closed trades are summed from its own table. The two books share one account,
so the account's value is theirs together; their returns are never mixed.

Read only. It never places or cancels an order, and it reads the paper
account only:

    python -m tools.alpaca.portfolio

The `/portfolio` command in Discord (`tools.alpaca.bot`) shows the
same card on demand.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from research import pit_store
from tools.alpaca.async_broker import AsyncBroker

BOOKS = (("Earnings", "paper_fill"), ("Copy", "copy_fill"))


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


async def snapshot(broker, as_of: Optional[str] = None) -> Dict[str, Any]:
    """The account, its positions largest first, and each book's trades."""
    as_of = as_of or _today()
    account = await broker.get_account()
    positions = await broker.get_all_positions()
    books = {name: pit_store.book_summary(as_of, table)
             for name, table in BOOKS}
    rows = []
    for position in sorted(positions, key=lambda p: -abs(p["market_value"])):
        held_by = [name for name, book in books.items()
                   if position["symbol"] in book["open_tickers"]]
        rows.append({**position, "books": held_by})
    return {
        "status": "ok", "as_of": as_of,
        "equity": account["equity"], "last_equity": account["last_equity"],
        "cash": account["cash"], "positions": rows,
        "books": {name: {key: book[key]
                         for key in ("open", "closed", "mean_return")}
                  for name, book in books.items()},
    }


async def read() -> Dict[str, Any]:
    """One snapshot of the paper account. Paper, always."""
    async with AsyncBroker(paper=True) as broker:
        return await snapshot(broker)


def main() -> int:
    from research import discord_post

    elapsed = discord_post.stopwatch()
    pit_store.init_schema()
    try:
        result = asyncio.run(read())
    except Exception as exc:  # noqa: BLE001 - reported, then non-zero
        result = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(result, indent=2))
        return discord_post.finish("research-portfolio", 1, result,
                                   seconds=elapsed())
    print(json.dumps(result, indent=2, default=str))
    return discord_post.finish("research-portfolio", 0, result,
                               seconds=elapsed())


if __name__ == "__main__":  # pragma: no cover - exercised via main()
    raise SystemExit(main())
