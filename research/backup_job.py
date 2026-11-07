"""Nightly copy of the point-in-time store.

Refuses to replace yesterday's copy with an empty file. A mistyped volume
gives an empty source path, and a backup whose failure looks like a valid
database is worse than no backup. The file name carries the UTC date.
"""
from __future__ import annotations

import datetime
import pathlib
import sqlite3
import sys

from research import discord_post


SOURCE = pathlib.Path("/app/data/pit.db")
BACKUPS = pathlib.Path("/backups")


def main() -> int:
    elapsed = discord_post.stopwatch()
    if not SOURCE.exists():
        message = f"{SOURCE} is not there -- check the volume mount"
        print(message, file=sys.stderr)
        return discord_post.finish(
            "research-backup", 1, {"bytes": 0, "removed": 0, "error": message},
            seconds=elapsed())
    BACKUPS.mkdir(parents=True, exist_ok=True)
    today = datetime.datetime.now(datetime.timezone.utc).date()
    dest = BACKUPS / f"pit-{today.isoformat()}.db"
    source, target = sqlite3.connect(SOURCE), sqlite3.connect(dest)
    try:
        source.backup(target)
    finally:
        target.close()
        source.close()
    stale = sorted(BACKUPS.glob("pit-*.db"))[:-14]
    for path in stale:
        path.unlink()
    print(f"wrote {dest} ({dest.stat().st_size} bytes), "
          f"removed {len(stale)} older than the last 14")
    return discord_post.finish("research-backup", 0, {
        "bytes": dest.stat().st_size,
        "removed": len(stale),
    }, seconds=elapsed())


if __name__ == "__main__":
    raise SystemExit(main())
