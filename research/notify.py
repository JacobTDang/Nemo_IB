"""Post a cron failure, or a prune result, to #ops.

The scheduled jobs post from inside the process. This command is the cron
line's failure branch: the process already exited, and the log tail is the
only record left. A successful ``docker container prune`` has no process of
its own, so that line calls ``--ok`` instead.

    python -m research.notify --job research-copy --log /srv/nemo/logs/research-copy.log
    python -m research.notify --job docker-prune --ok
"""
from __future__ import annotations

import argparse

from research import discord_post


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="research.notify")
    parser.add_argument("--job", required=True)
    parser.add_argument("--log", default=None)
    parser.add_argument("--ok", action="store_true")
    parser.add_argument("--exit-code", dest="exit_code", type=int, default=None)
    args = parser.parse_args(argv)

    if args.ok or args.exit_code == 0:
        return discord_post.finish(args.job, 0, {"ok": True})

    tail = discord_post.log_tail(args.log) if args.log else ""
    error = tail or "the job exited non-zero and left no log tail"
    return discord_post.finish(args.job, 1, {"ok": False, "error": error})


if __name__ == "__main__":
    raise SystemExit(main())
