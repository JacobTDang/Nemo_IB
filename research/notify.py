"""Post a cron failure, a skipped run, or a prune result, to #ops.

The scheduled jobs post from inside the process. This command is the cron
line's failure branch: the process already exited, and the log tail is the
only record left. It posts nothing when the process reported its own failure,
which the process shows by printing a marker as the last line of the log. A
successful ``docker container prune`` has no process of its own, so that line
calls ``--ok``.

    python -m research.notify --job research-copy --log /srv/nemo/logs/research-copy.log --code 1
    python -m research.notify --job docker-prune --ok

``--code`` is the exit status of the failed line. 99 means ``flock`` found the
lock held, and 75 means the job ran and its own Discord post failed.
"""
from __future__ import annotations

import argparse
import sys

from research import discord_post


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="research.notify")
    parser.add_argument("--job", required=True)
    parser.add_argument("--log", default=None)
    parser.add_argument("--ok", action="store_true")
    parser.add_argument("--code", type=int, default=1)
    args = parser.parse_args(argv)

    if args.ok:
        return discord_post.finish(args.job, 0, {"ok": True})

    if args.log and discord_post.already_reported(args.log):
        print(f"{args.job}: the job already reported this failure to Discord")
        return 0

    tail = discord_post.log_tail(args.log) if args.log else ""
    error = tail or "the job exited non-zero and left no log tail"
    try:
        discord_post.deliver("ops", [
            discord_post.cron_message(args.job, args.code, error)])
    except discord_post.DiscordError as exc:
        print(f"{args.job}: {discord_post.redact(str(exc))}", file=sys.stderr,
              flush=True)
        return discord_post.POST_FAILED
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
