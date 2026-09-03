"""Least-recently-used eviction for the filing cache.

edgartools caches every document it fetches under `~/.edgar` and never removes
one. In the homelab image that path is a 512MB tmpfs, chosen deliberately: the
cache is disposable and rebuilding it costs only re-fetching. But a bound with
no eviction does not limit a cache, it schedules an outage -- the mount reached
100% in the running deployment and every SEC read began failing with
`[Errno 28] No space left on device`.

Sizes are what the filesystem allocates (`st_blocks`), because that is what
fills a tmpfs: tens of thousands of tiny metadata files each cost a page.

Pruning is by access time, so a filing being read repeatedly survives while one
fetched once months ago does not. It trims to a fraction of the cap rather than
to the cap itself, because a single filing can be 89MB and trimming to exactly
the limit would refill on the next fetch.

Two callers drive it, on the same cap and the same interval. The HTTP servers
run `prune_and_log` as a background task off the app's lifespan, sleeping
between passes. The batch jobs have no event loop and no idle moment to sleep
in -- they are a tight sequential sweep over thousands of names into the same
tmpfs -- so they ask `prune_if_due` between names instead.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Defaults sit under the 512MB tmpfs in deploy/docker-compose.yml. Overridable
# because the same code runs on a laptop where ~/.edgar is ordinary disk.
DEFAULT_TARGET_FRACTION = 0.7


def cap_bytes() -> int:
    """Read on use, not at import, so the value a test or a compose file sets
    is the value that applies. Read once at import, an override would be
    accepted silently and ignored."""
    return int(os.environ.get("NEMO_FILING_CACHE_CAP_MB", "400")) * 1024 * 1024


def interval_seconds() -> int:
    return max(int(os.environ.get("NEMO_FILING_CACHE_INTERVAL_S", "300")), 1)


def cache_dir() -> Path:
    return Path(os.environ.get("NEMO_FILING_CACHE_DIR",
                               str(Path.home() / ".edgar")))


def prune(root: Any, cap_bytes: int | None = None,
          target_fraction: float = DEFAULT_TARGET_FRACTION) -> dict:
    """Evict least-recently-used files until `root` fits under the target.

    Returns what it did rather than logging and forgetting: a prune that
    quietly failed would leave the disk full while reporting that it cleaned.
    """
    root = Path(root)
    if cap_bytes is None:
        cap_bytes = globals()["cap_bytes"]()
    report: dict[str, Any] = {
        "existed": root.is_dir(), "bytes_before": 0, "bytes_after": 0,
        "bytes_removed": 0, "removed_files": 0, "failed": [],
    }
    if not report["existed"]:
        return report

    entries = []
    for path in root.rglob("*"):
        try:
            if not path.is_file():
                continue
            stat = path.stat()
        except OSError as exc:            # vanished mid-walk, or unreadable
            report["failed"].append(f"{path}: {type(exc).__name__}: {exc}")
            continue
        # What the filesystem charges, not the apparent length. tmpfs bills
        # whole pages, and edgartools' hishel cache writes three files per
        # object -- body, a sub-kilobyte .meta, an empty .lock -- so on a
        # live cache the page charge ran 1.39x the st_size sum and the 512m
        # mount filled under a 400MB cap without a single prune (issue #98).
        entries.append((stat.st_atime, stat.st_blocks * 512, path))

    total = sum(size for _, size, _ in entries)
    report["bytes_before"] = total
    report["bytes_after"] = total
    if total <= cap_bytes:
        return report

    target = int(cap_bytes * target_fraction)
    entries.sort(key=lambda item: item[0])          # oldest access first

    for _, size, path in entries:
        if total <= target:
            break
        try:
            os.remove(path)
        except OSError as exc:
            report["failed"].append(f"{path}: {type(exc).__name__}: {exc}")
            continue
        total -= size
        report["bytes_removed"] += size
        report["removed_files"] += 1

    report["bytes_after"] = total
    return report


def prune_and_log(root: Any = None, cap: int | None = None) -> dict:
    """Prune, and say what happened. Never raises: a janitor that kills the
    server it tidies for is worse than a full cache."""
    try:
        report = prune(root if root is not None else cache_dir(), cap)
    except Exception as exc:              # noqa: BLE001 - reported, not masked
        log.error("filing-cache prune failed: %s: %s", type(exc).__name__, exc)
        return {"failed": [f"{type(exc).__name__}: {exc}"], "removed_files": 0}

    if report["removed_files"]:
        log.info("filing-cache: evicted %d file(s), freed %.1f MB, now %.1f MB",
                 report["removed_files"], report["bytes_removed"] / 1048576,
                 report["bytes_after"] / 1048576)
    for failure in report["failed"]:
        log.warning("filing-cache: could not remove %s", failure)
    return report


# When the last prune ran, on the monotonic clock so a system time change
# cannot make one overdue by years or postpone the next one indefinitely.
# Process-global because the thing being rationed is a directory, not a
# caller: a batch job is one sweep in one process, and the servers run their
# own janitor on a timer instead.
_last_prune: float | None = None


def prune_if_due(root: Any = None, cap: int | None = None) -> dict | None:
    """`prune_and_log`, but at most once per `interval_seconds()`.

    For the callers that cannot sleep. A batch job sweeping the eligible
    universe fetches documents into the same capped tmpfs the servers do, and
    the janitor that keeps the servers alive is an asyncio task the jobs never
    start -- so the jobs ask between names whether a prune is owed. Ungated,
    that would rglob the whole cache between every SEC request.

    Returns None when a prune is not due, so a caller can tell that apart from
    a prune that ran and found nothing to remove.
    """
    global _last_prune

    now = time.monotonic()
    if _last_prune is not None and now - _last_prune < interval_seconds():
        return None
    # Stamped before the walk, not after: a prune of a full cache takes real
    # time, and dating it from the end would let a slow one push the next
    # one out by its own duration.
    _last_prune = now
    return prune_and_log(root, cap)
