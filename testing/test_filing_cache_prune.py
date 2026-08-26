"""A capped cache with no eviction is a server that stops working.

`/root/.edgar` is a 512MB tmpfs. edgartools caches every filing document it
fetches and never removes one; a single INTC filing measured 89MB. In the
running deployment the mount reached 100% and calls began failing outright
with `[Errno 28] No space left on device` -- not a slow tool, not a partial
answer, a hard error on every SEC read.

The cap was deliberate and is right: the cache is disposable and RAM-backed.
What was missing is the other half. A bound without eviction does not limit
the cache, it schedules an outage for whenever the bound is reached.

Eviction is least-recently-used and leaves headroom rather than trimming to
exactly the cap, so a single large filing arriving right after a prune does
not immediately refill it.
"""
import os
import time

import pytest

from tools import filing_cache


def _write(path, size, atime=None):
    path.write_bytes(b"x" * size)
    if atime is not None:
        os.utime(path, (atime, atime))
    return path


def test_a_cache_under_its_cap_is_left_alone(tmp_path):
    _write(tmp_path / "a.txt", 100)
    report = filing_cache.prune(tmp_path, cap_bytes=10_000)

    assert report["removed_files"] == 0
    assert (tmp_path / "a.txt").exists()


def test_the_least_recently_used_file_goes_first(tmp_path):
    now = time.time()
    _write(tmp_path / "old.txt", 4_000, atime=now - 9_000)
    _write(tmp_path / "new.txt", 4_000, atime=now)

    # 8,000 bytes against a 6,000 cap trimming to 4,500: exactly one file
    # has to go, so which one is the whole question.
    filing_cache.prune(tmp_path, cap_bytes=6_000, target_fraction=0.75)

    assert not (tmp_path / "old.txt").exists()
    assert (tmp_path / "new.txt").exists(), "evicted the file still in use"


def test_pruning_leaves_headroom_rather_than_trimming_to_the_cap(tmp_path):
    now = time.time()
    for i in range(10):
        _write(tmp_path / f"f{i}.txt", 1_000, atime=now - (10 - i) * 100)

    # 10,000 bytes against an 8,000 cap: over it, so a prune is due, and the
    # target of 4,800 is what it must reach rather than merely 8,000.
    filing_cache.prune(tmp_path, cap_bytes=8_000, target_fraction=0.6)

    remaining = sum(p.stat().st_size for p in tmp_path.iterdir())
    assert remaining <= 4_800, "trimmed to the cap instead of below it"
    assert remaining > 0, "emptied the cache instead of trimming it"


def test_a_cache_directory_that_does_not_exist_is_not_an_error(tmp_path):
    report = filing_cache.prune(tmp_path / "never-created", cap_bytes=1_000)
    assert report["removed_files"] == 0
    assert report["existed"] is False


def test_a_file_that_cannot_be_removed_is_reported_not_swallowed(tmp_path,
                                                                monkeypatch):
    """A prune that silently fails leaves the disk full and says it cleaned."""
    _write(tmp_path / "stuck.txt", 8_000)

    def _refuse(path):
        raise PermissionError("read-only")

    monkeypatch.setattr(filing_cache.os, "remove", _refuse)
    report = filing_cache.prune(tmp_path, cap_bytes=1_000)

    assert report["failed"], "a failed removal was not reported"
    assert report["removed_files"] == 0


def test_nested_files_are_counted_and_evictable(tmp_path):
    """edgartools stores filings in per-CIK subdirectories."""
    nested = tmp_path / "0001045810" / "10-K"
    nested.mkdir(parents=True)
    now = time.time()
    _write(nested / "big.txt", 9_000, atime=now - 5_000)
    _write(tmp_path / "small.txt", 500, atime=now)

    report = filing_cache.prune(tmp_path, cap_bytes=2_000, target_fraction=0.5)

    assert report["removed_files"] == 1
    assert not (nested / "big.txt").exists()
    assert (tmp_path / "small.txt").exists()


def test_the_report_says_how_much_it_freed(tmp_path):
    now = time.time()
    _write(tmp_path / "a.txt", 6_000, atime=now - 1_000)
    report = filing_cache.prune(tmp_path, cap_bytes=1_000, target_fraction=0.5)

    assert report["bytes_removed"] == 6_000
    assert report["bytes_before"] == 6_000
    assert report["bytes_after"] == 0


# --- the janitor has to actually run -----------------------------------------
#
# A pruner nothing calls is a pruner that does not exist. The cache filled in a
# long-lived container, which is exactly the process that has a lifespan to
# hang it on.


def test_the_server_lifespan_prunes_the_cache(monkeypatch):
    import asyncio

    from mcp.server import Server

    from tools import mcp_http

    calls = []
    monkeypatch.setattr(mcp_http.filing_cache, "prune_and_log",
                        lambda *a, **k: calls.append(1))
    monkeypatch.setenv("NEMO_FILING_CACHE_INTERVAL_S", "1")

    app = mcp_http.build_app(Server("prune-test"), auth_token=None)

    async def drive():
        async with app.router.lifespan_context(app):
            await asyncio.sleep(0.1)

    asyncio.run(drive())
    assert calls, "the lifespan came and went without pruning the cache"


def test_the_janitor_never_takes_the_server_down_with_it(monkeypatch, tmp_path):
    """A tidy-up that crashes the process it tidies for is worse than a full
    cache."""
    from tools import filing_cache

    def _explode(*a, **k):
        raise OSError("the mount went away")

    monkeypatch.setattr(filing_cache, "prune", _explode)
    report = filing_cache.prune_and_log(tmp_path)
    assert report["failed"], "the failure was swallowed without a trace"


def test_the_cap_is_read_when_it_is_used_not_when_the_module_loads(monkeypatch,
                                                                   tmp_path):
    """Read at import, an override in compose or a test would be accepted
    silently and ignored -- the worst kind of setting."""
    _write(tmp_path / "a.txt", 3_000, atime=time.time() - 1_000)

    monkeypatch.setenv("NEMO_FILING_CACHE_CAP_MB", "1")   # 1MB: nothing to do
    assert filing_cache.prune(tmp_path)["removed_files"] == 0

    monkeypatch.setattr(filing_cache, "cap_bytes", lambda: 1_000)
    assert filing_cache.prune(tmp_path)["removed_files"] == 1


def test_the_interval_is_read_the_same_way(monkeypatch):
    monkeypatch.setenv("NEMO_FILING_CACHE_INTERVAL_S", "42")
    assert filing_cache.interval_seconds() == 42
    monkeypatch.setenv("NEMO_FILING_CACHE_INTERVAL_S", "0")
    assert filing_cache.interval_seconds() == 1, "a zero interval would spin"
