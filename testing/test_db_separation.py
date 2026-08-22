"""The disposable tool cache must not share a file with book state, and both
connections must tolerate concurrent writers."""
from __future__ import annotations

import os
import sqlite3
import sys
import threading

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def test_cache_and_state_use_different_files():
    from agent.cache import CACHE_DB_PATH
    from state.schema import DB_PATH

    assert os.path.abspath(CACHE_DB_PATH) != os.path.abspath(DB_PATH)
    assert os.path.basename(CACHE_DB_PATH) == "tool_cache.db"


def test_db_paths_are_absolute():
    # A CWD-relative path silently creates a second database when a process
    # starts from a different directory.
    from agent.cache import CACHE_DB_PATH
    from state.schema import DB_PATH

    assert os.path.isabs(DB_PATH), f"DB_PATH is relative: {DB_PATH}"
    assert os.path.isabs(CACHE_DB_PATH), f"CACHE_DB_PATH is relative: {CACHE_DB_PATH}"


def test_db_path_env_override(monkeypatch, tmp_path):
    import importlib

    target = str(tmp_path / "override.db")
    monkeypatch.setenv("NEMO_DB_PATH", target)

    import state.schema

    importlib.reload(state.schema)
    assert state.schema.DB_PATH == target

    monkeypatch.delenv("NEMO_DB_PATH")
    importlib.reload(state.schema)


def test_state_connection_uses_wal(tmp_path):
    from state.schema import get_connection

    conn = get_connection(str(tmp_path / "wal_check.db"))
    try:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode.lower() == "wal"
    finally:
        conn.close()


def test_concurrent_writers_do_not_lock(tmp_path):
    """Default journal mode raises 'database is locked' here; WAL plus a
    busy_timeout does not."""
    from state.schema import get_connection, init_schema

    db = str(tmp_path / "concurrent.db")
    init_schema(db)

    errors = []

    def _writer(n):
        try:
            conn = get_connection(db)
            for i in range(20):
                conn.execute(
                    "INSERT OR REPLACE INTO watchlist(ticker, priority) VALUES (?, ?)",
                    (f"T{n}{i}", 1),
                )
                conn.commit()
            conn.close()
        except sqlite3.OperationalError as e:
            errors.append(f"writer {n}: {e}")

    threads = [threading.Thread(target=_writer, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"concurrent writes failed: {errors}"
