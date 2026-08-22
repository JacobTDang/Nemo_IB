"""Phase 0: schema initialization + watchlist CRUD."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib

from state.schema import (
  init_schema, get_connection, add_to_watchlist,
  get_watchlist, remove_from_watchlist,
)


def test_init_creates_all_tables():
  # Durable state DB: only the book-state tables live here. The disposable
  # tool/news/scrape caches live in a separate db_cache/tool_cache.db (see
  # agent/cache.py) and are checked by test_init_creates_cache_tables below.
  init_schema()
  conn = get_connection()
  try:
    names = {r['name'] for r in conn.execute(
      "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    required = {'watchlist', 'events', 'theses', 'positions', 'orders'}
    missing = required - names
    assert not missing, f"missing tables: {missing}"
    print(f"PASS: all {len(required)} expected state tables exist")
  finally:
    conn.close()


def test_init_creates_cache_tables(monkeypatch, tmp_path):
  # Disposable cache DB, split off from state in this refactor. Point it at a
  # tmp path so the test proves fresh creation rather than reusing whatever
  # already sits in the developer's local db_cache/tool_cache.db.
  target = str(tmp_path / "tool_cache.db")
  monkeypatch.setenv("NEMO_CACHE_DB_PATH", target)

  import agent.cache
  importlib.reload(agent.cache)
  try:
    cache = agent.cache.Session_Cache()
    try:
      names = {r[0] for r in cache.cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
      ).fetchall()}
      required = {'tool_cache', 'news_cache', 'scrape_cache'}
      missing = required - names
      assert not missing, f"missing tables: {missing}"
      print(f"PASS: all {len(required)} expected cache tables exist")
    finally:
      cache.connection.close()
  finally:
    monkeypatch.delenv("NEMO_CACHE_DB_PATH", raising=False)
    importlib.reload(agent.cache)


def test_watchlist_crud():
  add_to_watchlist('TEST_TICKER_X', priority=2, notes='test')
  wl = get_watchlist()
  assert 'TEST_TICKER_X' in wl
  remove_from_watchlist('TEST_TICKER_X')
  wl2 = get_watchlist()
  assert 'TEST_TICKER_X' not in wl2
  print(f"PASS: watchlist add/get/remove works (current: {wl2})")


def test_idempotent_add():
  add_to_watchlist('IDEMPOTENT_TEST')
  add_to_watchlist('IDEMPOTENT_TEST')
  add_to_watchlist('IDEMPOTENT_TEST')
  conn = get_connection()
  try:
    count = conn.execute(
      "SELECT COUNT(*) c FROM watchlist WHERE ticker='IDEMPOTENT_TEST'"
    ).fetchone()['c']
    assert count == 1, f"expected 1 row, got {count}"
  finally:
    conn.close()
  remove_from_watchlist('IDEMPOTENT_TEST')
  print("PASS: duplicate adds are idempotent")


if __name__ == "__main__":
  test_init_creates_all_tables()
  test_watchlist_crud()
  test_idempotent_add()
  print("\nAll tests passed.")
