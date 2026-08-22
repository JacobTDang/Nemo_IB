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


# ---------------------------------------------------------------------------
# sqlite-vec absence reporting
# ---------------------------------------------------------------------------
# The homelab container image deliberately ships without the RAG stack (no
# torch, no sentence-transformers, no sqlite-vec). init_schema must still
# succeed there, and the message it prints must read as an expected capability
# gap -- naming the lost feature and the extra that restores it -- rather than
# as a bare failure. It must not be silenced: a missing extension is worth
# reporting wherever RAG is supposed to work.

def _init_schema_without_sqlite_vec(tmp_path, capsys):
  """Run init_schema with sqlite_vec unimportable; return (stdout, exception)."""
  import builtins

  real_import = builtins.__import__

  def fake_import(name, *args, **kwargs):
    if name == "sqlite_vec":
      raise ImportError("No module named 'sqlite_vec'")
    return real_import(name, *args, **kwargs)

  builtins.__import__ = fake_import
  raised = None
  try:
    init_schema(str(tmp_path / "novec.db"))
  except Exception as exc:      # noqa: BLE001 - the test asserts on this
    raised = exc
  finally:
    builtins.__import__ = real_import
  return capsys.readouterr().out, raised


def test_missing_sqlite_vec_does_not_break_init(tmp_path, capsys):
  """The rest of the schema must still be created without sqlite-vec."""
  _out, raised = _init_schema_without_sqlite_vec(tmp_path, capsys)
  assert raised is None, f"init_schema raised without sqlite-vec: {raised!r}"

  conn = get_connection(str(tmp_path / "novec.db"))
  try:
    names = {r['name'] for r in conn.execute(
      "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    assert {'watchlist', 'events', 'theses', 'rag_chunks'} <= names
  finally:
    conn.close()


def test_missing_sqlite_vec_message_is_accurate(tmp_path, capsys):
  out, _raised = _init_schema_without_sqlite_vec(tmp_path, capsys)
  lower = out.lower()

  # Still reported -- not silenced.
  assert "sqlite-vec" in lower or "sqlite_vec" in lower, \
    f"missing sqlite-vec was not reported at all:\n{out}"

  # Says what is actually lost, in user-facing terms.
  assert "vector search" in lower, \
    f"message does not name the lost capability (vector search):\n{out}"

  # Frames it as expected when the RAG extras are absent, not as a failure.
  assert "expected" in lower, \
    f"message does not say this is expected without the RAG extras:\n{out}"
  assert "rag" in lower, f"message does not mention RAG:\n{out}"

  # Must not read as a bare failure/warning about a table that "wasn't created".
  assert "warning:" not in lower, \
    f"message still reads as a bare warning:\n{out}"

  # The underlying cause is still surfaced, not swallowed.
  assert "no module named" in lower, \
    f"underlying exception detail was swallowed:\n{out}"


def test_sqlite_vec_present_creates_embeddings_table(tmp_path):
  """When sqlite-vec IS installed the vec0 table must actually be created."""
  import pytest
  try:
    import sqlite_vec  # noqa: F401
  except ImportError:
    pytest.skip("sqlite-vec not installed in this environment")

  db = str(tmp_path / "withvec.db")
  init_schema(db)
  conn = get_connection(db)
  try:
    names = {r['name'] for r in conn.execute(
      "SELECT name FROM sqlite_master WHERE name='rag_chunk_embeddings'"
    ).fetchall()}
    assert 'rag_chunk_embeddings' in names
  finally:
    conn.close()
