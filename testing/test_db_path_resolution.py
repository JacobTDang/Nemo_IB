"""The NEMO_DB_PATH / NEMO_CACHE_DB_PATH overrides must take effect when they
are set, not only when they happen to be set before this module is imported.

A default argument binds once at function-definition time, so
`def get_connection(db_path=DB_PATH)` freezes the path at import and silently
ignores any later override. Per-module test isolation depends on this working.
"""
import os


def test_state_path_follows_env_after_import(monkeypatch, tmp_path):
    from state import schema
    target = tmp_path / "redirected_session.db"
    monkeypatch.setenv("NEMO_DB_PATH", str(target))
    assert schema.current_db_path() == str(target)


def test_cache_path_follows_env_after_import(monkeypatch, tmp_path):
    from agent import cache
    target = tmp_path / "redirected_cache.db"
    monkeypatch.setenv("NEMO_CACHE_DB_PATH", str(target))
    assert cache.current_cache_db_path() == str(target)


def test_get_connection_writes_to_the_overridden_path(monkeypatch, tmp_path):
    from state import schema
    target = tmp_path / "written.db"
    monkeypatch.setenv("NEMO_DB_PATH", str(target))
    conn = schema.get_connection()
    try:
        conn.execute("CREATE TABLE probe(x INTEGER)")
        conn.commit()
    finally:
        conn.close()
    assert target.exists(), "get_connection ignored NEMO_DB_PATH set after import"


def test_cache_writes_to_the_overridden_path(monkeypatch, tmp_path):
    """Session_Cache opened the module constant, so it ignored a later override
    exactly the way get_connection did."""
    from agent import cache
    target = tmp_path / "written_cache.db"
    monkeypatch.setenv("NEMO_CACHE_DB_PATH", str(target))
    session = cache.Session_Cache()
    try:
        assert target.exists(), "Session_Cache ignored NEMO_CACHE_DB_PATH set after import"
    finally:
        session.connection.close()


def test_defaults_are_absolute_when_env_is_unset(monkeypatch):
    monkeypatch.delenv("NEMO_DB_PATH", raising=False)
    monkeypatch.delenv("NEMO_CACHE_DB_PATH", raising=False)
    from state import schema
    from agent import cache
    assert os.path.isabs(schema.current_db_path())
    assert os.path.isabs(cache.current_cache_db_path())
