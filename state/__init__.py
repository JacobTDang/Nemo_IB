"""Persistent state layer for the autonomous monitoring system.

This package owns the SQLite tables that outlive a single agent run:
  - watchlist     (Phase 0)
  - events        (Phase 1, news watcher writes; thesis maintainer reads)
  - theses        (Phase 2, analyze_node writes; future runs read)
  - positions     (Phase 6, execution agent writes)
  - orders        (Phase 6, execution agent writes)

The existing Session_Cache (agent/cache.py) holds short-lived tool/news/scrape
caches in the same db_cache/session.db. Tables created here coexist with those.
"""
from state.schema import init_schema, get_connection

__all__ = ['init_schema', 'get_connection']
