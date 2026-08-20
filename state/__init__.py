"""Persistent state layer for the autonomous monitoring system.

This package owns the SQLite tables that outlive a single agent run:
  - watchlist     (Phase 0)
  - events        (Phase 1, news watcher writes; thesis maintainer reads)
  - theses        (Phase 2, analyze_node writes; future runs read)
  - positions     (Phase 6, execution agent writes)
  - orders        (Phase 6, execution agent writes)

The disposable tool/news/scrape caches live separately in db_cache/tool_cache.db
(see agent/cache.py) so that clearing a cache can never take book state with it.
"""
from state.schema import init_schema, get_connection

__all__ = ['init_schema', 'get_connection']
