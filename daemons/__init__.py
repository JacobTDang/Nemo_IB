"""Long-running daemons that observe the world and write to state/.

Each daemon is meant to run as its own process. They share state via the
SQLite DB at db_cache/session.db (events, theses, positions tables).
"""
