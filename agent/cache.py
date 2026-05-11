import sqlite3
import hashlib
import json
import os
from datetime import datetime
from typing import Optional, Dict, Any

class Session_Cache():
    def __init__(self):
        os.makedirs("db_cache", exist_ok=True)
        self.connection = sqlite3.connect("db_cache/session.db")
        self.cursor = self.connection.cursor()
        self.create_session()

    def create_session(self) -> None:
        self.cursor.execute("""CREATE TABLE IF NOT EXISTS tool_cache(
                            tool_name TEXT,
                            args_hash TEXT,
                            args_json TEXT,
                            result_json TEXT,
                            created_at TIMESTAMP
                            )""")

        self.cursor.execute("""CREATE TABLE IF NOT EXISTS news_cache(
                            article_name TEXT,
                            args_hash TEXT,
                            args_json TEXT,
                            result_article TEXT,
                            created_at TIMESTAMP
        )""")

        # URL-keyed scrape cache. INSERT OR REPLACE on PRIMARY KEY makes writes idempotent
        # so re-scraping the same URL within a session just refreshes the row.
        self.cursor.execute("""CREATE TABLE IF NOT EXISTS scrape_cache(
                            url TEXT PRIMARY KEY,
                            content_json TEXT,
                            created_at TIMESTAMP
        )""")
        self.connection.commit()

    def get(self, tool_name: str, args: Dict[str, str]) -> Optional[Dict[str, str]]:
        # create the args string,
        json_args_hash = hashlib.sha256(json.dumps(args, sort_keys = True).encode()).hexdigest()

        self.cursor.execute("""
                            SELECT result_json FROM tool_cache WHERE tool_name = ? AND args_hash = ?
                            """,
                            (tool_name, json_args_hash)
                            )
        # grab the query result
        row = self.cursor.fetchone()

        if row is None:
            return None


        # turn the str back into a dict
        result_dict = json.loads(row[0])
        return result_dict

    def put(self, tool_name: str, args: Dict[str,Any], result_args: Dict[str, Any]) -> None:
        args_json,result_json = json.dumps(args, sort_keys=True), json.dumps(result_args, sort_keys=True)
        # create the hash
        args_hash= hashlib.sha256(json.dumps(args, sort_keys = True).encode()).hexdigest()

        created_at = datetime.now().isoformat()

        self.cursor.execute("""
                            INSERT INTO tool_cache (tool_name, args_hash, args_json, result_json, created_at) VALUES (?,?,?,?,?)
                            """,
                            (tool_name, args_hash, args_json, result_json, created_at)
                            )
        self.connection.commit()
    
    @staticmethod
    def _articles_hash(articles: list) -> str:
        """Hash news by article headline + datetime, not by query args.

        Same set of articles always produces the same hash regardless of the
        from_date/to_date used to fetch them. This prevents stale cache misses
        when run dates shift by one day but the article set hasn't changed.
        """
        keys = sorted(
            f"{a.get('headline', '')}|{a.get('datetime', '')}"
            for a in articles
            if isinstance(a, dict)
        )
        return hashlib.sha256(json.dumps(keys).encode()).hexdigest()

    def get_news(self, tool_name: str, articles: list) -> Optional[Dict[str, Any]]:
        """Retrieve a cached news analysis result keyed by article content.

        Returns None on cache miss. Pass the raw article list returned by
        Finnhub -- the hash is derived from article headlines and datetimes
        so the same articles always return the same cached analysis.
        """
        articles_hash = self._articles_hash(articles)
        self.cursor.execute("""
                            SELECT result_article FROM news_cache WHERE article_name = ? AND args_hash = ?
                            """,
                            (tool_name, articles_hash))
        row = self.cursor.fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def put_news(self, tool_name: str, articles: list, result: Dict[str, Any]) -> None:
        """Cache a news analysis result keyed by article content hash."""
        articles_hash = self._articles_hash(articles)
        # Store a compact summary of what was hashed for human inspection
        articles_json = json.dumps(sorted(
            f"{a.get('headline', '')}|{a.get('datetime', '')}"
            for a in articles
            if isinstance(a, dict)
        ))
        result_json = json.dumps(result, sort_keys=True)
        created_at = datetime.now().isoformat()

        self.cursor.execute("""
                            INSERT INTO news_cache (article_name, args_hash, args_json, result_article, created_at) VALUES (?,?,?,?,?)
                            """,
                            (tool_name, articles_hash, articles_json, result_json, created_at))
        self.connection.commit()


    def get_scrape(self, url: str) -> Optional[Dict[str, Any]]:
        """Retrieve a cached scrape result by URL. Returns None on miss."""
        self.cursor.execute(
            "SELECT content_json FROM scrape_cache WHERE url = ?",
            (url,)
        )
        row = self.cursor.fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def put_scrape(self, url: str, result: Dict[str, Any]) -> None:
        """Cache a scraped page by URL. Never stores failed scrapes."""
        if not (isinstance(result, dict) and result.get('success')):
            return
        content_json = json.dumps(result, sort_keys=True)
        created_at = datetime.now().isoformat()
        self.cursor.execute(
            """INSERT OR REPLACE INTO scrape_cache (url, content_json, created_at)
               VALUES (?, ?, ?)""",
            (url, content_json, created_at)
        )
        self.connection.commit()

    def clear(self):
        self.cursor.execute("DELETE FROM tool_cache")
        self.cursor.execute("DELETE FROM news_cache")
        self.cursor.execute("DELETE FROM scrape_cache")
        self.connection.commit()


if __name__ == "__main__":
    pass
