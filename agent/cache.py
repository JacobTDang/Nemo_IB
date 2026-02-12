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

    def clear(self):
        self.cursor.execute("""
                            DELETE FROM tool_cache

                            """)
        self.connection.commit()


if __name__ == "__main__":
    pass
