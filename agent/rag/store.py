"""CRUD layer over `rag_chunks` + `rag_chunk_embeddings`.

The two tables are kept in lockstep via a single transaction per chunk:
  1. INSERT into `rag_chunks` -> sqlite assigns chunk_id via AUTOINCREMENT
  2. INSERT into `rag_chunk_embeddings` with rowid = chunk_id
This way vector_search can JOIN by chunk_id to recover metadata.

Cosine similarity is computed by sqlite-vec's `vec_distance_cosine`, which
returns a distance in [0, 2] for L2-normalized vectors. We convert to
similarity in [0, 1] via `similarity = 1 - distance / 2`.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import sqlite_vec

from state.schema import get_connection


def _to_blob(embedding: np.ndarray) -> bytes:
    """Serialize a float32 numpy array to the sqlite-vec wire format."""
    arr = np.asarray(embedding, dtype=np.float32).flatten()
    return sqlite_vec.serialize_float32(arr.tolist())


def insert_chunk(chunk_dict: Dict[str, Any], embedding: np.ndarray) -> int:
    """Insert a chunk + its embedding atomically. Returns the new chunk_id.

    `chunk_dict` may include any subset of these keys (all others default
    to NULL): doc_id (required), ticker, source_tool, doc_type, filing_date,
    item_number, section_heading, chunk_text (required), chunk_offset,
    chunk_sequence.

    `embedding` must be a 384-dim float32 array (anything else is coerced).
    """
    doc_id = chunk_dict.get("doc_id")
    chunk_text = chunk_dict.get("chunk_text")
    if not doc_id:
        raise ValueError("insert_chunk requires chunk_dict['doc_id']")
    if chunk_text is None:
        raise ValueError("insert_chunk requires chunk_dict['chunk_text']")

    conn = get_connection()
    try:
        conn.execute("BEGIN")
        cur = conn.execute(
            """INSERT INTO rag_chunks(
                doc_id, ticker, source_tool, doc_type, filing_date,
                item_number, section_heading, chunk_text, chunk_offset,
                chunk_sequence, created_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                doc_id,
                chunk_dict.get("ticker"),
                chunk_dict.get("source_tool"),
                chunk_dict.get("doc_type"),
                chunk_dict.get("filing_date"),
                chunk_dict.get("item_number"),
                chunk_dict.get("section_heading"),
                chunk_text,
                chunk_dict.get("chunk_offset"),
                chunk_dict.get("chunk_sequence"),
                chunk_dict.get("created_at") or datetime.now().isoformat(),
            ),
        )
        chunk_id = cur.lastrowid
        conn.execute(
            "INSERT INTO rag_chunk_embeddings(rowid, embedding) VALUES (?, ?)",
            (chunk_id, _to_blob(embedding)),
        )
        conn.commit()
        return int(chunk_id)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def vector_search(
    query_embedding: np.ndarray,
    top_k: int = 10,
    ticker_filter: Optional[str] = None,
    doc_type_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Run a k-NN search against `rag_chunk_embeddings`.

    Optional metadata filters are applied AFTER the vector search using the
    JOIN against rag_chunks. We over-fetch when filters are supplied so the
    final result still contains roughly top_k matching rows.
    """
    needs_filter = bool(ticker_filter or doc_type_filter)
    fetch_k = top_k * 5 if needs_filter else top_k

    conn = get_connection()
    try:
        # sqlite-vec's MATCH-based syntax + k=? limit. The JOIN against
        # rag_chunks resolves metadata.
        sql = """
            SELECT
                c.chunk_id, c.doc_id, c.ticker, c.source_tool, c.doc_type,
                c.filing_date, c.item_number, c.section_heading,
                c.chunk_text, c.chunk_offset, c.chunk_sequence, c.created_at,
                e.distance AS distance
            FROM rag_chunk_embeddings e
            JOIN rag_chunks c ON c.chunk_id = e.rowid
            WHERE e.embedding MATCH ?
              AND k = ?
        """
        params: List[Any] = [_to_blob(query_embedding), fetch_k]
        if ticker_filter:
            sql += " AND c.ticker = ?"
            params.append(ticker_filter)
        if doc_type_filter:
            sql += " AND c.doc_type = ?"
            params.append(doc_type_filter)
        sql += " ORDER BY e.distance ASC"

        rows = conn.execute(sql, params).fetchall()
        results: List[Dict[str, Any]] = []
        for r in rows[:top_k]:
            d = dict(r)
            # Cosine distance is in [0, 2] for L2-normalized vectors;
            # convert to similarity in [0, 1].
            dist = d.pop("distance")
            try:
                d["similarity"] = max(0.0, min(1.0, 1.0 - float(dist) / 2.0))
            except (TypeError, ValueError):
                d["similarity"] = 0.0
            results.append(d)
        return results
    finally:
        conn.close()


def delete_by_doc_id(doc_id: str) -> int:
    """Remove all chunks (and their embeddings) for a given doc_id.

    Returns the number of rag_chunks rows deleted. Idempotent — calling on
    an unknown doc_id is a 0-row no-op.
    """
    conn = get_connection()
    try:
        conn.execute("BEGIN")
        # Look up chunk_ids first so we can delete from the vec0 table by
        # rowid (vec0 doesn't support DELETE ... WHERE non-rowid).
        rows = conn.execute(
            "SELECT chunk_id FROM rag_chunks WHERE doc_id = ?", (doc_id,)
        ).fetchall()
        ids = [r["chunk_id"] for r in rows]
        for cid in ids:
            conn.execute(
                "DELETE FROM rag_chunk_embeddings WHERE rowid = ?", (cid,)
            )
        cur = conn.execute(
            "DELETE FROM rag_chunks WHERE doc_id = ?", (doc_id,)
        )
        deleted = cur.rowcount
        conn.commit()
        return int(deleted)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def count_chunks(filter: Optional[Dict[str, Any]] = None) -> int:
    """Count chunks, optionally restricted by metadata filters.

    `filter` keys supported: ticker, doc_type, source_tool, doc_id. Any
    unsupported keys are ignored.
    """
    conn = get_connection()
    try:
        sql = "SELECT COUNT(*) AS n FROM rag_chunks"
        params: List[Any] = []
        if filter:
            clauses = []
            for col in ("ticker", "doc_type", "source_tool", "doc_id"):
                if col in filter and filter[col] is not None:
                    clauses.append(f"{col} = ?")
                    params.append(filter[col])
            if clauses:
                sql += " WHERE " + " AND ".join(clauses)
        row = conn.execute(sql, params).fetchone()
        return int(row["n"]) if row else 0
    finally:
        conn.close()
