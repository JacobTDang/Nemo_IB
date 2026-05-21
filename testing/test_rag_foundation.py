"""Stress-test the RAG foundation layer (chunker, embedder, store, schema).

Run from project root:
  ./.venv/Scripts/python.exe testing/test_rag_foundation.py

Each assertion goes through `_check(name, condition, hint)` which counts
passes / fails. Final line is `PASS N FAIL M` so this is easy to grep from
CI. Tests use a unique "FOUND_TEST_" doc_id prefix so they can clean up
after themselves without touching production rag_chunks rows.
"""
from __future__ import annotations

import os
import sys
import time
import traceback

# Make project root importable when run as a script.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np

from agent.rag import chunker, embedder, store
from state.schema import get_connection, init_schema


# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------

_PASS = 0
_FAIL = 0
_FAILURES: list = []


def _check(name: str, condition: bool, hint: str = "") -> None:
    global _PASS, _FAIL
    if condition:
        _PASS += 1
        print(f"  PASS  {name}")
    else:
        _FAIL += 1
        msg = f"  FAIL  {name}" + (f"  ({hint})" if hint else "")
        print(msg)
        _FAILURES.append(msg)


def _section(title: str) -> None:
    print(f"\n--- {title} ---")


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ---------------------------------------------------------------------------
# Cleanup helper — strip any leftover FOUND_TEST_ rows from a prior run.
# ---------------------------------------------------------------------------

DOC_PREFIX = "FOUND_TEST_"


def _cleanup_test_rows() -> None:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT chunk_id FROM rag_chunks WHERE doc_id LIKE ?",
            (DOC_PREFIX + "%",),
        ).fetchall()
        for r in rows:
            try:
                conn.execute(
                    "DELETE FROM rag_chunk_embeddings WHERE rowid = ?",
                    (r["chunk_id"],),
                )
            except Exception:
                pass
        conn.execute("DELETE FROM rag_chunks WHERE doc_id LIKE ?", (DOC_PREFIX + "%",))
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 1. Chunker tests
# ---------------------------------------------------------------------------

def test_chunker() -> None:
    _section("Chunker")

    # Empty
    chunks = chunker.chunk_text("")
    _check("empty string -> 0 chunks", len(chunks) == 0, f"got {len(chunks)}")
    chunks = chunker.chunk_text("   \n\n  ")
    _check("whitespace-only -> 0 chunks", len(chunks) == 0, f"got {len(chunks)}")

    # Short text — should produce exactly one chunk.
    short = (
        "Microsoft reported a strong quarter driven by Azure cloud "
        "revenue growth of 30% YoY. CFO Amy Hood guided to similar "
        "growth next quarter."
    )
    chunks = chunker.chunk_text(short, target_tokens=500, overlap_tokens=50)
    _check("short text -> 1 chunk", len(chunks) == 1, f"got {len(chunks)}")
    _check(
        "short chunk has sequence=0",
        chunks and chunks[0]["chunk_sequence"] == 0,
        "sequence wrong",
    )
    _check(
        "short chunk preserves text",
        chunks and "Azure" in chunks[0]["chunk_text"],
        "text missing",
    )

    # Long text — many paragraphs to force multi-chunk emission.
    paragraph = (
        "Azure cloud services delivered 30 percent revenue growth driven by "
        "enterprise AI workloads, accelerating GPU-backed compute consumption "
        "across financial-services and healthcare verticals. The segment "
        "remains the primary engine of Microsoft's commercial business. "
    ) * 6
    long_text = "\n\n".join([paragraph] * 12)
    chunks = chunker.chunk_text(long_text, target_tokens=200, overlap_tokens=30)
    _check("long text -> multiple chunks", len(chunks) >= 3, f"got {len(chunks)}")

    seqs = [c["chunk_sequence"] for c in chunks]
    _check(
        "chunk_sequence is contiguous 0..N-1",
        seqs == list(range(len(chunks))),
        f"got {seqs}",
    )

    offsets = [c["chunk_offset"] for c in chunks]
    monotone = all(offsets[i] <= offsets[i + 1] for i in range(len(offsets) - 1))
    _check("chunk_offset monotonically increasing", monotone, f"offsets={offsets}")

    # Section heading preservation
    chunks_h = chunker.chunk_text(long_text, target_tokens=200,
                                   overlap_tokens=30,
                                   section_heading="Item 1A. Risk Factors")
    all_heading = all(c["section_heading"] == "Item 1A. Risk Factors"
                      for c in chunks_h)
    _check("section_heading preserved on every chunk", all_heading,
           "heading missing from some chunks")

    # Token-count sizing — every chunk except possibly the last is within
    # roughly the target band. Allow generous slack for short paragraph
    # remainders (the chunker emits whole paragraphs).
    target = 200
    in_band = 0
    for c in chunks[:-1]:
        if c["token_count"] >= target * 0.5:
            in_band += 1
    _check(
        "most non-final chunks reach >= 50% of target_tokens",
        in_band >= max(1, len(chunks) - 1) // 2,
        f"only {in_band}/{len(chunks)-1} chunks reached threshold",
    )


# ---------------------------------------------------------------------------
# 2. Embedder tests
# ---------------------------------------------------------------------------

def test_embedder() -> None:
    _section("Embedder (first call may take 10-30s to download model)")

    t0 = time.time()
    v = embedder.embed("Hello world")
    dt = time.time() - t0
    print(f"  first embed took {dt:.2f}s")

    _check("embed() returns numpy array", isinstance(v, np.ndarray))
    _check("embed() shape == (384,)", v.shape == (384,), f"got {v.shape}")
    _check("embed() dtype == float32", v.dtype == np.float32, f"got {v.dtype}")

    # Determinism
    v2 = embedder.embed("Hello world")
    _check("identical inputs -> identical embeddings",
           np.allclose(v, v2, atol=1e-6),
           f"max diff {np.max(np.abs(v - v2))}")

    # Distinctness
    v3 = embedder.embed("Microsoft Azure cloud growth in financial services")
    sim_distinct = _cosine(v, v3)
    _check("different inputs -> cosine < 0.99",
           sim_distinct < 0.99,
           f"sim={sim_distinct:.4f}")

    # Batch
    texts = ["alpha", "beta", "gamma", "delta", "epsilon"]
    batch = embedder.embed_batch(texts)
    _check("embed_batch shape == (5, 384)", batch.shape == (5, 384),
           f"got {batch.shape}")
    _check("embed_batch dtype == float32", batch.dtype == np.float32)

    # Semantic similarity ordering — related Azure phrases should be closer
    # to each other than an Azure phrase vs an unrelated tax-policy phrase.
    e_a = embedder.embed("Azure cloud revenue growth")
    e_b = embedder.embed("Microsoft Azure cloud growth")
    e_c = embedder.embed("Federal tax policy reform")
    sim_related = _cosine(e_a, e_b)
    sim_unrelated = _cosine(embedder.embed("Azure"), e_c)
    _check(
        "Azure phrases more similar than Azure vs tax policy",
        sim_related > sim_unrelated,
        f"related={sim_related:.3f} vs unrelated={sim_unrelated:.3f}",
    )


# ---------------------------------------------------------------------------
# 3. Store tests
# ---------------------------------------------------------------------------

def test_store() -> None:
    _section("Store")

    # Ensure schema is up.
    init_schema()
    _cleanup_test_rows()

    docs = [
        "Azure cloud revenue accelerated 30% YoY this quarter.",
        "Office 365 commercial seats grew 14% YoY.",
        "Activision Blizzard integration is on track with margin lift.",
        "GitHub Copilot paid seats exceeded 1.3 million.",
        "LinkedIn Marketing Solutions saw weakness in EMEA.",
        "Windows OEM revenue declined 8% on weak PC demand.",
        "Surface devices revenue fell 21% YoY.",
        "Server products and cloud services rose 22%.",
        "Operating margin expanded 120 basis points.",
        "Free cash flow conversion remained above 90 percent.",
    ]

    doc_id = DOC_PREFIX + "STORE_BATCH_1"
    embs = embedder.embed_batch(docs)
    inserted_ids = []
    for i, (text, vec) in enumerate(zip(docs, embs)):
        cid = store.insert_chunk(
            {
                "doc_id": doc_id,
                "ticker": "MSFT",
                "source_tool": "test_rag_foundation",
                "doc_type": "TEST",
                "chunk_text": text,
                "chunk_sequence": i,
                "chunk_offset": i * 100,
            },
            vec,
        )
        inserted_ids.append(cid)

    _check("insert_chunk returns integer ids", all(isinstance(i, int) for i in inserted_ids))
    _check("count_chunks(doc_id) == 10",
           store.count_chunks({"doc_id": doc_id}) == 10,
           f"got {store.count_chunks({'doc_id': doc_id})}")

    # Vector search — exact text of chunk[3] should retrieve chunk[3] top.
    query = embedder.embed(docs[3])
    results = store.vector_search(query, top_k=5, ticker_filter="MSFT")
    _check("vector_search returns up to top_k", len(results) <= 5, f"got {len(results)}")
    _check("vector_search returns results", len(results) > 0)
    _check(
        "top result for exact chunk[3] text is chunk[3]",
        results and results[0]["chunk_text"] == docs[3],
        f"top={results[0]['chunk_text'] if results else 'NONE'}",
    )
    _check(
        "similarity score in [0, 1]",
        results and 0.0 <= results[0]["similarity"] <= 1.0,
        f"sim={results[0]['similarity'] if results else 'NONE'}",
    )

    # Ticker filter — insert a non-MSFT doc and verify it's filtered.
    doc_id_other = DOC_PREFIX + "STORE_NVDA_1"
    other_text = "NVIDIA Hopper GPU shipments accelerated through hyperscaler demand."
    cid_other = store.insert_chunk(
        {
            "doc_id": doc_id_other,
            "ticker": "NVDA",
            "source_tool": "test_rag_foundation",
            "doc_type": "TEST",
            "chunk_text": other_text,
            "chunk_sequence": 0,
            "chunk_offset": 0,
        },
        embedder.embed(other_text),
    )
    _check("inserted NVDA chunk", isinstance(cid_other, int))

    msft_only = store.vector_search(embedder.embed("GPU"), top_k=20,
                                     ticker_filter="MSFT")
    all_msft = all(r["ticker"] == "MSFT" for r in msft_only)
    _check("ticker_filter='MSFT' returns only MSFT rows", all_msft,
           f"got tickers {[r['ticker'] for r in msft_only]}")

    nvda_only = store.vector_search(embedder.embed("GPU"), top_k=20,
                                     ticker_filter="NVDA")
    all_nvda = all(r["ticker"] == "NVDA" for r in nvda_only)
    _check("ticker_filter='NVDA' returns only NVDA rows", all_nvda,
           f"got tickers {[r['ticker'] for r in nvda_only]}")

    # doc_type filter
    typed = store.vector_search(embedder.embed("revenue"), top_k=5,
                                 doc_type_filter="TEST")
    _check("doc_type_filter applies",
           all(r["doc_type"] == "TEST" for r in typed) and len(typed) > 0,
           "doc_type filter failed")

    # delete_by_doc_id removes chunks AND vec rows
    n_before = store.count_chunks()
    deleted = store.delete_by_doc_id(doc_id)
    _check("delete_by_doc_id returns 10", deleted == 10, f"got {deleted}")
    _check("count_chunks(doc_id) == 0 after delete",
           store.count_chunks({"doc_id": doc_id}) == 0)
    n_after = store.count_chunks()
    _check("total count decreased by 10", n_before - n_after == 10,
           f"before={n_before} after={n_after}")

    # Confirm vec0 rows for those ids are gone too.
    conn = get_connection()
    try:
        # Issue a MATCH-based search constrained by the deleted ids.
        ids_csv = ",".join(str(i) for i in inserted_ids)
        rows = conn.execute(
            f"SELECT rowid FROM rag_chunk_embeddings WHERE rowid IN ({ids_csv})"
        ).fetchall()
    finally:
        conn.close()
    _check("vec0 embeddings removed for deleted chunks", len(rows) == 0,
           f"found {len(rows)} leftover vec rows")

    # Clean up the NVDA test row
    store.delete_by_doc_id(doc_id_other)

    # Schema idempotency — running init_schema again should not break anything
    init_schema()
    init_schema()
    _check("init_schema is idempotent (no exception)", True)


# ---------------------------------------------------------------------------
# 4. End-to-end mini
# ---------------------------------------------------------------------------

def test_e2e() -> None:
    _section("End-to-end mini")

    # Build a ~2000-char synthetic document with three distinct topics so
    # we know which chunk should win.
    paragraphs = [
        # Chunk 0 region
        "Microsoft reported a strong fiscal Q1 with revenue of $56.5 billion, "
        "up 13% year over year. Productivity and Business Processes segment "
        "grew 13% to $18.6 billion driven by Office 365 commercial. ",
        # Chunk 1 region — Azure heavy
        "Intelligent Cloud revenue was $24.3 billion, up 19% year over year. "
        "Server products and cloud services revenue increased 21% driven by "
        "Azure and other cloud services revenue growth of 29%. Azure remains "
        "the primary growth driver of the commercial business in fiscal 2024. "
        "Azure consumption was strong across financial services and healthcare. ",
        # Chunk 2 region — capital returns
        "More Personal Computing revenue was $13.7 billion. Windows OEM revenue "
        "decreased 4%. Surface revenue decreased 17% reflecting weak PC demand. "
        "The company returned $9.1 billion to shareholders in the quarter via "
        "share repurchases and dividends. Free cash flow was $20.7 billion. ",
    ]
    doc = "\n\n".join(paragraphs * 3)
    _check("synthetic doc length sane", 1500 <= len(doc) <= 5000, f"len={len(doc)}")

    chunks = chunker.chunk_text(doc, target_tokens=80, overlap_tokens=10,
                                 section_heading="Earnings Highlights")
    _check("e2e chunker produces multiple chunks", len(chunks) >= 2,
           f"got {len(chunks)}")

    doc_id = DOC_PREFIX + "E2E_DOC_1"
    embs = embedder.embed_batch([c["chunk_text"] for c in chunks])
    for i, (c, e) in enumerate(zip(chunks, embs)):
        store.insert_chunk(
            {
                "doc_id":          doc_id,
                "ticker":          "MSFT",
                "source_tool":     "test_rag_foundation",
                "doc_type":        "E2E",
                "section_heading": c["section_heading"],
                "chunk_text":      c["chunk_text"],
                "chunk_offset":    c["chunk_offset"],
                "chunk_sequence":  c["chunk_sequence"],
            },
            e,
        )

    # Query for Azure-specific content; chunk(s) about Azure should top.
    q_emb = embedder.embed(
        "Azure cloud services revenue growth in financial services and healthcare"
    )
    results = store.vector_search(q_emb, top_k=3, ticker_filter="MSFT",
                                  doc_type_filter="E2E")
    _check("e2e search returns >=1 result", len(results) > 0)
    top_text = results[0]["chunk_text"] if results else ""
    _check(
        "top e2e result mentions Azure",
        "Azure" in top_text,
        f"top text: {top_text[:80]!r}",
    )

    # Cleanup
    store.delete_by_doc_id(doc_id)
    _check("e2e doc cleaned up",
           store.count_chunks({"doc_id": doc_id}) == 0)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 60)
    print("RAG Foundation Test Suite")
    print("=" * 60)

    init_schema()

    try:
        test_chunker()
    except Exception:
        traceback.print_exc()
        _check("chunker suite ran without exception", False, "exception above")

    try:
        test_embedder()
    except Exception:
        traceback.print_exc()
        _check("embedder suite ran without exception", False, "exception above")

    try:
        test_store()
    except Exception:
        traceback.print_exc()
        _check("store suite ran without exception", False, "exception above")

    try:
        test_e2e()
    except Exception:
        traceback.print_exc()
        _check("e2e suite ran without exception", False, "exception above")

    # Final cleanup just in case.
    _cleanup_test_rows()

    print("\n" + "=" * 60)
    print(f"PASS {_PASS} FAIL {_FAIL}")
    if _FAILURES:
        print("\nFailures:")
        for f in _FAILURES:
            print(f)
    print("=" * 60)
    return 0 if _FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
