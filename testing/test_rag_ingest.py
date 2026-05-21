"""Stress-test the RAG ingestion layer (`agent.rag.ingest`) and the
bootstrap corpus script.

Run from project root:
    ./.venv/Scripts/python.exe testing/test_rag_ingest.py

Mirrors the harness style of `test_rag_foundation.py` — each assertion goes
through `_check(name, condition, hint)` and the final line is
`PASS N FAIL M`. Synthetic test docs use the `INGEST_TEST_` prefix so they
can be cleaned up without disturbing bootstrap-generated rows (which are
expected to persist across runs).
"""
from __future__ import annotations

import os
import subprocess
import sys
import traceback
from typing import Any, Dict, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agent.rag import ingest, store  # noqa: E402
from state.schema import get_connection, init_schema  # noqa: E402


# ---------------------------------------------------------------------------
# Harness
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


DOC_PREFIX = "INGEST_TEST_"


def _cleanup_test_rows() -> None:
    """Remove any chunks whose doc_id starts with INGEST_TEST_."""
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
# 1. ingest_document basics
# ---------------------------------------------------------------------------

def test_ingest_document_basics() -> None:
    _section("ingest_document basics")

    # Empty text
    res = ingest.ingest_document(
        "",
        {"ticker": "MSFT", "doc_type": "TEST"},
        doc_id=DOC_PREFIX + "EMPTY",
    )
    _check("empty text -> 0 chunks", res["chunks_inserted"] == 0,
           f"got {res['chunks_inserted']}")
    _check("empty returns a doc_id", isinstance(res.get("doc_id"), str) and bool(res["doc_id"]))
    _check("empty does not raise", True)

    # Whitespace-only
    res = ingest.ingest_document(
        "   \n\n  ",
        {"ticker": "MSFT", "doc_type": "TEST"},
        doc_id=DOC_PREFIX + "WHITESPACE",
    )
    _check("whitespace-only -> 0 chunks", res["chunks_inserted"] == 0)

    # Short text -> 1 chunk
    short_text = (
        "Microsoft reported Azure cloud growth of 30 percent year over year. "
        "Operating margin expanded. Free cash flow remained strong."
    )
    doc_id_short = DOC_PREFIX + "SHORT_1"
    res = ingest.ingest_document(
        short_text,
        {"ticker": "MSFT", "doc_type": "TEST", "source_tool": "test_rag_ingest"},
        doc_id=doc_id_short,
    )
    _check("short text -> 1 chunk", res["chunks_inserted"] == 1,
           f"got {res['chunks_inserted']}")
    _check("short returns ticker", res["ticker"] == "MSFT")
    _check("short returns doc_type", res["doc_type"] == "TEST")
    _check(
        "short inserted into store",
        store.count_chunks({"doc_id": doc_id_short}) == 1,
        f"count={store.count_chunks({'doc_id': doc_id_short})}",
    )

    # Re-ingesting same content + metadata -> same doc_id, no duplicate
    res2 = ingest.ingest_document(
        short_text,
        {"ticker": "MSFT", "doc_type": "TEST", "source_tool": "test_rag_ingest"},
        doc_id=doc_id_short,
    )
    _check("re-ingest same doc_id -> still 1 chunk", res2["chunks_inserted"] == 1)
    _check(
        "no duplicate row after re-ingest",
        store.count_chunks({"doc_id": doc_id_short}) == 1,
        f"count={store.count_chunks({'doc_id': doc_id_short})}",
    )

    # Auto doc_id is stable: same input -> same id
    auto1 = ingest.ingest_document(
        "stable-content-A",
        {"ticker": "FAKE", "doc_type": "TEST", "source_tool": "test_rag_ingest",
         "_test_marker": DOC_PREFIX + "AUTO_A"},
    )
    auto2 = ingest.ingest_document(
        "stable-content-A",
        {"ticker": "FAKE", "doc_type": "TEST", "source_tool": "test_rag_ingest",
         "_test_marker": DOC_PREFIX + "AUTO_A"},
    )
    _check("auto doc_id stable across runs", auto1["doc_id"] == auto2["doc_id"],
           f"{auto1['doc_id']} vs {auto2['doc_id']}")
    # Cleanup the auto-id rows we just made.
    store.delete_by_doc_id(auto1["doc_id"])

    # Different content -> different doc_id
    autoX = ingest.ingest_document(
        "stable-content-A",
        {"ticker": "FAKE", "doc_type": "TEST", "source_tool": "test_rag_ingest",
         "_test_marker": DOC_PREFIX + "AUTO_X"},
    )
    autoY = ingest.ingest_document(
        "stable-content-B",
        {"ticker": "FAKE", "doc_type": "TEST", "source_tool": "test_rag_ingest",
         "_test_marker": DOC_PREFIX + "AUTO_Y"},
    )
    _check("different content -> different doc_id", autoX["doc_id"] != autoY["doc_id"],
           f"{autoX['doc_id']} == {autoY['doc_id']}")
    store.delete_by_doc_id(autoX["doc_id"])
    store.delete_by_doc_id(autoY["doc_id"])


# ---------------------------------------------------------------------------
# 2. ingest_extractor_output adapters
# ---------------------------------------------------------------------------

def _long_paragraph(seed: str) -> str:
    """Build a multi-paragraph blob long enough for the chunker to emit a
    chunk for each section but not so long that embedding takes minutes.
    """
    para = (
        f"{seed} describes operational dynamics across global supply chains, "
        "competitive positioning, and execution risk in the current cycle. "
        "Margin expansion has been driven by mix shift and pricing discipline "
        "while capital intensity remains within historical bands. "
    ) * 4
    return "\n\n".join([para] * 3)


def test_extract_risk_factors_adapter() -> None:
    _section("ingest_extractor_output: extract_risk_factors")

    # Build a synthetic Item 1A result with three section headings whose
    # offsets carve the text into three sub-documents.
    sec_a = _long_paragraph("SECTION_ALPHA")
    sec_b = _long_paragraph("SECTION_BETA")
    sec_c = _long_paragraph("SECTION_GAMMA")
    h_a = "OPERATIONAL RISKS"
    h_b = "CYBERSECURITY RISKS"
    h_c = "REGULATORY RISKS"
    text = (
        h_a + "\n\n" + sec_a + "\n\n"
        + h_b + "\n\n" + sec_b + "\n\n"
        + h_c + "\n\n" + sec_c
    )
    off_a = 0
    off_b = text.find(h_b)
    off_c = text.find(h_c)
    result = {
        "ticker":               "INGTST",
        "success":              True,
        "item":                 "1A",
        "text":                 text,
        "filing_date":          "2025-01-15",
        "filing_url":           "https://example.invalid/rf",
        "section_headings": [
            {"heading": h_a, "offset_in_section": off_a},
            {"heading": h_b, "offset_in_section": off_b},
            {"heading": h_c, "offset_in_section": off_c},
        ],
    }
    out = ingest.ingest_extractor_output("extract_risk_factors", result)

    _check("risk_factors: tool_name preserved", out["tool_name"] == "extract_risk_factors")
    _check("risk_factors: 3 docs ingested", out["docs_ingested"] == 3,
           f"got {out['docs_ingested']}")
    _check("risk_factors: chunks > 0", out["total_chunks_inserted"] > 0)

    # Verify each heading appears as section_heading on a stored chunk.
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT DISTINCT section_heading FROM rag_chunks "
            "WHERE ticker = ? AND doc_type = '10K_risk_factors'",
            ("INGTST",),
        ).fetchall()
        headings = {r["section_heading"] for r in rows}
    finally:
        conn.close()
    for h in (h_a, h_b, h_c):
        _check(f"risk_factors: heading '{h}' present", h in headings,
               f"got {headings}")

    # Cleanup the three docs.
    for h in (h_a, h_b, h_c):
        slug = "".join(c if c.isalnum() else "_" for c in h).strip("_").lower()[:50]
        # doc_id pattern: {ticker}_{doc_type}_{date_slug}_{heading_slug}
        doc_id = f"INGTST_10K_risk_factors_2025_01_15_{slug}"
        store.delete_by_doc_id(doc_id)


def test_extract_mda_adapter() -> None:
    _section("ingest_extractor_output: extract_mda")

    text = (
        "EXECUTIVE SUMMARY\n\n" + _long_paragraph("MDA_SUMMARY") + "\n\n"
        "LIQUIDITY\n\n" + _long_paragraph("MDA_LIQ")
    )
    off1 = 0
    off2 = text.find("LIQUIDITY")
    result = {
        "ticker":          "MDATEST",
        "success":         True,
        "item":            "7",
        "text":            text,
        "filing_date":     "2025-03-01",
        "section_headings": [
            {"heading": "EXECUTIVE SUMMARY", "offset_in_section": off1},
            {"heading": "LIQUIDITY", "offset_in_section": off2},
        ],
    }
    out = ingest.ingest_extractor_output("extract_mda", result)
    _check("mda: 2 docs ingested", out["docs_ingested"] == 2,
           f"got {out['docs_ingested']}")
    _check("mda: chunks > 0", out["total_chunks_inserted"] > 0)

    # Verify doc_type tagged correctly
    conn = get_connection()
    try:
        n = conn.execute(
            "SELECT COUNT(*) AS n FROM rag_chunks "
            "WHERE ticker = ? AND doc_type = '10K_mda'",
            ("MDATEST",),
        ).fetchone()["n"]
    finally:
        conn.close()
    _check("mda: chunks tagged as 10K_mda", n > 0, f"n={n}")

    for h in ("executive_summary", "liquidity"):
        store.delete_by_doc_id(f"MDATEST_10K_mda_2025_03_01_{h}")


def test_earnings_releases_adapter() -> None:
    _section("ingest_extractor_output: get_earnings_releases")

    releases: List[Dict[str, Any]] = []
    accessions = []
    for i in range(4):
        acc = f"0000{i}-25-{1000 + i}"
        accessions.append(acc)
        releases.append({
            "filing_date":      f"2025-0{i+1}-15",
            "accession_number": acc,
            "items":            ["2.02"],
            "text":             _long_paragraph(f"EARNINGS_Q{i+1}"),
        })
    result = {
        "ticker":   "EARNTST",
        "success":  True,
        "releases": releases,
    }
    out = ingest.ingest_extractor_output("get_earnings_releases", result)
    _check("earnings: 4 docs ingested", out["docs_ingested"] == 4,
           f"got {out['docs_ingested']}")
    _check("earnings: chunks > 0", out["total_chunks_inserted"] > 0)

    # Each accession should have its own doc_id row(s)
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT DISTINCT doc_id FROM rag_chunks "
            "WHERE ticker = ? AND doc_type = 'earnings_release'",
            ("EARNTST",),
        ).fetchall()
        doc_ids = {r["doc_id"] for r in rows}
    finally:
        conn.close()
    _check("earnings: 4 distinct doc_ids", len(doc_ids) == 4, f"got {doc_ids}")
    for acc in accessions:
        slug = "".join(c if c.isalnum() else "_" for c in acc).strip("_").lower()[:40]
        _check(
            f"earnings: doc_id for accession {acc}",
            any(slug in d for d in doc_ids),
            f"missing slug {slug} in {doc_ids}",
        )

    for d in doc_ids:
        store.delete_by_doc_id(d)


def test_supply_chain_adapter() -> None:
    _section("ingest_extractor_output: get_supply_chain")

    triggers = []
    for i in range(5):
        triggers.append({
            "trigger":  "compete with",
            "sentence": (
                f"In the {i}th product category we compete with established "
                f"vendors including FooCorp and BarCorp across several "
                f"verticals where pricing and feature differentiation are "
                f"the primary competitive levers driving customer choice."
            ),
        })
    result = {
        "ticker":            "SCHTEST",
        "success":           True,
        "filing_date":       "2025-04-01",
        "trigger_sentences": triggers,
    }
    out = ingest.ingest_extractor_output("get_supply_chain", result)
    _check("supply_chain: 1 doc ingested", out["docs_ingested"] == 1,
           f"got {out['docs_ingested']}")
    _check("supply_chain: chunks >= 1", out["total_chunks_inserted"] >= 1,
           f"got {out['total_chunks_inserted']}")

    doc_id = "SCHTEST_supply_chain_2025_04_01"
    n = store.count_chunks({"doc_id": doc_id})
    _check("supply_chain: doc_id derived correctly", n >= 1, f"n={n}")
    store.delete_by_doc_id(doc_id)


def test_failed_extractor() -> None:
    _section("ingest_extractor_output: failed / empty results")

    # success=False -> 0 docs, no exception
    out = ingest.ingest_extractor_output(
        "extract_risk_factors",
        {"ticker": "FAILTST", "success": False, "error": "no filing"},
    )
    _check("failed result -> 0 docs", out["docs_ingested"] == 0)
    _check("failed result -> 0 chunks", out["total_chunks_inserted"] == 0)

    # Empty releases list
    out = ingest.ingest_extractor_output(
        "get_earnings_releases",
        {"ticker": "EMPTYTST", "success": True, "releases": []},
    )
    _check("empty releases -> 0 docs", out["docs_ingested"] == 0)

    # Non-dict input
    out = ingest.ingest_extractor_output("extract_mda", None)
    _check("None input -> 0 docs, no crash", out["docs_ingested"] == 0)

    # Unknown tool
    out = ingest.ingest_extractor_output("unknown_tool", {"success": True})
    _check("unknown tool -> 0 docs", out["docs_ingested"] == 0)


# ---------------------------------------------------------------------------
# 3. Bootstrap script smoke + idempotency
# ---------------------------------------------------------------------------

def _python_exe() -> str:
    """Prefer the project's venv python if present; else current interpreter."""
    venv_py = os.path.join(PROJECT_ROOT, ".venv", "Scripts", "python.exe")
    if os.path.exists(venv_py):
        return venv_py
    venv_py2 = os.path.join(PROJECT_ROOT, "venv", "Scripts", "python.exe")
    if os.path.exists(venv_py2):
        return venv_py2
    return sys.executable


def _run_bootstrap() -> Dict[str, Any]:
    script = os.path.join(PROJECT_ROOT, "scripts", "bootstrap_rag_corpus.py")
    proc = subprocess.run(
        [_python_exe(), script],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    return {
        "returncode": proc.returncode,
        "stdout":     proc.stdout,
        "stderr":     proc.stderr,
    }


def test_bootstrap_smoke() -> None:
    _section("Bootstrap script smoke")

    res = _run_bootstrap()
    _check("bootstrap exits 0", res["returncode"] == 0,
           f"stderr: {res['stderr'][:300]}")
    if res["returncode"] != 0:
        print("---- bootstrap stdout ----")
        print(res["stdout"][-2000:])
        print("---- bootstrap stderr ----")
        print(res["stderr"][-2000:])

    total = store.count_chunks()
    _check("total chunks > 100", total > 100, f"total={total}")
    print(f"  (info) total rag_chunks rows: {total}")

    # Verify multiple doc_types present
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT DISTINCT doc_type FROM rag_chunks WHERE doc_type IS NOT NULL"
        ).fetchall()
        doc_types = {r["doc_type"] for r in rows}
    finally:
        conn.close()
    print(f"  (info) doc_types present: {sorted(doc_types)}")

    for expected in ("analogue", "analyst_writeup", "rule"):
        _check(f"doc_type '{expected}' present", expected in doc_types,
               f"got {doc_types}")

    # At least one ticker-tagged ticker should exist (MSFT writeups).
    msft_count = store.count_chunks({"ticker": "MSFT"})
    _check("MSFT chunks > 20", msft_count > 20, f"msft_count={msft_count}")
    print(f"  (info) MSFT chunks: {msft_count}")


def test_bootstrap_idempotency() -> None:
    _section("Bootstrap idempotency")

    n_before = store.count_chunks()
    res = _run_bootstrap()
    _check("second bootstrap exits 0", res["returncode"] == 0,
           f"stderr: {res['stderr'][:300]}")
    n_after = store.count_chunks()
    _check(
        "chunk count unchanged after second run",
        n_before == n_after,
        f"before={n_before} after={n_after}",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 60)
    print("RAG Ingestion Test Suite")
    print("=" * 60)

    init_schema()
    _cleanup_test_rows()

    suites = [
        ("ingest_document basics",       test_ingest_document_basics),
        ("extract_risk_factors adapter", test_extract_risk_factors_adapter),
        ("extract_mda adapter",          test_extract_mda_adapter),
        ("earnings_releases adapter",    test_earnings_releases_adapter),
        ("supply_chain adapter",         test_supply_chain_adapter),
        ("failed extractor",             test_failed_extractor),
        ("bootstrap smoke",              test_bootstrap_smoke),
        ("bootstrap idempotency",        test_bootstrap_idempotency),
    ]
    for name, fn in suites:
        try:
            fn()
        except Exception:
            traceback.print_exc()
            _check(f"{name} ran without exception", False, "exception above")

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
