"""Bootstrap the initial RAG corpus from on-disk knowledge artifacts.

Loads four buckets of pre-existing content into the rag_chunks vector store
so the rest of the system has a usable retrieval surface from the moment
the database initializes:

  1. knowledge/analogues.md    — one doc per "## N. Title" section.
  2. testing/fixtures/phase_B3_*_log.md — MSFT analyst write-ups (3 files).
  3. testing/output/phase_B3_pilot.json — structured extractor bundle; each
     call with text-bearing output is routed through ingest_extractor_output.
  4. rules/*.md                — one doc per rule file.

The script is idempotent: every ingest path uses a deterministic doc_id
and `delete_by_doc_id` is called inside `ingest_document` before insertion,
so re-running produces the same chunk count instead of duplicating.

Run from project root:
    ./.venv/Scripts/python.exe scripts/bootstrap_rag_corpus.py
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from typing import Any, Dict, List, Tuple

# Make project root importable when invoked as a script.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agent.rag.ingest import ingest_document, ingest_extractor_output  # noqa: E402
from agent.rag import store  # noqa: E402
from state.schema import init_schema  # noqa: E402


# Tools in phase_B3_pilot.json whose results carry text we want to ingest.
# (extract_8k_events is NOT in the four supported adapters and is excluded.)
TEXT_BEARING_TOOLS = {
    "extract_risk_factors",
    "extract_mda",
    "get_earnings_releases",
    "get_supply_chain",
}


# ---------------------------------------------------------------------------
# 1. knowledge/analogues.md
# ---------------------------------------------------------------------------

ANALOGUE_HEADING_RE = re.compile(r"^##\s+(\d+)\.\s+(.+?)\s*$", re.MULTILINE)


def _slugify(text: str, max_len: int = 60) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", text or "").strip("_").lower()
    return (s[:max_len] or "untitled")


def _parse_analogues(path: str) -> List[Tuple[str, str]]:
    """Return [(title_slug, body_text)] for each `## N. Title` block.

    Body includes the heading line itself so chunk text retains the title
    even when the section is short enough to be a single chunk.
    """
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    matches = list(ANALOGUE_HEADING_RE.finditer(text))
    if not matches:
        return []

    sections: List[Tuple[str, str]] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        title = m.group(2).strip()
        slug = _slugify(title)
        sections.append((slug, body))
    return sections


def ingest_analogues(path: str) -> Dict[str, int]:
    sections = _parse_analogues(path)
    summary = {"file": path, "docs": 0, "chunks": 0}
    for slug, body in sections:
        doc_id = f"analogue_{slug}"
        meta = {
            "source_tool":     "bootstrap_rag_corpus",
            "doc_type":        "analogue",
            "section_heading": slug.replace("_", " ").title(),
        }
        # Small chunks per analogue — sections are short (~1KB each) but
        # dense; fine-grained chunks let retrieval pinpoint the specific
        # element (setup, what worked, lesson, modern rhyme triggers).
        res = ingest_document(body, meta, doc_id=doc_id,
                              target_tokens=80, overlap_tokens=15)
        summary["docs"] += 1
        summary["chunks"] += res["chunks_inserted"]
    print(f"  analogues: {os.path.basename(path)} -> {summary['docs']} docs, {summary['chunks']} chunks")
    return summary


# ---------------------------------------------------------------------------
# 2. testing/fixtures/phase_B3_*_log.md
# ---------------------------------------------------------------------------

def ingest_writeups(paths: List[str]) -> Dict[str, int]:
    summary = {"docs": 0, "chunks": 0}
    for p in paths:
        if not os.path.exists(p):
            print(f"  writeups: SKIP missing {p}")
            continue
        with open(p, "r", encoding="utf-8") as f:
            text = f.read()
        stem = os.path.splitext(os.path.basename(p))[0]
        doc_id = f"writeup_{stem}"
        meta = {
            "ticker":      "MSFT",
            "source_tool": "bootstrap_rag_corpus",
            "doc_type":    "analyst_writeup",
            "section_heading": stem.replace("_", " ").title(),
        }
        # Writeups are long-form analyses — fine-grained chunks make per-
        # paragraph retrieval (e.g. "what was the variant perception?")
        # tractable instead of returning a whole chapter at a time.
        res = ingest_document(text, meta, doc_id=doc_id,
                              target_tokens=120, overlap_tokens=20)
        summary["docs"] += 1
        summary["chunks"] += res["chunks_inserted"]
        print(f"  writeups: {os.path.basename(p)} -> 1 doc, {res['chunks_inserted']} chunks")
    return summary


# ---------------------------------------------------------------------------
# 3. testing/output/phase_B3_pilot.json — extractor bundle
# ---------------------------------------------------------------------------

def ingest_pilot_bundle(path: str) -> Dict[str, int]:
    summary = {"docs": 0, "chunks": 0}
    if not os.path.exists(path):
        print(f"  pilot bundle: SKIP missing {path}")
        return summary

    try:
        with open(path, "r", encoding="utf-8") as f:
            bundle = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  pilot bundle: SKIP parse error {e}")
        return summary

    ticker_hint = bundle.get("ticker")

    # Iterate raw call log — only adapt calls whose tool we recognize.
    for call in bundle.get("calls", []):
        if not isinstance(call, dict):
            continue
        tool = call.get("tool")
        if tool not in TEXT_BEARING_TOOLS:
            continue
        result = call.get("result")
        if not isinstance(result, dict):
            continue
        res = ingest_extractor_output(tool, result, ticker_hint=ticker_hint)
        summary["docs"] += res["docs_ingested"]
        summary["chunks"] += res["total_chunks_inserted"]
        print(f"  pilot bundle: {tool} -> {res['docs_ingested']} docs, {res['total_chunks_inserted']} chunks")

    # Optional synthesis section — handle both common shapes.
    synthesis_text = None
    for key in ("synthesis", "synthesis_text", "verdict_writeup", "writeup"):
        v = bundle.get(key)
        if isinstance(v, str) and v.strip():
            synthesis_text = v
            break
    if synthesis_text:
        stem = os.path.splitext(os.path.basename(path))[0]
        doc_id = f"writeup_{stem}_synthesis"
        meta = {
            "ticker":      ticker_hint,
            "source_tool": "bootstrap_rag_corpus",
            "doc_type":    "analyst_writeup",
            "section_heading": "Phase B3 Pilot Synthesis",
        }
        res = ingest_document(synthesis_text, meta, doc_id=doc_id)
        summary["docs"] += 1
        summary["chunks"] += res["chunks_inserted"]
        print(f"  pilot bundle: synthesis -> 1 doc, {res['chunks_inserted']} chunks")

    print(f"  pilot bundle TOTAL: {summary['docs']} docs, {summary['chunks']} chunks")
    return summary


# ---------------------------------------------------------------------------
# 4. rules/*.md
# ---------------------------------------------------------------------------

def ingest_rules(dir_path: str) -> Dict[str, int]:
    summary = {"docs": 0, "chunks": 0}
    pattern = os.path.join(dir_path, "*.md")
    for p in sorted(glob.glob(pattern)):
        with open(p, "r", encoding="utf-8") as f:
            text = f.read()
        stem = os.path.splitext(os.path.basename(p))[0]
        doc_id = f"rule_{stem}"
        meta = {
            "source_tool": "bootstrap_rag_corpus",
            "doc_type":    "rule",
            "section_heading": stem.replace("_", " ").title(),
        }
        # Rules files are dense bullet lists — small chunks let retrieval
        # pinpoint the specific principle that matched a query.
        res = ingest_document(text, meta, doc_id=doc_id,
                              target_tokens=120, overlap_tokens=20)
        summary["docs"] += 1
        summary["chunks"] += res["chunks_inserted"]
        print(f"  rules: {os.path.basename(p)} -> 1 doc, {res['chunks_inserted']} chunks")
    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    init_schema()

    print("=" * 60)
    print("Bootstrapping RAG corpus")
    print("=" * 60)

    grand_docs = 0
    grand_chunks = 0

    print("\n[1/4] analogues")
    s1 = ingest_analogues(os.path.join(PROJECT_ROOT, "knowledge", "analogues.md"))
    grand_docs += s1["docs"]
    grand_chunks += s1["chunks"]

    print("\n[2/4] analyst writeups")
    writeup_paths = [
        os.path.join(PROJECT_ROOT, "testing", "fixtures", "phase_B3_pilot_log.md"),
        os.path.join(PROJECT_ROOT, "testing", "fixtures", "phase_B3_rerun_log.md"),
        os.path.join(PROJECT_ROOT, "testing", "fixtures", "phase_B3_direct_log.md"),
    ]
    s2 = ingest_writeups(writeup_paths)
    grand_docs += s2["docs"]
    grand_chunks += s2["chunks"]

    print("\n[3/4] phase B3 pilot bundle")
    s3 = ingest_pilot_bundle(
        os.path.join(PROJECT_ROOT, "testing", "output", "phase_B3_pilot.json")
    )
    grand_docs += s3["docs"]
    grand_chunks += s3["chunks"]

    print("\n[4/4] rules")
    s4 = ingest_rules(os.path.join(PROJECT_ROOT, "rules"))
    grand_docs += s4["docs"]
    grand_chunks += s4["chunks"]

    total_in_db = store.count_chunks()
    print("\n" + "=" * 60)
    print(f"TOTAL: {grand_docs} docs ingested, {grand_chunks} chunks written")
    print(f"rag_chunks rows in store: {total_in_db}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
