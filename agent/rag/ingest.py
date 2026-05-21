"""High-level ingestion layer that feeds documents into the RAG store.

`agent.rag.chunker`, `agent.rag.embedder`, and `agent.rag.store` already
provide the low-level primitives. This module wires them together into two
entry points:

  ingest_document(text, metadata, doc_id=None) -> dict
      Chunk arbitrary text, embed each chunk in a batch, insert into the
      vector store under a stable doc_id. Idempotent — calling with the
      same text + metadata produces the same doc_id, and prior chunks are
      cleared before re-insertion.

  ingest_extractor_output(tool_name, result_dict, ticker_hint=None) -> dict
      Adapter that knows the result shapes of the four SEC extractor tools
      (extract_risk_factors, extract_mda, get_earnings_releases,
      get_supply_chain) and converts each into one or more ingest_document
      calls with sensible doc_id / doc_type / section metadata.

The module never mutates inputs and never touches the network — embedding
is local via sentence-transformers, storage is the local sqlite db.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Optional

from agent.rag import chunker, embedder, store


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stable_doc_id(text: str, metadata: Dict[str, Any]) -> str:
    """Deterministic doc_id derived from text + metadata.

    24-char prefix of sha256 keeps the id short while preserving enough
    bits to avoid collisions across our corpus (~10^5 docs).
    """
    payload = (text or "") + "||" + repr(sorted((metadata or {}).items()))
    return hashlib.sha256(payload.encode("utf-8", errors="replace")).hexdigest()[:24]


def _slugify(value: Optional[str], max_len: int = 60) -> str:
    """Cheap slug: lowercase, non-alnum -> underscore, collapse runs."""
    if not value:
        return "unknown"
    s = re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_").lower()
    return (s[:max_len] or "unknown")


# ---------------------------------------------------------------------------
# Core: ingest_document
# ---------------------------------------------------------------------------

def ingest_document(
    text: str,
    metadata: Dict[str, Any],
    doc_id: Optional[str] = None,
    target_tokens: int = 500,
    overlap_tokens: int = 50,
) -> Dict[str, Any]:
    """Chunk + embed + store `text` under a stable doc_id.

    Args:
      text: source text. Empty / whitespace-only input is a no-op.
      metadata: dict that may contain ticker, source_tool, doc_type,
        filing_date, item_number, section_heading. Unknown keys are
        ignored by the store.
      doc_id: optional explicit doc_id. When None, an idempotent id is
        generated from (text, metadata).
      target_tokens: chunker target size. Default 500 matches MiniLM's
        comfortable context window. Smaller values produce denser corpora
        for dense source documents like analyst write-ups.
      overlap_tokens: chunker overlap. Default 50.

    Returns dict with doc_id, chunks_inserted, ticker, doc_type.
    """
    metadata = metadata or {}
    if doc_id is None:
        doc_id = _stable_doc_id(text or "", metadata)

    # Clear any prior chunks for this doc_id so re-ingest is a clean replace.
    store.delete_by_doc_id(doc_id)

    result = {
        "doc_id":          doc_id,
        "chunks_inserted": 0,
        "ticker":          metadata.get("ticker"),
        "doc_type":        metadata.get("doc_type"),
    }

    if not text or not text.strip():
        return result

    chunks = chunker.chunk_text(
        text,
        target_tokens=target_tokens,
        overlap_tokens=overlap_tokens,
        section_heading=metadata.get("section_heading"),
    )
    if not chunks:
        return result

    # Batch-embed all chunks in one model call — this is the bottleneck
    # so batching matters more than any other optimization here.
    embeddings = embedder.embed_batch([c["chunk_text"] for c in chunks])

    inserted = 0
    for chunk, vec in zip(chunks, embeddings):
        store.insert_chunk(
            {
                "doc_id":          doc_id,
                "ticker":          metadata.get("ticker"),
                "source_tool":     metadata.get("source_tool"),
                "doc_type":        metadata.get("doc_type"),
                "filing_date":     metadata.get("filing_date"),
                "item_number":     metadata.get("item_number"),
                "section_heading": chunk.get("section_heading") or metadata.get("section_heading"),
                "chunk_text":      chunk["chunk_text"],
                "chunk_offset":    chunk.get("chunk_offset"),
                "chunk_sequence":  chunk.get("chunk_sequence"),
            },
            vec,
        )
        inserted += 1

    result["chunks_inserted"] = inserted
    return result


# ---------------------------------------------------------------------------
# Adapters per extractor tool
# ---------------------------------------------------------------------------

def _split_by_section_headings(text: str, headings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Slice `text` into sub-documents by the offsets in `headings`.

    Each heading dict has keys 'heading' and 'offset_in_section' (matches
    sec_utils.extract_{risk_factors,mda}).

    Returns a list of {heading, text, offset} dicts in document order.
    If no headings or text doesn't span them, returns a single-section
    fallback covering the whole text.
    """
    if not text:
        return []
    if not headings:
        return [{"heading": None, "text": text, "offset": 0}]

    # Sort by offset just in case the producer didn't.
    sorted_h = sorted(
        [h for h in headings if isinstance(h.get("offset_in_section"), int)],
        key=lambda h: h["offset_in_section"],
    )
    if not sorted_h:
        return [{"heading": None, "text": text, "offset": 0}]

    sections: List[Dict[str, Any]] = []
    # Preamble (text before first heading) gets its own section so we don't
    # silently drop the lead-in paragraphs.
    first_off = sorted_h[0]["offset_in_section"]
    if first_off > 0:
        preamble = text[:first_off]
        if preamble.strip():
            sections.append({"heading": None, "text": preamble, "offset": 0})

    for i, h in enumerate(sorted_h):
        start = h["offset_in_section"]
        end = sorted_h[i + 1]["offset_in_section"] if i + 1 < len(sorted_h) else len(text)
        body = text[start:end]
        if body.strip():
            sections.append({
                "heading": h.get("heading"),
                "text":    body,
                "offset":  start,
            })
    return sections


def _ingest_risk_or_mda(
    tool_name: str,
    result_dict: Dict[str, Any],
    doc_type: str,
    ticker_hint: Optional[str],
) -> Dict[str, Any]:
    """Common path for extract_risk_factors and extract_mda — both return
    text + section_headings + filing metadata in the same shape.
    """
    ticker = (result_dict.get("ticker") or ticker_hint or "UNKNOWN").upper()
    text = result_dict.get("text") or ""
    filing_date = result_dict.get("filing_date")
    item_number = result_dict.get("item")
    headings = result_dict.get("section_headings") or []

    out = {
        "tool_name":              tool_name,
        "ticker":                 ticker,
        "docs_ingested":          0,
        "total_chunks_inserted":  0,
    }
    if not text.strip():
        return out

    sections = _split_by_section_headings(text, headings)
    fd_slug = _slugify(filing_date, 12)

    for sec in sections:
        heading = sec.get("heading")
        heading_slug = _slugify(heading or "preamble", 50)
        doc_id = f"{ticker}_{doc_type}_{fd_slug}_{heading_slug}"
        meta = {
            "ticker":          ticker,
            "source_tool":     tool_name,
            "doc_type":        doc_type,
            "filing_date":     filing_date,
            "item_number":     item_number,
            "section_heading": heading,
        }
        res = ingest_document(sec["text"], meta, doc_id=doc_id)
        out["docs_ingested"] += 1
        out["total_chunks_inserted"] += res["chunks_inserted"]
    return out


def _ingest_earnings_releases(
    result_dict: Dict[str, Any],
    ticker_hint: Optional[str],
) -> Dict[str, Any]:
    ticker = (result_dict.get("ticker") or ticker_hint or "UNKNOWN").upper()
    releases = result_dict.get("releases") or []
    out = {
        "tool_name":             "get_earnings_releases",
        "ticker":                ticker,
        "docs_ingested":         0,
        "total_chunks_inserted": 0,
    }
    for rel in releases:
        if not isinstance(rel, dict):
            continue
        text = rel.get("text") or ""
        if not text.strip():
            continue
        accession = rel.get("accession_number") or rel.get("filing_date") or "unknown"
        accession_slug = _slugify(accession, 40)
        doc_id = f"{ticker}_earnings_{accession_slug}"
        meta = {
            "ticker":      ticker,
            "source_tool": "get_earnings_releases",
            "doc_type":    "earnings_release",
            "filing_date": rel.get("filing_date"),
            "item_number": None,
            "section_heading": f"Earnings Release {rel.get('filing_date') or ''}".strip(),
        }
        res = ingest_document(text, meta, doc_id=doc_id)
        out["docs_ingested"] += 1
        out["total_chunks_inserted"] += res["chunks_inserted"]
    return out


def _ingest_supply_chain(
    result_dict: Dict[str, Any],
    ticker_hint: Optional[str],
) -> Dict[str, Any]:
    ticker = (result_dict.get("ticker") or ticker_hint or "UNKNOWN").upper()
    triggers = result_dict.get("trigger_sentences") or []
    filing_date = result_dict.get("filing_date")
    out = {
        "tool_name":             "get_supply_chain",
        "ticker":                ticker,
        "docs_ingested":         0,
        "total_chunks_inserted": 0,
    }
    if not triggers:
        return out

    # Join triggers into a single text body, one labeled sentence per line
    # block. Paragraphs split by blank lines so the chunker can size them.
    lines: List[str] = []
    for t in triggers:
        if not isinstance(t, dict):
            continue
        sentence = (t.get("sentence") or "").strip()
        if not sentence:
            continue
        label = (t.get("trigger") or "trigger").strip()
        lines.append(f"[{label}] {sentence}")
    if not lines:
        return out

    body = "\n\n".join(lines)
    fd_slug = _slugify(filing_date, 12)
    doc_id = f"{ticker}_supply_chain_{fd_slug}"
    meta = {
        "ticker":      ticker,
        "source_tool": "get_supply_chain",
        "doc_type":    "supply_chain_signals",
        "filing_date": filing_date,
        "section_heading": "Supply Chain Signals",
    }
    res = ingest_document(body, meta, doc_id=doc_id)
    out["docs_ingested"] += 1
    out["total_chunks_inserted"] += res["chunks_inserted"]
    return out


def ingest_extractor_output(
    tool_name: str,
    result_dict: Dict[str, Any],
    ticker_hint: Optional[str] = None,
) -> Dict[str, Any]:
    """Route a raw extractor result through the appropriate ingestion shape.

    Supported tool_name values:
      - extract_risk_factors  -> per-section sub-documents, doc_type='10K_risk_factors'
      - extract_mda           -> per-section sub-documents, doc_type='10K_mda'
      - get_earnings_releases -> one doc per release, doc_type='earnings_release'
      - get_supply_chain      -> one doc with concatenated trigger sentences

    Returns {tool_name, ticker, docs_ingested, total_chunks_inserted}.
    Failed results (`success=False` or empty payload) return a zero-doc
    summary without raising.
    """
    empty = {
        "tool_name":             tool_name,
        "ticker":                (ticker_hint or "UNKNOWN").upper() if ticker_hint else "UNKNOWN",
        "docs_ingested":         0,
        "total_chunks_inserted": 0,
    }
    if not isinstance(result_dict, dict):
        return empty
    if result_dict.get("success") is False:
        # Failed extraction — record nothing, never crash.
        ticker = (result_dict.get("ticker") or ticker_hint or "UNKNOWN")
        empty["ticker"] = ticker.upper() if isinstance(ticker, str) else "UNKNOWN"
        return empty

    if tool_name == "extract_risk_factors":
        return _ingest_risk_or_mda(tool_name, result_dict, "10K_risk_factors", ticker_hint)
    if tool_name == "extract_mda":
        return _ingest_risk_or_mda(tool_name, result_dict, "10K_mda", ticker_hint)
    if tool_name == "get_earnings_releases":
        return _ingest_earnings_releases(result_dict, ticker_hint)
    if tool_name == "get_supply_chain":
        return _ingest_supply_chain(result_dict, ticker_hint)

    # Unknown tool — no-op. Callers can add explicit routes if needed.
    return empty
