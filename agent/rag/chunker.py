"""Paragraph-aware text chunker for the RAG ingest pipeline.

Splits a long document into overlapping chunks sized for the sentence-
transformers MiniLM model (which has a 512-token context window). The chunker
is intentionally simple — it does not understand document structure beyond
blank-line paragraph breaks. Section headings, when known by the caller, are
attached to every chunk so downstream retrieval can show context.

Token counts come from `agent.falsifier_evaluator._tokenize` so chunk sizing
is consistent with other parts of the codebase that reason about token
budgets.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

from agent.falsifier_evaluator import _tokenize


_PARA_SPLIT = re.compile(r"\n\s*\n")


def _count_tokens(text: str) -> int:
    """Token count via the project's shared tokenizer."""
    return len(_tokenize(text))


def _take_overlap_tokens(text: str, n_tokens: int) -> str:
    """Return the suffix of `text` whose token count is roughly `n_tokens`.

    We don't need to be exact — we want a small bridge of context between
    consecutive chunks so embeddings retain continuity. We walk backwards
    through whitespace-split words and stop once we've collected n_tokens
    significant tokens (matching the falsifier tokenizer's definition).
    """
    if n_tokens <= 0 or not text:
        return ""
    words = text.split()
    if not words:
        return ""
    # Walk from the tail; build a candidate suffix and stop once its token
    # count reaches n_tokens. This is O(len(words)) but bounded by overlap
    # size so it's cheap.
    for i in range(len(words) - 1, -1, -1):
        candidate = " ".join(words[i:])
        if _count_tokens(candidate) >= n_tokens:
            return candidate
    return " ".join(words)


def chunk_text(
    text: str,
    target_tokens: int = 500,
    overlap_tokens: int = 50,
    section_heading: Optional[str] = None,
) -> List[Dict]:
    """Split `text` into overlapping, paragraph-aware chunks.

    Args:
      text: source text to chunk. Empty or whitespace-only text returns [].
      target_tokens: chunks grow greedily until their token count reaches
        this size. The final chunk may be smaller.
      overlap_tokens: approximate token count of context to carry over from
        the previous chunk into the next. 0 disables overlap.
      section_heading: optional heading to attach to every emitted chunk.

    Returns:
      A list of dicts with keys:
        chunk_text       — the chunk's text (overlap + paragraphs)
        chunk_offset     — character offset of the chunk's first paragraph
                           in the original text (overlap excluded)
        chunk_sequence   — 0-based ordinal of this chunk in the document
        section_heading  — passthrough of the heading argument
        token_count      — approximate token count of chunk_text
    """
    if not text or not text.strip():
        return []

    # Split into paragraphs while tracking each paragraph's start offset in
    # the original text. We use finditer on the separator pattern to walk
    # consistent positions.
    paragraphs: List[Dict] = []
    last_end = 0
    for m in _PARA_SPLIT.finditer(text):
        para = text[last_end:m.start()]
        if para.strip():
            paragraphs.append({"text": para, "offset": last_end})
        last_end = m.end()
    tail = text[last_end:]
    if tail.strip():
        paragraphs.append({"text": tail, "offset": last_end})

    if not paragraphs:
        return []

    chunks: List[Dict] = []
    cur_paragraphs: List[Dict] = []
    cur_tokens = 0
    cur_offset: Optional[int] = None
    prev_overlap = ""
    sequence = 0

    def _emit():
        nonlocal cur_paragraphs, cur_tokens, cur_offset, prev_overlap, sequence
        if not cur_paragraphs:
            return
        body = "\n\n".join(p["text"] for p in cur_paragraphs)
        if prev_overlap:
            chunk_text_full = prev_overlap + "\n\n" + body
        else:
            chunk_text_full = body
        chunks.append({
            "chunk_text":      chunk_text_full,
            "chunk_offset":    cur_offset if cur_offset is not None else 0,
            "chunk_sequence":  sequence,
            "section_heading": section_heading,
            "token_count":     _count_tokens(chunk_text_full),
        })
        # Carry tail tokens into the next chunk as overlap context.
        prev_overlap = _take_overlap_tokens(body, overlap_tokens)
        sequence += 1
        cur_paragraphs = []
        cur_tokens = 0
        cur_offset = None

    for para in paragraphs:
        ptoks = _count_tokens(para["text"])

        # Edge case: a single paragraph already exceeds target_tokens. We
        # still emit it as its own chunk rather than splitting mid-sentence,
        # because the embedding model truncates internally and chunks are
        # only "soft" sized.
        if cur_paragraphs and cur_tokens + ptoks > target_tokens and cur_tokens >= target_tokens // 2:
            _emit()

        if not cur_paragraphs:
            cur_offset = para["offset"]
        cur_paragraphs.append(para)
        cur_tokens += ptoks

        if cur_tokens >= target_tokens:
            _emit()

    # Tail flush
    if cur_paragraphs:
        _emit()

    return chunks
