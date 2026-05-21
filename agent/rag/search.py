"""Public retrieval entry point for the RAG layer.

This module is the single function the rest of the codebase (MCP tools,
analyst workflows, LangGraph nodes) calls when it needs semantic search over
the ingested corpus. It is a thin, opinionated wrapper around:

  1. `agent.rag.embedder.embed`     -- turn the query into a 384-dim vector
  2. `agent.rag.store.vector_search` -- k-NN against `rag_chunk_embeddings`
  3. A small filter/format pass     -- drop below-threshold rows, attach a
                                       short text preview, build the response
                                       envelope.

We deliberately do NOT re-embed, re-chunk, or re-rank here. Retrieval-time
work that touches the vector store should live in `store.py`. This module is
purely glue.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from agent.rag.embedder import embed
from agent.rag.store import vector_search


_PREVIEW_CHARS = 300


def rag_search(
    query: str,
    ticker: Optional[str] = None,
    doc_type: Optional[str] = None,
    top_k: int = 10,
    min_score: float = 0.0,
) -> Dict[str, Any]:
    """Semantic search over the RAG corpus.

    Args:
      query:     natural-language question or topic
      ticker:    optional ticker filter passed through to vector_search
      doc_type:  optional doc_type filter (e.g. 'analogue', '10K_mda')
      top_k:     k passed to vector_search (k-NN limit before filtering)
      min_score: drop results whose similarity is strictly below this value

    Returns:
      Dict with the shape documented in the agent contract:
        {
          'query', 'filters', 'top_k', 'results_count', 'results': [...],
          ('note', if results_count == 0)
        }
      Each result row carries chunk metadata, the full chunk_text, a short
      chunk_text_preview, and the similarity score in [0, 1].
    """
    # Empty/whitespace queries embed to a (near-)zero vector and produce
    # nonsense rankings. Bail early with an explicit empty envelope.
    if query is None or not query.strip():
        return {
            'query': query or '',
            'filters': {'ticker': ticker, 'doc_type': doc_type},
            'top_k': top_k,
            'results_count': 0,
            'results': [],
            'note': 'empty query',
        }

    q_embedding = embed(query)
    raw = vector_search(
        q_embedding,
        top_k=top_k,
        ticker_filter=ticker,
        doc_type_filter=doc_type,
    )

    # Filter on min_score and format. vector_search already orders ASC by
    # distance, which equals DESC by similarity for normalized vectors --
    # but we re-sort explicitly so the contract holds even if the upstream
    # ever changes.
    filtered = []
    for row in raw:
        sim = float(row.get('similarity', 0.0))
        if sim < min_score:
            continue
        chunk_text = row.get('chunk_text') or ''
        filtered.append({
            'chunk_id':           row.get('chunk_id'),
            'doc_id':             row.get('doc_id'),
            'ticker':             row.get('ticker'),
            'doc_type':           row.get('doc_type'),
            'filing_date':        row.get('filing_date'),
            'item_number':        row.get('item_number'),
            'section_heading':    row.get('section_heading'),
            'source_tool':        row.get('source_tool'),
            'chunk_text':         chunk_text,
            'chunk_text_preview': chunk_text[:_PREVIEW_CHARS],
            'similarity':         sim,
        })

    filtered.sort(key=lambda r: r['similarity'], reverse=True)

    out: Dict[str, Any] = {
        'query':         query,
        'filters':       {'ticker': ticker, 'doc_type': doc_type},
        'top_k':         top_k,
        'results_count': len(filtered),
        'results':       filtered,
    }
    if not filtered:
        out['note'] = 'no matches above min_score'
    return out
