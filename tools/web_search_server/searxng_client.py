"""Thin httpx client for a local SearXNG instance.

SearXNG aggregates results from Google, Bing, Brave, DuckDuckGo, Startpage, and Qwant
in one query. Run the container with `docker compose up -d searxng` from the project root.
"""
import os
import sys
from typing import List, Dict

import httpx

SEARXNG_URL = os.getenv("SEARXNG_URL", "http://localhost:8888")
_DEFAULT_TIMEOUT = 15.0
_DEFAULT_ENGINES = "google,bing,brave,duckduckgo,startpage"


async def searxng_search(query: str, max_results: int = 5,
                         engines: str = _DEFAULT_ENGINES) -> List[Dict]:
  """Query SearXNG localhost and return search results.

  Returns list of dicts shaped for execution_engine consumption:
    [{title, link, snippet, source}]
  """
  try:
    async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client:
      response = await client.get(
        f"{SEARXNG_URL}/search",
        params={
          "q": query,
          "format": "json",
          "engines": engines,
          "safesearch": 0,
          "categories": "general",
        }
      )
      response.raise_for_status()
      data = response.json()
  except httpx.ConnectError:
    print(f"[SearXNG] Cannot reach {SEARXNG_URL}. Is the container running? "
          "Run `docker compose up -d searxng`.", file=sys.stderr, flush=True)
    return []
  except Exception as e:
    print(f"[SearXNG] Query failed: {e}", file=sys.stderr, flush=True)
    return []

  raw_results = data.get("results", [])
  out: List[Dict] = []
  seen = set()
  for r in raw_results:
    url = r.get("url", "")
    if not url or url in seen:
      continue
    seen.add(url)
    out.append({
      "title": r.get("title", ""),
      "link": url,
      "snippet": r.get("content", ""),
      "source": ",".join(r.get("engines", [])) or "searxng",
    })
    if len(out) >= max_results:
      break

  print(f"[SearXNG] {len(out)} results for '{query[:60]}'", file=sys.stderr, flush=True)
  return out
