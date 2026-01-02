import requests
import httpx
from ddgs import DDGS
import urllib.parse
from typing import List, Dict, Optional

def search_duckduckgo(query: str, max_results = 10) -> List[Dict]:
  try:
    search_results = []

    with DDGS() as ddgs:
      for result in ddgs.text(query, max_results=max_results):
        search_results.append({
          'title': result.get('title', ''),
          'link': result.get('href', ''),
          'snippet': result.get('body',''),
          'source': 'duckduckgo'
        })

    return search_results

  except:
    print("Duckduckgo search failed")
    return []

if __name__ == "__main__":
  print(search_duckduckgo("MSFT earnings"))
