from ddgs import DDGS
from typing import List, Dict, Optional

def search_duckduckgo(query: str, max_results = 5) -> List[Dict]:
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

  except Exception as e:  # Catch the actual error
    print(f"DuckDuckGo search failed: {e}")
    return []

if __name__ == "__main__":
  print(search_duckduckgo("MSFT earnings"))
