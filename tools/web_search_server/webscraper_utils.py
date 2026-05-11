"""Legacy webscraper module — superseded by searxng_client.py and scraper.py.

Kept as an empty module to avoid breaking imports during git history bisects.
All functions previously here (search_duckduckgo, web_scrape, SessionManager,
batch_scrape, 5 extraction heuristics, content quality scoring) were removed
in favor of Trafilatura + Crawl4AI.
"""
