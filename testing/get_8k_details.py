
import sys
import os
import json

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools.web_search_server.8k_and_DEF14A_utils import SECFilingParser

def get_detailed_8k(ticker):
    parser = SECFilingParser()
    print(f"Fetching detailed 8-K events for {ticker}...\n")
    
    # Increase limit to get more history if needed
    events = parser.extract_8k_events(ticker, limit=5)
    
    if not events['success']:
        print(f"Error: {events.get('error')}")
        return

    print(f"Found {events['total_events']} events for {ticker}:")
    print("-" * 80)
    
    # Sort by date descending
    sorted_dates = sorted(events['events_by_date'].keys(), reverse=True)
    
    for date in sorted_dates:
        evt = events['events_by_date'][date]
        print(f"Date: {date}")
        print(f"Type: {evt['event_type']} ({evt.get('category')})")
        print(f"Items: {evt['sec_items']}")
        print(f"Source: {evt.get('items_source')}")
        print(f"Confidence: {evt.get('confidence')}")
        
        # In a real application, you might want to fetch and show the text snippet here.
        # The current extract_8k_events doesn't return the full text body to keep the dict small,
        # but the classification result 'matched_keywords' gives a hint.
        print(f"Keywords Found: {evt.get('matched_keywords', [])}") 
        
        # If 'key_facts' are available (extracted in the parser but maybe not fully populated in output dict depending on implementation)
        if 'key_facts' in evt:
             print(f"Key Facts: {evt['key_facts']}")

        print("-" * 80)

if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "GOOGL"
    get_detailed_8k(ticker)
