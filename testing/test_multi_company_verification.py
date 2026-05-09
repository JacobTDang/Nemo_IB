
import sys
import os
import importlib.util
import time
import random
import json
from datetime import datetime

# Add project root to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_dir)

# Import the module with a numeric start using importlib
module_path = os.path.join(root_dir, 'tools', 'web_search_server', '8K_and_DEF14A_utils.py')
spec = importlib.util.spec_from_file_location("sec_utils_8k", module_path)
sec_utils = importlib.util.module_from_spec(spec)
sys.modules["sec_utils_8k"] = sec_utils
spec.loader.exec_module(sec_utils)

# Import TestStockUniverse
try:
    from testing.test_sec_xbrl_functions import TestStockUniverse
    companies = TestStockUniverse.get_all_tickers()
except ImportError:
    print("Could not import TestStockUniverse. Using fallback list.")
    companies = ["MSFT", "AAPL", "NVDA", "GOOGL", "META", "AMZN", "TSLA", "BRK.B", "LLY", "V", 
                 "JPM", "XOM", "WMT", "UNH", "MA", "PG", "JNJ", "HD", "MRK", "COST", "ABBV", 
                 "CVX", "CRM", "BAC", "PEP", "KO", "AMD", "NFLX", "ADBE", "DIS"] # Fallback

def test_company(ticker):
    print(f"\n{'='*40} {ticker} {'='*40}")
    
    results = {
        'ticker': ticker,
        'comp_extracted': False,
        'comp_verified': False,
        'gov_extracted': False,
        'gov_verified': False,
        'alerts': [],
        'timestamp': datetime.now().isoformat()
    }

    # 1. Compensation
    try:
        comp_res = sec_utils.extract_proxy_compensation(ticker)
        if comp_res['success']:
            results['comp_extracted'] = True
            quality = comp_res.get('quality_report', {})
            alerts = quality.get('potential_error_alerts', [])
            
            if not alerts:
                results['comp_verified'] = True
                print(f"SUCCESS - Compensation: Verified (Top Exec: ${comp_res['executives'][0]['total_compensation']:,.0f})")
            else:
                results['alerts'].extend([f"[COMP] {a}" for a in alerts])
                print(f"ALERT - Compensation Alerts: {len(alerts)}")
        else:
            print(f"ERROR - Compensation Extraction Failed: {comp_res.get('error')}")
    except Exception as e:
        print(f"EXCEPTION - Compensation: {e}")

    # 2. Governance
    try:
        gov_res = sec_utils.extract_governance_data(ticker)
        if gov_res['success']:
            results['gov_extracted'] = True
            quality = gov_res.get('quality_report', {})
            alerts = quality.get('potential_error_alerts', [])
            
            if not alerts:
                results['gov_verified'] = True
                print(f"SUCCESS - Governance: Verified ({len(gov_res['board_members'])} members)")
            else:
                results['alerts'].extend([f"[GOV] {a}" for a in alerts])
                print(f"ALERT - Governance Alerts: {len(alerts)}")
        else:
            print(f"ERROR - Governance Extraction Failed: {gov_res.get('error')}")
    except Exception as e:
        print(f"EXCEPTION - Governance: {e}")
        
    return results

if __name__ == "__main__":
    
    # Shuffle to mix it up
    random.shuffle(companies)
    # Take 100 (or all if less) - let's do all 500 as requested
    test_set = companies 
    
    print(f"Starting Batch Verification on {len(test_set)} Companies...")
    print("="*80)
    
    summary_stats = {
        'total': 0,
        'comp_success': 0,
        'comp_verified': 0,
        'gov_success': 0,
        'gov_verified': 0,
        'alerts_total': 0
    }
    
    all_alerts = []
    full_results_log = []
    
    # Ensure output directory exists
    output_dir = os.path.join(root_dir, 'testing', 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'sp500_verification_results.json')

    start_time = time.time()
    
    for i, ticker in enumerate(test_set):
        print(f"\nProcessing {i+1}/{len(test_set)}...")
        res = test_company(ticker)
        full_results_log.append(res)
        
        summary_stats['total'] += 1
        if res['comp_extracted']: summary_stats['comp_success'] += 1
        if res['comp_verified']: summary_stats['comp_verified'] += 1
        if res['gov_extracted']: summary_stats['gov_success'] += 1
        if res['gov_verified']: summary_stats['gov_verified'] += 1
        
        if res['alerts']:
            summary_stats['alerts_total'] += len(res['alerts'])
            all_alerts.append((ticker, res['alerts']))
            
        # Save progress every 10 companies
        if (i + 1) % 10 == 0:
            with open(output_file, 'w') as f:
                json.dump(full_results_log, f, indent=2)
            print(f"--> Progress saved to {output_file}")
            
        # Rate limiting: 2 seconds delay
        time.sleep(2)

    duration = time.time() - start_time
    
    # Final save
    with open(output_file, 'w') as f:
        json.dump(full_results_log, f, indent=2)
    
    print("\n" + "="*80)
    print("FINAL SUMMARY REPORT")
    print("="*80)
    print(f"Total Companies Processed: {summary_stats['total']}")
    print(f"Duration: {duration:.1f} seconds")
    print("-" * 40)
    print(f"Compensation Extraction: {summary_stats['comp_success']}/{summary_stats['total']} ({summary_stats['comp_success']/summary_stats['total']*100:.1f}%)")
    print(f"Compensation Verified:   {summary_stats['comp_verified']}/{summary_stats['comp_success']} verified without alerts")
    print("-" * 40)
    print(f"Governance Extraction:   {summary_stats['gov_success']}/{summary_stats['total']} ({summary_stats['gov_success']/summary_stats['total']*100:.1f}%)")
    print(f"Governance Verified:     {summary_stats['gov_verified']}/{summary_stats['gov_success']} verified without alerts")
    print("="*80)
    
    if all_alerts:
        print("\nTOP ALERTS (First 20):")
        for ticker, alerts in all_alerts[:20]:
            print(f"\n{ticker}:")
            for alert in alerts:
                print(f"  - {alert}")
