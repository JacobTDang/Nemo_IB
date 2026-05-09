"""Debug: check what OperatingIncomeLoss facts exist for AAPL"""
from tools.web_search_server.sec_utils import get_latest_filing
from edgar import set_identity
import os
import pandas as pd

set_identity(f"{os.getenv('NAME', 'Analyst')} {os.getenv('SEC_EMAIL', 'a@b.com')}")
filing = get_latest_filing('AAPL', '10-K')
xbrl = filing['xbrl_data']

# Get ALL OperatingIncomeLoss facts
facts = xbrl.facts.query().by_concept('us-gaap:OperatingIncomeLoss').to_dataframe()
facts['period_start_dt'] = pd.to_datetime(facts['period_start'])
facts['period_end_dt'] = pd.to_datetime(facts['period_end'])
facts['duration_days'] = (facts['period_end_dt'] - facts['period_start_dt']).dt.days

# Show annual facts (350+ days)
annual = facts[facts['duration_days'] >= 350].sort_values('period_end_dt', ascending=False)
latest_period = annual['period_end_dt'].max()
latest_facts = annual[annual['period_end_dt'] == latest_period]

print(f"Latest period end: {latest_period}")
print(f"Number of facts for latest period: {len(latest_facts)}")
print()
for i, (_, row) in enumerate(latest_facts.iterrows()):
    val = row['numeric_value']
    dims = row.get('dimensions', 'none')
    print(f"  Fact {i}: ${val/1e9:.2f}B  dims={dims}")
print()
print(f"iloc[0] picks: ${latest_facts.iloc[0]['numeric_value']/1e9:.2f}B")
print(f"max picks:     ${latest_facts['numeric_value'].max()/1e9:.2f}B")

# Also check D&A
print("\n--- D&A Facts ---")
da_facts = xbrl.facts.query().by_concept('us-gaap:DepreciationDepletionAndAmortization').to_dataframe()
da_facts['period_start_dt'] = pd.to_datetime(da_facts['period_start'])
da_facts['period_end_dt'] = pd.to_datetime(da_facts['period_end'])
da_facts['duration_days'] = (da_facts['period_end_dt'] - da_facts['period_start_dt']).dt.days
da_annual = da_facts[da_facts['duration_days'] >= 350]
da_latest = da_annual[da_annual['period_end_dt'] == da_annual['period_end_dt'].max()]
print(f"D&A facts for latest period: {len(da_latest)}")
for i, (_, row) in enumerate(da_latest.iterrows()):
    val = row['numeric_value']
    dims = row.get('dimensions', 'none')
    print(f"  Fact {i}: ${val/1e9:.2f}B  dims={dims}")

# What EBITDA should be
print("\n--- Calculation ---")
op_max = latest_facts['numeric_value'].max()
da_max = da_latest['numeric_value'].max()
print(f"Operating Income (max): ${op_max/1e9:.2f}B")
print(f"D&A (max): ${da_max/1e9:.2f}B")
print(f"EBITDA (max + max): ${(op_max + da_max)/1e9:.2f}B")
print(f"EBITDA margin: {(op_max + da_max) / 416.16e9 * 100:.2f}%")
