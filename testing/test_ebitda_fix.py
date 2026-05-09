"""Test that EBITDA margin fix picks consolidated total, not segment"""
from tools.web_search_server.sec_utils import get_ebitda_margin

result = get_ebitda_margin('AAPL', '10-K')
if result.get('success'):
    print(f"EBITDA Margin: {result['ebitda_margin_percent']:.2f}%")
    print(f"EBITDA Amount: ${result['ebitda_amount']/1e9:.2f}B")
    print(f"Operating Income: ${result['operating_income']/1e9:.2f}B")
    print(f"D&A: ${result['d&a']/1e9:.2f}B")
    print(f"Revenue: ${result['revenue']/1e9:.2f}B")
else:
    print(f"Error: {result}")
