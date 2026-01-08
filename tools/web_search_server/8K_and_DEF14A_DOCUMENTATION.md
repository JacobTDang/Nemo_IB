A production-grade parser for SEC 8-K and DEF 14A filings that extracts executive compensation, board governance data, and material events with strict truth-grounding and evidence tracking.

Thank you claude sonnet 4.5, they could never make me hate you

## Key Features

- Truth-grounded extraction: All assertions backed by evidence
- Validation framework: Catches extraction errors before they propagate
- Quality metrics: Transparency into data completeness and confidence
- LLM-ready output: Raw table text for complex visual formats

## Functions Available

### SECFilingParser Class Methods

#### extract_8k_events(ticker, limit=10, debug=False) -> Dict[str, Any]
**Parameters:**
- `ticker` (str): Stock symbol
- `limit` (int): Max filings to process (default: 10)
- `debug` (bool): Print debug output (default: False)

**When to use:** Extract material corporate events from 8-K filings for due diligence, M&A analysis, or event-driven investment strategies

**Expected output:**
```python
{
    'ticker': 'AAPL',
    'events_by_date': {
        '2025-01-15': {
            'event_type': 'management_change',
            'category': 'MANAGEMENT_CHANGE',
            'sec_items': ['5.02', '9.01'],
            'items_source': 'eightk_obj',
            'confidence': 0.85,
            'evidence': {
                'strategy': 'eightk_obj',
                'snippet': 'Item 5.02. Departure of Directors...',
                'location': {'line_range': [45, 52]}
            },
            'text': 'Full 8-K text...'
        }
    },
    'total_events': 5,
    'success': True,
    'quality_report': {
        'assertions_missing_evidence': 0,
        'evidence_coverage_pct': 100.0,
        'eight_k_avg_confidence': 0.82,
        'validation_assertions_passed': True
    }
}
```

#### extract_proxy_compensation(ticker, debug=False) -> Dict[str, Any]
**Parameters:**
- `ticker` (str): Stock symbol
- `debug` (bool): Print debug output (default: False)

**When to use:** Analyze executive compensation for comparable company analysis, governance research, or compensation benchmarking

**Expected output:**
```python
{
    'ticker': 'AAPL',
    'executives': [
        {
            'name': 'Tim Cook',
            'total_compensation': 63209845.0,
            'year': 2024,
            'extraction_method': 'table',
            'evidence': {
                'strategy': 'table_parse',
                'snippet': 'Name: Tim Cook | Total: 63209845',
                'location': {'table_index': 94, 'row_index': 0},
                'column_header': 'Total ($)',
                'table_kv': {
                    'Name': 'Tim Cook',
                    'Total ($)': '63209845'
                }
            },
            'validation_passed': True,
            'validation_alerts': []
        }
    ],
    'candidates': [],
    'success': True,
    'quality_report': {
        'assertions_missing_evidence': 0,
        'evidence_coverage_pct': 100.0,
        'validation_assertions_passed': True
    }
}
```

#### extract_governance_data(ticker, debug=False) -> Dict[str, Any]
**Parameters:**
- `ticker` (str): Stock symbol
- `debug` (bool): Print debug output (default: False)

**When to use:** Evaluate board composition and independence for ESG analysis, governance scoring, or investment committee reports. For complex table formats (icons, visual markers), use `raw_table_text` with your LLM.

**Expected output:**
```python
{
    'ticker': 'AAPL',
    'board_members': [
        {
            'name': 'Arthur D. Levinson',
            'age': 74,
            'tenure': 'Since 2011',
            'independent': True,
            'extraction_method': 'table',
            'evidence': {
                'strategy': 'table_parse',
                'snippet': 'Name: Arthur D. Levinson | Age: 74 | Independent: Yes',
                'location': {'table_index': 12, 'row_index': 1},
                'column_header': 'Independent',
                'table_kv': {
                    'Name': 'Arthur D. Levinson',
                    'Age': '74',
                    'Independent': 'Yes'
                }
            },
            'validation_passed': True,
            'validation_alerts': []
        }
    ],
    'candidates': [],
    'success': True,
    'governance_metrics': {
        'total': 8,
        'independent_true': 7,
        'independent_known_total': 8,
        'independent_unknown': 0,
        'independence_pct_known': 100.0
    },
    'quality_report': {
        'assertions_missing_evidence': 0,
        'evidence_coverage_pct': 100.0,
        'validation_assertions_passed': True
    },
    'raw_table_text': 'Name Age Since Independent\nArthur D. Levinson 74 2011 Yes\n...'
}
```

**Note on raw_table_text:** For tables with visual markers (checkmarks, icons) that the parser cannot interpret, `raw_table_text` provides clean table text for LLM processing. See LLM integration example below.

### Standalone Convenience Functions

#### extract_8k_events(ticker, limit=10) -> Dict[str, Any]
**Parameters:**
- `ticker` (str): Stock symbol
- `limit` (int): Max filings (default: 10)

**When to use:** One-line extraction of 8-K events for scripting

**Expected output:** Same as class method

#### extract_proxy_compensation(ticker) -> Dict[str, Any]
**Parameters:**
- `ticker` (str): Stock symbol

**When to use:** One-line executive compensation extraction

**Expected output:** Same as class method

#### extract_governance_data(ticker) -> Dict[str, Any]
**Parameters:**
- `ticker` (str): Stock symbol

**When to use:** One-line board governance analysis

**Expected output:** Same as class method

## Usage Examples

### Basic Usage
```python
from sec_filing_parser_v7_2 import SECFilingParser

# Using class methods (recommended)
parser = SECFilingParser()
comp_data = parser.extract_proxy_compensation("AAPL")
gov_data = parser.extract_governance_data("AAPL")
events = parser.extract_8k_events("AAPL", limit=5)

# Using convenience functions (quick scripts)
comp_data = extract_proxy_compensation("AAPL")
```

### Debug Mode
```python
parser = SECFilingParser()

# Enable debug output to see extraction process
result = parser.extract_governance_data("GOOGL", debug=True)
```

### LLM Integration for Complex Tables
```python
# For tables with icons or visual markers
result = parser.extract_governance_data("GOOGL")

# Check if independence needs LLM processing
if result['governance_metrics']['independent_unknown'] > 0:
    # Pass raw table to your LLM
    prompt = f"""
    Here is a board table where blank Independent column means NOT independent:

    {result['raw_table_text']}

    Which directors are independent? Return JSON with names and boolean.
    """

    # Your LLM processes it
    llm_response = your_llm.generate(prompt)

    # Merge LLM results with parser data
    for director in result['board_members']:
        if director['independent'] is None:
            # Update from LLM response
            director['independent'] = llm_result[director['name']]
```

### Accessing Evidence
```python
result = parser.extract_proxy_compensation("AAPL")

for exec in result['executives']:
    print(f"Name: {exec['name']}")
    print(f"Compensation: ${exec['total_compensation']:,.0f}")
    print(f"Evidence: {exec['evidence']['snippet'][:100]}...")
    print(f"Source: Table {exec['evidence']['location']['table_index']}, Row {exec['evidence']['location']['row_index']}")
```

### Quality Validation
```python
result = parser.extract_governance_data("AAPL")

# Check quality metrics
quality = result['quality_report']
print(f"Evidence coverage: {quality['evidence_coverage_pct']}%")
print(f"Validation passed: {quality['validation_assertions_passed']}")

# Check for candidates (unvalidated data)
if result['candidates']:
    print(f"Warning: {len(result['candidates'])} entries failed validation")
    for candidate in result['candidates']:
        print(f"  - {candidate['name']}: {candidate['validation_alerts']}")
```

## Installation
```bash
pip install sec-edgar-downloader pandas beautifulsoup4 lxml --break-system-packages

# Set SEC identity (required by SEC.gov)
export NAME="Your Name"
export SEC_EMAIL="your.email@company.com"
```

## Command Line Testing
```bash
python 8K_and_DEF14A_utils.py
```

## Truth-Grounding Guarantees

All extracted data is truth-grounded:

1. **Evidence requirement**: Every assertion includes source evidence (snippet, location)
2. **Validation framework**: Data that fails validation is moved to `candidates`, not included in main results
3. **Quality metrics**: Transparency into extraction quality and data completeness
4. **Honest unknowns**: Fields are set to `None` when data cannot be reliably determined

## Output Structure

### Common Fields

All extraction methods return:
- `ticker`: Stock symbol
- `success`: Boolean indicating if extraction succeeded
- `quality_report`: Metrics on extraction quality
- `candidates`: Data that failed validation (if any)

### Error Handling
```python
result = parser.extract_governance_data("INVALID")

if not result['success']:
    print(f"Error: {result.get('error', 'Unknown error')}")
else:
    # Process successful result
    pass
```
