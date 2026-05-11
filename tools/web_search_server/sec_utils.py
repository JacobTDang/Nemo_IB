from typing import Any, Dict, Optional
from edgar import Company, set_identity
from edgar.xbrl import XBRL
import pandas as pd
import os
import re
import sys
# useful documentation for edgartools xbrl: https://edgartools.readthedocs.io/en/latest/getting-xbrl/

# set identity first
NAME = os.getenv('NAME', 'Investment Analyst')
SEC_EMAIL = os.getenv('SEC_EMAIL', 'analyst@example.com')

# Module-level cache: prevents the 4 SEC tools (ebitda, capex, tax, depreciation) from each
# downloading and parsing the same 10-K XBRL independently. Within one MCP server process
# lifetime the filing content doesn't change, so caching is safe.
_filing_cache: Dict[tuple, Any] = {}


def filter_instant_data(xbrl, concept: str) -> Optional[Dict[str, Any]]:
  """Filter XBRL for instant (point-in-time) facts -- balance sheet items.

  Balance sheet items use 'period_instant' rather than 'period_start'/'period_end'.
  Returns the most recent fact and its as-of date.
  """
  try:
    facts = xbrl.facts.query().by_concept(concept).to_dataframe()
    if facts.empty:
      return None
    # Filter to instant-type rows (balance sheet); period_instant has the as-of date
    if 'period_type' in facts.columns:
      facts = facts[facts['period_type'] == 'instant']
    if facts.empty:
      return None
    facts['instant_dt'] = pd.to_datetime(facts['period_instant'])
    latest = facts[facts['instant_dt'] == facts['instant_dt'].max()]
    if len(latest) > 1:
      # Consolidated total = largest absolute value among segments
      latest = latest.loc[latest['numeric_value'].abs().idxmax()]
    else:
      latest = latest.iloc[0]
    return {
      'value': latest['numeric_value'],
      'concept_used': concept,
      'period_end': latest['period_instant'],
    }
  except Exception:
    return None


def filter_annual_data(xbrl, concept: str, form_type: str = '10-K') -> Optional[Dict[str, Any]]:
  """
  Helper function to filter XBRL facts for period data based on form type
  Returns latest period data or None if not found
  - 10-K: Annual data (350+ days)
  - 10-Q: Quarterly data (80-95 days)
  - Others: Most recent available
  """
  try:
    facts = xbrl.facts.query().by_concept(concept).to_dataframe()

    if facts.empty:
      return None

    # Filter for appropriate periods based on form type
    facts['period_start_dt'] = pd.to_datetime(facts['period_start'])
    facts['period_end_dt'] = pd.to_datetime(facts['period_end'])
    facts['duration_days'] = (facts['period_end_dt'] - facts['period_start_dt']).dt.days

    # Set period filters based on form type
    if form_type == '10-K':
      # Annual data (350+ days for fiscal year variations)
      target_periods = facts[facts['duration_days'] >= 350]
    elif form_type == '10-Q':
      # Quarterly data (80-95 days typically)
      target_periods = facts[(facts['duration_days'] >= 80) & (facts['duration_days'] <= 95)]
    else:
      # For other forms (8-K, S-1, etc.), get most recent regardless of duration
      target_periods = facts

    if not target_periods.empty:
      period_data = target_periods[target_periods['period_end_dt'] == target_periods['period_end_dt'].max()]
      # XBRL often has multiple facts for the same concept and period
      # (segments + consolidated total). The consolidated total is always
      # the largest positive value, so take the max for any concept that
      # can have segment breakdowns.
      if len(period_data) > 1:
        period_data = period_data.loc[period_data['numeric_value'].idxmax()]
      else:
        period_data = period_data.iloc[0]
    else:
      # Fallback to most recent data if no target periods found
      period_data = facts[facts['period_end_dt'] == facts['period_end_dt'].max()]
      if len(period_data) > 1:
        period_data = period_data.loc[period_data['numeric_value'].idxmax()]
      else:
        period_data = period_data.iloc[0]

    if period_data is not None and not (hasattr(period_data, 'empty') and period_data.empty):
      latest_row = period_data

      return {
        'value': latest_row['numeric_value'],
        'concept_used': concept,
        'period_end': latest_row['period_end'],
        'duration_days': latest_row['duration_days']
      }

    return None

  except Exception:
    return None

def get_latest_filing(ticker: str, form_type: str = '10-K') -> Optional[Dict[str, Any]]:
  """Get latest SEC filing with XBRL data.

  Results are cached in-process by (ticker, form_type) key.
  When multiple SEC tools run in the same parallel batch (e.g. get_ebitda_margin,
  get_capex_pct_revenue, get_tax_rate, get_depreciation all called concurrently),
  the cache prevents 4 duplicate 10-K downloads that would trigger SEC rate limiting.
  """
  cache_key = (ticker.upper(), form_type)
  if cache_key in _filing_cache:
    return _filing_cache[cache_key]

  result = None
  try:
    # Set identity globally first
    set_identity(f"{NAME} {SEC_EMAIL}")

    # Create company object and get filings
    company = Company(ticker)
    filings = company.get_filings(form=form_type)

    if filings:
      latest_filing = filings[0]

      # Get XBRL data
      try:
        xbrl_data = latest_filing.xbrl()
      except Exception:
        xbrl_data = None

      # Get filing URL
      url = None
      for attr in ['filing_url', 'url', 'filing_details_url', 'document_url']:
        if hasattr(latest_filing, attr):
          url = getattr(latest_filing, attr)
          break

      result = {
        'filing_date': latest_filing.filing_date,
        'url': url,
        'accession_number': latest_filing.accession_number,
        'filing_object': latest_filing,
        'xbrl_data': xbrl_data
      }

  except Exception:
    result = None

  _filing_cache[cache_key] = result
  return result

def get_disclosures_names(ticker:str, form_type: str = '10-K') -> Dict[str, Any]:
  # get the disclosure name for agent to use
  try:
    filing_data = get_latest_filing(ticker, form_type)
    if filing_data and filing_data['xbrl_data']:
      xbrl=filing_data['xbrl_data']
      disclosures = []
      try:

        statements = xbrl.statements

        # Get all disclosure statements
        disclosure_statements = statements.disclosures()

        for disclosure in disclosure_statements:
          # Get role/type and clean it up for readability
          if hasattr(disclosure, 'role_or_type'):
            role = disclosure.role_or_type
            # Extract the disclosure name from the URL
            if '/' in role:
              disclosure_name = role.split('/')[-1]
            else:
              disclosure_name = role
            disclosures.append(disclosure_name)

        if disclosures:
          return {
            'ticker': ticker,
            'success': True,
            'error': None,
            'disclosure_names': disclosures
          }
        else:
          return{
            'ticker': ticker,
            'success': False,
            'error': f'Unable to find any disclosure in {ticker} statements',
            'disclosure_names': None
          }

      except Exception as e:
        return{
          'ticker': ticker,
          'success': False,
          'error': f"Unable to find disclosure in disclosure concepts for {ticker}: {str(e)}",
          'disclosure_names': None
        }


  except Exception as e:
    return {
      'ticker': ticker,
      'success': False,
      'error': f'Unable to get disclosures for {ticker}: {str(e)}',
      'disclosure_names': None
    }
  return {
    'ticker': ticker,
    'success': False,
    'error': f'Unable to get disclosures for {ticker}',
    'disclosure_names': None
  }

def extract_disclosure_data(ticker: str, disclosure_name: str, form_type: str = '10-K') -> Dict[str, Any]:

  try:
    latest_filing = get_latest_filing(ticker, form_type)
    if latest_filing and latest_filing['xbrl_data']:
      xbrl = latest_filing['xbrl_data']

      try:
        statement = xbrl.statements
        disclosures =  statement.disclosures()

        # Find the specific disclosure by name
        target_disclosure = None
        for disclosure in disclosures:
          if hasattr(disclosure, 'role_or_type'):
            role = disclosure.role_or_type
            # Extract the disclosure name from the URL
            if '/' in role:
              current_name = role.split('/')[-1]
            else:
              current_name = role

            if current_name == disclosure_name:
              target_disclosure = disclosure
              break

        if target_disclosure:
          print(f'Found disclosure: {disclosure_name}', file=sys.stderr, flush=True)

          # Get summary info about the disclosure
          disclosure_summary = {
            'name': disclosure_name,
            'role_or_type': target_disclosure.role_or_type if hasattr(target_disclosure, 'role_or_type') else None,
            'primary_concept': target_disclosure.primary_concept if hasattr(target_disclosure, 'primary_concept') else None,
            'success': True
          }

          # Try to get DataFrame but filter out text-heavy data
          if hasattr(target_disclosure, 'to_dataframe'):
            try:
              df = target_disclosure.to_dataframe()
              print(f'DataFrame shape: {df.shape if df is not None else None}', file=sys.stderr, flush=True)

              if df is not None and not df.empty:
                disclosure_summary['data_shape'] = df.shape
                disclosure_summary['columns'] = df.columns.tolist()

                # Check if this is mostly text data (like HTML)
                text_heavy = False
                for col in df.columns:
                  if df[col].dtype == 'object':  # String columns
                    sample_text = str(df[col].iloc[0]) if not df[col].isna().iloc[0] else ""
                    if len(sample_text) > 1000 or '<' in sample_text:  # HTML or very long text
                      text_heavy = True
                      break

                if text_heavy:
                  print("This disclosure contains mostly text/HTML data - extracting clean text", file=sys.stderr, flush=True)
                  disclosure_summary['data_type'] = 'text_heavy'

                  # Extract clean text from HTML
                  for col in df.columns:
                    if df[col].dtype == 'object' and not df[col].isna().iloc[0]:
                      raw_content = str(df[col].iloc[0])
                      if '<' in raw_content:  # HTML content
                        # Remove HTML tags but keep the text content
                        clean_text = re.sub(r'<[^>]+>', ' ', raw_content)
                        # Clean up extra whitespace
                        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
                        # Remove special characters like \xa0
                        clean_text = re.sub(r'[\xa0\u00a0]', ' ', clean_text)
                        clean_text = re.sub(r'\s+', ' ', clean_text).strip()

                        disclosure_summary[f'clean_text_{col}'] = clean_text
                        print(f"Extracted clean text length: {len(clean_text)} characters", file=sys.stderr, flush=True)

                else:
                  # Only include actual data for numerical/structured disclosures
                  disclosure_summary['data_type'] = 'structured'
                  disclosure_summary['sample_data'] = df.head(3).to_dict("records")

              print(f'Disclosure summary: {disclosure_summary}', file=sys.stderr, flush=True)

            except Exception as e:
              print(f'Error converting to dataframe: {e}', file=sys.stderr, flush=True)

          return disclosure_summary

        else:
          print(f'debug: Unable to find disclosure: {disclosure_name}', file=sys.stderr, flush=True)
          print(f'Available disclosures: {[d.role_or_type.split("/")[-1] if hasattr(d, "role_or_type") and "/" in d.role_or_type else str(d) for d in disclosures[:5]]}...', file=sys.stderr, flush=True)


      except Exception as e:
        return {
          'error': f'Unable to get statement'
        }
    else:
      # failed to get the filing data
      return {
        'error': f"Unable to get latest filing for {ticker}"
      }
  except Exception as e:
    return {
      'error': f"Unable to get {disclosure_name} for {ticker}: {str(e)}",
      'success': False
    }


  return {}

def get_revenue_base(ticker: str, form_type: str= "10-K") -> Dict[str, Any]:
  # this is the company's recurring revenue from its primary business operations. It will be the starting point for nearly all financial analysis
  try:
    filing_data = get_latest_filing(ticker, form_type)

    if filing_data and filing_data['xbrl_data']:
      xbrl = filing_data['xbrl_data']


      # Try different revenue concept names - prioritize Google's specific concepts
      revenue_concepts = [
        'us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax',  # Google's main revenue concept
        'us-gaap:Revenues',  # Total revenues (most comprehensive)
        'us-gaap:SalesRevenueNet',
        'RevenueFromContractWithCustomerExcludingAssessedTax',  # Without prefix
        'Revenues',
        'Revenue',
        'TotalRevenues',
        'SalesRevenueNet'
      ]

      for concept in revenue_concepts:
        result = filter_annual_data(xbrl, concept, form_type)
        if result:
          return {
            'ticker': ticker,
            'revenue_base': float(result['value']),  # keep in raw dollars
            'concept_used': result['concept_used'],
            'period_end': result['period_end'],
            'filing_date': filing_data['filing_date'],
            'success': True
          }

      return {
        'ticker': ticker,
        'error': 'No revenue concept found',
        'success': False
      }

    return {
      'ticker': ticker,
      'error': 'No XBRL data available',
      'success': False
    }

  except Exception as e:
    return {
      'ticker': ticker,
      'error': f"Unable to get filing data: {str(e)}",
      'success': False
    }

def get_ebitda_margin(ticker: str, form_type: str = '10-K') -> Dict[str, Any]:
  # Ignores interst, taxes, and non-cash expenses, so it allows you to compare the underlying profit generation of a company from there core operations
  # how to ususally calculate it
  # 1. Find the operating income from income statement
  # 2. Find the depreciation & amorization on the cash flow statement
  # 3. Calcualte ebitda = operating income + depreication & amorization, and then divide the sum by revenue
  try:
    ebitda = {} # use to store operating income and depreciation & amorization
    filing = get_latest_filing(ticker, form_type)
    if filing and filing['xbrl_data']:
      xbrl = filing['xbrl_data']

      # get the operating income from income statement
      operating_income_concepts = [
      'us-gaap:OperatingIncomeLoss',        # 95% of companies
      'OperatingIncomeLoss',                # Fallback
      'IncomeLossFromContinuingOperations'  # Edge cases
      ]

      for concept in operating_income_concepts:
        result = filter_annual_data(xbrl, concept, form_type)
        if result:
          ebitda['operating_income'] = result['value']
          ebitda['operating_income_concept_used'] = result['concept_used']
          ebitda['operating_income_period_end'] = result['period_end']
          break


      # get the depreciation & amorization from the cash flow statement
      cashflow_statement_concepts = [
      'us-gaap:DepreciationDepletionAndAmortization',  # First choice - combined total
      'us-gaap:DepreciationAndAmortization',           # Alternative combined
      'DepreciationDepletionAndAmortization'           # Fallback without prefix
      ]


      for concept in cashflow_statement_concepts:
        result = filter_annual_data(xbrl, concept, form_type)
        if result:
          ebitda['d&a'] = result['value']
          ebitda['d&a_concept_used'] = result['concept_used']
          ebitda['d&a_period_end'] = result['period_end']
          break

      if 'd&a' not in ebitda or not ebitda['d&a']:
        # Try to get individual components and add them together
        individual_concepts = [
          'us-gaap:Depreciation',
          'us-gaap:AmortizationOfIntangibleAssets'
        ]

        depreciation_value = 0
        amortization_value = 0
        concepts_used = []
        period_end = None

        for concept in individual_concepts:
          result = filter_annual_data(xbrl, concept, form_type)
          if result:
            if 'Depreciation' in concept:
              depreciation_value = result['value']
            elif 'Amortization' in concept:
              amortization_value = result['value']
            concepts_used.append(result['concept_used'])
            if not period_end:
              period_end = result['period_end']

        # Only set D&A if we found at least one component
        if depreciation_value > 0 or amortization_value > 0:
          ebitda['d&a'] = depreciation_value + amortization_value
          ebitda['d&a_concept_used'] = ' + '.join(concepts_used)
          ebitda['d&a_period_end'] = period_end

      # Get revenue for margin calculation
      revenue_data = get_revenue_base(ticker, form_type)
      if not revenue_data['success']:
        return revenue_data  # Return the error from revenue function

      revenue = revenue_data['revenue_base']  # already in raw dollars

      # Calculate EBITDA and EBITDA margin
      if 'operating_income' not in ebitda or 'd&a' not in ebitda:
        missing = []
        if 'operating_income' not in ebitda: missing.append('operating_income')
        if 'd&a' not in ebitda: missing.append('d&a')
        return {
          'error': f"Missing EBITDA components for {ticker}: {', '.join(missing)}",
          'success': False
        }
      ebitda_amount = ebitda['operating_income'] + ebitda['d&a']
      ebitda_margin_percent = (ebitda_amount / revenue) * 100

      return {
        'error': None,
        'success': True,
        'ticker': ticker,
        'ebitda_margin_percent': float(ebitda_margin_percent),
        'ebitda_amount': float(ebitda_amount),
        'operating_income': float(ebitda['operating_income']),
        'd&a': float(ebitda['d&a']),
        'revenue': float(revenue),
        'operating_income_concept_used': ebitda.get('operating_income_concept_used'),
        'd&a_concept_used': ebitda.get('d&a_concept_used'),
        'period_end': ebitda.get('operating_income_period_end')
      }
    else:
      return {
        'error': f'Unable to get latest filing for {ticker}',
        'success': False
      }


  except Exception as e:
    return {
      'error': f'Failed to get latest file for {ticker}',
      'success': False
    }

def get_capex_pct_revenue(ticker: str, form_type: str = '10-K') -> Dict[str, Any]:
  # function to get capital expenditures: Capex is the money that the company spends to buy, maintain, or upgrade physical assets
  # this metric will show CapEX as percentage of revenue, it shows how much a company is reinvesting back into its assets
  # CapEx % of revenue = capital expedeitures / total revenue
  # can find it on cash flow statement under 'cash flow from investing activities'
  try:
    filing = get_latest_filing(ticker, form_type)

    if filing and filing['xbrl_data']:
      xbrl = filing['xbrl_data']

      primary_capex_concepts = [
      'us-gaap:PaymentsToAcquirePropertyPlantAndEquipment',  # Total PP&E (most common)
      'us-gaap:PaymentsForCapitalExpenditures',              # Direct total CapEx
      'PaymentsToAcquirePropertyPlantAndEquipment',          # Fallback
      'CapitalExpenditures'                                  # Basic total
      ]
      total_capex = 0
      capex_concept_used = None

      for concept in primary_capex_concepts:
        result = filter_annual_data(xbrl, concept, form_type)
        if result:
         total_capex = abs(result['value'])
         capex_concept_used = result['concept_used']
         break

       # unable to find anything in the primary concepts, so we move to components
       # thinking about this, there are many issues that can arise from this
       # issue 1. could be more concepts outside of component concepts
       # issue 2. could overlap when adding capital expenditures
       # please fix in future, this will be just a place holder
      if total_capex == 0:
        component_concepts = [
        'us-gaap:PaymentsToAcquireBuildings',
        'us-gaap:PaymentsToAcquireMachineryAndEquipment',
        'us-gaap:PaymentsToAcquireComputerSoftwareAndEquipment',
        'us-gaap:PaymentsToAcquireOtherPropertyPlantAndEquipment'
        ]
        print(f'WARNING for {ticker}: Might not account for all capital expenditures. Possible overlap of capital expenditures. Using concepts: {component_concepts}', file=sys.stderr)
        for concept in component_concepts:
          result = filter_annual_data(xbrl, concept, form_type)
          if result:
            total_capex += result['value']

        if total_capex == 0:
          return{
            'error': f'Unable to find any concepts for {ticker}',
            'success': False
          }

      # now that we have the capex value we can get the percentage
      revenue_data = get_revenue_base(ticker, form_type)
      if not revenue_data['success']:
        return revenue_data # return the revenue error
      revenue = revenue_data['revenue_base']  # already in raw dollars
      capex_pct = (total_capex / revenue) * 100

      return{
        'error': None,
        'success': True,
        'ticker': ticker,
        'total_capex': float(total_capex),  # keep in raw dollars
        'revenue': float(revenue),  # keep in raw dollars
        'capex_pct_revenue': float(capex_pct),
        'capex_concept_used': capex_concept_used,
        'period_end': revenue_data['period_end']
      }
    else:
      return {
        'error': f"Unable to get xbrl data for: {ticker}",
        'success': False
      }

  except Exception:
    return{
      'error': f'Unable to get filing for {ticker}',
      'success': False
    }


def get_tax_rate(ticker: str, form_type: str = '10-K') -> Dict[str, Any]:
  # returns the effective/actual tax rate that the company pays on its profits
  # can find it on the income statement in 'income before provision for income taxes or similar wording' and 'provision for income taxes
  # formula: Effective tax rate = provision for income taxes / earnings before taxes
  try:
    filing = get_latest_filing(ticker, form_type)
    if filing and filing['xbrl_data']:
      xbrl = filing['xbrl_data']

      tax_expense_concepts = [
      'us-gaap:IncomeTaxExpenseBenefit',                           # Most common - total tax expense
      'us-gaap:ProvisionForIncomeTaxes',                           # Alternative provision concept
      'us-gaap:IncomeTaxesPaid',                                   # Cash taxes paid
      'us-gaap:CurrentIncomeTaxExpense',                           # Current year tax expense
      'IncomeTaxExpenseBenefit',                                   # Without prefix
      'ProvisionForIncomeTaxes',                                   # Without prefix
      'IncomeTaxExpense'                                           # Basic form
      ]
      tax_expense = 0.0 # bc panda dataframe return np.float64
      tax_concept_used = None
      for concept in tax_expense_concepts:
        result = filter_annual_data(xbrl, concept, form_type)
        if result:
          tax_expense = float(result['value'])
          tax_concept_used = concept
          break

      pretax_income_concepts = [
        'us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest',  # Full concept
        'us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxes',                                          # Most common
        'us-gaap:EarningsBeforeIncomeTaxes',                                                                    # Alternative
        'IncomeLossFromContinuingOperationsBeforeIncomeTaxes',                                                  # Without prefix
        'EarningsBeforeIncomeTaxes',                                                                            # Without prefix
        'IncomeBeforeTaxes'                                                                                     # Basic form
      ]

      pretax_income = 0.0
      pretax_concept_used = None
      for concept in pretax_income_concepts:
        result = filter_annual_data(xbrl, concept, form_type)
        if result:
          pretax_income = float(result['value'])
          pretax_concept_used = concept
          break

      # calulate the effective tax rate: Effective tax rate = provision for income taxes / earnings before taxes
      if pretax_income != 0: # prevent divide by 0 error
        effective_tax_rate = (tax_expense / pretax_income) * 100
      else:
        return{
          'error': f"pretax income is 0, unable to calculate effective tax rate",
          'success': False
        }

      return{
        'error': None,
        'success': True,
        'effective_tax_rate': effective_tax_rate,
        'tax_expense': tax_expense,
        'tax_concept_used': tax_concept_used,
        'pretax_income': pretax_income,
        'pretax_concept_used': pretax_concept_used
      }

    else:
      return{
        'error': f'Unable to get xbrl data',
        'success': False
      }
  except Exception as e:
    return{
      'error': f'Unable to get filing for {ticker}: {str(e)}',
      'success': False
    }


def get_depreciation(ticker: str, form_type: str = '10-K') -> Dict[str, Any]:
  # this is the accounting method of allocating the cost of a physical asset over its uselife. It is a non cash expense is will be expressed as a percentage of revenue
  # formula: depreication % of revenue = depreication & amorization / total revenue
  # this will be helpful beacuse it helps us find the age and cost structure of a company's assets
  # find it on the cash flow statement, usually under "cash flow from operating activities"
  try:
    filing = get_latest_filing(ticker, form_type)
    if filing and filing['xbrl_data']:
      xbrl = filing['xbrl_data']

      depreciation_concepts = [
      'us-gaap:DepreciationDepletionAndAmortization',           # Combined D&A (most common)
      'us-gaap:Depreciation',                                   # Depreciation only
      'us-gaap:DepreciationAndAmortization',                    # Alternative combined
      'us-gaap:DepreciationAmortizationAndAccretionNet',        # With accretion
      'us-gaap:DepreciationDepletionAndAmortizationExcludingAmortizationOfDebtIssuanceCosts',  # Excluding debt costs
      'DepreciationDepletionAndAmortization',                   # Without prefix
      'Depreciation',                                           # Basic depreciation
      'DepreciationAndAmortization'                            # Basic combined
      ]

      d_a_value = 0.0
      d_a_concept = None
      for concept in depreciation_concepts:
        results = filter_annual_data(xbrl, concept, form_type)
        if results:
          d_a_value = float(results['value'])
          d_a_concept = concept
          break

      if d_a_value == 0.0:
        return{
          'error': f"Unable to find concept for {ticker}: Concepts used = {depreciation_concepts}",
          'success': False
        }

      revenue_data = get_revenue_base(ticker, form_type)

      if not revenue_data['success']:
        return revenue_data # just return revenue error

      revenue = revenue_data['revenue_base']  # already in raw dollars

      # now we have the d_a value and revenue so we can calulate deprceication %
      d_a_pct = (d_a_value / revenue) * 100

      return{
        'error': None,
        'success': True,
        'd&a_pct': d_a_pct,
        'concept': d_a_concept,
        'd&a': d_a_value,
        'revenue': revenue
      }

    else:
      return{
        'error': f"Unable to get filing for {ticker}",
        'success': False
      }

  except Exception as e:
    return{
      'error': f"Unable to get filing for {ticker}",
      'success':False
    }


def _get_revenue_from_xbrl(xbrl, form_type: str):
  """Helper: try the two most common revenue concepts; return (value, period_end) or None."""
  for concept in ('us-gaap:Revenues',
                  'us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax',
                  'us-gaap:SalesRevenueNet'):
    d = filter_annual_data(xbrl, concept, form_type)
    if d:
      return d['value'], d['period_end']
  return None


def get_margin_breakdown(ticker: str, form_type: str = '10-K') -> Dict[str, Any]:
  """Extract gross margin, SG&A %, R&D % from the latest filing.

  Returns ticker, revenue, gross_profit, sga, rnd, *_pct_revenue values, plus
  concepts_used for traceability. Banks (no COGS) typically have no GrossProfit
  XBRL concept; absence is expected, not an error.
  """
  try:
    filing_data = get_latest_filing(ticker, form_type)
    if not filing_data or not filing_data.get('xbrl_data'):
      return {'error': f'No filing found for {ticker}', 'success': False}

    xbrl = filing_data['xbrl_data']
    rev_tuple = _get_revenue_from_xbrl(xbrl, form_type)
    if rev_tuple is None:
      return {'ticker': ticker, 'error': 'No revenue concept found', 'success': False}
    revenue, period_end = rev_tuple

    result = {'ticker': ticker, 'revenue': revenue, 'period_end': period_end, 'success': True}
    concepts_used = {}

    for c in ('us-gaap:GrossProfit',):
      gp = filter_annual_data(xbrl, c, form_type)
      if gp:
        result['gross_profit'] = gp['value']
        result['gross_margin_pct'] = (gp['value'] / revenue) * 100
        concepts_used['gross_profit'] = c
        break

    for c in ('us-gaap:SellingGeneralAndAdministrativeExpense',
              'us-gaap:GeneralAndAdministrativeExpense'):
      sga = filter_annual_data(xbrl, c, form_type)
      if sga:
        result['sga'] = sga['value']
        result['sga_pct_revenue'] = (sga['value'] / revenue) * 100
        concepts_used['sga'] = c
        break

    for c in ('us-gaap:ResearchAndDevelopmentExpense',):
      rnd = filter_annual_data(xbrl, c, form_type)
      if rnd:
        result['rnd'] = rnd['value']
        result['rnd_pct_revenue'] = (rnd['value'] / revenue) * 100
        concepts_used['rnd'] = c
        break

    result['concepts_used'] = concepts_used

    if 'gross_profit' not in result:
      print(f"[Validate SEC] {ticker}: gross_profit XBRL concept not found (expected for banks/financials)",
            file=sys.stderr, flush=True)
    return result

  except Exception as e:
    return {'ticker': ticker, 'error': f'get_margin_breakdown failed: {e}', 'success': False}


def get_historical_fcf(ticker: str, form_type: str = '10-K') -> Dict[str, Any]:
  """Extract operating cash flow, capex, and compute FCF and FCF margin from latest filing."""
  try:
    filing_data = get_latest_filing(ticker, form_type)
    if not filing_data or not filing_data.get('xbrl_data'):
      return {'error': f'No filing found for {ticker}', 'success': False}

    xbrl = filing_data['xbrl_data']
    ocf = None
    for c in ('us-gaap:NetCashProvidedByUsedInOperatingActivities',
              'us-gaap:NetCashProvidedByUsedInOperatingActivitiesContinuingOperations'):
      d = filter_annual_data(xbrl, c, form_type)
      if d:
        ocf = d['value']
        break

    capex = None
    for c in ('us-gaap:PaymentsToAcquirePropertyPlantAndEquipment',
              'us-gaap:PaymentsForCapitalImprovements'):
      d = filter_annual_data(xbrl, c, form_type)
      if d:
        capex = abs(d['value'])  # capex usually reported negative on CF statement
        break

    if ocf is None:
      return {'ticker': ticker, 'error': 'OCF concept not found', 'success': False}

    fcf = ocf - (capex or 0)
    rev_tuple = _get_revenue_from_xbrl(xbrl, form_type)
    revenue = rev_tuple[0] if rev_tuple else None
    period_end = rev_tuple[1] if rev_tuple else None

    return {
      'ticker': ticker,
      'operating_cash_flow': ocf,
      'capex': capex,
      'free_cash_flow': fcf,
      'fcf_margin_pct': (fcf / revenue * 100) if revenue else None,
      'period_end': period_end,
      'success': True
    }
  except Exception as e:
    return {'ticker': ticker, 'error': f'get_historical_fcf failed: {e}', 'success': False}


def get_working_capital(ticker: str, form_type: str = '10-K') -> Dict[str, Any]:
  """Extract current assets/liabilities and compute NWC + NWC % of revenue.

  Balance sheet items are XBRL instant facts (point-in-time), not duration facts,
  so this uses filter_instant_data rather than filter_annual_data.
  """
  try:
    filing_data = get_latest_filing(ticker, form_type)
    if not filing_data or not filing_data.get('xbrl_data'):
      return {'error': 'No filing', 'success': False}

    xbrl = filing_data['xbrl_data']
    ca = filter_instant_data(xbrl, 'us-gaap:AssetsCurrent')
    cl = filter_instant_data(xbrl, 'us-gaap:LiabilitiesCurrent')
    ar = filter_instant_data(xbrl, 'us-gaap:AccountsReceivableNetCurrent')
    inv = filter_instant_data(xbrl, 'us-gaap:InventoryNet')
    ap = filter_instant_data(xbrl, 'us-gaap:AccountsPayableCurrent')
    rev_tuple = _get_revenue_from_xbrl(xbrl, form_type)

    if not (ca and cl):
      return {'ticker': ticker, 'error': 'Current assets/liabilities not found', 'success': False}

    nwc = ca['value'] - cl['value']
    revenue = rev_tuple[0] if rev_tuple else None

    return {
      'ticker': ticker,
      'current_assets': ca['value'],
      'current_liabilities': cl['value'],
      'net_working_capital': nwc,
      'nwc_pct_revenue': (nwc / revenue * 100) if revenue else None,
      'accounts_receivable': ar['value'] if ar else None,
      'inventory': inv['value'] if inv else None,
      'accounts_payable': ap['value'] if ap else None,
      'period_end': ca['period_end'],
      'success': True
    }
  except Exception as e:
    return {'ticker': ticker, 'error': f'get_working_capital failed: {e}', 'success': False}


if __name__ == "__main__":
  # Test diverse companies across different industries
  test_companies = [
    "AAPL",  # Tech/Manufacturing (Apple)
    "GOOGL", # Tech/Services (Google)
    "JPM",   # Banking (JPMorgan Chase)
    "JNJ",   # Healthcare/Pharma (Johnson & Johnson)
    "WMT",   # Retail (Walmart)
    "XOM",   # Energy (ExxonMobil)
    "BAC",   # Banking (Bank of America)
    "MSFT"   # Tech/Software (Microsoft)
  ]

  # Test SEC form type support
  print("Testing SEC Form Type Support:")
  print("=" * 60)

  test_ticker = "AAPL"
  forms_to_test = ['10-K', '10-Q', '8-K', 'S-1', 'DEF 14A', '13F']

  for form in forms_to_test:
    try:
      filing_data = get_latest_filing(test_ticker, form)
      if filing_data:
        print(f"✓ {form}: SUCCESS - Found filing dated {filing_data['filing_date']}")
        # Check if XBRL is available
        if filing_data['xbrl_data']:
          print(f"  XBRL: Available")
        else:
          print(f"  XBRL: Not available")
      else:
        print(f"✗ {form}: No filings found")
    except Exception as e:
      print(f"✗ {form}: ERROR - {str(e)}")
    print("-" * 40)

  print("Testing Different Form Types - Revenue Comparison:")
  print("=" * 60)

  test_ticker = "AAPL"
  form_types = ['10-K', '10-Q']

  for form_type in form_types:
    print(f"\n{form_type} DATA:")
    print("-" * 30)

    try:
      # Test revenue
      revenue_result = get_revenue_base(test_ticker, form_type)
      if revenue_result['success']:
        print(f"Revenue: ${revenue_result['revenue_base']/1e9:.1f}B")
        print(f"Period End: {revenue_result['period_end']}")
        print(f"Concept: {revenue_result['concept_used']}")

      # Test EBITDA
      ebitda_result = get_ebitda_margin(test_ticker, form_type)
      if ebitda_result['success']:
        print(f"EBITDA Margin: {ebitda_result['ebitda_margin_percent']:.2f}%")
        print(f"EBITDA Amount: ${ebitda_result['ebitda_amount']/1e9:.1f}B")

      # Test CapEx
      capex_result = get_capex_pct_revenue(test_ticker, form_type)
      if capex_result['success']:
        print(f"CapEx % of Revenue: {capex_result['capex_pct_revenue']:.2f}%")
        print(f"Total CapEx: ${capex_result['total_capex']/1e9:.2f}B")

      # Test Tax Rate
      tax_result = get_tax_rate(test_ticker, form_type)
      if tax_result['success']:
        print(f"Effective Tax Rate: {tax_result['effective_tax_rate']:.2f}%")

    except Exception as e:
      print(f"ERROR with {form_type}: {str(e)}")

  print("\n" + "=" * 60)
  print("Testing 8-K, DEF 14A Disclosure Data:")
  print("=" * 60)

  test_ticker = "AAPL"
  special_forms = ['8-K', 'DEF 14A']

  for form_type in special_forms:
    print(f"\n{form_type} DISCLOSURES:")
    print("-" * 40)

    try:
      # Get disclosure names first
      disclosures_result = get_disclosures_names(test_ticker, form_type)
      if disclosures_result['success']:
        print(f"Found {len(disclosures_result['disclosure_names'])} disclosures:")
        for i, disclosure in enumerate(disclosures_result['disclosure_names'][:5]):  # Show first 5
          print(f"  {i+1}. {disclosure}")

        # Try to extract data from first disclosure
        if disclosures_result['disclosure_names']:
          first_disclosure = disclosures_result['disclosure_names'][0]
          print(f"\nExtracting data from: {first_disclosure}")
          disclosure_data = extract_disclosure_data(test_ticker, first_disclosure, form_type)
          if 'clean_text' in str(disclosure_data):
            print("Found text-based disclosure data")
          elif 'sample_data' in str(disclosure_data):
            print("Found structured disclosure data")
          else:
            print("No structured data found")
      else:
        print(f"Error getting disclosures: {disclosures_result['error']}")

    except Exception as e:
      print(f"ERROR with {form_type}: {str(e)}")

  print("\n" + "=" * 60)
  print("Investigating Filing Structure and Content:")
  print("=" * 60)

  test_ticker = "AAPL"
  investigation_forms = ['8-K', 'DEF 14A']

  for form_type in investigation_forms:
    print(f"\n{form_type} FILING STRUCTURE:")
    print("-" * 50)

    try:
      # Get the raw filing object
      filing_data = get_latest_filing(test_ticker, form_type)
      if filing_data:
        filing = filing_data['filing_object']
        print(f"Filing Date: {filing_data['filing_date']}")
        print(f"Accession Number: {filing_data['accession_number']}")
        print(f"URL: {filing_data['url']}")

        # Check what attributes the filing object has
        print(f"\nFiling Object Attributes:")
        attrs = [attr for attr in dir(filing) if not attr.startswith('_')]
        for attr in attrs[:10]:  # Show first 10 attributes
          print(f"  - {attr}")

        # Try to get the actual document content
        try:
          # Check if filing has documents
          if hasattr(filing, 'documents'):
            docs = filing.documents
            print(f"\nNumber of Documents: {len(docs) if docs else 'None'}")
            if docs:
              for i, doc in enumerate(docs[:3]):  # Show first 3 docs
                print(f"  Doc {i+1}: {doc.document if hasattr(doc, 'document') else 'Unknown'}")

          # Check if filing has html content
          if hasattr(filing, 'html'):
            html_content = filing.html()
            print(f"\nHTML Content Length: {len(html_content)} characters")
            print(f"HTML Preview (first 500 chars):\n{html_content[:500]}...")

          # Check if filing has text content
          if hasattr(filing, 'text'):
            text_content = filing.text()
            print(f"\nText Content Length: {len(text_content)} characters")
            print(f"Text Preview (first 500 chars):\n{text_content[:500]}...")

        except Exception as e:
          print(f"Error accessing content: {e}")

        # For DEF 14A, check for tables
        if form_type == 'DEF 14A':
          try:
            if hasattr(filing, 'tables'):
              tables = filing.tables()
              print(f"\nTables found: {len(tables) if tables else 0}")
              if tables:
                for i, table in enumerate(tables[:2]):  # Show first 2 tables
                  print(f"  Table {i+1} shape: {table.shape if hasattr(table, 'shape') else 'Unknown'}")
          except Exception as e:
            print(f"Error accessing tables: {e}")

        # Check XBRL structure
        if filing_data['xbrl_data']:
          xbrl = filing_data['xbrl_data']
          print(f"\nXBRL Structure:")

          # Check statements
          if hasattr(xbrl, 'statements'):
            statements = xbrl.statements
            print(f"  Statements available: {len(statements) if statements else 0}")

          # Check facts
          if hasattr(xbrl, 'facts'):
            facts = xbrl.facts
            print(f"  Facts available: {len(facts) if hasattr(facts, '__len__') else 'Unknown'}")

          # Check concepts
          if hasattr(xbrl, 'concepts'):
            concepts = xbrl.concepts
            print(f"  Concepts available: {len(concepts) if hasattr(concepts, '__len__') else 'Unknown'}")

      else:
        print(f"No {form_type} filing found")

    except Exception as e:
      print(f"ERROR investigating {form_type}: {str(e)}")

  print("\n" + "=" * 60)
  print("Testing EBITDA margins across different industries:")
  print("=" * 60)

  for ticker in test_companies:
    try:
      result = get_ebitda_margin(ticker, '10-K')
      if result['success']:
        print(f"{ticker}: {result['ebitda_margin_percent']:.2f}% EBITDA margin "
              f"(Revenue: ${result['revenue']/1e9:.1f}B, Concept: {result['operating_income_concept_used']})")
      else:
        print(f"{ticker}: ERROR - {result['error']}")
    except Exception as e:
      print(f"{ticker}: EXCEPTION - {str(e)}")
    print("-" * 50)
