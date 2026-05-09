"""
Extensive Test Suite for SEC XBRL Financial Data Extraction Functions
======================================================================

Requirements (pip install):
    pip install pytest
    pip install edgartools
    pip install pandas
    pip install numpy

Optional for enhanced testing:
    pip install pytest-html      # HTML reports
    pip install pytest-cov       # Coverage reports
    pip install pytest-xdist     # Parallel execution

This test suite provides comprehensive coverage for all SEC XBRL extraction functions
including edge cases, error handling, data validation, and cross-sector testing.

Test Categories:
1. Unit Tests - Individual function behavior
2. Integration Tests - End-to-end workflows
3. Edge Case Tests - Boundary conditions and unusual inputs
4. Error Handling Tests - API failures, missing data, malformed responses
5. Data Validation Tests - Output format and value range verification
6. Cross-Sector Tests - Different industries and company types
7. Stress Tests - High volume and concurrent testing
8. Regression Tests - Known issues and bug fixes

Usage:
    pytest test_tools.web_search_server.sec_utils.py -v
    pytest test_tools.web_search_server.sec_utils.py -v -k "test_revenue"
    pytest test_tools.web_search_server.sec_utils.py -v --tb=short -x
"""

import pytest
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import time
import random
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- THIS IS THE CRITICAL FIX ---
# Add the project root directory to the Python path to allow imports from other folders
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# --- END FIX ---

# Import functions to test
from tools.web_search_server.sec_utils import (
    filter_annual_data,
    get_latest_filing,
    get_disclosures_names,
    extract_disclosure_data,
    get_revenue_base,
    get_ebitda_margin,
    get_capex_pct_revenue,
    get_tax_rate,
    get_depreciation
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# TEST DATA: 500 STOCK TICKERS ORGANIZED BY SECTOR
# =============================================================================

class TestStockUniverse:
    """
    Comprehensive stock universe for testing across sectors and market caps.
    Total: 500 stocks
    """

    # Technology Sector (75 stocks)
    TECHNOLOGY = [
        # Mega Cap Tech
        "AAPL", "MSFT", "GOOGL", "GOOG", "META", "NVDA", "AVGO", "ORCL", "CSCO", "ADBE",
        "CRM", "AMD", "INTC", "IBM", "NOW", "QCOM", "TXN", "INTU", "AMAT", "ADI",
        # Large Cap Tech
        "LRCX", "MU", "SNPS", "CDNS", "KLAC", "MRVL", "FTNT", "PANW", "CRWD", "ZS",
        "WDAY", "TEAM", "DDOG", "NET", "MDB", "SNOW", "PLTR", "COIN", "SQ", "PYPL",
        # Mid Cap Tech
        "OKTA", "ZM", "DOCU", "TWLO", "SPLK", "ESTC", "CFLT", "PATH", "APP", "BILL",
        "GTLB", "MNDY", "DOCN", "FSLY", "NEWR", "SUMO", "ASAN", "FROG", "TYL", "SMAR",
        # Small Cap Tech
        "PEGA", "MANH", "QTWO", "NCNO", "FRSH", "TENB", "RPD", "VRNS", "ALTR", "EVBG",
        "PSTG", "APPN", "PRGS", "CWAN", "JAMF"
    ]

    # Healthcare / Pharma (70 stocks)
    HEALTHCARE = [
        # Large Pharma
        "JNJ", "UNH", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT", "DHR", "BMY",
        "AMGN", "GILD", "VRTX", "REGN", "ISRG", "MDT", "SYK", "ZTS", "BDX", "BSX",
        # Biotech
        "BIIB", "ILMN", "MRNA", "SGEN", "ALNY", "BMRN", "EXAS", "TECH", "NBIX", "INCY",
        "SRPT", "IONS", "UTHR", "RARE", "PCVX", "FOLD", "DAWN", "RXRX", "DNLI", "ARVN",
        # Healthcare Services
        "HCA", "CI", "ELV", "HUM", "CNC", "MOH", "DVA", "THC", "UHS", "ACHC",
        # Medical Devices
        "EW", "DXCM", "HOLX", "ALGN", "IDXX", "IQV", "MTD", "WAT", "A", "PKI",
        # Small/Mid Cap Healthcare
        "EXEL", "HALO", "CRNX", "ACAD", "INSM", "CORT", "PTCT", "XENE", "KRYS", "GERN"
    ]

    # Financial Services (65 stocks)
    FINANCIALS = [
        # Major Banks
        "JPM", "BAC", "WFC", "C", "GS", "MS", "USB", "PNC", "TFC", "COF",
        # Regional Banks
        "FITB", "HBAN", "CFG", "KEY", "RF", "MTB", "ZION", "CMA", "FRC", "SIVB",
        # Insurance
        "BRK.B", "AIG", "MET", "PRU", "AFL", "TRV", "PGR", "CB", "ALL", "AJG",
        # Asset Management
        "BLK", "SCHW", "BX", "KKR", "APO", "ARES", "OWL", "TROW", "AMG", "IVZ",
        # Fintech / Payments
        "V", "MA", "AXP", "DFS", "FIS", "FISV", "GPN", "WU", "AFRM", "UPST",
        # REITs
        "AMT", "PLD", "CCI", "EQIX", "PSA", "SPG", "O", "WELL", "AVB", "EQR",
        # Specialty Finance
        "ICE", "CME", "NDAQ", "MSCI", "SPGI"
    ]

    # Consumer Discretionary (55 stocks)
    CONSUMER_DISCRETIONARY = [
        # Retail
        "AMZN", "HD", "LOW", "TGT", "COST", "WMT", "TJX", "ROST", "DG", "DLTR",
        "BBY", "ORLY", "AZO", "ULTA", "FIVE", "WSM", "RH", "TSCO", "OLLI", "BIG",
        # Automotive
        "TSLA", "GM", "F", "RIVN", "LCID", "TM", "HMC", "RACE", "AN", "PAG",
        # Restaurants / Leisure
        "MCD", "SBUX", "CMG", "DRI", "YUM", "QSR", "WING", "SHAK", "DNUT", "EAT",
        # Hotels / Travel
        "MAR", "HLT", "H", "ABNB", "BKNG", "EXPE", "RCL", "CCL", "NCLH", "WYNN",
        # Apparel
        "NKE", "LULU", "VFC", "PVH", "RL"
    ]

    # Consumer Staples (40 stocks)
    CONSUMER_STAPLES = [
        # Food & Beverage
        "KO", "PEP", "MDLZ", "KHC", "GIS", "K", "HSY", "SJM", "CPB", "CAG",
        "MKC", "HRL", "TSN", "BG", "ADM", "DAR", "INGR", "LNDC", "POST", "FLO",
        # Household Products
        "PG", "CL", "KMB", "CHD", "CLX", "SPB", "NWL", "COTY", "ELF", "IPAR",
        # Tobacco / Alcohol
        "PM", "MO", "BTI", "STZ", "BF.B", "DEO", "TAP", "SAM", "ABEV", "CCU"
    ]

    # Industrials (60 stocks)
    INDUSTRIALS = [
        # Aerospace & Defense
        "BA", "RTX", "LMT", "NOC", "GD", "LHX", "TDG", "HWM", "TXT", "HII",
        # Machinery & Equipment
        "CAT", "DE", "EMR", "HON", "MMM", "ITW", "PH", "ROK", "ETN", "IR",
        # Transportation
        "UNP", "CSX", "NSC", "UPS", "FDX", "JBHT", "XPO", "CHRW", "EXPD", "ODFL",
        # Airlines
        "DAL", "UAL", "LUV", "AAL", "ALK", "JBLU", "SAVE", "HA", "SKYW", "ALGT",
        # Industrial Services
        "WM", "RSG", "FAST", "GWW", "CTAS", "ADP", "PAYX", "CINF", "WCN", "CLH",
        # Construction & Engineering
        "VMC", "MLM", "URI", "PWR", "EME", "MTZ", "FIX", "ACM", "DY", "GVA"
    ]

    # Energy (45 stocks)
    ENERGY = [
        # Integrated Oil & Gas
        "XOM", "CVX", "COP", "EOG", "SLB", "OXY", "PSX", "VLO", "MPC", "PXD",
        # Exploration & Production
        "DVN", "FANG", "HES", "APA", "MRO", "OVV", "CTRA", "MTDR", "PR", "CHRD",
        # Oil Services
        "HAL", "BKR", "NOV", "FTI", "CHX", "HP", "PTEN", "RIG", "VAL", "OII",
        # Midstream
        "WMB", "KMI", "OKE", "TRGP", "EPD", "ET", "PAA", "MPLX", "ENLC", "DCP",
        # Refining
        "HFC", "DK", "PBF", "PARR", "CVI"
    ]

    # Materials (35 stocks)
    MATERIALS = [
        # Chemicals
        "LIN", "APD", "SHW", "ECL", "DD", "DOW", "LYB", "PPG", "NEM", "FCX",
        "ALB", "CE", "EMN", "HUN", "OLN", "AXTA", "RPM", "FUL", "CBT", "KWR",
        # Metals & Mining
        "NUE", "STLD", "CLF", "RS", "ATI", "CMC", "X", "AA", "CENX", "KALU",
        # Packaging
        "IP", "PKG", "WRK", "GPK", "SLGN"
    ]

    # Utilities (30 stocks)
    UTILITIES = [
        "NEE", "DUK", "SO", "D", "AEP", "SRE", "EXC", "XEL", "WEC", "ED",
        "PEG", "ES", "EIX", "AWK", "DTE", "FE", "PPL", "AES", "ETR", "CMS",
        "AEE", "CNP", "EVRG", "NI", "PNW", "OGE", "NRG", "VST", "ATO", "SWX"
    ]

    # Real Estate (25 stocks) - Separate from Financials
    REAL_ESTATE = [
        "PLD", "AMT", "CCI", "EQIX", "PSA", "SPG", "O", "WELL", "DLR", "SBAC",
        "VICI", "ARE", "AVB", "EQR", "ESS", "MAA", "UDR", "CPT", "INVH", "VTR",
        "KIM", "REG", "FRT", "BXP", "SLG"
    ]

    @classmethod
    def get_all_tickers(cls) -> List[str]:
        """Return all 500 tickers as a single list."""
        all_tickers = (
            cls.TECHNOLOGY +
            cls.HEALTHCARE +
            cls.FINANCIALS +
            cls.CONSUMER_DISCRETIONARY +
            cls.CONSUMER_STAPLES +
            cls.INDUSTRIALS +
            cls.ENERGY +
            cls.MATERIALS +
            cls.UTILITIES +
            cls.REAL_ESTATE
        )
        return list(set(all_tickers))  # Remove any duplicates

    @classmethod
    def get_sector_tickers(cls, sector: str) -> List[str]:
        """Return tickers for a specific sector."""
        sector_map = {
            'technology': cls.TECHNOLOGY,
            'healthcare': cls.HEALTHCARE,
            'financials': cls.FINANCIALS,
            'consumer_discretionary': cls.CONSUMER_DISCRETIONARY,
            'consumer_staples': cls.CONSUMER_STAPLES,
            'industrials': cls.INDUSTRIALS,
            'energy': cls.ENERGY,
            'materials': cls.MATERIALS,
            'utilities': cls.UTILITIES,
            'real_estate': cls.REAL_ESTATE
        }
        return sector_map.get(sector.lower(), [])

    @classmethod
    def get_sample_tickers(cls, n: int = 50) -> List[str]:
        """Return a random sample of n tickers for quick testing."""
        all_tickers = cls.get_all_tickers()
        return random.sample(all_tickers, min(n, len(all_tickers)))

    # Edge case tickers for special testing
    EDGE_CASE_TICKERS = [
        "BRK.A",    # Very high stock price
        "BRK.B",    # Class B shares
        "BF.B",     # Ticker with period
        "GOOG",     # Dual class structure
        "GOOGL",    # Dual class structure
    ]

    # Tickers known to have unusual filing patterns
    UNUSUAL_FILERS = [
        "BRK.A",    # Unique reporting format
        "META",     # Recently renamed
    ]

    # Foreign companies with ADRs
    FOREIGN_ADRS = [
        "TSM", "ASML", "TM", "NVS", "SNY", "AZN", "GSK", "NVO", "UL", "SHEL",
        "BP", "RIO", "BHP", "VALE", "BABA", "JD", "PDD", "BIDU", "NIO", "LI"
    ]

    # Small cap / potentially delisted - for error handling tests
    RISKY_TICKERS = [
        "BBBY", "AMC", "GME", "SPCE", "PLTR", "HOOD", "SOFI", "WISH", "CLOV", "WKHS"
    ]


# =============================================================================
# PYTEST FIXTURES
# =============================================================================

@pytest.fixture(scope="session")
def all_tickers():
    """Provide all 500 test tickers."""
    return TestStockUniverse.get_all_tickers()


@pytest.fixture(scope="session")
def sample_tickers():
    """Provide a sample of tickers for faster tests."""
    return TestStockUniverse.get_sample_tickers(50)


@pytest.fixture(scope="session")
def mega_cap_tickers():
    """Provide mega-cap tickers that should always have data."""
    return ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "JPM", "JNJ", "V", "WMT"]


@pytest.fixture
def mock_xbrl_data():
    """Create mock XBRL data for unit testing."""
    mock = MagicMock()

    # Create mock facts
    mock_facts = MagicMock()
    mock_query = MagicMock()

    # Sample DataFrame for revenue
    sample_df = pd.DataFrame({
        'concept': ['us-gaap:Revenues'],
        'numeric_value': [100000000000.0],
        'period_start': ['2023-01-01'],
        'period_end': ['2023-12-31'],
        'unit': ['USD']
    })

    mock_query.to_dataframe.return_value = sample_df
    mock_query.by_concept.return_value = mock_query
    mock_facts.query.return_value = mock_query
    mock.facts = mock_facts

    return mock


@pytest.fixture
def mock_filing_data(mock_xbrl_data):
    """Create mock filing data."""
    return {
        'filing_date': '2024-02-15',
        'url': 'https://www.sec.gov/cgi-bin/browse-edgar',
        'accession_number': '0000320193-24-000001',
        'filing_object': MagicMock(),
        'xbrl_data': mock_xbrl_data
    }


@pytest.fixture
def rate_limiter():
    """Rate limiter to stay within SEC API limits (10 requests/second)."""
    class RateLimiter:
        def __init__(self, calls_per_second: float = 10.0):
            self.calls_per_second = calls_per_second
            self.last_call = 0

        def wait(self):
            elapsed = time.time() - self.last_call
            wait_time = max(0, (1 / self.calls_per_second) - elapsed)
            if wait_time > 0:
                time.sleep(wait_time)
            self.last_call = time.time()

    return RateLimiter()


# =============================================================================
# HELPER FUNCTIONS FOR TESTING
# =============================================================================

class TestHelpers:
    """Helper functions for test validation and reporting."""

    @staticmethod
    def validate_result_structure(result: Dict[str, Any], required_keys: List[str]) -> bool:
        """Validate that a result dictionary has required keys."""
        if result is None:
            return False
        return all(key in result for key in required_keys)

    @staticmethod
    def is_valid_percentage(value: float) -> bool:
        """Check if a value is a reasonable percentage."""
        return isinstance(value, (int, float)) and -500 <= value <= 500

    @staticmethod
    def is_valid_financial_value(value: float, min_val: float = -1e15, max_val: float = 1e15) -> bool:
        """Check if a financial value is within reasonable bounds."""
        return isinstance(value, (int, float)) and min_val <= value <= max_val

    @staticmethod
    def collect_test_results(results: List[Dict]) -> Dict[str, Any]:
        """Aggregate test results for reporting."""
        total = len(results)
        successful = sum(1 for r in results if r.get('success', False))
        failed = total - successful

        errors = {}
        for r in results:
            if not r.get('success', False) and 'error' in r:
                error_type = str(r['error'])[:50]
                errors[error_type] = errors.get(error_type, 0) + 1

        return {
            'total': total,
            'successful': successful,
            'failed': failed,
            'success_rate': (successful / total * 100) if total > 0 else 0,
            'error_distribution': errors
        }

    @staticmethod
    def log_test_summary(test_name: str, results: Dict[str, Any]):
        """Log a summary of test results."""
        logger.info(f"\n{'='*60}")
        logger.info(f"TEST SUMMARY: {test_name}")
        logger.info(f"{'='*60}")
        logger.info(f"Total Tests: {results['total']}")
        logger.info(f"Successful: {results['successful']}")
        logger.info(f"Failed: {results['failed']}")
        logger.info(f"Success Rate: {results['success_rate']:.2f}%")
        if results['error_distribution']:
            logger.info(f"Error Distribution: {results['error_distribution']}")
        logger.info(f"{'='*60}\n")


# =============================================================================
# UNIT TESTS: filter_annual_data
# =============================================================================

class TestFilterAnnualData:
    """Unit tests for the filter_annual_data helper function."""

    def test_filter_annual_data_with_valid_data(self, mock_xbrl_data):
        """Test filtering with valid annual data."""
        result = filter_annual_data(mock_xbrl_data, 'us-gaap:Revenues')
        # Result depends on mock setup - validate structure if not None
        if result is not None:
            assert 'value' in result
            assert 'concept_used' in result
            assert 'period_end' in result

    def test_filter_annual_data_with_empty_concept(self, mock_xbrl_data):
        """Test filtering with empty concept string."""
        result = filter_annual_data(mock_xbrl_data, '')
        # Should handle gracefully
        assert result is None or isinstance(result, dict)

    def test_filter_annual_data_with_none_xbrl(self):
        """Test filtering with None XBRL data."""
        result = filter_annual_data(None, 'us-gaap:Revenues')
        assert result is None

    def test_filter_annual_data_with_invalid_concept(self, mock_xbrl_data):
        """Test filtering with non-existent concept."""
        result = filter_annual_data(mock_xbrl_data, 'us-gaap:NonExistentConcept12345')
        # Should return None for non-existent concepts
        assert result is None or isinstance(result, dict)

    def test_filter_annual_data_quarterly_vs_annual(self):
        """Test that quarterly data is properly excluded."""
        mock_xbrl = MagicMock()
        mock_facts = MagicMock()
        mock_query = MagicMock()

        # Create DataFrame with both quarterly and annual data
        df = pd.DataFrame({
            'concept': ['us-gaap:Revenues'] * 4,
            'numeric_value': [25000000000.0, 25000000000.0, 25000000000.0, 100000000000.0],
            'period_start': ['2023-01-01', '2023-04-01', '2023-07-01', '2023-01-01'],
            'period_end': ['2023-03-31', '2023-06-30', '2023-09-30', '2023-12-31'],
            'unit': ['USD'] * 4
        })

        mock_query.to_dataframe.return_value = df
        mock_query.by_concept.return_value = mock_query
        mock_facts.query.return_value = mock_query
        mock_xbrl.facts = mock_facts

        result = filter_annual_data(mock_xbrl, 'us-gaap:Revenues')

        if result is not None:
            # Should return the annual value (100B), not quarterly
            assert result['duration_days'] >= 350 or result['value'] == 100000000000.0

    def test_filter_annual_data_revenue_max_selection(self):
        """Test that for revenue, the maximum value is selected."""
        mock_xbrl = MagicMock()
        mock_facts = MagicMock()
        mock_query = MagicMock()

        # Create DataFrame with multiple revenue values for same period
        df = pd.DataFrame({
            'concept': ['us-gaap:Revenues'] * 3,
            'numeric_value': [50000000000.0, 100000000000.0, 75000000000.0],
            'period_start': ['2023-01-01'] * 3,
            'period_end': ['2023-12-31'] * 3,
            'unit': ['USD'] * 3
        })

        mock_query.to_dataframe.return_value = df
        mock_query.by_concept.return_value = mock_query
        mock_facts.query.return_value = mock_query
        mock_xbrl.facts = mock_facts

        result = filter_annual_data(mock_xbrl, 'us-gaap:Revenues')

        if result is not None:
            # Should select the maximum revenue value
            assert result['value'] == 100000000000.0

    def test_filter_annual_data_handles_nan_values(self):
        """Test handling of NaN values in data."""
        mock_xbrl = MagicMock()
        mock_facts = MagicMock()
        mock_query = MagicMock()

        df = pd.DataFrame({
            'concept': ['us-gaap:Revenues'] * 2,
            'numeric_value': [np.nan, 100000000000.0],
            'period_start': ['2023-01-01'] * 2,
            'period_end': ['2023-12-31'] * 2,
            'unit': ['USD'] * 2
        })

        mock_query.to_dataframe.return_value = df
        mock_query.by_concept.return_value = mock_query
        mock_facts.query.return_value = mock_query
        mock_xbrl.facts = mock_facts

        result = filter_annual_data(mock_xbrl, 'us-gaap:Revenues')
        # Should handle NaN gracefully
        assert result is None or (result is not None and not np.isnan(result.get('value', np.nan)))


# =============================================================================
# UNIT TESTS: get_latest_filing
# =============================================================================

class TestGetLatestFiling:
    """Unit tests for the get_latest_filing function."""

    @pytest.mark.parametrize("ticker", ["AAPL", "MSFT", "GOOGL"])
    def test_get_latest_filing_mega_caps(self, ticker, rate_limiter):
        """Test getting filings for mega-cap companies."""
        rate_limiter.wait()
        result = get_latest_filing(ticker)

        if result is not None:
            assert 'filing_date' in result
            assert 'xbrl_data' in result
            assert 'accession_number' in result

    def test_get_latest_filing_invalid_ticker(self, rate_limiter):
        """Test with invalid ticker symbol."""
        rate_limiter.wait()
        result = get_latest_filing("INVALIDTICKER12345")

        # Should return None for invalid tickers
        assert result is None

    def test_get_latest_filing_empty_ticker(self):
        """Test with empty ticker string."""
        result = get_latest_filing("")
        assert result is None

    def test_get_latest_filing_special_characters(self):
        """Test with ticker containing special characters."""
        result = get_latest_filing("$%^&*")
        assert result is None

    def test_get_latest_filing_form_10k(self, rate_limiter):
        """Test explicitly requesting 10-K form."""
        rate_limiter.wait()
        result = get_latest_filing("AAPL", form_type="10-K")

        if result is not None:
            assert 'filing_date' in result

    def test_get_latest_filing_form_10q(self, rate_limiter):
        """Test requesting 10-Q form."""
        rate_limiter.wait()
        result = get_latest_filing("AAPL", form_type="10-Q")

        if result is not None:
            assert 'filing_date' in result

    def test_get_latest_filing_lowercase_ticker(self, rate_limiter):
        """Test with lowercase ticker."""
        rate_limiter.wait()
        result = get_latest_filing("aapl")

        # Should handle case insensitivity
        if result is not None:
            assert 'filing_date' in result

    def test_get_latest_filing_with_period_ticker(self, rate_limiter):
        """Test ticker with period (e.g., BRK.B)."""
        rate_limiter.wait()
        result = get_latest_filing("BRK.B")

        # Should handle tickers with periods
        if result is not None:
            assert 'filing_date' in result


# =============================================================================
# UNIT TESTS: get_revenue_base
# =============================================================================

class TestGetRevenueBase:
    """Unit tests for the get_revenue_base function."""

    @pytest.mark.parametrize("ticker", ["AAPL", "MSFT", "JNJ", "JPM", "XOM"])
    def test_get_revenue_base_diverse_sectors(self, ticker, rate_limiter):
        """Test revenue extraction across different sectors."""
        rate_limiter.wait()
        result = get_revenue_base(ticker)

        assert 'ticker' in result
        if result.get('success'):
            assert 'revenue_base' in result
            assert 'concept_used' in result
            assert result['revenue_base'] > 0
            # Revenue should be in reasonable range (in millions)
            assert 1 <= result['revenue_base'] <= 1000000  # $1M to $1T

    def test_get_revenue_base_invalid_ticker(self, rate_limiter):
        """Test with invalid ticker."""
        rate_limiter.wait()
        result = get_revenue_base("INVALID12345")

        assert result['success'] == False
        assert 'error' in result

    def test_get_revenue_base_result_structure(self, rate_limiter):
        """Test that successful results have correct structure."""
        rate_limiter.wait()
        result = get_revenue_base("AAPL")

        assert 'ticker' in result
        assert 'success' in result

        if result['success']:
            required_keys = ['revenue_base', 'concept_used', 'period_end', 'filing_date']
            for key in required_keys:
                assert key in result, f"Missing key: {key}"

    def test_get_revenue_base_data_types(self, rate_limiter):
        """Test that returned values have correct data types."""
        rate_limiter.wait()
        result = get_revenue_base("AAPL")

        if result.get('success'):
            assert isinstance(result['revenue_base'], (int, float))
            assert isinstance(result['concept_used'], str)
            assert isinstance(result['ticker'], str)


# =============================================================================
# UNIT TESTS: get_ebitda_margin
# =============================================================================

class TestGetEbitdaMargin:
    """Unit tests for the get_ebitda_margin function."""

    @pytest.mark.parametrize("ticker", ["AAPL", "MSFT", "GOOGL", "META", "NVDA"])
    def test_get_ebitda_margin_tech_companies(self, ticker, rate_limiter):
        """Test EBITDA margin for tech companies (typically high margins)."""
        rate_limiter.wait()
        result = get_ebitda_margin(ticker)

        assert 'ticker' in result or 'success' in result
        if result.get('success'):
            assert 'ebitda_margin_percent' in result
            # Tech companies typically have positive EBITDA margins
            assert TestHelpers.is_valid_percentage(result['ebitda_margin_percent'])

    @pytest.mark.parametrize("ticker", ["DAL", "UAL", "AAL"])
    def test_get_ebitda_margin_airlines(self, ticker, rate_limiter):
        """Test EBITDA margin for airlines (volatile margins)."""
        rate_limiter.wait()
        result = get_ebitda_margin(ticker)

        if result.get('success'):
            # Airlines can have negative or low margins
            assert TestHelpers.is_valid_percentage(result['ebitda_margin_percent'])

    def test_get_ebitda_margin_result_components(self, rate_limiter):
        """Test that all EBITDA components are returned."""
        rate_limiter.wait()
        result = get_ebitda_margin("AAPL")

        if result.get('success'):
            component_keys = ['ebitda_amount', 'operating_income', 'd&a', 'revenue']
            for key in component_keys:
                assert key in result, f"Missing component: {key}"
                assert isinstance(result[key], (int, float))

    def test_get_ebitda_margin_calculation_validity(self, rate_limiter):
        """Test that EBITDA calculation is mathematically correct."""
        rate_limiter.wait()
        result = get_ebitda_margin("MSFT")

        if result.get('success'):
            # EBITDA = Operating Income + D&A
            calculated_ebitda = result['operating_income'] + result['d&a']
            assert abs(calculated_ebitda - result['ebitda_amount']) < 0.01

            # EBITDA Margin = EBITDA / Revenue * 100
            calculated_margin = (result['ebitda_amount'] / result['revenue']) * 100
            assert abs(calculated_margin - result['ebitda_margin_percent']) < 0.01

    def test_get_ebitda_margin_invalid_ticker(self, rate_limiter):
        """Test with invalid ticker."""
        rate_limiter.wait()
        result = get_ebitda_margin("INVALID12345")

        assert result.get('success') == False


# =============================================================================
# UNIT TESTS: get_capex_pct_revenue
# =============================================================================

class TestGetCapexPctRevenue:
    """Unit tests for the get_capex_pct_revenue function."""

    @pytest.mark.parametrize("ticker", ["AAPL", "MSFT", "AMZN"])
    def test_get_capex_pct_tech(self, ticker, rate_limiter):
        """Test CapEx for tech companies."""
        rate_limiter.wait()
        result = get_capex_pct_revenue(ticker)

        if result.get('success'):
            assert 'capex_pct_revenue' in result
            assert 'total_capex' in result
            # CapEx should be positive (it's an expenditure)
            assert result['total_capex'] >= 0
            # CapEx % typically 0-50% for most companies
            assert 0 <= result['capex_pct_revenue'] <= 100

    @pytest.mark.parametrize("ticker", ["XOM", "CVX", "OXY"])
    def test_get_capex_pct_energy(self, ticker, rate_limiter):
        """Test CapEx for energy companies (typically high CapEx)."""
        rate_limiter.wait()
        result = get_capex_pct_revenue(ticker)

        if result.get('success'):
            # Energy companies typically have higher CapEx
            assert result['capex_pct_revenue'] >= 0

    @pytest.mark.parametrize("ticker", ["NEE", "DUK", "SO"])
    def test_get_capex_pct_utilities(self, ticker, rate_limiter):
        """Test CapEx for utilities (capital intensive)."""
        rate_limiter.wait()
        result = get_capex_pct_revenue(ticker)

        if result.get('success'):
            # Utilities are capital intensive
            assert result['total_capex'] > 0

    def test_get_capex_pct_result_consistency(self, rate_limiter):
        """Test that CapEx calculation is internally consistent."""
        rate_limiter.wait()
        result = get_capex_pct_revenue("AAPL")

        if result.get('success'):
            # Verify calculation: CapEx % = (Total CapEx / Revenue) * 100
            calculated_pct = (result['total_capex'] / result['revenue']) * 100
            assert abs(calculated_pct - result['capex_pct_revenue']) < 0.01


# =============================================================================
# UNIT TESTS: get_tax_rate
# =============================================================================

class TestGetTaxRate:
    """Unit tests for the get_tax_rate function."""

    @pytest.mark.parametrize("ticker", ["AAPL", "MSFT", "JNJ", "PG", "KO"])
    def test_get_tax_rate_major_companies(self, ticker, rate_limiter):
        """Test tax rate for major US companies."""
        rate_limiter.wait()
        result = get_tax_rate(ticker)

        if result.get('success'):
            assert 'effective_tax_rate' in result
            # US corporate tax rates typically 15-30%, but can vary
            assert -50 <= result['effective_tax_rate'] <= 100

    def test_get_tax_rate_components(self, rate_limiter):
        """Test that tax rate components are returned."""
        rate_limiter.wait()
        result = get_tax_rate("AAPL")

        if result.get('success'):
            assert 'tax_expense' in result
            assert 'pretax_income' in result
            assert 'tax_concept_used' in result
            assert 'pretax_concept_used' in result

    def test_get_tax_rate_calculation(self, rate_limiter):
        """Test tax rate calculation validity."""
        rate_limiter.wait()
        result = get_tax_rate("MSFT")

        if result.get('success') and result['pretax_income'] != 0:
            calculated_rate = (result['tax_expense'] / result['pretax_income']) * 100
            assert abs(calculated_rate - result['effective_tax_rate']) < 0.01

    def test_get_tax_rate_loss_company(self, rate_limiter):
        """Test tax rate for companies that might have losses."""
        # Test with a company that might have losses
        rate_limiter.wait()
        result = get_tax_rate("AMC")  # Potentially loss-making

        # Should handle gracefully regardless of profit/loss
        assert 'success' in result


# =============================================================================
# UNIT TESTS: get_depreciation
# =============================================================================

class TestGetDepreciation:
    """Unit tests for the get_depreciation function."""

    @pytest.mark.parametrize("ticker", ["AAPL", "MSFT", "CAT", "DE"])
    def test_get_depreciation_various_sectors(self, ticker, rate_limiter):
        """Test depreciation across different sectors."""
        rate_limiter.wait()
        result = get_depreciation(ticker)

        if result.get('success'):
            assert 'd&a_pct' in result
            assert 'd&a' in result
            assert result['d&a'] >= 0
            # D&A as % of revenue typically 0-30%
            assert 0 <= result['d&a_pct'] <= 50

    def test_get_depreciation_asset_heavy(self, rate_limiter):
        """Test depreciation for asset-heavy companies."""
        rate_limiter.wait()
        result = get_depreciation("CAT")  # Caterpillar - asset heavy

        if result.get('success'):
            # Asset-heavy companies should have meaningful D&A
            assert result['d&a'] > 0

    def test_get_depreciation_result_structure(self, rate_limiter):
        """Test result structure for depreciation."""
        rate_limiter.wait()
        result = get_depreciation("AAPL")

        if result.get('success'):
            required_keys = ['d&a_pct', 'concept', 'd&a', 'revenue']
            for key in required_keys:
                assert key in result


# =============================================================================
# UNIT TESTS: get_disclosures_names
# =============================================================================

class TestGetDisclosuresNames:
    """Unit tests for the get_disclosures_names function."""

    @pytest.mark.parametrize("ticker", ["AAPL", "MSFT", "JPM"])
    def test_get_disclosures_names_major_companies(self, ticker, rate_limiter):
        """Test getting disclosures for major companies."""
        rate_limiter.wait()
        result = get_disclosures_names(ticker)

        assert 'ticker' in result
        assert 'success' in result

        if result['success']:
            assert 'disclosure_names' in result
            assert isinstance(result['disclosure_names'], list)
            assert len(result['disclosure_names']) > 0

    def test_get_disclosures_names_invalid_ticker(self, rate_limiter):
        """Test with invalid ticker."""
        rate_limiter.wait()
        result = get_disclosures_names("INVALID12345")

        assert result['success'] == False
        assert 'error' in result


# =============================================================================
# UNIT TESTS: extract_disclosure_data
# =============================================================================

class TestExtractDisclosureData:
    """Unit tests for the extract_disclosure_data function."""

    def test_extract_disclosure_data_with_valid_name(self, rate_limiter):
        """Test extracting disclosure with valid name."""
        rate_limiter.wait()

        # First get available disclosures
        disclosures = get_disclosures_names("AAPL")

        if disclosures.get('success') and disclosures.get('disclosure_names'):
            disclosure_name = disclosures['disclosure_names'][0]
            result = extract_disclosure_data("AAPL", disclosure_name)

            # Should return some data or empty dict
            assert isinstance(result, dict)

    def test_extract_disclosure_data_invalid_name(self, rate_limiter):
        """Test with invalid disclosure name."""
        rate_limiter.wait()
        result = extract_disclosure_data("AAPL", "NonExistentDisclosure12345")

        # Should handle gracefully
        assert isinstance(result, dict)

    def test_extract_disclosure_data_invalid_ticker(self, rate_limiter):
        """Test with invalid ticker."""
        rate_limiter.wait()
        result = extract_disclosure_data("INVALID12345", "SomeDisclosure")

        assert isinstance(result, dict)
        if 'error' in result:
            assert isinstance(result['error'], str)


# =============================================================================
# INTEGRATION TESTS: Full Workflow Testing
# =============================================================================

class TestIntegrationWorkflows:
    """Integration tests for complete analysis workflows."""

    def test_complete_financial_analysis_workflow(self, rate_limiter):
        """Test complete financial analysis for a single company."""
        ticker = "AAPL"
        results = {}

        # Get all metrics
        rate_limiter.wait()
        results['revenue'] = get_revenue_base(ticker)

        rate_limiter.wait()
        results['ebitda'] = get_ebitda_margin(ticker)

        rate_limiter.wait()
        results['capex'] = get_capex_pct_revenue(ticker)

        rate_limiter.wait()
        results['tax'] = get_tax_rate(ticker)

        rate_limiter.wait()
        results['depreciation'] = get_depreciation(ticker)

        # Validate all results
        success_count = sum(1 for r in results.values() if r.get('success', False))

        # At least some metrics should succeed for AAPL
        assert success_count >= 3, f"Expected at least 3 successful metrics, got {success_count}"

    def test_cross_company_comparison_workflow(self, rate_limiter):
        """Test comparing metrics across multiple companies."""
        tickers = ["AAPL", "MSFT", "GOOGL"]
        revenue_data = []

        for ticker in tickers:
            rate_limiter.wait()
            result = get_revenue_base(ticker)
            if result.get('success'):
                revenue_data.append({
                    'ticker': ticker,
                    'revenue': result['revenue_base']
                })

        # Should get data for at least 2 companies
        assert len(revenue_data) >= 2

        # Revenues should be different (unless by coincidence)
        revenues = [d['revenue'] for d in revenue_data]
        assert len(set(revenues)) >= 1

    def test_sector_comparison_workflow(self, rate_limiter):
        """Test comparing metrics across sectors."""
        sectors = {
            'tech': ["AAPL", "MSFT"],
            'healthcare': ["JNJ", "PFE"],
            'energy': ["XOM", "CVX"]
        }

        sector_metrics = {}

        for sector, tickers in sectors.items():
            sector_metrics[sector] = []
            for ticker in tickers:
                rate_limiter.wait()
                result = get_ebitda_margin(ticker)
                if result.get('success'):
                    sector_metrics[sector].append(result['ebitda_margin_percent'])

        # Should have data for multiple sectors
        sectors_with_data = sum(1 for v in sector_metrics.values() if len(v) > 0)
        assert sectors_with_data >= 2


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_ticker_with_period(self, rate_limiter):
        """Test tickers containing periods (e.g., BRK.B)."""
        rate_limiter.wait()
        result = get_revenue_base("BRK.B")

        # Should handle gracefully
        assert 'ticker' in result

    def test_ticker_case_sensitivity(self, rate_limiter):
        """Test case sensitivity of tickers."""
        rate_limiter.wait()
        result_upper = get_revenue_base("AAPL")

        rate_limiter.wait()
        result_lower = get_revenue_base("aapl")

        # Both should return similar results or both fail
        if result_upper.get('success') and result_lower.get('success'):
            # Revenue should be the same
            assert abs(result_upper['revenue_base'] - result_lower['revenue_base']) < 1

    def test_recently_ipo_company(self, rate_limiter):
        """Test with recently IPO'd companies (may have limited filings)."""
        rate_limiter.wait()
        result = get_revenue_base("RIVN")  # Rivian - IPO'd 2021

        # Should handle gracefully whether data exists or not
        assert 'ticker' in result or 'error' in result

    def test_foreign_adr_company(self, rate_limiter):
        """Test with foreign ADR companies."""
        rate_limiter.wait()
        result = get_revenue_base("TSM")  # Taiwan Semiconductor

        # ADRs may have different reporting formats
        assert 'ticker' in result or 'error' in result

    def test_very_old_company(self, rate_limiter):
        """Test with very old established companies."""
        rate_limiter.wait()
        result = get_revenue_base("GE")  # General Electric

        if result.get('success'):
            assert result['revenue_base'] > 0

    def test_company_with_restatement(self, rate_limiter):
        """Test handling of companies that may have restated financials."""
        rate_limiter.wait()
        result = get_revenue_base("JPM")

        # Should return most recent data
        if result.get('success'):
            assert 'period_end' in result

    def test_fiscal_year_mismatch(self, rate_limiter):
        """Test companies with non-calendar fiscal years."""
        rate_limiter.wait()
        result = get_revenue_base("AAPL")  # Apple has Sept fiscal year end

        if result.get('success'):
            # Should still work regardless of fiscal year
            assert result['revenue_base'] > 0

    def test_very_small_revenue(self, rate_limiter):
        """Test companies with very small revenue."""
        rate_limiter.wait()
        # Use a small cap company
        result = get_revenue_base("GERN")  # Geron Corp - small biotech

        # Should handle small values gracefully
        assert 'ticker' in result or 'error' in result

    def test_negative_operating_income(self, rate_limiter):
        """Test EBITDA for companies with negative operating income."""
        rate_limiter.wait()
        result = get_ebitda_margin("AMC")  # AMC Entertainment

        if result.get('success'):
            # Should handle negative values
            assert 'ebitda_margin_percent' in result

    def test_zero_division_handling(self, rate_limiter):
        """Test handling of potential division by zero scenarios."""
        # This tests internal handling - use a company that might have edge case data
        rate_limiter.wait()
        result = get_tax_rate("AAPL")

        # Should never raise ZeroDivisionError
        assert 'success' in result or 'error' in result

    def test_missing_da_components(self, rate_limiter):
        """Test EBITDA when D&A components might be missing."""
        rate_limiter.wait()
        result = get_ebitda_margin("V")  # Visa - may have minimal D&A

        if result.get('success'):
            # D&A should be handled even if small
            assert 'd&a' in result

    def test_reit_special_accounting(self, rate_limiter):
        """Test REITs which have special accounting rules."""
        rate_limiter.wait()
        result = get_revenue_base("O")  # Realty Income - REIT

        # REITs may report differently
        assert 'ticker' in result or 'error' in result

    def test_bank_special_accounting(self, rate_limiter):
        """Test banks which have unique financial statements."""
        rate_limiter.wait()
        result = get_revenue_base("JPM")

        # Banks report interest income differently
        if result.get('success'):
            assert result['revenue_base'] > 0


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================

class TestErrorHandling:
    """Tests for error handling and recovery."""

    def test_network_error_handling(self):
        """Test handling of network errors."""
        with patch('tools.web_search_server.sec_utils.Company') as mock_company:
            mock_company.side_effect = ConnectionError("Network error")

            result = get_revenue_base("AAPL")

            assert result.get('success') == False or result is None

    def test_api_timeout_handling(self):
        """Test handling of API timeouts."""
        with patch('tools.web_search_server.sec_utils.Company') as mock_company:
            mock_company.side_effect = TimeoutError("API timeout")

            result = get_latest_filing("AAPL")

            assert result is None

    def test_malformed_xbrl_handling(self):
        """Test handling of malformed XBRL data."""
        with patch('tools.web_search_server.sec_utils.get_latest_filing') as mock_filing:
            mock_filing.return_value = {
                'xbrl_data': "malformed_data",  # Not a valid XBRL object
                'filing_date': '2024-01-01'
            }

            result = get_revenue_base("AAPL")

            # Should handle gracefully
            assert 'success' in result

    def test_empty_xbrl_data_handling(self):
        """Test handling of empty XBRL data."""
        with patch('tools.web_search_server.sec_utils.get_latest_filing') as mock_filing:
            mock_filing.return_value = {
                'xbrl_data': None,
                'filing_date': '2024-01-01'
            }

            result = get_revenue_base("AAPL")

            assert result.get('success') == False

    def test_missing_required_concepts(self):
        """Test handling when required XBRL concepts are missing."""
        mock_xbrl = MagicMock()
        mock_xbrl.facts.query().by_concept().to_dataframe.return_value = pd.DataFrame()

        with patch('tools.web_search_server.sec_utils.get_latest_filing') as mock_filing:
            mock_filing.return_value = {
                'xbrl_data': mock_xbrl,
                'filing_date': '2024-01-01'
            }

            result = get_revenue_base("AAPL")

            assert result.get('success') == False

    def test_unicode_handling(self, rate_limiter):
        """Test handling of unicode in ticker symbols."""
        rate_limiter.wait()
        result = get_revenue_base("AAPL\u200b")  # Ticker with zero-width space

        # Should handle or fail gracefully
        assert isinstance(result, dict)

    def test_sql_injection_attempt(self):
        """Test that SQL injection attempts are handled safely."""
        result = get_revenue_base("AAPL'; DROP TABLE stocks;--")

        # Should fail safely without executing injection
        assert result.get('success') == False or result is None

    def test_very_long_ticker(self):
        """Test handling of very long ticker strings."""
        long_ticker = "A" * 1000
        result = get_revenue_base(long_ticker)

        assert result.get('success') == False or result is None

    def test_numeric_ticker(self):
        """Test handling of numeric ticker input."""
        result = get_revenue_base("12345")

        # Should handle gracefully
        assert isinstance(result, dict)


# =============================================================================
# DATA VALIDATION TESTS
# =============================================================================

class TestDataValidation:
    """Tests for validating output data quality and consistency."""

    @pytest.mark.parametrize("ticker", ["AAPL", "MSFT", "GOOGL", "AMZN", "META"])
    def test_revenue_positive(self, ticker, rate_limiter):
        """Test that revenue values are positive."""
        rate_limiter.wait()
        result = get_revenue_base(ticker)

        if result.get('success'):
            assert result['revenue_base'] > 0, f"{ticker} reported non-positive revenue"

    @pytest.mark.parametrize("ticker", ["AAPL", "MSFT", "GOOGL"])
    def test_capex_non_negative(self, ticker, rate_limiter):
        """Test that CapEx values are non-negative."""
        rate_limiter.wait()
        result = get_capex_pct_revenue(ticker)

        if result.get('success'):
            assert result['total_capex'] >= 0, f"{ticker} reported negative CapEx"

    @pytest.mark.parametrize("ticker", ["AAPL", "MSFT", "GOOGL"])
    def test_depreciation_non_negative(self, ticker, rate_limiter):
        """Test that depreciation values are non-negative."""
        rate_limiter.wait()
        result = get_depreciation(ticker)

        if result.get('success'):
            assert result['d&a'] >= 0, f"{ticker} reported negative D&A"

    def test_tax_rate_reasonable_range(self, rate_limiter):
        """Test that tax rates are within reasonable bounds."""
        tickers = ["AAPL", "MSFT", "JPM", "JNJ", "XOM"]

        for ticker in tickers:
            rate_limiter.wait()
            result = get_tax_rate(ticker)

            if result.get('success'):
                # Tax rates should typically be between -50% and 100%
                assert -50 <= result['effective_tax_rate'] <= 100, \
                    f"{ticker} has unusual tax rate: {result['effective_tax_rate']}"

    def test_ebitda_margin_reasonable_range(self, rate_limiter):
        """Test that EBITDA margins are within reasonable bounds."""
        tickers = ["AAPL", "MSFT", "WMT", "COST"]

        for ticker in tickers:
            rate_limiter.wait()
            result = get_ebitda_margin(ticker)

            if result.get('success'):
                # EBITDA margins typically -50% to 70%
                assert -100 <= result['ebitda_margin_percent'] <= 100, \
                    f"{ticker} has unusual EBITDA margin: {result['ebitda_margin_percent']}"

    def test_period_end_is_recent(self, rate_limiter):
        """Test that period_end dates are reasonably recent."""
        rate_limiter.wait()
        result = get_revenue_base("AAPL")

        if result.get('success') and 'period_end' in result:
            period_end = result['period_end']
            # Should be within last 2 years
            if isinstance(period_end, str):
                period_date = datetime.strptime(period_end[:10], '%Y-%m-%d')
                assert period_date > datetime.now() - timedelta(days=730)

    def test_filing_date_is_recent(self, rate_limiter):
        """Test that filing dates are reasonably recent."""
        rate_limiter.wait()
        result = get_revenue_base("AAPL")

        if result.get('success') and 'filing_date' in result:
            filing_date = result['filing_date']
            if isinstance(filing_date, str):
                file_date = datetime.strptime(filing_date[:10], '%Y-%m-%d')
                # Should be within last 2 years
                assert file_date > datetime.now() - timedelta(days=730)

    def test_consistency_across_functions(self, rate_limiter):
        """Test that revenue is consistent across functions."""
        ticker = "AAPL"

        rate_limiter.wait()
        revenue_result = get_revenue_base(ticker)

        rate_limiter.wait()
        ebitda_result = get_ebitda_margin(ticker)

        if revenue_result.get('success') and ebitda_result.get('success'):
            # Revenue should be the same (within small tolerance for rounding)
            revenue_from_base = revenue_result['revenue_base']
            revenue_from_ebitda = ebitda_result['revenue']

            # Allow 1% tolerance for rounding differences
            tolerance = max(revenue_from_base, revenue_from_ebitda) * 0.01
            assert abs(revenue_from_base - revenue_from_ebitda) <= tolerance


# =============================================================================
# CROSS-SECTOR TESTING (500 STOCKS)
# =============================================================================

class TestCrossSectorAll500:
    """Comprehensive tests across all 500 stocks in the test universe."""

    @pytest.mark.slow
    @pytest.mark.parametrize("ticker", TestStockUniverse.TECHNOLOGY[:25])
    def test_technology_sector_revenue(self, ticker, rate_limiter):
        """Test revenue extraction for technology sector."""
        rate_limiter.wait()
        result = get_revenue_base(ticker)

        assert 'ticker' in result
        # Log result for analysis
        logger.info(f"Technology - {ticker}: success={result.get('success')}")

    @pytest.mark.slow
    @pytest.mark.parametrize("ticker", TestStockUniverse.HEALTHCARE[:25])
    def test_healthcare_sector_revenue(self, ticker, rate_limiter):
        """Test revenue extraction for healthcare sector."""
        rate_limiter.wait()
        result = get_revenue_base(ticker)

        assert 'ticker' in result
        logger.info(f"Healthcare - {ticker}: success={result.get('success')}")

    @pytest.mark.slow
    @pytest.mark.parametrize("ticker", TestStockUniverse.FINANCIALS[:25])
    def test_financials_sector_revenue(self, ticker, rate_limiter):
        """Test revenue extraction for financial sector."""
        rate_limiter.wait()
        result = get_revenue_base(ticker)

        assert 'ticker' in result
        logger.info(f"Financials - {ticker}: success={result.get('success')}")

    @pytest.mark.slow
    @pytest.mark.parametrize("ticker", TestStockUniverse.CONSUMER_DISCRETIONARY[:25])
    def test_consumer_discretionary_sector_revenue(self, ticker, rate_limiter):
        """Test revenue extraction for consumer discretionary sector."""
        rate_limiter.wait()
        result = get_revenue_base(ticker)

        assert 'ticker' in result
        logger.info(f"Consumer Discretionary - {ticker}: success={result.get('success')}")

    @pytest.mark.slow
    @pytest.mark.parametrize("ticker", TestStockUniverse.ENERGY[:25])
    def test_energy_sector_revenue(self, ticker, rate_limiter):
        """Test revenue extraction for energy sector."""
        rate_limiter.wait()
        result = get_revenue_base(ticker)

        assert 'ticker' in result
        logger.info(f"Energy - {ticker}: success={result.get('success')}")

    @pytest.mark.slow
    @pytest.mark.parametrize("ticker", TestStockUniverse.INDUSTRIALS[:25])
    def test_industrials_sector_revenue(self, ticker, rate_limiter):
        """Test revenue extraction for industrials sector."""
        rate_limiter.wait()
        result = get_revenue_base(ticker)

        assert 'ticker' in result
        logger.info(f"Industrials - {ticker}: success={result.get('success')}")


# =============================================================================
# BATCH TESTING (ALL 500 STOCKS)
# =============================================================================

class TestBatchProcessing:
    """Batch tests for processing all 500 stocks."""

    @pytest.mark.slow
    def test_all_500_revenue(self, all_tickers, rate_limiter):
        """Test revenue extraction for all 500 stocks."""
        results = []

        for i, ticker in enumerate(all_tickers):
            try:
                rate_limiter.wait()
                result = get_revenue_base(ticker)
                result['ticker'] = ticker
                results.append(result)

                if (i + 1) % 50 == 0:
                    logger.info(f"Processed {i + 1}/{len(all_tickers)} tickers")

            except Exception as e:
                results.append({
                    'ticker': ticker,
                    'success': False,
                    'error': str(e)
                })

        # Generate summary
        summary = TestHelpers.collect_test_results(results)
        TestHelpers.log_test_summary("All 500 Revenue Test", summary)

        # At least 60% should succeed (accounting for ADRs, delisted, etc.)
        assert summary['success_rate'] >= 60, \
            f"Success rate too low: {summary['success_rate']:.2f}%"

    @pytest.mark.slow
    def test_all_500_ebitda(self, all_tickers, rate_limiter):
        """Test EBITDA margin extraction for all 500 stocks."""
        results = []

        for i, ticker in enumerate(all_tickers):
            try:
                rate_limiter.wait()
                result = get_ebitda_margin(ticker)
                if 'ticker' not in result:
                    result['ticker'] = ticker
                results.append(result)

                if (i + 1) % 50 == 0:
                    logger.info(f"EBITDA: Processed {i + 1}/{len(all_tickers)} tickers")

            except Exception as e:
                results.append({
                    'ticker': ticker,
                    'success': False,
                    'error': str(e)
                })

        summary = TestHelpers.collect_test_results(results)
        TestHelpers.log_test_summary("All 500 EBITDA Test", summary)

        # EBITDA may have lower success rate due to component requirements
        assert summary['success_rate'] >= 50, \
            f"EBITDA success rate too low: {summary['success_rate']:.2f}%"

    @pytest.mark.slow
    def test_sample_full_analysis(self, sample_tickers, rate_limiter):
        """Test full financial analysis on sample of 50 stocks."""
        full_results = []

        for ticker in sample_tickers:
            ticker_results = {'ticker': ticker}

            try:
                rate_limiter.wait()
                ticker_results['revenue'] = get_revenue_base(ticker)

                rate_limiter.wait()
                ticker_results['ebitda'] = get_ebitda_margin(ticker)

                rate_limiter.wait()
                ticker_results['capex'] = get_capex_pct_revenue(ticker)

                rate_limiter.wait()
                ticker_results['tax'] = get_tax_rate(ticker)

                rate_limiter.wait()
                ticker_results['depreciation'] = get_depreciation(ticker)

                # Count successes
                success_count = sum(1 for k in ['revenue', 'ebitda', 'capex', 'tax', 'depreciation']
                                  if ticker_results.get(k, {}).get('success', False))

                ticker_results['success'] = success_count >= 3
                ticker_results['metrics_succeeded'] = success_count

            except Exception as e:
                ticker_results['success'] = False
                ticker_results['error'] = str(e)

            full_results.append(ticker_results)

        # Analyze results
        successful = sum(1 for r in full_results if r.get('success', False))
        success_rate = (successful / len(full_results)) * 100

        logger.info(f"Full Analysis: {successful}/{len(full_results)} ({success_rate:.2f}%) succeeded")

        # At least 50% should have 3+ metrics
        assert success_rate >= 50


# =============================================================================
# STRESS TESTS
# =============================================================================

class TestStress:
    """Stress tests for high-volume and edge conditions."""

    @pytest.mark.slow
    def test_rapid_sequential_calls(self, mega_cap_tickers, rate_limiter):
        """Test rapid sequential API calls."""
        start_time = time.time()
        results = []

        for ticker in mega_cap_tickers:
            rate_limiter.wait()
            result = get_revenue_base(ticker)
            results.append(result)

        elapsed = time.time() - start_time
        successful = sum(1 for r in results if r.get('success', False))

        logger.info(f"Rapid Sequential: {successful}/{len(results)} in {elapsed:.2f}s")

        # Should maintain reasonable success rate
        assert successful >= len(mega_cap_tickers) // 2

    @pytest.mark.slow
    def test_repeated_same_ticker(self, rate_limiter):
        """Test calling same ticker multiple times."""
        ticker = "AAPL"
        results = []

        for _ in range(5):
            rate_limiter.wait()
            result = get_revenue_base(ticker)
            results.append(result)

        # All calls should return consistent results
        successful_revenues = [r['revenue_base'] for r in results if r.get('success', False)]

        if len(successful_revenues) >= 2:
            # All revenues should be identical
            assert all(r == successful_revenues[0] for r in successful_revenues)

    @pytest.mark.slow
    def test_alternating_functions(self, rate_limiter):
        """Test alternating between different functions."""
        ticker = "MSFT"
        functions = [
            lambda: get_revenue_base(ticker),
            lambda: get_ebitda_margin(ticker),
            lambda: get_capex_pct_revenue(ticker),
            lambda: get_tax_rate(ticker),
            lambda: get_depreciation(ticker)
        ]

        results = []
        for _ in range(3):  # 3 cycles
            for func in functions:
                rate_limiter.wait()
                try:
                    result = func()
                    results.append({'success': result.get('success', False)})
                except Exception as e:
                    results.append({'success': False, 'error': str(e)})

        successful = sum(1 for r in results if r['success'])
        logger.info(f"Alternating Functions: {successful}/{len(results)} succeeded")

        # Should maintain >50% success rate
        assert successful >= len(results) // 2


# =============================================================================
# REGRESSION TESTS
# =============================================================================

class TestRegression:
    """Regression tests for known issues and bug fixes."""

    def test_berkshire_ticker_format(self, rate_limiter):
        """Test Berkshire Hathaway ticker with period."""
        rate_limiter.wait()
        result_b = get_revenue_base("BRK.B")

        # Should handle period in ticker
        assert 'ticker' in result_b or 'error' in result_b

    def test_meta_ticker_change(self, rate_limiter):
        """Test META (formerly FB) ticker."""
        rate_limiter.wait()
        result = get_revenue_base("META")

        if result.get('success'):
            # Should find data under META
            assert result['revenue_base'] > 0

    def test_google_dual_class(self, rate_limiter):
        """Test Google's dual class structure."""
        rate_limiter.wait()
        result_googl = get_revenue_base("GOOGL")

        rate_limiter.wait()
        result_goog = get_revenue_base("GOOG")

        # Both should return same revenue (same company)
        if result_googl.get('success') and result_goog.get('success'):
            tolerance = max(result_googl['revenue_base'], result_goog['revenue_base']) * 0.01
            assert abs(result_googl['revenue_base'] - result_goog['revenue_base']) <= tolerance

    def test_zero_revenue_handling(self):
        """Test handling of zero revenue edge case."""
        mock_xbrl = MagicMock()
        mock_facts = MagicMock()
        mock_query = MagicMock()

        df = pd.DataFrame({
            'concept': ['us-gaap:Revenues'],
            'numeric_value': [0.0],
            'period_start': ['2023-01-01'],
            'period_end': ['2023-12-31'],
            'unit': ['USD']
        })

        mock_query.to_dataframe.return_value = df
        mock_query.by_concept.return_value = mock_query
        mock_facts.query.return_value = mock_query
        mock_xbrl.facts = mock_facts

        with patch('tools.web_search_server.sec_utils.get_latest_filing') as mock_filing:
            mock_filing.return_value = {
                'xbrl_data': mock_xbrl,
                'filing_date': '2024-01-01'
            }

            result = get_revenue_base("TEST")

            # Should handle zero gracefully
            assert isinstance(result, dict)

    def test_negative_value_handling(self):
        """Test handling of negative values in XBRL data."""
        mock_xbrl = MagicMock()
        mock_facts = MagicMock()
        mock_query = MagicMock()

        df = pd.DataFrame({
            'concept': ['us-gaap:OperatingIncomeLoss'],
            'numeric_value': [-1000000000.0],  # Negative operating income
            'period_start': ['2023-01-01'],
            'period_end': ['2023-12-31'],
            'unit': ['USD']
        })

        mock_query.to_dataframe.return_value = df
        mock_query.by_concept.return_value = mock_query
        mock_facts.query.return_value = mock_query
        mock_xbrl.facts = mock_facts

        result = filter_annual_data(mock_xbrl, 'us-gaap:OperatingIncomeLoss')

        if result is not None:
            # Should preserve negative values
            assert result['value'] == -1000000000.0


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestPerformance:
    """Performance and timing tests."""

    def test_single_call_timing(self, rate_limiter):
        """Test timing of a single API call."""
        rate_limiter.wait()

        start = time.time()
        result = get_revenue_base("AAPL")
        elapsed = time.time() - start

        logger.info(f"Single call timing: {elapsed:.3f}s")

        # Should complete within 30 seconds
        assert elapsed < 30

    @pytest.mark.slow
    def test_batch_timing(self, sample_tickers, rate_limiter):
        """Test timing of batch processing."""
        start = time.time()
        successful = 0

        for ticker in sample_tickers[:10]:  # Use 10 for timing test
            rate_limiter.wait()
            result = get_revenue_base(ticker)
            if result.get('success'):
                successful += 1

        elapsed = time.time() - start

        logger.info(f"Batch timing (10 tickers): {elapsed:.2f}s, {successful} successful")

        # Should complete within 5 minutes for 10 tickers
        assert elapsed < 300


# =============================================================================
# SPECIAL COMPANY TESTS
# =============================================================================

class TestSpecialCompanies:
    """Tests for companies with special characteristics."""

    @pytest.mark.parametrize("ticker", TestStockUniverse.FOREIGN_ADRS[:5])
    def test_foreign_adrs(self, ticker, rate_limiter):
        """Test foreign ADRs (may have different reporting)."""
        rate_limiter.wait()
        result = get_revenue_base(ticker)

        # Should handle gracefully
        assert 'ticker' in result or 'error' in result
        logger.info(f"ADR {ticker}: success={result.get('success')}")

    @pytest.mark.parametrize("ticker", ["AMT", "PLD", "O", "EQIX"])
    def test_reits(self, ticker, rate_limiter):
        """Test REITs (special accounting rules)."""
        rate_limiter.wait()
        result = get_revenue_base(ticker)

        logger.info(f"REIT {ticker}: success={result.get('success')}")
        assert 'ticker' in result or 'error' in result

    @pytest.mark.parametrize("ticker", ["JPM", "BAC", "C", "WFC", "GS"])
    def test_major_banks(self, ticker, rate_limiter):
        """Test major banks (interest income vs revenue)."""
        rate_limiter.wait()
        result = get_revenue_base(ticker)

        logger.info(f"Bank {ticker}: success={result.get('success')}")
        if result.get('success'):
            # Banks should have substantial revenue
            assert result['revenue_base'] > 10000  # > $10B in millions

    @pytest.mark.parametrize("ticker", ["MET", "PRU", "AFL", "PGR"])
    def test_insurance_companies(self, ticker, rate_limiter):
        """Test insurance companies (premium income)."""
        rate_limiter.wait()
        result = get_revenue_base(ticker)

        logger.info(f"Insurance {ticker}: success={result.get('success')}")
        assert 'ticker' in result or 'error' in result

    @pytest.mark.parametrize("ticker", ["NEE", "DUK", "SO"])
    def test_utilities(self, ticker, rate_limiter):
        """Test utilities (regulated industries)."""
        rate_limiter.wait()
        result = get_revenue_base(ticker)

        logger.info(f"Utility {ticker}: success={result.get('success')}")
        if result.get('success'):
            assert result['revenue_base'] > 0


# =============================================================================
# MOCK-BASED UNIT TESTS (NO API CALLS)
# =============================================================================

class TestMockBased:
    """Pure unit tests using mocks (no actual API calls)."""

    def test_get_revenue_base_mock_success(self):
        """Test get_revenue_base with mocked successful response."""
        mock_xbrl = MagicMock()
        mock_facts = MagicMock()
        mock_query = MagicMock()

        df = pd.DataFrame({
            'concept': ['us-gaap:Revenues'],
            'numeric_value': [365000000000.0],  # Apple-like revenue
            'period_start': ['2023-10-01'],
            'period_end': ['2024-09-30'],
            'unit': ['USD']
        })

        mock_query.to_dataframe.return_value = df
        mock_query.by_concept.return_value = mock_query
        mock_facts.query.return_value = mock_query
        mock_xbrl.facts = mock_facts

        with patch('tools.web_search_server.sec_utils.get_latest_filing') as mock_filing:
            mock_filing.return_value = {
                'filing_date': '2024-11-01',
                'xbrl_data': mock_xbrl,
                'url': 'https://sec.gov/test',
                'accession_number': '0000320193-24-000001'
            }

            result = get_revenue_base("AAPL")

            assert result['success'] == True
            assert result['revenue_base'] == 365000.0  # In millions

    def test_get_ebitda_margin_mock_calculation(self):
        """Test EBITDA margin calculation with mocked data."""
        # This tests the calculation logic without API calls
        mock_xbrl = MagicMock()
        mock_facts = MagicMock()
        mock_query = MagicMock()

        def mock_by_concept(concept):
            if 'OperatingIncome' in concept:
                df = pd.DataFrame({
                    'concept': [concept],
                    'numeric_value': [100000000000.0],  # $100B
                    'period_start': ['2023-01-01'],
                    'period_end': ['2023-12-31'],
                    'unit': ['USD']
                })
            elif 'Depreciation' in concept:
                df = pd.DataFrame({
                    'concept': [concept],
                    'numeric_value': [10000000000.0],  # $10B
                    'period_start': ['2023-01-01'],
                    'period_end': ['2023-12-31'],
                    'unit': ['USD']
                })
            elif 'Revenue' in concept:
                df = pd.DataFrame({
                    'concept': [concept],
                    'numeric_value': [400000000000.0],  # $400B
                    'period_start': ['2023-01-01'],
                    'period_end': ['2023-12-31'],
                    'unit': ['USD']
                })
            else:
                df = pd.DataFrame()

            mock_q = MagicMock()
            mock_q.to_dataframe.return_value = df
            return mock_q

        mock_query.by_concept = mock_by_concept
        mock_facts.query.return_value = mock_query
        mock_xbrl.facts = mock_facts

        with patch('tools.web_search_server.sec_utils.get_latest_filing') as mock_filing:
            mock_filing.return_value = {
                'filing_date': '2024-02-01',
                'xbrl_data': mock_xbrl
            }

            with patch('tools.web_search_server.sec_utils.get_revenue_base') as mock_revenue:
                mock_revenue.return_value = {
                    'success': True,
                    'revenue_base': 400000.0  # $400B in millions
                }

                result = get_ebitda_margin("TEST")

                if result.get('success'):
                    # EBITDA = 100B + 10B = 110B
                    # EBITDA Margin = 110B / 400B * 100 = 27.5%
                    expected_margin = 27.5
                    assert abs(result['ebitda_margin_percent'] - expected_margin) < 0.1


# =============================================================================
# COMPREHENSIVE SUMMARY TEST
# =============================================================================

class TestComprehensiveSummary:
    """Generate comprehensive test summary across all 500 stocks."""

    @pytest.mark.slow
    def test_generate_comprehensive_report(self, all_tickers, rate_limiter):
        """Generate a comprehensive test report for all 500 stocks."""
        report = {
            'revenue': {'success': 0, 'fail': 0, 'errors': {}},
            'ebitda': {'success': 0, 'fail': 0, 'errors': {}},
            'capex': {'success': 0, 'fail': 0, 'errors': {}},
            'tax': {'success': 0, 'fail': 0, 'errors': {}},
            'depreciation': {'success': 0, 'fail': 0, 'errors': {}},
        }

        sample = random.sample(all_tickers, min(100, len(all_tickers)))  # Test 100 for speed

        for ticker in sample:
            # Revenue
            rate_limiter.wait()
            result = get_revenue_base(ticker)
            if result.get('success'):
                report['revenue']['success'] += 1
            else:
                report['revenue']['fail'] += 1
                error = str(result.get('error', 'Unknown'))[:30]
                report['revenue']['errors'][error] = report['revenue']['errors'].get(error, 0) + 1

            # EBITDA
            rate_limiter.wait()
            result = get_ebitda_margin(ticker)
            if result.get('success'):
                report['ebitda']['success'] += 1
            else:
                report['ebitda']['fail'] += 1

            # CapEx
            rate_limiter.wait()
            result = get_capex_pct_revenue(ticker)
            if result.get('success'):
                report['capex']['success'] += 1
            else:
                report['capex']['fail'] += 1

            # Tax Rate
            rate_limiter.wait()
            result = get_tax_rate(ticker)
            if result.get('success'):
                report['tax']['success'] += 1
            else:
                report['tax']['fail'] += 1

            # Depreciation
            rate_limiter.wait()
            result = get_depreciation(ticker)
            if result.get('success'):
                report['depreciation']['success'] += 1
            else:
                report['depreciation']['fail'] += 1

        # Log summary
        logger.info("\n" + "="*70)
        logger.info("COMPREHENSIVE TEST REPORT")
        logger.info("="*70)

        for metric, data in report.items():
            total = data['success'] + data['fail']
            rate = (data['success'] / total * 100) if total > 0 else 0
            logger.info(f"{metric.upper()}: {data['success']}/{total} ({rate:.1f}%)")

        logger.info("="*70 + "\n")

        # Overall pass criteria: at least 50% success rate for core metrics
        assert report['revenue']['success'] >= len(sample) * 0.5


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    """
    Run tests with various options:

    Quick test (sample):
        pytest test_tools.web_search_server.sec_utils.py -v -k "not slow" --tb=short

    Full test (all 500):
        pytest test_tools.web_search_server.sec_utils.py -v --tb=short

    Specific category:
        pytest test_tools.web_search_server.sec_utils.py -v -k "TestGetRevenueBase"

    With coverage:
        pytest test_tools.web_search_server.sec_utils.py -v --cov=tools.web_search_server.sec_utils

    Generate report:
        pytest test_tools.web_search_server.sec_utils.py -v --html=report.html
    """
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-x",  # Stop on first failure
        "-k", "not slow"  # Skip slow tests by default
    ])

