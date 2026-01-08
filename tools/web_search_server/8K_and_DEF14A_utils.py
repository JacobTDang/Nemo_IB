# SEC Filing Parser v7.2 - Production-Grade Truth-Grounding
#
# Critical improvements from v7.1:
# 1. Case-insensitive Item anchoring + subtype keyword evidence for 8-K
# 2. Facts vs Candidates separation (no unvalidated data in facts)
# 3. Structured evidence validation (validate right cell, not any number)
# 4. Robust Summary Comp Table detection with variants
# 5. Hard gates for board table (must have age OR since)
# 6. Role token stopwords for name cleaning edge cases
# 7. Truth tests and assertions

from typing import Any, Dict, List, Optional, Tuple, Set, Union, cast
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from datetime import datetime
from edgar import Company, set_identity
from io import StringIO
import pandas as pd
import logging
import re
import os
import numpy as np

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None  # Optional dependency

# Logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)

# Configuration
NAME = os.getenv('NAME', 'Investment Analyst')
SEC_EMAIL = os.getenv('SEC_EMAIL', 'analyst@example.com')

# Regex constants
_ITEM_NUM_RE = re.compile(r'(\d+\.\d+)')
_MONEY_RE = re.compile(r'(?:\$\s*\d{1,3}(?:,\d{3})+(?:\.\d{2})?|\d{1,3}(?:,\d{3})+(?:\.\d{2})?)')
_YEAR_RE = re.compile(r'\b(19|20)\d{2}\b')

class EventCategory(Enum):
    MATERIAL_AGREEMENT = auto()
    FINANCIAL_RESULTS = auto()
    ASSET_TRANSACTION = auto()
    BANKRUPTCY = auto()
    MANAGEMENT_CHANGE = auto()
    CORPORATE_GOVERNANCE = auto()
    SECURITIES = auto()
    REGULATORY = auto()
    OTHER = auto()

@dataclass
class EvidenceLocation:
    section_hint: Optional[str] = None
    table_index: Optional[int] = None
    row_index: Optional[int] = None
    line_range: Optional[List[int]] = None

@dataclass
class Evidence:
    strategy: str
    snippet: str
    location: EvidenceLocation = field(default_factory=EvidenceLocation)
    column_header: Optional[str] = None
    table_kv: Optional[Dict[str, str]] = None  # NEW: Structured key-value pairs

    def _norm_for_match(self, s: str) -> str:
        """Normalize string for matching by removing parentheticals and punctuation."""
        s = re.sub(r'\(.*?\)', ' ', s)              # Drop parentheticals like (Chair)
        s = re.sub(r'[^a-zA-Z0-9\s]', ' ', s)       # Drop punctuation
        s = re.sub(r'\s+', ' ', s).strip().lower()
        return s

    def validate_name(self, name: str) -> bool:
        """Validate that the evidence snippet contains the name."""
        normalized_name = self._norm_for_match(name)
        normalized_snippet = self._norm_for_match(self.snippet)
        return normalized_name in normalized_snippet

    def validate_amount(self, amount: float, tolerance: float = 0.05) -> bool:
        """IMPROVED: Structured validation - check the right cell first."""
        # NEW: If we have structured table data and column header, validate that specific cell
        if self.table_kv and self.column_header:
            # Find the value in the key-value map that corresponds to our column header
            col_lower = self.column_header.lower()
            for key, value in self.table_kv.items():
                if 'total' in key.lower() and 'total' in col_lower:
                    # This is likely our total column
                    return self._numeric_close(value, amount, tolerance)

        # Fallback: Extract all numbers from snippet and compare
        money_patterns = [
            r'\$\s*([\d,]+(?:\.\d{2})?)',
            r'([\d,]+(?:\.\d{2})?)\s*(?:million|m)',
            r'\(([\d,]+(?:\.\d{2})?)\)',
            r'([\d,]+(?:\.\d{2})?)',
        ]

        for pattern in money_patterns:
            matches = re.findall(pattern, self.snippet, re.IGNORECASE)
            for match_str in matches:
                try:
                    normalized = match_str.replace(',', '')
                    value = float(normalized)

                    if re.search(rf'{re.escape(match_str)}\s*(?:million|m)', self.snippet, re.IGNORECASE):
                        value *= 1_000_000

                    if abs(value - amount) / amount <= tolerance:
                        return True
                except:
                    continue

        return False

    def _numeric_close(self, value_str: str, target: float, tolerance: float) -> bool:
        """Check if a string value is numerically close to target."""
        try:
            normalized = re.sub(r'[^\d.]', '', value_str)
            if not normalized:
                return False
            value = float(normalized)
            return abs(value - target) / target <= tolerance
        except:
            return False

    def validate_label(self, label: str, check_header: bool = True) -> bool:
        """Validate label in snippet and optionally in column header."""
        if label.lower() in self.snippet.lower():
            return True
        if check_header and self.column_header and label.lower() in self.column_header.lower():
            return True
        return False

@dataclass
class ExtractionQuality:
    assertions_missing_evidence: int = 0
    evidence_coverage_pct: float = 0.0
    eight_k_avg_confidence: float = 0.0
    potential_error_alerts: List[str] = field(default_factory=list)
    evidence_validation_failures: int = 0
    fields_nullified_by_validation: int = 0
    validation_assertions_passed: bool = True  # NEW: Truth tests passed

@dataclass
class SECItemDefinition:
    item_number: str
    title: str
    description: str
    category: EventCategory
    keywords: List[str]
    sub_types: Dict[str, List[str]] = field(default_factory=dict)
    weight: float = 1.0

@dataclass
class ClassificationResult:
    event_type: str
    sub_type: Optional[str]
    confidence: float
    matched_keywords: List[str]
    sec_item: Optional[SECItemDefinition]
    category: EventCategory
    items_source: str = "unknown"
    evidence: Optional[Evidence] = None

@dataclass
class ExtractedName:
    full_name: str
    first_name: str = field(default="")
    last_name: str = field(default="")
    title: Optional[str] = field(default=None)
    confidence: float = field(default=1.0)
    age: Optional[int] = field(default=None)
    evidence: Optional[Evidence] = None

# SEC 8-K Items Registry
SEC_8K_ITEMS: Dict[str, SECItemDefinition] = {
    "1.01": SECItemDefinition("1.01", "Entry into a Material Definitive Agreement", "Material contracts", EventCategory.MATERIAL_AGREEMENT,
        ["agreement", "contract", "amendment", "entered into", "executed"], {"acquisition": ["acquisition", "merger"], "debt_financing": ["credit", "loan", "debt"]}, 1.5),
    "1.02": SECItemDefinition("1.02", "Termination of a Material Definitive Agreement", "Termination of contracts", EventCategory.MATERIAL_AGREEMENT, ["termination", "terminated", "cancelled"], {}, 1.4),
    "1.03": SECItemDefinition("1.03", "Bankruptcy or Receivership", "Bankruptcy filings", EventCategory.BANKRUPTCY, ["bankruptcy", "chapter 11", "chapter 7", "insolvency"], {}, 2.0),
    "1.05": SECItemDefinition("1.05", "Material Cybersecurity Incidents", "Cybersecurity disclosure", EventCategory.REGULATORY, ["cybersecurity", "data breach", "security incident"], {}, 1.8),
    "2.01": SECItemDefinition("2.01", "Completion of Acquisition or Disposition", "Completed transactions", EventCategory.ASSET_TRANSACTION,
        ["completed", "closed", "acquisition", "disposition", "sale"], {"acquisition_complete": ["acquired", "closed acquisition"], "disposition": ["sold", "divested"]}, 1.8),
    "2.02": SECItemDefinition("2.02", "Results of Operations and Financial Condition", "Earnings releases", EventCategory.FINANCIAL_RESULTS,
        ["results", "earnings", "revenue", "income", "quarter", "fiscal"], {"earnings_release": ["earnings", "quarterly results"], "guidance": ["guidance", "outlook"]}, 1.6),
    "2.03": SECItemDefinition("2.03", "Creation of a Direct Financial Obligation", "New debt", EventCategory.MATERIAL_AGREEMENT, ["obligation", "debt", "credit facility", "notes", "bonds"], {}, 1.5),
    "2.04": SECItemDefinition("2.04", "Triggering Events for Financial Obligation", "Debt triggers", EventCategory.FINANCIAL_RESULTS, ["acceleration", "default", "covenant", "breach"], {}, 1.6),
    "2.05": SECItemDefinition("2.05", "Exit or Disposal Activities", "Restructuring", EventCategory.ASSET_TRANSACTION, ["restructuring", "exit", "layoff", "workforce reduction", "closure"], {}, 1.4),
    "2.06": SECItemDefinition("2.06", "Material Impairments", "Asset impairment", EventCategory.FINANCIAL_RESULTS, ["impairment", "write-down", "write-off", "goodwill"], {}, 1.5),
    "3.01": SECItemDefinition("3.01", "Notice of Delisting", "Listing issues", EventCategory.SECURITIES, ["delisting", "nasdaq", "nyse", "compliance", "deficiency"], {}, 1.7),
    "3.02": SECItemDefinition("3.02", "Unregistered Sales of Equity", "Private placements", EventCategory.SECURITIES, ["unregistered", "private placement", "regulation d"], {}, 1.3),
    "3.03": SECItemDefinition("3.03", "Material Modification to Rights", "Shareholder rights changes", EventCategory.CORPORATE_GOVERNANCE, ["rights", "shareholder", "modification"], {}, 1.3),
    "4.01": SECItemDefinition("4.01", "Changes in Certifying Accountant", "Auditor changes", EventCategory.CORPORATE_GOVERNANCE, ["auditor", "accountant", "dismissed", "resigned", "engaged"], {}, 1.5),
    "4.02": SECItemDefinition("4.02", "Non-Reliance on Financial Statements", "Restatements", EventCategory.FINANCIAL_RESULTS, ["restatement", "non-reliance", "correction", "error"], {}, 2.0),
    "5.01": SECItemDefinition("5.01", "Changes in Control", "Control changes", EventCategory.CORPORATE_GOVERNANCE, ["change of control", "controlling", "takeover"], {}, 2.0),
    "5.02": SECItemDefinition("5.02", "Departure/Appointment of Officers", "Management changes", EventCategory.MANAGEMENT_CHANGE,
        ["appointed", "resigned", "retired", "elected", "departure", "stepped down", "named"], {"departure": ["resigned", "retired", "left", "stepped down"], "appointment": ["appointed", "named", "elected", "promoted"]}, 1.7),
    "5.03": SECItemDefinition("5.03", "Amendments to Bylaws", "Charter changes", EventCategory.CORPORATE_GOVERNANCE, ["amendment", "bylaws", "charter", "articles"], {}, 1.1),
    "5.07": SECItemDefinition("5.07", "Shareholder Vote", "Vote results", EventCategory.CORPORATE_GOVERNANCE, ["vote", "shareholder meeting", "annual meeting", "election"], {}, 1.0),
    "7.01": SECItemDefinition("7.01", "Regulation FD Disclosure", "Material disclosure", EventCategory.OTHER, ["regulation fd", "disclosure", "investor", "presentation"], {}, 0.9),
    "8.01": SECItemDefinition("8.01", "Other Events", "Other material events", EventCategory.OTHER, [],
        {
            "litigation": ["litigation", "lawsuit", "settlement", "fine", "penalty", "complaint"],
            "regulatory": ["regulatory", "fda", "approval", "compliance"],
            "debt_offering": ["notes", "offering", "indenture", "senior notes", "debentures"]
        }, 0.5),
    "9.01": SECItemDefinition("9.01", "Financial Statements and Exhibits", "Exhibits", EventCategory.OTHER, ["exhibit"], {}, 0.1),
}

# Validation Constants
VALIDATION_CONFIG: Dict[str, Any] = {
    "min_director_age": 21,
    "max_director_age": 99,
    "max_tenure_years": 60,
    "min_independence_ratio": 0.5,
    "max_compensation": 1_000_000_000,
    "min_compensation": 10_000,
}

# Title regex
COMMON_TITLES = [
    r'Chief Executive Officer', r'CEO', r'Chief Financial Officer', r'CFO',
    r'Chief Operating Officer', r'COO', r'Chief Technology Officer', r'CTO',
    r'Chief Legal Officer', r'CLO', r'General Counsel', r'President', r'Chairman',
    r'Chair', r'Director', r'Senior Vice President', r'SVP', r'Executive Vice President',
    r'EVP', r'Vice President', r'VP', r'Secretary', r'Treasurer', r'Trustee', r'Nominee',
    r'Co-Founder', r'Founder', r'Member', r'Head of'
]
TITLE_REGEX = re.compile(r'\b(?:' + '|'.join(COMMON_TITLES) + r')\b.*', re.IGNORECASE)

# NEW: Role token stopwords for edge cases
ROLE_TOKENS = {
    'senior', 'vice', 'chief', 'officer', 'president', 'director',
    'chair', 'executive', 'global', 'secretary', 'treasurer', 'member',
    'former', 'interim', 'acting'  # NEW: Add qualifiers that precede titles
}

# IMPROVED: More comprehensive Summary Compensation Table indicators
SUMMARY_COMP_TABLE_INDICATORS = [
    'salary', 'bonus',
    'stock award', 'stock awards',
    'option award', 'option awards',
    'non-equity', 'non equity',
    'incentive plan',
    'all other comp', 'all other compensation',
    'total', 'total ($)', 'total compensation'
]

# Board table indicators
BOARD_TABLE_POSITIVE = ['age', 'director', 'nominee', 'board', 'since', 'joined']
BOARD_TABLE_NEGATIVE = ['committee', 'meeting', 'attendance']

# =============================================================================
# HELPER FUNCTIONS & CLEANING
# =============================================================================

def normalize_item_numbers(items: List[Any]) -> List[str]:
    out: List[str] = []
    for it in items:
        if it is None: continue
        s = str(it).strip()
        m = _ITEM_NUM_RE.search(s)
        if m: out.append(m.group(1))
        elif re.fullmatch(r'\d+\.\d+', s): out.append(s)
    return sorted(set(out), key=lambda x: float(x))

def safe_get_text(filing: Any) -> str:
    if filing is None: return ""
    try:
        if hasattr(filing, 'text') and callable(filing.text):
            result = filing.text()
            return str(result) if result is not None else ""
    except: pass
    return ""

def safe_get_html(filing: Any) -> str:
    if filing is None: return ""
    try:
        if hasattr(filing, 'html') and callable(filing.html):
            result = filing.html()
            return str(result) if result is not None else ""
    except: pass
    return ""

def safe_get_filing_date(filing: Any) -> Optional[str]:
    if filing is None: return None
    try:
        val = getattr(filing, 'filing_date', None)
        return str(val) if val is not None else None
    except: return None

def _find_item_anchor(full_text: str, item_num: str) -> Optional[int]:
    """NEW: Case-insensitive Item anchor finding."""
    m = re.search(rf'(?i)\bitem\s+{re.escape(item_num)}\b', full_text)
    return m.start() if m else None

def clean_name_cell(s: str) -> str:
    """IMPROVED: Handles multi-word titles and role token stopwords."""
    if not s or not isinstance(s, str): return ""

    s = re.sub(r'\(.*?\)', '', s)
    s = s.replace('"', '').replace("'", "")

    if ',' in s:
        parts = s.split(',')
        s = parts[0].strip()

    # Try full title regex first
    title_match = TITLE_REGEX.search(s)
    if title_match:
        s = s[:title_match.start()].strip()
    else:
        # NEW: Fallback to role token stopwords
        # Skip leading role tokens, but stop once we've started capturing a name
        parts = s.split()
        clean = []
        for p in parts:
            token = p.lower().strip('.').strip(',')
            if token in ROLE_TOKENS:
                if not clean:
                    continue  # Skip leading role words
                break  # Stop once we've started capturing a name
            clean.append(p)
        s = ' '.join(clean)

    s = re.sub(r'\s*age\s*\d+', '', s, flags=re.IGNORECASE)
    s = re.sub(r'[^a-zA-ZÀ-ÿ\.\-\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()

    # NEW: Remove trailing qualifiers that often precede titles
    s = re.sub(r'\b(?:Former|Interim|Acting)\b$', '', s, flags=re.IGNORECASE).strip()

    return s

def create_table_evidence(row: pd.Series, df: pd.DataFrame) -> Tuple[str, Dict[str, str]]:
    """NEW: Returns both snippet AND structured key-value pairs."""
    kv = {}
    pairs = []
    for header, value in zip(df.columns, row.values):
        h = str(header).strip()
        v = str(value).strip()

        # Skip empty, unnamed, or "nan" headers
        if not h or h.lower() in {'unnamed', 'nan', ''}:
            continue
        if not v or v.lower() == 'nan':
            continue

        kv[h] = v
        pairs.append(f"{h}: {v}")

    snippet = " | ".join(pairs)
    return snippet, kv

def parse_independent_cell(val: Any) -> Optional[bool]:
    """
    Parse independence cell values, handling common formats including icons.
    Returns True for independent, False for non-independent, None for unknown.
    """
    if val is None:
        return None
    s = str(val).strip().lower()
    if not s or s in {"nan", "—", "-", "–", ""}:
        return None
    # Explicit yes/true indicators
    if s in {"yes", "y", "true", "✓", "✔", "x", "●", "•", "independent"}:
        return True
    # Explicit no/false indicators
    if s in {"no", "n", "false"}:
        return False
    # Numeric indicators (sometimes used)
    if s in {"1"}:
        return True
    if s in {"0"}:
        return False
    return None

def norm_name_key(name: str) -> str:
    """Normalize names for matching (handles commas, Jr., parentheticals)."""
    name = re.sub(r'\(.*?\)', ' ', name)
    name = re.sub(r'[^a-zA-Z0-9\s]', ' ', name)
    name = re.sub(r'\s+', ' ', name).strip().lower()
    return name

def flatten_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return df

    if isinstance(df.columns, pd.MultiIndex):
        new_cols = []
        for col in df.columns.values:
            parts = [str(c).strip() for c in col if pd.notna(c) and 'Unnamed' not in str(c)]
            new_cols.append(' '.join(parts).strip())
        df.columns = pd.Index(new_cols)
    else:
        df.columns = pd.Index([str(c).strip() if 'Unnamed' not in str(c) else '' for c in df.columns])

    if len(df) > 0:
        for i in range(min(5, len(df))):
            row_vals = df.iloc[i].values
            row_text = ' '.join(str(v).lower() for v in row_vals)
            # IMPROVED: Require at least 2 header keywords to prevent false promotion
            header_keywords = ['name', 'salary', 'bonus', 'stock', 'option', 'non-equity', 'all other', 'total', 'age', 'since', 'independent']
            keyword_hits = sum(1 for k in header_keywords if k in row_text)
            if keyword_hits >= 2:
                new_header: List[str] = []
                for j, val in enumerate(df.iloc[i]):
                    val_s = str(val).strip()
                    if not val_s:
                        val_s = str(df.columns[j]) if j < len(df.columns) else f"Col_{j}"
                    new_header.append(val_s)
                df.columns = pd.Index(new_header)
                df = df.iloc[i+1:].reset_index(drop=True)
                break

    return df

def is_summary_compensation_table(df: pd.DataFrame) -> Tuple[bool, int]:
    """IMPROVED: More robust with normalization and variants."""
    header_text = ' '.join(str(c).lower() for c in df.columns)

    # Normalize: remove punctuation, collapse spaces
    header_text = re.sub(r'[^a-z0-9\s]', ' ', header_text)
    header_text = re.sub(r'\s+', ' ', header_text)

    hits = sum(1 for indicator in SUMMARY_COMP_TABLE_INDICATORS if indicator in header_text)

    # Must have at least 4 of the indicators
    is_summary_comp = hits >= 4

    return is_summary_comp, hits

def score_board_table(df: pd.DataFrame, name_count: int) -> int:
    """IMPROVED: Hard gates with more flexibility."""
    header_text = ' '.join(str(c).lower() for c in df.columns)

    # CRITICAL: Exclude compensation tables
    comp_indicators = ['salary', 'bonus', 'stock award', 'option award', 'total ($)', 'fees earned']
    comp_hits = sum(1 for indicator in comp_indicators if indicator in header_text)
    if comp_hits >= 3:
        return -100

    # NEW: Heavily penalize matrix tables (committee grids with names as headers)
    def looks_like_person_header(h: str) -> bool:
        h = h.strip()
        if len(h.split()) < 2:
            return False
        if any(c.isdigit() for c in h):
            return False
        # Typical person-name patterns
        return bool(re.match(r'^[A-Z][a-z]+(\.|\'|"|")?(\s+[A-Z]\.)?(\s+[A-Z][a-z]+)+', h))

    namey_headers = sum(1 for c in df.columns if looks_like_person_header(str(c)))
    if namey_headers >= 2:
        return -200  # Strong rejection for matrix/grid tables

    # IMPROVED: More flexible hard gate
    # Require: (Age OR Since) OR (Independent AND Director/Nominee)
    has_age = 'age' in header_text
    has_since = any(x in header_text for x in ['since', 'joined', 'tenure'])
    has_independent = 'independent' in header_text or 'independence' in header_text
    has_director_term = 'director' in header_text or 'nominee' in header_text or 'board' in header_text

    # Must have age/since OR (independent + director term)
    if not (has_age or has_since or (has_independent and has_director_term)):
        return -50  # Exclude non-board tables

    score = 0

    # NEW: Huge bonus for having the ideal roster structure
    ideal_structure = has_age and has_independent and (has_since or has_director_term)
    if ideal_structure:
        score += 50  # This should strongly prefer proper roster tables

    # Positive signals
    for term in BOARD_TABLE_POSITIVE:
        if term in header_text:
            score += 3

    # Strong positive: Age AND (Director or Since)
    has_director_or_since = any(x in header_text for x in ['director', 'nominee', 'since'])
    if has_age and has_director_or_since:
        score += 10

    # Also strong positive: Independent AND Director
    if has_independent and has_director_term:
        score += 8

    # +2 per valid name
    score += name_count * 2

    # Negative signals
    for term in BOARD_TABLE_NEGATIVE:
        if term in header_text:
            if term == 'committee' and not has_age and 'director' not in header_text and not has_independent:
                score -= 15
            else:
                score -= 3

    # Size checks
    if len(df) > 20: score -= 10
    if name_count < 5: score -= 5  # Relaxed from 6 to 5
    if name_count > 20: score -= 5

    return score

class ColumnIdentifier:
    @staticmethod
    def identify(df: pd.DataFrame) -> Dict[str, Optional[int]]:
        mapping: Dict[str, Optional[int]] = {
            'name': None,
            'age': None,
            'total_comp': None,
            'independent': None,
            'year': None,
            'since': None
        }

        scores = {col_idx: {k: 0 for k in mapping} for col_idx in range(len(df.columns))}

        for idx, col in enumerate(df.columns):
            h = str(col).lower()
            if any(x in h for x in ['name', 'executive', 'director', 'nominee', 'person']): scores[idx]['name'] += 10
            if 'age' in h: scores[idx]['age'] += 10
            if ('total' in h or 'compensation' in h) and 'subtotal' not in h: scores[idx]['total_comp'] += 8
            if 'total' in h and '$' in h: scores[idx]['total_comp'] += 5
            if 'independent' in h or 'independence' in h: scores[idx]['independent'] += 10
            if 'since' in h or 'joined' in h: scores[idx]['since'] += 10
            if 'year' in h or 'fiscal' in h: scores[idx]['year'] += 10

        sample = df.head(8)
        for idx, col in enumerate(df.columns):
            valid_names = 0
            valid_ages = 0
            valid_money = 0
            valid_years = 0
            valid_bools = 0

            for val in sample.iloc[:, idx]:
                s = str(val).strip()
                if not s or s.lower() == 'nan': continue

                if re.match(r'^\d{2}$', s):
                    try:
                        if 20 <= int(s) <= 99: valid_ages += 1
                    except: pass

                if _MONEY_RE.fullmatch(s):
                    valid_money += 1

                if re.match(r'^(19|20)\d{2}$', s):
                    valid_years += 1

                col_header = str(df.columns[idx]).lower()
                if 'independent' in col_header or 'independence' in col_header:
                    if s.lower() in ['yes', 'no', '✓', 'true', 'false']:
                        valid_bools += 1

                if len(s) > 3 and not any(c.isdigit() for c in s) and len(s.split()) >= 2:
                    if not any(x in s.lower() for x in ['committee', 'board', 'audit', 'meeting', 'llc', 'inc', 'fund', 'trust']):
                        valid_names += 1

            if valid_ages > 2: scores[idx]['age'] += 5
            if valid_money > 2: scores[idx]['total_comp'] += 5
            if valid_years > 2: scores[idx]['year'] += 5
            if valid_bools > 1: scores[idx]['independent'] += 5
            if valid_names > 2: scores[idx]['name'] += 5

        for key in mapping:
            best_score = 0
            best_idx = None
            for idx, s_dict in scores.items():
                if s_dict[key] > best_score and s_dict[key] >= 5:  # FIXED: >= instead of >
                    best_score = s_dict[key]
                    best_idx = idx
            mapping[key] = best_idx

        # NEW: Fallback for name column when headers don't explicitly say "name"
        if mapping['name'] is None:
            best_idx = None
            best_valid = 0
            sample = df.head(8)
            for idx in range(len(df.columns)):
                valid_names = 0
                for val in sample.iloc[:, idx]:
                    s = str(val).strip()
                    if not s or s.lower() == 'nan':
                        continue
                    if len(s) > 3 and not any(c.isdigit() for c in s) and len(s.split()) >= 2:
                        if not any(x in s.lower() for x in ['committee', 'board', 'audit', 'meeting', 'llc', 'inc', 'fund', 'trust']):
                            valid_names += 1
                if valid_names > best_valid:
                    best_valid = valid_names
                    best_idx = idx
            if best_idx is not None and best_valid >= 3:
                mapping['name'] = best_idx

        return mapping

# =============================================================================
# EVENT CLASSIFIER
# =============================================================================

class DynamicEventClassifier:
    def __init__(self) -> None:
        self.sec_items = SEC_8K_ITEMS

    def classify_from_items(self, items: List[str], full_text: str = "", items_source: str = "unknown") -> ClassificationResult:
        substantive = [i for i in items if i not in {"9.01", "8.01"}]
        if not substantive:
            if "8.01" in items: substantive = ["8.01"]
            elif "9.01" in items: substantive = ["9.01"]
        if not substantive: return ClassificationResult("unknown", None, 0.0, [], None, EventCategory.OTHER, items_source)

        candidates = []
        for item_num in substantive:
            item_def = self.sec_items.get(item_num)
            if item_def:
                score, sub_type, matched_kw = self._score_item_match(item_def, full_text.lower())

                # NEW: Gate 8.01 subtype classification
                if item_num == "8.01" and sub_type and len(matched_kw) < 2:
                    # Weak evidence - don't assign subtype
                    sub_type = None
                    score = max(score * 0.5, 0.3)  # Reduce confidence

                snippet = ""
                line_range = None
                if full_text:
                    # NEW: Case-insensitive Item anchor
                    idx = _find_item_anchor(full_text, item_num)
                    if idx is None and matched_kw:
                        idx = full_text.lower().find(matched_kw[0].lower())

                    if idx != -1 and idx is not None:
                        start = max(0, idx - 50)
                        end = min(len(full_text), idx + 250)
                        snippet = full_text[start:end].strip()

                        lines_before = full_text[:start].count('\n')
                        lines_snippet = snippet.count('\n')
                        line_range = [lines_before, lines_before + lines_snippet]

                evidence = Evidence(
                    strategy=items_source if items_source != "unknown" else "regex",
                    snippet=snippet[:300],
                    location=EvidenceLocation(line_range=line_range)
                )

                candidates.append((item_def.weight, score, sub_type, matched_kw, item_def, evidence))

        if not candidates: return ClassificationResult("unknown", None, 0.0, [], None, EventCategory.OTHER, items_source)

        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        weight, score, sub_type, matched_kw, item_def, evidence = candidates[0]
        base_confidence = 0.6 if items_source == "eightk_obj" else 0.4

        return ClassificationResult(
            sub_type or self._category_to_str(item_def.category),
            sub_type,
            min(base_confidence + score * 0.4, 1.0),
            matched_kw,
            item_def,
            item_def.category,
            items_source,
            evidence=evidence
        )

    def _score_item_match(self, item_def: SECItemDefinition, text: str) -> Tuple[float, Optional[str], List[str]]:
        """IMPROVED: Return subtype keywords too for better evidence anchoring."""
        matched = [kw for kw in item_def.keywords if kw.lower() in text]

        best_sub_type, best_sub_score, best_sub_matches = None, 0, []
        for sub_type, sub_keywords in item_def.sub_types.items():
            sub_matches = [kw for kw in sub_keywords if kw.lower() in text]
            if len(sub_matches) > best_sub_score:
                best_sub_score = len(sub_matches)
                best_sub_type = sub_type
                best_sub_matches = sub_matches

        total_kw = len(item_def.keywords) + sum(len(kws) for kws in item_def.sub_types.values())
        combined_matches = list(set(matched + best_sub_matches))
        score = (len(combined_matches) / total_kw) if total_kw > 0 else 0.5
        return score, best_sub_type, combined_matches

    def _category_to_str(self, cat: EventCategory) -> str:
        return cat.name.lower()

# =============================================================================
# NAME EXTRACTOR
# =============================================================================

class DynamicNameExtractor:
    def extract_names(self, text: str) -> List[ExtractedName]:
        names_found: List[ExtractedName] = []
        seen_names: Set[str] = set()

        age_patterns = [
            r'\b([A-Z][a-z]+(?:[ \t]+[A-Z]\.?)?[ \t]+[A-Z][a-z]+)\s*,\s*age\s*(\d{2})\b',
            r'\b([A-Z][a-z]+(?:[ \t]+[A-Z]\.?)?[ \t]+[A-Z][a-z]+)\s*\(\s*(\d{2})\s*\)',
            # NEW: Flexible pattern for "Name ... Age: 52" or "Name ... Age 52" within 60 chars
            r'\b([A-Z][a-z]+(?:[ \t]+[A-Z]\.?)?[ \t]+[A-Z][a-z]+)\b.{0,60}?\bage[:\s]+(\d{2})\b',
        ]
        for pattern in age_patterns:
            for match in re.finditer(pattern, text):
                name = match.group(1).strip()
                try: age = int(match.group(2))
                except: age = None

                if self._validate_name(name) and name.lower() not in seen_names:
                    seen_names.add(name.lower())

                    start, end = match.span()
                    snippet_start = max(0, start - 50)
                    snippet_end = min(len(text), end + 50)
                    snippet = text[snippet_start:snippet_end].strip()

                    lines_before = text[:snippet_start].count('\n')
                    line_range = [lines_before, lines_before + snippet.count('\n')]

                    evidence = Evidence(
                        strategy="regex",
                        snippet=snippet[:300],
                        location=EvidenceLocation(line_range=line_range)
                    )

                    names_found.append(ExtractedName(full_name=name, confidence=0.9, age=age, evidence=evidence))

        return names_found

    def extract_name_from_cell(self, cell_value: str) -> Optional[ExtractedName]:
        if not cell_value or not isinstance(cell_value, str):
            return None

        name = clean_name_cell(cell_value)

        if not self._validate_name(name):
            return None

        age = None
        age_match = re.search(r'\((\d{2})\)|\s+(\d{2})\s*$', cell_value)
        if age_match:
            try:
                age = int(age_match.group(1) or age_match.group(2))
                if not (20 <= age <= 99):
                    age = None
            except:
                age = None

        return ExtractedName(
            full_name=name,
            confidence=0.9,
            age=age
        )

    def extract_names_generic(self, text: str) -> List[ExtractedName]:
        """
        Extract names without requiring age context.
        Strict: still uses _validate_name().
        Used for independence lists that don't include ages.
        """
        names_found: List[ExtractedName] = []
        seen: Set[str] = set()

        # Supports: First Last, First M. Last, First Middle Last, suffix Jr.
        pattern = re.compile(
            r'\b([A-Z][a-z]+(?:\s+[A-Z]\.)?(?:\s+[A-Z][a-z]+){1,3}(?:\s+Jr\.)?)\b'
        )

        for m in pattern.finditer(text):
            name = m.group(1).strip()
            if not self._validate_name(name):
                continue

            key = name.lower()
            if key in seen:
                continue
            seen.add(key)

            start, end = m.span()
            snippet_start = max(0, start - 80)
            snippet_end = min(len(text), end + 80)
            snippet = text[snippet_start:snippet_end].strip()

            lines_before = text[:snippet_start].count("\n")
            evidence = Evidence(
                strategy="regex_generic",
                snippet=snippet[:300],
                location=EvidenceLocation(line_range=[lines_before, lines_before + snippet.count("\n")])
            )

            names_found.append(ExtractedName(full_name=name, confidence=0.75, evidence=evidence))

        return names_found

    def _validate_name(self, name: str) -> bool:
        if not name or len(name) < 4 or len(name) > 60: return False
        parts = name.split()
        if len(parts) < 2 or len(parts) > 5: return False

        blocked = {
            'stock', 'plan', 'proxy', 'statement', 'form', 'total', 'salary', 'bonus', 'pay', 'ratio',
            'the', 'and', 'for', 'of', 'to', 'in', 'on', 'with', 'values', 'framework', 'engagement',
            'conduct', 'policy', 'oversight', 'governance', 'compensation', 'business', 'progress',
            'highlights', 'summary', 'across', 'our', 'officers', 'committee', 'board', 'llc', 'inc',
            'l.p.', 'fund', 'trust', 'group', 'investment', 'aggregate', 'associates', 'advisors',
            'holdings', 'capital', 'management', 'partners', 'power', 'voting', 'beneficial', 'ownership'
        }
        if any(p.lower() in blocked for p in parts): return False
        if any(c.isdigit() for c in name): return False

        capital_words = sum(1 for p in parts if len(p) > 0 and p[0].isupper())
        if capital_words < 2: return False

        return True

# =============================================================================
# FILING-GROUNDED VALIDATOR
# =============================================================================

class FilingGroundedValidator:
    @staticmethod
    def validate_compensation(name: str, amount: float, evidence: Optional[Evidence]) -> Tuple[bool, List[str]]:
        alerts = []

        if not evidence:
            alerts.append(f"Missing evidence for {name}'s compensation of ${amount:,.0f}")
            return False, alerts

        if not evidence.validate_name(name):
            alerts.append(f"Evidence snippet does not contain name '{name}'")
            return False, alerts

        if not evidence.validate_amount(amount):
            alerts.append(f"Evidence does not contain compensation amount ${amount:,.0f}")
            return False, alerts

        if not evidence.validate_label('total') and not evidence.validate_label('compensation'):
            alerts.append(f"Evidence for {name} lacks 'Total' or 'Compensation' label")
            return False, alerts

        return True, alerts

    @staticmethod
    def validate_board_member(name: str, independent: Optional[bool], evidence: Optional[Evidence],
                             age: Optional[int] = None, tenure: Optional[str] = None) -> Tuple[bool, List[str]]:
        alerts = []

        if not evidence:
            alerts.append(f"Missing evidence for board member {name}")
            return False, alerts

        if not evidence.validate_name(name):
            alerts.append(f"Evidence snippet does not contain name '{name}'")
            return False, alerts

        # NEW: Require at least one attribute for table-derived board members
        if age is None and tenure is None and independent is None:
            alerts.append(f"Board member {name} has no age/tenure/independence fields")
            return False, alerts

        if independent is not None:
            if evidence.column_header:
                if 'independent' not in evidence.column_header.lower():
                    alerts.append(f"Independence claimed for {name} but column header is '{evidence.column_header}'")
                    return False, alerts
            else:
                has_independent_label = 'independent' in evidence.snippet.lower()
                if not has_independent_label:
                    alerts.append(f"Independence claimed for {name} but evidence lacks 'independent' label")
                    return False, alerts

        return True, alerts

# =============================================================================
# TRUTH TESTS
# =============================================================================

def run_truth_tests(executives: List[Dict[str, Any]], board_members: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
    """NEW: Truth tests to ensure data integrity."""
    alerts = []

    # Test 1: Table evidence must contain the extracted name
    for e in executives:
        if e.get("total_compensation") is not None and e.get("evidence"):
            try:
                ev = Evidence(**e["evidence"])
                if not ev.validate_name(e["name"]):
                    alerts.append(f"TRUTH TEST FAILED: Exec {e['name']} compensation without name in evidence")
            except: pass

    # Test 2: If compensation is non-null, evidence must validate total label
    for e in executives:
        if e.get("total_compensation") is not None and e.get("evidence"):
            try:
                ev = Evidence(**e["evidence"])
                if not ev.validate_label("total"):
                    alerts.append(f"TRUTH TEST FAILED: Exec {e['name']} compensation without 'total' label")
            except: pass

    # Test 3: Board members must not use compensation table evidence
    for m in board_members:
        if m.get("extraction_method") == "table" and m.get("evidence"):
            snippet_lower = m["evidence"]["snippet"].lower()
            if any(k in snippet_lower for k in ["salary", "bonus", "stock award", "total ($)"]):
                alerts.append(f"TRUTH TEST FAILED: Board member {m['name']} extracted from compensation table")

    # Test 4: If independent is asserted, evidence must contain 'independent' (snippet or header)
    for m in board_members:
        if m.get("independent") in (True, False) and m.get("evidence"):
            snip = (m["evidence"].get("snippet") or "").lower()
            hdr = (m["evidence"].get("column_header") or "").lower()
            if "independent" not in snip and "independent" not in hdr:
                alerts.append(
                    f"TRUTH TEST FAILED: {m['name']} has independence asserted without 'independent' in evidence"
                )

    passed = len(alerts) == 0
    return passed, alerts

# =============================================================================
# MAIN PARSER CLASS
# =============================================================================

class SECFilingParser:
    def __init__(self, name: Optional[str] = None, email: Optional[str] = None) -> None:
        self.name, self.email = name or NAME, email or SEC_EMAIL
        self.event_classifier = DynamicEventClassifier()
        self.name_extractor = DynamicNameExtractor()
        self.validator = FilingGroundedValidator()

    def _set_identity(self) -> None:
        set_identity(f"{self.name} {self.email}")

    def _generate_quality_report(self, results: List[Dict[str, Any]], truth_tests_passed: bool, truth_alerts: List[str]) -> Dict[str, Any]:
        if not results:
            return ExtractionQuality().__dict__

        missing_evidence = 0
        total_confidence = 0.0
        covered_count = 0
        validation_failures = 0
        fields_nullified = 0

        for res in results:
            if 'event_type' in res:
                if res.get('evidence'):
                    covered_count += 1
                else:
                    missing_evidence += 1
                total_confidence += res.get('confidence', 0.0)

            # Count validated items (not candidates)
            validated_items = res.get('executives', []) or res.get('board_members', [])
            if validated_items:
                for item in validated_items:
                    if item.get('evidence'):
                        covered_count += 1
                        if 'validation_passed' in item and not item['validation_passed']:
                            validation_failures += 1
                        if item.get('field_nullified'):
                            fields_nullified += 1
                    else:
                        missing_evidence += 1

        total_items = covered_count + missing_evidence
        evidence_coverage = (covered_count / total_items) if total_items > 0 else 0.0
        avg_confidence = (total_confidence / len([r for r in results if 'confidence' in r])) if any('confidence' in r for r in results) else 0.0

        return {
            "assertions_missing_evidence": missing_evidence,
            "evidence_coverage_pct": round(evidence_coverage * 100, 2),
            "eight_k_avg_confidence": round(avg_confidence, 2),
            "evidence_validation_failures": validation_failures,
            "fields_nullified_by_validation": fields_nullified,
            "validation_assertions_passed": truth_tests_passed,
            "truth_test_alerts": truth_alerts
        }

    def extract_8k_events(self, ticker: str, limit: int = 10) -> Dict[str, Any]:
        try:
            self._set_identity()
            company = Company(ticker)
            filings = company.get_filings(form='8-K')
            if not filings: return {'ticker': ticker, 'success': False, 'error': 'No 8-K found'}
            events_by_date = {}
            for i, filing in enumerate(filings):
                if i >= limit: break
                text = safe_get_text(filing)
                items, source = safe_get_8k_items(filing)
                if not items:
                    items = normalize_item_numbers(re.findall(r'Item\s+(\d+\.\d+)', text, re.I))
                    source = "regex"

                cls = self.event_classifier.classify_from_items(items, text, source)
                date_key = safe_get_filing_date(filing) or f"unknown_{i}"

                orig_date = date_key
                k = 1
                while date_key in events_by_date:
                    date_key = f"{orig_date}_{k}"
                    k += 1

                events_by_date[date_key] = {
                    'event_type': cls.event_type,
                    'category': cls.category.name if cls.category else None,
                    'sec_items': items,
                    'items_source': source,
                    'confidence': cls.confidence,
                    'evidence': asdict(cls.evidence) if cls.evidence else None,
                    'text': text  # Full 8-K text (no truncation)
                }

            truth_passed, truth_alerts = True, []
            quality = self._generate_quality_report(list(events_by_date.values()), truth_passed, truth_alerts)

            return {
                'ticker': ticker,
                'events_by_date': events_by_date,
                'total_events': len(events_by_date),
                'success': True,
                'quality_report': quality
            }
        except Exception as e: return {'ticker': ticker, 'success': False, 'error': str(e)}

    def extract_proxy_compensation(self, ticker: str) -> Dict[str, Any]:
        try:
            self._set_identity()
            filings = Company(ticker).get_filings(form='DEF 14A')
            if not filings: return {'ticker': ticker, 'executives': [], 'candidates': [], 'success': False, 'error': 'No DEF 14A found'}
            latest = filings[0]
            html, text = safe_get_html(latest), safe_get_text(latest)

            execs = self._extract_compensation_from_tables(html)

            if not execs:
                execs = self._extract_compensation_from_text(text)

            unique_execs = {}
            for e in execs:
                name_key = e['name'].lower()
                if name_key not in unique_execs or e['total_compensation'] > unique_execs[name_key]['total_compensation']:
                    unique_execs[name_key] = e

            final_execs = sorted(unique_execs.values(), key=lambda x: x['total_compensation'], reverse=True)

            # NEW: Separate validated facts from candidates
            validated, candidates = [], []
            for exec_data in final_execs:
                evidence = Evidence(**exec_data['evidence']) if exec_data.get('evidence') else None
                passed, alerts = self.validator.validate_compensation(
                    exec_data['name'],
                    exec_data['total_compensation'],
                    evidence
                )
                exec_data['validation_passed'] = passed
                exec_data['validation_alerts'] = alerts

                if passed:
                    validated.append(exec_data)
                else:
                    # Move to candidates
                    exec_data['total_compensation_raw'] = exec_data['total_compensation']
                    exec_data['total_compensation'] = None
                    exec_data['field_nullified'] = True
                    candidates.append(exec_data)

            # Run truth tests
            truth_passed, truth_alerts = run_truth_tests(validated, [])

            quality = self._generate_quality_report([{'executives': validated}], truth_passed, truth_alerts)

            all_alerts = []
            for exec_data in candidates:
                all_alerts.extend(exec_data.get('validation_alerts', []))
            quality['potential_error_alerts'] = all_alerts

            return {
                'ticker': ticker,
                'executives': validated,  # NEW: Only validated facts
                'candidates': candidates,  # NEW: Unvalidated entries
                'success': len(validated) > 0,
                'quality_report': quality
            }
        except Exception as e: return {'ticker': ticker, 'success': False, 'error': str(e)}

    def _extract_compensation_from_tables(self, html: str) -> List[Dict[str, Any]]:
        best_execs = []
        best_score = 0

        if not html: return best_execs
        try:
            try:
                dfs = pd.read_html(StringIO(html))
            except:
                dfs = pd.read_html(StringIO(html), flavor='html5lib')

            for table_idx, df in enumerate(dfs):
                df = flatten_dataframe(df)

                if self._is_beneficial_ownership_table(df):
                    continue

                is_summary_comp, indicator_count = is_summary_compensation_table(df)
                if not is_summary_comp:
                    continue

                cols = ColumnIdentifier.identify(df)

                if cols['name'] is None or cols['total_comp'] is None: continue

                target_year = None
                if cols['year'] is not None:
                    try:
                        years = pd.to_numeric(df.iloc[:, cols['year']], errors='coerce').dropna()
                        if not years.empty:
                            target_year = int(years.max())
                    except: pass

                current_table: List[Dict[str, Any]] = []
                for row_idx, row in df.iterrows():
                    try:
                        if target_year and cols['year'] is not None:
                            try:
                                row_year = int(float(str(row.iloc[cols['year']]).strip()))
                                if row_year != target_year: continue
                            except: pass

                        name_raw = str(row.iloc[cols['name']])

                        name_obj = self.name_extractor.extract_name_from_cell(name_raw)
                        if not name_obj:
                            continue

                        name = name_obj.full_name

                        val_raw = str(row.iloc[cols['total_comp']])
                        val_str = re.sub(r'[^\d.]', '', val_raw)
                        if not val_str: continue
                        total = float(val_str)

                        if VALIDATION_CONFIG['min_compensation'] <= total <= VALIDATION_CONFIG['max_compensation']:
                            # NEW: Structured evidence with key-value pairs
                            row_snippet, row_kv = create_table_evidence(row, df)
                            ev_row_idx = int(row_idx) if isinstance(row_idx, (int, float, np.integer)) else None

                            total_comp_header = str(df.columns[cols['total_comp']]) if cols['total_comp'] is not None else None

                            evidence = Evidence(
                                strategy="table_parse",
                                snippet=row_snippet[:500],
                                location=EvidenceLocation(
                                    section_hint=f"Summary Compensation Table ({target_year})" if target_year else "Summary Compensation Table",
                                    table_index=table_idx,
                                    row_index=ev_row_idx
                                ),
                                column_header=total_comp_header,
                                table_kv=row_kv  # NEW: Structured data
                            )
                            current_table.append({
                                'name': name,
                                'total_compensation': total,
                                'extraction_method': 'table',
                                'year': target_year,
                                'evidence': asdict(evidence)
                            })
                    except: pass

                if current_table and indicator_count > best_score:
                    best_score = indicator_count
                    best_execs = current_table

        except Exception as e:
            pass
        return best_execs

    def _is_beneficial_ownership_table(self, df: pd.DataFrame) -> bool:
        try:
            header_text = ' '.join(str(c).lower() for c in df.columns)

            if 'beneficial owner' in header_text or 'beneficial ownership' in header_text:
                return True

            ownership_keywords = [
                'beneficial', 'ownership', 'voting power', 'shares', 'class a', 'class b',
                'percent', 'aggregate', 'sole', 'shared', '% of class'
            ]

            matches = sum(1 for k in ownership_keywords if k in header_text)

            if matches >= 3:
                return True

            if 'shares' in header_text and ('%' in header_text or 'voting' in header_text):
                return True

            return False
        except:
            return False

    def _is_matrix_table(self, df: pd.DataFrame) -> bool:
        """
        Detect matrix/grid tables where both headers and values contain names.
        These are typically committee assignment matrices or voting grids.

        Examples:
        - Committee membership matrices
        - Director voting/election grids
        - Cross-reference tables
        """
        try:
            if len(df) < 2 or len(df.columns) < 2:
                return False

            # Check if multiple column headers look like person names
            header_names = 0
            for col in df.columns:
                col_str = str(col).strip()
                if len(col_str) > 5 and len(col_str.split()) >= 2:
                    # Check if it looks like a person name (has capital letters, no obvious keywords)
                    if not any(x in col_str.lower() for x in ['committee', 'independent', 'age', 'since', 'nominee', 'class', 'total', 'shares']):
                        # Basic name pattern: multiple words with capitals
                        words = col_str.split()
                        if sum(1 for w in words if w and w[0].isupper()) >= 2:
                            header_names += 1

            # If 2+ headers look like names, check if cells also contain names
            if header_names >= 2:
                # Sample first 3 rows
                cell_names = 0
                for idx, row in df.head(3).iterrows():
                    for val in row.values:
                        val_str = str(val).strip()
                        if len(val_str) > 5 and len(val_str.split()) >= 2:
                            # Looks like a name (multiple capitalized words)
                            words = val_str.split()
                            if sum(1 for w in words if w and len(w) > 0 and w[0].isupper()) >= 2:
                                cell_names += 1

                # If we have name-like headers AND name-like cell values, it's a matrix
                if cell_names >= 3:
                    return True

            return False
        except:
            return False

    def _is_any_compensation_table(self, df: pd.DataFrame) -> bool:
        try:
            header_text = ' '.join(str(c).lower() for c in df.columns)

            # IMPROVED: Include director compensation indicators
            comp_keywords = [
                'salary', 'bonus', 'stock award', 'option award',
                'non-equity', 'all other compensation', 'total ($)',
                'fees earned', 'total compensation'
            ]

            hits = sum(1 for k in comp_keywords if k in header_text)

            return hits >= 3
        except:
            return False

    def _extract_compensation_from_text(self, text: str) -> List[Dict[str, Any]]:
        execs = []

        section_match = re.search(r'(?i)summary compensation table.*?(?=\n\s*\n[A-Z][A-Z\s]{20,}|\Z)', text, re.DOTALL)
        search_text = section_match.group(0) if section_match else text

        names = self.name_extractor.extract_names(search_text)
        for name_obj in names:
            comp_res = self._extract_row_context_compensation(name_obj.full_name, search_text)
            if comp_res:
                execs.append({
                    'name': name_obj.full_name,
                    'total_compensation': comp_res['total_compensation'],
                    'extraction_method': 'text',
                    'evidence': comp_res['evidence']
                })
        return execs

    def _extract_row_context_compensation(self, name: str, text: str) -> Optional[Dict[str, Any]]:
        try:
            lines = text.splitlines()
            for i, line in enumerate(lines):
                if name.lower() in line.lower():
                    start_idx = max(0, i-2)
                    end_idx = min(len(lines), i+5)
                    ctx_lines = lines[start_idx:end_idx]
                    ctx = ' '.join(ctx_lines)

                    matches = list(_MONEY_RE.finditer(ctx))
                    if matches:
                        for match in matches:
                            amount_str = match.group()
                            amount_pos = match.start()

                            line_start = max(0, ctx.rfind('\n', 0, amount_pos))
                            prev_line_start = max(0, ctx.rfind('\n', 0, line_start - 1))

                            search_text = ctx[prev_line_start:amount_pos].lower()

                            has_total = 'total' in search_text
                            has_comp_context = 'compensation' in search_text.lower() or 'summary compensation' in ctx[:amount_pos].lower()

                            if has_total and has_comp_context:
                                try:
                                    val = float(amount_str.replace('$', '').replace(',', ''))
                                    if VALIDATION_CONFIG['min_compensation'] <= val <= VALIDATION_CONFIG['max_compensation']:
                                        evidence = Evidence(
                                            strategy="row_context",
                                            snippet=ctx[:400],
                                            location=EvidenceLocation(line_range=[start_idx, end_idx])
                                        )
                                        return {
                                            'total_compensation': val,
                                            'evidence': asdict(evidence)
                                        }
                                except: pass
        except: pass
        return None

    def _extract_independent_directors_from_text(self, text: str) -> Dict[str, Evidence]:
        """
        Returns {normalized_name: Evidence} for directors explicitly stated as independent.
        Safe: runs regex only on small windows around relevant phrases.
        """
        out: Dict[str, Evidence] = {}
        if not text:
            return out

        # Build windows around likely independence mentions
        needles = ["independent director", "director independence", "independence", "the board has determined"]
        lower = text.lower()

        hits: List[int] = []
        for n in needles:
            start = 0
            while True:
                idx = lower.find(n, start)
                if idx == -1:
                    break
                hits.append(idx)
                start = idx + len(n)

        windows: List[str] = []
        if hits:
            for idx in hits[:30]:  # cap windows
                a = max(0, idx - 3000)
                b = min(len(text), idx + 3000)
                windows.append(text[a:b])
        else:
            # fallback: first chunk only (still bounded)
            windows = [text[:120_000]]

        patterns = [
            # "The Board has determined ... independent directors are: X, Y, Z"
            r'(?i)(?:the\s+board\s+(?:of\s+directors\s+)?has\s+determined|we\s+have\s+determined|the\s+board\s+determined)'
            r'(?:.{0,300}?)(?:following|the\s+following)(?:.{0,120}?)independent\s+director[s]?\s*(?:are|include|:|—|–)\s*'
            r'([A-Z][^\.\n]{10,450}?)(?:\.\s|\n\n|\Z)',

            # "Independent directors are: X, Y, Z"
            r'(?i)\bindependent\s+director[s]?\s*(?:are|include|:|—|–)\s*'
            r'([A-Z][^\.\n]{10,450}?)(?:\.\s|\n\n|\Z)',

            # "The following directors are independent: X, Y, Z"
            r'(?i)(?:the\s+)?following\s+directors?\s+(?:are|is)\s+independent\s*[:—–]?\s*'
            r'([A-Z][^\.\n]{10,450}?)(?:\.\s|\n\n|\Z)',
        ]

        for w in windows:
            for pat in patterns:
                try:
                    for m in re.finditer(pat, w):
                        chunk = m.group(1)
                        if len(chunk) > 700:
                            continue

                        names = self.name_extractor.extract_names(chunk)
                        if not names:
                            names = self.name_extractor.extract_names_generic(chunk)

                        if not names:
                            continue

                        start, end = m.span()
                        snippet = w[max(0, start - 150):min(len(w), end + 150)].strip()
                        lines_before = w[:max(0, start - 150)].count('\n')

                        ev = Evidence(
                            strategy="independence_list",
                            snippet=snippet[:400],
                            location=EvidenceLocation(line_range=[lines_before, lines_before + snippet.count('\n')])
                        )

                        for n in names:
                            out[norm_name_key(n.full_name)] = ev
                except Exception:
                    # Skip problematic patterns
                    continue

        return out

    def extract_governance_data(self, ticker: str, debug: bool = False) -> Dict[str, Any]:
        try:
            self._set_identity()
            filings = Company(ticker).get_filings(form='DEF 14A')
            if not filings: return {'ticker': ticker, 'board_members': [], 'candidates': [], 'success': False}
            latest = filings[0]
            html, text = safe_get_html(latest), safe_get_text(latest)

            members, board_df = self._extract_board_from_tables(html, debug=debug)

            if len(members) < 3:
                text_members = self._extract_board_from_text(text)
                existing_names = {m['name'].lower() for m in members}
                for tm in text_members:
                    if tm['name'].lower() not in existing_names:
                        members.append(tm)

            unique_members = {m['name'].lower(): m for m in members}.values()
            final_members = list(unique_members)

            # NEW: Extract raw table text for LLM processing from DataFrame
            raw_table_text = ""
            if board_df is not None:
                # Convert DataFrame to clean text representation
                # Replace NaN with empty string and clean up spacing
                df_clean = board_df.fillna('')
                raw_table_text = df_clean.to_string(index=False, max_colwidth=100)

                # Clean up excessive whitespace
                lines = raw_table_text.split('\n')
                cleaned_lines = []
                for line in lines:
                    # Remove excessive spaces
                    cleaned = ' '.join(line.split())
                    if cleaned:  # Skip empty lines
                        cleaned_lines.append(cleaned)
                raw_table_text = '\n'.join(cleaned_lines)

            if debug:
                print(f"\n[DEBUG] System status:")
                print(f"  BeautifulSoup available: {BeautifulSoup is not None}")
                if BeautifulSoup is None:
                    print(f"  ⚠️  Install BeautifulSoup for HTML icon extraction:")
                    print(f"      pip install beautifulsoup4 lxml --break-system-packages")

                print(f"\n[DEBUG] After table extraction:")
                ind_counts = {"True": 0, "False": 0, "None": 0}
                for m in final_members:
                    ind_val = m.get('independent')
                    if ind_val is True:
                        ind_counts["True"] += 1
                    elif ind_val is False:
                        ind_counts["False"] += 1
                    else:
                        ind_counts["None"] += 1
                print(f"  Independence: {ind_counts['True']} True, {ind_counts['False']} False, {ind_counts['None']} None")
                if ind_counts["None"] > 0 and BeautifulSoup is None:
                    print(f"  ⚠️  WARNING: {ind_counts['None']} directors with unknown independence")
                    print(f"  ⚠️  BeautifulSoup NOT installed - HTML icons cannot be extracted")
                    print(f"  ⚠️  To fix: pip install beautifulsoup4 lxml --break-system-packages")

            # NEW: Fill independence with explicit text evidence when available
            ind_map = self._extract_independent_directors_from_text(text)

            if debug:
                print(f"\n[DEBUG] Text independence extraction:")
                print(f"  ind_map size: {len(ind_map)}")
                if ind_map:
                    print(f"  ind_map keys: {list(ind_map.keys())[:10]}")  # Show first 10
                print(f"  final_members count: {len(final_members)}")
                print(f"  final_members names: {[m['name'] for m in final_members]}")

            for m in final_members:
                if m.get("independent") is None:
                    key = norm_name_key(m["name"])  # Use normalized key for matching
                    if key in ind_map:
                        m["independent"] = True
                        m["evidence"] = asdict(ind_map[key])  # Overwrite with better evidence for independence
                        m["extraction_method"] = (m.get("extraction_method") or "table") + "+text_independence"
                        if debug:
                            print(f"  ✓ Enriched {m['name']} with independence=True")

            # NEW: Separate validated facts from candidates
            validated, candidates = [], []
            for member_data in final_members:
                evidence = Evidence(**member_data['evidence']) if member_data.get('evidence') else None
                passed, alerts = self.validator.validate_board_member(
                    member_data['name'],
                    member_data.get('independent'),
                    evidence,
                    age=member_data.get('age'),
                    tenure=member_data.get('tenure')
                )
                member_data['validation_passed'] = passed
                member_data['validation_alerts'] = alerts

                if passed:
                    validated.append(member_data)
                else:
                    # Move to candidates
                    if member_data.get('independent') is not None:
                        member_data['independent_raw'] = member_data['independent']
                        member_data['independent'] = None
                        member_data['field_nullified'] = True
                    candidates.append(member_data)

            # Run truth tests
            truth_passed, truth_alerts = run_truth_tests([], validated)

            quality = self._generate_quality_report([{'board_members': validated}], truth_passed, truth_alerts)

            all_alerts = []
            for member_data in candidates:
                all_alerts.extend(member_data.get('validation_alerts', []))
            quality['potential_error_alerts'] = all_alerts

            return {
                'ticker': ticker,
                'board_members': validated,  # NEW: Only validated facts
                'candidates': candidates,  # NEW: Unvalidated entries
                'success': len(validated) > 0,
                'governance_metrics': self._calc_gov_metrics(validated),
                'quality_report': quality,
                'raw_table_text': raw_table_text  # NEW: Raw table for LLM processing
            }
        except Exception as e: return {'ticker': ticker, 'success': False, 'error': str(e)}

    def _html_board_independence_map(
        self,
        html: str,
        table_idx: int,
        name_col_idx: Optional[int] = None,
        indep_col_idx: Optional[int] = None,
        debug: bool = False
    ) -> Dict[str, Optional[bool]]:
        """
        Parse the raw HTML table to recover Independence icons that pandas.read_html drops.
        Returns {norm_name_key(name): True/False/None}.
        Rule: treat blank as False only if we observed at least one True in the column.

        Args:
            html: Raw HTML
            table_idx: Table index (from pandas)
            name_col_idx: Name column index from pandas (optional, will auto-detect if None)
            indep_col_idx: Independence column index from pandas (optional, will auto-detect if None)
            debug: Print debug output
        """
        out: Dict[str, Optional[bool]] = {}
        if not html:
            if debug:
                print(f"    [HTML Icon] No HTML provided")
            return out

        if BeautifulSoup is None:
            if debug:
                print(f"    [HTML Icon] BeautifulSoup not installed")
            # BeautifulSoup not installed - cannot parse HTML icons
            return out

        try:
            soup = BeautifulSoup(html, "lxml")
        except:
            try:
                soup = BeautifulSoup(html, "html.parser")
            except:
                if debug:
                    print(f"    [HTML Icon] Failed to parse HTML")
                return out

        tables = soup.find_all("table")
        if debug:
            print(f"    [HTML Icon] Found {len(tables)} tables, looking for table_idx={table_idx}")

        if table_idx < 0 or table_idx >= len(tables):
            if debug:
                print(f"    [HTML Icon] Table index out of range")
            return out

        table = tables[table_idx]
        rows = table.find_all("tr")
        if not rows:
            if debug:
                print(f"    [HTML Icon] No rows in table")
            return out

        # Initialize to satisfy type checker
        header_row_idx: int = 0
        name_idx: Optional[int]
        indep_idx: Optional[int]

        # Use pandas column indices if provided, otherwise try to detect
        if name_col_idx is not None and indep_col_idx is not None:
            name_idx = name_col_idx
            indep_idx = indep_col_idx
            header_row_idx = 0  # Assume first row is header

            if debug:
                print(f"    [HTML Icon] Using pandas column indices: name={name_idx}, independent={indep_idx}")
        else:
            # Fallback: try to detect headers (original logic)
            headers = []

            # Try first 3 rows to find headers
            for row_idx in range(min(3, len(rows))):
                header_cells = rows[row_idx].find_all(["th", "td"])
                test_headers = []

                for cell in header_cells:
                    # Try multiple extraction methods
                    txt = cell.get_text(" ", strip=True)

                    # If empty, try getting text from all descendant elements
                    if not txt:
                        txt = " ".join(el.get_text(strip=True) for el in cell.find_all(string=True))

                    test_headers.append(txt.lower() if txt else "")

                # Check if this row has actual headers (not empty, has key terms)
                non_empty = sum(1 for h in test_headers if h)
                has_key_terms = any(term in " ".join(test_headers) for term in ["name", "age", "director", "independent", "since"])

                if non_empty >= 3 and has_key_terms:
                    headers = test_headers
                    header_row_idx = row_idx
                    break

            # Fallback: use first row even if mostly empty
            if not headers:
                header_cells = rows[0].find_all(["th", "td"])
                headers = [c.get_text(" ", strip=True).lower() for c in header_cells]
                header_row_idx = 0

            if debug:
                print(f"    [HTML Icon] Header row: {header_row_idx}, headers: {headers[:10]}")  # First 10

            name_idx = None
            indep_idx = None
            for i, h in enumerate(headers):
                if indep_idx is None and ("independent" in h or "independence" in h):
                    indep_idx = i
                if name_idx is None and ("name" in h or "director" in h or "nominee" in h):
                    name_idx = i

            if debug:
                print(f"    [HTML Icon] name_idx={name_idx}, indep_idx={indep_idx}")

            if indep_idx is None or name_idx is None:
                if debug:
                    print(f"    [HTML Icon] Missing name or independence column")
                return out

            # Type checker: name_idx and indep_idx are guaranteed to be int here
            assert name_idx is not None and indep_idx is not None

        saw_true = False
        pending_blank: List[str] = []

        def cell_indicator(cell) -> str:
            # Prefer explicit text
            txt = cell.get_text(" ", strip=True)
            if txt:
                return txt

            # Try aria/alt/title (common for icons)
            for attr in ["aria-label", "title"]:
                v = cell.get(attr)
                if v:
                    return str(v)

            img = cell.find("img")
            if img:
                for attr in ["alt", "title", "aria-label"]:
                    v = img.get(attr)
                    if v:
                        return str(v)
                return "✓"  # icon present

            svg = cell.find("svg")
            if svg:
                return "✓"  # icon present

            # Any span with class hinting check
            span = cell.find("span")
            if span:
                cls = " ".join(span.get("class", []))
                if "check" in cls.lower() or "tick" in cls.lower():
                    return "✓"

            return ""

        extracted_count = 0
        total_rows = 0
        skipped_header = 0
        too_few_cells = 0
        no_name = 0

        for row_idx, tr in enumerate(rows):
            total_rows += 1

            # Skip header row
            if row_idx <= header_row_idx:
                skipped_header += 1
                continue

            cells = tr.find_all(["td", "th"])

            if debug and row_idx == header_row_idx + 1:  # First data row
                print(f"    [HTML Icon] First data row has {len(cells)} cells (need > {max(indep_idx, name_idx)})")

            if len(cells) <= max(indep_idx, name_idx):
                too_few_cells += 1
                continue

            raw_name = cells[name_idx].get_text(" ", strip=True)
            name = clean_name_cell(raw_name)
            if not name:
                no_name += 1
                continue

            ind_raw = cell_indicator(cells[indep_idx])
            ind = parse_independent_cell(ind_raw)

            key = norm_name_key(name)

            if debug and extracted_count < 3:  # Show first 3 directors
                print(f"    [HTML Icon] Sample row {extracted_count + 1}: name='{name}', ind_raw='{ind_raw}', ind={ind}")
                extracted_count += 1

            if ind is True:
                saw_true = True
                out[key] = True
            elif ind is False:
                out[key] = False
            else:
                # blank/unknown for now, may become False if column is binary
                pending_blank.append(key)
                out[key] = None

        # If we saw at least one True marker, interpret blanks as No (common roster pattern)
        if saw_true:
            for k in pending_blank:
                if out.get(k) is None:
                    out[k] = False

        if debug:
            print(f"    [HTML Icon] Row processing: {total_rows} total rows, {skipped_header} header, {too_few_cells} too few cells, {no_name} no name")
            print(f"    [HTML Icon] Final: extracted {len(out)} entries, saw_true={saw_true}, pending_blank={len(pending_blank)}")
            if out:
                sample = list(out.items())[:3]
                print(f"    [HTML Icon] Sample entries: {sample}")

        return out

    def _extract_raw_board_table_text(self, html: str, table_idx: int) -> str:
        """
        Extract raw text representation of the board table for LLM processing.
        Returns a formatted text version of the table that an LLM can easily parse.
        """
        if not html or BeautifulSoup is None:
            return ""

        try:
            soup = BeautifulSoup(html, "lxml")
        except:
            try:
                soup = BeautifulSoup(html, "html.parser")
            except:
                return ""

        tables = soup.find_all("table")
        if table_idx < 0 or table_idx >= len(tables):
            return ""

        table = tables[table_idx]

        # Extract as clean text
        text_lines = []
        for row in table.find_all("tr"):
            cells = row.find_all(["td", "th"])
            row_text = " | ".join(cell.get_text(" ", strip=True) for cell in cells)
            if row_text.strip():
                text_lines.append(row_text)

        return "\n".join(text_lines)

    def _extract_board_from_tables(self, html: str, debug: bool = False) -> tuple[List[Dict[str, Any]], Optional[Any]]:
        """
        Extract board members from HTML tables.
        Returns: (members_list, best_dataframe)
        """
        best_members = []
        best_score = -10_000  # FIXED: Allow negative-scoring tables to win
        best_df = None  # NEW: Track the best DataFrame
        fallback_candidates = []  # NEW: Track tables that almost qualified

        if not html: return best_members, None
        try:
            try:
                dfs = pd.read_html(StringIO(html))
            except:
                dfs = pd.read_html(StringIO(html), flavor='html5lib')

            for table_idx, df in enumerate(dfs):
                df = flatten_dataframe(df)
                if len(df) < 3: continue

                if self._is_beneficial_ownership_table(df):
                    continue

                if self._is_any_compensation_table(df):
                    continue

                cols = ColumnIdentifier.identify(df)
                if cols['name'] is None: continue

                # CRITICAL: Hard gate - real director table must have at least one attribute column
                # (age OR since OR independent). Otherwise it's likely a layout/committee matrix.
                if cols['age'] is None and cols['since'] is None and cols['independent'] is None:
                    continue

                # IMPROVED: Check headers AND preview rows for board indicators
                header_text = ' '.join(str(c).lower() for c in df.columns)
                preview_text = ' '.join(
                    ' '.join(str(v).lower() for v in df.iloc[r].values)
                    for r in range(min(2, len(df)))
                )

                has_board_indicator = any(term in (header_text + " " + preview_text) for term in [
                    'director', 'nominee', 'age', 'independent', 'since',
                    'board', 'term expires'
                ])

                if not has_board_indicator:
                    continue

                # NEW: Extract independence from HTML icons if column exists
                ind_html_map = {}
                if cols.get('independent') is not None:
                    if debug:
                        print(f"\n[DEBUG] HTML icon extraction for table {table_idx}:")
                    # Pass pandas column indices to BeautifulSoup
                    ind_html_map = self._html_board_independence_map(
                        html,
                        table_idx,
                        name_col_idx=cols['name'],
                        indep_col_idx=cols['independent'],
                        debug=debug
                    )
                    if debug:
                        print(f"  HTML ind_html_map size: {len(ind_html_map)}")
                    if not ind_html_map and BeautifulSoup is None:
                        if debug:
                            print(f"  BeautifulSoup not available - HTML icons cannot be extracted")
                        # BeautifulSoup not installed - HTML icons can't be extracted
                        pass  # Silently fail, will use text enrichment fallback

                current_table: List[Dict[str, Any]] = []
                valid_name_count = 0

                for row_idx, row in df.iterrows():
                    try:
                        name_raw = str(row.iloc[cols['name']])

                        name_obj = self.name_extractor.extract_name_from_cell(name_raw)
                        if not name_obj:
                            continue

                        name = name_obj.full_name
                        valid_name_count += 1

                        age = name_obj.age
                        if not age and cols['age'] is not None:
                            try:
                                age_match = re.search(r'\d+', str(row.iloc[cols['age']]))
                                if age_match:
                                    age = int(age_match.group())
                            except: pass

                        tenure = None
                        if cols['since'] is not None:
                            m = _YEAR_RE.search(str(row.iloc[cols['since']]))
                            if m: tenure = f"Since {m.group()}"

                        independent = None
                        independent_header = None
                        if cols['independent'] is not None:
                            col_header = str(df.columns[cols['independent']])
                            if 'independent' in col_header.lower() or 'independence' in col_header.lower():
                                independent_header = col_header
                                # NEW: Use robust parser for independence values
                                independent = parse_independent_cell(row.iloc[cols['independent']])

                                # NEW: Backfill from HTML icon map if pandas value is missing
                                if independent is None and ind_html_map:
                                    independent = ind_html_map.get(norm_name_key(name))

                        row_snippet, row_kv = create_table_evidence(row, df)

                        # NEW: Inject independence into evidence if we have it
                        if independent is not None:
                            ind_str = "Yes" if independent else "No"
                            row_kv["Independent"] = ind_str
                            row_snippet = (row_snippet + f" | Independent: {ind_str}")[:500]

                        ev_row_idx = int(row_idx) if isinstance(row_idx, (int, float, np.integer)) else None

                        evidence = Evidence(
                            strategy="table_parse",
                            snippet=row_snippet[:500],
                            location=EvidenceLocation(
                                section_hint="Board of Directors Table",
                                table_index=table_idx,
                                row_index=ev_row_idx
                            ),
                            column_header=independent_header,
                            table_kv=row_kv
                        )

                        current_table.append({
                            'name': name,
                            'age': age,
                            'tenure': tenure,
                            'independent': independent,
                            'extraction_method': 'table',
                            'evidence': asdict(evidence)
                        })
                    except: pass

                if current_table:
                    # CRITICAL: Must yield at least 5 directors to be considered "the board table"
                    if len(current_table) < 5:
                        continue

                    score = score_board_table(df, valid_name_count)

                    if score > best_score:
                        best_score = score
                        best_members = current_table
                        best_df = df  # NEW: Save the DataFrame too

                    # NEW: Track as fallback if has director indicators and decent size
                    if score > -50 and valid_name_count >= 5 and len(current_table) >= 5:
                        fallback_candidates.append((score, current_table))

            # NEW: If no table passed hard gates, use best fallback
            if not best_members and fallback_candidates:
                fallback_candidates.sort(key=lambda x: x[0], reverse=True)
                best_members = fallback_candidates[0][1]

        except: pass
        return best_members, best_df

    def _extract_board_from_text(self, text: str) -> List[Dict[str, Any]]:
        members = []

        section_patterns = [
            r'(?i)(Election of Directors.*?)(?=\n\s*\n[A-Z][A-Z\s]{20,}|\Z)',
            r'(?i)(Director Nominees.*?)(?=\n\s*\n[A-Z][A-Z\s]{20,}|\Z)',
            r'(?i)(Board of Directors.*?)(?=\n\s*\n[A-Z][A-Z\s]{20,}|\Z)',
            r'(?i)(Proposal\s+\d+.*?Director.*?)(?=\n\s*\n[A-Z][A-Z\s]{20,}|\Z)'
        ]

        section_text = None
        for pattern in section_patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                section_text = match.group(1)
                break

        if not section_text:
            section_text = text

        names = self.name_extractor.extract_names(section_text)

        for n in names:
            if self._is_likely_director_in_section(n.full_name, section_text):
                ind = self._analyze_independence(section_text, n.full_name)
                members.append({
                    'name': n.full_name,
                    'age': n.age,
                    'independent': ind['is_independent'],
                    'extraction_method': 'text',
                    'evidence': asdict(n.evidence) if n.evidence else None
                })
        return members

    def _is_likely_director_in_section(self, name: str, section_text: str) -> bool:
        try:
            name_idx = section_text.lower().find(name.lower())
            if name_idx == -1:
                return False

            start = max(0, name_idx - 500)
            end = min(len(section_text), name_idx + len(name) + 500)
            context = section_text[start:end].lower()

            return 'director' in context or 'nominee' in context or 'board' in context
        except:
            return False

    def _analyze_independence(self, text: str, name: str) -> Dict[str, Any]:
        try:
            ctx = re.search(rf'.{{0,300}}{re.escape(name)}.{{0,300}}', text, re.I | re.S)
            context = ctx.group(0).lower() if ctx else ""
            score = 0.0
            if 'independent' in context: score += 0.5
            if any(t in context for t in ['ceo', 'cfo', 'chief financial', 'chief executive', 'president']): score -= 0.8
            if 'executive chairman' in context: score -= 0.8
            return {'is_independent': True if score >= 0.5 else False if score <= -0.5 else None}
        except: return {'is_independent': None}

    def _calc_gov_metrics(self, members: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not members: return {}

        total = len(members)
        known = [m for m in members if m.get("independent") in (True, False)]
        known_total = len(known)
        ind_true = sum(1 for m in known if m.get("independent") is True)
        unknown = total - known_total

        return {
            "total": total,
            "independent_true": ind_true,
            "independent_known_total": known_total,
            "independent_unknown": unknown,
            "independence_pct_known": round((ind_true / known_total) * 100, 1) if known_total > 0 else None
        }

def safe_get_8k_items(filing: Any) -> Tuple[List[str], str]:
    items_list: List[str] = []
    source = "regex"
    if filing is None: return items_list, source
    try:
        if hasattr(filing, 'obj') and callable(filing.obj):
            eightk_obj = filing.obj()
            obj_items = getattr(eightk_obj, 'items', None)
            if obj_items:
                norm = normalize_item_numbers(list(obj_items))
                if norm: return norm, "eightk_obj"
    except: pass
    return items_list, source

# Wrapper functions
def extract_8k_events(ticker: str, limit: int = 10): return SECFilingParser().extract_8k_events(ticker, limit)
def extract_proxy_compensation(ticker: str): return SECFilingParser().extract_proxy_compensation(ticker)
def extract_governance_data(ticker: str, debug: bool = False):
    return SECFilingParser().extract_governance_data(ticker, debug=debug)

if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "GOOGL"
    parser = SECFilingParser()
    print("="*70)
    print(f"SEC Filing Parser v7.2 - Production-Grade Truth-Grounding Test for {ticker}")
    print("="*70)
    print("\n[1/3] 8-K Events")
    res8k = parser.extract_8k_events(ticker, 3)
    print(res8k)
    print("\n[2/3] Compensation")
    rescomp = parser.extract_proxy_compensation(ticker)
    print(rescomp)
    print("\n[3/3] Governance")
    resgov = parser.extract_governance_data(ticker, debug=True)
    print(resgov)
