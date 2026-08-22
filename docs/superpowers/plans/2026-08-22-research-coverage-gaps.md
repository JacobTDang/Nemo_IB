# Research Coverage Gaps Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close seven primary-source blind spots in the SEC layer, the largest being that
share count and therefore dilution are invisible.

**Architecture:** A new `sec_series.py` provides the one capability nothing in the codebase
has — walking multiple filings and resolving each XBRL fact's context to its dimension
members. Three deliverables sit on it. The rest reuse existing extractors.

**Tech Stack:** Python 3.12, edgartools, yfinance, pytest.

## Global Constraints

- Run everything with `.venv/bin/python`, never bare `python3`.
- Live EDGAR tests are gated behind the network marker and rate-limited to ~10 req/s.
- A tool that cannot find its concept returns an explicit not-covered result. Never zero,
  never an empty list, never a synthesised value.
- Mock data only in tests.
- No new dependencies. edgartools and yfinance are already present.
- Commit messages must not mention Claude or any AI tool.
- `dei:EntityCommonStockSharesOutstanding` facts carry no dimension column. Resolve class
  identity through `xbrl.contexts[context_ref].dimensions`. `by_dimension()` returns empty
  for this concept — do not use it.

---

### Task 1: `sec_series.py` — the multi-filing accessor

**Files:**
- Create: `tools/web_search_server/sec_series.py`
- Test: `testing/test_sec_series.py`

**Interfaces:**
- Produces:
  - `ConceptFact` — dataclass with `value: float`, `period: str`, `dimensions: dict[str, str]`, `context_ref: str`
  - `FilingPoint` — dataclass with `filing_date: str`, `form: str`, `accession: str`, `facts: list[ConceptFact]`
  - `fetch_concept_series(ticker, concept, form="10-Q", limit=8) -> list[FilingPoint]`
  - `resolve_dimensions(xbrl, context_ref) -> dict[str, str]`
  - `NotCovered` — exception raised when a concept appears in no filing

- [ ] **Step 1: Write the failing test for dimension resolution**

```python
def test_resolve_dimensions_returns_empty_for_undimensioned_context():
    class FakeXBRL:
        contexts = {"c-1": type("C", (), {"dimensions": {}})()}
    assert resolve_dimensions(FakeXBRL(), "c-1") == {}


def test_resolve_dimensions_returns_members():
    class FakeXBRL:
        contexts = {"c-28": type("C", (), {
            "dimensions": {"us-gaap:StatementClassOfStockAxis": "us-gaap:CommonClassAMember"}})()}
    assert resolve_dimensions(FakeXBRL(), "c-28") == {
        "us-gaap:StatementClassOfStockAxis": "us-gaap:CommonClassAMember"}


def test_resolve_dimensions_missing_context_is_empty_not_crash():
    class FakeXBRL:
        contexts = {}
    assert resolve_dimensions(FakeXBRL(), "c-99") == {}
```

- [ ] **Step 2: Run and verify it fails**

Run: `SKIP_NETWORK_TESTS=1 .venv/bin/python -m pytest testing/test_sec_series.py -v`
Expected: `ModuleNotFoundError: tools.web_search_server.sec_series`

- [ ] **Step 3: Implement the module**

Implement `ConceptFact`, `FilingPoint`, `NotCovered`, `resolve_dimensions`, and
`fetch_concept_series`. `fetch_concept_series` walks `Company(ticker).get_filings(form=...)`,
calls `.xbrl()` per filing, queries the concept, and builds a `ConceptFact` per row with
dimensions resolved from `xbrl.contexts`. Rate-limit between filings.

- [ ] **Step 4: Run and verify it passes**

Run: `SKIP_NETWORK_TESTS=1 .venv/bin/python -m pytest testing/test_sec_series.py -v`
Expected: PASS

- [ ] **Step 5: Add the live golden test**

Assert MSFT returns one undimensioned fact per filing and GOOGL returns three
dimensioned facts, with `goog:CapitalClassCMember` among the members. Network-gated.

- [ ] **Step 6: Commit**

---

### Task 2: D1 — share count, dilution, shelf activity

**Files:**
- Create: `tools/web_search_server/dilution.py`
- Modify: `tools/web_search_server/web_search.py` (register tools)
- Test: `testing/test_dilution.py`

**Interfaces:**
- Consumes: `fetch_concept_series`, `resolve_dimensions`, `NotCovered` from Task 1.
- Produces:
  - `get_share_count_series(ticker, limit=8) -> dict` with `by_class`, `total`, `periods`, `change_pct`
  - `get_shelf_activity(ticker, lookback_days=730) -> dict` with `s3_registrations`, `b5_takedowns`
  - `_class_label(member: str) -> str` mapping a member tag to a human label

- [ ] **Step 1: Write the failing tests**

Cover: single-class totals; multi-class sums all classes; a company-specific member is
counted, not dropped; a missing concept raises `NotCovered` rather than returning 0.

- [ ] **Step 2: Run and verify they fail**
- [ ] **Step 3: Implement**
- [ ] **Step 4: Run and verify they pass**
- [ ] **Step 5: Golden test — GOOGL sums to 12,230,000,000 across A/B/C; MSFT single class**
- [ ] **Step 6: Shelf detection against a known S-3 filer**
- [ ] **Step 7: Register both as MCP tools**
- [ ] **Step 8: Commit**

---

### Task 3: D2 — stock-based compensation

**Files:** Create `tools/web_search_server/sbc.py`; test `testing/test_sbc.py`

**Interfaces:**
- Consumes: `fetch_concept_series` from Task 1.
- Produces: `get_sbc_series(ticker, limit=8) -> dict` with `series`, `pct_of_revenue`, `pct_of_ocf`.

Concept chain: `us-gaap:ShareBasedCompensation`,
`us-gaap:AllocatedShareBasedCompensationExpense`,
`us-gaap:ShareBasedCompensationArrangementByShareBasedPaymentAwardCompensationCost`.

- [ ] Steps 1-6: failing test, verify fail, implement, verify pass, golden test, commit

---

### Task 4: D3 — debt maturity schedule

**Files:** Create `tools/web_search_server/debt_maturity.py`; test `testing/test_debt_maturity.py`

**Interfaces:**
- Produces: `get_debt_maturity_schedule(ticker) -> dict` with `by_year`, `thereafter`,
  `total`, `coverage: "full" | "partial" | "not_covered"`.

Concepts: `LongTermDebtMaturitiesRepaymentsOfPrincipalInNextTwelveMonths` through
`InYearFive`, plus `Thereafter`.

**This deliverable must report coverage honestly.** When concepts are absent, return
`coverage: "not_covered"` with the concepts tried. Never synthesise from total debt.

- [ ] Steps 1-6: failing test, verify fail, implement, verify pass, golden test, commit

---

### Task 5: D4 and D5 — litigation and customer concentration

**Files:** Modify `tools/web_search_server/sec_utils.py`; test `testing/test_sec_sections.py`

**Interfaces:**
- Produces: `extract_litigation(ticker, form_type="10-K") -> dict`,
  `extract_customer_concentration(ticker, form_type="10-K") -> dict` with
  `named_customers: list[{name, pct_of_revenue}]` and `disclosure_text`.

Reuse the existing section extractor that serves MD&A and risk factors. Assert on known
text from a named filing, never on length.

- [ ] Steps 1-6: failing test, verify fail, implement, verify pass, golden test, commit

---

### Task 6: D6 — dividends and splits

**Files:** Modify `tools/financial_modeling_engine/analysis_tools.py`; test `testing/test_corporate_actions.py`

**Interfaces:**
- Produces: `get_corporate_actions(ticker, years=10) -> dict` with `dividends`, `splits`,
  `latest_split_ratio`, `latest_split_date`.

Golden: NVDA has a 10:1 split dated 2024-06-10.

- [ ] Steps 1-6: failing test, verify fail, implement, verify pass, golden test, commit

---

### Task 7: D7 — peer discovery by SIC

**Files:** Modify `tools/web_search_server/sec_utils.py`; test `testing/test_peer_discovery.py`

**Interfaces:**
- Produces: `get_sic_code(ticker) -> dict`, `find_peers_by_sic(ticker, limit=20) -> dict`.

- [ ] Steps 1-6: failing test, verify fail, implement, verify pass, golden test, commit

---

### Task 8: Tier 2 — the broad sweep

**Files:** Create `testing/test_research_coverage_sweep.py`

Roughly 50 tickers spanning megacap, mid, small, REIT, bank, biotech, recent IPO.
Network-gated, rate-limited.

Assertions per ticker:
- No crash. A failure returns a not-covered result, never an exception.
- Shape matches the documented contract.
- **Market-cap reconciliation:** `total_shares × price` within tolerance of yfinance
  market cap. This is the assertion that catches a dropped share class.
- Coverage rate recorded per deliverable and printed at the end.

- [ ] Steps 1-5: write sweep, run it, record coverage rates, document them, commit

---

## Verification

```bash
SKIP_NETWORK_TESTS=1 .venv/bin/python -m pytest testing/ -q   # offline: no new failures
.venv/bin/python -m pytest testing/test_research_coverage_sweep.py -v   # live, rate-limited
```

Coverage rates per deliverable go in `docs/known_issues.md`. A tool that works for 60% of
filers is useful; a tool that silently works for 60% is not.
