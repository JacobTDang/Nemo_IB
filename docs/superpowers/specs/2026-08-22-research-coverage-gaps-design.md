# Research Coverage Gaps — Design

**Date:** 2026-08-22
**Status:** approved, in progress
**Depends on:** `2026-08-20-truth-source-refactor-design.md` (PR #11)
**Scope:** new primary-source extraction tools. No transport change, no containerization.

## Problem

The SEC layer answers "what did this company earn?" well and "how many shares are there?"
not at all. Seven gaps, each of which makes an analyst confidently wrong rather than
merely uninformed:

| Gap | Current coverage | Consequence |
|---|---|---|
| Share count over time | point-in-time snapshot from yfinance only | dilution is invisible; improving per-share metrics can be pure denominator growth |
| Shelf registrations | **zero** references to `S-3` or `424B5` in the codebase | the mechanism by which dilution actually happens is unobserved |
| Stock-based compensation | **zero** | the largest GAAP-to-adjusted bridge item, and the other dilution engine |
| Debt maturity schedule | `calculate_credit_profile` exists but has no maturity data | "leverage is 3x" means something different when the wall is next year |
| Litigation (Item 3) | **zero** | material legal exposure unreported |
| Customer concentration | **zero** | "one customer was 34% of revenue" is often the entire thesis |
| Dividends and splits | **zero** | un-split-adjusted per-share history is silently wrong |
| Peer set | `comparable_company_analysis` requires hand-fed peers | comps depend on the analyst already knowing the answer |

SEC form coverage today, counted across `tools/`: 59 references to `10-K`, 6 to
`DEF 14A`, 5 to `8-K`, 3 to `10-Q`, 3 to `SC 13D`/`13G`, 1 each to `S-1` and `13F-HR`.
Zero to `424B5`, `S-3`, `20-F`, or `6-K`.

## The architectural gap

`get_segment_financials` produces five years of history from a **single** filing,
because one 10-K's XBRL carries comparative periods. That works for income-statement
items and hides the real limitation: **nothing in the SEC layer ever iterates multiple
filings.** `get_historical_fcf`, despite its name, reads only the latest filing and
returns one period.

Share count lives on the cover page as one instant per filing, so a dilution series is
impossible without walking filings over time. That capability does not exist.

The enabler is installed and unused: edgartools exposes `XBRLS`, `StatementStitcher`,
`StitchedFactQuery`, and `StitchedFactsView`.

## Proven mechanism

Validated live against EDGAR on 2026-08-22 before writing this spec.

**Multi-filing extraction works.** MSFT `dei:EntityCommonStockSharesOutstanding` across
three 10-Qs: 7,432,377,655 (2025-10-29) → 7,425,629,076 (2026-01-28) → 7,428,434,704
(2026-04-29).

**Multi-class is the trap.** GOOGL returns **three** facts per filing with identical
labels, distinguishable only by `context_ref`:

| context | dimension member | shares |
|---|---|---|
| `c-28` | `us-gaap:CommonClassAMember` | 5,868,000,000 |
| `c-29` | `us-gaap:CommonClassBMember` | 835,000,000 |
| `c-30` | `goog:CapitalClassCMember` | 5,527,000,000 |

Total 12,230,000,000. **An extractor taking the first row reports 5.868bn — a 52%
understatement that looks entirely plausible.**

Two facts constrain the implementation:

1. `to_dataframe()` carries **no dimension column**. Class identity must be resolved via
   `xbrl.contexts[context_ref].dimensions`, which is a plain dict.
2. `facts.query().by_dimension("us-gaap:StatementClassOfStockAxis", member)` returns
   **empty** for every member, despite the facts existing. The dimension-query path used
   by `get_segment_financials` does not work here. Do not reach for it.
3. Class C is tagged `goog:CapitalClassCMember` — a **company-specific** member, not
   `us-gaap:CommonClassCMember`. Whitelisting standard members silently drops classes.
   Resolve whatever the filer used.

Single-class companies return one fact with **no dimensions at all**. Both paths must work.

## Deliverables

### D0 — `sec_series.py`, the multi-filing accessor

A new module. `tools/web_search_server/sec_utils.py` is already 2,912 lines and adding a
second access pattern to it would make the file harder to hold in context, not easier.

Responsibilities, and nothing else:

- Walk the last N filings of a form type for a ticker.
- Extract one concept per filing.
- Resolve each fact's `context_ref` to its dimension members.
- Return a dated series, with a per-dimension breakdown when dimensions are present and a
  single value when they are not.

Rate-limited to SEC's fair-access ceiling. D1, D2, and D3 consume it.

### D1 — Share count, dilution, and shelf activity

Series of `dei:EntityCommonStockSharesOutstanding` over the last N periods:
per-class breakdown, total, and period-over-period change.

Plus shelf detection from the filing index: `S-3` registrations and `424B5` takedowns in
the window. This is new form coverage, not a new concept.

The tool must state which share classes it found. Reporting a bare total hides the case
where a class was missed.

### D2 — Stock-based compensation

`us-gaap:ShareBasedCompensation` series, expressed as raw dollars, percent of revenue,
and percent of operating cash flow. Falls back through the concept chain the same way
`get_revenue_base` does.

### D3 — Debt maturity schedule

The `us-gaap:LongTermDebtMaturitiesRepaymentsOfPrincipalIn*` family — next twelve months
through year five, plus thereafter.

**This is the deliverable most likely to have coverage gaps.** Maturity tagging is
inconsistent across filers. When the concepts are absent, say so explicitly. Do not
synthesise a schedule from total debt, and do not return an empty schedule that reads
like "no maturities."

### D4 — Litigation, Item 3

Reuse the existing section extractor that serves MD&A and risk factors. Point it at
Item 3 of the 10-K.

### D5 — Customer concentration

Extract major-customer disclosure. Where a filer names a customer and a revenue
percentage, return both; where the disclosure is qualitative, return the text rather
than a null.

### D6 — Dividends and splits

From yfinance. Dividend history and split history with dates and ratios. Cheap, and its
absence makes historical per-share comparisons wrong without warning.

### D7 — Peer discovery

SEC submissions metadata already returns the SIC code. Expose peer discovery by SIC so
`comparable_company_analysis` stops requiring the analyst to supply the answer.

## Testing strategy

Two tiers. Neither alone is sufficient: golden values catch wrong-but-plausible numbers,
the sweep catches coverage holes.

### Tier 1 — hand-verified golden values

Exact assertions against figures read from the filings themselves, not ranges.

| Ticker | Why it is in the basket |
|---|---|
| GOOGL | multi-class, company-specific Class C tag, the 52% trap |
| MSFT | single class, no dimensions — the other code path |
| NVDA | 10:1 split in 2024; share counts must be split-consistent |
| a heavy-ATM biotech | dilution actually visible period over period |
| a REIT | different concept usage |
| a bank | different statement structure |
| a recent IPO | short history; the series must degrade rather than crash |

### Tier 2 — broad sweep, roughly 50 tickers

Shape, no-crash, and sanity bounds. Cross-checked against yfinance where an equivalent
exists. Rate-limited to SEC's ~10 requests per second and gated behind the network marker
so it never runs offline.

**The bound that catches the GOOGL class of bug:** total shares × price must fall within
tolerance of market capitalisation. A dropped share class fails this immediately, which
a shape assertion would not.

### Rules

- A tool that cannot find its concept returns an explicit "not covered" result. It never
  returns zero, an empty list, or a synthesised value.
- Coverage rate across the sweep is recorded per deliverable. A tool that works for 60%
  of filers is useful; a tool that silently works for 60% is not.
- No mock data outside tests.

## Risks

| Risk | Mitigation |
|---|---|
| A share class is missed and the total looks plausible | market-cap reconciliation in Tier 2; per-class breakdown always reported |
| Debt maturity concepts are absent for many filers | explicit not-covered result; coverage rate recorded rather than hidden |
| Live EDGAR tests are slow and rate-limited | network-gated, rate-limited, excluded from the offline suite |
| `sec_utils.py` grows further | new access pattern goes in `sec_series.py` |
| Golden values drift as companies file | assertions pin a stated filing date, not "latest" |
| Section extraction returns the wrong item | assert on known text from the named filing, not on length |

## Explicitly out of scope

**Foreign issuers.** `20-F` and `6-K` use different taxonomies and a different filing
regime. Every ADR is currently invisible to the SEC pipeline, which is a real gap, but
folding it in would roughly double this spec. It gets its own.

Owned by other specs, not touched here: the containerization and HTTP transport work, the
trustworthy-baseline test repair, and the primary-source swap's routing of
`get_financial_statements` to EDGAR.

**Note for the primary-source swap:** its D4 proposes deleting `get_revenue_base` and
`get_ebitda_margin` if nothing references them. D2 here uses the same concept-chain
pattern and D1's market-cap reconciliation may consume revenue. Check before deleting.
