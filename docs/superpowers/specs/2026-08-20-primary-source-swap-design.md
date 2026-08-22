# Primary-Source Swap — Design

**Date:** 2026-08-20
**Status:** scoped, not started
**Depends on:** `2026-08-20-truth-source-refactor-design.md` landing first
**Scope:** financial-statement and estimate sourcing only. No transport change, no containerization.

## Problem

Two Finnhub endpoints return `HTTP 403 — "You don't have access to this resource"`
on the current tier, verified live 2026-08-20:

- `/stock/financials` — backs `get_financial_statements`
- `/stock/eps-estimate`, `/revenue-estimate`, `/ebitda-estimate` — back `get_forward_estimates`

Both silently degrade to a yfinance fallback tagged `_source: yfinance_fallback`. The
tag means the degradation is detectable, but nothing consumes the tag, so in practice
four skills each believe they are reading Finnhub data and are not.

The deeper problem is sourcing direction. Financial statements are **public SEC
filings**. The repo already extracts them from EDGAR XBRL — the primary source Finnhub
resells. Yet the extractors are unused while the reseller path is load-bearing:

| Tool | Backing source | Skills using it |
|---|---|---|
| `get_financial_statements` | Finnhub 403 → yfinance | 4 |
| `get_forward_estimates` | Finnhub 403 → yfinance | 4 |
| `get_segment_financials` | SEC XBRL | 4 |
| `get_historical_fcf` | SEC XBRL | 1 |
| `get_revenue_base` | SEC XBRL | **0** |
| `get_ebitda_margin` | SEC XBRL | **0** |
| `get_margin_breakdown` | SEC XBRL | **0** |

This is the same defect the truth-source refactor addressed for options: two paths to
one fact, with consumers pointed at the weaker one.

## Guiding principle

Carried forward from the truth-source refactor:

> Go to the filing, not to a reseller of the filing.

Split the two endpoints by whether their data is public:

- **Financial statements are public.** SEC EDGAR is authoritative and free. Finnhub
  adds nothing but normalization. **Swap to EDGAR.**
- **Consensus estimates are not public.** They are the licensed aggregation of private
  sell-side research (LSEG I/B/E/S, FactSet, Visible Alpha, Zacks). No free
  authoritative source exists because the underlying data was never public. **Cannot be
  built; must be bought or approximated.**

## Deliverables

### D1 — Route `get_financial_statements` to EDGAR

Re-back the tool with the repo's existing SEC XBRL extractors instead of yfinance.
Verified accurate: `get_revenue_base('MSFT')` returns `331,839,000,000` raw dollars,
matching Microsoft's reported FY2026 revenue exactly.

The tool's `statement` parameter (`ic` / `bs` / `cf`) maps onto extractor families in
`tools/web_search_server/sec_utils.py`. Preserve the existing response envelope so the
four consuming skills need no edit — this is a sourcing change, not an interface change.

Keep yfinance as an explicit **third** fallback for non-SEC filers (ADRs, foreign
issuers filing 20-F/6-K rather than 10-K/10-Q), tagged as such. Do not silently return
yfinance data under an EDGAR label.

**Ordering constraint:** the truth-source refactor's D4 splits `Session_Cache` off the
state database. EDGAR extraction is slow and cache-heavy, so this deliverable must land
after that split or it will write cache rows into the state file.

### D2 — Make the estimate degradation loud

`get_forward_estimates` advertises "next 4-6 quarters" from three endpoints that all
403. Consensus cannot be self-built, so this deliverable does not replace the data — it
stops the tool from misrepresenting what it returned.

- Surface `_source` in the tool's top-level response, not buried in the payload.
- When the Finnhub path 403s, populate a `data_gap` field naming exactly what is
  missing: multi-quarter forward curve, EBITDA estimates, and estimate dispersion.
- Skills that gate on consensus must treat a `yfinance_fallback` source as reduced
  confidence rather than equivalent data.

### D3 — Source next-quarter consensus from the free calendar path

Verified 2026-08-20: `/calendar/earnings` returns `epsEstimate` and `revenueEstimate`
on 66% of rows and is **not** paywalled.

```
ADSK  2026-08-27       epsEst=3.1811  revEst=2,050,834,684
AFRM  2026-08-27 amc   epsEst=0.3504  revEst=1,128,021,784
```

`docs/known_issues.md` already states the calendar `eps_estimate` is THE scoring bar for
`/earnings-eval`, so the pre-earnings pipeline's most important number already comes
through this path. Make that explicit rather than incidental: next-quarter consensus
reads from the calendar, and `get_forward_estimates` is consulted only for the
multi-quarter curve it cannot currently supply.

### D4 — Wire up or delete the orphaned extractors

`get_revenue_base`, `get_ebitda_margin`, and `get_margin_breakdown` are referenced by
zero skills. After D1 they may be reached indirectly through
`get_financial_statements`. Decide per tool: reachable via D1, referenced directly by a
skill, or deleted. An extractor that is neither is dead code.

### D5 — Fix the 7 unit-scale tests

`test_get_revenue_base_diverse_sectors` and two mock-based siblings assert
`1 <= revenue_base <= 1000000`, a millions-scale range. Live output is raw dollars, so
the assertion is wrong by ~332,000x. **The production code is correct; the tests encode
a bad expectation.** Fix the tests, not the extractor. Confirmed against both
`sec_utils.py`'s own "keep in raw dollars" comments and Microsoft's reported revenue.

## Explicitly out of scope

Owned by the truth-source refactor, do not touch here: the options merge, the openbb
deletion, the FinBERT removal, the `Session_Cache` split, and the `/preearnings-research`
sentiment edit.

Not in either spec: buying an estimates vendor. If dispersion (analyst count, high/low
spread) proves necessary for `/expectations-hurdle-check`, price Financial Modeling
Prep, EODHD, and Tiingo against Finnhub's paid tier before defaulting to a Finnhub
upgrade. That is a purchasing decision, not an engineering one.

## Testing strategy

1. **Parity harness.** For a basket spanning sectors and filer sizes, assert EDGAR-sourced
   statements agree with the current yfinance fallback within tolerance on revenue,
   operating income, and total assets. Disagreements beyond tolerance are findings to
   adjudicate against the actual 10-K, not tolerances to widen.
2. **Non-SEC filer coverage.** Include at least one ADR / 20-F filer to prove the
   third-tier fallback engages and is labelled honestly rather than returning empty.
3. **Envelope stability.** Assert the response shape is unchanged so the four consuming
   skills keep working without edits.
4. **Fix-the-test gate (D5).** The 7 unit-scale tests must assert raw dollars against a
   known reported figure, not a magnitude range.
5. **Degradation visibility (D2).** Assert that a 403 produces a populated `data_gap`
   and a top-level `_source`, and that neither is silently absent.

## Risks

| Risk | Mitigation |
|---|---|
| EDGAR is slower than the Finnhub/yfinance path | Land after the cache split; measure before and after and record the delta |
| XBRL tags vary across filers, so extraction is less uniform than a vendor's normalized feed | Parity harness spans sectors and filer sizes; `sec_utils.diff_10k` already has known megacap issues (`known_issues.md`) |
| A skill depends on a yfinance-only field EDGAR does not carry | Envelope-stability test catches it before skills break |
| SEC fair-access rate limiting under a now-real `SEC_EMAIL` | ~10 req/s ceiling; batch and cache. `test_all_500_revenue` already demonstrates this hazard |
