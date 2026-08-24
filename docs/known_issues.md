# Known Issues

Findings worth fixing but not blocking the current build. Each entry
includes what surfaced it, what the fix looks like, and rough priority.

## Pre-Earnings Pipeline — bugs, limitations, backlog (2026-06-05)

Branch `preearnings-pipeline`, written at merge time. 314 tests passing.

### Known bugs / limitations

1. **yfinance IV sentinels** — yfinance sometimes reports implied vol as
   a near-zero sentinel (~1e-5) on illiquid strikes. Mitigation shipped:
   skew is set to `None` instead of a garbage number when either leg IV
   is a sentinel; the skill treats `skew=None` as a data gap. Residual:
   no alternative IV source, so skew coverage is thinner on small caps.

2. **Short interest is 2-3 weeks stale** — FINRA publishes SI
   bi-monthly with a lag. `classify_positioning` can call crowded_short
   on positioning that has already partially unwound. No fix available
   without a paid feed (Ortex — see backlog).

3. **Surprise/reaction pairing residuals** — `pair_surprises_with_reactions`
   now uses optimal max-cardinality / min-distance assignment plus the
   `filter_earnings_cadence` pre-filter, but non-earnings 8-Ks inside the
   cadence window can still slip through. The accurate fallback (parsing
   each 8-K for Item 2.02) is too slow for the live pipeline and is
   documented as a SOLO-only diagnostic.

4. **`select_scoring_bar` sign-flip limits** — basis validation uses a
   growth-bounds check (0.3–2.5x year-ago actual). A company swinging
   from loss to profit (or vice versa) breaks the ratio logic; the
   function flags `divergent_unverifiable` rather than guessing, which
   is correct but means no bar validation for turnaround quarters.

5. **`quotes_stale` after hours** — options chain quotes go to 0-bid /
   0-ask after the close. Mitigation shipped: legs rebuild from
   `last_price` and the result carries `quotes_stale=true`. Residual:
   last_price can be hours old; event-move math inherits that staleness.

6. **RESOLVED: Put-call parity guard not live until next MCP restart** —
   the guard (commit 22e102b) rebuilds both legs from last_price when
   |C−P−(S−K)|/S > 5%. Originally shipped as a separate MCP tool, so the
   already-running altdata server process predated the commit and needed
   a restart to pick it up. As of the `truth-source-refactor` merge the
   guard ships inline inside `_straddle_legs` in
   `tools/financial_modeling_engine/utils.py`, which is imported fresh on
   every call rather than living in a long-running server process, so
   there is no restart-lag window anymore.

7. **Eval price-move attribution gap** — `/earnings-eval` grades the
   1-day move against the EPS-surprise outcome, but the move is often
   driven by guidance, not the printed quarter. Improvement idea: add an
   `outcome_driver` tag (eps / guidance / multiple) to `record_eval`
   notes so the track record can separate "right on EPS, wrong on
   guide" cases.

8. **Google Trends 429s** — pytrends gets rate-limited intermittently.
   The skill records `data_gap` and the `na` weight redistributes, so
   the pipeline degrades gracefully, but trends coverage is unreliable.

9. **`get_earnings_transcripts` returns 8-K releases, not call
   transcripts** — for CHWY it produced pairs_found=0 because the
   "transcripts" were press releases. Call-sentiment signals are
   effectively press-release-sentiment for companies without free
   transcript sources.

10. **Stale vendor calendar dates** — Finnhub's earnings calendar showed
    GME reporting 6-08 when news proved it reported 6-02 (dropped from
    the betting slate). Mitigation shipped: news-digest sub-agents carry
    a `calendar_conflicts` field so digests can contradict the calendar;
    the skill must treat a confirmed conflict as disqualifying.

### Improvement backlog (ordered)

1. **TIME-GATED: weight refit + /track-record skill** — after ~10 evals
   exist, refit `DEFAULT_DIRECTION_WEIGHTS` against realized accuracy
   and build an aggregate track-record view. First eval points land
   6-11/6-12 (ORCL, CHWY, ADBE, RH).
2. **BUDGET-GATED: Tier 2 data** — SimilarWeb scaffold was shipped and
   key-gated as `get_web_traffic_signal` (`tools/preearnings/web_traffic.py`);
   removed in `truth-source-refactor` alongside the broken FinBERT sentiment
   tool. Re-scaffolding this is still backlog; next are Panjiva (shipments),
   AlphaSense (transcripts), Ortex (live SI), card panels.
3. **Predict-the-guide** — model the guidance number separately from
   the quarter print; most post-print moves are guide-driven (see bug 7).
4. **KPI-level consensus scoring** — score KPI beats/misses against
   KPI-level consensus where available, not just EPS.
5. **Sentry live tick** — wire the pre-earnings escalation branch in
   `/sentry-tick` to a scheduled daemon so research fires 5-7 days out
   without manual invocation.
6. **Fresh sessions for eval week** — use `run_manifest` resumability to
   run each eval in a clean session instead of one long context.

### Live predictions awaiting eval

| Ticker | Earnings | Hour | Prediction | Conf | Action |
|---|---|---|---|---|---|
| ORCL | 2026-06-10 | amc | likely_beat | 0.52 | no_position (crowded long) |
| CHWY | 2026-06-10 | bmo | in_line | 0.53 | no_position (in_line rule) |
| ADBE | 2026-06-11 | amc | in_line | 0.44 | no_position (in_line rule) |
| RH | 2026-06-11 | amc | likely_miss | 0.39 | no_position (57% SI squeeze risk; Dec −23% miss closed UP +5.7%) |

Eval reminders: quarter matching window is [−75d, +45d] around the
earnings date; use bmo/amc reaction conventions; the Finnhub calendar
eps_estimate is THE scoring bar; the stored prediction is frozen — the
scoring call structurally cannot rewrite it.

---

## Discipline / process gaps surfaced by real runs

### `/portfolio-fit` silently skipped during /equity-deep-research

**Surfaced by:** AAPL deep-research run on 2026-05-21. Synthesis listed
"`/portfolio-fit` skipped (paper account read declined this turn)" in
data_gaps — but the skill still produced a sizing recommendation.

**Why it's a problem:** the /equity-deep-research Step 18 hard rules
require portfolio-fit to run before any sizing call. If portfolio-fit
fails, the synthesis should either:
1. Treat it as empty book (then sizing is informational only)
2. Refuse to produce a sizing recommendation entirely

It did neither. The verdict bypassed a required gate.

**Likely cause:** alpaca `get_paper_positions` was the slow MCP call we
just fixed in `tools/alpaca/server.py` (added 10s timeout in commit
e9ea9ff). Pre-fix, when get_paper_positions hung, the skill silently
fell through to "skip portfolio-fit". Post-fix, the alpaca call will
fail fast in 10s — but the skill should STILL handle the failure
explicitly, not silently degrade.

**Fix:** update `.claude/skills/equity-deep-research/SKILL.md` Step 18
to add a hard rule: if portfolio-fit fails to read the book (timeout,
crash, missing creds), the synthesis must NOT produce a sizing
recommendation — only an informational verdict.

**Priority:** medium. Process-level discipline fix, not a code bug.

---

### `diff_10k` Item 1A extractor failed on AAPL

**Surfaced by:** same AAPL run. data_gap reported "diff_10k Item 1A
failed (extractor couldn't isolate section)."

**Why it's a problem:** Item 1A (risk factors) diff is one of the
highest-signal extractors for catching what management is newly worried
about. If it's broken for AAPL, it's probably broken for any megacap
with non-standard 10-K formatting.

**Fix:** investigate `tools/web_search_server/sec_utils.py:diff_10k`
section-isolation logic. AAPL's 10-K likely has Item 1A nested under a
different heading hierarchy than smaller-cap filings the extractor was
tuned for.

**Priority:** medium-high. Affects research quality for the megacaps
that get researched most often.

---

### `extract_forward_signals` returns boilerplate for low-signal companies

**Surfaced by:** same AAPL run. Returns generic "we expect / anticipate
/ plan" boilerplate that adds no signal.

**Why it's a problem:** Apple specifically doesn't pre-announce
material guidance. The extractor pattern-matches on language without
considering whether the language is informative. False positives.

**Fix:** add a heuristic to detect "this company doesn't pre-announce
meaningfully" — e.g., if all matches are from generic risk-factor or
safe-harbor sections, return an empty result with note=`low_signal_
company` instead of the boilerplate. Could check company-specific
filing style (Apple, Berkshire, etc. are known low-signal).

**Priority:** low. Cosmetic noise rather than wrong signal.

---

### Scenario DCF lacks built-in terminal-multiple sensitivity

**Surfaced by:** AAPL synthesis used 22x terminal multiple in base
case. Bull case DCF printed $195 — 36% below spot of $305. The verdict
("watchlist until ≤$240") implicitly assumed 27x terminal, which
contradicts the 22x used in the DCF. Internal numbers don't agree with
each other.

**Why it's a problem:** any DCF where bull case is dramatically below
spot has one of three explanations: market is wrong, growth/margin
assumptions are too conservative, OR terminal multiple is wrong. The
skill should force the analyst to check which by varying terminal
multiple before publishing the verdict.

**Fix:** update `.claude/skills/scenario-builder/SKILL.md` to require
terminal-multiple sensitivity: for each case, show implied price at
terminal multiples of {20x, 25x, 30x, 35x} not just one. The analyst
must explicitly justify which multiple they're committing to.

Alternatively, `/equity-deep-research` Step 16 hard rule: if scenario
bull < 0.85 × spot, run `/valuation-check` with reverse-DCF mode to
back-solve what terminal multiple the market is pricing.

**Priority:** medium. Improves the most important quantitative input
to the verdict.

---

## Infrastructure / known fragility

### MCP stdio path can hang on alpaca calls (mostly fixed)

**Surfaced by:** Phase D smoke tests + user observation 2026-05-21.
`get_paper_positions` and `get_paper_account` taking 30s+ via MCP
while direct Python returns in < 1s.

**Status:** fixed in commit e9ea9ff — all 4 alpaca MCP tools now
bounded by `asyncio.wait_for` (10s reads, 20s writes). Hangs surface
as `broker_timeout` errors instead of blocking indefinitely.

**Remaining concern:** root cause of the stdio buffering quirk is not
understood. Symptom is suppressed but the underlying buffering issue
could affect other MCP servers similarly. Worth investigation if it
recurs on a non-alpaca MCP server.

**Priority:** low (symptom mitigated, root cause unknown).

---

### bun not always on PATH for slack_channel plugin

**Surfaced by:** new clone / fresh launcher run. Claude Code silently
fails to load `--dangerously-load-development-channels server:slack`
when bun isn't on PATH.

**Status:** mitigated in commit f7c4998 — launcher now checks bun on
PATH, falls back to `$HOME/.bun/bin/bun.exe`, and fails loudly with an
install URL if neither is found.

**Priority:** low (handled).

## Research coverage tools: measured coverage rates (2026-08-22)

Seven new primary-source tools closed the blind spots listed in
`docs/superpowers/specs/2026-08-22-research-coverage-gaps-design.md`. Coverage was
measured across a 35-ticker basket spanning megacaps, multi-class filers, banks,
REITs, biotech, serial issuers, industrials, energy, consumer staples, and recent
IPOs (`testing/test_research_coverage_sweep.py`).

| Tool | Coverage | Notes |
|---|---|---|
| `get_corporate_actions` | 35/35 (100%) | yfinance, not filing-dependent |
| `get_share_count_series` | 32/35 (91.4%) | cover-page tag; every filer should report it |
| `extract_customer_concentration` | 32/35 (91.4%) | text scan, not a numbered item |
| `get_sbc_series` | 31/35 (88.6%) | concept chain across three tags |
| `get_debt_maturity_schedule` | 20/35 (57.1%) | **full 14, partial 6, not covered 15** |

**`get_debt_maturity_schedule` is the one to treat carefully.** Maturity tagging is
inconsistent: MSFT, AAPL and T tag all six buckets while Ford and PLUG tag none,
and Ford is among the largest debt issuers in the market. The tool returns
`coverage: "not_covered"` in that case. **Never read that as "no debt matures."**

### Two bugs the sweep caught that unit tests could not

1. **Biogen double-counted.** BIIB emits its share-count fact twice with identical
   value, period and `context_ref`. Summing reported 295.5M shares against a real
   147.75M. Shape assertions passed; the market-cap reconciliation
   (`shares x price` against reported market cap) caught it at 100% off. Facts are
   now de-duplicated on the value/period/context triple.

2. **Failure changed the response shape.** `extract_customer_concentration` omitted
   `has_concentration` on its error path, so a caller reading the documented field
   got a `KeyError` rather than `False`.

Both are why the sweep asserts reconciliation rather than structure alone.

### Known limitation: litigation is usually a cross-reference

`extract_litigation` returns Item 3 verbatim, and most large filers use Item 3 only
to point at a contingencies note — MSFT's is 204 characters, NVDA's 199. The
`cross_referenced_only` flag marks this. The substance is in the notes to the
financial statements, so a short result means "look in the notes", not "no
litigation".

## MCP stdio hangs on long-lived nemo_openbb / nemo_sentry processes (overnight 2026-05-22; deeper investigation 2026-05-31)

**`nemo_openbb` REMOVED in `truth-source-refactor`.** The server
(`tools/openbb_server/`) is deleted, along with `testing/test_openbb_server.py`
and all `openbb-*` manifest pins. Reasons: (1) all four of its tools were
fully redundant with tools already living closer to source in servers we
keep — `obb_insider_trading` → finnhub `get_insider_transactions` +
`get_insider_sentiment`; `obb_analyst_consensus` → finnhub
`get_analyst_recommendations` + `get_forward_estimates` +
`get_analyst_revisions_history`; `obb_news_company` → finnhub
`get_company_news`; `obb_options_chain` → the merged `get_options_metrics`;
(2) zero `.claude/skills/` references called any `nemo_openbb` tool; and
(3) it is the server documented below as the indefinite-hang MCP process
that direct-Python-invocation workarounds were needed for. The nemo_openbb
paragraphs below are kept as historical record of that investigation — they
no longer describe a live tool. The nemo_sentry material in this section is
still current; that server was not touched by this refactor.

**Surfaced by:** Several MCP tool invocations hung indefinitely from the
Claude Code client side: `mcp__nemo_openbb__obb_insider_trading`,
`mcp__nemo_openbb__obb_analyst_consensus`, `mcp__nemo_sentry__sentry_get_queue`.
User reported "they have been calling/hanging the whole night".

### Update 2026-05-31 — two distinct failure modes confirmed

Process inspection found TWO instances of every MCP server running
simultaneously (one from `.venv\Scripts\python.exe`, one from the
uv-managed `~/.bun/python/...`). Each pair was 8.5 minutes old at
inspection time. The venv-python instances are tiny stubs (4MB, 0 CPU,
~65 handles); the uv-python instances are the active workers
(90-370MB, real CPU). Likely caused by running two Claude Code
sessions against the same project, each spawning its own MCP servers
under the user-scope registration.

A fresh isolated stdio MCP client + nemo_sentry server pair completes
`sentry_get_queue` in **2.0 seconds** end-to-end. So nemo_sentry's
code is healthy; the wedged behavior comes from the long-lived
session-bound process accumulating state. Restart CC to fix.

A fresh isolated stdio MCP client + nemo_openbb server pair **hangs
indefinitely** on the same tool call that completes in 7.4s when
invoked via direct Python. Exit code 143 (terminated). Stderr empty.
No stdout pollution from OpenBB SDK detected. Root cause is in the
interaction between OpenBB's heavy startup (~50 extensions) and the
MCP framework's asyncio TaskGroup — exact mechanism not yet pinned
down.

**Practical status:**
- nemo_sentry MCP tools: USABLE if CC was just restarted; degrade
  after several hours of uptime. Workaround: restart CC.
- nemo_openbb MCP tools: NOT USABLE via Claude Code's MCP layer in
  the current build. Workaround: invoke the handler directly via
  Python (`from tools.openbb_server.server import OpenBBServer;
  asyncio.run(...)`). Reliable, ~6-7s per call.

**Fix shipped (discovery-expansion-followup branch):**
- Lazy-import openbb on first tool invocation via `_get_obb()` singleton.
  Server startup is now < 1s so the MCP `initialize` handshake completes
  before Claude Code's manager timeout fires (was the root cause of SIGTERM).
- Added `asyncio.wait_for(..., timeout=45s)` around every `asyncio.to_thread`
  call so any residual hang surfaces as a structured `openbb_timeout` error
  rather than blocking the stdio loop indefinitely.

**What I measured:** invoking each handler directly via Python (bypassing
MCP stdio) completes cleanly:
- `sentry_get_queue`: 1.5s
- `obb_insider_trading`: 6.6s
- `obb_analyst_consensus`: 7.4s

So the handler logic is fine. The hang is somewhere in the MCP stdio
transport between Claude Code's MCP client and the long-lived server
process. `claude mcp list` reports all servers as "Connected" because
that health-check probably spawns a fresh process; the actual server
the session is communicating with may be in a stuck state.

**Likely causes (most → least probable):**
1. OpenBB SDK state accumulation. OpenBB auto-loads ~50 extensions at
   import. Over hours of uptime per-process state (cookies, cached
   tokens, connection pools) likely grows and slows or wedges.
2. yfinance backing the OpenBB calls rate-limiting on the per-process
   request count. A long-lived process accumulates a high request
   count.
3. Claude Code's MCP client lost sync with the server's stdio buffer
   (one side wrote more than the other consumed; deadlock).

**Workaround:** restart the Claude Code session, which respawns the MCP
server processes fresh. Symptoms returned after several hours of
uptime; not immediate.

**Fix shipped (discovery-expansion-followup branch):**
- nemo_sentry `call_tool` now dispatches all 19 synchronous tool methods
  via `asyncio.to_thread` with a 30s `asyncio.wait_for` timeout. A wedged
  long-lived process surfaces as `sentry_timeout` error in < 30s instead of
  blocking the stdio pipe forever. Restart CC to get a fresh process if the
  timeout is hit repeatedly.

**Priority:** medium. Workaround (restart) is cheap; bug only surfaces
after extended uptime.

## IPO exchange filter was over-strict (fixed)

**Surfaced by:** live testing on 2026-05-22 — 4 IPOs returned from the
5-day calendar, all dropped as `wrong_exchange`.

**Root cause:** `IPO_VALID_EXCHANGES` was a set
`{'NASDAQ', 'NYSE', 'NYSE ARCA', 'NYSE AMERICAN'}` doing exact
membership check. Finnhub actually returns full exchange names like
`'NASDAQ Capital'`, `'NASDAQ Global'`, `'NYSE MKT'`, `'NYSE Arca'` —
none of which match the set exactly.

**Fix:** switched to `IPO_VALID_EXCHANGE_PREFIXES = ('NASDAQ', 'NYSE')`
with `exchange.startswith(prefix)` check. Re-tested: 4 IPOs now pass
the exchange filter (all correctly drop to `below_min_cap` because
they're micro-caps below the $1B floor). Shipped on
`discovery-expansion-followup` branch.

**Priority:** done.

## Sentry daemons are running mixed-version code (overnight 2026-05-22)

**Surfaced by:** querying `sentry_get_discovery_status` showed all 5
new per-channel counters at 0 even though `sentry_queue` has 2 new
rag_analogue candidates (NVDA + MSFT) that landed at 08:52:59 today.

**Why it happens:** The triage daemon's `_maybe_run_daily_discovery`
calls `sentry_discovery.run_all()` via a fresh import — so it picks up
the NEW channels (rag_analogue, ipo, universe_insider, screener). But
the same daemon's `_record_discovery_run` was loaded at daemon startup
hours ago, before the new code shipped, so it doesn't know how to
write the new counter columns. Net effect: new channels run, queue
gets new candidates, but the audit row in `sentry_discovery_runs`
shows zeros for the new counters.

**Workaround:** restart the daemons via nemo.bat. Once restarted, the
daemon process picks up the new code and the counters land correctly.

**Real fix:** not required — this is expected behavior of mid-flight
code deployment. The lesson is to restart daemons after any commit
touching `daemons/*.py`.

**Priority:** none. Expected behavior.

## Accounting-identity sweep (2026-08-22)

**Surfaced by:** `testing/test_accounting_identities.py` — a network-gated
sweep over 32 filers (megacap, bank, REIT, retailer, biotech, industrial,
energy, minority-interest-heavy, multi-class) asserting relationships that
arithmetic forces to hold for every filer in every year. No golden values.
Run against each filer's latest 10-K.

The point of the approach: every serious extraction defect in this project
passed its own tests, because nothing compared one number to another. MSFT's
total assets read 207.7bn against a real 758.4bn and looked entirely
plausible. These checks catch that class of defect without anyone having
predicted the specific bug.

### Identity results (32 filers)

As first measured (2026-08-22), and re-measured after findings 1, 2 and 3
were fixed on `tier1-data-gaps` (2026-08-24):

| identity | holds | violated | not checkable | after the fix |
|---|---|---|---|---|
| 1a `Assets` == `LiabilitiesAndStockholdersEquity` | 32 | 0 | 0 | 32 / 0 / 0 |
| 1b `Assets` == `Liabilities` + equity + mezzanine | 25 | 0 | 7 | 25 / 0 / 7 |
| 2 component <= parent (24 concept pairs) | 32 | 0 | 0 | 32 / 0 / 0 |
| 2b reported revenue is the consolidated total | 21 | 11 | 0 | **32 / 0 / 0** |
| 3 segment revenue vs consolidated | 20 | 5 | 7 | 22 / 5 / 5 |
| 4 geographic revenue vs consolidated | 25 | 0 | 7 | 25 / 0 / 7 |
| 5 debt buckets vs long-term debt | 18 | 0 | 14 | 18 / 0 / 14 |
| 6a FCF == OCF − capex | 19 | 13 | 0 | **29 / 0 / 3** |
| 6b accruals == net income − OCF | 32 | 0 | 0 | 32 / 0 / 0 |
| 7 float <= shares × price | 31 | 0 | 1 | 31 / 0 / 1 |

The three filers 6a can no longer check are JPM, BAC and WFC: a bank tags no
capital-expenditure element, and `get_historical_fcf` now says so instead of
returning operating cash flow as free cash flow. The five 3 gains are filers
whose consolidated revenue the identity could not previously read.

Identities 1a, 1b, 2, 6b and 7 hold everywhere they can be evaluated. The
balance sheet foots to **exactly zero** for all 32 filers, so those checks
carry no meaningful tolerance — the allowance is 1e-9 of total assets, which
exists only so a float64 sum at JPM's 4.4tn scale cannot fail on
representation.

Not-checkable is a coverage fact, not a pass:

- **1b** — AMZN, CHTR, FOXA, HON, T, TGT and WMT do not tag
  `us-gaap:Liabilities` at all. Identity 1a covers them.
- **5** — 14 filers either tag a partial ladder or none. `get_debt_maturity_
  schedule`'s own `coverage` field already says so; the identity is only
  asserted where coverage is `full`.
- **3 / 4** — banks and single-segment filers.

### Confirmed wrong numbers

#### 1. FIXED — `sec_utils.filter_annual_data` returned dimensioned and prefix-matched facts (P0)

This is the shared read path behind `get_revenue_base`, `get_ebitda_margin`,
`get_capex_pct_revenue`, `get_tax_rate`, `get_depreciation`,
`get_margin_breakdown`, `get_historical_fcf` and `get_working_capital`. Two
defects compound:

1. It passes `xbrl.facts.query().by_concept(...)` straight through. `by_concept`
   is a **prefix match** — the same bug fixed in `sec_series._concept_matches`
   and never fixed here.
2. It resolves multiple facts for a period with `idxmax` — "the consolidated
   total is always the largest positive value". That is false. Segment,
   geography and parent-company-only facts sit in the same frame, and where the
   filer does not tag the requested concept undimensioned, one of those wins.

Measured against the filings:

| filer | tool reports | filing reports | what the tool actually returned |
|---|---|---|---|
| GOOGL | 342,721,000,000 | 402,836,000,000 | the Google Services **segment** |
| BA | 41,332,000,000 | 89,463,000,000 | a dimensioned slice |
| XOM | 226,909,000,000 | 332,238,000,000 | a dimensioned slice |
| GE | 30,163,000,000 | 45,855,000,000 | a dimensioned slice |
| WFC | 10,498,000,000 | order of magnitude larger | a dimensioned slice |
| SPG | 12,461,291,000 | 6,364,505,000 | a dimensioned fact, ~2x the total |
| CVX | 231,370,000,000 | 189,031,000,000 | a dimensioned fact |
| CAT | 73,955,000,000 | 67,589,000,000 | the reportable-segment aggregation member |

Alphabet's revenue is understated by 15% and Boeing's by 54% by the function
whose own docstring calls it "the starting point for nearly all analysis".

The same path drives operating cash flow in `get_historical_fcf`. For a bank
the consolidated figure is often negative, so `idxmax` picks the
parent-company-only Schedule I cash flow (`srt:ConsolidatedEntitiesAxis` /
`srt:ParentCompanyMember`):

| filer | tool reports | consolidated statement |
|---|---|---|
| JPM | +44,468,000,000 | **−147,782,000,000** |
| GS | +17,007,000,000 | **−45,154,000,000** |
| WFC | +25,946,000,000 | **−19,001,000,000** |
| BAC | +46,937,000,000 | +12,613,000,000 |

Three of the four flip sign. GE is the prefix-match half of the same defect:
8,543,000,000 is `NetCashProvidedByUsedInOperatingActivitiesContinuingOperations`
reached by prefix, against 8,537,000,000 for the concept actually asked for.

GS is the provenance-only case: the **value** 58,283,000,000 is Goldman's total
net revenues and is correct, but it is reported under `concept_used:
us-gaap:Revenues`, which GS does not tag at all — the fact is
`us-gaap:RevenuesNetOfInterestExpense`, reached by prefix match. A caller
reconciling against the filing would find nothing under the named concept.

**Fixed** on `tier1-data-gaps` (2026-08-24). `filter_annual_data` and
`filter_instant_data` route through `sec_series.concept_point` for exact-
concept filtering and `FilingPoint.undimensioned()` for consolidated selection,
so neither mechanism is reimplemented. `concept_used` names the element the
value is tagged under rather than the one that was asked for, which closes the
GS provenance case. A concept tagged only on dimensions returns None, so the
caller's chain moves to the element the filer does tag rather than being handed
a segment.

`latest_undimensioned()` grew an optional `span_days` window: a 10-Q carries the
year-to-date duration beside the quarter and both end on the same day, so
ranking by period alone returns nine months where three were asked for.

Regressions: `testing/test_consolidated_fact_selection.py`, one per figure
above, offline for the mechanism and network-gated for the filings.

#### 2. FIXED — revenue concept chains preferred the ASC 606 subset (P1)

Every revenue chain in the codebase tries
`us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax` before
`us-gaap:Revenues`. ASC 606 contract revenue is a *component* of revenue
whenever a filer earns anything outside a customer contract. A REIT earns
almost all of it under ASC 842 lease accounting:

- **AMT**: reported 935,900,000 against 10,644,600,000 of revenue — 8.8% of it.
- **WMT**: 706,413,000,000 against 713,163,000,000.

**Fixed** on `tier1-data-gaps` (2026-08-24). One shared chain,
`sec_utils.REVENUE_CONCEPTS`, broadest element first:

    us-gaap:Revenues
    us-gaap:RevenuesNetOfInterestExpense
    us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax
    us-gaap:SalesRevenueNet

`RevenuesNetOfInterestExpense` is named explicitly because it is the total-
revenue line on a bank's income statement and the only element GS and WFC tag
undimensioned — GS does not tag `us-gaap:Revenues` at all. It used to be
reached by accident through the prefix match and reported under the wrong name.
The unprefixed spellings that trailed the old chain are gone; no filing tags
them.

Reordering alone would have made two filers worse, which is why finding 1 had
to ship with it: with the order corrected and the selection still `idxmax`, the
largest `us-gaap:Revenues` fact is 452.209bn for XOM against a consolidated
332.238bn, and 48.024bn for GE against 45.855bn.

`REVENUE_ELEMENTS` in the identity test gained the same bank element, so
identity 2b can see a bank's revenue rather than reporting that it came off an
element the check does not know about.

#### 3. FIXED — `get_historical_fcf` reported operating cash flow as free cash flow (P0)

The capex chain is two concepts wide — `PaymentsToAcquirePropertyPlantAnd
Equipment` and `PaymentsForCapitalImprovements` — and `fcf = ocf - (capex or 0)`
turns a missing capex into zero rather than into "not covered".

| filer | reported FCF | capex the filing tags | real FCF |
|---|---|---|---|
| AMZN | 139,514,000,000 | 131,819,000,000 | **7,695,000,000** |
| T | 40,284,000,000 | 20,842,000,000 | 19,442,000,000 |
| NVDA | 102,718,000,000 | 6,042,000,000 | 96,676,000,000 |
| CVX | 33,939,000,000 | 17,347,000,000 | 16,592,000,000 |
| HD | 16,325,000,000 | 3,679,000,000 | 12,646,000,000 |
| SPG | 4,136,551,000 | 934,346,000 | 3,202,205,000 |
| REGN | 4,978,900,000 | 898,400,000 | 4,080,500,000 |
| PLD | 5,008,434,000 | 2,781,260,000 | 2,227,174,000 |

Amazon's free cash flow is overstated **18x**. All of these filers tag
`us-gaap:PaymentsToAcquireProductiveAssets` (PLD:
`PaymentsToDevelopRealEstateAssets`), which the chain never tries.

**Fixed** on `tier1-data-gaps` (2026-08-24). `sec_utils.CAPEX_CONCEPTS` is five
elements wide and shared with `get_capex_pct_revenue`, so the two tools cannot
disagree about what a filer tags. Every filer in the table above now reports the
real FCF in the right-hand column.

A filer that genuinely tags no capex — JPM, BAC and WFC in this basket — gets
`success: False`, `coverage: "not_covered"` and the list of elements tried,
never `ocf - 0`. The operating cash flow that was read is returned in the same
payload so nothing found is discarded. Identity 6a records those three as not
checkable, which is the honest verdict: free cash flow is not a meaningful
figure for a bank and the filing does not supply the input.

`get_capex_pct_revenue` gained coverage from the same change: HD, GE, CVX and T
were returning "Unable to find any concepts" and now resolve, and AMZN, NVDA,
REGN and O were returning a small unrelated fact the prefix match had reached
(O: 300,000 against a real 131,800,000).

#### 4. FIXED — `get_segment_financials` picked the wrong fact within a segment (P0)

`_annual_series` queried `by_dimension(axis, member)` and took the first row of
the result. Facts carrying the segment axis *plus another* axis
(`srt:ConsolidationItemsAxis`, `srt:ProductOrServiceAxis`,
`srt:StatementGeographicalAxis`) are in that result and won arbitrarily.

**GE**: Commercial Engines & Services reported **−62,000,000** of revenue. That
is the `us-gaap:IntersegmentEliminationMember` context. The segment-only context
in the same filing carries **33,252,000,000**. Both GE segments came back
negative and the tool reported total segment revenue of −1,748,000,000.

**Fixed** on `tier1-data-gaps` (2026-08-24). The selection rule, and what each
half of it is for:

- a fact whose only dimension is the segment axis is the segment's figure;
- where a filer tags none, the fact additionally qualified by
  `srt:ConsolidationItemsAxis = us-gaap:OperatingSegmentsMember` is — that is
  the segment column of the reconciliation table rather than a breakdown of it.
  AAPL, BA, COST, HON, JPM, NVDA, PLD, SPG and WFC tag their segments only that
  way, so a rule that demanded the segment axis alone would report nine filers
  in this basket as tagging nothing;
- anything else — a product, a geography, an intersegment elimination, a
  corporate reconciling item — is a piece of a segment or an adjustment to it,
  never the segment;
- the choice between the two is made **once per filing**, by which resolves more
  members, so every segment of one filer is on one basis. Per-member preference
  mixes them: AMT tags five members' non-lease revenue on the segment axis alone
  and all seven members' total revenue on the operating-segments column.

The concept chain is `sec_utils.REVENUE_CONCEPTS`, the same broadest-first chain
finding 2 established, for the same reason: AMT's ASC 606 element is the one
tagged on the segment axis alone and it is 8.8% of revenue. The read routes
through `sec_series.concept_point`, so the exact-concept filter and the
dimension resolution are the ones the rest of the module uses rather than a
third mechanism — which is also what recovers the banks, whose segment revenue
is `us-gaap:RevenuesNetOfInterestExpense` and used to be reached by prefix
accident and reported under `us-gaap:Revenues`.

Measured against the filings, latest segment total as a share of consolidated
revenue:

| filer | before | after | what changed |
|---|---|---|---|
| GE | −1,748,000,000 (−3.8%) | 42,120,000,000 (91.9%) | intersegment-elimination contexts, both segments negative |
| WFC | 6.1% | 101.4% | the ASC 606 fee-income fragment; segments are tagged `RevenuesNetOfInterestExpense` |
| HON | 30.2% | 99.9% | a product breakdown answered for the segment |
| BIIB | 35.9% | 100.0% | ditto |
| BA | 50.1% | 100.2% | ditto |
| GOOGL | 70.7% | 100.0% | Google Services answered by a `ProductOrServiceAxis` slice |
| PLD | 91.0% | 100.0% | ditto |
| T | 98.7% | 99.7% | ditto |
| META | 98.7% | 100.0% | ditto |
| WMT | 99.1% | 100.0% | segment total revenue rather than the net-sales fragment |
| CVX | 184.8% | 97.6% | the aggregation member read off its `OperatingSegmentsMember` context |
| XOM | 41.4% | **not extractable** | every segment fact also carries a geography axis |

AAPL, AMZN, COST, FOXA, HD, MSFT, NVDA, SPG, TGT and VRTX are unchanged. GS and
JPM return the same figures under the element they are actually tagged with
rather than under `us-gaap:Revenues`.

XOM is the case the tool now refuses rather than approximates: `success: False`,
naming the members and the reason. 41.4% was the sum of whichever
geography-qualified facts happened to come back first, and it is not a figure
the filing reports.

Regression: `testing/test_segment_fact_selection.py`, offline for the mechanism
against GE's, CAT's, AMT's, AAPL's and XOM's real context shapes, and
network-gated for the filings.

#### 5. `get_segment_financials` has no overlap detection (P1)

`get_geographic_revenue` detects nested members and sets `members_overlap`. The
segment tool has no equivalent, so aggregation and parent members are summed
alongside the members they aggregate:

- **CAT**: `us-gaap:ReportableSegmentAggregationBeforeOtherOperatingSegment
  Member` is 37,106m — exactly Construction 14,064 + Financial Products 2,841 +
  Power & Energy 15,558 + Resource 4,643. Segments sum to 109.8% of revenue.
- **CVX**: same member, 184.8%.
- **LEN**: `len:LennarHomebuildingEastCentralWestHoustonandOtherMember` is the
  sum of the five Homebuilding regions. 187.8%.
- **AMT**: `amt:PropertyMember` is the parent of the five Property regions.
  105.6%.

**Fix:** mirror `get_geographic_revenue` — compare the sum against consolidated
revenue and flag the overlap rather than trying to recognise parent members from
tag names, which cannot be done.

#### 6. `get_share_count_series` calls a corporate separation a buyback (P2)

HON's share count halves from 633,653,119 (2026-04-23 10-Q) to 316,940,010
(2026-07-23 10-Q). Both figures are correct — Honeywell separated — but the tool
reports `direction: "buyback"` and `change_pct: -50.1%`. A 50% "buyback" in one
quarter is not a buyback.

This also broke the first draft of identity 7: comparing the post-separation
10-Q share count against the 10-K's public float read as a 180% violation of an
identity that was not violated. The check now reads the share count off the same
cover page as the float, so both describe the same capital structure.

#### 7. `forward_metrics` reports a credential failure as "not disclosed" (P2)

`get_geographic_revenue` and `_series_for` wrap each concept attempt in
`except Exception: continue`. With `SEC_EMAIL` unset, `get_geographic_revenue`
returns *"NVDA does not disaggregate revenue by geography in its 10-K"* —
a statement about NVIDIA, caused by a missing environment variable. NVDA
discloses four geographies. `earnings_quality._series` swallows only
`NotCovered` and documents why; `forward_metrics` should match it.

### Legitimate violations, encoded narrowly

- **Mezzanine equity.** SPG carries 233,306,000 of redeemable interests, which
  sit between liabilities and equity and belong to neither total. Identity 1b
  fails by exactly that amount without the term. Added as an explicit term, not
  absorbed by a wider tolerance.
- **Minority interest.** Total equity is
  `StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest` where
  tagged, otherwise `StockholdersEquity` + `MinorityInterest`. Checking the
  parent figure alone would report every filer with a subsidiary as broken
  (CHTR's NCI is 4.5bn). With the fallback, all 25 checkable filers foot exactly
  — including all four banks, where a naive check would have produced the
  loudest false alarms.
- **MSFT's debt ladder, 14.5% from the balance sheet.** Adjudicated against the
  filing, which reconciles it itself: face value 46,136 − 1,081 discount − 11
  hedge − 4,750 premium on debt exchange = 40,294 carrying. Both sides are
  correct. Encoded as a named per-filer exception at 16%, leaving the general
  tolerance at 10% (worst other filer: T at 7.4%).
- **`members_overlap` on geographic revenue.** BAC, CAT, GE, JPM, META, MRNA and
  VRTX tag nested regions. The tool already detects this; identity 4 respects
  the flag and checks the things the flag does not imply instead.

### Rate limiting is a silent-failure mode

Roughly 70 SEC requests per filer across seven tools. Two thousand back to back
earns a 429 that blocks the host for ~9 minutes — and every tool reports it as
"No filing found" or "not covered". A throttled run therefore produces false
"not checkable" verdicts and can make a defect register look stale.

The sweep detects rate-limited results, retries with backoff, and
`test_no_tool_result_is_a_throttled_request` fails the run if any survive, so a
degraded sweep can never be read as coverage.

### Unresolved

- **XOM segment revenue is not extractable as tagged.** XOM tags every segment
  revenue fact in combination with `srt:StatementGeographicalAxis`; there is no
  segment-only fact. A correct segment figure requires summing across the
  geography axis, which no current tool does.
- **Two facts in the same context, same period, same label.** AMZN tags
  `IncomeTaxExpenseBenefit` twice undimensioned in c-1 — 19,087,000,000 and a
  rounded 19,100,000,000 — and TGT tags pre-tax income as 4,767,000,000 and
  4,800,000,000. `latest_undimensioned` ranks by period, so a tie falls to
  document order, which happens to be the precise figure in both cases. It is
  not a rule: the frame carries a `decimals` column that would settle it and
  `ConceptFact` does not keep it. Same class as the AMT `NetIncomeLoss` case
  under the reconciliation sweep's defect 5.
- **`get_ebitda_margin` cannot serve a filer that tags no operating income.**
  Eleven of the 32 — every bank, both REITs in the basket, XOM, CVX, GE, BIIB,
  FOXA and LEN — return a "missing EBITDA components" error where they used to
  return an EBITDA built on pre-tax income reached by prefix match (JPM:
  72,595,000,000; GS: a fact worth $1). The error is correct and the coverage
  gap is real: there is no consolidated operating income in those filings.
- **`get_margin_breakdown` gross profit is unavailable for AMT, GE, HON and
  RIOT.** Each tags `us-gaap:GrossProfit` only on a dimension, or not for an
  annual period. The tool used to report the dimensioned segment figure as
  consolidated.
- **Identity 5 reaches only 18 of 32 filers**, and identity 3 only 27, because
  of genuine tagging gaps rather than tool defects.
- **Real-estate acquisition spend** (`PaymentsToAcquireRealEstate`,
  `PaymentsToAcquireCommercialRealEstate`) is deliberately excluded from the
  capex identity. Whether a REIT's property acquisitions belong in free cash
  flow is a judgement call, not an identity.

**Priority:** findings 1, 2 and 3 were P0/P1 and are fixed — the sweep now runs
14/14 green with 2b at 32/0/0 and 6a at 29/0/3. Findings 4 and 5
(`get_segment_financials`) are still open and are the only entries left in
`KNOWN_DEFECTS` in `testing/test_accounting_identities.py`;
`test_known_defect_register_is_not_stale` fails the moment one is fixed, so the
register cannot outlive the bug.

## Cross-source reconciliation sweep (2026-08-22)

Branch `tier1-data-gaps`. New test:
`testing/test_cross_source_reconciliation.py`, network-gated the same way as
`testing/test_research_coverage_sweep.py`. 37 tickers, 444 comparisons,
about 7 minutes when EDGAR is not throttling and roughly twice that when it
is.

The premise: every defect this project has shipped passed its own tests,
because a test written beside the code inherits the code's blind spots.
Reconciliation does not. The same fact is available from SEC XBRL, from
Yahoo and from Finnhub, and where they disagree materially at least one is
wrong. Nothing below was being looked for; all of it fell out of comparing
feeds.

Every disagreement was adjudicated by pulling the raw XBRL facts out of the
filing — concept, label, statement and context dimensions — not by preferring
whichever source looked nicer. **No tolerance was widened to make a check
pass.** Adjudicated cases live in the test's `ADJUDICATED` table; a
disagreement outside it fails the run, and an entry inside it that starts
agreeing also fails, so entries get deleted when bugs get fixed rather than
outliving them.

### Reconciliation table

As first measured (2026-08-22). `revenue_annual` was re-measured after
defect 1 was fixed and reads **34 / 2 / 1 / 0**; nothing else in the table
moved.

| check | agree | disagree | not comparable | absent |
|---|---|---|---|---|
| revenue_annual (SEC XBRL vs vendor 10-K) | 27 | 9 | 1 | 0 |
| revenue_ttm (Yahoo vs Finnhub) | 27 | 7 | 2 | 1 |
| vendor_internal_revenue (Finnhub vs Finnhub) | 33 | 3 | 0 | 1 |
| shares (SEC dei vs Yahoo) | 29 | 5 | 0 | 3 |
| shares_finnhub (SEC dei vs Finnhub profile) | 36 | 0 | 0 | 1 |
| market_cap (SEC shares x price vs Yahoo) | 32 | 2 | 0 | 3 |
| market_cap_internal (Yahoo vs Yahoo) | 29 | 5 | 0 | 3 |
| enterprise_value_internal (Yahoo vs Yahoo) | 27 | 3 | 4 | 3 |
| net_income (SEC XBRL vs vendor) | 33 | 2 | 2 | 0 |
| operating_cash_flow (SEC XBRL vs vendor) | 33 | 0 | 4 | 0 |
| total_assets (SEC XBRL vs vendor) | 35 | 0 | 2 | 0 |
| pe_ratio (ours vs Yahoo's own trailingPE) | 29 | 1 | 0 | 7 |

Tolerances come from the measured spread rather than being chosen. Each band
sits above the widest drift among the pairs that genuinely agree and below the
narrowest genuine disagreement: 1% on period-matched statement lines (widest
agreement 0.24%), 2% on annual revenue (1.14% vs 3.6%), 2% on share counts
(1.82% vs 15%), 5% on market cap (4.01% vs 13.4%), 3% on TTM revenue (1.55%
vs 3.3%), 10% on Finnhub's internal revenue consistency (8.1% vs 115%), 20%
on P/E (16.5% vs 96.9%). The one band not set that way is enterprise value at
10%: market cap + debt − cash omits minority interest and preferred, which a
proper EV includes, and that incompleteness — not vendor drift — is what puts
AMT at 5.2% while the rest of the basket sits inside 3.2%. The failure that
check exists to catch is 5500% away.

SPG straddles that 10% band as its price moves — measured at 9.7% and 10.3%
a day apart — and it was adjudicated rather than accommodated. Widening the
band to swallow it would have cost the AMT-sized signal and bought nothing,
and chasing it turned out to be the strongest single piece of evidence in the
sweep (see the SPG entry under vendor behaviour below).

Basket: MSFT AAPL GOOGL AMZN META NVDA (megacap), WSM DKS EXPD (mid),
PLUG RIOT SAVA (small), O SPG PLD AMT (REIT), JPM BAC WFC GS (bank),
BIIB MRNA REGN (biotech), WMT COST TGT HD (retail), ARM RDDT CART (recent
IPO), GOOGL META RDDT DKS (multi-class), F CAT GE (industrial), XOM CVX
(energy), TSM ASML (foreign/ADR).

**A second caveat: SEC rate limiting silently degrades this table.** A
throttled EDGAR request comes back looking exactly like a company that does
not tag a concept, so a 429 storm turns into false "not covered" rows and,
worse, could turn into a false disagreement. One evidence-gathering run was
hit this way and lost six `revenue_annual` comparisons and eight share
counts. Three things keep it honest: `_sec_call` retries with exponential
backoff and evicts the poisoned negative cache entry, every check carries a
minimum-comparisons floor that fails loudly rather than passing on an empty
set, and the table above was taken from a clean run and cross-checked against
an earlier clean run. The two clean runs produced the same 36 disagreements;
the 37th, SPG's enterprise value, is a price-dependent case that crossed the
band between them and is adjudicated below. None of the adjudications came
from a throttled run: every one was settled by reading the filing's raw XBRL
facts directly.

All 37 material disagreements are adjudicated. Nothing in the table is
recorded as "sources differ, unclear why".

**One caveat on the table.** Finnhub's `/stock/financials` is paywalled on
the free tier, so `get_financial_statements` falls back to yfinance and tags
itself `_source: yfinance_fallback`. Every statement comparison above is
therefore SEC XBRL against Yahoo, not against Finnhub. Finnhub contributes
genuinely independent data only through `/stock/metric` and `/stock/profile2`.
`test_vendor_statements_declare_their_true_source` asserts the tag is present
so the table cannot be misread as a three-way agreement.

### Confirmed defects, in severity order

**1. FIXED — `get_revenue_base` returned the wrong revenue for 9 of the 36
filers it could be compared on.**
`tools/web_search_server/sec_utils.py`. Two independent causes, both
verified against the filings. Fixed on `tier1-data-gaps` (2026-08-24); the
re-run reads **34 agree / 2 disagree / 1 not comparable**, and the two
remaining are a definitional split rather than the same defect — see the end
of this entry.

*Cause A — the ASC 606 element is tried ahead of `us-gaap:Revenues`.* For a
lessor, a bank or a segment-reporting filer that element covers a fragment
of revenue, and the consolidated total sits on `us-gaap:Revenues`:

| ticker | tool | filing | error | what the tool actually returned |
|---|---|---|---|---|
| AMT | 0.9359bn | 10.6446bn | -91.2% | the fact AMT labels "Total non-lease revenue"; tower rents are ASC 842 lease income |
| WFC | 10.498bn | 83.699bn | -87.5% | the fact WFC labels "Fee income" |
| GE | 30.163bn | 45.855bn | -34.2% | the fact GE labels "Sales" |
| XOM | 226.909bn | 332.238bn | -29.9% | one segment, off `DisclosuresaboutSegments...SalesandOtherOperatingRevenuesDetails` |
| GOOGL | 342.721bn | 402.836bn | -14.9% | Google Services. Every ASC 606 fact in Alphabet's FY2025 10-K is on the segment disclosure; none is on the income statement |

*Cause B — `filter_annual_data` takes the largest fact for the period.* Its
comment states "the consolidated total is always the largest positive value".
It is not: a segment aggregate is struck before intersegment eliminations,
and a joint-venture disclosure is not the filer's revenue at all.

| ticker | tool | filing (undimensioned context) | error | context the tool picked |
|---|---|---|---|---|
| SPG | 12.4613bn | 6.3645bn | +95.8% | `EquityMethodInvestmentNonconsolidatedInvesteeAxis=spg:PlatformInvestmentsExcludingTrgAndKlepierre` — the unconsolidated JVs' revenue |
| CVX | 231.370bn | 184.432bn | +25.5% | `srt:ConsolidationItemsAxis=us-gaap:OperatingSegmentsMember` |
| CAT | 73.955bn | 67.589bn | +9.4% | a dimensioned context; consolidated is c-1 |
| RIOT | 670.718m | 647.435m | +3.6% | `OperatingSegmentsMember` + `ReportableSegmentAggregationBeforeOtherOperatingSegment` |

Fixing cause A alone makes XOM and GE worse, not better: the largest
`us-gaap:Revenues` fact is 452.209bn for XOM (against a consolidated
332.238bn) and 48.024bn for GE (against 45.855bn). Both causes have to go.

The fix already existed in this codebase. `sec_series.FilingPoint.undimensioned()`
and `latest_undimensioned()` select the consolidated fact by *absence of
dimensions* rather than by size, and every tool built on them —
`get_accruals_quality`, `get_working_capital_trends`, `get_share_count_series`
— reconciled cleanly here. `filter_annual_data` now routes through
`sec_series.concept_point` and that same selection rather than reimplementing
either half, and the revenue chain is reordered broadest-first with
`us-gaap:RevenuesNetOfInterestExpense` named explicitly for the banks. The
mechanism is written up under the accounting-identity sweep's findings 1 and 2
above.

Seven of the nine now agree and their `ADJUDICATED` entries are deleted. **XOM
and CVX still disagree, and not for the same reason.** Both now return the
consolidated total the filer tags, and Yahoo reports the narrower sales-only
line:

| ticker | tool now | Yahoo | gap | what the two figures are |
|---|---|---|---|---|
| XOM | 332.238bn | 323.905bn | +2.6% | `us-gaap:Revenues`, "total revenues and other income", against the `ProductOrServiceAxis=SalesAndOtherOperatingRevenue` line |
| CVX | 189.031bn | 184.432bn | +2.5% | `us-gaap:Revenues`, "total revenues and other income", against the ASC 606 "sales and other operating revenues" element |

Identity 2b requires the broader element — a tool returning the narrower one
while the filer tags a larger revenue element undimensioned is reporting a
fragment — so the definitional call is settled in favour of `us-gaap:Revenues`,
and the divergence from Yahoo is recorded rather than tuned away. This also
settles the "XOM's revenue has two defensible answers" item under **Not
resolved** below.

**2. `get_market_data` mixes currencies inside a single response.**
`tools/financial_modeling_engine/utils.py`. For a foreign filer `marketCap`
and `currentPrice` come back in the quote currency while `totalRevenue`,
`ebitda`, `netIncomeToCommon`, `totalDebt`, `totalCash` and `enterpriseValue`
come back in the reporting currency, and `pe_ratio`/`pb_ratio` divide one by
the other. The response carries no currency field at all.

- TSM: `pe_ratio` = **0.98** against Yahoo's own `trailingPE` of **31.24** —
  off by the TWD/USD rate. `pb_ratio` = 0.41 on the same mix.
- ASML: `pe_ratio` = 63.68 against Yahoo's 59.97, inflated by EUR/USD.

Yahoo already publishes the correct `trailingPE` in the same `info` dict the
tool has in hand.

**3. `get_market_data` publishes EV multiples three orders of magnitude out
without a plausibility check.** ASML's yfinance `enterpriseValue` reads
37,631bn against a 677bn market cap and roughly 2bn of net debt. The tool
divides it straight into revenue and EBIT and returns
`ev_revenue = 1065x`, `ev_ebitda = 2790x`, `ev_ebit = 3330x`. Reconciling EV
against market cap + debt − cash from the same response catches it: every
non-financial in the basket lands within 5.2% apart from SPG (adjudicated
separately below), and ASML and TSM at 5500%.

**4. `get_market_data.revenue_ttm` does not reconcile to Yahoo's own
quarters.** Settled by summing the four most recent reported quarters from
yfinance's own quarterly income statement. Finnhub's implied TTM
(`marketCapitalization / psTTM`) matches that sum to within 0.0004% — the
rounding of `psTTM` itself. Yahoo's `info.totalRevenue`, which the tool
returns as `revenue_ttm` and divides into enterprise value for `ev_revenue`,
matches no four-quarter window of Yahoo's own statements:

| ticker | Yahoo `info.totalRevenue` | sum of last 4 reported quarters (= Finnhub) | gap |
|---|---|---|---|
| PLD | 9.657bn | 9.1898bn | +5.1% |
| SPG | 6.941bn | 6.6486bn | +4.4% |
| RIOT | 674.5m | 653.3m | +3.3% |

Finnhub is right here and Yahoo is the unreconcilable side — the reverse of
the usual direction, which is the point of running the check both ways.

**5. `get_accruals_quality` picks arbitrarily between two conflicting facts
in the same context.** AMT tags `us-gaap:NetIncomeLoss` twice in the
undimensioned context c-1: 2.6285bn labelled "Net income (loss)" on
`CONSOLIDATEDSTATEMENTSOFEQUITY`, and 2.5295bn labelled "NET INCOME
ATTRIBUTABLE TO AMERICAN TOWER CORPORATION COMMON STOCKHOLDERS" on the income
statement. `_by_period` keeps whichever appears first in document order and
returns the equity-statement figure. The income statement and both vendors
say 2.5295bn. Net income reads 3.9% high and the accrual ratio inherits it.

**6. A rate limit is reported as "No XBRL data available", and cached.**
`get_latest_filing` in `sec_utils.py` wraps the whole fetch in
`except Exception: result = None`, caches the None, and the caller renders it
as "No XBRL data available" — a statement about the filer. Reproduced twice
while building this sweep: once from an HTTP 429 partway through the basket
(19 tickers turned "uncovered" in one step) and once from an unset
`SEC_EMAIL`. Because the failure is cached, a retry inside the same process
cannot recover; the harness has to evict the entry, which is why
`_sec_call()` in the new test reaches into `_filing_cache_lru`. Contrast
`get_accruals_quality` and `get_share_count_series`, which surfaced the same
429 as "Too Many Requests".

**7. `get_revenue_base` raises for 20-F/40-F filers.** The foreign branch
`return _foreign_revenue_base(ticker)` sits *above* the function's own
`try`, so anything the foreign reader raises propagates instead of coming
back as `{'success': False, ...}`. Observed live: a missing `SEC_EMAIL`
raised `ValueError` out of `get_revenue_base('TSM', '20-F')` while the same
condition on the domestic path returned a result dict.

**8. `get_revenue_base` returns two different `period_end` formats.** The
10-K path returns `'2025-12-31'`; the 20-F path returns the raw XBRL period
key, `'duration_2025-01-01_2025-12-31'`. Any caller parsing it as a date
breaks on every foreign private issuer. The test normalises both forms and
notes why — without that, ASML, TSM and ARM all read "not comparable" when in
fact all three match the vendor to the dollar (EUR 32,667,300,000 /
TWD 3,809,054,300,000 / USD 4,920,000,000).

**9. `get_basic_financials` contradicts itself, and drops currency.**
`tools/news_agregator/finnhub_server.py`, `_condense_basic_financials`.

- *Two TTM revenues from one payload.* `marketCapitalization / psTTM` and
  `revenuePerShareTTM x shareOutstanding` are the same quantity by
  construction. For banks they are not: JPM 332.6bn vs 138.6bn (2.4x), BAC
  192.1bn vs 87.4bn, WFC 141.1bn vs 65.7bn. Yahoo says 186.3 / 113.9 / 83.0bn.
  Non-financials agree within 8%. Nothing in the response says which basis
  either figure is on.
- *No currency.* The condenser keeps `marketCapitalization` and
  `enterpriseValue` but drops every currency signal. TSM's
  `marketCapitalization` is 63,145,320 — **millions of TWD** (~$2.0tn), read
  as USD it is $63tn. `get_company_profile` keeps `currency` in its allow-list
  and correctly reports TWD for the same company.
- *Stale.* `/stock/metric`'s `marketCapitalization` drifts from the live
  figure by up to 12% across the basket and by **-56.5% for MRNA** (25.19bn
  against 57.94bn). `/stock/profile2`'s market cap matched Yahoo's to the
  dollar for MRNA, META and DKS, so the staleness is specific to the metric
  endpoint. `psTTM`, `peTTM` and the rest of that block inherit it.

**10. `get_share_count_series` returns ordinary shares for an ADR with no
note.** TSM: SEC dei reports 25,932,524,521 ordinary shares, Yahoo 5,186,474,013
ADS, and the ratio is 5.00003 — TSM's 5:1 ADR ratio. Finnhub's profile also
reports ordinary shares, so two of three sources agree and the tool is right in
its own unit. But the quote is per ADS, so `latest_total x price` gives a
$10.86tn market cap against a real $2.17tn, and nothing in the result warns
that the two cannot be multiplied.

**11. `get_market_data` returns a null market cap for HD and TGT.**
Reproducible across repeated calls: `marketCap` and `sharesOutstanding` come
back `None` while `currentPrice` and `revenue` are populated. `get_data` then
omits `pe_ratio` and `pb_ratio` from the dict entirely rather than setting them
to None, so a caller indexing them gets a `KeyError` on exactly the tickers
where the data is thinnest. SEC dei has HD at 997,116,682 shares and TGT at
454,191,112, so the fallback exists.

**12. `_yf_financial_statements` misses an operating-cash-flow label.** Its
`cf` map has only `"Operating Cash Flow"`. ASML's yfinance cash-flow statement
carries no such row — only `"Cash Flow From Continuing Operating Activities"` —
so `operatingCashFlow` comes back absent for it. (RIOT is a genuine vendor
gap by contrast: the row exists and is NaN for 2022-2025.)

### Vendor behaviour our outputs inherit — adjudicated, not our bug

- **yfinance `sharesOutstanding` is the quoted class, not the company.**
  GOOGL 5.867bn vs an SEC dei total of 12.23bn (+108%), META 2.205 vs 2.5475bn,
  RDDT 146.1 vs 192.4m, DKS 65.9 vs 89.5m. Settled two ways: Finnhub's profile
  count agrees with SEC dei for all 36 tickers where both exist, and Yahoo's
  own `marketCap` reconciles with the SEC total at the quoted price rather than
  with its own share count. The consequence is that `get_market_data` returns a
  `marketCap` and a `sharesOutstanding` that cannot both be right — 52% apart
  for GOOGL, 26% for DKS, 24% for RDDT, 13% for META.
- **yfinance `marketCap` for SPG includes operating-partnership units.** Three
  independent measures agree on the share count and Yahoo's `marketCap` is the
  one that does not: SEC dei says 323,559,515, Yahoo's own `sharesOutstanding`
  says 323,551,515 (0.002% apart), Finnhub's profile agrees, and Yahoo's
  `marketCap` divided by Yahoo's own price implies 379.6m — 56.1m more, which
  is the SPG LP unit count. The fourth measure is Yahoo's own
  `enterpriseValue`: it reconciles to SEC-shares x price + debt − cash within
  **0.78%** (99.73bn against a reported 100.50bn) but sits 10.3% away from its
  own `marketCap` + debt − cash. So Yahoo builds EV on the common-only market
  cap while publishing an OP-unit-inclusive `marketCap` in the same payload.
  A fully-exchanged market cap is a defensible number; shipping it beside a
  `sharesOutstanding` and an `enterpriseValue` that both assume the other one
  is not. `get_market_data` passes all three through untouched, so a caller can
  compute SPG's equity value two ways from one response and get 70.6bn or
  83.7bn.
- **Bank revenue, gross vs net of interest expense.** Finnhub's `psTTM` implies
  gross revenue including total interest income; Yahoo's `totalRevenue` is net.
  GS 135.0 vs 67.6bn, JPM 332.6 vs 186.3bn, WFC 141.1 vs 83.0bn, BAC 192.1 vs
  113.9bn. Both are real measures of a bank. Neither source labels which.

### Definitional divergence worth a warning, not a fix

**SPG net income, 5.36412bn vs 4.6276bn (+15.9%).** SPG tags no
`us-gaap:NetIncomeLoss` at all — only `NetIncomeLossAvailableToCommon
StockholdersBasic` and siblings — so `get_accruals_quality`'s concept chain
falls through to `us-gaap:ProfitLoss` = "Consolidated Net Income", which
includes the operating partnership's noncontrolling interests. Yahoo reports
4.6276bn to common. The 15.9% gap is SPG's NCI. `concepts_used` does name
ProfitLoss, but nothing warns that this figure is not the per-share numerator,
and for an UPREIT the two differ by a sixth.

### Not resolved

- **XOM's "revenue" has two defensible answers — decided.** `us-gaap:Revenues`
  undimensioned = 332.238bn ("total revenues and other income", including
  equity-affiliate and other income); the `ProductOrServiceAxis=
  SalesAndOtherOperatingRevenue` line = 323.905bn, which is what Yahoo reports.
  The tool used to return 226.909bn, which is neither. `get_revenue_base` now
  returns 332.238bn, because identity 2b forbids returning a narrower element
  while a broader one is tagged undimensioned. CVX is the same call at 189.031
  against 184.432bn. Both stay in `ADJUDICATED` as definitional divergence.
- **ASML's `enterpriseValue` of 37,631bn.** Not EUR, not USD, not a
  share-count artefact — could not determine what Yahoo computed. The
  actionable half (we publish it as an EV multiple unchecked) is defect 3.
- **SAVA has no Yahoo quote.** `Quote not found for symbol: SAVA` on every
  attempt, while SEC has 10-Q filings through 2026-07 and a 48,307,896 share
  count. Ticker change or Yahoo delisting; not investigated further. All 11
  price-side checks report absent for it, which is the correct behaviour.

### Priority

1 (revenue) is fixed. 2 (currency mix) and 5 (net income) are wrong numbers
reaching valuation code and are the next to fix. 6 (rate limit as "no data") is the
highest-leverage robustness fix: it silently converts an outage into a claim
about a company. 3, 4, 9 and 11 are wrong or missing numbers with narrower
blast radius. 7, 8 and 12 are contract and coverage fixes.
