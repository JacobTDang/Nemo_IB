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

| identity | holds | violated | not checkable |
|---|---|---|---|
| 1a `Assets` == `LiabilitiesAndStockholdersEquity` | 32 | 0 | 0 |
| 1b `Assets` == `Liabilities` + equity + mezzanine | 25 | 0 | 7 |
| 2 component <= parent (24 concept pairs) | 32 | 0 | 0 |
| 2b reported revenue is the consolidated total | 21 | 11 | 0 |
| 3 segment revenue vs consolidated | 20 | 5 | 7 |
| 4 geographic revenue vs consolidated | 25 | 0 | 7 |
| 5 debt buckets vs long-term debt | 18 | 0 | 14 |
| 6a FCF == OCF − capex | 19 | 13 | 0 |
| 6b accruals == net income − OCF | 32 | 0 | 0 |
| 7 float <= shares × price | 31 | 0 | 1 |

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

#### 1. `sec_utils.filter_annual_data` returns dimensioned and prefix-matched facts (P0)

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

**Fix:** apply `sec_series._concept_matches` to the frame, and prefer
undimensioned facts over `idxmax`. Not shipped here: `filter_annual_data` is
the read path for ten tools and several rely on its current selection, so the
change needs its own regression pass over the golden tests.

#### 2. Revenue concept chains prefer the ASC 606 subset (P1)

Every revenue chain in the codebase tries
`us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax` before
`us-gaap:Revenues`. ASC 606 contract revenue is a *component* of revenue
whenever a filer earns anything outside a customer contract. A REIT earns
almost all of it under ASC 842 lease accounting:

- **AMT**: reported 935,900,000 against 10,644,600,000 of revenue — 8.8% of it.
- **WMT**: 706,413,000,000 against 713,163,000,000.

**Fix:** where a filer tags both, take `us-gaap:Revenues`. The containment
identity (`ASC 606 <= Revenues`) holds for every filer in the basket, so
"largest wins" is safe.

#### 3. `get_historical_fcf` reports operating cash flow as free cash flow (P0)

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

**Fix:** widen the chain, and return `coverage: "not_covered"` instead of
substituting zero — a silent fallback masking a missing input is exactly the
failure mode this project keeps relearning.

#### 4. `get_segment_financials` picks the wrong fact within a segment (P0)

`_annual_series` queries `by_dimension(axis, member)` and takes the first row of
the result. Facts carrying the segment axis *plus another* axis
(`srt:ConsolidationItemsAxis`, `srt:ProductOrServiceAxis`) are in that result
and win arbitrarily.

**GE**: Commercial Engines & Services reports **−62,000,000** of revenue. That is
the `us-gaap:IntersegmentEliminationMember` context. The segment-only context in
the same filing carries **33,252,000,000**. Both GE segments come back negative
and the tool reports total segment revenue of −1,748,000,000.

The same defect understates BA (50.1% of consolidated), HON (30.2%), XOM (41.4%)
and BIIB (35.9%).

A prototype that selects the fact whose only dimension is the segment axis fixes
GE (91.9%), BA (100.2%), HON (99.9%), BIIB (100.0%) and CVX (97.6%).

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
- **`filter_annual_data` is not fixed.** Ten tools read through it; the fix is
  clear (exact-concept filter, prefer undimensioned) but needs its own
  regression pass.
- **Identity 5 reaches only 18 of 32 filers**, and identity 3 only 27, because
  of genuine tagging gaps rather than tool defects.
- **Real-estate acquisition spend** (`PaymentsToAcquireRealEstate`,
  `PaymentsToAcquireCommercialRealEstate`) is deliberately excluded from the
  capex identity. Whether a REIT's property acquisitions belong in free cash
  flow is a judgement call, not an identity.

**Priority:** findings 1, 3 and 4 are P0 — they produce confidently wrong
numbers on megacaps and banks. Each is registered in
`KNOWN_DEFECTS` in `testing/test_accounting_identities.py`, and
`test_known_defect_register_is_not_stale` fails the moment one is fixed, so the
register cannot outlive the bug.
