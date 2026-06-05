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

6. **Put-call parity guard not live until next MCP restart** — the guard
   (commit 22e102b) rebuilds both legs from last_price when
   |C−P−(S−K)|/S > 5%. The running altdata server process predates the
   commit; restart Claude Code to activate it.

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
2. **BUDGET-GATED: Tier 2 data** — SimilarWeb scaffold is shipped and
   key-gated (`tools/preearnings/web_traffic.py`); next are Panjiva
   (shipments), AlphaSense (transcripts), Ortex (live SI), card panels.
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

## MCP stdio hangs on long-lived nemo_openbb / nemo_sentry processes (overnight 2026-05-22; deeper investigation 2026-05-31)

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
