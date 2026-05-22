# Known Issues

Findings worth fixing but not blocking the current build. Each entry
includes what surfaced it, what the fix looks like, and rough priority.

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
