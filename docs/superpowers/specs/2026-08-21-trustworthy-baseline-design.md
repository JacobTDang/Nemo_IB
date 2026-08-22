# Trustworthy Baseline — Design

**Date:** 2026-08-21
**Status:** approved, not started
**Depends on:** `2026-08-20-truth-source-refactor-design.md` (PR #11) landing first
**Blocks:** `2026-08-20-primary-source-swap-design.md` and any future work
**Scope:** test-suite trustworthiness, environment buildability, and the production
defects that measurement exposed. No new features. No transport change.

## Problem

The offline suite reports **80 failed / 759 passed / 310 skipped / 5 errors**. It has
never been green in this environment, so the red is treated as background noise. That
is the defect. A baseline nobody trusts cannot catch a regression, and this has already
cost real money once: `test_phase_00_schema.py:19` was silently invalidated by the
truth-source refactor's cache/state split and passed anyway, because this machine's
`session.db` still held abandoned tables. It was caught only by the final whole-branch
review, by luck rather than by process.

Worse, the noise is not merely uninformative — it is **wrong in both directions**.
Measured 2026-08-21:

- `test_phase_A3` + `A4` + `06b` contribute 4 failures in-suite but pass **29/29** when
  run in isolation. The suite invents failures.
- `test_falsifier_watcher_e2e` fails 6 in-suite but **8** in isolation. The suite hides
  failures.

A fresh database for the whole run moves the total by one (80 → 81), which proves the
contamination is generated *during* a run, not inherited from previous ones.

## Guiding principle

Carried forward from the truth-source refactor, which asked "does this tool return a
fact about the world, or a fact about my book?" The analogous seam here:

> Is this red telling me something about my code, or something about my machine?

Every failing test must be one or the other, and the suite must say which. A test that
cannot run offline should say so and skip; a test that fails should mean a defect.
Anything that fails for an environmental reason is miscategorised, not unlucky.

## Evidence

Full root-cause taxonomy, measured rather than assumed. Counts sum to 85 (80 failures +
5 errors).

| Count | Class | How it was established |
|---:|---|---|
| 28 | Genuinely need a live LLM | retried real endpoints under a fake key until timeout |
| 14 | **Pure logic, blocked by an eager credential check** | the 4 affected files went 0 → **17 passed in 1.47s with zero network** under a fake key (14 previously failing, 3 already green) |
| 10 | Need the project's SearXNG on `:8888` | repo `docker-compose.yml` maps `8888:8080`; port is dead. The healthy `firecrawl-searxng-1` container does not publish its port to the host |
| 10 | Need `CLAUDE.md`, which is gitignored | `.gitignore:3`; no copy exists anywhere in the repo |
| 7 | Need `.mcp.json`, deliberately deleted | `00abf63` moved it to `.bak`; `d12e25c` deleted the `.bak` |
| 6 | `falsifier_alerts` table never created | absent from `init_schema()` |
| 5 | Script-style functions, not tests | parameters resolved as fixtures |
| 4 | Cross-test coupling | 29/29 pass in isolation |
| 1 | Stale premise — lxml now parses natively | the test's own assertion message says so |

**The 14 is the central finding.** `Financial_Modeling_Agent.__init__` raises
`ValueError` when `GROQ_API_KEY` is absent. `test_fix_14_regime_weights`,
`test_fix_01_lbo_hy_spread`, `test_bugfix_02_credit_guard`, and
`test_fix_04_beat_rate_signals` construct that agent only to reach deterministic
methods — LBO spread math, regime weights, credit guards, prompt strings. Eager
validation in a constructor holds 14 real tests hostage to a credential their code
paths never use.

## Deliverables

### D1 — Lazy credential resolution, with an explicit eager check

Move client construction out of `__init__` and onto first use in the two base classes
that own the credential check: `agent/groq_template.py:35-45` and
`agent/openrouter_template.py:139-152`. The same `ValueError` still raises, with the
same message — later, not never.

`agent/Financial_Modeling_Agent.py` needs no edit. It extends `OpenRouterModel` and
calls `super().__init__()` at line 147, so it inherits the fix. All four affected test
files reach the check through that inheritance, not through their own code.

Deferring the check alone would trade a test problem for a worse production one: a
research run would die ten minutes in rather than at startup. So this deliverable also
adds an explicit `validate_credentials()` that MCP server entrypoints call on boot.
Production keeps failing fast; tests construct freely. **Shipping the lazy half without
the eager half is a regression, not a fix.**

Unblocks 14 tests.

### D2 — One service-gate module

`testing/_gates.py` exporting `requires_groq`, `requires_openrouter`, `requires_searxng`,
`requires_playbook`, and `requires_sec`. The two LLM gates are separate because the
providers are separately configured and separately broken: `OPENROUTER_API_KEY` is set,
while `GROQ_API_KEY` is present as a name with an empty value. A single `requires_llm`
would skip OpenRouter-backed tests that can actually run. Each skips with a reason naming exactly what is missing, and flips
to a hard failure under `NEMO_REQUIRE_SERVICES=1`.

This replaces the dead `network` alias at `testing/test_altdata_tools.py:31`, which is a
`skipif` rather than a marker and so collects zero tests under `-m network` suite-wide.

Covers 28 LLM + 10 SearXNG + 10 playbook tests.

**Known weakness, stated rather than hidden:** strict mode only helps if someone runs
it. Otherwise "skipped" decays into "deleted" without anyone noticing. This deliverable
ships a documented one-line invocation; nothing enforces it until CI exists. That gap is
accepted, not solved.

### D3 — Per-module database isolation

A `conftest.py` fixture pointing `NEMO_DB_PATH` and `NEMO_CACHE_DB_PATH` at a per-module
`tmp_path`. Both overrides already exist — the truth-source refactor added them — so
this is wiring, not new machinery.

Fixes the 4 coupling failures and the 2 failures that contaminated state was masking.

### D4 — `falsifier_alerts` belongs in the schema

Move the DDL from `daemons/falsifier_watcher.py:58` (`_ensure_alerts_table`) into
`state/schema.init_schema()`, and delete the lazy creator. Two components use this
table and only one creates it.

Also delete the fallback at `tools/sentry_server/server.py:720` that returns
`'falsifier_alerts table does not exist yet'` instead of raising. That silent fallback
is precisely why a missing table went unnoticed, and it violates the project's
fail-loud rule.

### D5 — Retire tests of deleted designs

- Delete `testing/test_phase_B2_settings_valid.py` (7 tests). It asserts `.mcp.json`
  exists and parses. That file was intentionally removed in `00abf63` and its backup
  deleted in `d12e25c`. The test guards a design that was abandoned.
- Retire the stale-premise test at `testing/test_phase_B3b_governance_xml_strip.py:62`,
  which asserts raw lxml string parsing *fails* in order to justify a workaround. It now
  passes. **Check whether the workaround it guarded is dead code and remove it if so** —
  retiring the test without checking would leave the dead branch behind.
- Convert the 5 script-style functions in `test_finnhub_tools.py`,
  `test_fred_integration.py`, and `test_multi_company_verification.py` into real tests,
  or rename them out of collection. Two of the pipelines the truth-source refactor kept
  are currently untested because of this.

### D6 — Manifest and environment

- Remove `[tool.uv] environments = ["sys_platform == 'win32'"]` (`pyproject.toml:196-199`),
  which makes `uv sync` refuse to resolve anywhere but Windows.
- Add `pytest-asyncio` as a dev dependency. Approved 2026-08-21. Test-only; without it
  the 11 async tests cannot run at all.
- Re-mark `test_sec_xbrl_functions.py:1414::test_all_500_revenue` as network rather than
  `slow`. It is a sequential ~500-ticker live-EDGAR loop, and `SEC_EMAIL` is now a real
  identity.

### D7 — Fail loud, and clean up

- `SEC_EMAIL` silently defaults to `analyst@example.com` in five places:
  `tools/web_search_server/sec_utils.py:14`, `tools/web_search_server/hf_letters.py:31`,
  `daemons/edgar_firehose.py:60`, `daemons/gdelt_poller.py:59`,
  `daemons/rss_aggregator.py:58`. Under SEC fair-access rules this misrepresents identity.
  Fail loud instead.
- README documents Alpaca env var names the code does not read.
- Fix `testing/test_db_separation.py:22`, which asserts
  `basename(CACHE_DB_PATH) == "tool_cache.db"` and therefore fails the moment anyone
  sets the `NEMO_CACHE_DB_PATH` override that same file exists to protect. Assert the
  invariant (different files), and test the default separately. This defect was
  introduced on the truth-source branch.
- **The OpenRouter model pool trusts a malformed override.** Three stacked defects,
  confirmed at runtime — every `OpenRouterModel` built without an explicit model name
  currently defaults to the literal string
  `'# optional override; if unset, pool auto-resolves'`:
  1. `.env.example:3` ships `PRIMARY_REASONING_MODEL=` followed by an inline comment.
     dotenv reads the comment as the value, so every clone inherits this. Fix the
     template, not just one machine's `.env`.
  2. `_build_reasoning_pool()` (`agent/openrouter_template.py:56-86`) puts the override
     at position 0 without checking it looks like `vendor/model`.
  3. `_verify_model_alive()` (`:24-42`) ends in `except Exception: return True`, so the
     malformed id's error was swallowed and the string was declared alive. **This is the
     load-bearing defect** — the pool's own guard is what let the garbage through, and a
     bare `except: return True` is precisely the silent fallback the project forbids.
     Distinguish a malformed-request error from a transient one (rate limit, auth)
     instead of treating every non-404 as healthy.
- Delete the untracked scratch files `debug_test.xlsx` and `simple_test.xlsx`.

## Which components actually need an LLM

Established 2026-08-21 by importing every server with all LLM keys unset, then tracing
each tool handler. This scopes what D2 gates and what the 24/7 server needs.

| Component | Needs an LLM? | Evidence |
|---|---|---|
| All 8 MCP servers | **No** | all import with every key unset; every handler traces to deterministic code |
| `rss_aggregator`, `gdelt_poller`, `news_watcher` | **Yes — Groq** | all three import `Materiality_Classifier(GroqModel)` |
| `main.py` front-end | **Yes — Groq + OpenRouter** | `WorkFlow.__init__` constructs `Bull_Agent`/`Bear_Agent`/`Arbiter_Agent`, all `GroqModel` |
| `falsifier_watcher`, `sentry_triage`, `sentry_discovery` | No | `falsifier_evaluator`, `event_scorer`, `rag.search` are deterministic |

The apparent tool-layer dependencies on the agent cluster are not real: `Risk_Officer` is
plain dataclasses, `analysis_tools.py:145` is a comment, and `tools/alpaca/server.py:206`
imports `ArbiterVerdict`, a pydantic schema.

**Consequence:** the truth-source servers need no LLM credential at all. Everything the
28 gated LLM tests cover belongs to the news daemons or to the second front-end.

**Current state:** `GROQ_API_KEY` is set to an empty value, so `main.py` and all three
news daemons raise on construction today. This is configuration, not a code defect, and
is not in scope here beyond being recorded.

## Deferred: the second front-end

`main.py` drives a LangGraph workflow over 12 `agent/*_Agent.py` modules that reference
only each other — zero references from `tools/`, `daemons/`, or `.claude/skills/`. It
reaches only 4 of the 8 MCP servers (no alpaca, sentry, or altdata), so it cannot trade,
run the sentry queue, or use the merged options tool.

It resembles dead code but is not: `main.py:9` imports it, and it is the only headless
path that runs without a Claude Code session. **Keep or retire is a product decision and
gets its own spec.** This spec gates its tests and changes nothing about it.


## Ordering

D1 and D3 land first, then **re-measure the baseline before starting anything else**.
Both change which tests fail, and the remaining deliverables are sized against numbers
that will no longer be true. Planning D5's deletions against today's counts would be
building on a figure already known to be wrong.

D2 depends on D1 (the 14 must be unblocked before the rest can be gated correctly).
D4, D5, D6, D7 are independent of each other and of everything above.

## Testing strategy

1. **The success criterion is the test.** `SKIP_NETWORK_TESTS=1` with no keys must give
   0 failed and 0 errors. `NEMO_REQUIRE_SERVICES=1` with keys and containers must give
   0 failed and 0 skipped. Both numbers go in the README.
2. **Isolation proof (D3).** `test_phase_A3`, `A4`, `06b`, and `test_phase_00_schema`
   must pass *in-suite*, not only standalone. Standalone passing is the current state
   and is not evidence.
3. **Credential proof (D1).** Construct each agent with no key and call a deterministic
   method — succeeds. Call an LLM method — raises `ValueError` with the original
   message. Call `validate_credentials()` with no key — raises at boot.
4. **Gate proof (D2).** With keys absent, the gated tests skip and the reason names the
   missing dependency. With `NEMO_REQUIRE_SERVICES=1` and keys still absent, the same
   tests **fail**. A gate that cannot fail is not a gate.
5. **Do not confuse skipped with passing.** The 28 LLM tests must be observed passing
   with real keys present before their skip is accepted. Otherwise D2 converts 28 broken
   tests into 28 invisible ones. This mirrors the sentinel-data guard from the options
   merge: agreement under degenerate conditions is not evidence.
6. **Schema proof (D4).** Against a fresh database, `init_schema()` alone must create
   `falsifier_alerts`, and the sentry tool must raise rather than return a note.
7. **Model-pool proof (D7).** Given `PRIMARY_REASONING_MODEL` set to a non-model string,
   the pool must reject it rather than seat it at position 0. Test `_verify_model_alive`
   directly against a malformed id and against a simulated rate-limit error: the first
   must return False, the second True. Asserting only the pool's final contents would
   pass even if the override were dropped for the wrong reason.

## Risks

| Risk | Mitigation |
|---|---|
| D1 defers a credential error into a long-running pipeline | `validate_credentials()` at MCP boot; the eager half is not optional |
| D2 turns 48 failures into 48 skips that nobody ever runs | Strict mode ships with a documented invocation; deliverable 5 of the testing strategy requires observing them pass at least once |
| Re-measuring after D1/D3 reveals more work than scoped | Expected. The re-measure gate exists to surface it before the rest is planned, not after |
| Deleting `test_phase_B2` loses real coverage | Verified against git history: the file it asserts was deliberately removed in `00abf63`/`d12e25c` |
| D5's lxml cleanup removes a workaround still needed on another platform | Check the workaround's call sites and lxml version pin before deleting; retire the test regardless |
| Fail-loud `SEC_EMAIL` breaks a working daemon at start-up | That is the intent — a daemon misrepresenting identity to SEC should not start. Set the variable |

## Explicitly out of scope

Owned by the primary-source swap, do not touch here: routing
`get_financial_statements` to EDGAR, the `get_forward_estimates` degradation, the
orphaned SEC extractors, and the 7 unit-scale tests in
`test_get_revenue_base_diverse_sectors` (those are that spec's D5).

Not in any spec yet: RVOL/ADV/ATR, 424B5 dilution detection, a VIX regime gate, borrow
rate, and an earnings-aware front-expiry selector. Those are net-new capability and need
their own brainstorm.

**Note for the primary-source swap:** its D4 proposes deleting `get_revenue_base` and
`get_ebitda_margin` if nothing references them. `testing/test_phase_B1_playbook_lint.py`
lists both in `REQUIRED_TOOL_NAMES`. That lint becomes gated by D2 rather than deleted,
so it survives this spec and will need updating there.

## Not addressed

CI does not exist. Every gate here is runnable but unenforced, which is why D2's
weakness is stated rather than solved. Standing up CI is the natural successor to this
spec and is deliberately not folded into it.
