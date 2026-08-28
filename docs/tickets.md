# Tickets

Open issues in ticket form, newest batch first. `docs/known_issues.md` is the
long-form record of investigations; this file is the actionable queue.

**Status** — `OPEN`, `IN PROGRESS`, `DONE`, `WONTFIX`
**Severity** — `S1` breaks or leaks something, `S2` wrong output, `S3` degraded
signal or coverage, `S4` cosmetic

---

## NEMO-1 · Live OpenRouter key is printed in pytest tracebacks

**Severity** S1 · **Status** OPEN · **Component** `agent/openrouter_template.py`

`_verify_model_alive(model_id, api_key, timeout=10.0)` takes the credential as
a positional parameter. pytest renders function locals in a failure traceback,
so any failing run of `testing/test_bugfix_01_model_resolution.py` writes the
whole key to stdout.

**Repro** — `.venv/bin/python -m pytest testing/test_bugfix_01_model_resolution.py -q`
with any `OPENROUTER_API_KEY` set. The key appears in the traceback header as
`api_key = '…'`. Observed 2026-08-28.

**Impact** — CI logs, a pasted terminal, or a captured artefact carries a live
credential. This is the only entry here that leaks a secret.

**Fix** — stop passing the key as a parameter: read it from the environment
inside the function, or wrap it in a type whose `__repr__` redacts. A
`# pytest: no-locals` style suppression is not enough; the value should not be
in a frame that can be rendered.

**Done when** — a deliberately failing liveness probe emits no key material.

---

## NEMO-2 · Policy-signals test cannot separate an empty provider from the regression it guards

**Severity** S3 · **Status** OPEN · **Component** `testing/test_altdata_tools.py`, `tools/altdata_server/server.py`

`test_policy_signals_returns_required_fields` asserts `bill_count > 0` to catch
the old dedup bug where `obj.get("id")` was always `None` and every bill was
silently dropped. GovTrack answered `success: True` with zero bills during a
42-minute full run, and the test read that as the regression. It passes when
run alone.

**Repro** — intermittent; observed once in a full-suite run on 2026-08-28,
not reproducible in isolation.

**Impact** — a flaky failure that points at a dedup bug that is not there. The
risk of "fixing" it wrongly is real: relaxing the assertion removes a guard
that caught an actual bug.

**Fix** — key the assertion on `degraded` / `partial_errors`, which the
response already carries, so a throttled or empty provider reports as such
instead of as a dedup regression. Do not simply drop the `> 0` check.

**Done when** — an empty GovTrack response and a dedup regression produce
different, correctly-named failures.

---

## NEMO-3 · `debt_maturity` covers 57% of the sweep basket

**Severity** S3 · **Status** OPEN · **Component** `tools/web_search_server/debt_maturity.py`

Coverage sweep on the 35-name basket: 14 full, 6 partial, 15 not covered.
Neighbouring tools sit at 82–94%.

**Repro** — `.venv/bin/python -m pytest testing/test_research_coverage_sweep.py -q -s`
and read the printed table.

**Impact** — unknown whether this is a filer-tagging limit (fine, and worth
documenting on the tool) or an extractor gap (worth fixing). Right now a
caller cannot tell either.

**Fix** — sample the 15 not-covered filers and classify. If tagging, surface it
as a documented limit on the response; if extraction, fix the concept chain.

---

## NEMO-4 · `OPENROUTER_API_KEY` in `.env` is malformed

**Severity** S2 · **Status** OPEN · **Component** configuration, not code

Stored as `v1-…`. An OpenRouter key begins `sk-or-v1-`, so the provider answers
`401 Missing Authentication header`. The `sk-or-` prefix looks to have been
lost when the value was pasted.

**Fails** — `test_resolved_model_is_alive`, `test_old_dead_model_still_dead`,
`test_probe_extracts_explicit_ticker_symbol`.

**Note** — these three previously **skipped**. `.env` was not loaded for tests,
so the key read as absent and the gate skipped rather than ran. Loading it at
session start (commit `abafb39`) turned three silent skips into three honest
failures. The tests did not regress; they started running.

`OPENROUTER_NEMOTRON`, `OPENROUTER_GLM` and `GROQ_API_KEY` are empty — check
whether that is deliberate.

**Fix** — correct the key in `.env`. No code change.

---

## NEMO-5 · No local RAG corpus, so five tests cannot pass

**Severity** S4 · **Status** OPEN · **Component** environment

`test_rag_search` (4 tests) returns `top_ids=[]`;
`test_rag_ingest::test_bootstrap_smoke` finds 23 chunks against a `>100` bar.
Reproduces identically on a pristine worktree at HEAD, so this is not a
regression.

**Impact** — five permanently-red tests train the eye to ignore red. The RAG
stack is not in the image either, and `rag_search`/`rag_ingest` are
capability-gated out of it.

**Fix** — either ingest a corpus locally, or gate these behind a marker that
skips when no index is present, in the same way the network tests are gated.
Skipping is only acceptable with `STRICT_GATES=1` still able to force them.

---

## NEMO-6 · DKS TTM revenue disagrees between vendors by 10.1%

**Severity** S3 · **Status** OPEN · **Component** data, not code

`market_data` 21.145bn vs `finnhub_metric` 19.206bn TTM revenue (+10.1%),
above the reconciliation tolerance in
`test_cross_source_reconciliation::test_ttm_revenue_agrees_between_vendors`.

**Impact** — a real disagreement between two providers on one name. The
tolerance was deliberately not widened to make the suite green, because the
test is doing its job.

**Fix** — determine which vendor is stale (DKS has had corporate activity that
would move TTM revenue and the two vendors update on different cadences), and
either record it as an adjudicated divergence in `known_issues.md` or add a
staleness note to the tool that is behind.

---

## NEMO-7 · `paper_order.sue` does not record which variant produced it

**Severity** S2 · **Status** OPEN · **Component** `research/pit_store.py:151`, `research/scanner.py:76`

`SIGNAL_VARIANT` selects between `ts` and `af` (a sigma) and `cs` (a percentile
rank). The scan summary reports `variant` (`scanner.py:569`), but the
`paper_order` row does not, and no table in the store persists it — `grep -n
variant research/pit_store.py` returns nothing, and `run_log` carries only
job/dates/status.

**Trigger** — change `SIGNAL_VARIANT` between two runs, which is a one-line
edit and the intended way to switch variants.

**Impact** — the book silently mixes two incomparable quantities under one
`sue` column, and `research-score` scores them together. The codebase already
states this rule for the coefficients ("a sigma and a rank are different
quantities and pricing them with one number would make them look comparable");
the order record does not enforce it. Any calibration drawn from a book
spanning a variant change is drawn from a mixture.

**Fix** — add a `variant` column to `paper_order` via `_MIGRATIONS` and write
`row.get("variant")` in `record_paper_orders`. Have `research-score` group by
it, or refuse to score a mixed window.

**Done when** — a book containing two variants either scores them separately or
refuses, and a `paper_order` row alone says which quantity its `sue` is.

---

## NEMO-8 · Seeded-consensus provenance stops before the order record

**Severity** S3 · **Status** OPEN · **Component** `research/sue.py:1371`, `research/scanner.py`

`sue.py` computes `seeded_quarters` and `recorded_quarters` — how much of a
sigma rests on quarters reconstructed by `research-seed` rather than observed —
and the reasoning for why that matters is written out at `sue.py:1282`. Nothing
in `scanner.py` reads either field, and neither reaches `paper_order`.

**Trigger** — latent until the analyst leg activates. `sue_af` needs eight
quarters of recorded consensus and currently refuses, so no order today rests
on seeded data. It will when the recorder has accrued enough.

**Impact** — an order whose signal rests on four reconstructions and two
observations will be indistinguishable from one resting on six observations,
in exactly the record built to answer that question later.

**Fix** — carry `seeded_quarters`/`recorded_quarters` through the scanner onto
the order row, alongside NEMO-7's `variant`. Both are the same shape of fix and
should land together.

**Done when** — a filed order states how much of its signal was reconstructed.

---

## NEMO-9 · Half-day closes are classified as during-market-hours

**Severity** S3 · **Status** OPEN · **Component** `research/announcements.py:39`

`MARKET_CLOSE = time(16, 0)` is fixed. US equities close at 13:00 ET on roughly
three sessions a year — the day after Thanksgiving, Christmas Eve on a weekday,
and July 3rd in some years. `grep -rn "half.day\|early_close" research/` returns
nothing, so no code path knows about them.

**Trigger** — an Item 2.02 filing accepted between 13:00 and 16:00 ET on a
half-day session.

**Impact** — `classify()` returns `dmh` for a release that was genuinely after
the close. `timing` decides which session an order may enter, and
`scoring.py` splits its results `by_timing`, so the release lands in the wrong
bucket on both sides. The DST reasoning in that same docstring is careful and
correct; this is the other calendar dependency, and it is not handled.

**Note** — narrow. It needs a half day *and* a filer releasing in that window,
so it may never have fired. Worth pinning before the analyst leg starts scoring
`by_timing` seriously.

**Fix** — take the close from an exchange calendar rather than a constant, or
carry a small table of half-day dates and use 13:00 on those. Refusing to
classify (`unknown`) on a half day would also be honest and is cheaper.

**Done when** — a 14:30 ET release on the day after Thanksgiving classifies
`amc`, and a test pins it.
