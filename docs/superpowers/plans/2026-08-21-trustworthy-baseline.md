# Trustworthy Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the offline test suite report zero failures and zero errors, so that any
red means a defect in the code rather than a fact about the machine.

**Architecture:** Fix the two things that make today's numbers untrustworthy first —
database paths that bind at import time (so tests cannot be isolated) and credential
checks that fire in constructors (so pure-logic tests need keys they never use) — then
re-measure before sizing the rest. Everything that genuinely needs a live service gets
an explicit gate that skips with a reason and can be flipped to a hard failure.

**Tech Stack:** Python 3.12, pytest, SQLite, `python-dotenv`, `openai` SDK pointed at
Groq and OpenRouter.

## Global Constraints

- Run every command with the project venv: `.venv/bin/python`, never bare `python3`.
- Offline runs set `SKIP_NETWORK_TESTS=1`. Never remove that gate to make a test pass.
- Mock data belongs only in tests, never in a production code path.
- No silent fallbacks. An error that cannot be handled must propagate.
- Ask before adding a dependency. `pytest-asyncio` is the only one pre-approved here.
- Never commit or echo the contents of `.env`.
- Commit messages must not mention Claude or any AI tool.
- Target: `SKIP_NETWORK_TESTS=1 .venv/bin/python -m pytest testing/ -q` reports
  **0 failed, 0 errors**.

---

### Task 1: Resolve database paths at call time

Today `state/schema.py:385` reads `def get_connection(db_path: str = DB_PATH)`. A default
argument is evaluated once, when the function is defined, so `DB_PATH` is frozen at import.
Setting `NEMO_DB_PATH` afterwards has no effect. This is why per-test database isolation
is impossible and why the suite's contamination cannot currently be fixed from the test
side. The override works only for a process launched with the variable already set.

**Files:**
- Modify: `state/schema.py:18-22` (constant), `state/schema.py:385` (`get_connection`), `state/schema.py:414` (`init_schema`)
- Modify: `agent/cache.py:11-15` (constant) and its connection helper
- Test: `testing/test_db_path_resolution.py` (create)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `state.schema.current_db_path() -> str` and `agent.cache.current_cache_db_path() -> str`.
  Both read the environment on every call. `get_connection(db_path: str | None = None)` and
  `init_schema(db_path: str | None = None)` resolve `None` through those helpers.
  Task 2 depends on this behaviour.

- [ ] **Step 1: Write the failing test**

Create `testing/test_db_path_resolution.py`:

```python
"""The NEMO_DB_PATH / NEMO_CACHE_DB_PATH overrides must take effect when they
are set, not only when they happen to be set before this module is imported.

A default argument binds once at function-definition time, so
`def get_connection(db_path=DB_PATH)` freezes the path at import and silently
ignores any later override. Per-module test isolation depends on this working.
"""
import os
import importlib


def test_state_path_follows_env_after_import(monkeypatch, tmp_path):
    from state import schema
    target = tmp_path / "redirected_session.db"
    monkeypatch.setenv("NEMO_DB_PATH", str(target))
    assert schema.current_db_path() == str(target)


def test_cache_path_follows_env_after_import(monkeypatch, tmp_path):
    from agent import cache
    target = tmp_path / "redirected_cache.db"
    monkeypatch.setenv("NEMO_CACHE_DB_PATH", str(target))
    assert cache.current_cache_db_path() == str(target)


def test_get_connection_writes_to_the_overridden_path(monkeypatch, tmp_path):
    from state import schema
    target = tmp_path / "written.db"
    monkeypatch.setenv("NEMO_DB_PATH", str(target))
    conn = schema.get_connection()
    try:
        conn.execute("CREATE TABLE probe(x INTEGER)")
        conn.commit()
    finally:
        conn.close()
    assert target.exists(), "get_connection ignored NEMO_DB_PATH set after import"


def test_defaults_are_absolute_when_env_is_unset(monkeypatch):
    monkeypatch.delenv("NEMO_DB_PATH", raising=False)
    monkeypatch.delenv("NEMO_CACHE_DB_PATH", raising=False)
    from state import schema
    from agent import cache
    assert os.path.isabs(schema.current_db_path())
    assert os.path.isabs(cache.current_cache_db_path())
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `SKIP_NETWORK_TESTS=1 .venv/bin/python -m pytest testing/test_db_path_resolution.py -v`

Expected: FAIL with `AttributeError: module 'state.schema' has no attribute 'current_db_path'`.

- [ ] **Step 3: Add call-time resolution to `state/schema.py`**

Replace the `DB_PATH` constant block (around line 18-22) with:

```python
_DEFAULT_DB_PATH = os.path.join(_REPO_ROOT, "db_cache", "session.db")


def current_db_path() -> str:
    """Resolve the state database path on every call.

    Read per call rather than bound at import so a test fixture or an
    embedding process can redirect the database after this module loads.
    """
    return os.environ.get("NEMO_DB_PATH", _DEFAULT_DB_PATH)


# Convenience alias for readers that want the path at import time. Callers that
# need to honour a later override must call current_db_path() instead.
DB_PATH = current_db_path()
```

Change the two function signatures:

```python
def get_connection(db_path: str | None = None) -> sqlite3.Connection:
```

and immediately inside its body, before `os.makedirs(...)`:

```python
    if db_path is None:
        db_path = current_db_path()
```

```python
def init_schema(db_path: str | None = None) -> None:
```

and at the top of its body:

```python
    if db_path is None:
        db_path = current_db_path()
```

- [ ] **Step 4: Add call-time resolution to `agent/cache.py`**

Apply the same shape. Replace the `CACHE_DB_PATH` constant block with:

```python
_DEFAULT_CACHE_DB_PATH = os.path.join(_REPO_ROOT, "db_cache", "tool_cache.db")


def current_cache_db_path() -> str:
    """Resolve the tool-cache database path on every call. See
    state.schema.current_db_path for why this is not a bound default."""
    return os.environ.get("NEMO_CACHE_DB_PATH", _DEFAULT_CACHE_DB_PATH)


CACHE_DB_PATH = current_cache_db_path()
```

Then find every place in `agent/cache.py` that opens a connection and make it call
`current_cache_db_path()` instead of reading the module constant. Confirm none remain:

Run: `grep -n "CACHE_DB_PATH" agent/cache.py`

Every hit must be either the constant definition or inside `current_cache_db_path`.

- [ ] **Step 5: Run the new test and verify it passes**

Run: `SKIP_NETWORK_TESTS=1 .venv/bin/python -m pytest testing/test_db_path_resolution.py -v`

Expected: PASS, 4 passed.

- [ ] **Step 6: Verify no caller broke**

Every existing caller uses the no-argument form, so behaviour must be unchanged.

Run: `SKIP_NETWORK_TESTS=1 .venv/bin/python -m pytest testing/test_db_separation.py testing/test_phase_00_schema.py -v`

Expected: `test_db_separation.py::test_cache_and_state_use_different_files` may still
fail — it is over-specified and gets fixed in Task 10. Every other test passes.

- [ ] **Step 7: Commit**

```bash
git add state/schema.py agent/cache.py testing/test_db_path_resolution.py
git commit -m "resolve database paths at call time

A default argument binds once at function definition, so get_connection's
db_path default froze NEMO_DB_PATH at import and ignored any later override.
Per-module test isolation is impossible until the path is read per call."
```

---

### Task 2: Isolate the database per test module

With Task 1 in place, each test module can be given its own database. Measured evidence
that this is needed: `test_phase_A3`, `test_phase_A4`, and `test_phase_06b` contribute 4
failures in-suite but pass 29/29 in isolation, while `test_falsifier_watcher_e2e` fails 6
in-suite and 8 in isolation. The contamination both invents and masks failures.

**Files:**
- Create: `testing/conftest.py`
- Test: the existing suite is the test; no new test file.

**Interfaces:**
- Consumes: `state.schema.current_db_path`, `agent.cache.current_cache_db_path` from Task 1.
- Produces: an autouse module-scoped fixture. No importable symbols.

- [ ] **Step 1: Confirm the failures exist in-suite and not in isolation**

Run: `SKIP_NETWORK_TESTS=1 .venv/bin/python -m pytest testing/test_phase_A3_get_positions.py testing/test_phase_A4_risk_check.py testing/test_phase_06b_positions_orders.py -q -p no:randomly`

Expected: 29 passed. This is the "in isolation" baseline that proves the code is fine.

- [ ] **Step 2: Create the isolation fixture**

Create `testing/conftest.py`:

```python
"""Per-module database isolation.

Tests write positions, theses, orders, and alert rows into one SQLite file.
Later modules then observe earlier modules' rows. That invents failures -- a
reconcile test sees another module's positions and reports a discrepancy --
and it masks them, because a schema test can pass on tables a previous module
happened to create. Measured: three position/risk modules contribute 4
failures in-suite and pass 29/29 alone.

Module scope rather than function scope: many of these tests build state
across several test functions in the same file and would break under
per-function isolation.
"""
import os

import pytest


@pytest.fixture(autouse=True, scope="module")
def isolated_databases(tmp_path_factory, request):
    module_name = request.module.__name__.rsplit(".", 1)[-1]
    directory = tmp_path_factory.mktemp(f"db_{module_name}")

    previous = {}
    for variable, filename in (
        ("NEMO_DB_PATH", "session.db"),
        ("NEMO_CACHE_DB_PATH", "tool_cache.db"),
    ):
        previous[variable] = os.environ.get(variable)
        os.environ[variable] = str(directory / filename)

    yield

    for variable, value in previous.items():
        if value is None:
            os.environ.pop(variable, None)
        else:
            os.environ[variable] = value
```

- [ ] **Step 3: Verify the position and risk tests now pass in-suite**

Run: `SKIP_NETWORK_TESTS=1 .venv/bin/python -m pytest testing/ -q -p no:randomly 2>&1 | grep -E "test_phase_A3|test_phase_A4|test_phase_06b|test_phase_00_schema"`

Expected: no output. These modules produce no `FAILED` lines when the whole suite runs.

- [ ] **Step 4: Verify the falsifier failures are now visible rather than masked**

Run: `SKIP_NETWORK_TESTS=1 .venv/bin/python -m pytest testing/test_falsifier_watcher_e2e.py -q -p no:randomly 2>&1 | tail -3`

Expected: 8 failed. This is correct — the missing `falsifier_alerts` table is a real
defect and Task 6 fixes it. Seeing 8 rather than 6 confirms the masking is gone.

- [ ] **Step 5: Commit**

```bash
git add testing/conftest.py
git commit -m "isolate the database per test module

Tests shared one SQLite file, so later modules saw earlier modules' rows.
That invented 4 failures in the position and risk suites and masked 2 real
failures in the falsifier suite."
```

---

### Task 3: Resolve credentials lazily, and check them explicitly at boot

`agent/groq_template.py:39` and `agent/openrouter_template.py:149` raise in `__init__`.
Fourteen tests construct `Financial_Modeling_Agent` — which inherits that constructor via
`OpenRouterModel` — purely to reach deterministic methods: LBO spread maths, regime
weights, credit guards, prompt strings. Measured: those four files go from 14 failures to
17 passed in 1.47 seconds with zero network when a fake key is present.

Deferring the check alone would be a regression, trading a test problem for a worse
production one: a long research run would die partway instead of at startup. So this task
also adds an explicit `validate_credentials()` for server entrypoints to call on boot.
**Shipping the lazy half without the eager half is not acceptable.**

**Files:**
- Modify: `agent/groq_template.py:35-46`
- Modify: `agent/openrouter_template.py:139-165`
- Test: `testing/test_lazy_credentials.py` (create)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: on both `GroqModel` and `OpenRouterModel` —
  `validate_credentials() -> None` (raises `ValueError` when the key is missing),
  a `client` property that builds and caches an `OpenAI` client on first access,
  and `_resolve_api_key() -> str`. `OpenRouterModel` also exposes a `fallback_client`
  property. Every existing `self.client.chat.completions.create(...)` call site keeps
  working unchanged.

- [ ] **Step 1: Write the failing test**

Create `testing/test_lazy_credentials.py`:

```python
"""Constructing a model must not require a credential.

Fourteen tests instantiate Financial_Modeling_Agent only to reach deterministic
methods. Validating the key in __init__ held those tests hostage to a
credential their code paths never use. The key is still required -- it is
checked on first real use, and explicitly at boot via validate_credentials().

load_dotenv is patched out in every test because it would otherwise repopulate
the environment from the developer's .env and defeat the point.
"""
import pytest


def _no_dotenv(monkeypatch, module):
    monkeypatch.setattr(module, "load_dotenv", lambda *a, **k: None)


def test_groq_constructs_without_a_key(monkeypatch):
    import agent.groq_template as gt
    _no_dotenv(monkeypatch, gt)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    model = gt.GroqModel()
    assert model.model_name


def test_groq_client_access_raises_without_a_key(monkeypatch):
    import agent.groq_template as gt
    _no_dotenv(monkeypatch, gt)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    model = gt.GroqModel()
    with pytest.raises(ValueError, match="GROQ_API_KEY not found"):
        _ = model.client


def test_groq_validate_credentials_raises_without_a_key(monkeypatch):
    import agent.groq_template as gt
    _no_dotenv(monkeypatch, gt)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GROQ_API_KEY not found"):
        gt.GroqModel().validate_credentials()


def test_groq_client_is_built_once(monkeypatch):
    import agent.groq_template as gt
    _no_dotenv(monkeypatch, gt)
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")
    model = gt.GroqModel()
    assert model.client is model.client


def test_openrouter_constructs_without_a_key(monkeypatch):
    import agent.openrouter_template as ot
    _no_dotenv(monkeypatch, ot)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    model = ot.OpenRouterModel(model_name="vendor/model:free")
    assert model.model_name == "vendor/model:free"


def test_openrouter_client_access_raises_without_a_key(monkeypatch):
    import agent.openrouter_template as ot
    _no_dotenv(monkeypatch, ot)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_NEMOTRON", raising=False)
    model = ot.OpenRouterModel(model_name="vendor/model:free")
    with pytest.raises(ValueError, match="No API key found"):
        _ = model.client


def test_openrouter_validate_credentials_raises_without_a_key(monkeypatch):
    import agent.openrouter_template as ot
    _no_dotenv(monkeypatch, ot)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="No API key found"):
        ot.OpenRouterModel(model_name="vendor/model:free").validate_credentials()


def test_financial_modeling_agent_constructs_without_a_key(monkeypatch):
    """The concrete reason this task exists: 14 pure-logic tests build this."""
    import agent.openrouter_template as ot
    _no_dotenv(monkeypatch, ot)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    from agent.Financial_Modeling_Agent import Financial_Modeling_Agent
    assert Financial_Modeling_Agent() is not None
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `SKIP_NETWORK_TESTS=1 .venv/bin/python -m pytest testing/test_lazy_credentials.py -v`

Expected: FAIL. `test_groq_constructs_without_a_key` raises
`ValueError: GROQ_API_KEY not found in environment.`

- [ ] **Step 3: Make `GroqModel` lazy**

In `agent/groq_template.py`, replace the body of `__init__` and add three members:

```python
  def __init__(self, model_name: str = 'llama-3.3-70b-versatile', api_key_env: str = "GROQ_API_KEY"):
    load_dotenv()
    self._api_key_env = api_key_env
    self._client = None
    self.model_name = model_name
    self.conversatoin_history = []  # Typo kept for codebase consistency

  def _resolve_api_key(self) -> str:
    api_key = os.getenv(self._api_key_env)
    if not api_key:
      raise ValueError(f"{self._api_key_env} not found in environment. Add it to your .env file.")
    return api_key

  def validate_credentials(self) -> None:
    """Fail fast at process start. Server entrypoints call this on boot so a
    missing key surfaces immediately rather than partway through a run."""
    self._resolve_api_key()

  @property
  def client(self) -> OpenAI:
    if self._client is None:
      self._client = OpenAI(
        api_key=self._resolve_api_key(),
        base_url="https://api.groq.com/openai/v1",
        timeout=self.CLIENT_TIMEOUT
      )
    return self._client
```

- [ ] **Step 4: Make `OpenRouterModel` lazy**

In `agent/openrouter_template.py`, replace the body of `__init__` and add four members:

```python
  def __init__(self, model_name: str = None, api_key_env: str = "OPENROUTER_API_KEY"):
    load_dotenv()
    if model_name is None:
      model_name = PRIMARY_REASONING_MODEL
    self._api_key_env = api_key_env
    self._client = None
    self._fallback_client = None
    self.model_name = model_name
    self.conversatoin_history = []

  def _resolve_api_key(self) -> str:
    # Try the requested env var first, then fall back to the main key. A single
    # OPENROUTER_API_KEY is always enough; model-specific keys are optional extras.
    api_key = os.getenv(self._api_key_env) or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
      raise ValueError(f"No API key found. Set OPENROUTER_API_KEY (or {self._api_key_env}) in your .env file.")
    return api_key

  def validate_credentials(self) -> None:
    """Fail fast at process start. See GroqModel.validate_credentials."""
    self._resolve_api_key()

  @property
  def client(self) -> OpenAI:
    if self._client is None:
      self._client = OpenAI(
        api_key=self._resolve_api_key(),
        base_url="https://openrouter.ai/api/v1",
        timeout=self.CLIENT_TIMEOUT
      )
    return self._client

  @property
  def fallback_client(self) -> OpenAI:
    # Prefer OPENROUTER_GLM, otherwise reuse the main key. Reusing it is fine --
    # the fallback only triggers when the primary model fails.
    if self._fallback_client is None:
      fallback_key = os.getenv("OPENROUTER_GLM") or self._resolve_api_key()
      self._fallback_client = OpenAI(
        api_key=fallback_key,
        base_url="https://openrouter.ai/api/v1",
        timeout=self.CLIENT_TIMEOUT
      )
    return self._fallback_client
```

- [ ] **Step 5: Run the new test and verify it passes**

Run: `SKIP_NETWORK_TESTS=1 .venv/bin/python -m pytest testing/test_lazy_credentials.py -v`

Expected: PASS, 8 passed.

- [ ] **Step 6: Verify the 14 hostage tests now run without a key**

Run: `SKIP_NETWORK_TESTS=1 env -u GROQ_API_KEY -u OPENROUTER_API_KEY .venv/bin/python -m pytest testing/test_fix_14_regime_weights.py testing/test_fix_01_lbo_hy_spread.py testing/test_bugfix_02_credit_guard.py testing/test_fix_04_beat_rate_signals.py -q -p no:randomly 2>&1 | tail -3`

Expected: 17 passed. If any test now fails on a *network* call rather than a key, it was
never pure logic — report that rather than working around it.

- [ ] **Step 7: Wire the eager check into the server entrypoints**

Only servers that actually construct an LLM model need this. Find them:

Run: `grep -rn "GroqModel(\|OpenRouterModel(\|Materiality_Classifier(" --include="*.py" daemons/ tools/`

For each daemon that constructs one at startup, call `validate_credentials()` on the
instance immediately after construction, before entering its main loop. Do not add it to
any of the 8 MCP servers — none of them constructs an LLM model, verified by importing
all 8 with every key unset.

- [ ] **Step 8: Commit**

```bash
git add agent/groq_template.py agent/openrouter_template.py testing/test_lazy_credentials.py daemons/
git commit -m "resolve LLM credentials lazily, check them explicitly at boot

Validating the key in __init__ meant 14 pure-logic tests -- LBO spread maths,
regime weights, credit guards, prompt strings -- could not run without a
credential their code paths never use. The client is now built on first use.

validate_credentials() keeps the fail-fast behaviour where it matters: daemon
entrypoints call it on boot, so a missing key still surfaces at startup rather
than partway through a run."
```

---

### Task 4: Re-measure the baseline

The spec requires this gate. Tasks 1-3 change which tests fail, and every later task is
sized against numbers that are now stale. Do not skip this and do not estimate — measure.

**Files:**
- Modify: `docs/known_issues.md`

**Interfaces:**
- Consumes: the state of the tree after Task 3.
- Produces: a recorded baseline that later tasks are checked against.

- [ ] **Step 1: Run the full offline suite and capture the result**

Run: `SKIP_NETWORK_TESTS=1 .venv/bin/python -m pytest testing/ -q --no-header --tb=no -p no:randomly > /tmp/baseline_after_d1_d3.txt 2>&1; tail -3 /tmp/baseline_after_d1_d3.txt`

- [ ] **Step 2: Break the remaining failures down by cause**

Run: `grep -E "^FAILED " /tmp/baseline_after_d1_d3.txt | sed 's/^FAILED //; s/::.*//' | sort | uniq -c | sort -rn`

- [ ] **Step 3: Record the new baseline in `docs/known_issues.md`**

Add a dated section stating the before and after totals, the per-file breakdown from
Step 2, and — importantly — any failure whose cause is **not** already one of the classes
in the spec's evidence table. A new class means the spec's taxonomy was incomplete;
report it rather than absorbing it silently.

- [ ] **Step 4: Commit**

```bash
git add docs/known_issues.md
git commit -m "record the post-isolation baseline

Tasks 1-3 change which tests fail, so the remaining work is sized against
this measurement rather than the pre-isolation numbers."
```

---

### Task 5: Gate the tests that need a live service

**Files:**
- Create: `testing/_gates.py`
- Modify: `testing/test_altdata_tools.py:31` (remove the dead `network` alias)
- Modify: the LLM, scraper, and playbook test modules identified in Task 4
- Test: `testing/test_gates.py` (create)

**Interfaces:**
- Consumes: the Task 4 baseline, which lists exactly which modules still fail and why.
- Produces: `testing/_gates.py` exporting `requires_groq`, `requires_openrouter`,
  `requires_searxng`, `requires_playbook`, `requires_sec` — each a `pytest.mark`
  decorator usable as a module-level `pytestmark` — plus
  `service_missing(name: str) -> str | None` returning a human-readable reason or `None`.

- [ ] **Step 1: Write the failing test**

Create `testing/test_gates.py`:

```python
"""A gate that cannot fail is not a gate.

Skipping a test when its dependency is absent is only honest if the skip can
be turned back into a failure on demand. Otherwise "skipped" silently decays
into "deleted" and nobody notices that 38 tests stopped running.
"""
import os
import pathlib
import subprocess
import sys
import tempfile
import textwrap


def test_missing_groq_key_is_reported():
    from testing import _gates
    os.environ.pop("GROQ_API_KEY", None)
    assert _gates.service_missing("groq") is not None


def test_present_groq_key_is_not_reported(monkeypatch):
    from testing import _gates
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")
    assert _gates.service_missing("groq") is None


def test_empty_key_counts_as_missing(monkeypatch):
    """GROQ_API_KEY is present in .env with an empty value. An empty string is
    a missing credential, not a configured one."""
    from testing import _gates
    monkeypatch.setenv("GROQ_API_KEY", "")
    assert _gates.service_missing("groq") is not None


def _run_probe(env_extra):
    """Run a one-test probe in a subprocess with a controlled environment.

    A subprocess is required because _gates.STRICT is read at import time, so
    NEMO_REQUIRE_SERVICES cannot be toggled within a running session.

    The probe file is written into testing/ rather than a tempdir on purpose:
    testing/conftest.py owns the strict-mode hook, and a file outside that
    directory would not pick the conftest up. The gate would then silently
    never fire and this test would pass for the wrong reason.
    """
    probe = textwrap.dedent("""
        import pytest
        from testing._gates import requires_groq

        pytestmark = requires_groq

        def test_needs_groq():
            assert True
    """)
    testing_dir = pathlib.Path(__file__).resolve().parent
    handle, name = tempfile.mkstemp(
        prefix="test_probe_gate_", suffix=".py", dir=testing_dir)
    os.close(handle)
    path = pathlib.Path(name)
    path.write_text(probe)
    try:
        env = {**os.environ, **env_extra}
        env.pop("GROQ_API_KEY", None)
        env["PYTHONPATH"] = str(testing_dir.parent)
        env["SKIP_NETWORK_TESTS"] = "1"
        return subprocess.run(
            [sys.executable, "-m", "pytest", str(path),
             "-q", "--no-header", "-p", "no:randomly"],
            capture_output=True, text=True, env=env,
        )
    finally:
        path.unlink(missing_ok=True)


def test_gate_skips_when_service_absent():
    result = _run_probe({})
    assert "1 skipped" in result.stdout, result.stdout


def test_gate_fails_when_strict_mode_is_on():
    result = _run_probe({"NEMO_REQUIRE_SERVICES": "1"})
    assert "1 failed" in result.stdout, result.stdout
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `SKIP_NETWORK_TESTS=1 .venv/bin/python -m pytest testing/test_gates.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'testing._gates'`.

- [ ] **Step 3: Implement the gate module**

Create `testing/_gates.py`:

```python
"""Service gates for tests that need something this machine may not have.

Two rules:

1. A skip must name what is missing. "skipped" with no reason is how a broken
   test hides.
2. Every skip must be convertible into a failure. Set NEMO_REQUIRE_SERVICES=1
   and a gated test fails instead of skipping, so a run with full credentials
   proves the gated tests still work rather than merely still exist.

The LLM gates are split by provider because the providers are separately
configured and separately broken: OPENROUTER_API_KEY is set on this machine
while GROQ_API_KEY is present with an empty value.
"""
import os
import socket
from urllib.parse import urlparse

import pytest

STRICT = os.environ.get("NEMO_REQUIRE_SERVICES") == "1"


def _env_key(name: str) -> bool:
    # An empty value is a missing credential, not a configured one.
    return bool(os.environ.get(name, "").strip())


def _searxng_reachable() -> bool:
    url = os.environ.get("SEARXNG_URL", "http://localhost:8888")
    parsed = urlparse(url)
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


def _playbook_present() -> bool:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.exists(os.path.join(root, "CLAUDE.md"))


def service_missing(name: str) -> str | None:
    """Return a human-readable reason the service is unavailable, or None."""
    if name == "groq":
        return None if _env_key("GROQ_API_KEY") else (
            "GROQ_API_KEY is unset or empty; set it in .env to run this test")
    if name == "openrouter":
        return None if _env_key("OPENROUTER_API_KEY") else (
            "OPENROUTER_API_KEY is unset or empty; set it in .env to run this test")
    if name == "searxng":
        return None if _searxng_reachable() else (
            "SearXNG is not reachable at SEARXNG_URL (default http://localhost:8888); "
            "start it with: docker compose up -d searxng")
    if name == "playbook":
        return None if _playbook_present() else (
            "CLAUDE.md is absent. It is gitignored, so this lint only runs on a "
            "machine that has the local playbook")
    if name == "sec":
        return None if _env_key("SEC_EMAIL") else (
            "SEC_EMAIL is unset; SEC fair access requires a real contact identity")
    raise ValueError(f"unknown service gate: {name!r}")


def _gate(name: str):
    reason = service_missing(name)
    if reason is None:
        return pytest.mark.skipif(False, reason="")
    if STRICT:
        # A gate that cannot fail is not a gate. The marker is consumed by the
        # pytest_pyfunc_call hook in conftest.py, which fails during the call
        # phase. Failing in setup instead would report an error, not a failure.
        return pytest.mark.fail_missing_service(name=name, reason=reason)
    return pytest.mark.skipif(True, reason=reason)


requires_groq = _gate("groq")
requires_openrouter = _gate("openrouter")
requires_searxng = _gate("searxng")
requires_playbook = _gate("playbook")
requires_sec = _gate("sec")
```

Add the strict-mode hook to `testing/conftest.py` (created in Task 2), appended below the
existing fixture:

```python
def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "fail_missing_service(name, reason): gated test that must fail, not "
        "skip, when NEMO_REQUIRE_SERVICES=1 and the service is unavailable",
    )


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem):
    """Under NEMO_REQUIRE_SERVICES=1, a gated test fails instead of skipping.

    This fires in the call phase deliberately. The obvious alternative --
    pytest.fail() inside pytest_runtest_setup -- reports the test as an ERROR
    rather than a FAILURE, which muddies the "0 failed, 0 errors" target.
    Verified: setup gives `1 error`, this gives `1 failed`.

    tryfirst so it runs ahead of pytest-asyncio's own pytest_pyfunc_call
    implementation and therefore gates async tests too.
    """
    marker = pyfuncitem.get_closest_marker("fail_missing_service")
    if marker is not None:
        pytest.fail(
            f"NEMO_REQUIRE_SERVICES=1 but {marker.kwargs['name']} is "
            f"unavailable: {marker.kwargs['reason']}",
            pytrace=False,
        )
```

- [ ] **Step 4: Run the gate test and verify it passes**

Run: `SKIP_NETWORK_TESTS=1 .venv/bin/python -m pytest testing/test_gates.py -v`

Expected: PASS, 5 passed.

- [ ] **Step 5: Apply the gates to the modules the Task 4 baseline named**

For each module still failing on a missing service, add a module-level `pytestmark`
immediately after its imports. Groq-backed modules — those whose failure text is
`GROQ_API_KEY not found` — get `requires_groq`. OpenRouter-backed modules get
`requires_openrouter`. The `test_scraper_*` modules get `requires_searxng`.
`test_phase_B1_playbook_lint.py` and `test_phase_C1_brief_dry_run.py` get
`requires_playbook`.

```python
from testing._gates import requires_groq

pytestmark = requires_groq
```

Where a module already defines `pytestmark`, combine rather than replace:

```python
pytestmark = [pytest.mark.skipif(SKIP_NETWORK, reason="network"), requires_groq]
```

- [ ] **Step 6: Remove the dead `network` alias**

`testing/test_altdata_tools.py:31` defines `network = pytest.mark.skipif(...)`. It is a
skipif bound to a name, not a registered marker, so `-m network` collects zero tests
suite-wide. Delete the alias and replace its usages with the appropriate gate.

Run: `grep -rn "^network = pytest.mark" testing/`

Expected: no output.

- [ ] **Step 7: Verify the offline suite is clean of service failures**

Run: `SKIP_NETWORK_TESTS=1 .venv/bin/python -m pytest testing/ -q --no-header --tb=no -p no:randomly 2>&1 | tail -3`

Expected: every remaining failure belongs to Tasks 6-10. No failure mentions a missing
key, an unreachable SearXNG, or a missing `CLAUDE.md`.

- [ ] **Step 8: Commit**

```bash
git add testing/_gates.py testing/test_gates.py testing/conftest.py testing/
git commit -m "gate tests that need a live service

Each gate skips with a reason naming what is missing, and flips to a hard
failure under NEMO_REQUIRE_SERVICES=1 so a fully-credentialled run proves the
gated tests still work rather than merely still exist.

Replaces the network alias in test_altdata_tools, which was a skipif bound to
a name rather than a registered marker, so -m network collected nothing."
```

---

### Task 6: Create `falsifier_alerts` in the schema, and stop swallowing its absence

Two components use this table and only one creates it.
`daemons/falsifier_watcher.py:58` creates it lazily inside `_ensure_alerts_table`, called
only from `tick()` at line 311. `tools/sentry_server/server.py:708` queries it and, when
it is absent, returns `'falsifier_alerts table does not exist yet'` instead of raising.
That fallback is why a missing table went unnoticed.

**Files:**
- Modify: `state/schema.py` (append to `CREATE_SCHEMA`, which ends around line 337)
- Modify: `daemons/falsifier_watcher.py:58-82` (delete `_ensure_alerts_table`), `:311` (its call)
- Modify: `tools/sentry_server/server.py:706-721` (delete the silent fallback)
- Test: `testing/test_falsifier_alerts_schema.py` (create)

**Interfaces:**
- Consumes: `state.schema.init_schema(db_path=None)` from Task 1, `isolated_databases` from Task 2.
- Produces: the `falsifier_alerts` table as part of the base schema. `_ensure_alerts_table`
  no longer exists — `testing/test_falsifier_watcher_e2e.py:34` imports it and must be updated.

- [ ] **Step 1: Write the failing test**

Create `testing/test_falsifier_alerts_schema.py`:

```python
"""falsifier_alerts is part of the schema, not something a daemon creates on
the way past.

The daemon created it lazily and the sentry tool swallowed its absence, so a
table two components depend on could simply not exist and nothing said so.
"""
import sqlite3

import pytest

from state.schema import init_schema, get_connection


def test_init_schema_creates_falsifier_alerts(monkeypatch, tmp_path):
    monkeypatch.setenv("NEMO_DB_PATH", str(tmp_path / "fresh.db"))
    init_schema()
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='falsifier_alerts'"
        ).fetchone()
    finally:
        conn.close()
    assert row is not None, "init_schema did not create falsifier_alerts"


def test_falsifier_alerts_has_the_idempotency_constraint(monkeypatch, tmp_path):
    """The table exists to stop the watcher firing twice on one triple."""
    monkeypatch.setenv("NEMO_DB_PATH", str(tmp_path / "fresh.db"))
    init_schema()
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO falsifier_alerts(thesis_id, falsifier_hash, evidence_id) "
            "VALUES (1, 'abc', 'ev1')")
        conn.commit()
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO falsifier_alerts(thesis_id, falsifier_hash, evidence_id) "
                "VALUES (1, 'abc', 'ev1')")
            conn.commit()
    finally:
        conn.close()


def test_ensure_alerts_table_helper_is_gone():
    """The lazy creator is deleted, not merely unused -- leaving it invites a
    second source of truth for the same DDL."""
    import daemons.falsifier_watcher as fw
    assert not hasattr(fw, "_ensure_alerts_table")
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `SKIP_NETWORK_TESTS=1 .venv/bin/python -m pytest testing/test_falsifier_alerts_schema.py -v`

Expected: FAIL — `init_schema did not create falsifier_alerts`.

- [ ] **Step 3: Add the DDL to `CREATE_SCHEMA`**

In `state/schema.py`, immediately before the closing `]` of `CREATE_SCHEMA` (after the
`idx_pe_layers_natural` line), add:

```python
    """CREATE TABLE IF NOT EXISTS falsifier_alerts(
        alert_id        INTEGER PRIMARY KEY AUTOINCREMENT,
        thesis_id       INTEGER NOT NULL,
        ticker          TEXT,
        falsifier_hash  TEXT,
        falsifier_text  TEXT,
        evidence_id     TEXT,
        score           REAL,
        reason          TEXT,
        fired_at        TIMESTAMP,
        UNIQUE(thesis_id, falsifier_hash, evidence_id)
    )""",
    "CREATE INDEX IF NOT EXISTS idx_falsifier_alerts_thesis ON falsifier_alerts(thesis_id, fired_at)",
```

- [ ] **Step 4: Delete the lazy creator**

In `daemons/falsifier_watcher.py`, delete the whole `_ensure_alerts_table` function
(lines 58-82) and its call at line 311. `tick()` already calls `init_schema()` — confirm
that, and if it does not, add `init_schema()` in its place.

Update `testing/test_falsifier_watcher_e2e.py:34`, which imports `_ensure_alerts_table`,
to stop importing it, and line 364, which calls it, to call `init_schema()` instead.

- [ ] **Step 5: Delete the silent fallback in the sentry tool**

In `tools/sentry_server/server.py`, replace the nested try/except at lines 706-721 so the
query runs without the inner `except sqlite3.OperationalError` that returns a note:

```python
      rows = conn.execute(
        """SELECT a.*, t.ticker
           FROM falsifier_alerts a
           LEFT JOIN theses t ON t.thesis_id = a.thesis_id
           WHERE a.fired_at >= ?
           ORDER BY a.fired_at DESC LIMIT ?""",
        (cutoff, limit),
      ).fetchall()
      return _ok('sentry_active_falsifier_alerts',
                 {'alerts': [dict(r) for r in rows], 'count': len(rows)})
```

A missing table is now a real error. It cannot happen once the table is in the schema,
and if it does, the caller must hear about it.

- [ ] **Step 6: Run the tests and verify they pass**

Run: `SKIP_NETWORK_TESTS=1 .venv/bin/python -m pytest testing/test_falsifier_alerts_schema.py testing/test_falsifier_watcher_e2e.py -v 2>&1 | tail -5`

Expected: all pass. `test_falsifier_watcher_e2e` goes from 8 failed to 0.

- [ ] **Step 7: Commit**

```bash
git add state/schema.py daemons/falsifier_watcher.py tools/sentry_server/server.py testing/
git commit -m "create falsifier_alerts in the schema, stop swallowing its absence

Two components used this table and only the daemon created it, lazily. The
sentry tool returned a note instead of raising when it was missing, which is
why the gap went unnoticed for so long."
```

---

### Task 7: Retire tests of designs that no longer exist

**Files:**
- Delete: `testing/test_phase_B2_settings_valid.py`
- Modify: `testing/test_phase_B3b_governance_xml_strip.py:62`
- Modify: `testing/test_finnhub_tools.py`, `testing/test_fred_integration.py`, `testing/test_multi_company_verification.py`

**Interfaces:**
- Consumes: nothing.
- Produces: nothing importable.

- [ ] **Step 1: Confirm `.mcp.json` was deliberately removed**

Run: `git log --oneline --all -- .mcp.json .mcp.json.bak`

Expected: `00abf63` moved `.mcp.json` to `.bak`; `d12e25c` deleted the `.bak` as a dead
artifact. The 7 tests in `test_phase_B2_settings_valid.py` assert a file the project
chose to remove. Confirm this before deleting — if the history says otherwise, stop and
report.

- [ ] **Step 2: Delete the stale settings test**

```bash
git rm testing/test_phase_B2_settings_valid.py
```

- [ ] **Step 3: Investigate the lxml workaround before touching its test**

`testing/test_phase_B3b_governance_xml_strip.py:62` fails with
`test premise wrong: raw lxml string parse already works`. The test asserts that raw lxml
parsing *fails*, in order to justify a workaround. It now succeeds.

Find the workaround the test was guarding:

Run: `grep -rn "xml_declaration\|XML declaration\|lstrip\|encoding=" --include="*.py" agent/ tools/ | grep -i "xml" | head`

If the workaround still exists and is now unnecessary, remove it in this task. If it is
still needed for a different reason, keep it and say so in the commit message. **Do not
retire the test without resolving this** — that would leave a dead branch behind.

- [ ] **Step 4: Rewrite the stale test to assert current behaviour**

Replace the premise assertion with one that documents what is true now: that the parser
handles an XML declaration directly. Keep the test — the behaviour is still worth
pinning, only the premise was wrong.

- [ ] **Step 5: Fix the five script-style functions**

`test_finnhub_tools.py::test_envelope`, `test_fred_integration.py` (3 functions), and
`test_multi_company_verification.py::test_company` are standalone scripts whose parameters
pytest tries to resolve as fixtures, producing 5 errors.

For each, either convert it into a real test with no unresolvable parameters, or rename
the file so pytest stops collecting it (e.g. `scripts/check_finnhub_tools.py`). Prefer
conversion: `test_finnhub_tools.py` and `test_fred_integration.py` cover two pipelines
the truth-source refactor deliberately kept, and both are currently untested.

- [ ] **Step 6: Verify no collection errors remain**

Run: `SKIP_NETWORK_TESTS=1 .venv/bin/python -m pytest testing/ -q --no-header --tb=no -p no:randomly 2>&1 | tail -3`

Expected: 0 errors.

- [ ] **Step 7: Commit**

```bash
git add -A testing/
git commit -m "retire tests of designs that no longer exist

test_phase_B2_settings_valid asserted .mcp.json exists; that file was removed
in 00abf63 and its backup deleted in d12e25c. The governance xml test asserted
that raw lxml parsing fails in order to justify a workaround, and it no longer
does. Five script-style functions errored because pytest read their parameters
as fixtures."
```

---

### Task 8: Fix the manifests and the environment

**Files:**
- Modify: `pyproject.toml:196-199` (`[tool.uv] environments`), and the dev dependency list
- Modify: `testing/test_sec_xbrl_functions.py:1414`
- Test: `testing/test_manifest_integrity.py` (exists — extend it)

**Interfaces:**
- Consumes: `testing/_gates.py` from Task 5.
- Produces: nothing importable.

- [ ] **Step 1: Write the failing test**

Add to `testing/test_manifest_integrity.py`:

```python
def test_uv_is_not_pinned_to_a_single_platform():
    """[tool.uv] environments = ["sys_platform == 'win32'"] makes uv sync refuse
    to resolve anywhere but Windows, which blocks every non-Windows checkout."""
    import tomllib, pathlib
    root = pathlib.Path(__file__).resolve().parent.parent
    data = tomllib.loads((root / "pyproject.toml").read_text())
    environments = data.get("tool", {}).get("uv", {}).get("environments", [])
    assert environments == [] or len(environments) > 1, (
        f"uv is pinned to a single platform: {environments}")


def test_pytest_asyncio_is_declared():
    """Without it, 11 async tests across 8 files cannot run at all."""
    import tomllib, pathlib
    root = pathlib.Path(__file__).resolve().parent.parent
    text = (root / "pyproject.toml").read_text()
    assert "pytest-asyncio" in text
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `SKIP_NETWORK_TESTS=1 .venv/bin/python -m pytest testing/test_manifest_integrity.py -v -k "uv_is_not_pinned or pytest_asyncio"`

Expected: FAIL on both.

- [ ] **Step 3: Remove the single-platform uv pin**

Delete lines 196-199 of `pyproject.toml`:

```toml
[tool.uv]
environments = [
  "sys_platform == 'win32'",
]
```

If any Windows-only dependency needs isolating, that belongs on the dependency itself as
a marker — `pywin32` already carries `; sys_platform == "win32"` — not as a global
resolution environment.

- [ ] **Step 4: Add pytest-asyncio**

Add `pytest-asyncio` to the dev dependency group in `pyproject.toml`, matching the
existing style of that group. Approved by the project owner on 2026-08-21; it is
test-only and does not enter the runtime dependency set.

Then configure it so the existing `async def` tests are collected. Add to the pytest
config section of `pyproject.toml`:

```toml
asyncio_mode = "auto"
```

- [ ] **Step 5: Install and verify the async tests now run**

Run: `.venv/bin/python -m pip install pytest-asyncio`

Run: `SKIP_NETWORK_TESTS=1 .venv/bin/python -m pytest testing/ -q --no-header --tb=no -p no:randomly 2>&1 | grep -c "async def functions are not natively supported"`

Expected: `0`.

- [ ] **Step 6: Re-mark the 500-ticker EDGAR loop**

`testing/test_sec_xbrl_functions.py:1414` marks `test_all_500_revenue` as
`@pytest.mark.slow`. It is a sequential live-EDGAR loop over roughly 500 tickers, and
`SEC_EMAIL` is now a real identity, so running it accidentally is a fair-access hazard.
Replace the marker:

```python
    @pytest.mark.network
    def test_all_500_revenue(self, all_tickers, rate_limiter):
```

Register `network` as a real marker in `pyproject.toml`'s pytest config so `-m network`
and `-m "not network"` both work — unlike the alias Task 5 deleted.

- [ ] **Step 7: Run the manifest tests and verify they pass**

Run: `SKIP_NETWORK_TESTS=1 .venv/bin/python -m pytest testing/test_manifest_integrity.py -v`

Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml uv.lock testing/test_manifest_integrity.py testing/test_sec_xbrl_functions.py
git commit -m "unpin uv from Windows, declare pytest-asyncio, re-mark the EDGAR loop

[tool.uv] environments pinned resolution to win32, so uv sync refused to run
on any other platform. pytest-asyncio was undeclared, leaving 11 async tests
uncollectable. test_all_500_revenue was marked slow rather than network,
which matters now that SEC_EMAIL carries a real identity."
```

---

### Task 9: Stop the model pool trusting a malformed override

Three stacked defects make every `OpenRouterModel` built without an explicit model name
default to the literal string `'# optional override; if unset, pool auto-resolves'`.
Confirmed at runtime.

**Files:**
- Modify: `.env.example:3`
- Modify: `agent/openrouter_template.py:24-42` (`_verify_model_alive`), `:56-86` (`_build_reasoning_pool`)
- Test: `testing/test_model_pool.py` (create)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `agent.openrouter_template._is_valid_model_id(model_id: str) -> bool`.
  `_verify_model_alive` keeps its signature and returns `False` for a malformed id.

- [ ] **Step 1: Write the failing test**

Create `testing/test_model_pool.py`:

```python
"""The pool must not seat a malformed override at position 0.

.env.example ships `PRIMARY_REASONING_MODEL=` followed by an inline comment.
dotenv reads that comment as the value, _build_reasoning_pool puts the override
first, and _verify_model_alive's `except Exception: return True` declared the
comment string alive. The result: every default OpenRouter agent resolved to a
model named "# optional override; if unset, pool auto-resolves".
"""
import pytest

import agent.openrouter_template as ot


@pytest.mark.parametrize("bad", [
    "# optional override; if unset, pool auto-resolves",
    "",
    "   ",
    "not-a-model",
    "vendor/",
    "/model",
])
def test_malformed_model_ids_are_rejected(bad):
    assert ot._is_valid_model_id(bad) is False


@pytest.mark.parametrize("good", [
    "deepseek/deepseek-chat-v3.1:free",
    "z-ai/glm-4.5-air:free",
    "meta-llama/llama-3.3-70b-instruct:free",
])
def test_well_formed_model_ids_are_accepted(good):
    assert ot._is_valid_model_id(good) is True


def test_verify_model_alive_rejects_malformed_without_calling_the_api(monkeypatch):
    """A malformed id is dead on inspection. It must not cost a network call."""
    def explode(*args, **kwargs):
        raise AssertionError("network call made for a malformed model id")
    monkeypatch.setattr(ot, "OpenAI", explode)
    assert ot._verify_model_alive("# optional override", "key") is False


def test_transient_errors_still_count_as_alive(monkeypatch):
    """A rate limit or auth failure does not prove the model is dead."""
    class FakeClient:
        def __init__(self, *a, **k):
            self.chat = self
            self.completions = self
        def create(self, **kwargs):
            raise RuntimeError("429 rate limit")
    monkeypatch.setattr(ot, "OpenAI", FakeClient)
    assert ot._verify_model_alive("vendor/model:free", "key") is True


def test_env_example_does_not_ship_a_comment_as_a_value():
    """The template is the root cause -- every clone inherits it."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent
    for line in (root / ".env.example").read_text().splitlines():
        if line.startswith("PRIMARY_REASONING_MODEL="):
            value = line.split("=", 1)[1].strip()
            assert not value.startswith("#"), (
                f"dotenv reads this comment as the value: {value!r}")
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `SKIP_NETWORK_TESTS=1 .venv/bin/python -m pytest testing/test_model_pool.py -v`

Expected: FAIL with `AttributeError: module 'agent.openrouter_template' has no attribute '_is_valid_model_id'`.

- [ ] **Step 3: Add id validation and use it in the liveness check**

In `agent/openrouter_template.py`, add above `_verify_model_alive`:

```python
def _is_valid_model_id(model_id: str) -> bool:
  """OpenRouter ids are `vendor/model`, optionally `:tag`.

  Guarding the shape here matters because PRIMARY_REASONING_MODEL is read from
  the environment, and dotenv will happily hand back an inline comment as a
  value. A malformed id must never reach the pool.
  """
  if not model_id or not model_id.strip():
    return False
  candidate = model_id.strip()
  if candidate.startswith("#"):
    return False
  vendor, separator, model = candidate.partition("/")
  return bool(separator) and bool(vendor.strip()) and bool(model.strip())
```

Then change `_verify_model_alive` so a malformed id is rejected before any network call,
and so a genuinely malformed request is not treated as healthy:

```python
def _verify_model_alive(model_id: str, api_key: str, timeout: float = 10.0) -> bool:
  """Send a 1-token completion to check the model endpoint exists.

  Returns False for a malformed id (rejected without a network call) and for an
  explicit 404. Returns True when the error is transient -- rate limit, auth,
  timeout -- since none of those prove the model is dead.
  """
  if not _is_valid_model_id(model_id):
    return False
  try:
    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1", timeout=timeout)
    client.chat.completions.create(
      model=model_id,
      messages=[{"role": "user", "content": "ping"}],
      max_tokens=1,
    )
    return True
  except NotFoundError:
    return False
  except Exception:
    return True  # rate limit, auth, etc. don't prove the model is dead
```

- [ ] **Step 4: Fix the environment template**

`.env.example:3` currently reads:

```
PRIMARY_REASONING_MODEL=         # optional override; if unset, pool auto-resolves
```

Replace it so the comment sits on its own line, where dotenv cannot read it as a value:

```
# PRIMARY_REASONING_MODEL: optional override. Leave unset and the pool auto-resolves.
PRIMARY_REASONING_MODEL=
```

- [ ] **Step 5: Tell the owner to fix their local `.env`**

`.env` is gitignored and machine-local, so this task cannot repair it. Line 3 of the
owner's `.env` has the same defect. Report it in the task summary: the line must become
`PRIMARY_REASONING_MODEL=` with the comment moved above it, or the running system keeps
resolving to the comment string. **Do not edit `.env` — report it.**

- [ ] **Step 6: Run the tests and verify they pass**

Run: `SKIP_NETWORK_TESTS=1 .venv/bin/python -m pytest testing/test_model_pool.py -v`

Expected: PASS, 12 passed.

- [ ] **Step 7: Commit**

```bash
git add agent/openrouter_template.py .env.example testing/test_model_pool.py
git commit -m "reject malformed model ids instead of seating them in the pool

.env.example shipped PRIMARY_REASONING_MODEL= followed by an inline comment,
which dotenv reads as the value. The pool seated it first and the liveness
check's bare `except Exception: return True` declared it alive, so every
default OpenRouter agent resolved to a model named after the comment."
```

---

### Task 10: Fail loud on SEC identity, and clear the remaining debris

**Files:**
- Modify: `tools/web_search_server/sec_utils.py:14`, `tools/web_search_server/hf_letters.py:31`, `daemons/edgar_firehose.py:60`, `daemons/gdelt_poller.py:59`, `daemons/rss_aggregator.py:58`
- Modify: `testing/test_db_separation.py:22`
- Modify: `README.md` (Alpaca environment variable names)
- Delete: `debug_test.xlsx`, `simple_test.xlsx`
- Test: `testing/test_sec_identity.py` (create)

**Interfaces:**
- Consumes: `testing/_gates.py` from Task 5.
- Produces: `tools.web_search_server.sec_utils.require_sec_email() -> str`.

- [ ] **Step 1: Write the failing test**

Create `testing/test_sec_identity.py`:

```python
"""SEC fair access requires a real contact identity in the User-Agent.

Defaulting to analyst@example.com misrepresents the caller to the SEC and does
it silently, in five separate files. A missing SEC_EMAIL must stop the process,
not quietly forge an identity.
"""
import pytest


def test_require_sec_email_raises_when_unset(monkeypatch):
    from tools.web_search_server import sec_utils
    monkeypatch.delenv("SEC_EMAIL", raising=False)
    with pytest.raises(ValueError, match="SEC_EMAIL"):
        sec_utils.require_sec_email()


def test_require_sec_email_raises_when_empty(monkeypatch):
    from tools.web_search_server import sec_utils
    monkeypatch.setenv("SEC_EMAIL", "   ")
    with pytest.raises(ValueError, match="SEC_EMAIL"):
        sec_utils.require_sec_email()


def test_require_sec_email_returns_the_configured_value(monkeypatch):
    from tools.web_search_server import sec_utils
    monkeypatch.setenv("SEC_EMAIL", "someone@example.org")
    assert sec_utils.require_sec_email() == "someone@example.org"


def test_no_module_still_defaults_to_the_placeholder():
    """All five call sites must be converted, not just the first one."""
    import pathlib, subprocess
    root = pathlib.Path(__file__).resolve().parent.parent
    result = subprocess.run(
        ["grep", "-rn", "analyst@example.com", "--include=*.py", str(root)],
        capture_output=True, text=True)
    hits = [line for line in result.stdout.splitlines() if "/.venv/" not in line]
    assert hits == [], "placeholder SEC identity still present:\n" + "\n".join(hits)
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `SKIP_NETWORK_TESTS=1 .venv/bin/python -m pytest testing/test_sec_identity.py -v`

Expected: FAIL — `require_sec_email` does not exist, and the grep finds 5 hits.

- [ ] **Step 3: Add the loud accessor**

In `tools/web_search_server/sec_utils.py`, replace line 14:

```python
def require_sec_email() -> str:
  """Return the configured SEC contact identity, or refuse to continue.

  SEC fair access asks for a real contact address in the User-Agent. Defaulting
  to a placeholder misrepresents the caller, so an unset value stops the caller
  rather than forging one.
  """
  value = os.getenv('SEC_EMAIL', '').strip()
  if not value:
    raise ValueError(
      "SEC_EMAIL is not set. SEC fair access requires a real contact address "
      "in the User-Agent header. Set SEC_EMAIL in your .env file.")
  return value
```

- [ ] **Step 4: Convert the other four call sites**

Replace the `os.getenv('SEC_EMAIL', 'analyst@example.com')` default in
`tools/web_search_server/hf_letters.py:31`, `daemons/edgar_firehose.py:60`,
`daemons/gdelt_poller.py:59`, and `daemons/rss_aggregator.py:58` with a call to
`require_sec_email()`. In the three daemons the User-Agent is built at module import, so
move the call into the function that builds the header rather than leaving it at module
level — otherwise importing the daemon raises before its entrypoint can report a clean
error.

- [ ] **Step 5: Fix the over-specified database separation test**

`testing/test_db_separation.py:22` asserts
`os.path.basename(CACHE_DB_PATH) == "tool_cache.db"`, which fails the moment anyone sets
`NEMO_CACHE_DB_PATH` — the very override the file exists to protect, and which Task 2's
fixture now sets for every module. Replace that assertion with the real invariant, and
test the default separately:

```python
def test_cache_and_state_use_different_files():
    from agent.cache import current_cache_db_path
    from state.schema import current_db_path

    assert os.path.abspath(current_cache_db_path()) != os.path.abspath(current_db_path())


def test_default_filenames_when_no_override_is_set(monkeypatch):
    monkeypatch.delenv("NEMO_CACHE_DB_PATH", raising=False)
    monkeypatch.delenv("NEMO_DB_PATH", raising=False)
    from agent.cache import current_cache_db_path
    from state.schema import current_db_path

    assert os.path.basename(current_cache_db_path()) == "tool_cache.db"
    assert os.path.basename(current_db_path()) == "session.db"
```

- [ ] **Step 6: Correct the README's Alpaca variable names**

Find what the code actually reads:

Run: `grep -rn "ALPACA_" --include="*.py" tools/alpaca/ | grep getenv`

Update `README.md` so the documented names match. Names that do not match are worse than
absent — they send a reader to configure a variable nothing reads.

- [ ] **Step 7: Delete the scratch files**

```bash
rm -f debug_test.xlsx simple_test.xlsx
```

Both are untracked, were never committed on any branch, and are leftover artifacts.

- [ ] **Step 8: Run the tests and verify they pass**

Run: `SKIP_NETWORK_TESTS=1 .venv/bin/python -m pytest testing/test_sec_identity.py testing/test_db_separation.py -v`

Expected: all pass.

- [ ] **Step 9: Run the full offline suite and confirm the goal**

Run: `SKIP_NETWORK_TESTS=1 .venv/bin/python -m pytest testing/ -q --no-header --tb=no -p no:randomly 2>&1 | tail -3`

Expected: **0 failed, 0 errors.** If anything still fails, report it with its cause
rather than adjusting the target.

- [ ] **Step 10: Commit**

```bash
git add -A
git commit -m "fail loud on missing SEC identity, fix remaining test debt

SEC_EMAIL defaulted to analyst@example.com in five files, silently
misrepresenting the caller under SEC fair access. test_db_separation asserted
a hardcoded filename, so it broke under the NEMO_CACHE_DB_PATH override it
exists to protect. README documented Alpaca variable names the code does not
read."
```

---

## Verification

After Task 10, prove both halves of the success criterion.

**Offline, no credentials:**

```bash
SKIP_NETWORK_TESTS=1 .venv/bin/python -m pytest testing/ -q --no-header --tb=no -p no:randomly 2>&1 | tail -3
```

Expected: 0 failed, 0 errors.

**Strict, with credentials and services:**

```bash
docker compose up -d searxng
NEMO_REQUIRE_SERVICES=1 .venv/bin/python -m pytest testing/ -q --no-header --tb=no -p no:randomly 2>&1 | tail -3
```

Expected: 0 failed, 0 skipped-for-missing-service. This run requires `GROQ_API_KEY` and
`OPENROUTER_API_KEY` to be set to real values. **Until this run has been observed passing
at least once, the gates added in Task 5 are unproven** — they may be converting broken
tests into invisible ones rather than absent services into honest skips. Record the
result either way; if the credentials are unavailable, say so explicitly rather than
reporting the offline number alone.
