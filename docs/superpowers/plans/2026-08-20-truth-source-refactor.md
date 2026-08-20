# Truth-Source Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collapse the MCP tool layer toward a stateless data layer — delete the redundant openbb server, merge the duplicate options tool, remove FinBERT, and split the disposable cache off the state database.

**Architecture:** Merge-first with differential testing. The duplicate options path is unified into `get_options_metrics` while the old altdata implementation still exists, so the two can be compared live on the same ticker before anything is deleted. Deletions follow, then database hygiene, then the skill update last — after the tools it references are final.

**Tech Stack:** Python 3.12, pytest 9, MCP stdio servers, yfinance, SQLite (+ sqlite-vec), sentence-transformers.

**Spec:** `docs/superpowers/specs/2026-08-20-truth-source-refactor-design.md`

## Global Constraints

- Branch is `truth-source-refactor`. Do not commit to `main`.
- Python 3.12 only (`requires-python = ">=3.12,<3.13"`).
- No new dependencies. This refactor only removes and repins.
- Mock data belongs in tests only, never in a production code path.
- No silent fallbacks. A failure surfaces as a structured error envelope; it does not degrade to a fabricated number.
- Every removal is preceded by a repo-wide grep for consumers. Do not trust a thing's own docstring or marker description — the `slow` marker says "FinBERT" but has 14 users in `testing/test_sec_xbrl_functions.py`.
- Tests that hit the network use `@pytest.mark.network` and must pass with `SKIP_NETWORK_TESTS=1` set (skipped, not failed).
- Commit after every task. Commit messages describe what changed and why, nothing else.
- The `.venv` python is `.venv/bin/python` on macOS/Linux, `.venv\Scripts\python.exe` on Windows. Commands below use `pytest` directly; prefix with the venv python if not activated.

---

### Task 0: Baseline and safe workspace

Establishes the number every later regression gate is a delta against, and stops the daemons that write the database Task 6 modifies.

**Run Task 0.5 first if no `.venv` exists** — on arm64 the environment cannot be built
until the CUDA torch pin is removed, so there is nothing to measure a baseline against.

**Baseline caveat:** `testing/` holds scratch scripts alongside real tests
(`debug_client.py`, `debug_ebitda.py`, `get_8k_details.py`,
`run_full_orchestration_AMD.py`, `run_orchestration_resume_AMD.py`, `simple_test.py`).
pytest will not collect most of them, but do not treat "116 files" as a test count.
Record what pytest actually collects.

**Offline-gate caveat:** only `testing/test_altdata_tools.py` uses the `network`
marker. Roughly 15 other files make live calls without one, so `SKIP_NETWORK_TESTS=1`
does not fully isolate an offline run. Expect network-dependent failures in the
baseline and record them as pre-existing rather than chasing them.

**Files:**
- Create: `docs/superpowers/plans/baseline.txt`

- [ ] **Step 1: Stop the daemons**

Six daemons hold open handles on `db_cache/session.db`. Task 6 changes how that file is opened; a running daemon would keep mixed-version code alive (see `docs/known_issues.md`, "Sentry daemons are running mixed-version code").

On Windows: `powershell -File scripts/stop_daemons.ps1`
On macOS/Linux, or if no daemons are running, skip — confirm with:

```bash
ps aux | grep -E "daemons\.(edgar_firehose|news_watcher|falsifier_watcher|sentry_triage|rss_aggregator|gdelt_poller)" | grep -v grep
```

Expected: no output.

- [ ] **Step 2: Record the collection baseline**

```bash
SKIP_NETWORK_TESTS=1 pytest testing/ --collect-only -q 2>&1 | tail -3 | tee docs/superpowers/plans/baseline.txt
```

Expected: a line like `N tests collected`. Record N. `docs/known_issues.md` cites 314 as of the pre-earnings merge; the real number is whatever this prints.

- [ ] **Step 3: Record the pass baseline**

```bash
SKIP_NETWORK_TESTS=1 pytest testing/ -q 2>&1 | tail -5 | tee -a docs/superpowers/plans/baseline.txt
```

Expected: a summary line. Append any pre-existing failures to `baseline.txt` — those are not yours to fix, but you must know them so you do not later mistake them for regressions.

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/plans/baseline.txt
git commit -m "record test baseline before truth-source refactor"
```

---

### Task 0.5: Make the environment installable

**Pulled forward from Task 5.** `pyproject.toml` pins `torch==2.5.1+cu121` against a
CUDA-only index. Those wheels do not exist for arm64, so `uv sync` cannot resolve on
an Apple Silicon machine — the environment cannot be built at all until this changes.
Every remaining torch consumer is CPU-bound, so the CUDA pin buys nothing.

This task is independent of the FinBERT deletion and must run before any other task.

**Files:**
- Modify: `pyproject.toml:7-9, 203-211`, `requirements.txt`

- [ ] **Step 1: Repin torch to CPU wheels**

In `pyproject.toml`, change lines 7-9:

```toml
  "torch==2.5.1",
  "torchvision==0.20.1",
  "torchaudio==2.5.1",
```

Delete the `[[tool.uv.index]]` block named `pytorch-cu121` (around line 203) and the
`[tool.uv.sources]` entries pinning torch/torchvision/torchaudio to it (around lines
209-211). Apply the same version changes to `requirements.txt`.

Leave `accelerate` and `bitsandbytes` alone — they are removed in Task 5, after the
dead code that uses them is deleted.

- [ ] **Step 2: Verify no CUDA pin remains**

```bash
grep -n "cu121\|pytorch-cu121" pyproject.toml requirements.txt
```

Expected: no output.

- [ ] **Step 3: Build the environment**

```bash
uv venv --python 3.12 .venv
uv pip install -r requirements.txt 2>&1 | tail -20
```

Expected: resolution succeeds. If it fails on a package unrelated to torch, record the
package and stop — a second unsatisfiable pin is a finding, not something to work
around silently.

- [ ] **Step 4: Verify torch and the embedder import**

```bash
.venv/bin/python -c "import torch; print('torch', torch.__version__)"
.venv/bin/python -c "from agent.rag.embedder import embed; print('embedder ok')"
```

Expected: a plain version with no `+cu121` suffix, then `embedder ok`. The second
command downloads the ~80MB MiniLM model on first run.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml requirements.txt
git commit -m "repin torch to CPU wheels

The +cu121 pin targets a CUDA-only index with no arm64 wheels, so uv sync
could not resolve on Apple Silicon at all. Every torch consumer in the repo
is CPU-bound -- finbert_runner ran device=-1 and MiniLM embedding is trivial
on CPU -- so the CUDA build bought nothing."
```

---

### Task 1: Move the options math helpers into the financial engine

Relocates four functions and two constants that carry the documented options bug fixes. Pure move — no behavior change — so the migrated tests must pass immediately.

**Files:**
- Modify: `tools/financial_modeling_engine/utils.py` (add helpers near the top, before `get_options_metrics` at line 619)
- Create: `testing/test_options_merge.py`
- Reference (do not modify yet): `tools/altdata_server/server.py:145-235`

**Interfaces:**
- Produces, all importable from `tools.financial_modeling_engine.utils`:
  - `_safe_float(v: Any, default: float = 0.0) -> float`
  - `_leg_price(opt: Dict) -> Tuple[float, bool]`
  - `compute_implied_move(spot: float, atm_call_ask: float, atm_put_ask: float) -> Dict[str, Any]`
  - `_find_atm_options(rows: List[Dict], spot: float, target_expiry: Optional[str] = None) -> Tuple[Optional[Dict], Optional[Dict], Optional[str]]`
  - `_us_market_today() -> datetime.date`
  - `_ATM_GAP_THRESHOLD: float = 0.08`, `_PARITY_TOLERANCE: float = 0.05`

- [ ] **Step 1: Write the failing tests**

Create `testing/test_options_merge.py`. These are the options tests from `testing/test_altdata_tools.py` retargeted at the new import path. Do not delete the originals yet — Task 3 does that, after the differential test proves the merge.

```python
"""Options math helpers, post-merge into the financial engine.

These encode four bugs documented in docs/known_issues.md: put-call parity
violation, after-hours last_price fallback, yfinance NaN/sentinel handling,
and the zero-spot guard. They are the safety net for the merge.
"""
from __future__ import annotations

import os
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from tools.financial_modeling_engine.utils import (  # noqa: E402
    _find_atm_options,
    _leg_price,
    _safe_float,
    compute_implied_move,
)


def test_safe_float_handles_nan():
    assert _safe_float(float("nan")) == 0.0
    assert _safe_float(float("nan"), default=1.5) == 1.5
    assert _safe_float(None) == 0.0
    assert _safe_float("abc") == 0.0
    assert _safe_float("2.5") == 2.5


def test_leg_price_prefers_live_ask():
    price, stale = _leg_price({"ask": 3.0, "last_price": 9.0, "bid": 1.0})
    assert price == 3.0
    assert stale is False


def test_leg_price_falls_back_to_last_when_ask_zero():
    price, stale = _leg_price({"ask": 0.0, "last_price": 4.25, "bid": 1.0})
    assert price == 4.25
    assert stale is True


def test_leg_price_falls_back_to_bid_when_no_last():
    price, stale = _leg_price({"ask": 0.0, "last_price": 0.0, "bid": 2.0})
    assert price == 2.0
    assert stale is True


def test_leg_price_all_zero():
    price, stale = _leg_price({"ask": 0.0, "last_price": 0.0, "bid": 0.0})
    assert price == 0.0
    assert stale is True


def test_leg_price_nan_ask_does_not_mask_ask_price():
    # A NaN ask is truthy; `ask or ask_price` would swallow a valid ask_price.
    price, stale = _leg_price({"ask": float("nan"), "ask_price": 5.0})
    assert price == 5.0
    assert stale is False


def test_compute_implied_move_basic_math():
    out = compute_implied_move(spot=100.0, atm_call_ask=3.0, atm_put_ask=2.0)
    assert out["straddle_cost"] == 5.0
    assert out["implied_move_pct"] == 0.05


def test_compute_implied_move_zero_spot():
    out = compute_implied_move(spot=0.0, atm_call_ask=3.0, atm_put_ask=2.0)
    assert out["implied_move_pct"] == 0.0


def test_compute_implied_move_nan_ask_no_nan_output():
    out = compute_implied_move(spot=100.0, atm_call_ask=float("nan"), atm_put_ask=2.0)
    assert out["straddle_cost"] == 2.0
    assert out["implied_move_pct"] == out["implied_move_pct"]  # not NaN


def _rows(expiry: str = "2099-01-15"):
    return [
        {"expiration": expiry, "option_type": "call", "strike": 95.0, "ask": 7.0},
        {"expiration": expiry, "option_type": "call", "strike": 100.0, "ask": 3.0},
        {"expiration": expiry, "option_type": "put", "strike": 100.0, "ask": 2.0},
        {"expiration": expiry, "option_type": "put", "strike": 105.0, "ask": 8.0},
    ]


def test_find_atm_options_selects_nearest_strike():
    call, put, expiry = _find_atm_options(_rows(), spot=100.0)
    assert call["strike"] == 100.0
    assert put["strike"] == 100.0
    assert expiry == "2099-01-15"


def test_find_atm_options_empty_returns_none():
    assert _find_atm_options([], spot=100.0) == (None, None, None)


def test_find_atm_uses_last_price_after_hours():
    # ask == 0 everywhere (market closed); selection must still find the strike.
    rows = [
        {"expiration": "2099-01-15", "option_type": "call", "strike": 100.0,
         "ask": 0.0, "last_price": 3.0},
        {"expiration": "2099-01-15", "option_type": "put", "strike": 100.0,
         "ask": 0.0, "last_price": 2.0},
    ]
    call, put, _ = _find_atm_options(rows, spot=100.0)
    assert call is not None and put is not None
    assert call["strike"] == 100.0


def test_skew_classification():
    rows = [
        {"expiration": "2099-08-15", "option_type": "call", "strike": 100,
         "ask": 3.0, "implied_volatility": 0.30},
        {"expiration": "2099-08-15", "option_type": "put", "strike": 100,
         "ask": 4.0, "implied_volatility": 0.38},
    ]
    call, put, _ = _find_atm_options(rows, 100.0, "2099-08-15")
    skew_diff = float(put["implied_volatility"]) - float(call["implied_volatility"])
    assert skew_diff > 0.03  # put_heavy
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
SKIP_NETWORK_TESTS=1 pytest testing/test_options_merge.py -q
```

Expected: collection error — `ImportError: cannot import name '_find_atm_options' from 'tools.financial_modeling_engine.utils'`.

- [ ] **Step 3: Move the helpers**

Copy `tools/altdata_server/server.py` lines 145-235 verbatim into `tools/financial_modeling_engine/utils.py`, inserted immediately **before** `def get_options_metrics` (currently line 619). This is `_safe_float`, `_leg_price`, `compute_implied_move`, `_ATM_GAP_THRESHOLD`, `_PARITY_TOLERANCE`, `_us_market_today`, and `_find_atm_options` — including the full body of `_find_atm_options` through its `nearest_atm` inner function and the parity/gap guards that follow.

Keep the docstrings. They explain why each guard exists and are the only record of the bugs.

Add to the `utils.py` import block at the top:

```python
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional, Tuple
```

`utils.py` line 2 already imports `Dict, Any, List, Optional` — extend it with `Tuple` rather than duplicating the line. Do not remove the function-local `from datetime import ...` statements elsewhere in the file; they are unrelated.

Do **not** delete anything from `tools/altdata_server/server.py` in this task. Both copies coexist until Task 3.

**Note for Task 2:** lines 145-235 do *not* contain the put-call parity guard. That
guard lives inside the `options_implied_move` handler at roughly lines 1716-1732 and
is migrated separately in Task 2. Moving only this block would silently drop it.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
SKIP_NETWORK_TESTS=1 pytest testing/test_options_merge.py -q
```

Expected: `13 passed`.

- [ ] **Step 5: Verify the original tests still pass**

The old copies are untouched, so `test_altdata_tools.py` must be unaffected.

```bash
SKIP_NETWORK_TESTS=1 pytest testing/test_altdata_tools.py -q
```

Expected: same result as `baseline.txt` for this file.

- [ ] **Step 6: Commit**

```bash
git add tools/financial_modeling_engine/utils.py testing/test_options_merge.py
git commit -m "move options math helpers into the financial engine

Pure relocation of _safe_float, _leg_price, compute_implied_move and
_find_atm_options ahead of merging the duplicate options tool. Both copies
coexist until the differential test confirms the merge is faithful."
```

---

### Task 2: Merge the straddle into `get_options_metrics`

Adds the ATM straddle to the financial engine's options tool from a single chain fetch, and fixes the timezone bug the two implementations disagreed on.

**Files:**
- Modify: `tools/financial_modeling_engine/utils.py:619+` (`get_options_metrics`)
- Modify: `testing/test_options_merge.py` (add adapter tests)

**Interfaces:**
- Consumes from Task 1: `_find_atm_options`, `_leg_price`, `compute_implied_move`, `_us_market_today`.
- Produces: `_chain_to_rows(chain, expiry: str) -> List[Dict]` and
  `_straddle_legs(atm_call: Dict, atm_put: Dict, spot: float) -> Tuple[float, float, bool]`
  in `utils.py`, and `get_options_metrics(ticker: str) -> Dict[str, Any]` gains a top-level `implied_move` key:

```python
{
  "implied_move": {
    "implied_move_pct": float,   # straddle / spot
    "straddle_cost": float,
    "front_expiry": str,         # "YYYY-MM-DD"
    "quotes_stale": bool,        # True when any leg fell back off ask
  }
}
```

On failure `implied_move` is `{"error": "<reason>"}` — never a fabricated number, and never absent.

- [ ] **Step 1: Write the failing adapter test**

Append to `testing/test_options_merge.py`:

```python
def test_chain_to_rows_normalizes_yfinance_columns():
    import pandas as pd

    from tools.financial_modeling_engine.utils import _chain_to_rows

    class _Chain:
        calls = pd.DataFrame([{"strike": 100.0, "ask": 3.0, "bid": 2.8,
                               "lastPrice": 2.9, "impliedVolatility": 0.35}])
        puts = pd.DataFrame([{"strike": 100.0, "ask": 2.0, "bid": 1.8,
                              "lastPrice": 1.9, "impliedVolatility": 0.33}])

    rows = _chain_to_rows(_Chain(), "2099-01-15")

    assert len(rows) == 2
    call = next(r for r in rows if r["option_type"] == "call")
    assert call["expiration"] == "2099-01-15"
    assert call["strike"] == 100.0
    assert call["ask"] == 3.0
    assert call["last_price"] == 2.9          # camelCase -> snake_case
    assert call["implied_volatility"] == 0.35
    assert {r["option_type"] for r in rows} == {"call", "put"}
```

- [ ] **Step 2: Run it to verify it fails**

```bash
SKIP_NETWORK_TESTS=1 pytest testing/test_options_merge.py::test_chain_to_rows_normalizes_yfinance_columns -q
```

Expected: `ImportError: cannot import name '_chain_to_rows'`.

- [ ] **Step 3: Add the adapter**

`_find_atm_options` expects flat row dicts; `get_options_metrics` holds yfinance DataFrames. Add to `utils.py`, immediately after `_find_atm_options`:

```python
def _chain_to_rows(chain, expiry: str) -> List[Dict]:
  """Flatten one yfinance option_chain result into the row shape the ATM
  helpers expect. yfinance uses camelCase DataFrame columns; the helpers use
  snake_case keys, so the rename happens here and nowhere else."""
  rows: List[Dict] = []
  for df, otype in ((chain.calls, 'call'), (chain.puts, 'put')):
    for _, row in df.iterrows():
      rows.append({
        'expiration': expiry,
        'option_type': otype,
        'strike': _safe_float(row.get('strike')),
        'ask': _safe_float(row.get('ask')),
        'bid': _safe_float(row.get('bid')),
        'last_price': _safe_float(row.get('lastPrice')),
        'implied_volatility': _safe_float(row.get('impliedVolatility')),
      })
  return rows
```

- [ ] **Step 4: Run it to verify it passes**

```bash
SKIP_NETWORK_TESTS=1 pytest testing/test_options_merge.py::test_chain_to_rows_normalizes_yfinance_columns -q
```

Expected: `1 passed`.

- [ ] **Step 5: Write the failing parity-guard tests**

The put-call parity guard (commit 22e102b, `known_issues.md` #6) is **not** in the
helper block Task 1 moved — it lives in the altdata handler. Migrating it as a named
function makes it testable instead of buried in a 60-line handler.

Append to `testing/test_options_merge.py`:

```python
def test_straddle_legs_uses_ask_when_parity_holds():
    from tools.financial_modeling_engine.utils import _straddle_legs

    # C - P = 3 - 2 = 1; S - K = 100 - 99 = 1. Parity holds, asks are trusted.
    call = {"strike": 99.0, "ask": 3.0, "last_price": 2.5}
    put = {"strike": 99.0, "ask": 2.0, "last_price": 1.5}
    call_px, put_px, stale = _straddle_legs(call, put, spot=100.0)
    assert (call_px, put_px) == (3.0, 2.0)
    assert stale is False


def test_straddle_legs_rebuilds_on_parity_violation():
    """Live ORCL case: call 6.75 / put 28.35 at strike 237.5, spot 236.34 -- a
    $21 parity violation from junk wide quotes left at the close. A nonzero ask
    is not necessarily a sane ask."""
    from tools.financial_modeling_engine.utils import _straddle_legs

    call = {"strike": 237.5, "ask": 6.75, "last_price": 5.10}
    put = {"strike": 237.5, "ask": 28.35, "last_price": 6.40}
    call_px, put_px, stale = _straddle_legs(call, put, spot=236.34)

    assert (call_px, put_px) == (5.10, 6.40), "should rebuild both legs off last_price"
    assert stale is True


def test_straddle_legs_keeps_ask_when_no_fallback_available():
    from tools.financial_modeling_engine.utils import _straddle_legs

    # Parity violated but no last_price/bid to fall back to: keep the asks
    # rather than fabricating a number.
    call = {"strike": 237.5, "ask": 6.75}
    put = {"strike": 237.5, "ask": 28.35}
    call_px, put_px, stale = _straddle_legs(call, put, spot=236.34)
    assert (call_px, put_px) == (6.75, 28.35)
```

- [ ] **Step 6: Run to verify they fail**

```bash
SKIP_NETWORK_TESTS=1 pytest testing/test_options_merge.py -k straddle_legs -q
```

Expected: `ImportError: cannot import name '_straddle_legs'`.

- [ ] **Step 7: Migrate the parity guard**

Add to `utils.py`, immediately after `_chain_to_rows`. This is the guard from
`tools/altdata_server/server.py:1716-1732`, extracted into a named function:

```python
def _straddle_legs(atm_call: Dict, atm_put: Dict,
                   spot: float) -> Tuple[float, float, bool]:
  """Price both straddle legs, with a put-call parity sanity check.

  C - P should approximate S - K for same-strike legs. A nonzero ask is not
  necessarily a SANE ask -- junk wide quotes left at the close pass a bare >0
  check (live ORCL: call 6.75 / put 28.35 at strike 237.5 with spot 236.34, a
  $21 violation). On gross violation both legs are rebuilt from last_price/bid
  and the result is flagged stale.

  Returns (call_price, put_price, quotes_stale)."""
  call_px, call_stale = _leg_price(atm_call)
  put_px, put_stale = _leg_price(atm_put)
  stale = call_stale or put_stale

  strike = _safe_float(atm_call.get('strike'))
  if (call_px > 0 and put_px > 0 and spot > 0
      and abs((call_px - put_px) - (spot - strike)) / spot > _PARITY_TOLERANCE):
    def _no_ask(opt):
      return {k: v for k, v in opt.items() if k not in ('ask', 'ask_price')}
    c2, _ = _leg_price(_no_ask(atm_call))
    p2, _ = _leg_price(_no_ask(atm_put))
    # Only rebuild if a fallback actually exists -- otherwise keep the asks
    # rather than returning a fabricated zero.
    if c2 > 0 and p2 > 0:
      call_px, put_px = c2, p2
      stale = True

  return call_px, put_px, stale
```

- [ ] **Step 8: Run to verify they pass**

```bash
SKIP_NETWORK_TESTS=1 pytest testing/test_options_merge.py -k straddle_legs -q
```

Expected: `3 passed`.

- [ ] **Step 9: Fix the timezone bug**

In `get_options_metrics`, replace `today = _date.today()` (line 648) with:

```python
    today = _us_market_today()
```

The two implementations disagreed here: altdata used ET-aware dating, the financial engine used local date. Between 00:00 and 05:00 UTC — exactly when after-hours pre-earnings research runs — local date is a day ahead of ET, which drops a valid front expiry. See the `_us_market_today` docstring.

The `from datetime import datetime, date as _date` line inside `get_options_metrics` can keep `_date` if other references remain; remove it only if this was its sole use.

- [ ] **Step 10: Compute the straddle from the front expiry**

`get_options_metrics` already resolves the 7d point via `_find_expiry(7)` and fetches its chain for the term structure. Reuse that same fetched chain — do not add a network call. Inside the term-structure loop, capture the 7d chain, then after the loop insert:

```python
    # ATM straddle from the front expiry — reuses the 7d chain already
    # fetched for the term structure rather than refetching.
    if front_chain is not None:
      rows = _chain_to_rows(front_chain, front_expiry)
      call, put, chosen_expiry = _find_atm_options(rows, spot)
      if call is None or put is None:
        out['implied_move'] = {'error': 'no ATM strike within threshold'}
      else:
        call_px, put_px, stale = _straddle_legs(call, put, spot)
        moved = compute_implied_move(spot, call_px, put_px)
        out['implied_move'] = {
          **moved,
          'front_expiry': chosen_expiry,
          'quotes_stale': stale,
        }
    else:
      out['implied_move'] = {'error': 'front expiry chain unavailable'}
```

Declare `front_chain = None` and `front_expiry = None` before the term-structure loop, and inside it set both when `label == '7d'` and the fetch succeeded.

- [ ] **Step 11: Verify against a live ticker**

```bash
python -c "
from tools.financial_modeling_engine.utils import get_options_metrics
import json
r = get_options_metrics('MSFT')
print(json.dumps({k: r[k] for k in ('ticker','spot_price','implied_move')}, indent=2, default=str))
"
```

Expected: `implied_move` present with a plausible `implied_move_pct` (roughly 0.01–0.10 for a megacap outside earnings week), a `front_expiry` in the future, and `quotes_stale` true or false depending on market hours. If it prints an `error` key, do not proceed — diagnose first.

- [ ] **Step 12: Run the full options test file**

```bash
SKIP_NETWORK_TESTS=1 pytest testing/test_options_merge.py -q
```

Expected: `17 passed`.

- [ ] **Step 13: Commit**

```bash
git add tools/financial_modeling_engine/utils.py testing/test_options_merge.py
git commit -m "merge ATM straddle into get_options_metrics

The straddle now comes from the 7d chain already fetched for the term
structure, so one options tool serves both the event-move calculation and
the binary-event rule from a single network fetch.

Also adopts ET-aware dating. The financial engine used local date, which
between 00:00-05:00 UTC is a day ahead of ET and drops a valid front expiry
during after-hours research."
```

---

### Task 3: Differential test, then delete the old options path

Proves the merge is faithful by comparing against the live old implementation, then removes it. This is the only evidence that the merged numbers match; do not skip it.

**Files:**
- Create: `testing/test_options_differential.py` (temporary — deleted in Step 5)
- Modify: `tools/altdata_server/server.py` (remove `get_options_implied_move`, helpers, `options_runner.py` wiring)
- Delete: `tools/altdata_server/options_runner.py`
- Modify: `testing/test_altdata_tools.py` (remove migrated options tests)

- [ ] **Step 1: Write the differential test**

```python
"""Temporary: assert the merged get_options_metrics agrees with the altdata
implementation it replaces. Deleted once the old path is removed."""
from __future__ import annotations

import os
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

pytestmark = pytest.mark.network

if os.environ.get("SKIP_NETWORK_TESTS") == "1":
    pytest.skip("network tests disabled", allow_module_level=True)

TICKERS = ["MSFT", "AAPL", "NVDA"]


@pytest.mark.parametrize("ticker", TICKERS)
def test_merged_matches_altdata(ticker):
    from tools.altdata_server.options_runner import fetch_options_chain
    from tools.altdata_server.server import (
        _find_atm_options as old_find,
        _leg_price as old_leg,
        compute_implied_move as old_move,
    )
    from tools.financial_modeling_engine.utils import get_options_metrics

    new = get_options_metrics(ticker)
    assert new.get("success"), f"{ticker}: merged tool failed: {new.get('error')}"
    spot = new["spot_price"]

    rows = fetch_options_chain(ticker, near_days=60)
    o_call, o_put, o_expiry = old_find(rows, spot)
    if o_call is None or o_put is None:
        pytest.skip(f"{ticker}: old path found no ATM strike; nothing to compare")

    old_px = old_move(spot, old_leg(o_call)[0], old_leg(o_put)[0])
    merged = new["implied_move"]
    assert "error" not in merged, f"{ticker}: merged implied_move errored: {merged}"

    assert merged["front_expiry"] == o_expiry, (
        f"{ticker}: front expiry disagrees — merged {merged['front_expiry']} "
        f"vs old {o_expiry}"
    )

    # Tolerance, not equality: the two fetch the chain at slightly different
    # moments, so quotes can move between calls. A 15% relative gap means the
    # math diverged, not that the market ticked.
    a, b = merged["implied_move_pct"], old_px["implied_move_pct"]
    assert a > 0 and b > 0, f"{ticker}: zero implied move — merged {a}, old {b}"
    assert abs(a - b) / max(a, b) < 0.15, (
        f"{ticker}: implied move diverged — merged {a:.4f} vs old {b:.4f}"
    )
```

- [ ] **Step 1b: Add the sentinel guard**

**This guard is the point of the test.** A live probe on 2026-08-20 at 01:53 ET
(`marketState=PREPRE`) found every MSFT contract returning `bid=ask=openInterest=0`
and ATM IV pinned to the constant `0.0156` sentinel across all four tenors, making
skew exactly 0.0. Both implementations read that same feed. Without a guard they
agree perfectly on garbage and the test passes while proving nothing — the exact
false-confidence failure this task exists to prevent.

Insert at the top of `test_merged_matches_altdata`, before any comparison:

```python
    # Refuse to pass on sentinel data. yfinance serves bid=ask=0 and IV
    # sentinels outside market hours; two implementations reading the same
    # degraded feed agree trivially. Agreement is only evidence when the
    # underlying quotes are real.
    ts = new.get("term_structure", {})
    ivs = [v.get("atm_iv") for v in ts.values()
           if isinstance(v, dict) and v.get("atm_iv") is not None]
    assert ivs, f"{ticker}: no ATM IV in term structure — cannot verify"
    assert max(ivs) > 0.08, (
        f"{ticker}: IV sentinel data (max atm_iv={max(ivs):.4f}) — run this test "
        f"during US market hours (09:30-16:00 ET). Agreement on sentinel quotes "
        f"is not evidence the merge is correct."
    )
    assert len(set(round(v, 6) for v in ivs)) > 1, (
        f"{ticker}: identical ATM IV {ivs[0]} across all tenors — sentinel feed, "
        f"not a real term structure."
    )
```

- [ ] **Step 2: Run it — during US market hours only**

This test is only meaningful between 09:30 and 16:00 ET on a trading day. Check first:

```bash
.venv/bin/python -c "
from datetime import datetime
from zoneinfo import ZoneInfo
n = datetime.now(ZoneInfo('America/New_York'))
ok = n.weekday() < 5 and (9,30) <= (n.hour, n.minute) < (16,0)
print(f'{n:%Y-%m-%d %H:%M %Z} — {\"OK to run\" if ok else \"WAIT: market closed\"}')
"
```

If it says WAIT, stop and report that Task 3 is time-gated. Do not run the
differential test outside market hours and do not weaken the sentinel guard to
make it pass — a green run on sentinel data is worse than no run, because it
retires the risk without testing it.

```bash
.venv/bin/python -m pytest testing/test_options_differential.py -v
```

Expected: 3 passed (or skipped for a ticker with no ATM strike).

If a ticker fails on `front_expiry`, the ET-date fix from Task 2 Step 5 is likely missing or applied to only one side. If it fails on the implied-move tolerance, the ATM selection diverged — compare `o_call["strike"]` against the merged path's chosen strike before changing the tolerance. **Do not widen the tolerance, and do not lower the 0.08 sentinel floor, to make the
test pass.** Either would defeat the entire purpose of this task.

- [ ] **Step 3: Verify no consumer of the old tool remains**

```bash
grep -rn "get_options_implied_move\|options_runner\|_OPTIONS_RUNNER" \
  --exclude-dir=__pycache__ --exclude-dir=.git . | grep -v test_options_differential
```

Expected: hits only in `tools/altdata_server/server.py`, `testing/test_altdata_tools.py`, and `.claude/skills/preearnings-research/SKILL.md`. The skill is updated in Task 8; note the line numbers now.

- [ ] **Step 4: Delete the old options path**

From `tools/altdata_server/server.py` remove:
- the `get_options_implied_move` tool definition (around line 1403) and its `call_tool` dispatch branch (around line 1563)
- the `options_implied_move` handler method (around line 1659)
- `_safe_float`, `_leg_price`, `compute_implied_move`, `_find_atm_options`, `_us_market_today`, `_ATM_GAP_THRESHOLD`, `_PARITY_TOLERANCE` (lines 145-235)
- the `_OPTIONS_RUNNER` constant (line 66) and the `get_options_implied_move` line from the module docstring (line 9)

Delete `tools/altdata_server/options_runner.py`.

`_safe_float` may be used elsewhere in `server.py`. Check before removing it:

```bash
grep -c "_safe_float" tools/altdata_server/server.py
```

If other callers remain, keep `_safe_float` in `server.py` — the financial engine has its own copy and the two are independent.

From `testing/test_altdata_tools.py` remove the migrated tests (they now live in `test_options_merge.py`): `test_options_implied_move_basic_math`, `test_options_implied_move_zero_spot`, `test_find_atm_options_selects_nearest_strike`, `test_find_atm_options_empty_returns_none`, `test_skew_classification`, `test_safe_float_handles_nan`, `test_compute_implied_move_nan_ask_no_nan_output`, `test_leg_price_prefers_live_ask`, `test_leg_price_falls_back_to_last_when_ask_zero`, `test_leg_price_falls_back_to_bid_when_no_last`, `test_leg_price_all_zero`, `test_find_atm_uses_last_price_after_hours`, `test_options_handler_after_hours_uses_last_price`, `test_options_handler_parity_violation_falls_back_to_last_price`, and the `_OPTIONS_RUNNER` constant.

All of these now have counterparts in `testing/test_options_merge.py`, including the
two handler-level tests — `test_options_handler_parity_violation_falls_back_to_last_price`
is covered by `test_straddle_legs_rebuilds_on_parity_violation`, and
`test_options_handler_after_hours_uses_last_price` by
`test_find_atm_uses_last_price_after_hours` plus the `_leg_price` fallback tests.

- [ ] **Step 5: Delete the differential test**

```bash
rm testing/test_options_differential.py
```

It compared against an implementation that no longer exists. Leaving it would break collection.

- [ ] **Step 6: Verify the server still starts**

```bash
python -c "from tools.altdata_server.server import AltDataServer; print('ok')" 2>&1 | tail -2
```

Expected: `ok`. If the class has a different name, list it with `grep -n "^class" tools/altdata_server/server.py`.

- [ ] **Step 7: Run the suite**

```bash
SKIP_NETWORK_TESTS=1 pytest testing/ -q 2>&1 | tail -5
```

Expected: baseline count minus the deleted altdata options tests, plus 13 from `test_options_merge.py`. No failures beyond those recorded in `baseline.txt`.

- [ ] **Step 8: Commit**

```bash
git add -A tools/altdata_server testing/
git commit -m "remove the duplicate options implementation from altdata

get_options_metrics now serves both the straddle and the term structure.
A differential test confirmed the merged numbers matched the old path on
MSFT/AAPL/NVDA before deletion."
```

---

### Task 4: Delete the openbb server

All four tools are covered by servers we keep. It is also the documented indefinite-hang server.

**Files:**
- Delete: `tools/openbb_server/`, `testing/test_openbb_server.py`
- Modify: `pyproject.toml`, `requirements.txt`

- [ ] **Step 1: Verify nothing consumes it**

```bash
grep -rn "openbb_server\|obb_insider_trading\|obb_analyst_consensus\|obb_news_company\|obb_options_chain" \
  --exclude-dir=__pycache__ --exclude-dir=.git . | grep -v "^./testing/test_openbb_server.py"
```

Expected: no hits outside the server's own directory. Task 3 already removed the `options_chain_rows` parameter that coupled altdata to it. If a `.claude/skills/` hit appears, stop — the spec asserts zero skill references and that assertion would be wrong.

- [ ] **Step 2: Delete**

```bash
git rm -r tools/openbb_server testing/test_openbb_server.py
```

- [ ] **Step 3: Remove the dependency pins**

From `pyproject.toml`, delete every line matching `"openbb`. From `requirements.txt`, delete every line matching `^openbb`.

```bash
grep -c "openbb" pyproject.toml requirements.txt
```

Expected: `0` for both.

- [ ] **Step 4: Verify no import survives**

```bash
grep -rn "import openbb\|from openbb" --exclude-dir=__pycache__ --exclude-dir=.git .
```

Expected: no output.

- [ ] **Step 5: Run the suite**

```bash
SKIP_NETWORK_TESTS=1 pytest testing/ -q 2>&1 | tail -5
```

Expected: Task 3's count minus 7 (`test_openbb_server.py`). No new failures.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "delete the openbb MCP server

All four tools are covered closer to source by finnhub (insider transactions,
analyst consensus, company news) and the merged get_options_metrics. No skill
referenced it, and it is the indefinite-hang server documented in
docs/known_issues.md. Removes ~40 openbb-* pins."
```

---

### Task 5: Delete FinBERT, web-traffic, and dead template code

Removes the only judgment tool in the data layer, plus a key-gated tool that was never wired.

**Files:**
- Delete: `tools/altdata_server/finbert_runner.py`, `tools/preearnings/web_traffic.py`, `testing/test_preearnings_web_traffic.py`, `agent/huggingface_template.py`
- Modify: `tools/altdata_server/server.py`, `testing/test_altdata_tools.py`, `pyproject.toml`, `requirements.txt`

- [ ] **Step 1: Grep for consumers before deleting**

```bash
grep -rn "finbert\|FinBERT\|web_traffic\|get_web_traffic_signal\|huggingface_template" \
  --exclude-dir=__pycache__ --exclude-dir=.git . | grep -v "^./docs/"
```

Record every hit. Expected: `tools/altdata_server/` (runner + server wiring), `tools/preearnings/web_traffic.py`, the two test files, `agent/huggingface_template.py`, and `.claude/skills/preearnings-research/SKILL.md:31`. The skill is Task 8.

- [ ] **Step 2: Confirm `huggingface_template.py` is dead**

```bash
grep -rn "huggingface_template" --exclude-dir=__pycache__ agent/ tools/ daemons/ testing/ main.py
```

Expected: no output other than the file itself. It is imported by nothing — it is a template sibling of `ollama_template.py` and `openrouter_template.py`, and it owns the only remaining `accelerate`/`bitsandbytes` usage.

- [ ] **Step 3: Delete the files**

```bash
git rm tools/altdata_server/finbert_runner.py \
       tools/preearnings/web_traffic.py \
       testing/test_preearnings_web_traffic.py \
       agent/huggingface_template.py
```

- [ ] **Step 4: Remove the server wiring**

From `tools/altdata_server/server.py` remove the `get_finbert_sentiment` and `get_web_traffic_signal` tool definitions, their `call_tool` dispatch branches, their handler methods, the `_FINBERT_RUNNER` constant (line 65), and their lines from the module docstring.

From `testing/test_altdata_tools.py` remove `test_finbert_runner_positive_sentiment`, `test_finbert_runner_negative_sentiment`, `test_finbert_runner_empty_texts_fails_clean`, `test_finbert_runner_unknown_tool_fails_clean`, and the `_FINBERT_RUNNER` constant.

- [ ] **Step 5: Keep the `slow` marker, fix only its description**

The marker has 14 users in `testing/test_sec_xbrl_functions.py`. Removing it breaks that suite. In `pyproject.toml` under `[tool.pytest.ini_options]`, change:

```toml
    "slow: model-loading tests (FinBERT) excluded from quick runs via -m 'not slow'",
```

to:

```toml
    "slow: long-running tests excluded from quick runs via -m 'not slow'",
```

- [ ] **Step 6: Verify the marker still works**

```bash
SKIP_NETWORK_TESTS=1 pytest testing/test_sec_xbrl_functions.py -m slow --collect-only -q 2>&1 | tail -3
```

Expected: 14 tests collected.

- [ ] **Step 7: Drop accelerate and bitsandbytes**

The torch CPU repin already happened in Task 0.5. What remains is the two pins whose
only consumer was `agent/huggingface_template.py`, deleted in Step 3 of this task.

Remove `"accelerate==1.12.0"` and `"bitsandbytes==0.49.0"` from `pyproject.toml` and
`requirements.txt`.

Leave `"transformers==4.57.3"` — sentence-transformers requires it, so it stays as a
transitive dependency; dropping the direct pin is out of scope.

- [ ] **Step 8: Verify the embedder still imports**

```bash
python -c "from agent.rag.embedder import embed; print('embedder ok')"
```

Expected: `embedder ok`. This is the check that torch survived the repin. If it fails with a torch import error, the CPU wheels are not installed — run `pip install -r requirements.txt` before continuing.

- [ ] **Step 9: Run the suite**

```bash
SKIP_NETWORK_TESTS=1 pytest testing/ -q 2>&1 | tail -5
```

Expected: Task 4's count minus the 4 FinBERT tests and the web-traffic test file. No new failures.

- [ ] **Step 10: Commit**

```bash
git add -A
git commit -m "remove FinBERT sentiment, web-traffic, and dead template code

Sentiment is a judgment about text, not a fact about the world, and the
news-digest sub-agent already reads the article text in full -- FinBERT was a
second pass over data the agent was holding. Scoring moves to the agent.

Drops accelerate/bitsandbytes and repins torch from +cu121 to CPU wheels;
every remaining consumer is CPU-bound. torch itself stays, since the RAG
embedder needs it via sentence-transformers.

The slow pytest marker stays -- 14 SEC XBRL tests use it. Only its
FinBERT-specific description changed."
```

---

### Task 6: Split the cache off the state database

`Session_Cache` and the state stores share one SQLite file with no WAL and no busy_timeout, while six daemons write concurrently.

**Files:**
- Modify: `agent/cache.py:9-11`, `state/schema.py:11,374-395`
- Create: `testing/test_db_separation.py`

**Interfaces:**
- Produces: `state.schema.DB_PATH` becomes an absolute path, overridable via `NEMO_DB_PATH`. `agent.cache.CACHE_DB_PATH` is a new module-level absolute path, overridable via `NEMO_CACHE_DB_PATH`.

- [ ] **Step 1: Write the failing tests**

```python
"""The disposable tool cache must not share a file with book state, and both
connections must tolerate concurrent writers."""
from __future__ import annotations

import os
import sqlite3
import sys
import threading

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def test_cache_and_state_use_different_files():
    from agent.cache import CACHE_DB_PATH
    from state.schema import DB_PATH

    assert os.path.abspath(CACHE_DB_PATH) != os.path.abspath(DB_PATH)
    assert os.path.basename(CACHE_DB_PATH) == "tool_cache.db"


def test_db_paths_are_absolute():
    # A CWD-relative path silently creates a second database when a process
    # starts from a different directory.
    from agent.cache import CACHE_DB_PATH
    from state.schema import DB_PATH

    assert os.path.isabs(DB_PATH), f"DB_PATH is relative: {DB_PATH}"
    assert os.path.isabs(CACHE_DB_PATH), f"CACHE_DB_PATH is relative: {CACHE_DB_PATH}"


def test_db_path_env_override(monkeypatch, tmp_path):
    import importlib

    target = str(tmp_path / "override.db")
    monkeypatch.setenv("NEMO_DB_PATH", target)

    import state.schema

    importlib.reload(state.schema)
    assert state.schema.DB_PATH == target

    monkeypatch.delenv("NEMO_DB_PATH")
    importlib.reload(state.schema)


def test_state_connection_uses_wal(tmp_path):
    from state.schema import get_connection

    conn = get_connection(str(tmp_path / "wal_check.db"))
    try:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode.lower() == "wal"
    finally:
        conn.close()


def test_concurrent_writers_do_not_lock(tmp_path):
    """Default journal mode raises 'database is locked' here; WAL plus a
    busy_timeout does not."""
    from state.schema import get_connection, init_schema

    db = str(tmp_path / "concurrent.db")
    init_schema(db)

    errors = []

    def _writer(n):
        try:
            conn = get_connection(db)
            for i in range(20):
                conn.execute(
                    "INSERT OR REPLACE INTO watchlist(ticker, priority) VALUES (?, ?)",
                    (f"T{n}{i}", 1),
                )
                conn.commit()
            conn.close()
        except sqlite3.OperationalError as e:
            errors.append(f"writer {n}: {e}")

    threads = [threading.Thread(target=_writer, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"concurrent writes failed: {errors}"
```

- [ ] **Step 2: Run to verify they fail**

```bash
SKIP_NETWORK_TESTS=1 pytest testing/test_db_separation.py -q
```

Expected: `ImportError: cannot import name 'CACHE_DB_PATH'` plus failures on the absolute-path and WAL assertions.

- [ ] **Step 3: Update `state/schema.py`**

Replace line 11:

```python
DB_PATH = os.path.join("db_cache", "session.db")
```

with:

```python
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Absolute, not CWD-relative: a relative path silently opens a different
# database when a process starts from another directory, and the mismatch
# surfaces as an empty result rather than an error.
DB_PATH = os.environ.get(
    "NEMO_DB_PATH", os.path.join(_REPO_ROOT, "db_cache", "session.db")
)
```

In `get_connection` (line 374), after `conn.row_factory = sqlite3.Row`, add:

```python
    # WAL lets readers proceed during a write; busy_timeout makes a blocked
    # writer wait rather than raise immediately. Six daemons write this file
    # concurrently.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
```

- [ ] **Step 4: Update `agent/cache.py`**

Replace lines 9-11:

```python
class Session_Cache():
    def __init__(self):
        os.makedirs("db_cache", exist_ok=True)
        self.connection = sqlite3.connect("db_cache/session.db")
```

with:

```python
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Disposable tool/news/scrape caches. Deliberately a different file from the
# state database -- cache rows are regenerable, book state is not, and mixing
# them means a cache-clearing mistake can take theses and positions with it.
CACHE_DB_PATH = os.environ.get(
    "NEMO_CACHE_DB_PATH", os.path.join(_REPO_ROOT, "db_cache", "tool_cache.db")
)


class Session_Cache():
    def __init__(self):
        os.makedirs(os.path.dirname(CACHE_DB_PATH), exist_ok=True)
        self.connection = sqlite3.connect(CACHE_DB_PATH, timeout=5.0)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA busy_timeout=5000")
```

Existing cache rows in `session.db` are abandoned, not migrated — the cache is disposable and repopulates on use. Leave the old tables in place; the state tables share that file and dropping them is out of scope.

- [ ] **Step 5: Run to verify they pass**

```bash
SKIP_NETWORK_TESTS=1 pytest testing/test_db_separation.py -q
```

Expected: `5 passed`.

- [ ] **Step 6: Verify the state layer still reads real data**

```bash
python -c "
from state.schema import DB_PATH, init_schema, get_connection
print('DB_PATH:', DB_PATH)
init_schema()
c = get_connection()
print('journal_mode:', c.execute('PRAGMA journal_mode').fetchone()[0])
print('tables:', len(c.execute(\"SELECT name FROM sqlite_master WHERE type='table'\").fetchall()))
c.close()
"
```

Expected: an absolute `DB_PATH`, `journal_mode: wal`, and a non-zero table count. A count of zero means it opened a fresh file — the path is wrong.

- [ ] **Step 7: Run the suite**

```bash
SKIP_NETWORK_TESTS=1 pytest testing/ -q 2>&1 | tail -5
```

Expected: Task 5's count plus 5. Watch `testing/test_scraper_03_cache.py` specifically — it exercises `Session_Cache` and is the most likely casualty.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "split the tool cache off the state database

Session_Cache wrote db_cache/session.db, the same file holding theses,
positions and the sentry queue. It now uses tool_cache.db.

Both paths become absolute and env-overridable -- the previous CWD-relative
DB_PATH would silently open a different database from a process started in
another directory, surfacing as an empty result rather than an error.

Adds WAL and a 5s busy_timeout to both connections; six daemons write
concurrently and neither connection configured locking."
```

---

### Task 7: Repair the dependency manifests

`sentence-transformers` is in `requirements.txt` only, so `uv sync` builds an environment where RAG fails at first use.

**Files:**
- Modify: `pyproject.toml`
- Create: `testing/test_manifest_integrity.py`

- [ ] **Step 1: Write the failing test**

```python
"""Guards the manifest drift that broke RAG under uv sync."""
from __future__ import annotations

import os
import re

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _pyproject_deps() -> set:
    text = open(os.path.join(_REPO, "pyproject.toml")).read()
    block = text.split("dependencies = [", 1)[1].split("]", 1)[0]
    return {m.split("==")[0].strip().lower()
            for m in re.findall(r'"([^"]+)"', block)}


def test_sentence_transformers_is_declared():
    # agent/rag/embedder.py imports it; without the pin, uv sync produces an
    # environment where the first rag_search fails.
    assert "sentence-transformers" in _pyproject_deps()


def test_no_openbb_pins_remain():
    assert not [d for d in _pyproject_deps() if d.startswith("openbb")]


def test_no_cuda_pinned_torch():
    text = open(os.path.join(_REPO, "pyproject.toml")).read()
    assert "+cu121" not in text
    assert "pytorch-cu121" not in text


def test_removed_direct_deps_are_gone():
    deps = _pyproject_deps()
    for pkg in ("accelerate", "bitsandbytes"):
        assert pkg not in deps, f"{pkg} still pinned but has no consumer"
```

- [ ] **Step 2: Run to verify it fails**

```bash
SKIP_NETWORK_TESTS=1 pytest testing/test_manifest_integrity.py -q
```

Expected: `test_sentence_transformers_is_declared` fails. The other three should already pass from Tasks 4 and 5 — if any fails, that task's manifest edits are incomplete.

- [ ] **Step 3: Add the missing pin**

In `pyproject.toml`, add to the `dependencies` list, alphabetically near the other `s` entries:

```toml
  "sentence-transformers==5.5.1",
```

The version matches `requirements.txt:193`.

- [ ] **Step 4: Run to verify it passes**

```bash
SKIP_NETWORK_TESTS=1 pytest testing/test_manifest_integrity.py -q
```

Expected: `4 passed`.

- [ ] **Step 5: Verify a clean resolve**

```bash
uv lock --check 2>&1 | tail -5
```

If it reports the lock is stale, run `uv lock` and commit the regenerated `uv.lock`. If `uv` is unavailable, skip and note it — `requirements.txt` remains the installed source of truth.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock testing/test_manifest_integrity.py
git commit -m "declare sentence-transformers in pyproject

It was in requirements.txt only, so uv sync produced an environment where
agent/rag/embedder.py failed at the first rag_search. Adds a test guarding
this and the openbb/cuda removals."
```

---

### Task 8: Update the pre-earnings skill

Lands last, after the tools it references are final.

**Files:**
- Modify: `.claude/skills/preearnings-research/SKILL.md:29-35, 74-90`

- [ ] **Step 1: Replace the FinBERT step in the news-digest template**

At `SKILL.md:29-31`, the template currently reads:

```
Read mcp__nemo_finnhub__get_company_news({TICKER}, last 30d) IN FULL, then
run mcp__nemo_altdata__get_finbert_sentiment on the headline+summary texts.
```

Replace with:

```
Read mcp__nemo_finnhub__get_company_news({TICKER}, last 30d) IN FULL and
score sentiment yourself from the headline and summary text.
```

- [ ] **Step 2: Replace the output field**

In the same JSON template, replace `"finbert_net_score":X` with:

```
"net_sentiment":X,
```

Keep the same scale so downstream weighting is unchanged. Search the file for every other `finbert_net_score` occurrence and rename each:

```bash
grep -n "finbert" .claude/skills/preearnings-research/SKILL.md
```

Expected after editing: no output.

- [ ] **Step 3: Point the asymmetry step at the single options tool**

At `SKILL.md:77-78`, the asymmetry-inputs step lists `get_options_metrics` and `get_options_implied_move(ticker, spot_price)` as separate calls. Replace that pair with a single `get_options_metrics(ticker)`, and note that the straddle now arrives as its `implied_move` sub-object:

```
   - `get_short_interest` (SI % float, days-to-cover), `get_options_metrics`
     (IV skew, put/call volume ratio, term structure, AND the ATM straddle as
     `implied_move`), `get_price_history` (3M momentum + ~500 daily bars), and
     reuse the Layer-0 `get_analyst_revisions_history` pct_bullish.
```

Keep the IV-sentinel rule and the event-move extraction paragraph unchanged — both still apply. Update the event-move text to note that `front_iv_7d` and `back_iv_30d` come from the same call that provides the straddle.

Also note `implied_move.quotes_stale`: when true, the straddle rebuilt from `last_price` and the event-move math inherits that staleness (`known_issues.md` #5).

- [ ] **Step 3b: Fix the stale example in `earnings-eval`**

`.claude/skills/earnings-eval/SKILL.md:140` lists `finbert_sentiment` as a row in the
signal-attribution illustration table. It is an EXAMPLE, not a tool call — nothing
breaks — but it names a signal that no longer exists. Replace that row with a signal
the system still produces, e.g.:

```
| news_sentiment | neutral | — |
```

Do not remove the table or the `google_trends` row above it.

- [ ] **Step 4: Verify no stale tool references**

```bash
grep -rn "get_finbert_sentiment\|get_options_implied_move\|get_web_traffic_signal\|obb_\|finbert" .claude/skills/
```

Expected: no output. Note this grep now also catches bare `finbert`, which is how the
`earnings-eval` example row was missed on the first pass.

- [ ] **Step 5: Full suite**

```bash
SKIP_NETWORK_TESTS=1 pytest testing/ -q 2>&1 | tail -5
```

Expected: no failures beyond `baseline.txt`.

- [ ] **Step 6: Live network run**

```bash
pytest testing/ -q -m network 2>&1 | tail -8
```

Expected: no failures. These hit real APIs and are the only check that the merged options path works against live yfinance data.

- [ ] **Step 7: End-to-end acceptance**

Restart Claude Code so the MCP servers respawn with the new code — `docs/known_issues.md` documents that a running server keeps executing its startup-time code. Then run:

```
/preearnings-research <TICKER>
```

Pick a ticker with earnings inside 5-10 days. Pass conditions:
- no `data_gap` for options inputs
- a `net_sentiment` value present in the news digest, agent-produced
- the asymmetry classification completes without falling through to `na`
- `implied_move` present with a plausible `implied_move_pct`

- [ ] **Step 8: Commit**

```bash
git add .claude/skills/preearnings-research/SKILL.md
git commit -m "update preearnings skill for the merged options tool

Sentiment is scored by the news-digest sub-agent from text it already reads
in full, replacing the FinBERT call. Options inputs come from a single
get_options_metrics call that now carries the ATM straddle."
```

- [ ] **Step 9: Restart the daemons**

Task 0 stopped them. On Windows, relaunch via `nemo.bat`. Verify:

```bash
ps aux | grep -E "daemons\." | grep -v grep | wc -l
```

Expected: 6.

---

## Verification summary

| Deliverable | Task | Gate |
|---|---|---|
| D1 merged options tool | 1-3 | 13 unit tests + live differential on 3 tickers |
| D2 delete openbb | 4 | zero `openbb` imports repo-wide; suite minus 7 |
| D3 delete FinBERT/web-traffic | 5 | `slow` marker still collects 14 SEC tests; embedder imports |
| D4 cache/state split + WAL | 6 | 5 tests incl. a 4-thread concurrent-writer test |
| D5 manifest repair | 7 | 4 manifest-integrity tests; `uv lock --check` |
| D6 skill update | 8 | zero stale tool references; live `/preearnings-research` run |
