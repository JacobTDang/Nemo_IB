"""Every entry point in `research/` builds its own schema before it reads.

Nothing in production ever created the point-in-time tables. Each job assumed
the store was already there, which is true in the normal deployment order --
`research-daily` runs at 22:30 and creates it, and the rest find a schema
waiting. Nothing enforces that order. A `docker compose run --rm research-watch`
on a new volume, a `research-seed` on the 1st before the first nightly, or any
rescheduling makes some other job the first one, and it dies on a raw
`no such table` traceback before it does anything.

It was fixed one module at a time -- `daily_job`, then `scanner`, `replay` and
`scoring` -- and each fix came with a test of that one module, which is why the
next three stayed broken with a green suite. So this is a sweep rather than a
seventh case: it finds the job modules itself and fails the day an eighth is
added without the call, instead of the day someone runs it on a fresh volume.

A job module is one that defines `main()` and touches `pit_store`. That is the
definition the bug has: a module with no `main()` is never the first command,
and one that never opens the store cannot find its tables missing.

`status` is the one exception, and it is exempt for the opposite reason rather
than by oversight. It reads and changes nothing, so building a schema would
mean creating an empty store as a side effect of looking at one -- and then
reporting nine jobs as "never run" against a database this command had just
invented. It answers a missing store by saying so, which
`testing/test_status.py` pins.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

RESEARCH = pathlib.Path(__file__).resolve().parent.parent / "research"


def _tree(path: pathlib.Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _main_of(tree: ast.AST) -> ast.FunctionDef | None:
    """The module-level `main`, which is what a `python -m` run enters.

    Module level only. A `main` nested in a class or another function is not
    the entry point `python -m research.<module>` calls, and counting one would
    let a module pass this check without being fixed.
    """
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            return node
    return None


def _touches_pit_store(tree: ast.AST) -> bool:
    return any(isinstance(node, ast.Name) and node.id == "pit_store"
               for node in ast.walk(tree))


def _calls_init_schema(node: ast.AST) -> bool:
    """`pit_store.init_schema()`, written out, inside this function.

    Matched on the call rather than on the name appearing anywhere, so a
    mention in a docstring or a comment about init_schema does not satisfy it.
    """
    for inner in ast.walk(node):
        if not isinstance(inner, ast.Call):
            continue
        func = inner.func
        if (isinstance(func, ast.Attribute) and func.attr == "init_schema"
                and isinstance(func.value, ast.Name)
                and func.value.id == "pit_store"):
            return True
    return False


# Readers, not jobs. See the module docstring.
EXEMPT = {"status"}


def _job_modules() -> list[str]:
    found = []
    for path in sorted(RESEARCH.glob("*.py")):
        tree = _tree(path)
        if (_main_of(tree) is not None and _touches_pit_store(tree)
                and path.stem not in EXEMPT):
            found.append(path.stem)
    return found


JOBS = _job_modules()

# The seven that existed when the sweep was written. Named so that a detector
# broken into finding nothing fails loudly here rather than passing every
# module vacuously -- a sweep that silently stops sweeping is the same failure
# as the per-module tests it replaces.
KNOWN_JOBS = {"activist_watch", "announcements", "daily_job", "replay",
              "scanner", "scoring", "seed_consensus"}


def test_the_sweep_still_finds_the_jobs_it_was_written_against():
    missing = sorted(KNOWN_JOBS - set(JOBS))
    assert not missing, (
        f"{missing} define main() and use pit_store but this sweep no longer "
        f"sees them, so it is checking less than it claims")


@pytest.mark.parametrize("module", JOBS)
def test_a_job_module_creates_its_own_schema(module):
    main = _main_of(_tree(RESEARCH / f"{module}.py"))
    assert _calls_init_schema(main), (
        f"`research/{module}.py` defines main() and opens the point-in-time "
        f"store, but never calls pit_store.init_schema(). Run it first on a "
        f"fresh volume and it dies on `no such table`. init_schema is "
        f"idempotent, so the call costs nothing when the store exists.")
