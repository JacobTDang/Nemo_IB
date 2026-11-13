"""A module with a `main()` is not runnable until it has a `__main__` guard.

`python -m research.supply_chain` is written in `deploy/README.md` as the way to
read one name's filing by hand. Without

    if __name__ == "__main__":
        raise SystemExit(main())

that command exits **0** and does nothing at all: no rows, no run-log entry, no
output. The operator sees success, which is the worst reading available, and
the failure is invisible from the outside -- the module imports cleanly, every
test that calls `main()` passes, and only the command line is dead.

This is a sweep because the mistake is invisible per module. It is written once
by copying the bottom of a neighbouring file, and the one file that forgot it
was the one whose instructions live in a runbook. So: every module in
`research/` that defines a module-level `main()` must also enter it from
`__main__`, and one of them is actually run that way here rather than only
parsed.
"""
from __future__ import annotations

import ast
import pathlib
import subprocess
import sys

import pytest

RESEARCH = pathlib.Path(__file__).resolve().parent.parent / "research"


def _mains() -> list[str]:
    """Modules with a module-level `main()` -- the ones a `python -m` run enters.

    Module level only: a `main` nested inside a class or another function is
    not what `python -m research.<module>` calls, and counting one would let a
    module pass without being runnable.
    """
    found = []
    for path in sorted(RESEARCH.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if any(isinstance(node, ast.FunctionDef) and node.name == "main"
               for node in tree.body):
            found.append(path.stem)
    return found


MODULES = _mains()


def _guard(tree: ast.AST) -> ast.If | None:
    for node in tree.body:
        if not isinstance(node, ast.If):
            continue
        if ast.unparse(node.test) != "__name__ == '__main__'":
            continue
        return node
    return None


def test_the_sweep_finds_the_entry_points_it_was_written_against():
    # A detector broken into finding nothing passes every module vacuously --
    # the same failure as the per-module tests this replaces.
    assert {"daily_job", "supply_chain", "scanner"} <= set(MODULES), MODULES


@pytest.mark.parametrize("module", MODULES)
def test_every_module_with_a_main_enters_it_from_dunder_main(module):
    path = RESEARCH / f"{module}.py"
    node = _guard(ast.parse(path.read_text(encoding="utf-8")))

    assert node is not None, (
        f"`research/{module}.py` defines main() but has no "
        f"`if __name__ == '__main__':` guard, so `python -m research.{module}` "
        f"exits 0 having done nothing")
    # Raising SystemExit is what carries main()'s exit code out to the shell;
    # a bare `main()` call would drop it and every failure would read as 0.
    body = ast.unparse(node)
    assert "SystemExit" in body, (
        f"`research/{module}.py` calls main() from `__main__` without "
        f"SystemExit, so its non-zero exit code never reaches the shell: {body}")


def test_the_runbook_command_actually_runs(tmp_path):
    """The guard is proved by running the command, not by parsing for it.

    `--help` needs no network, no key and no store, and an unguarded module
    prints nothing at all -- which is exactly the symptom, so this catches the
    regression the AST check is standing in for.
    """
    done = subprocess.run(
        [sys.executable, "-m", "research.supply_chain", "--help"],
        cwd=RESEARCH.parent, capture_output=True, text=True, timeout=120,
        env={"PATH": "/usr/bin:/bin", "NEMO_PIT_DB": str(tmp_path / "pit.db")})

    assert done.returncode == 0, done.stderr
    assert "usage:" in done.stdout.lower(), (
        f"`python -m research.supply_chain --help` printed nothing: {done.stdout!r}")
