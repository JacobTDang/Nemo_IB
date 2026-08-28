"""`Secret` is a redaction primitive, so it gets exactly one definition.

Issue #17 introduced it in `agent/openrouter_template.py`. Issue #59 needed it
in five more modules and #60 in a seventh, and each of those decisions declined
to import the original for a reason that was correct at the time:

* `agent/groq_template.py` was imported *by* `openrouter_template` for
  `CredentialsMissing`, so importing back was a cycle.
* `agent/Execution_Agent.py` is a 112-module import; reaching
  `openrouter_template` cost 1146 in 0.23s, and the one module that talks to
  the broker should not load openai and ollama to reach a value class.
* `tools/alpaca_server/`, `tools/news_agregator/` and `tools/altdata_server/`
  are data sources. `testing/test_agent_package_boundary.py` exists to keep the
  LLM layer out of their image, and `openrouter_template` also called
  `sys.stdout.reconfigure()` at import -- on an MCP stdio server, stdout is the
  wire.

`agent/openrouter_template.py` itself was deleted with the LangGraph/OpenRouter
layer (issue #63), so the list below is one shorter. The rule it was the first
violation of is unchanged, and `agent/groq_template.py` still reconfigures
stdout at import, so the constraint on the shared home is exactly as tight.

Seven copies of a security-critical class is still seven chances for one of
them to drift, and the divergence would only ever show up as the leak. What all
seven objections have in common is that they are objections to importing *that
module*, not to importing the type: a module that imports nothing can be
imported from anywhere. `common/secret.py` is that module, and this file is
what holds it to the two properties that let it be.

Every credential here is a synthetic sentinel built in this file. Nothing reads
a real one -- a test that loads a live credential to prove it does not escape
is itself the escape.
"""
import ast
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
HOME = REPO / "common" / "secret.py"

# Where the class must not come back. Everything under these is either not ours
# or not shipped.
SKIP = {".venv", ".git", "__pycache__", ".pytest_cache", ".ruff_cache",
        "node_modules", "testing"}

# The modules that hold a credential and therefore name the type.
CONSUMERS = [
    "agent.groq_template",
    "agent.Execution_Agent",
    "tools.alpaca_server.alpaca_server",
    "tools.news_agregator.finnhub_utils",
    "tools.news_agregator.fred_utils",
    "tools.altdata_server.server",
]

_SENTINEL = "synthetic0000shared0000secret0000sentinel"
_BODY = "synthetic0000"


def _source_files():
    for path in REPO.rglob("*.py"):
        if not SKIP & set(path.relative_to(REPO).parts):
            yield path


def test_the_class_is_defined_in_exactly_one_place():
    """The whole point. Seven definitions cannot be kept identical by care."""
    defined = sorted(
        str(path.relative_to(REPO))
        for path in _source_files()
        if any(isinstance(node, ast.ClassDef) and node.name == "Secret"
               for node in ast.walk(ast.parse(
                   path.read_text(encoding="utf-8", errors="replace"))))
    )
    assert defined == ["common/secret.py"], (
        f"`Secret` is defined in {defined}. A redaction primitive with more "
        f"than one definition drifts, and the drift shows up as the leak.")


@pytest.mark.parametrize("module", CONSUMERS)
def test_every_module_that_holds_a_credential_uses_that_one(module):
    """Not "behaves the same as" -- is the same object.

    The mirrors were pinned to the original by comparing behaviour, which
    catches a drift only in the behaviour someone thought to compare. Identity
    has nothing left to compare.
    """
    import importlib

    from common.secret import Secret as Shared

    carried = getattr(importlib.import_module(module), "Secret", None)
    assert carried is not None, f"{module} no longer names Secret"
    assert carried is Shared, (
        f"{module} carries its own Secret again; import it from common.secret")


# --------------------------------------------------------------------------
# The two properties that let it be imported from anywhere
# --------------------------------------------------------------------------

@pytest.mark.parametrize("relative", ["common/__init__.py", "common/secret.py"])
def test_the_shared_module_imports_nothing(relative):
    """This is the constraint the seven mirrors were working around.

    A shared home that imported anything would put that thing into every image
    that reaches for a credential -- which is how the LLM layer got into a
    data-source image the first time. Importing nothing at all is a stronger
    guarantee than importing only safe things, and it is checkable.
    """
    tree = ast.parse((REPO / relative).read_text())
    imports = sorted(
        ast.unparse(node) for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom)))
    assert not imports, (
        f"{relative} imports {imports}. Every module that holds a credential "
        f"imports this one, so whatever it imports they all import.")


def test_importing_it_loads_nothing_that_is_not_already_there():
    """Measured rather than reasoned: import it in a clean interpreter."""
    code = ("import sys\n"
            "before = set(sys.modules)\n"
            "import common.secret\n"
            "print('\\n'.join(sorted(set(sys.modules) - before)))\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, cwd=str(REPO))
    assert out.returncode == 0, out.stderr[-800:]
    loaded = set(out.stdout.split()) - {"common", "common.secret"}
    assert not loaded, (
        f"importing common.secret pulled in {sorted(loaded)}; it is imported "
        f"by five data-source modules and must cost them nothing")


def test_it_never_reaches_for_the_transport():
    """`agent/groq_template.py` calls `sys.stdout.reconfigure()` as a
    side effect of being imported. `tools/altdata_server/server.py` is an MCP
    **stdio** server: stdout is its wire, and a credential helper that writes
    to it -- or reconfigures it -- corrupts the protocol rather than the log.

    Read off the syntax tree rather than the text, so the paragraph above
    explaining the hazard does not itself count as one.
    """
    forbidden = {"sys", "print", "logging", "warnings", "input", "breakpoint"}
    tree = ast.parse(HOME.read_text())
    reached = sorted({node.id for node in ast.walk(tree)
                      if isinstance(node, ast.Name) and node.id in forbidden})
    assert not reached, (
        f"common/secret.py reaches {reached}; it is imported by an MCP stdio "
        f"server whose stdout is the protocol, and by a broker agent whose "
        f"credential must not reach a log either")


# --------------------------------------------------------------------------
# The behaviour every mirror was tested for, now tested once
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "render",
    [repr, str, lambda s: f"{s}", lambda s: f"{s!r}", lambda s: f"{s!s}",
     lambda s: "{}".format(s),                       # noqa: UP032 - the point
     lambda s: "%s" % (s,), lambda s: str([s]), lambda s: str({"key": s})],
    ids=["repr", "str", "fstring", "fstring_r", "fstring_s", "format",
         "percent", "in_list", "in_dict"],
)
def test_no_way_of_rendering_it_shows_the_value(render):
    from common.secret import Secret

    rendered = render(Secret(_SENTINEL))
    assert _BODY not in rendered, "part of the value survived rendering"


def test_it_still_reveals_the_value_when_asked():
    """Redaction that also breaks authentication is not a fix."""
    from common.secret import Secret

    assert Secret(_SENTINEL).reveal() == _SENTINEL


def test_an_unset_credential_reads_as_absent():
    from common.secret import Secret

    assert not Secret("")
    assert Secret(_SENTINEL)


def test_it_scrubs_the_value_out_of_provider_text():
    from common.secret import Secret

    echoed = f"401 Unauthorized: token={_SENTINEL} was rejected"
    scrubbed = Secret(_SENTINEL).scrub(echoed)
    assert _BODY not in scrubbed
    assert "401 Unauthorized" in scrubbed, "scrubbing ate the diagnosis"


def test_the_instance_carries_no_dict_to_render():
    """`__slots__` is not a micro-optimisation here. Without it the value is in
    `vars(instance)`, which a debugger, a crash reporter and `--showlocals` all
    render for an object whose own `__repr__` says nothing."""
    from common.secret import Secret

    with pytest.raises(TypeError):
        vars(Secret(_SENTINEL))
