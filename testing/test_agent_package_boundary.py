"""Importing a utility from `agent` must not drag in the LLM layer.

`agent/__init__.py` was a single line -- `from .News_Processing_Agent import
News_Processing_Agent` -- so any server importing `agent.cache` also loaded
News_Processing_Agent, groq_template and openai. Nothing consumed that
re-export; the real consumers use submodule imports like
`from agent import sentry_budget`, which work regardless.

The cost was not just startup time. It put the whole orchestration layer, and
transitively LangGraph and LangChain, into a data-source image that will never
run any of it.
"""
import subprocess
import sys


def _loaded_after(import_line: str) -> set:
    """Import something in a clean interpreter, report the modules it pulled in."""
    code = (
        "import sys\n"
        "before = set(sys.modules)\n"
        f"{import_line}\n"
        "print('\\n'.join(sorted(set(sys.modules) - before)))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, cwd=".")
    assert out.returncode == 0, out.stderr[-800:]
    return set(out.stdout.split())


def test_importing_the_cache_does_not_load_the_llm_layer():
    loaded = _loaded_after("import agent.cache")
    llm = {m for m in loaded
           if "groq_template" in m or "openrouter_template" in m
           or "_Agent" in m or m.startswith("langgraph") or m.startswith("langchain")}
    assert not llm, f"agent.cache pulled in the LLM layer: {sorted(llm)}"


def test_importing_the_cache_does_not_load_langgraph():
    """LangGraph is only ever used by agent/workflows/analysis_workflow.py."""
    loaded = _loaded_after("import agent.cache")
    assert not {m for m in loaded if m.split(".")[0] in ("langgraph", "langchain")}


def test_submodule_imports_still_work():
    """The pattern real consumers use must keep working."""
    loaded = _loaded_after("from agent import cache")
    assert any(m == "agent.cache" for m in loaded)


# --- the shared credential type, which crosses the boundary on purpose ------
#
# `Secret` was copied into seven modules rather than imported, and the reason
# given for four of them was this file: importing `agent.openrouter_template`
# for a value class would have put openai, ollama and httpx into a data-source
# image that runs none of them. Issue #61 removed the copies by removing the
# reason -- `common/secret.py` imports nothing, so importing it imports
# nothing. That claim is the one this file is here to check, and it is checked
# the same way: in a clean interpreter, by measurement.

def test_the_shared_credential_type_imports_nothing_at_all():
    """Not "nothing heavy" -- nothing. It is imported by every module that
    holds a credential, on both sides of the boundary, so whatever it pulled in
    they would all pull in."""
    loaded = _loaded_after("import common.secret")
    assert loaded <= {"common", "common.secret"}, (
        f"common.secret pulled in {sorted(loaded - {'common', 'common.secret'})}")


def test_a_data_source_module_reaching_it_still_loads_no_llm_layer():
    """The property the copies existed to preserve, now that they are gone."""
    for module in ("tools.news_agregator.fred_utils",
                   "tools.news_agregator.finnhub_utils",
                   "tools.altdata_server.server"):
        loaded = _loaded_after(f"import {module}")
        llm = {m for m in loaded
               if m.split(".")[0] in ("openai", "ollama", "langgraph",
                                      "langchain")
               or "groq_template" in m or "openrouter_template" in m
               or m.endswith("_Agent")}
        assert not llm, f"{module} now pulls in the LLM layer: {sorted(llm)}"


# --- what issue #63 makes checkable that was not checkable before -----------
#
# Until the LangGraph/OpenRouter layer was retired, "the LLM stack is out of
# the data-source image" was a property of one COPY list in the Dockerfile:
# langgraph and langchain were installed in the project venv, and one module
# -- agent/workflows/analysis_workflow.py -- imported them. The tests above
# measure the blast radius of that import.
#
# The layer is gone, so the property is now absolute rather than local: not
# "agent.cache does not reach LangGraph" but "nothing does, and it is not
# installable". Those are the assertions below. They are cheap, and they are
# what stops the stack coming back one convenient import at a time.

import ast
import pathlib

_REPO = pathlib.Path(__file__).resolve().parent.parent

_RETIRED = ("langgraph", "langchain", "langchain_core", "langchain_community",
            "langchain_classic", "langchain_text_splitters", "langgraph_sdk",
            "langgraph_checkpoint", "langgraph_prebuilt", "langsmith")

_SKIP_DIRS = {".venv", ".git", "__pycache__", ".pytest_cache", ".ruff_cache",
              "node_modules", ".superpowers"}


def _python_sources():
    for path in _REPO.rglob("*.py"):
        if not _SKIP_DIRS & set(path.relative_to(_REPO).parts):
            yield path


def test_nothing_in_the_tree_imports_the_retired_orchestration_stack():
    """Read off the syntax tree, not the text, so the prose above -- and the
    package names in the assertions further up this file -- do not count as
    imports of the thing they are about."""
    offenders = []
    for path in _python_sources():
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module] if node.module else []
            else:
                continue
            for name in names:
                if name.split(".")[0] in _RETIRED:
                    offenders.append(
                        f"{path.relative_to(_REPO)}:{node.lineno}: {name}")

    assert not offenders, (
        "the LangGraph/LangChain stack is imported again by:\n  "
        + "\n  ".join(offenders)
        + "\nIt was retired with the orchestration layer (issue #63) and is no "
          "longer a project dependency, so this import cannot resolve in a "
          "clean environment.")


_RETIRED_DISTS = {
    "langchain-classic", "langchain-community", "langchain-core",
    "langchain-text-splitters", "langgraph", "langgraph-checkpoint",
    "langgraph-prebuilt", "langgraph-sdk", "langsmith",
}


def test_the_retired_stack_is_not_a_project_dependency():
    """An import that cannot resolve is caught by the test above only on a
    machine where the package is genuinely absent. This one is true everywhere:
    it reads the manifests, so a re-added pin fails here even in a venv that
    still happens to carry the wheels."""
    pinned = []

    pyproject = (_REPO / "pyproject.toml").read_text(encoding="utf-8")
    for line in pyproject.splitlines():
        stripped = line.strip().lstrip('"').strip()
        if stripped.split("==")[0].replace("_", "-").lower() in _RETIRED_DISTS:
            pinned.append(f"pyproject.toml: {line.strip()}")

    # requirements.txt is UTF-16 on this repo; decoding it as UTF-8 silently
    # produces one unsplittable line, which would pass this test by matching
    # nothing at all.
    raw = (_REPO / "requirements.txt").read_bytes()
    requirements = raw.decode("utf-16" if raw[:2] in (b"\xff\xfe", b"\xfe\xff")
                              else "utf-8")
    assert len(requirements.splitlines()) > 50, (
        "requirements.txt decoded to almost nothing, so the scan below is "
        "vacuous -- check the file encoding")
    for line in requirements.splitlines():
        if line.strip().split("==")[0].replace("_", "-").lower() in _RETIRED_DISTS:
            pinned.append(f"requirements.txt: {line.strip()}")

    assert not pinned, (
        "the retired orchestration stack is pinned again:\n  "
        + "\n  ".join(pinned))


# --- the image's own agent/ list, checked rather than commented -------------

def _agent_modules_the_dockerfile_ships():
    """The agent modules `COPY agent/... ./agent/` puts in the image."""
    shipped = set()
    for line in (_REPO / "Dockerfile").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("COPY agent/"):
            continue
        source = line.split()[1]                      # agent/cache.py
        stem = source[len("agent/"):].removesuffix(".py").rstrip("/")
        shipped.add("agent" if stem == "__init__" else f"agent.{stem}")
    return shipped


def test_the_dockerfile_still_names_the_agent_modules_it_copies():
    """Guard on the parser: a COPY list this test reads as empty would make
    the assertion below trivially true."""
    shipped = _agent_modules_the_dockerfile_ships()
    assert "agent" in shipped and "agent.cache" in shipped, (
        f"the Dockerfile's agent COPY list parsed as {sorted(shipped)}; the "
        f"check below cannot be trusted")


def test_a_shipped_server_imports_no_agent_module_the_image_leaves_behind():
    """The Dockerfile copies four files out of agent/ rather than the package.

    A server that grows a module-scope `from agent.<x> import ...` for an `x`
    outside that list imports fine here and fails at container start, where the
    file simply is not there. The comment above the COPY lines says this was
    "verified by importing all five servers and recording what loads" -- which
    was true when it was written and is a one-time act. This does it on every
    run.

    Module scope only: the RAG tools reach `agent.rag` from inside a function
    and are capability-gated on it being importable, which is exactly how a
    module the image deliberately omits is supposed to be reached.
    """
    servers = ("tools.news_agregator.fred_server",
               "tools.news_agregator.finnhub_server",
               "tools.web_search_server.web_search",
               "tools.financial_modeling_engine.analysis_tools",
               "tools.altdata_server.server")
    shipped = _agent_modules_the_dockerfile_ships()

    for server in servers:
        loaded = _loaded_after(f"import {server}")
        reached = {m for m in loaded if m == "agent" or m.startswith("agent.")}
        missing = reached - shipped
        assert not missing, (
            f"{server} imports {sorted(missing)} at module scope, which the "
            f"Dockerfile does not COPY into the image. Either add it to the "
            f"COPY list or move the import inside the function that needs it.")
