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
