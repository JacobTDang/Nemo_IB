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
