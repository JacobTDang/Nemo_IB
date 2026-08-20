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
