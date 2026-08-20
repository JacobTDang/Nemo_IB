"""Guards the manifest drift that broke RAG under uv sync.

The D5 rule from the truth-source-refactor design spec is "apply every
removal to BOTH manifests" -- pyproject.toml AND requirements.txt. Every
invariant here is therefore checked against both files.
"""
from __future__ import annotations

import os
import re

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _pyproject_text() -> str:
    return open(os.path.join(_REPO, "pyproject.toml"), encoding="utf-8").read()


def _pyproject_deps() -> set:
    text = _pyproject_text()
    block = text.split("dependencies = [", 1)[1].split("]", 1)[0]
    return {m.split("==")[0].strip().lower()
            for m in re.findall(r'"([^"]+)"', block)}


def _requirements_text() -> str:
    # requirements.txt is UTF-16LE with a BOM and CRLF line endings (uv export
    # default on this project's Windows-originated lockfile). Decoding as
    # plain utf-8 would garble or fail outright.
    raw = open(os.path.join(_REPO, "requirements.txt"), "rb").read()
    return raw.decode("utf-16")


def _requirements_deps() -> set:
    text = _requirements_text()
    deps = set()
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        deps.add(line.split("==")[0].strip().lower())
    return deps


def test_sentence_transformers_is_declared():
    # agent/rag/embedder.py imports it; without the pin, uv sync produces an
    # environment where the first rag_search fails.
    assert "sentence-transformers" in _pyproject_deps()
    assert "sentence-transformers" in _requirements_deps()


def test_no_openbb_pins_remain():
    assert not [d for d in _pyproject_deps() if d.startswith("openbb")]
    assert not [d for d in _requirements_deps() if d.startswith("openbb")]


def test_no_cuda_pinned_torch():
    assert "+cu121" not in _pyproject_text()
    assert "pytorch-cu121" not in _pyproject_text()
    assert "+cu121" not in _requirements_text()
    assert "pytorch-cu121" not in _requirements_text()


def test_removed_direct_deps_are_gone():
    pyproject_deps = _pyproject_deps()
    requirements_deps = _requirements_deps()
    for pkg in ("accelerate", "bitsandbytes"):
        assert pkg not in pyproject_deps, f"{pkg} still pinned in pyproject.toml but has no consumer"
        assert pkg not in requirements_deps, f"{pkg} still pinned in requirements.txt but has no consumer"


def test_no_direct_transformers_pin():
    # transformers has no direct importer anywhere in the repo (grep -rn
    # "import transformers" returns nothing) -- it should only be present as
    # a transitive dependency of sentence-transformers, i.e. installed but
    # not directly pinned in either manifest.
    assert "transformers" not in _pyproject_deps()
    assert "transformers" not in _requirements_deps()
