"""What the shipped servers import must be what the image installs.

The homelab image installs only the `server` dependency group, and the
Dockerfile's build-time check imports each server to prove it loads. A module
imported inside a function is invisible to both: the build passes, the image
ships, and the ImportError arrives on the first real call.

`get_congress_trades` was written that way -- `pdfplumber` imported inside
`fetch_house_ptr`, `bs4` inside `parse_senate_ptr` -- and neither was in the
server group. Nothing would have failed until someone asked the deployed
server about a House filing.
"""
import ast
import pathlib
import sys
import tomllib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Modules the servers ship with, and the distribution that provides each.
# Anything imported but unmapped fails the test rather than passing quietly.
# The authoritative stdlib set, rather than a hand-kept list that drifts.
STDLIB = set(sys.stdlib_module_names)

MODULE_TO_DISTRIBUTION = {
    "aiohttp": "aiohttp", "bs4": "beautifulsoup4", "ddgs": "ddgs",
    "dotenv": "python-dotenv", "edgar": "edgartools", "httpx": "httpx",
    "mcp": "mcp", "numpy": "numpy", "pandas": "pandas",
    "pdfplumber": "pdfplumber", "requests": "requests",
    "trafilatura": "trafilatura", "yfinance": "yfinance",
}

# The packages the image copies in. Kept in step with the Dockerfile's COPY.
SHIPPED = ["tools/altdata_server", "tools/news_agregator",
           "tools/web_search_server", "tools/financial_modeling_engine"]


def _server_group() -> set:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    return {dep.split("==")[0].strip().lower()
            for dep in data["dependency-groups"]["server"]}


def _imports(path: pathlib.Path) -> set:
    """Every top-level module name imported, at module scope or inside a def."""
    tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:      # a relative import is our own code
                continue
            if node.module:
                found.add(node.module.split(".")[0])
    return found


def _source_files():
    for package in SHIPPED:
        for path in (ROOT / package).rglob("*.py"):
            if "__pycache__" not in str(path):
                yield path


@pytest.mark.parametrize("path", sorted(_source_files(), key=str),
                         ids=lambda p: str(p.relative_to(ROOT)))
def test_every_import_is_installed_in_the_image(path):
    declared = _server_group()
    local = {"tools", "agent", "state", "daemons", "data", "testing"}

    for module in sorted(_imports(path)):
        if module in STDLIB or module in local:
            continue
        distribution = MODULE_TO_DISTRIBUTION.get(module)
        assert distribution is not None, (
            f"{path.relative_to(ROOT)} imports {module!r}, which this test "
            f"does not know how to map to a distribution. Add it to "
            f"MODULE_TO_DISTRIBUTION so the check can tell whether the image "
            f"installs it.")
        assert distribution.lower() in declared, (
            f"{path.relative_to(ROOT)} imports {module!r} but "
            f"{distribution!r} is not in the `server` dependency group, so it "
            f"is absent from the homelab image. A lazy import makes this fail "
            f"on the first call rather than at build time.")
