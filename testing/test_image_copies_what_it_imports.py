"""Every shared module the servers import must be COPYed into the image.

`tools/` is copied selectively rather than wholesale, because `COPY tools/`
would ship alpaca, sentry and excel -- excluded from running but present on
disk, which on a LAN-reachable host is the difference between "no trading
tools" and "trading tools nobody happens to start". That is the right call.

Its cost is a hand-maintained list of the top-level modules, and that list has
now been wrong three times: `response_meta.py`, `manifest.py` and
`filing_cache.py` were each added, imported by every server, and left out of
the COPY. The failure is total and silent until runtime -- the image builds,
the container starts, and every server dies on ImportError.

So the list is checked against the imports rather than maintained beside them.
"""
import ast
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCKERFILE = ROOT / "Dockerfile"

# The packages the image copies wholesale. A module under one of these needs no
# COPY line of its own.
SHIPPED_PACKAGES = {"news_agregator", "web_search_server",
                    "financial_modeling_engine", "altdata_server",
                    "preearnings"}


def _top_level_modules_available() -> set:
    """Modules that live directly in tools/, as files rather than packages."""
    return {p.stem for p in (ROOT / "tools").glob("*.py")
            if p.stem != "__init__"}


def _copied_top_level() -> set:
    """The top-level tools/*.py the Dockerfile actually copies."""
    text = DOCKERFILE.read_text()
    copied = set()
    for line in text.splitlines():
        if not line.startswith("COPY "):
            continue
        for match in re.finditer(r"tools/([A-Za-z_][A-Za-z0-9_]*)\.py", line):
            copied.add(match.group(1))
    return copied


def _shipped_sources() -> list:
    """Every file that ends up in the image: the shipped packages, and the
    top-level modules the Dockerfile already copies.

    The top-level ones matter as much as the packages. `filing_cache` is
    imported by `tools/mcp_http.py`, so a scan that looked only inside the
    server packages would have declared the COPY list complete while the
    image could not start.
    """
    sources = []
    for package in SHIPPED_PACKAGES:
        sources.extend((ROOT / "tools" / package).rglob("*.py"))
    for module in _copied_top_level():
        path = ROOT / "tools" / f"{module}.py"
        if path.exists():
            sources.append(path)
    return sources


def _imported_top_level() -> dict:
    """{module: the file that imports it} for every tools/*.py the image
    reaches, from the servers and from the transport alike."""
    available = _top_level_modules_available()
    found = {}
    if True:
        for path in _shipped_sources():
            try:
                tree = ast.parse(path.read_text())
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.ImportFrom) and node.module:
                    if node.module == "tools":
                        names = [a.name for a in node.names]
                    elif node.module.startswith("tools."):
                        rest = node.module.split(".")
                        if len(rest) == 2:
                            names = [rest[1]]
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        parts = alias.name.split(".")
                        if len(parts) == 2 and parts[0] == "tools":
                            names.append(parts[1])
                for name in names:
                    if name in available:
                        found.setdefault(name, str(path.relative_to(ROOT)))
    return found


def test_the_dockerfile_copies_every_shared_module_the_servers_import():
    imported = _imported_top_level()
    copied = _copied_top_level()
    missing = {m: src for m, src in imported.items() if m not in copied}

    assert not missing, (
        "these modules are imported by a shipped server but never copied into "
        "the image, so every server dies on ImportError at container start: "
        + ", ".join(f"tools/{m}.py (imported by {src})"
                    for m, src in sorted(missing.items())))


def test_the_build_time_import_check_covers_the_transport():
    """The RUN python -c import check is what turns a missing COPY into a
    failed build rather than a failed deployment."""
    text = DOCKERFILE.read_text()
    assert "tools.mcp_http" in text, (
        "the build no longer verifies the HTTP transport imports, so a missing "
        "shared module would ship")


@pytest.mark.parametrize("module", sorted(_imported_top_level()))
def test_each_imported_module_is_named_in_the_copy_line(module):
    assert module in _copied_top_level(), (
        f"tools/{module}.py is imported by a shipped server and not copied")
