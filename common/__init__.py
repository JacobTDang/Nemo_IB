"""Types small enough, and dependency-free enough, to live below everything.

One rule, and it is the reason this package can exist at all: **nothing in here
imports anything**. Not a third-party package, not `agent`, not `tools`, not
another module in this package.

That is what makes it importable from every layer at once. `agent/` and
`tools/` are kept apart on purpose -- `testing/test_agent_package_boundary.py`
exists to stop the LLM layer reaching a data-source image -- and a shared module
that imported something would smuggle that something across the boundary for
every consumer. A module that imports nothing cannot.

Keep it empty for the same reason `agent/__init__.py` is empty: a convenience
import at this level is paid for by every module that touches the package, and
these are touched by all of them.
"""
