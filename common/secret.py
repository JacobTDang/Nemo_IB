"""The credential wrapper, in one place.

`Secret` holds a credential where nothing can render it and hands it back only
to code that asks for it by name.

**Why the value has to be hidden rather than the printers fixed.** pytest
prints a frame's arguments at the head of every traceback entry, and every
local under `--showlocals`. So a key sitting in a parameter or a variable is
written to stdout by the first test that fails anywhere below it: the live
OpenRouter key went into CI logs, pasted terminals and captured artefacts that
way (issue #17), and issue #59 found the same shape around the *live Alpaca*
credentials, which can move money. Suppressing frame rendering in pytest
configuration would have fixed pytest and nothing else -- a log line, a
debugger, a crash reporter and a print written next year are the same
disclosure by another route. Keeping the value behind `reveal()` leaves nothing
renderable to render, which closes all of them at once.

**Why this module imports nothing.** It is imported by seven modules that do
not otherwise share a layer: the two LLM templates, the broker agent, and four
data-source servers whose image deliberately excludes the LLM stack
(`testing/test_agent_package_boundary.py`). Anything imported here would be
imported by all of them. `tools/altdata_server/server.py` adds a second, harder
constraint: it is an MCP **stdio** server, so stdout is its wire -- and the
module this class used to live in calls `sys.stdout.reconfigure()` as a side
effect of being imported. Nothing here touches the transport, logs, or prints.

Those two properties are what let the seven copies this replaced become one;
`testing/test_the_credential_type_has_one_home.py` holds them.
"""


class Secret:
    """A credential that renders as a placeholder instead of as itself.

    Use it at the read -- `Secret(os.getenv("FINNHUB_API_KEY") or "")` -- so the
    value never exists as a bare string in a frame at all. Call `reveal()` at
    the point of use (an SDK constructor, a request header, a query parameter)
    and never bind the result to a name, or the value is back in a frame.
    """

    # No instance dict, so there is nothing for a debugger, a crash reporter or
    # --showlocals to render for an object whose own __repr__ says nothing.
    __slots__ = ("_value",)

    PLACEHOLDER = "<redacted>"

    def __init__(self, value: str = ""):
        self._value = value or ""

    def reveal(self) -> str:
        """The raw credential."""
        return self._value

    def scrub(self, text: str) -> str:
        """`text` with the credential replaced by the placeholder.

        Provider error bodies are printed on every retry and, in the data
        servers, returned to the MCP caller in an `error` field -- which leaves
        the process entirely rather than merely reaching a log. A provider that
        echoes the offending credential back, or an SDK that quotes the request
        URL the key was a query parameter of, would otherwise put it there.
        """
        if not self._value:
            return text
        return text.replace(self._value, self.PLACEHOLDER)

    def __repr__(self) -> str:
        return self.PLACEHOLDER

    __str__ = __repr__

    def __bool__(self) -> bool:
        """Callers test the credential for presence; an unset key reads absent."""
        return bool(self._value)
