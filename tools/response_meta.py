"""Additive provenance metadata for MCP tool responses."""

import copy
from datetime import datetime, timezone


SCHEMA_VERSION = "1"
_ALLOWED_COVERAGE = frozenset({"full", "partial", "not_covered", "unknown"})


def warning(code: str, message: str, **extra) -> dict:
    """Build a structured warning that callers can inspect programmatically."""
    return {"code": code, "message": message, **extra}


def _validate_warnings(entries) -> None:
    if not isinstance(entries, list):
        raise TypeError(
            f"warnings must be a list of dicts, received {type(entries).__name__}"
        )
    for entry in entries:
        if not isinstance(entry, dict):
            raise TypeError(
                "warning entries must be dicts with code and message, "
                f"received {type(entry).__name__}"
            )
        if "code" not in entry or "message" not in entry:
            raise TypeError("warning entries must contain code and message")


def _validate_coverage(value) -> None:
    if value not in _ALLOWED_COVERAGE:
        allowed = ", ".join(sorted(_ALLOWED_COVERAGE))
        raise ValueError(f"invalid coverage {value!r}; allowed values: {allowed}")


def annotate(
    payload,
    *,
    provider,
    source_url=None,
    data_as_of=None,
    cached=False,
    coverage=None,
    warnings=None,
    success=None,
):
    """Return ``payload`` with provenance added, mutating nothing else."""
    if not isinstance(payload, dict):
        raise TypeError(f"payload must be a dict, received {type(payload).__name__}")

    # Deep, not shallow. A shallow copy leaves `result["data"]` as the same
    # object the tool returned, so a caller editing the annotated response
    # rewrites the original -- and, behind a cache, the stored entry with it.
    result = copy.deepcopy(payload)

    effective_coverage = result.get(
        "coverage", "unknown" if coverage is None else coverage
    )
    _validate_coverage(effective_coverage)

    existing_warnings = result.get("warnings", [])
    _validate_warnings(existing_warnings)
    supplied_warnings = [] if warnings is None else warnings
    _validate_warnings(supplied_warnings)

    if "success" not in result:
        if success is not None:
            result["success"] = success
        else:
            result["success"] = not bool(result.get("error"))

    defaults = {
        "provider": provider,
        "source_url": source_url,
        "data_as_of": data_as_of,
        "retrieved_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "cached": cached,
        "coverage": effective_coverage,
        "schema_version": SCHEMA_VERSION,
    }
    for key, value in defaults.items():
        result.setdefault(key, value)

    if warnings is not None:
        # Append only what is not already there. Annotation happens at more
        # than one boundary -- a helper and again at dispatch -- and stacking
        # the same caveat twice would have a reader counting one staleness
        # warning as two.
        merged = list(existing_warnings)
        for entry in supplied_warnings:
            if entry not in merged:
                merged.append(entry)
        result["warnings"] = merged
    else:
        result.setdefault("warnings", [])

    return result


# --------------------------------------------------------------- the seam

# Keys that unambiguously name the period the DATA describes. Deliberately
# short. `timestamp` is when we asked, `filing_date` is when a document was
# filed, `date` could be either -- promoting any of them would hand a caller
# confident provenance that is wrong, and a caller comparing two observations
# on a bad `data_as_of` compares different periods believing they match.
_PERIOD_KEYS = ("data_as_of", "period_end", "as_of")

_SOURCE_URL_KEYS = ("source_url", "filing_url", "url")


def _first_present(payload: dict, keys) -> "object | None":
    for key in keys:
        value = payload.get(key)
        if value:
            return value
    return None


def _record_success() -> None:
    """Tell /ready that this server just answered something successfully.

    Imported here rather than at module scope on purpose. Every server imports
    `response_meta`, `tools.mcp_http` imports uvicorn, starlette and the MCP
    session manager, and a server that never serves over HTTP still wants
    provenance -- a module-level import would make all of that a hard
    dependency of annotation and invite an import cycle.

    Every failure is swallowed, which this module otherwise never does. The
    distinction is that this is observability: a missing `mcp_http`, or a clock
    error inside `record_success`, must cost the readiness timestamp and
    nothing else. Turning it into a tool failure would be a strictly worse bug
    than the one it fixes.
    """
    try:
        import importlib

        # import_module, not `from tools import mcp_http`: it consults
        # sys.modules first, so a test that makes the module unavailable
        # actually makes it unavailable rather than reaching a parent
        # package attribute left over from an earlier import.
        mcp_http = importlib.import_module("tools.mcp_http")
        mcp_http.record_success()
    except Exception:  # noqa: BLE001 -- telemetry must never fail a tool call
        pass


def annotating(provider, *, per_tool=None, warnings_per_tool=None):
    """Wrap an MCP `call_tool(name, args)` so every response carries provenance.

    Applied at the dispatcher rather than in each tool. `web_search` alone
    returns TextContent from fifty places; editing each is fifty chances to
    differ and one to be forgotten, and a tool added later would ship without
    provenance and nothing would say so.

    `per_tool` overrides the provider where a server reads more than one
    upstream -- altdata answers from the House Clerk, the Senate eFD,
    USAspending, GovTrack, FinMind and public ATS boards, and calling all of
    that "altdata" would name the aggregator instead of the source.

    Anything that is not a JSON object is passed through untouched: a list has
    no place to put provenance without changing its shape, and inventing one
    here would be the guess this module exists to avoid.

    The same seam records the successful call that /ready reports. Nothing else
    sees every tool's result, so anywhere else is 96 places to remember and one
    to forget -- and readiness on all five servers currently reads false on a
    working stack precisely because nobody called `record_success()`.
    """
    import functools
    import json

    per_tool = per_tool or {}
    warnings_per_tool = warnings_per_tool or {}

    def decorate(handler):
        @functools.wraps(handler)
        async def wrapper(name, args, *rest, **kwargs):
            contents = await handler(name, args, *rest, **kwargs)
            if not contents:
                # No content at all is not an answer, so it records nothing.
                # A raised handler never reaches here either: the exception
                # propagates untouched, which is the honest report.
                return contents

            resolved = per_tool.get(name, provider)
            extra = warnings_per_tool.get(name)

            # Whether this call answered the question, for /ready. A response
            # that could not carry provenance -- prose, a JSON list -- still
            # completed, so only an explicit `success: false` counts against
            # it. "Could not annotate" is not "failed".
            succeeded = True

            for item in contents:
                text = getattr(item, "text", None)
                if not text:
                    continue
                try:
                    payload = json.loads(text)
                except (ValueError, TypeError):
                    continue                      # prose, not a response body
                if not isinstance(payload, dict):
                    continue                      # a list has nowhere to put it

                annotated = annotate(
                    payload,
                    provider=resolved,
                    source_url=_first_present(payload, _SOURCE_URL_KEYS),
                    data_as_of=_first_present(payload, _PERIOD_KEYS),
                    warnings=list(extra) if extra else None,
                )
                # annotate() has already resolved success, including inferring
                # it from an `error` field, so this reads one answer rather
                # than a second opinion that could disagree with the body the
                # caller receives.
                if annotated.get("success") is False:
                    succeeded = False
                item.text = json.dumps(annotated, default=str)

            # Once per call, not once per content block: /ready reads a
            # timestamp, and a server answering nothing but errors must keep
            # reporting itself unready -- that is the entire question the
            # check asks.
            if succeeded:
                _record_success()
            return contents

        # Exposed so a manifest can read what responses actually report rather
        # than restating it. Two lists of providers drift; one does not.
        wrapper.annotation_config = {
            "provider": provider,
            "per_tool": dict(per_tool),
            "warnings_per_tool": {k: list(v) for k, v in warnings_per_tool.items()},
        }
        return wrapper

    return decorate
