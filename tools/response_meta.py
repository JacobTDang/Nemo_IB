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
