"""Deterministic pre-earnings review checks (/preearnings-review).

The reviewer is a strict, read-only auditor that runs AFTER /preearnings-research
and BEFORE acting on the prediction. It inspects the persisted research layers +
eval row — it gathers no new research and never mutates the prediction.

Severity model: any FAIL -> not_actionable; any WARN -> sound_with_warnings;
all PASS -> sound. Pure functions; nothing company-specific hardcoded.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# Components whose claims drive the DIRECTION score — uncited numbers here are
# disqualifying, not just messy.
_DIRECTION_CRITICAL_PREFIXES = ("peer_readthrough", "guidance", "kpi")

# Markers in layer payloads that indicate degraded/stale inputs.
_STALE_MARKERS = ("quotes_stale", "suspect", "sentinel", "data_gap", "stale")


def _r(check: str, status: str, detail: str, fix: Optional[str] = None) -> Dict[str, Any]:
    return {"check": check, "status": status, "detail": detail, "fix": fix}


def _is_direction_critical(component: str) -> bool:
    return any(component.startswith(p) for p in _DIRECTION_CRITICAL_PREFIXES)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_citations(layers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Every layer must carry sources; direction-critical layers with no
    citations are disqualifying."""
    out = []
    for layer in layers:
        comp = layer.get("component", "")
        if comp in ("synthesis", "review"):
            continue
        sources = layer.get("sources") or []
        valid = [s for s in sources if s.get("claim") and s.get("tool")]
        if not valid:
            sev = "fail" if _is_direction_critical(comp) else "warn"
            out.append(_r("citations", sev,
                          f"{comp}: no cited claims",
                          f"re-run {comp} with the cite-or-drop contract"))
        elif len(valid) < len(sources):
            out.append(_r("citations", "warn",
                          f"{comp}: {len(sources) - len(valid)} malformed source entries",
                          f"fix source entries on {comp}"))
    if not out:
        out.append(_r("citations", "pass", "all layers carry cited claims"))
    return out


def check_freshness(layers: List[Dict[str, Any]],
                    now: Optional[datetime] = None,
                    max_age_hours: float = 24.0) -> List[Dict[str, Any]]:
    """Components must be recent; very stale (3x window) is disqualifying."""
    now = now or datetime.now(timezone.utc)
    out = []
    for layer in layers:
        comp = layer.get("component", "")
        if comp == "review":
            continue
        created = layer.get("created_at")
        try:
            dt = datetime.fromisoformat(str(created))
        except (TypeError, ValueError):
            out.append(_r("freshness", "warn", f"{comp}: unparseable created_at",
                          f"re-persist {comp}"))
            continue
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        age_h = (now - dt).total_seconds() / 3600.0
        if age_h > 3 * max_age_hours:
            out.append(_r("freshness", "fail",
                          f"{comp}: {age_h:.0f}h old (> {3 * max_age_hours:.0f}h)",
                          f"re-run {comp} before acting"))
        elif age_h > max_age_hours:
            out.append(_r("freshness", "warn",
                          f"{comp}: {age_h:.0f}h old (> {max_age_hours:.0f}h)",
                          f"refresh {comp}"))
    if not out:
        out.append(_r("freshness", "pass", "all components within the freshness window"))
    return out


def check_completeness(layers: List[Dict[str, Any]],
                       eval_row: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The research must actually be assembled: synthesis + eval row are
    required; direction and asymmetry components missing -> warn with fix."""
    out = []
    comps = {l.get("component", "") for l in layers}
    if "synthesis" not in comps:
        out.append(_r("completeness", "fail", "no synthesis layer persisted",
                      "run /preearnings-research Layer 3"))
    if eval_row is None:
        out.append(_r("completeness", "fail", "no prediction row in preearnings_evals",
                      "run /preearnings-research (record_eval)"))
    for comp, label in (("peer_readthrough", "peer readthrough"),
                        ("guidance", "guidance archaeology"),
                        ("positioning", "positioning")):
        if comp not in comps:
            out.append(_r("completeness", "warn", f"{label} component missing",
                          f"run the {label} step (or record it as data_gap)"))
    if not any(c.startswith("kpi") for c in comps):
        out.append(_r("completeness", "warn", "no KPI component persisted",
                      "run the KPI drill-down"))
    if "reaction" not in comps:
        out.append(_r("completeness", "warn", "reaction profile not computed",
                      "build pairs via pair_surprises_with_reactions and persist"))
    if not out:
        out.append(_r("completeness", "pass", "all expected components present"))
    return out


def check_contradictions(layers: List[Dict[str, Any]],
                         min_magnitude: float = 0.4) -> List[Dict[str, Any]]:
    """High-magnitude components pointing opposite ways must be NAMED — a
    weighted average that quietly nets them out hides the real disagreement."""
    bulls, bears = [], []
    for layer in layers:
        comp = layer.get("component", "")
        if comp in ("synthesis", "review"):
            continue
        mag = layer.get("magnitude")
        if mag is None or float(mag) < min_magnitude:
            continue
        if layer.get("direction") == "bullish":
            bulls.append(comp)
        elif layer.get("direction") == "bearish":
            bears.append(comp)
    if bulls and bears:
        return [_r("contradictions", "warn",
                   f"high-magnitude disagreement: bullish={bulls} vs bearish={bears}",
                   "name the disagreement in the verdict; do not average it away")]
    return [_r("contradictions", "pass", "no high-magnitude opposing signals")]


def check_stale_flags(layers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Degraded-input markers inside payloads must be acknowledged."""
    out = []
    for layer in layers:
        comp = layer.get("component", "")
        if comp == "review":
            continue
        blob = json.dumps(layer.get("payload", {}), default=str).lower()
        hits = sorted({m for m in _STALE_MARKERS if m in blob})
        if hits:
            out.append(_r("stale_flags", "warn",
                          f"{comp}: degraded-input markers {hits}",
                          f"acknowledge {comp} data quality in the verdict"))
    if not out:
        out.append(_r("stale_flags", "pass", "no degraded-input markers"))
    return out


def check_hard_rules(synthesis_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The persisted verdict must obey its own discipline invariants."""
    out = []
    p = synthesis_payload or {}
    pred = p.get("prediction")
    conf = p.get("confidence")
    sizing = p.get("sizing")
    cov = p.get("coverage")
    implied = p.get("implied_move_pct")
    low_conf = p.get("low_confidence")

    if pred == "in_line" and sizing != "no_position":
        out.append(_r("hard_rules", "fail",
                      f"in_line prediction with sizing={sizing}",
                      "in_line must be no_position"))
    if conf is not None and conf < 0.55 and sizing not in (None, "no_position"):
        out.append(_r("hard_rules", "fail",
                      f"confidence {conf} < 0.55 with sizing={sizing}",
                      "confidence < 0.55 must be no_position"))
    if implied is not None and implied > 0.20 and sizing not in (None, "no_position"):
        out.append(_r("hard_rules", "fail",
                      f"implied move {implied} > 0.20 with sizing={sizing}",
                      "binary-event discipline: no_position"))
    if (cov is not None and conf is not None and low_conf is not None
            and bool(low_conf) != bool(cov < 0.5 or conf < 0.5)):
        out.append(_r("hard_rules", "fail",
                      f"low_confidence={low_conf} inconsistent with coverage={cov}, confidence={conf}",
                      "recompute low_confidence"))
    if not out:
        out.append(_r("hard_rules", "pass", "verdict obeys discipline invariants"))
    return out


def check_db_consistency(eval_row: Optional[Dict[str, Any]],
                         synthesis_payload: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The eval row must match the latest synthesis — a stale prediction row
    means the scorecard would grade the wrong call."""
    if not eval_row or not synthesis_payload:
        return [_r("db_consistency", "warn", "cannot compare (missing row or synthesis)")]
    mismatches = []
    if eval_row.get("prediction") != synthesis_payload.get("prediction"):
        mismatches.append(
            f"prediction: eval={eval_row.get('prediction')} vs synthesis={synthesis_payload.get('prediction')}")
    ec, sc = eval_row.get("confidence"), synthesis_payload.get("confidence")
    if ec is not None and sc is not None and abs(float(ec) - float(sc)) > 0.005:
        mismatches.append(f"confidence: eval={ec} vs synthesis={sc}")
    if mismatches:
        return [_r("db_consistency", "fail", "; ".join(mismatches),
                   "re-run record_eval so the row matches the latest synthesis")]
    return [_r("db_consistency", "pass", "eval row matches latest synthesis")]


def check_estimate_dispersion(eps_a: Optional[float], eps_b: Optional[float],
                              label_a: str = "source_a", label_b: str = "source_b",
                              tol: float = 0.01) -> List[Dict[str, Any]]:
    """Different vendors can disagree on what consensus even IS. A >tol relative
    gap changes the bar the company must clear — it must be surfaced."""
    if eps_a is None or eps_b is None:
        return [_r("estimate_dispersion", "warn",
                   "could not compare consensus across sources",
                   "fetch a second consensus source")]
    base = min(abs(eps_a), abs(eps_b))
    if base == 0:
        return [_r("estimate_dispersion", "warn", "zero-consensus comparison skipped")]
    rel = abs(eps_a - eps_b) / base
    if rel > tol:
        return [_r("estimate_dispersion", "warn",
                   f"{label_a}={eps_a} vs {label_b}={eps_b} ({rel * 100:.1f}% apart) — "
                   f"the bar differs by source",
                   "state which consensus the prediction is scored against")]
    return [_r("estimate_dispersion", "pass",
               f"{label_a}={eps_a} vs {label_b}={eps_b} within {tol * 100:.0f}%")]


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

def run_review(
    layers: List[Dict[str, Any]],
    eval_row: Optional[Dict[str, Any]],
    dispersion: Optional[Dict[str, Any]] = None,
    now: Optional[datetime] = None,
    max_age_hours: float = 24.0,
) -> Dict[str, Any]:
    """Run every check and aggregate. dispersion (optional):
    {eps_a, eps_b, label_a, label_b} fetched live by the orchestrating skill."""
    synthesis = next((l for l in layers if l.get("component") == "synthesis"), None)
    syn_payload = (synthesis or {}).get("payload") or {}
    # the implied move lives on the eval row; surface it to hard rules
    if eval_row and "implied_move_pct" not in syn_payload:
        syn_payload = {**syn_payload, "implied_move_pct": eval_row.get("implied_move_pct")}

    checks: List[Dict[str, Any]] = []
    checks += check_completeness(layers, eval_row)
    checks += check_citations(layers)
    checks += check_freshness(layers, now=now, max_age_hours=max_age_hours)
    checks += check_contradictions(layers)
    checks += check_stale_flags(layers)
    checks += check_hard_rules(syn_payload)
    checks += check_db_consistency(eval_row, (synthesis or {}).get("payload"))
    if dispersion is not None:
        checks += check_estimate_dispersion(
            dispersion.get("eps_a"), dispersion.get("eps_b"),
            dispersion.get("label_a", "source_a"), dispersion.get("label_b", "source_b"))

    statuses = {c["status"] for c in checks}
    if "fail" in statuses:
        verdict = "not_actionable"
    elif "warn" in statuses:
        verdict = "sound_with_warnings"
    else:
        verdict = "sound"

    return {
        "verdict": verdict,
        "fails": [c for c in checks if c["status"] == "fail"],
        "warns": [c for c in checks if c["status"] == "warn"],
        "passes": [c for c in checks if c["status"] == "pass"],
        "fixes": [c["fix"] for c in checks if c["status"] != "pass" and c.get("fix")],
        "checks_run": len(checks),
    }
