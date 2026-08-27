"""Does an options payload say what it actually knows?

`get_options_metrics` computes three headline outputs from yfinance option
chains -- an IV term structure, an ATM straddle, and open-interest/volume
ratios -- and yfinance answers illiquid chains with numbers that are shaped
like answers and are not. Live 2026-08-26, DLNG returned `success: true`,
`data_quality: {"iv_status": "ok", "notes": []}`, an `atm_put_iv` of 4.0391
against an `atm_call_iv` of 0.875 on the same strike, and an `implied_move`
that was a bare `{"error": ...}` where every other ticker has a numeric dict.

The audit lives here rather than inside the computation so it is a pure
function of the payload: it can be run over a captured response, calibrated
against a basket of live tickers, and tested without a network. The tool
applies it at the boundary, and nothing it reports is invented -- every
finding names the numbers that produced it.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

from tools.response_meta import warning


# --- thresholds -----------------------------------------------------------
#
# Calibrated live 2026-08-26 against NVDA, TGT, AAPL, MSFT, SPY, F, PLUG,
# AMC, TLRY and RIG. Across all ten the widest same-expiry call/put IV ratio
# was 1.377 (SPY), no ATM leg IV exceeded 1.0, and no leg repeated its IV
# across two different expiry dates. DLNG breached all three.

# An at-the-money option on a listed equity does not imply a 300% annualized
# volatility. Above this the number is a failed solve, not a measurement.
_IMPLAUSIBLE_LEG_IV = 3.0

# Put-call parity ties same-strike, same-expiry IVs together. They differ by
# a few percent on a real chain; a factor of two means at least one leg was
# not solved from a live quote. Set at 2.0 -- 1.45x above the widest healthy
# reading measured, and well under DLNG's 4.62x.
_PARITY_DIVERGENCE_RATIO = 2.0

# The rule that already shipped: real ATM IVs for listed equities live in the
# 0.15-0.60 band, and everything under this is yfinance's inactive-contract
# sentinel rather than implied volatility.
_SENTINEL_IV_CEILING = 0.08

_LEGS = ("atm_call_iv", "atm_put_iv")

# Most severe first. `iv_status` reports the first finding that fires so the
# single field stays stable for callers that switch on it; `notes` and
# `iv_findings` carry every finding, because a payload with three problems
# that reports one has thrown two away.
_SEVERITY = (
  "implausible_iv_level",
  "sentinel_iv_level",
  "parity_divergence",
  "pinned_leg_iv",
  "constant_atm_iv",
)

_STATUS_FOR = {
  "implausible_iv_level": "suspect_iv_implausible",
  "sentinel_iv_level":    "suspect_iv_sentinel",
  "parity_divergence":    "suspect_iv_parity_violation",
  "pinned_leg_iv":        "suspect_iv_pinned",
  "constant_atm_iv":      "suspect_iv_constant",
}


def _numeric(value: Any) -> Optional[float]:
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    return None
  return float(value)


def audit_iv(term_structure: Any) -> Dict[str, Any]:
  """Read a term structure and say what is wrong with it.

  Returns `{"iv_status", "notes", "iv_findings"}`. `iv_status` is "ok" when
  nothing fires, "no_iv_data" when there was nothing to read, and otherwise
  a `suspect_*` string -- the prefix the preearnings-research skill matches
  on to drop IV inputs while keeping the volume ratios.
  """
  findings: List[str] = []
  notes: List[str] = []

  def _fire(code: str, note: str) -> None:
    if code not in findings:
      findings.append(code)
      notes.append(note)

  entries = [(label, entry) for label, entry in (term_structure or {}).items()
             if isinstance(entry, dict)] if isinstance(term_structure, dict) else []

  atm_ivs = [v for v in (_numeric(e.get("atm_iv")) for _, e in entries) if v is not None]
  leg_ivs = [v for _, e in entries for v in
             (_numeric(e.get(leg)) for leg in _LEGS) if v is not None]

  if not atm_ivs and not leg_ivs:
    return {"iv_status": "no_iv_data", "iv_findings": [],
            "notes": ["no ATM IV could be extracted"]}

  # 1. A level no listed equity's ATM option implies.
  implausible = sorted({v for v in leg_ivs if v > _IMPLAUSIBLE_LEG_IV}, reverse=True)
  if implausible:
    _fire("implausible_iv_level",
          f"ATM leg IV of {implausible[0]:g} "
          f"({implausible[0] * 100:.0f}%) is not an implied volatility -- it is "
          f"what a solver returns when it cannot solve. Threshold "
          f"{_IMPLAUSIBLE_LEG_IV:g}.")

  # 2. The rule that already shipped, unchanged.
  if atm_ivs and max(atm_ivs) < _SENTINEL_IV_CEILING:
    _fire("sentinel_iv_level",
          f"all ATM IVs < {_SENTINEL_IV_CEILING} -- likely yfinance sentinel "
          f"values, not real implied volatility")

  # 3. Put-call parity ties the two legs of one strike together.
  for label, entry in entries:
    call_iv = _numeric(entry.get("atm_call_iv"))
    put_iv = _numeric(entry.get("atm_put_iv"))
    if call_iv is None or put_iv is None or min(call_iv, put_iv) <= 0:
      continue
    ratio = max(call_iv, put_iv) / min(call_iv, put_iv)
    if ratio >= _PARITY_DIVERGENCE_RATIO:
      _fire("parity_divergence",
            f"{label} ({entry.get('expiry')}): atm_call_iv {call_iv:g} against "
            f"atm_put_iv {put_iv:g} on the same strike is a {ratio:.2f}x gap. "
            f"Put-call parity forces same-strike IVs to agree closely, so at "
            f"least one leg is not solved from a live quote.")

  # 4. A leg that does not move with tenor is not a term structure.
  #    Keyed on the expiry DATE: the 60d and 90d buckets routinely resolve to
  #    the same listed expiry on a short chain, and counting those as two
  #    readings would fire on every thin name in the market.
  for leg in _LEGS:
    by_value: Dict[float, set] = {}
    for _, entry in entries:
      value = _numeric(entry.get(leg))
      expiry = entry.get("expiry")
      if value is None or not expiry:
        continue
      by_value.setdefault(round(value, 4), set()).add(expiry)
    for value, expiries in by_value.items():
      if len(expiries) >= 2:
        _fire("pinned_leg_iv",
              f"{leg} is {value:g} at {len(expiries)} different expiries "
              f"({', '.join(sorted(expiries))}). Implied volatility varies "
              f"with tenor; an identical value across expiries is the solver "
              f"returning a default.")

  # 5. The second rule that already shipped, unchanged.
  if len(atm_ivs) > 1 and len({round(v, 4) for v in atm_ivs}) == 1:
    _fire("constant_atm_iv", "all ATM IVs identical across tenors -- suspect data")

  if not findings:
    return {"iv_status": "ok", "iv_findings": [], "notes": []}

  ordered = [code for code in _SEVERITY if code in findings]
  return {"iv_status": _STATUS_FOR[ordered[0]],
          "iv_findings": ordered,
          "notes": [notes[findings.index(code)] for code in ordered]}


def normalize_implied_move(block: Any) -> Tuple[Dict[str, Any], Optional[dict]]:
  """Give the straddle block the same shape whether or not it was priced.

  A bare `{"error": ...}` where every other ticker has `implied_move_pct`
  means `.get("implied_move_pct")` answers None for "no such key", and a
  caller writing `... or 0` gets a 0% implied move on a chain that could not
  be priced at all. `available` is the field to branch on; the numeric keys
  are always present and explicitly null when there is no number; and the
  `error` key stays because .claude/skills/preearnings-research documents it.

  Returns `(block, warning_or_None)`.
  """
  if not isinstance(block, dict):
    reason = "get_options_metrics returned no implied_move block"
    return ({"available": False, "reason": reason, "error": reason,
             "implied_move_pct": None, "straddle_cost": None},
            warning("implied_move_unavailable", _UNAVAILABLE_MESSAGE.format(reason=reason)))

  if block.get("implied_move_pct") is not None:
    return ({**block, "available": True}, None)

  reason = str(block.get("error") or block.get("reason")
               or "implied_move_pct was not computed and no reason was given")
  normalized = {**block,
                "available": False,
                "reason": reason,
                "error": block.get("error", reason),
                "implied_move_pct": None,
                "straddle_cost": block.get("straddle_cost")}
  return normalized, warning("implied_move_unavailable",
                             _UNAVAILABLE_MESSAGE.format(reason=reason))


_UNAVAILABLE_MESSAGE = (
  "The ATM straddle could not be priced, so there is no implied move in this "
  "response: {reason}. implied_move.implied_move_pct is null, which is not "
  "zero -- a workflow reading it as a number gets a calm setup on a chain "
  "that could not be priced. The term structure and the open-interest and "
  "volume ratios in this response are unaffected."
)

_SUSPECT_IV_MESSAGE = (
  "The ATM implied volatilities in this response do not survive their own "
  "consistency checks and must not be read as implied volatility. The "
  "open-interest and volume ratios are computed from contract counts, not "
  "from a solver, and remain usable. See data_quality.notes."
)


def audit_options_metrics(result: Any) -> Dict[str, Any]:
  """Apply the audit to a whole `get_options_metrics` payload.

  Returns a new dict; the input is not mutated. A response that already
  failed is passed straight through -- there is no chain to audit, and
  layering a second diagnosis over the first would bury it.
  """
  if not isinstance(result, dict) or result.get("success") is not True:
    return result

  out = copy.deepcopy(result)
  warnings: List[dict] = list(out.get("warnings") or [])

  if "implied_move" in out:
    out["implied_move"], move_warning = normalize_implied_move(out["implied_move"])
    if move_warning is not None and move_warning not in warnings:
      warnings.append(move_warning)

  if "term_structure" in out:
    audit = audit_iv(out["term_structure"])
    existing = out.get("data_quality")
    out["data_quality"] = {
      **(existing if isinstance(existing, dict) else {}),
      "iv_status": audit["iv_status"],
      "iv_findings": audit["iv_findings"],
      "notes": audit["notes"],
      # Contract counts, not solver output. They survive a bad IV solve, and
      # the preearnings-research skill depends on that split.
      "volume_data_usable": True,
    }
    if audit["iv_status"].startswith("suspect"):
      entry = warning("suspect_iv", _SUSPECT_IV_MESSAGE,
                      iv_status=audit["iv_status"], findings=audit["iv_findings"])
      if entry not in warnings:
        warnings.append(entry)

  if warnings:
    out["warnings"] = warnings
    # One of the three headline outputs is missing or unreadable, and the
    # rest of the payload is real. "partial" is the label response_meta has
    # for exactly that; reporting it as complete is the defect.
    out["coverage"] = "partial"
  return out
