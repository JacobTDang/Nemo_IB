"""
Financial Modeling Agent

Sits between the execution phase (data gathering) and the analysis phase
(investment thesis). Reads the variable store populated by tool execution,
decides which financial models are appropriate, sets scenario parameters,
and runs the pure-Python calculation functions from analysis_tools.py.

The LLM's job is narrow: choose what to run and set assumption parameters.
Python does all the arithmetic.
"""
from .openrouter_template import OpenRouterModel
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import sys
import json
from datetime import datetime

from tools.financial_modeling_engine.analysis_tools import (
  _dcf_math, _wacc_math, _lbo_math,
  _credit_profile_math, _scenario_dcf_math, _capital_returns_math,
  _ddm_math, _sensitivity_table_math,
)


# ---------------------------------------------------------------------------
# Decision schema
# LLM fills this in; Python uses it to drive calculations.
# ---------------------------------------------------------------------------

class ScenarioParams(BaseModel):
  bear_growth_y1: float       # Year 1 revenue growth, bear case (decimal e.g. 0.05)
  bear_growth_long_run: float # Year 5 revenue growth, bear case
  base_growth_y1: float       # Year 1 revenue growth, base case
  base_growth_long_run: float # Year 5 revenue growth, base case
  bull_growth_y1: float       # Year 1 revenue growth, bull case
  bull_growth_long_run: float # Year 5 revenue growth, bull case
  bear_margin_adj: float      # Additive adjustment to base ebitda_margin, e.g. -0.02
  bull_margin_adj: float      # Additive adjustment to base ebitda_margin, e.g. +0.02
  reasoning: str              # Brief note explaining how anchors were chosen


class LBOParams(BaseModel):
  entry_premium: float   # Premium to current market cap, e.g. 0.30 for 30%
  leverage_turns: float  # Acquisition debt as multiple of entry EBITDA, e.g. 4.5
  exit_multiple: float   # EV/EBITDA at exit
  hold_years: int        # Investment horizon, typically 5
  reasoning: str


class ModelingDecision(BaseModel):
  run_scenario_dcf: bool
  run_credit_profile: bool
  run_capital_returns: bool
  run_lbo: bool
  run_sensitivity_table: bool = False  # auto-true when run_scenario_dcf is true
  run_ddm: bool = False                # true for high-payout / utility / REIT / consumer staples
  scenario_params: Optional[ScenarioParams] = None  # required if run_scenario_dcf
  lbo_params: Optional[LBOParams] = None            # required if run_lbo
  reasoning: str


class Financial_Modeling_Agent(OpenRouterModel):
  """
  Hybrid financial modeling agent.

  LLM (GLM Air): decides what to run + sets scenario/LBO parameters.
  Python: executes the pure math functions from analysis_tools.py.

  Fails gracefully per model -- if one model errors, others still run.
  """
  response_schema = ModelingDecision
  MAX_OUTPUT_TOKENS = 2048   # JSON decision only; no prose output needed
  REASONING_EFFORT = None    # No deep reasoning -- just parameter selection

  def __init__(self, model_name: str = 'z-ai/glm-4.5-air:free'):
    super().__init__(model_name=model_name)

  # ---- Public entry point ------------------------------------------------

  def model(self,
            user_query: str,
            variables: Dict[str, Any],
            modeling_tools_context: str) -> Dict[str, Any]:
    """
    Run financial models appropriate to the query and available data.

    Args:
      user_query: original user request
      variables: flat variable store populated by execution phase
      modeling_tools_context: dynamically injected tool descriptions + data requirements

    Returns:
      model_outputs dict with results from each model that ran, plus the decision.
    """
    self.conversatoin_history = []

    system_prompt = self._build_system_prompt(modeling_tools_context)
    user_prompt = self._build_user_prompt(user_query, variables)

    print(f"\n[Modeling Agent] Deciding what to model...", file=sys.stderr, flush=True)

    response = self.generate_response(prompt=user_prompt, system_prompt=system_prompt)

    # Parse decision
    try:
      decision = self.parse_response(response)
    except Exception as e:
      print(f"[Modeling Agent] Parse failed: {e}. Using conservative defaults.", file=sys.stderr, flush=True)
      decision = self._conservative_defaults(variables)

    print(f"[Modeling Agent] Decision: DCF={decision.run_scenario_dcf} | "
          f"Credit={decision.run_credit_profile} | "
          f"CapReturns={decision.run_capital_returns} | "
          f"LBO={decision.run_lbo}", file=sys.stderr, flush=True)

    # Validate that signal inputs reached the decision
    beat_rate = variables.get('earnings_quality.beat_rate_pct')
    curve = variables.get('yield_curve_shape')
    nfci = variables.get('macro.NFCI')
    bull_y1 = decision.scenario_params.bull_growth_y1 if decision.scenario_params else None
    print(f"[Validate Signals] beat_rate={beat_rate} | curve={curve} | nfci={nfci} | chose bull_y1={bull_y1}",
          file=sys.stderr, flush=True)

    # Run calculations
    outputs: Dict[str, Any] = {
      'decision': decision.model_dump(),
      'models_run': [],
      'ticker': variables.get('ticker', ''),
    }

    if decision.run_scenario_dcf and decision.scenario_params:
      result = self._run_scenario_dcf(variables, decision.scenario_params)
      if result:
        outputs['scenario_dcf'] = result
        outputs['models_run'].append('scenario_dcf')

    if decision.run_credit_profile:
      result = self._run_credit_profile(variables)
      if result:
        outputs['credit_profile'] = result
        outputs['models_run'].append('credit_profile')

    if decision.run_capital_returns:
      result = self._run_capital_returns(variables)
      if result:
        outputs['capital_returns'] = result
        outputs['models_run'].append('capital_returns')

    if decision.run_lbo and decision.lbo_params:
      result = self._run_lbo(variables, decision.lbo_params)
      if result:
        outputs['lbo'] = result
        outputs['models_run'].append('lbo')

    # Sensitivity table auto-runs whenever scenario DCF runs
    if decision.run_sensitivity_table or decision.run_scenario_dcf:
      result = self._run_sensitivity_table(variables)
      if result:
        outputs['sensitivity_table'] = result
        outputs['models_run'].append('sensitivity_table')

    if decision.run_ddm:
      result = self._run_ddm(variables)
      if result:
        outputs['ddm'] = result
        outputs['models_run'].append('ddm')

    print(f"[Modeling Agent] Models run: {outputs['models_run']}", file=sys.stderr, flush=True)
    return outputs

  # ---- Prompt builders ----------------------------------------------------

  def _build_system_prompt(self, modeling_tools_context: str) -> str:
    current_date = datetime.now().strftime("%B %d, %Y")
    return f"""You are a financial modeling specialist at a bulge-bracket investment bank.
Today: {current_date}

YOUR ROLE:
Read the gathered variable store and decide which financial models to run and what parameters to use.
You make TWO decisions:
1. WHAT TO RUN: Select which models are appropriate for this query and the available data.
2. PARAMETERS: Set scenario growth rates, margin adjustments, and LBO terms.

You do NOT do the calculations. You output a JSON decision; Python runs the math.

AVAILABLE MODELS AND THEIR DATA REQUIREMENTS:
{modeling_tools_context}

PARAMETER GUIDANCE:
- base_growth_y1: use financials.revenueGrowthTTMYoy if present; else use forward estimate avg / 100.
- base_growth_long_run: use financials.revenueGrowth5Y if present; else GDP + 1-2% sector premium.
- bear_growth_y1: analyst forward low estimate (get_forward_estimates) / revenue_base, or base * 0.55.
- bull_growth_y1: analyst forward high estimate / revenue_base, or base * 1.45.
- bear/bull long_run: ±1-2pp from base long_run; floor bear at 0.01 (1%), cap bull at 0.20 (20%).
- bear_margin_adj: -0.01 to -0.03 (competitive pressure, macro headwinds).
- bull_margin_adj: +0.01 to +0.03 (operating leverage, pricing power).
- LBO leverage_turns: 3-4x for cyclicals, 4-5x for stable FCF businesses, 5-6x for very defensive.
- LBO exit_multiple: anchor to financials.evEbitdaTTM from variable store; apply -10% discount (buyout discount).
- LBO entry_premium: 0.25-0.35 for public targets; 0.30 is standard.
- LBO hold_years: always 5 unless query specifies otherwise.

SELECTION RULES:
- run_scenario_dcf: true when wacc, revenue_base, ebitda_margin, capex_pct_revenue, depreciation, tax_rate, shares_outstanding are all non-zero in the variable store.
- run_credit_profile: true when total_debt (or totalDebt) AND ebitda (or revenue_base + ebitda_margin) are present.
- run_capital_returns: true ONLY when dividendsPaid or repurchaseOfCapitalStock appear in the variable store with non-zero values.
- run_lbo: true ONLY when the query involves M&A, private equity, buyout potential, acquisition, or takeout analysis.
- run_sensitivity_table: true WHENEVER run_scenario_dcf is true (no extra cost, builds the football field).
- run_ddm: true when one or more applies: (a) cf.dividendsPaid is non-zero AND payout_ratio_pct > 50, (b) profile.gicsSubIndustry or profile.finnhubIndustry contains "Utility"/"Utilities"/"REIT"/"Consumer Staples"/"Tobacco", (c) the query mentions "dividend" or "income".

SIGNAL-BASED ADJUSTMENTS (apply cumulatively to scenario_params):
- If earnings_quality.beat_rate_pct > 75: management consistently beats; raise bull_growth_y1 by 0.02 (200bps).
- If earnings_quality.beat_rate_pct < 40: management misses; lower bull_growth_y1 by 0.02, lower base_growth_y1 by 0.01.
- If insider_sentiment.signal == "net_buying": raise bull_margin_adj by 0.005.
- If insider_sentiment.signal == "net_selling": lower bull_growth_y1 by 0.01.
- If yield_curve_shape == "inverted": use more conservative settings; lower bull_growth_y1 by 0.02 and make bear_margin_adj more negative by 0.01.
- If macro.NFCI > 0 (tightening financial conditions): compress bull case; lower bull_growth_long_run by 0.005.
- Use profile.finnhubIndustry / profile.gicsSubIndustry to inform sector-appropriate margins (e.g. utilities = capital-intensive, low margins; software = high margins).

OUTPUT ONLY VALID JSON matching the ModelingDecision schema. No text before or after."""

  def _build_user_prompt(self, user_query: str, variables: Dict[str, Any]) -> str:
    current_date = datetime.now().strftime("%B %d, %Y")

    # Show flat non-namespaced keys
    flat = {k: v for k, v in variables.items() if v not in (None, 0, '', [], {}) and '.' not in k}
    # Show a selection of namespaced keys relevant to modeling decisions
    namespaced_relevant = [
      'financials.revenueGrowthTTMYoy', 'financials.revenueGrowth5Y',
      'financials.evEbitdaTTM', 'financials.ebitdaCagr5Y',
      'macro.real_gdp_growth', 'macro.DGS10', 'macro.NFCI',
      'cf.dividendsPaid', 'cf.repurchaseOfCapitalStock', 'cf.operatingCashFlow',
      'ic.grossMargin', 'ic.netMargin',
      'earnings_quality.beat_rate_pct', 'earnings_quality.avg_surprise_pct',
      'insider_sentiment.signal', 'insider_sentiment.avg_mspr',
      'yield_curve_shape',
      'profile.finnhubIndustry', 'profile.gicsSubIndustry',
    ]
    ns_present = {k: variables[k] for k in namespaced_relevant if k in variables and variables[k]}

    lines = [f"TODAY: {current_date}", f"USER QUERY: {user_query}", "", "VARIABLE STORE (flat keys):"]
    for k, v in sorted(flat.items()):
      val_str = str(v)
      if len(val_str) > 80:
        val_str = val_str[:77] + '...'
      lines.append(f"  {k}: {val_str}")

    if ns_present:
      lines.append("")
      lines.append("VARIABLE STORE (namespaced keys relevant to modeling):")
      for k, v in ns_present.items():
        lines.append(f"  {k}: {v}")

    lines.append("")
    lines.append("Decide which models to run and set parameters based on the above data.")
    lines.append("If a required input is zero or missing, set run_<model>=false.")

    return "\n".join(lines)

  # ---- Conservative fallback when parse fails ----------------------------

  def _conservative_defaults(self, variables: Dict[str, Any]) -> ModelingDecision:
    """Return a safe ModelingDecision when LLM parse fails."""
    has_dcf_inputs = all(
      self._get(variables, k) > 0
      for k in ('wacc', 'revenue_base', 'ebitda_margin', 'shares_outstanding')
    )
    has_debt = self._get(variables, 'totalDebt', 'total_debt') > 0
    base_growth = self._get(variables, 'financials.revenueGrowthTTMYoy') / 100
    if base_growth <= 0:
      base_growth = 0.05
    base_margin = self._get(variables, 'ebitda_margin')
    if base_margin > 1:
      base_margin /= 100

    scenario = ScenarioParams(
      bear_growth_y1=max(0.01, base_growth * 0.55),
      bear_growth_long_run=max(0.01, base_growth * 0.40),
      base_growth_y1=base_growth,
      base_growth_long_run=max(0.02, base_growth * 0.60),
      bull_growth_y1=base_growth * 1.45,
      bull_growth_long_run=base_growth * 0.80,
      bear_margin_adj=-0.02,
      bull_margin_adj=+0.02,
      reasoning="Conservative defaults used after parse failure.",
    ) if has_dcf_inputs else None

    return ModelingDecision(
      run_scenario_dcf=has_dcf_inputs and scenario is not None,
      run_credit_profile=has_debt,
      run_capital_returns=False,
      run_lbo=False,
      scenario_params=scenario,
      lbo_params=None,
      reasoning="Parse failure fallback.",
    )

  # ---- Variable store helpers --------------------------------------------

  def _get(self, variables: Dict[str, Any], *keys, default: float = 0.0) -> float:
    """Return first non-zero value found among the given keys."""
    for key in keys:
      val = variables.get(key)
      if val is not None:
        try:
          f = float(val)
          if f != 0:
            return f
        except (TypeError, ValueError):
          continue
    return default

  # ---- Model runners ------------------------------------------------------

  def _build_growth_schedule(self, y1: float, long_run: float, years: int = 5) -> list:
    """Linear taper from y1 to long_run over `years` periods."""
    if years == 1:
      return [y1]
    schedule = []
    for i in range(years):
      t = i / (years - 1)
      rate = y1 + t * (long_run - y1)
      schedule.append(round(rate, 6))
    return schedule

  def _run_scenario_dcf(self, variables: Dict[str, Any],
                         params: ScenarioParams) -> Optional[Dict[str, Any]]:
    try:
      base_margin = self._get(variables, 'ebitda_margin')
      if base_margin > 1:
        base_margin /= 100

      base_inputs = {
        'revenue_base':      self._get(variables, 'revenue_base'),
        'capex_pct_revenue': self._get(variables, 'capex_pct_revenue'),
        'tax_rate':          self._get(variables, 'tax_rate'),
        'depreciation':      self._get(variables, 'depreciation'),
        'wacc':              self._get(variables, 'wacc'),
        'terminal_growth':   self._get(variables, 'terminal_growth'),
        'terminal_multiple': self._get(variables, 'terminal_multiple',
                                       'financials.evEbitdaTTM'),
        'cash':              self._get(variables, 'totalCash', 'cash'),
        'debt':              self._get(variables, 'totalDebt', 'debt'),
        'shares_outstanding': self._get(variables, 'sharesOutstanding', 'shares_outstanding'),
        'ticker':            variables.get('ticker', ''),
        # Overridden per scenario:
        'ebitda_margin':     base_margin,
        'revenue_growth':    [0, 0, 0, 0, 0],
      }

      bear_growth  = self._build_growth_schedule(params.bear_growth_y1, params.bear_growth_long_run)
      base_growth  = self._build_growth_schedule(params.base_growth_y1, params.base_growth_long_run)
      bull_growth  = self._build_growth_schedule(params.bull_growth_y1, params.bull_growth_long_run)
      bear_margin  = base_margin + params.bear_margin_adj
      base_margin_ = base_margin
      bull_margin  = base_margin + params.bull_margin_adj

      result = _scenario_dcf_math(
        base_inputs=base_inputs,
        bear_growth=bear_growth, base_growth=base_growth, bull_growth=bull_growth,
        bear_margin=bear_margin, base_margin=base_margin_, bull_margin=bull_margin,
      )
      result['scenario_assumptions'] = params.model_dump()

      # Regime-weighted expected price -- yield curve + NFCI inform probability weights
      curve = variables.get('yield_curve_shape', 'normal')
      nfci = variables.get('macro.NFCI', 0) or 0
      if curve == 'inverted' and nfci > 0:
        weights = {'bear': 0.50, 'base': 0.35, 'bull': 0.15}
        regime = 'late-cycle / tightening'
      elif curve == 'inverted':
        weights = {'bear': 0.40, 'base': 0.40, 'bull': 0.20}
        regime = 'late-cycle'
      elif nfci > 0.5:
        weights = {'bear': 0.35, 'base': 0.45, 'bull': 0.20}
        regime = 'tightening conditions'
      elif curve == 'flat':
        weights = {'bear': 0.30, 'base': 0.45, 'bull': 0.25}
        regime = 'transitional'
      else:
        weights = {'bear': 0.20, 'base': 0.50, 'bull': 0.30}
        regime = 'expansion'

      px = result['price_range']
      expected_price = (weights['bear'] * px['low']
                        + weights['base'] * px['mid']
                        + weights['bull'] * px['high'])
      result['regime_weighted'] = {
        'regime': regime,
        'weights': weights,
        'expected_price': round(expected_price, 2),
        'inputs': {'yield_curve': curve, 'nfci': nfci},
      }

      print(f"[Modeling Agent] Scenario DCF: Bear=${result['price_range']['low']:.2f} | "
            f"Base=${result['price_range']['mid']:.2f} | "
            f"Bull=${result['price_range']['high']:.2f} | "
            f"Regime={regime} | Weighted=${expected_price:.2f}", file=sys.stderr, flush=True)
      return result
    except Exception as e:
      print(f"[Modeling Agent] Scenario DCF failed: {e}", file=sys.stderr, flush=True)
      return None

  def _run_credit_profile(self, variables: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
      revenue   = self._get(variables, 'revenue_base')
      margin    = self._get(variables, 'ebitda_margin')
      if margin > 1:
        margin /= 100

      # Polymorphic fetcher fallback: if variable store missed, try multi-source
      ticker = variables.get('ticker', '')
      if (revenue <= 0 or margin <= 0) and ticker:
        from data.sources import get_revenue, get_ebitda_margin_pct
        if revenue <= 0:
          v, src = get_revenue(ticker, variables)
          if v is not None:
            revenue = v
            variables['revenue_base'] = revenue
            print(f"[Sources] revenue resolved via {src}: ${revenue/1e9:.2f}B",
                  file=sys.stderr, flush=True)
        if margin <= 0:
          v, src = get_ebitda_margin_pct(ticker, variables)
          if v is not None:
            margin = v / 100 if v > 1 else v
            variables['ebitda_margin'] = margin
            print(f"[Sources] ebitda_margin resolved via {src}: {margin*100:.2f}%",
                  file=sys.stderr, flush=True)

      # Fail-fast: without revenue or margin, ebitda = 0 and credit ratios explode
      if revenue <= 0 or margin <= 0:
        print(f"[Validate Credit] Skipped: revenue={revenue} margin={margin} "
              "(missing fundamentals)", file=sys.stderr, flush=True)
        return None

      dep_pct   = self._get(variables, 'depreciation')
      if dep_pct > 1:
        dep_pct /= 100
      capex_pct = self._get(variables, 'capex_pct_revenue')
      if capex_pct > 1:
        capex_pct /= 100

      ebitda   = revenue * margin
      dep_abs  = revenue * dep_pct
      capex_abs= revenue * capex_pct

      result = _credit_profile_math(
        total_debt=       self._get(variables, 'totalDebt', 'total_debt'),
        cash=             self._get(variables, 'totalCash', 'cash'),
        ebitda=           ebitda,
        interest_expense= self._get(variables, 'interestExpense', 'interest_expense'),
        depreciation_abs= dep_abs,
        capex_abs=        capex_abs,
        tax_rate=         self._get(variables, 'tax_rate'),
        market_cap=       self._get(variables, 'marketCap', 'market_cap'),
      )
      if result.get('error'):
        print(f"[Validate Credit] {result['error']}", file=sys.stderr, flush=True)
        return None
      print(f"[Modeling Agent] Credit Profile: {result['credit_label']} | "
            f"Net Debt/EBITDA: {result['net_debt_ebitda']:.1f}x", file=sys.stderr, flush=True)
      return result
    except Exception as e:
      print(f"[Modeling Agent] Credit Profile failed: {e}", file=sys.stderr, flush=True)
      return None

  def _run_capital_returns(self, variables: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
      revenue    = self._get(variables, 'revenue_base')
      margin     = self._get(variables, 'ebitda_margin')
      if margin > 1:
        margin /= 100
      market_cap = self._get(variables, 'marketCap', 'market_cap')

      # Polymorphic fetcher fallback
      ticker = variables.get('ticker', '')
      if (revenue <= 0 or margin <= 0 or market_cap <= 0) and ticker:
        from data.sources import get_revenue, get_ebitda_margin_pct, get_market_cap
        if revenue <= 0:
          v, src = get_revenue(ticker, variables)
          if v is not None: revenue, variables['revenue_base'] = v, v
        if margin <= 0:
          v, src = get_ebitda_margin_pct(ticker, variables)
          if v is not None:
            margin = v / 100 if v > 1 else v
            variables['ebitda_margin'] = margin
        if market_cap <= 0:
          v, src = get_market_cap(ticker, variables)
          if v is not None: market_cap, variables['marketCap'] = v, v

      # Fail-fast: yields are meaningless without revenue, margin, or market cap
      if revenue <= 0 or margin <= 0 or market_cap <= 0:
        print(f"[Validate CapReturns] Skipped: revenue={revenue} margin={margin} "
              f"market_cap={market_cap} (missing fundamentals)",
              file=sys.stderr, flush=True)
        return None

      dep_pct   = self._get(variables, 'depreciation')
      if dep_pct > 1:
        dep_pct /= 100
      capex_pct = self._get(variables, 'capex_pct_revenue')
      if capex_pct > 1:
        capex_pct /= 100

      result = _capital_returns_math(
        market_cap=          market_cap,
        ebitda=              revenue * margin,
        capex_abs=           revenue * capex_pct,
        tax_rate=            self._get(variables, 'tax_rate'),
        depreciation_abs=    revenue * dep_pct,
        dividends_paid=      self._get(variables, 'cf.dividendsPaid', 'dividendsPaid'),
        shares_repurchased=  self._get(variables, 'cf.repurchaseOfCapitalStock',
                                       'repurchaseOfCapitalStock'),
        shares_outstanding=  self._get(variables, 'sharesOutstanding', 'shares_outstanding'),
      )
      if result.get('error'):
        print(f"[Validate CapReturns] {result['error']}", file=sys.stderr, flush=True)
        return None
      print(f"[Modeling Agent] Capital Returns: TSY={result['total_shareholder_yield_pct']}% | "
            f"{result['sustainability']}", file=sys.stderr, flush=True)
      return result
    except Exception as e:
      print(f"[Modeling Agent] Capital Returns failed: {e}", file=sys.stderr, flush=True)
      return None

  def _run_lbo(self, variables: Dict[str, Any],
               params: LBOParams) -> Optional[Dict[str, Any]]:
    try:
      market_cap = self._get(variables, 'marketCap', 'market_cap')
      total_debt = self._get(variables, 'totalDebt', 'total_debt')
      cash       = self._get(variables, 'totalCash', 'cash')
      revenue    = self._get(variables, 'revenue_base')
      margin     = self._get(variables, 'ebitda_margin')
      if margin > 1:
        margin /= 100

      # Polymorphic fetcher fallback for LBO inputs
      ticker = variables.get('ticker', '')
      if (market_cap <= 0 or revenue <= 0 or margin <= 0) and ticker:
        from data.sources import get_revenue, get_ebitda_margin_pct, get_market_cap
        if revenue <= 0:
          v, src = get_revenue(ticker, variables)
          if v is not None: revenue, variables['revenue_base'] = v, v
        if margin <= 0:
          v, src = get_ebitda_margin_pct(ticker, variables)
          if v is not None:
            margin = v / 100 if v > 1 else v
            variables['ebitda_margin'] = margin
        if market_cap <= 0:
          v, src = get_market_cap(ticker, variables)
          if v is not None: market_cap, variables['marketCap'] = v, v

      # Fail-fast: LBO needs market_cap (for entry_ev) + revenue/margin (for ebitda)
      if market_cap <= 0 or revenue <= 0 or margin <= 0:
        print(f"[Validate LBO] Skipped: market_cap={market_cap} revenue={revenue} "
              f"margin={margin} (missing fundamentals)", file=sys.stderr, flush=True)
        return None

      net_debt   = total_debt - cash
      entry_ev   = (market_cap + net_debt) * (1 + params.entry_premium)

      # Debt rate: risk_free + HY spread (or fallback to 8%)
      risk_free  = self._get(variables, 'macro.DGS10', default=0.045)
      if risk_free > 1:
        risk_free /= 100
      # HY spread stored as basis points by _flatten_macro; convert to decimal.
      hy_spread_bps = self._get(variables, 'credit_spread.BAMLH0A0HYM2', 'credit_spread_hy', 'BAMLH0A0HYM2')
      if hy_spread_bps == 0:
        hy_spread = 0.035  # fallback when no macro data available
      else:
        hy_spread = hy_spread_bps / 10000  # 350bps -> 0.035
      debt_rate  = risk_free + max(0.025, hy_spread)
      print(f"[Validate LBO] rf={risk_free:.4f} + hy_spread={hy_spread:.4f} (raw_bps={hy_spread_bps}) = debt_rate={debt_rate:.4f}",
            file=sys.stderr, flush=True)

      # revenue/margin already validated and normalized at top of method
      revenue_base = revenue
      dep_pct      = self._get(variables, 'depreciation')
      if dep_pct > 1:
        dep_pct /= 100
      capex_pct    = self._get(variables, 'capex_pct_revenue')
      if capex_pct > 1:
        capex_pct /= 100
      tax_rate     = self._get(variables, 'tax_rate')

      # Use stored revenue_growth or build from base growth
      base_growth_y1   = self._get(variables, 'financials.revenueGrowthTTMYoy',
                                   default=5.0) / 100
      base_growth_long = self._get(variables, 'financials.revenueGrowth5Y',
                                   default=3.0) / 100
      revenue_growth   = self._build_growth_schedule(
        base_growth_y1, base_growth_long, years=params.hold_years
      )

      result = _lbo_math(
        entry_ev=        entry_ev,
        revenue_base=    revenue_base,
        ebitda_margin=   margin,
        capex_pct_revenue= capex_pct,
        depreciation=    dep_pct,
        tax_rate=        tax_rate,
        revenue_growth=  revenue_growth,
        debt_interest_rate= debt_rate,
        leverage_turns=  params.leverage_turns,
        exit_multiple=   params.exit_multiple,
        hold_years=      params.hold_years,
      )
      result['lbo_assumptions'] = params.model_dump()
      result['debt_rate_composition'] = {
        'risk_free': round(risk_free, 4),
        'hy_spread': round(hy_spread, 4),
        'all_in_rate': round(debt_rate, 4),
      }
      print(f"[Modeling Agent] LBO: IRR={result['irr_pct']}% | MOIC={result['moic']}x | "
            f"20%+ IRR: {result['achieves_20pct_irr']}", file=sys.stderr, flush=True)
      return result
    except Exception as e:
      print(f"[Modeling Agent] LBO failed: {e}", file=sys.stderr, flush=True)
      return None

  def _run_sensitivity_table(self, variables: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
      base_margin = self._get(variables, 'ebitda_margin')
      if base_margin > 1:
        base_margin /= 100
      ttm = self._get(variables, 'financials.revenueGrowthTTMYoy') / 100
      if ttm <= 0:
        ttm = 0.05
      long_run = self._get(variables, 'financials.revenueGrowth5Y') / 100
      if long_run <= 0:
        long_run = ttm * 0.5
      growth_schedule = self._build_growth_schedule(ttm, long_run)

      base_inputs = {
        'revenue_base':      self._get(variables, 'revenue_base'),
        'ebitda_margin':     base_margin,
        'capex_pct_revenue': self._get(variables, 'capex_pct_revenue'),
        'tax_rate':          self._get(variables, 'tax_rate'),
        'depreciation':      self._get(variables, 'depreciation'),
        'revenue_growth':    growth_schedule,
        'wacc':              self._get(variables, 'wacc'),
        'terminal_growth':   self._get(variables, 'terminal_growth', default=0.025),
        'terminal_multiple': self._get(variables, 'terminal_multiple', 'financials.evEbitdaTTM'),
        'cash':              self._get(variables, 'totalCash', 'cash'),
        'debt':              self._get(variables, 'totalDebt', 'debt'),
        'shares_outstanding': self._get(variables, 'sharesOutstanding', 'shares_outstanding'),
        'ticker':            variables.get('ticker', ''),
      }
      if base_inputs['wacc'] <= 0 or base_inputs['revenue_base'] <= 0:
        print(f"[Modeling Agent] Sensitivity skipped: missing WACC or revenue", file=sys.stderr, flush=True)
        return None

      result = _sensitivity_table_math(base_inputs)
      print(f"[Validate Sensitivity] {result['cells_filled']} cells, range "
            f"${result['min_price']:.2f}-${result['max_price']:.2f} | mid=${result['mid_price']:.2f}",
            file=sys.stderr, flush=True)
      return result
    except Exception as e:
      print(f"[Modeling Agent] Sensitivity table failed: {e}", file=sys.stderr, flush=True)
      return None

  def _run_ddm(self, variables: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
      # DPS = dividendsPaid / sharesOutstanding (absolute value; CF reports outflows as negative)
      div_paid = abs(self._get(variables, 'cf.dividendsPaid', 'dividendsPaid'))
      shares   = self._get(variables, 'sharesOutstanding', 'shares_outstanding')
      if div_paid <= 0 or shares <= 0:
        print(f"[Modeling Agent] DDM skipped: missing dividends or shares", file=sys.stderr, flush=True)
        return None
      current_dps = div_paid / shares

      # Cost of equity: from WACC calc if available, else fall back to risk_free + 0.06*beta
      cost_of_equity = self._get(variables, 'cost_of_equity')
      if cost_of_equity <= 0:
        rf = self._get(variables, 'risk_free_rate', 'macro.DGS10', default=0.045)
        if rf > 1:
          rf /= 100
        beta = self._get(variables, 'beta', default=1.0)
        cost_of_equity = rf + 0.06 * beta

      # Terminal growth: prefer macro GDP, fall back to 3%
      tg = self._get(variables, 'terminal_growth', 'macro.real_gdp_growth', default=0.03)
      if tg > 1:
        tg /= 100

      result = _ddm_math(
        current_dps=current_dps,
        cost_of_equity=cost_of_equity,
        terminal_growth=tg,
      )
      if result.get('success'):
        print(f"[Validate DDM] DPS={current_dps:.4f} Ke={cost_of_equity:.4f} g={tg:.4f} "
              f"-> ${result['intrinsic_value_per_share']:.2f}", file=sys.stderr, flush=True)
      return result
    except Exception as e:
      print(f"[Modeling Agent] DDM failed: {e}", file=sys.stderr, flush=True)
      return None
