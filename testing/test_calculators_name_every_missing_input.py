"""A calculator called without an input refuses, and names every one missing.

`calculate_dcf` was described for a planner that resolved its inputs from a
variable store: "PASS ONLY TICKER ... Never pass any numeric arguments", and
each parameter read "SET TO 0 -- auto-resolved". That planner,
agent/workflows/execution_engine.py, was retired in e87b649. Since then the
only reader of the description is an MCP client, and nothing resolves anything
for it.

A client that obeyed got "revenue_base is zero or absent". A client that
passed what it had got zero for the rest. Cursor passed revenue, margin, WACC,
growth, cash, debt and shares, and NVDA came back at $117.41 a share with
tax_rate 0, capex_pct_revenue 0 and depreciation 0 -- no tax and no capital
spending, success: true. With those three supplied the same inputs give
$93.46. calculate_wacc had the same shape: `{"market_cap": 3e12}` returned
wacc 0.0, success: true.

So an absent input is refused, and the refusal names every absent input, not
the first. A refusal that names one leaves a client three calls from the
answer one call could have given it. None is absence: JSON null arrives as
None. Zero is a value where a company can really have none -- no debt, no
cash, no tax -- and is refused where it cannot: a share count, a discount
rate, a market capitalisation.

calculate_scenario_dcf, calculate_lbo, calculate_credit_profile and
calculate_capital_returns carried the same description ("Data that must be in
the variable store before modeling") and the same flaw. calculate_scenario_dcf
read capex, tax, D&A, WACC, terminal growth, cash, debt and shares with
`args.get(name, 0)` and guarded only revenue_base, so a client that passed some
of them got a bear, base and bull price built on a company that pays no tax.
calculate_capital_returns read an absent dividends_paid, shares_repurchased and
shares_outstanding as 0, so a dividend payer reported a 0% dividend yield and
"Sustainable". calculate_credit_profile dropped the FCF yield without a word
when market_cap was absent. calculate_lbo invented 3% growth for an empty
revenue_growth and repeated the last rate to fill a hold longer than the list.
The same rules apply to all four as to calculate_dcf.
"""
import asyncio
import inspect
import json
import re

import mcp.types as types
import pytest

from tools.financial_modeling_engine import analysis_tools as at

DCF_REQUIRED = ["revenue_base", "ebitda_margin", "capex_pct_revenue",
                "tax_rate", "depreciation", "revenue_growth", "wacc",
                "terminal_growth", "cash", "debt", "shares_outstanding"]

# What Cursor passed on 2026-10-20, then the same with the three it left out.
CURSOR = dict(ticker="NVDA", revenue_base=302970011648,
              ebitda_margin=0.6643099852464511, wacc=0.160882,
              revenue_growth=[0.3, 0.22, 0.16, 0.12, 0.08],
              terminal_growth=0.03, cash=62469001216, debt=38860001280,
              shares_outstanding=24147000000)
DCF_FULL = {**CURSOR, "tax_rate": 0.15, "capex_pct_revenue": 0.04,
            "depreciation": 0.02}

WACC_REQUIRED = ["beta", "risk_free_rate", "cost_of_debt", "tax_rate",
                 "market_cap", "total_debt"]
WACC_FULL = dict(beta=1.09, risk_free_rate=0.042, equity_risk_premium=0.06,
                 cost_of_debt=0.019, tax_rate=0.176,
                 market_cap=3_130_000_000_000, total_debt=125_400_000_000)


def _call(tool, args):
    """Through the server's own CallTool handler, validation and all."""
    server = at.Financial_Analysis().server
    handler = server.request_handlers[types.CallToolRequest]
    result = asyncio.run(handler(types.CallToolRequest(
        method="tools/call",
        params=types.CallToolRequestParams(name=tool, arguments=args))))
    text = result.root.content[0].text
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None
    return result.root.isError, text, payload


def _listed(tool):
    server = at.Financial_Analysis().server
    listing = asyncio.run(server.request_handlers[types.ListToolsRequest](
        types.ListToolsRequest(method="tools/list")))
    return next(t for t in listing.root.tools if t.name == tool)


def _named(text, names):
    """Which of `names` the message names, as whole words.

    Whole words, so `debt` is not found inside `total_debt`.
    """
    return {n for n in names if re.search(rf"(?<![A-Za-z_]){n}(?![A-Za-z_])", text)}


def _refused(is_error, payload):
    if is_error:
        return True
    return isinstance(payload, dict) and payload.get("success") is False


def _without(args, *names):
    return {k: v for k, v in args.items() if k not in names}


# ---- calculate_dcf over MCP ------------------------------------------------

def test_a_ticker_alone_is_refused_naming_every_input():
    is_error, text, payload = _call("calculate_dcf", {"ticker": "NVDA"})

    assert _refused(is_error, payload), f"a ticker alone was valued: {text[:300]}"
    assert _named(text, DCF_REQUIRED) == set(DCF_REQUIRED), (
        f"the refusal names {sorted(_named(text, DCF_REQUIRED))}; a client "
        f"needs every absent input from one call: {text[:600]}")


def test_cursors_call_is_refused_naming_the_three_it_left_out():
    is_error, text, payload = _call("calculate_dcf", CURSOR)

    assert _refused(is_error, payload), (
        f"valued NVDA with no tax, capex or D&A: {text[:300]}")
    assert _named(text, DCF_REQUIRED) == {
        "tax_rate", "capex_pct_revenue", "depreciation"}, text[:600]


def test_a_null_is_absent_and_is_named_with_the_rest():
    args = {**_without(DCF_FULL, "capex_pct_revenue"),
            "tax_rate": None, "depreciation": None}
    is_error, text, payload = _call("calculate_dcf", args)

    assert _refused(is_error, payload), text[:300]
    assert _named(text, DCF_REQUIRED) == {
        "tax_rate", "capex_pct_revenue", "depreciation"}, text[:600]


def test_the_full_set_values_the_company_as_before():
    is_error, text, payload = _call("calculate_dcf", DCF_FULL)

    assert not is_error and payload.get("success") is not False, text[:300]
    assert payload["price_per_share"] == pytest.approx(93.46, abs=0.005)
    assert payload["enterprise_value"] == pytest.approx(2233215805319.72, rel=1e-9)


def test_zero_is_a_value_where_a_company_can_have_none():
    zeros = dict(cash=0, debt=0, tax_rate=0, capex_pct_revenue=0,
                 depreciation=0, terminal_growth=0)
    is_error, text, payload = _call("calculate_dcf", {**DCF_FULL, **zeros})

    assert not is_error and payload.get("success") is not False, text[:300]
    assert payload["price_per_share"] is not None
    for name, value in zeros.items():
        assert payload["assumptions"][name] == value, (
            f"{name} was passed as 0 and used as {payload['assumptions'][name]}")


@pytest.mark.parametrize("name", ["revenue_base", "wacc", "shares_outstanding"])
def test_zero_is_refused_where_a_company_cannot_have_none(name):
    is_error, text, payload = _call("calculate_dcf", {**DCF_FULL, name: 0})

    assert _refused(is_error, payload), f"{name}=0 was valued: {text[:300]}"
    assert name in text, f"the refusal does not name {name}: {text[:300]}"
    assert not (payload or {}).get("price_per_share")


# ---- calculate_dcf called as a method -------------------------------------

def test_the_method_names_every_absent_input_and_its_source():
    """The method is public and has no schema in front of it."""
    blocks = asyncio.run(at.Financial_Analysis().calculate_dcf({"ticker": "NVDA"}))
    payload = json.loads(blocks[0].text)

    assert payload.get("success") is False
    assert payload.get("missing_inputs") == DCF_REQUIRED
    for source in ("get_revenue_base", "get_ebitda_margin",
                   "get_capex_pct_revenue", "get_depreciation", "get_tax_rate",
                   "get_market_data", "calculate_wacc"):
        assert source in payload["error"], f"{source} not named: {payload['error']}"


# ---- what an MCP client is told -------------------------------------------

def test_the_dcf_description_asks_for_every_input_and_names_its_source():
    tool = _listed("calculate_dcf")

    for planner_only in ("PASS ONLY TICKER", "variable store",
                         "Never pass any numeric", "execution engine"):
        assert planner_only not in tool.description, (
            f"an MCP client has no planner, and is told {planner_only!r}")
    for name in DCF_REQUIRED:
        assert name in tool.description, f"{name} is not asked for"
    for source in ("get_revenue_base", "get_ebitda_margin",
                   "get_capex_pct_revenue", "get_depreciation", "get_tax_rate",
                   "get_market_data", "calculate_wacc"):
        assert source in tool.description, f"{source} is not named"


def test_the_dcf_schema_requires_every_input_it_refuses_without():
    schema = _listed("calculate_dcf").inputSchema

    assert set(DCF_REQUIRED) <= set(schema["required"]), (
        f"absent from required: {sorted(set(DCF_REQUIRED) - set(schema['required']))}")
    assert "terminal_multiple" not in schema["required"]
    for name, spec in schema["properties"].items():
        text = spec.get("description", "")
        assert "SET TO" not in text and "auto-resolved" not in text, (
            f"{name}: {text}")


# ---- calculate_wacc ---------------------------------------------------------

def test_wacc_with_no_arguments_is_refused_naming_every_input():
    is_error, text, payload = _call("calculate_wacc", {})

    assert _refused(is_error, payload), text[:300]
    assert _named(text, WACC_REQUIRED) == set(WACC_REQUIRED), text[:600]


def test_a_market_cap_alone_is_not_a_zero_cost_of_capital():
    is_error, text, payload = _call("calculate_wacc", {"market_cap": 3e12})

    assert _refused(is_error, payload), f"computed a WACC: {text[:300]}"
    assert _named(text, WACC_REQUIRED) == set(WACC_REQUIRED) - {"market_cap"}


def test_wacc_names_absent_and_null_inputs_together():
    args = {**_without(WACC_FULL, "cost_of_debt"), "tax_rate": None}
    is_error, text, payload = _call("calculate_wacc", args)

    assert _refused(is_error, payload), text[:300]
    assert _named(text, WACC_REQUIRED) == {"cost_of_debt", "tax_rate"}, text[:600]


def test_the_full_wacc_set_computes_as_before():
    is_error, text, payload = _call("calculate_wacc", WACC_FULL)

    assert not is_error and payload.get("success") is not False, text[:300]
    assert payload["wacc"] == pytest.approx(0.103866, abs=1e-6)


def test_a_debt_free_untaxed_company_has_a_cost_of_capital():
    args = {**WACC_FULL, "total_debt": 0, "cost_of_debt": 0, "tax_rate": 0}
    is_error, text, payload = _call("calculate_wacc", args)

    assert not is_error and payload.get("success") is not False, text[:300]
    assert payload["wacc"] == pytest.approx(payload["cost_of_equity"])


def test_a_zero_market_cap_is_refused():
    is_error, text, payload = _call("calculate_wacc", {**WACC_FULL, "market_cap": 0})

    assert _refused(is_error, payload), f"computed a WACC: {text[:300]}"
    assert "market_cap" in text


def test_the_equity_risk_premium_default_is_stated_in_the_result():
    is_error, text, payload = _call(
        "calculate_wacc", _without(WACC_FULL, "equity_risk_premium"))

    assert not is_error and payload.get("success") is not False, text[:300]
    assert payload["inputs"]["equity_risk_premium"] == 0.06


def test_the_wacc_method_names_every_absent_input():
    blocks = asyncio.run(at.Financial_Analysis().calculate_wacc({}))
    payload = json.loads(blocks[0].text)

    assert payload.get("success") is False
    assert payload.get("missing_inputs") == WACC_REQUIRED


def test_the_wacc_description_asks_for_every_input_and_names_its_source():
    tool = _listed("calculate_wacc")

    for planner_only in ("PASS NO ARGUMENTS", "variable store",
                         "execution engine", "have not yet run"):
        assert planner_only not in tool.description, (
            f"an MCP client has no planner, and is told {planner_only!r}")
    for name in WACC_REQUIRED + ["get_market_data", "get_tax_rate",
                                 "get_macro_snapshot"]:
        assert name in tool.description, f"{name} is not named"
    schema = tool.inputSchema
    assert set(WACC_REQUIRED) <= set(schema["required"])
    assert "equity_risk_premium" not in schema["required"]
    for name, spec in schema["properties"].items():
        text = spec.get("description", "")
        assert "SET TO" not in text and "auto-resolved" not in text, (
            f"{name}: {text}")


# ---- the other four calculators -------------------------------------------

SCENARIO_REQUIRED = ["bear_growth", "base_growth", "bull_growth", "bear_margin",
                     "base_margin", "bull_margin", "revenue_base",
                     "capex_pct_revenue", "tax_rate", "depreciation", "wacc",
                     "terminal_growth", "cash", "debt", "shares_outstanding"]
SCENARIO_FULL = dict(
    ticker="MSFT",
    bear_growth=[0.05, 0.04, 0.03, 0.03, 0.02],
    base_growth=[0.12, 0.11, 0.10, 0.09, 0.08],
    bull_growth=[0.20, 0.18, 0.16, 0.14, 0.12],
    bear_margin=0.45, base_margin=0.52, bull_margin=0.58,
    revenue_base=331_839_000_000, capex_pct_revenue=0.20, tax_rate=0.18,
    depreciation=0.08, wacc=0.085, terminal_growth=0.025,
    terminal_multiple=20.0, cash=94_000_000_000, debt=60_000_000_000,
    shares_outstanding=7_430_000_000)

LBO_REQUIRED = ["entry_ev", "revenue_base", "ebitda_margin",
                "capex_pct_revenue", "depreciation", "tax_rate",
                "revenue_growth", "debt_interest_rate", "leverage_turns",
                "exit_multiple"]
LBO_FULL = dict(
    ticker="TEST", entry_ev=200_000_000_000.0, revenue_base=50_000_000_000.0,
    ebitda_margin=0.35, capex_pct_revenue=0.05, depreciation=0.04,
    tax_rate=0.21, revenue_growth=[0.08, 0.07, 0.06, 0.05, 0.04],
    debt_interest_rate=0.09, leverage_turns=5.0, exit_multiple=11.0,
    hold_years=5)

CREDIT_REQUIRED = ["total_debt", "cash", "ebitda", "interest_expense",
                   "depreciation_abs", "capex_abs", "tax_rate", "market_cap"]
CREDIT_FULL = dict(
    ticker="MSFT", total_debt=125_400_000_000, cash=75_000_000_000,
    ebitda=137_000_000_000, interest_expense=2_900_000_000,
    depreciation_abs=22_000_000_000, capex_abs=44_000_000_000, tax_rate=0.176,
    market_cap=3_130_000_000_000)

CAPITAL_REQUIRED = ["market_cap", "ebitda", "capex_abs", "tax_rate",
                    "depreciation_abs", "dividends_paid", "shares_repurchased",
                    "shares_outstanding"]
CAPITAL_FULL = dict(
    ticker="MSFT", market_cap=3_130_000_000_000, ebitda=137_000_000_000,
    capex_abs=44_000_000_000, tax_rate=0.176, depreciation_abs=22_000_000_000,
    dividends_paid=-24_000_000_000, shares_repurchased=-17_000_000_000,
    shares_outstanding=7_430_000_000)

# What each description must name, and what a refusal must point a caller to.
FOUR = {
    "calculate_scenario_dcf": dict(
        required=SCENARIO_REQUIRED, full=SCENARIO_FULL,
        optional=["terminal_multiple"],
        described=["get_revenue_base", "get_ebitda_margin",
                   "get_capex_pct_revenue", "get_depreciation", "get_tax_rate",
                   "get_market_data", "calculate_wacc", "get_basic_financials",
                   "get_forward_estimates"],
        pointed_to=["get_revenue_base", "get_capex_pct_revenue",
                    "get_depreciation", "get_tax_rate", "get_market_data",
                    "calculate_wacc"]),
    "calculate_lbo": dict(
        required=LBO_REQUIRED, full=LBO_FULL, optional=["hold_years"],
        described=["get_market_data", "get_revenue_base", "get_ebitda_margin",
                   "get_capex_pct_revenue", "get_depreciation", "get_tax_rate",
                   "get_basic_financials", "get_macro_snapshot",
                   "get_credit_spreads"],
        pointed_to=["get_revenue_base", "get_ebitda_margin",
                    "get_capex_pct_revenue", "get_depreciation",
                    "get_tax_rate", "get_market_data"]),
    "calculate_credit_profile": dict(
        required=CREDIT_REQUIRED, full=CREDIT_FULL, optional=[],
        described=["get_market_data", "get_revenue_base", "get_ebitda_margin",
                   "get_capex_pct_revenue", "get_depreciation",
                   "get_tax_rate"],
        pointed_to=["get_market_data", "get_revenue_base", "get_ebitda_margin",
                    "get_depreciation", "get_tax_rate"]),
    "calculate_capital_returns": dict(
        required=CAPITAL_REQUIRED, full=CAPITAL_FULL, optional=[],
        described=["get_market_data", "get_revenue_base", "get_ebitda_margin",
                   "get_capex_pct_revenue", "get_depreciation", "get_tax_rate",
                   "get_financial_statements", "get_buyback_history"],
        pointed_to=["get_market_data", "get_revenue_base", "get_ebitda_margin",
                    "get_depreciation", "get_tax_rate",
                    "get_financial_statements"]),
}

PLANNER_WORDING = ("variable store", "execution engine",
                   "Financial Modeling Agent", "MODELING PHASE")

# What an answer carries and a refusal must not: no price, no yield, no IRR
# beside it. (A refusal over MCP also carries the server's provenance envelope.)
ANSWER_KEYS = {"bear", "base", "bull", "price_range", "moic", "irr_pct",
               "equity_proceeds", "net_debt_ebitda", "interest_coverage",
               "fcf_yield_pct", "dividend_yield_pct",
               "total_shareholder_yield_pct"}


def _ran(is_error, payload):
    return (not is_error and isinstance(payload, dict)
            and payload.get("success") is not False and "error" not in payload)


def _method(tool, args):
    blocks = asyncio.run(getattr(at.Financial_Analysis(), tool)(args))
    return json.loads(blocks[0].text)


@pytest.mark.parametrize("tool", FOUR)
def test_a_ticker_alone_is_refused_naming_every_input_of_the_four(tool):
    spec = FOUR[tool]
    is_error, text, payload = _call(tool, {"ticker": spec["full"]["ticker"]})

    assert _refused(is_error, payload), f"{tool} ran on a ticker alone: {text[:300]}"
    assert _named(text, spec["required"]) == set(spec["required"]), (
        f"the refusal names {sorted(_named(text, spec['required']))}; a client "
        f"needs every absent input from one call: {text[:600]}")


@pytest.mark.parametrize("tool", FOUR)
def test_an_absent_and_a_null_input_are_named_together_by_the_four(tool):
    spec = FOUR[tool]
    required = spec["required"]
    absent, null = required[1], required[-1]
    args = {**_without(spec["full"], absent), null: None}
    is_error, text, payload = _call(tool, args)

    assert _refused(is_error, payload), f"{tool} ran without {absent}: {text[:300]}"
    assert _named(text, required) == {absent, null}, text[:600]


@pytest.mark.parametrize("tool", FOUR)
def test_the_four_methods_name_every_absent_input_and_its_source(tool):
    """Each method is public and has no schema in front of it."""
    spec = FOUR[tool]
    payload = _method(tool, {"ticker": spec["full"]["ticker"]})

    assert payload.get("success") is False
    assert payload.get("missing_inputs") == spec["required"]
    for source in spec["pointed_to"]:
        assert source in payload["error"], f"{source} not named: {payload['error']}"


@pytest.mark.parametrize("tool", FOUR)
def test_the_four_methods_survive_an_empty_call(tool):
    payload = _method(tool, {})

    assert payload.get("success") is False
    assert payload.get("missing_inputs") == FOUR[tool]["required"]
    assert "KeyError" not in json.dumps(payload)


@pytest.mark.parametrize("tool", FOUR)
def test_the_four_descriptions_ask_for_every_input_and_name_its_source(tool):
    spec = FOUR[tool]
    listed = _listed(tool)

    for wording in PLANNER_WORDING:
        assert wording not in listed.description, (
            f"an MCP client has no planner, and is told {wording!r}")
    for name in spec["required"] + spec["optional"]:
        assert _named(listed.description, [name]), f"{name} is not asked for"
    for source in spec["described"]:
        assert source in listed.description, f"{source} is not named"


@pytest.mark.parametrize("tool", FOUR)
def test_the_four_schemas_require_exactly_the_inputs_they_refuse_without(tool):
    spec = FOUR[tool]
    schema = _listed(tool).inputSchema

    assert set(schema["required"]) == {"ticker", *spec["required"]}
    for name in spec["required"]:
        assert schema["properties"][name].get("description", "").startswith(
            "Required"), f"{name} is not described as required"
    for name in spec["optional"]:
        assert name not in schema["required"]
        assert schema["properties"][name]["description"].startswith("Optional")


# ---- calculate_scenario_dcf ------------------------------------------------

def test_the_full_scenario_set_values_the_company_as_before():
    is_error, text, payload = _call("calculate_scenario_dcf", SCENARIO_FULL)

    assert _ran(is_error, payload), text[:300]
    assert [payload[c]["price_per_share"] for c in ("bear", "base", "bull")] == [
        150.56, 258.33, 395.31]
    assert payload["price_range"] == {"low": 150.56, "mid": 258.33, "high": 395.31}
    assert "terminal_sensitivity" in payload


def test_zero_is_a_value_in_a_scenario_where_a_company_can_have_none():
    zeros = dict(cash=0, debt=0, tax_rate=0, capex_pct_revenue=0,
                 depreciation=0, terminal_growth=0)
    is_error, text, payload = _call("calculate_scenario_dcf",
                                    {**SCENARIO_FULL, **zeros})

    assert _ran(is_error, payload), text[:300]
    for case in ("bear", "base", "bull"):
        assert payload[case]["equity_value"] == payload[case]["enterprise_value"], (
            f"{case}: cash and debt were 0, so equity is the enterprise value")
    # The base case is calculate_dcf on the same inputs, so a zero that was
    # replaced on the way in would price it differently.
    _, _, dcf = _call("calculate_dcf", {
        **_without(SCENARIO_FULL, "bear_growth", "base_growth", "bull_growth",
                   "bear_margin", "base_margin", "bull_margin"),
        **zeros, "revenue_growth": SCENARIO_FULL["base_growth"],
        "ebitda_margin": SCENARIO_FULL["base_margin"]})
    assert payload["base"]["price_per_share"] == dcf["price_per_share"]


def test_a_scenario_without_terminal_growth_is_refused_not_given_two_and_a_half_percent():
    args = _without(SCENARIO_FULL, "terminal_growth")
    is_error, text, payload = _call("calculate_scenario_dcf", args)

    assert _refused(is_error, payload), f"valued with an invented 2.5%: {text[:300]}"
    assert _named(text, SCENARIO_REQUIRED) == {"terminal_growth"}, text[:600]


def test_a_scenario_without_a_terminal_multiple_prices_on_the_perpetuity_alone():
    is_error, text, payload = _call(
        "calculate_scenario_dcf", _without(SCENARIO_FULL, "terminal_multiple"))

    assert _ran(is_error, payload), text[:300]
    assert "terminal_sensitivity" not in payload
    assert payload["base"]["terminal_value_method"] == "perpetuity"


@pytest.mark.parametrize("name", ["bear_growth", "base_growth", "bull_growth"])
def test_an_empty_scenario_growth_list_is_absent(name):
    is_error, text, payload = _call("calculate_scenario_dcf",
                                    {**SCENARIO_FULL, name: []})

    assert _refused(is_error, payload), f"valued on no growth years: {text[:300]}"
    assert _named(text, SCENARIO_REQUIRED) == {name}, text[:600]


# ---- calculate_lbo ---------------------------------------------------------

def test_the_full_lbo_set_models_the_deal_as_before():
    is_error, text, payload = _call("calculate_lbo", LBO_FULL)

    assert _ran(is_error, payload), text[:300]
    assert payload["moic"] == 1.92
    assert payload["irr_pct"] == 13.93
    assert payload["equity_proceeds"] == pytest.approx(215_908_207_314.43)


def test_zero_is_a_value_in_an_lbo_where_a_deal_can_have_none():
    zeros = dict(capex_pct_revenue=0, depreciation=0, tax_rate=0,
                 debt_interest_rate=0, leverage_turns=0)
    args = {**LBO_FULL, **zeros, "revenue_growth": [0, 0, 0, 0, 0]}
    is_error, text, payload = _call("calculate_lbo", args)

    assert _ran(is_error, payload), text[:300]
    assert payload["debt_amount"] == 0
    assert payload["equity_invested"] == args["entry_ev"]
    assert payload["assumptions"]["debt_interest_rate"] == 0
    assert payload["assumptions"]["leverage_turns"] == 0
    for year in payload["year_by_year"]:
        assert year["revenue"] == args["revenue_base"], "0 growth is flat revenue"
        assert year["taxes"] == 0 and year["interest"] == 0
        assert year["fcf_after_service"] == year["ebitda"], (
            "no capex, no tax and no interest: all of EBITDA is free cash flow")


def test_an_lbo_with_no_hold_years_holds_five_and_says_so():
    is_error, text, payload = _call("calculate_lbo", _without(LBO_FULL, "hold_years"))

    assert _ran(is_error, payload), text[:300]
    assert payload["hold_years"] == 5
    assert payload["moic"] == 1.92


def test_an_lbo_with_empty_growth_is_refused_not_given_three_percent():
    is_error, text, payload = _call("calculate_lbo", {**LBO_FULL, "revenue_growth": []})

    assert _refused(is_error, payload), f"modelled on invented growth: {text[:300]}"
    assert _named(text, LBO_REQUIRED) == {"revenue_growth"}, text[:600]
    assert "moic" not in text


@pytest.mark.parametrize("hold_years,rates", [(7, 5), (5, 3)])
def test_an_lbo_whose_hold_outruns_its_growth_rates_is_refused(hold_years, rates):
    args = {**LBO_FULL, "hold_years": hold_years,
            "revenue_growth": [0.05] * rates}
    is_error, text, payload = _call("calculate_lbo", args)

    assert _refused(is_error, payload), (
        f"{rates} rates were stretched over {hold_years} years: {text[:300]}")
    assert "revenue_growth" in text and "hold_years" in text, text[:300]
    assert "moic" not in text


# ---- calculate_credit_profile ----------------------------------------------

def test_the_full_credit_set_profiles_the_company_as_before():
    is_error, text, payload = _call("calculate_credit_profile", CREDIT_FULL)

    assert _ran(is_error, payload), text[:300]
    assert payload["net_debt_ebitda"] == 0.37
    assert payload["interest_coverage"] == 39.66
    assert payload["fcf_yield_pct"] == 2.32


def test_zero_is_a_value_in_a_credit_profile_where_a_company_can_have_none():
    zeros = dict(total_debt=0, cash=0, interest_expense=0, depreciation_abs=0,
                 capex_abs=0, tax_rate=0)
    is_error, text, payload = _call("calculate_credit_profile",
                                    {**CREDIT_FULL, **zeros})

    assert _ran(is_error, payload), text[:300]
    assert payload["net_debt"] == 0 and payload["credit_label"] == "Net Cash"
    for name, value in zeros.items():
        assert payload["inputs"][name] == value, (
            f"{name} was passed as 0 and used as {payload['inputs'][name]}")


def test_a_company_with_no_interest_has_no_coverage_not_a_coverage_over_one_dollar():
    """Zero interest is a debt-free company. EBIT over a stand-in of $1 is a
    number with no meaning, shaped like an answer."""
    is_error, text, payload = _call("calculate_credit_profile",
                                    {**CREDIT_FULL, "interest_expense": 0})

    assert _ran(is_error, payload), text[:300]
    assert payload["interest_coverage"] is None, payload["interest_coverage"]
    assert "interest_expense" in payload["interest_coverage_note"]


def test_a_credit_profile_without_market_cap_is_refused_not_stripped_of_its_yield():
    is_error, text, payload = _call("calculate_credit_profile",
                                    _without(CREDIT_FULL, "market_cap"))

    assert _refused(is_error, payload), f"profiled with no FCF yield: {text[:300]}"
    assert _named(text, CREDIT_REQUIRED) == {"market_cap"}, text[:600]


# ---- calculate_capital_returns ---------------------------------------------

def test_the_full_capital_returns_set_profiles_the_company_as_before():
    is_error, text, payload = _call("calculate_capital_returns", CAPITAL_FULL)

    assert _ran(is_error, payload), text[:300]
    assert payload["dividend_yield_pct"] == 0.77
    assert payload["payout_ratio_pct"] == 56.35
    assert payload["dividend_per_share"] == 3.2301
    assert payload["buyback_per_share"] == 2.288


def test_zero_is_a_value_in_capital_returns_where_a_company_can_have_none():
    zeros = dict(dividends_paid=0, shares_repurchased=0, tax_rate=0,
                 capex_abs=0, depreciation_abs=0)
    is_error, text, payload = _call("calculate_capital_returns",
                                    {**CAPITAL_FULL, **zeros})

    assert _ran(is_error, payload), text[:300]
    assert payload["dividend_yield_pct"] == 0 and payload["buyback_yield_pct"] == 0
    assert payload["dividends_paid"] == 0 and payload["shares_repurchased"] == 0
    assert payload["fcf_estimate"] == CAPITAL_FULL["ebitda"], (
        "no tax, no D&A and no capex: free cash flow is EBITDA")
    assert payload["dividend_per_share"] == 0


def test_a_dividend_payer_is_not_reported_as_paying_nothing():
    """The call a client made once it had the five inputs the schema required."""
    five = _without(CAPITAL_FULL, "dividends_paid", "shares_repurchased",
                    "shares_outstanding")
    is_error, text, payload = _call("calculate_capital_returns", five)

    assert _refused(is_error, payload), f"reported a 0% yield: {text[:300]}"
    assert _named(text, CAPITAL_REQUIRED) == {
        "dividends_paid", "shares_repurchased", "shares_outstanding"}, text[:600]
    assert "dividend_yield_pct" not in text


# ---- zero where a company cannot have none ---------------------------------

@pytest.mark.parametrize("tool,name", [
    ("calculate_scenario_dcf", "revenue_base"),
    ("calculate_scenario_dcf", "wacc"),
    ("calculate_scenario_dcf", "shares_outstanding"),
    ("calculate_lbo", "entry_ev"),
    ("calculate_lbo", "revenue_base"),
    ("calculate_lbo", "ebitda_margin"),
    ("calculate_lbo", "exit_multiple"),
    ("calculate_credit_profile", "ebitda"),
    ("calculate_credit_profile", "market_cap"),
    ("calculate_capital_returns", "market_cap"),
    ("calculate_capital_returns", "ebitda"),
    ("calculate_capital_returns", "shares_outstanding"),
])
def test_zero_is_refused_by_the_four_where_a_company_cannot_have_none(tool, name):
    is_error, text, payload = _call(tool, {**FOUR[tool]["full"], name: 0})

    assert _refused(is_error, payload), f"{tool}: {name}=0 was used: {text[:300]}"
    assert name in text, f"the refusal does not name {name}: {text[:300]}"
    assert not set(payload or {}) & ANSWER_KEYS, (
        f"a result sits beside the refusal: {sorted(set(payload) & ANSWER_KEYS)}")


# ---- the math has no default for what the tools require --------------------

@pytest.mark.parametrize("function,names", [
    (at._capital_returns_math,
     ["dividends_paid", "shares_repurchased", "shares_outstanding"]),
    (at._credit_profile_math, ["market_cap"]),
])
def test_the_math_has_no_default_for_an_input_its_tool_requires(function, names):
    parameters = inspect.signature(function).parameters
    for name in names:
        assert parameters[name].default is inspect.Parameter.empty, (
            f"{function.__name__}({name}=...) reads an absent {name} as "
            f"{parameters[name].default!r}")
