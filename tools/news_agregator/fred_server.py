"""
FRED MCP Server -- 4 tools for macroeconomic data.

Provides macro context for IB analysis: interest rates, inflation, employment,
GDP growth, yield curve, and access to 800k+ FRED series.

Entry point: python -m tools.news_agregator.fred_server server
"""
from typing import Any, Dict, List, Optional
import asyncio
import json
import sys
from datetime import date, datetime, timedelta, timezone

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, ServerCapabilities
from mcp.server.models import InitializationOptions

from tools.news_agregator.fred_utils import FredClient, build_envelope


def json_serializer(obj):
  if isinstance(obj, (date, datetime)):
    return obj.isoformat()
  raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def safe_json_dumps(obj):
  return json.dumps(obj, default=json_serializer)


# -- Curated series IDs --

SNAPSHOT_SERIES = {
  # Rates
  "DGS10": "10Y Treasury Yield",
  "DGS2": "2Y Treasury Yield",
  "FEDFUNDS": "Federal Funds Rate",
  "T10Y2Y": "10Y-2Y Treasury Spread",
  "BAMLH0A0HYM2": "HY OAS Spread",
  # Inflation
  "CPIAUCSL": "CPI (All Urban Consumers)",
  "PCEPILFE": "Core PCE Price Index",
  # Employment
  "UNRATE": "Unemployment Rate",
  "PAYEMS": "Nonfarm Payrolls",
  "ICSA": "Initial Jobless Claims",
  # Growth
  "A191RL1Q225SBEA": "Real GDP Growth (QoQ Annualized)",
}

YIELD_CURVE_SERIES = {
  "DGS3MO": "3M",
  "DGS6MO": "6M",
  "DGS1": "1Y",
  "DGS2": "2Y",
  "DGS3": "3Y",
  "DGS5": "5Y",
  "DGS7": "7Y",
  "DGS10": "10Y",
  "DGS20": "20Y",
  "DGS30": "30Y",
}

# Series where YoY% change is more meaningful than the raw index level
YOY_SERIES = {"CPIAUCSL", "PCEPILFE"}

# Tool descriptions
macro_snapshot_description = """Retrieves a curated bundle of key macroeconomic indicators from FRED (Federal Reserve Economic Data).
Includes: 10Y/2Y Treasury yields, Fed Funds rate, HY spread, CPI/Core PCE inflation, unemployment, nonfarm payrolls, initial claims, real GDP growth.
Each indicator shows current value, 3-month-ago value, and 1-year-ago value for trend context.
Should use: When you need macro context for analysis, risk-free rate for WACC, inflation data, or employment conditions.
Should NOT use: For company-specific data (use SEC/financial tools) or specific FRED series (use get_fred_series)."""

treasury_yields_description = """Retrieves the full US Treasury yield curve (3M through 30Y) plus computed spreads and curve shape.
Returns: yield for each maturity, 10Y-2Y spread, 10Y-3M spread, and curve shape label (normal/inverted/flat).
Should use: When you need the risk-free rate for DCF/WACC, credit analysis, LBO debt pricing, or yield curve analysis.
Should NOT use: For non-Treasury macro data (use get_macro_snapshot)."""

fred_series_description = """Retrieves observations for any FRED series by its series ID.
FRED has 800,000+ series covering economics, finance, demographics, and more.
Should use: When you need a specific FRED series not covered by get_macro_snapshot or get_treasury_yields (e.g. M2 money supply, housing starts, consumer sentiment).
Should NOT use: For standard macro indicators (use get_macro_snapshot) or yield curve (use get_treasury_yields)."""

search_fred_description = """Searches across 800,000+ FRED series by keyword to find relevant series IDs.
Returns: top 20 results with series_id, title, frequency, units, and popularity score.
Should use: When you need to discover FRED series IDs for specific economic data (e.g. 'housing starts', 'consumer credit', 'industrial production').
Should NOT use: If you already know the series ID (use get_fred_series directly)."""


def _parse_float(value: str) -> Optional[float]:
  """Parse a FRED observation value, returning None for missing data markers."""
  if value in (".", "", None):
    return None
  try:
    return float(value)
  except (ValueError, TypeError):
    return None


def _get_observation_value(observations: List[Dict], index: int = -1) -> Optional[float]:
  """Get a parsed float from a FRED observations list at the given index."""
  if not observations:
    return None
  # Filter out null markers first
  valid = [o for o in observations if o.get("value") not in (".", "", None)]
  if not valid:
    return None
  try:
    return _parse_float(valid[index]["value"])
  except (IndexError, KeyError):
    return None


def _get_observation_date(observations: List[Dict], index: int = -1) -> Optional[str]:
  """Get the date string from a FRED observations list at the given index."""
  valid = [o for o in observations if o.get("value") not in (".", "", None)]
  if not valid:
    return None
  try:
    return valid[index].get("date")
  except IndexError:
    return None


def _compute_yoy_pct(observations: List[Dict]) -> Optional[float]:
  """Compute year-over-year percent change from FRED index-level observations.

  Looks for the latest value and a value ~12 months prior.
  """
  valid = [o for o in observations if o.get("value") not in (".", "", None)]
  if len(valid) < 13:
    return None
  latest = _parse_float(valid[-1]["value"])
  year_ago = _parse_float(valid[-13]["value"])  # monthly data, ~12 months back
  if latest is None or year_ago is None or year_ago == 0:
    return None
  return round((latest - year_ago) / year_ago * 100, 2)


def _condense_snapshot(series_data: Dict[str, Dict]) -> Dict[str, Any]:
  """Condense snapshot series into latest + 3M-ago + 1Y-ago per series.

  For CPI/PCE, computes YoY% instead of raw index level.
  For rates, shows basis-point change from 3M-ago and 1Y-ago.
  """
  condensed = {}

  for series_id, raw in series_data.items():
    if "error" in raw:
      condensed[series_id] = {"label": SNAPSHOT_SERIES.get(series_id, series_id), "error": raw["error"]}
      continue

    observations = raw.get("observations", [])
    label = SNAPSHOT_SERIES.get(series_id, series_id)

    if series_id in YOY_SERIES:
      yoy = _compute_yoy_pct(observations)
      latest_val = _get_observation_value(observations)
      condensed[series_id] = {
        "label": label,
        "yoy_pct": yoy,
        "latest_index": latest_val,
        "as_of": _get_observation_date(observations),
      }
    else:
      latest = _get_observation_value(observations)
      # Find ~3 months ago and ~1 year ago values
      valid = [o for o in observations if o.get("value") not in (".", "", None)]
      three_mo = _parse_float(valid[-4]["value"]) if len(valid) >= 4 else None
      one_yr = _parse_float(valid[-13]["value"]) if len(valid) >= 13 else None

      entry = {
        "label": label,
        "current": latest,
        "as_of": _get_observation_date(observations),
      }
      if three_mo is not None and latest is not None:
        entry["3m_ago"] = three_mo
        entry["3m_change_bps"] = round((latest - three_mo) * 100, 1)
      if one_yr is not None and latest is not None:
        entry["1y_ago"] = one_yr
        entry["1y_change_bps"] = round((latest - one_yr) * 100, 1)

      condensed[series_id] = entry

  return condensed


def _condense_yield_curve(series_data: Dict[str, Dict]) -> Dict[str, Any]:
  """Condense yield curve series into rates + computed spreads + shape."""
  curve = {}
  errors = []

  for series_id, raw in series_data.items():
    maturity = YIELD_CURVE_SERIES.get(series_id, series_id)
    if "error" in raw:
      errors.append(f"{maturity}: {raw['error']}")
      continue
    observations = raw.get("observations", [])
    rate = _get_observation_value(observations)
    if rate is not None:
      curve[maturity] = rate

  # Compute spreads
  spreads = {}
  if "10Y" in curve and "2Y" in curve:
    spreads["10Y_2Y"] = round(curve["10Y"] - curve["2Y"], 3)
  if "10Y" in curve and "3M" in curve:
    spreads["10Y_3M"] = round(curve["10Y"] - curve["3M"], 3)

  # Determine curve shape
  shape = "unknown"
  if "10Y_2Y" in spreads:
    if spreads["10Y_2Y"] < -0.1:
      shape = "inverted"
    elif spreads["10Y_2Y"] > 0.1:
      shape = "normal"
    else:
      shape = "flat"

  result = {
    "curve": curve,
    "spreads": spreads,
    "shape": shape,
  }
  if errors:
    result["errors"] = errors

  return result


def _condense_observations(raw: Dict[str, Any], cap: int = 60) -> Dict[str, Any]:
  """Condense generic FRED observations: filter nulls, cap count."""
  observations = raw.get("observations", [])
  # Filter FRED's "." null markers
  valid = [
    {"date": o["date"], "value": _parse_float(o["value"])}
    for o in observations
    if o.get("value") not in (".", "", None)
  ]
  # Cap at most recent
  valid = valid[-cap:]

  return {
    "series_id": raw.get("series_id", "unknown"),
    "total_observations": raw.get("count", len(valid)),
    "returned": len(valid),
    "observations": valid,
  }


def _condense_search(raw: Dict[str, Any], cap: int = 20) -> Dict[str, Any]:
  """Condense FRED series search results to top N with essential fields."""
  series_list = raw.get("seriess", [])
  results = []
  for s in series_list[:cap]:
    results.append({
      "series_id": s.get("id", ""),
      "title": s.get("title", ""),
      "frequency": s.get("frequency_short", ""),
      "units": s.get("units_short", ""),
      "popularity": s.get("popularity", 0),
      "last_updated": s.get("last_updated", ""),
    })

  return {
    "total_matches": raw.get("count", len(results)),
    "returned": len(results),
    "results": results,
  }


class FredServer:
  def __init__(self):
    self.server = Server("fred")
    self.client = FredClient()
    self._setup_handlers()

  def _setup_handlers(self):
    parent = self

    @self.server.list_tools()
    async def list_tools() -> List[Tool]:
      return [
        Tool(
          name="get_macro_snapshot",
          description=macro_snapshot_description,
          inputSchema={
            "type": "object",
            "properties": {},
            "required": []
          }
        ),
        Tool(
          name="get_treasury_yields",
          description=treasury_yields_description,
          inputSchema={
            "type": "object",
            "properties": {},
            "required": []
          }
        ),
        Tool(
          name="get_fred_series",
          description=fred_series_description,
          inputSchema={
            "type": "object",
            "properties": {
              "series_id": {
                "type": "string",
                "description": "FRED series ID (e.g. 'DGS10', 'CPIAUCSL', 'UNRATE')"
              },
              "start": {
                "type": "string",
                "description": "Start date in YYYY-MM-DD format (optional, defaults to 5 years ago)"
              },
              "end": {
                "type": "string",
                "description": "End date in YYYY-MM-DD format (optional, defaults to today)"
              },
              "frequency": {
                "type": "string",
                "description": "Data frequency: d (daily), w (weekly), m (monthly), q (quarterly), a (annual). Optional.",
                "enum": ["d", "w", "m", "q", "a"]
              }
            },
            "required": ["series_id"]
          }
        ),
        Tool(
          name="search_fred",
          description=search_fred_description,
          inputSchema={
            "type": "object",
            "properties": {
              "search_text": {
                "type": "string",
                "description": "Keywords to search for (e.g. 'housing starts', 'consumer sentiment')"
              }
            },
            "required": ["search_text"]
          }
        ),
      ]

    @self.server.call_tool()
    async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
      match name:
        case "get_macro_snapshot":
          return await parent.get_macro_snapshot()
        case "get_treasury_yields":
          return await parent.get_treasury_yields()
        case "get_fred_series":
          return await parent.get_fred_series(
            arguments["series_id"],
            arguments.get("start"),
            arguments.get("end"),
            arguments.get("frequency")
          )
        case "search_fred":
          return await parent.search_fred(arguments["search_text"])
        case _:
          return [TextContent(
            type="text",
            text=safe_json_dumps({"error": f"Unknown tool: {name}"})
          )]

  # -- Tool implementations --

  async def _fetch_series(self, series_id: str, lookback_months: int = 18) -> Dict[str, Any]:
    """Fetch observations for a single series with a default lookback window."""
    start = (datetime.now() - timedelta(days=lookback_months * 30)).strftime("%Y-%m-%d")
    return await self.client.get("/series/observations", {
      "series_id": series_id,
      "observation_start": start,
      "sort_order": "asc",
    })

  async def get_macro_snapshot(self) -> List[TextContent]:
    """Fetch all snapshot series concurrently and condense."""
    tasks = {sid: self._fetch_series(sid) for sid in SNAPSHOT_SERIES}
    results = await asyncio.gather(*tasks.values(), return_exceptions=True)

    series_data = {}
    errors = []
    for sid, result in zip(tasks.keys(), results):
      if isinstance(result, Exception):
        series_data[sid] = {"error": str(result)}
        errors.append(str(result))
      else:
        series_data[sid] = result

    condensed = _condense_snapshot(series_data)
    envelope = build_envelope(condensed, "macro_snapshot", "get_macro_snapshot",
                              api_calls_made=len(SNAPSHOT_SERIES), errors=errors)
    return [TextContent(type="text", text=safe_json_dumps(envelope))]

  async def get_treasury_yields(self) -> List[TextContent]:
    """Fetch all yield curve maturities concurrently and compute spreads."""
    tasks = {sid: self._fetch_series(sid, lookback_months=3) for sid in YIELD_CURVE_SERIES}
    results = await asyncio.gather(*tasks.values(), return_exceptions=True)

    series_data = {}
    errors = []
    for sid, result in zip(tasks.keys(), results):
      if isinstance(result, Exception):
        series_data[sid] = {"error": str(result)}
        errors.append(str(result))
      else:
        series_data[sid] = result

    condensed = _condense_yield_curve(series_data)
    envelope = build_envelope(condensed, "treasury_yields", "get_treasury_yields",
                              api_calls_made=len(YIELD_CURVE_SERIES), errors=errors)
    return [TextContent(type="text", text=safe_json_dumps(envelope))]

  async def get_fred_series(self, series_id: str, start: Optional[str] = None,
                            end: Optional[str] = None, frequency: Optional[str] = None) -> List[TextContent]:
    """Fetch observations for any FRED series."""
    params = {"series_id": series_id, "sort_order": "asc"}
    if start:
      params["observation_start"] = start
    else:
      params["observation_start"] = (datetime.now() - timedelta(days=5*365)).strftime("%Y-%m-%d")
    if end:
      params["observation_end"] = end
    if frequency:
      params["frequency"] = frequency

    result = await self.client.get("/series/observations", params)
    if "error" in result:
      envelope = build_envelope(result, series_id, "get_fred_series")
    else:
      # Attach series_id for downstream identification
      result["series_id"] = series_id
      condensed = _condense_observations(result)
      envelope = build_envelope(condensed, series_id, "get_fred_series")
    return [TextContent(type="text", text=safe_json_dumps(envelope))]

  async def search_fred(self, search_text: str) -> List[TextContent]:
    """Search FRED series by keyword."""
    result = await self.client.get("/series/search", {
      "search_text": search_text,
      "order_by": "popularity",
      "sort_order": "desc",
    })
    if "error" in result:
      envelope = build_envelope(result, search_text, "search_fred")
    else:
      condensed = _condense_search(result)
      envelope = build_envelope(condensed, search_text, "search_fred")
    return [TextContent(type="text", text=safe_json_dumps(envelope))]

  async def run_server(self):
    try:
      async with stdio_server() as (read_stream, write_stream):
        await self.server.run(read_stream, write_stream, InitializationOptions(
          server_name="fred",
          server_version="1.0.0",
          capabilities=ServerCapabilities()
        ))
        print("Successfully created fred process", file=sys.stderr, flush=True)
    except Exception as e:
      import traceback
      traceback.print_exc(file=sys.stderr)
      raise
    finally:
      await self.client.close()


if __name__ == "__main__":
  if len(sys.argv) < 2:
    print("Usage: python -m tools.news_agregator.fred_server server", file=sys.stderr)
    sys.exit(1)

  if sys.argv[1] == "server":
    print("Starting fred process", file=sys.stderr, flush=True)
    try:
      server = FredServer()
      asyncio.run(server.run_server())
    except Exception as e:
      print(f"SERVER: Exception in main: {e}", file=sys.stderr, flush=True)
      import traceback
      traceback.print_exc(file=sys.stderr)
      sys.exit(1)
  else:
    print(f"Unknown argument: {sys.argv[1]}", file=sys.stderr, flush=True)
    print("Usage: python -m tools.news_agregator.fred_server server", file=sys.stderr)
    sys.exit(1)
