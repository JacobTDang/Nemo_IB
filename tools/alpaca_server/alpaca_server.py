import os
from alpaca.common import requests
from dotenv import load_dotenv
from alpaca.trading.models import TradeAccount, Position
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest
from alpaca.data.requests import StockLatestQuoteRequest, StockLatestTradeRequest
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.enums import DataFeed
from typing import Dict, List, Optional

from datetime import datetime


class Secret:
    """A credential that renders as a placeholder instead of as itself.

    A key bound to a name is rendered by anything that dumps the frame or the
    module it lives in -- pytest under --showlocals, a debugger, a crash
    reporter. Keeping the value behind reveal() leaves nothing renderable to
    render, which closes all of those at once.

    Mirrored from agent/openrouter_template.py (issue #17) rather than
    imported from it. Nothing under tools/news_agregator or tools/alpaca_server
    imports from agent/ today, and agent.openrouter_template is the LLM layer:
    importing it here would put openai, ollama and httpx into a data-source
    image that will never run any of them, which is the coupling
    testing/test_agent_package_boundary.py exists to prevent. See that file
    for the full reasoning behind the type.
    """
    __slots__ = ("_value",)

    PLACEHOLDER = "<redacted>"

    def __init__(self, value: str = ""):
        self._value = value or ""

    def reveal(self) -> str:
        """The raw credential.

        Call this at the point of use -- an SDK constructor -- and never bind
        the result to a name, or the value is back in a frame.
        """
        return self._value

    def scrub(self, text: str) -> str:
        """`text` with the credential replaced by the placeholder."""
        if not self._value:
            return text
        return text.replace(self._value, self.PLACEHOLDER)

    def __repr__(self) -> str:
        return self.PLACEHOLDER

    __str__ = __repr__

    def __bool__(self) -> bool:
        return bool(self._value)


class alpaca_client:
    def __init__(self):
        self.trading_client, self.stock_history_client = self.setup_clients()

    def setup_clients(self) -> tuple[TradingClient, StockHistoricalDataClient]:
        load_dotenv()
        # Wrapped in the same expression that reads the environment: an
        # intermediate `API_KEY = os.getenv(...)` is what any frame dump
        # prints, and a name in this shape is picked up by a debugger or a
        # crash reporter walking the scope as readily as by pytest.
        key = Secret(os.getenv("ALPACA_API_KEY") or "")
        secret = Secret(os.getenv("ALPACA_SECRET") or "")
        if not key or not secret:
            # Refused here rather than handed on. alpaca-py accepts a bare
            # api_key and only complains about the missing partner, so a half
            # configured .env used to surface either as its ValueError or,
            # worse, as a 401 several calls later with nothing naming the
            # cause.
            raise RuntimeError(
                "Missing Alpaca credentials. Set ALPACA_API_KEY and "
                "ALPACA_SECRET in .env")
        return (TradingClient(api_key=key.reveal(), secret_key=secret.reveal(),
                              paper=True),
                StockHistoricalDataClient(api_key=key.reveal(),
                                          secret_key=secret.reveal()),
                )
 
    def order(self, ticker: str,
              amount: int,
              side: str,
              time_in_force=TimeInForce.GTC):

        # place an order down
        side = OrderSide.SELL if side == "SELL" else OrderSide.BUY
        order_data = MarketOrderRequest(
        symbol=ticker,
        notional=amount,
        side=side,
        time_in_force=time_in_force
        )
        order = self.trading_client.submit_order(order_data=order_data)

        # Just for debugging
        print(order)

        return order

    def get_strike_price(self, tickers: List[str]):
        request_params= StockLatestTradeRequest(symbol_or_symbols=tickers)
        latest_trades = self.stock_history_client.get_stock_latest_trade(request_params)
        for ticker in latest_trades:
            print(f"Stock price for {ticker}: {latest_trades[ticker].price}")


    def get_account_metrics(self):
        account: TradeAccount = self.trading_client.get_account()
        positions = self.get_holdings()
        return{
        "holdings": positions,
        "equity": account.equity,
        "buying_power": account.buying_power,
        "date_accessed": str(datetime.now())
        }

    def get_holdings(self):
        positions = self.trading_client.get_all_positions()
        holdings = []

        for p in positions:
            try:
                if not isinstance(p, Position):
                    print(f"Unexpected position type")
                    continue
                holdings.append({
                    "symbol": p.symbol,
                    "qty": p.qty,
                    "market_value": p.market_value,
                    "avg_entry_price": p.avg_entry_price,
                    "unrealized_pl": p.unrealized_pl,
                    "unrealized_plpc": p.unrealized_plpc,
                    "side": p.side,
                }) 
            except Exception as e:
                print(f"Error: {str(e)}")
        return holdings

if __name__ == "__main__":
    a = alpaca_client()
    print(a.get_account_metrics())
