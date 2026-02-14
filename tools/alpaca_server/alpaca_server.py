import os
from alpaca.common import requests
from dotenv import load_dotenv
from alpaca.trading.models import TradeAccount
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest
from alpaca.data.requests import StockLatestQuoteRequest, StockLatestTradeRequest
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.enums import DataFeed
from typing import Dict, List, Optional

class alpaca_client:
    def __init__(self):
        self.trading_client, self.stock_history_client = self.setup_clients()

    def setup_clients(self) -> tuple[TradingClient, StockHistoricalDataClient]:
        load_dotenv()
        API_KEY = os.getenv("ALPACA_API_KEY")
        API_SECRET = os.getenv("ALPACA_SECRET")
        return (TradingClient(api_key=API_KEY, secret_key=API_SECRET, paper=True),
                StockHistoricalDataClient(api_key=API_KEY, secret_key=API_SECRET),
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
        

        return{
        "equity": account.equity,
        "buying_power": account.buying_power
        }


if __name__ == "__main__":
    a = alpaca_client()
    print(a.get_account_metrics())
