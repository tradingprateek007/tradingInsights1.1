from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce


def get_account_with_client(client: TradingClient):
    """
    Get account information using an alpaca-py TradingClient.
    """
    return client.get_account()


def get_positions_with_client(client: TradingClient):
    """
    Get all open stock positions using an alpaca-py TradingClient.
    """
    return client.get_all_positions()


def place_order_with_client(client: TradingClient, symbol, qty, side, asset_type="stock"):
    """
    Submit a stock order using an alpaca-py TradingClient.
    Options are handled separately in trading_tab.py using direct REST calls.
    """
    if asset_type == "stock":
        order_req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.GTC
        )
        order = client.submit_order(order_req)
        return order
    else:
        # Options handled separately
        raise NotImplementedError("Option orders are handled via REST API in trading_tab.py")