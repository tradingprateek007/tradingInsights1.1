import os
from alpaca_trade_api.rest import REST
import streamlit as st

# Default client for owner
API_KEY = os.getenv("ALPACA_API_KEY", st.secrets.get("ALPACA_API_KEY"))
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", st.secrets.get("ALPACA_SECRET_KEY"))
BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

api = REST(API_KEY, SECRET_KEY, BASE_URL)


def get_account():
    return api.get_account()


def get_positions():
    return api.list_positions()


def place_order(symbol, qty, side, order_type="market", time_in_force="gtc"):
    order = api.submit_order(
        symbol=symbol,
        qty=qty,
        side=side,
        type=order_type,
        time_in_force=time_in_force,
    )
    return order


# ✅ New helpers with client argument
def get_account_with_client(client):
    return client.get_account()


def get_positions_with_client(client):
    return client.list_positions()


def place_order_with_client(api, symbol, qty, side, asset_type="stock"):
    if asset_type == "stock":
        # omit time_in_force entirely — Alpaca will default to correct one
        return api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type="market",
            time_in_force = "gtc"

        )
    else:
        return api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type="market",
        )