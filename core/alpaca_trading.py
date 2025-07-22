from core.utils import api

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