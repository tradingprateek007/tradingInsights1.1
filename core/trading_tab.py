import streamlit as st
import requests
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from core.alpaca_trading import get_account_with_client, get_positions_with_client


def fetch_options_contracts(symbol, api_key, secret_key, base_url="https://paper-api.alpaca.markets"):
    url = f"{base_url}/v2/options/contracts"
    headers = {
        "accept": "application/json",
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": secret_key
    }
    params = {"underlying_symbol": symbol}
    resp = requests.get(url, headers=headers, params=params)
    resp.raise_for_status()
    data = resp.json()
    if "option_contracts" not in data:
        raise KeyError(f"No contracts found in API response: {data}")
    return data["option_contracts"]


def get_options_positions(api_key, secret_key, base_url="https://paper-api.alpaca.markets"):
    url = f"{base_url}/v2/options/positions"
    headers = {
        "accept": "application/json",
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": secret_key
    }
    resp = requests.get(url, headers=headers)
    if resp.status_code == 404:
        # Endpoint not available in paper trading mode
        return None
    resp.raise_for_status()
    return resp.json()


def place_option_order(option_symbol, qty, side, api_key, secret_key, base_url="https://paper-api.alpaca.markets"):
    url = f"{base_url}/v2/options/orders"
    headers = {
        "accept": "application/json",
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": secret_key,
        "Content-Type": "application/json"
    }
    payload = {
        "symbol": option_symbol,
        "qty": qty,
        "side": side,
        "type": "market"
    }
    resp = requests.post(url, headers=headers, json=payload)
    resp.raise_for_status()
    return resp.json()


def render_trading_tab():
    st.title("📈 Make Trades with Alpaca (Paper)")

    base_url = "https://paper-api.alpaca.markets"

    st.subheader("🔒 Authenticate to Trade")

    pin = st.text_input("Enter PIN (if you’re the owner)", type="password")
    user_api_key = st.text_input("Your Alpaca API Key (if not owner)")
    user_secret_key = st.text_input("Your Alpaca Secret Key (if not owner)")

    if pin == st.secrets.get("OWNER_PIN"):
        api_key = st.secrets["ALPACA_API_KEY"]
        secret_key = st.secrets["ALPACA_SECRET_KEY"]
        st.success("✅ Owner credentials unlocked.")
    elif user_api_key and user_secret_key:
        api_key = user_api_key
        secret_key = user_secret_key
        st.success("✅ Using provided credentials.")
    else:
        st.warning("⚠️ Please enter valid credentials or PIN to continue.")
        st.stop()

    trading_client = TradingClient(api_key, secret_key, paper=True)

    acc = get_account_with_client(trading_client)
    st.metric("Account Equity", f"${acc.equity}")
    st.metric("Buying Power", f"${acc.buying_power}")

    st.subheader("📊 Your Current Positions")

    # Stock Positions
    stock_positions = get_positions_with_client(trading_client)
    if stock_positions:
        st.markdown("### 📈 Stock Positions")
        stock_data = []
        for p in stock_positions:
            stock_data.append({
                "Symbol": p.symbol,
                "Qty": p.qty,
                "Side": p.side,
                "Market Value": p.market_value
            })
        st.table(stock_data)
    else:
        st.info("No stock positions.")

    # Options Positions
    options_positions = get_options_positions(api_key, secret_key, base_url)
    if options_positions is None:
        st.info("ℹ️ Options positions are not available in paper trading mode.")
    elif options_positions:
        st.markdown("### 📝 Options Positions")
        options_data = []
        for p in options_positions:
            options_data.append({
                "Symbol": p['symbol'],
                "Qty": p['qty'],
                "Side": p['side'],
                "Market Value": p['market_value']
            })
        st.table(options_data)
    else:
        st.info("No options positions.")

    st.subheader("📋 Place a New Trade")

    asset_type = st.selectbox("Asset Type", ["Stock", "Option"])
    symbol = st.text_input("Ticker Symbol", value="AAPL").upper()

    option_symbol = None

    if asset_type == "Option" and symbol:
        try:
            contracts_json = fetch_options_contracts(symbol, api_key, secret_key, base_url)
            if not contracts_json:
                st.error("No option contracts found for this symbol.")
                st.stop()

            contract_choices = {
                f"{c['expiration_date']} | {c['strike_price']} | {c['type'].capitalize()}": c['symbol']
                for c in contracts_json
            }

            chosen_contract_display = st.selectbox(
                "Select Option Contract",
                list(contract_choices.keys())
            )

            option_symbol = contract_choices[chosen_contract_display]

        except Exception as e:
            st.error(f"Failed to fetch option contracts: {e}")
            st.stop()

    with st.form("trade_form"):
        qty = st.number_input("Quantity", min_value=1, value=1)
        side = st.selectbox("Side", ["buy", "sell"])

        submitted = st.form_submit_button("Submit Order")

    if submitted:
        try:
            if asset_type == "Stock":
                order_req = MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                    time_in_force=TimeInForce.GTC
                )
                order = trading_client.submit_order(order_req)
                st.success(f"✅ Stock order submitted: {order.id}")
            else:
                if not option_symbol:
                    st.error("❌ No valid option contract selected.")
                    return
                order = place_option_order(option_symbol, qty, side, api_key, secret_key, base_url)
                st.success(f"✅ Option order submitted: {order['id']}")
        except Exception as e:
            st.error(f"❌ Order failed: {e}")