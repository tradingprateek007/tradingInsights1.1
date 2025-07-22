import streamlit as st
from core.alpaca_trading import get_account, get_positions, place_order

def render_make_trades_tab():
    st.header("🛒 Make Trades")

    account = get_account()
    st.metric("Account Equity", f"${account.equity}")
    st.metric("Buying Power", f"${account.buying_power}")

    ticker = st.text_input("Ticker", "AAPL")
    qty = st.number_input("Quantity", min_value=1, value=1)
    side = st.selectbox("Side", ["buy", "sell"])

    if st.button("Submit Market Order"):
        try:
            order = place_order(
                symbol=ticker.upper(),
                qty=qty,
                side=side,
            )
            st.success(f"✅ Order submitted: {order.id}")
        except Exception as e:
            st.error(f"❌ Failed to submit order: {e}")

    st.subheader("Open Positions")
    positions = get_positions()
    if positions:
        st.table([{
            "Symbol": p.symbol,
            "Qty": p.qty,
            "Side": p.side,
            "Market Value": p.market_value
        } for p in positions])
    else:
        st.info("No open positions.")