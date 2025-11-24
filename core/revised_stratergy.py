import streamlit as st
import pandas as pd

from core.options_directional_utils import (
    fetch_price_data,
    compute_indicators,
    evaluate_signal
)


def render_revised_strategy_signal_tab():
    st.header("Options Directional Signal 🔍")

    st.markdown(
        "Enter a stock symbol to analyze key indicators and get a **directional options signal** "
        "(`Buy Call`, `Buy Put`, or `No Trade`).\n\n"
        "_Signal logic is tuned to your historical strengths: trend + momentum + volume._"
    )

    # --- Single ticker input ---
    col1, col2 = st.columns([2, 1])
    with col1:
        ticker = st.text_input(
            "Ticker",
            value="AAPL",
            key="revised_strategy_ticker",
        ).strip().upper()

    with col2:
        period = st.selectbox(
            "Lookback period",
            ["3mo", "6mo", "1y"],
            index=1,
            key="revised_strategy_period",
        )

    if not ticker:
        st.info("Enter a ticker to begin.")
        return

    # --- Fetch + compute indicators ---
    try:
        df = fetch_price_data(ticker, period=period, interval="1d")
        df = compute_indicators(df)
    except Exception as e:
        st.error(f"Error fetching or processing data for {ticker}: {e}")
        return

    if df.empty:
        st.warning("Not enough data after indicator calculations. Try a longer period.")
        return

    # Latest single row
    latest = df.iloc[-1]

    # Evaluate signal
    overall_signal, indicators = evaluate_signal(latest)

    # --- Summary Header ---
    date_str = latest.name.strftime("%Y-%m-%d")
    price = float(latest["Close"])  # critical fix (Series → float)

    if overall_signal == "BUY_CALL":
        st.success(
            f"**Signal: BUY CALL 🟢**\n"
            f"Date: {date_str} | Close: {price:.2f}"
        )
    elif overall_signal == "BUY_PUT":
        st.error(
            f"**Signal: BUY PUT 🔴**\n"
            f"Date: {date_str} | Close: {price:.2f}"
        )
    else:
        st.warning(
            f"**Signal: NO TRADE / NEUTRAL 🟡**\n"
            f"Date: {date_str} | Close: {price:.2f}"
        )

    # --- Show indicator hits ---
    st.subheader("Indicator Check: What’s Hitting?")

    rows = []
    for name, vals in indicators.items():
        rows.append({
            "Indicator": name,
            "Bullish bias": "✅" if vals["bullish"] else "❌",
            "Bearish bias": "❌" if vals["bullish"] else "❌" if not vals["bearish"] else "❌"
        })

    # Correct Bearish column logic
    for row in rows:
        ind_name = row["Indicator"]
        bears = indicators[ind_name]["bearish"]
        row["Bearish bias"] = "✅" if bears else "❌"

    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # --- Price Trend Chart ---
    st.subheader("Price & EMAs")
    st.line_chart(df.tail(100)[["Close", "EMA_21", "EMA_50"]])

    # --- Trade Management Rules ---
    st.subheader("Trade Management Guidelines (for the option position)")

    st.markdown(
        """
### ✔️ Entry Rules
- Prefer **8–14 DTE** options (your historical sweet spot).
- Choose **ATM or slightly OTM** strikes.
- Enter only when **3+ indicators agree**.

### ✔️ Stop Loss
- Cut the option at **–50% of premium paid**.
- Example: Buy at $2.00 → sell if it drops to ~$1.00.

### ✔️ Take Profit Targets
- Scale out at **+100%** (double).
- Close remaining at **+150–200%**, or when:
  - RSI > 70 for calls  
  - RSI < 30 for puts

### ✔️ Time Stop
- If **< 4 days to expiry** AND:
  - Option is still OTM, OR
  - Indicators flipped to neutral…
  → Close the position. Do NOT let it expire worthless.

(These rules come directly from your actual backtest.)
"""
    )