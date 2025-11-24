# core/gpt_strategy.py

import streamlit as st
import pandas as pd

from core.gpt_strategy_utils import (
    fetch_price_data,        # imported via options_directional_utils
    compute_gpt_indicators,
    evaluate_gpt_signal,
)


def render_gpt_strategy_tab():
    st.header("🤖 GPT Strategy – Directional Signal")

    st.markdown(
        """
This tab uses the **GPTStrategy**:

- Trend: **EMA20 vs EMA40**  
- Momentum: **RSI 5**  
- Impulse: **Heikin-Ashi color**  

It outputs: **BUY CALL**, **BUY PUT**, or **NO TRADE**.
"""
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        ticker = st.text_input(
            "Ticker",
            value="AMD",
            key="gpt_strat_ticker",
        ).strip().upper()

    with col2:
        period = st.selectbox(
            "Lookback period",
            ["6mo", "1y", "3y"],
            index=1,
            key="gpt_strat_period",
        )

    if not ticker:
        st.info("Enter a ticker to begin.")
        return

    try:
        df = fetch_price_data(ticker, period=period, interval="1d")
        df = compute_gpt_indicators(df)
    except Exception as e:
        st.error(f"Error fetching or processing data for {ticker}: {e}")
        return

    if df.empty:
        st.warning("Not enough data after indicator calculations.")
        return

    latest = df.iloc[-1]
    overall_signal, indicators = evaluate_gpt_signal(latest)

    date_str = latest.name.strftime("%Y-%m-%d")
    price = float(latest["Close"])

    if overall_signal == "BUY_CALL":
        st.success(
            f"**Signal: BUY CALL 🟢**  \nDate: {date_str} | Close: {price:.2f}"
        )
    elif overall_signal == "BUY_PUT":
        st.error(
            f"**Signal: BUY PUT 🔴**  \nDate: {date_str} | Close: {price:.2f}"
        )
    else:
        st.warning(
            f"**Signal: NO TRADE / NEUTRAL 🟡**  \nDate: {date_str} | Close: {price:.2f}"
        )

    # Indicator table
    st.subheader("Indicator Alignment")

    rows = []
    for name, vals in indicators.items():
        rows.append(
            {
                "Indicator": name,
                "Bullish": "✅" if vals["bullish"] else "❌",
                "Bearish": "✅" if vals["bearish"] else "❌",
            }
        )

    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # Trend + HA chart
    st.subheader("Price, EMA20/40 & Heikin-Ashi Direction")

    last_n = df.tail(120)
    chart_df = last_n[["Close", "EMA_20", "EMA_40"]].copy()
    st.line_chart(chart_df)

    st.markdown(
        """
### Trade Management (for options based on GPTStrategy)

- Prefer **5–10 DTE** for lottos.  
- Only trade when **all 3 signals align**.  
- **Stop loss:** when Heikin-Ashi flips color against you *or* the option is down ~50%.  
- **Take profit:** when you get 2–3 strong bars in your favor **and** RSI5 goes extreme (>75 for calls, <25 for puts), or when your premium is up **100–150%**.  
- **Time stop:** if nothing happens within **6 trading days**, exit and recycle risk.
"""
    )